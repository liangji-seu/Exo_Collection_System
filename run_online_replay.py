"""Read-only replay of recorded online sessions; never imports motor/control code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def load_session(path):
    root = Path(path).resolve()
    session = json.loads((root / ".exo/online_session.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / ".exo/manifest.json").read_text(encoding="utf-8"))
    start, stop = session["start_host_monotonic_ns"], session["stop_host_monotonic_ns"]
    result = {"path": root, "session": session, "state": manifest["state"],
              "duration": (stop - start) / 1e9, "imu": [], "encoder": None}
    for filename in ("imu.h5", "encoder.h5"):
        file_path = root / "raw" / filename
        if not file_path.exists():
            continue
        with h5py.File(file_path, "r") as file:
            stamps = file["samples/host_monotonic_ns"][:]
            mask = (stamps >= start) & (stamps < stop)
            times = (stamps[mask].astype(np.int64) - start) / 1e9
            data = file["samples/data"][:][mask]
            channels = [value.decode() for value in file["metadata/channels"][:]]
            units = [value.decode() for value in file["metadata/units"][:]]
            metadata = json.loads(file["metadata/device"][()])
        if filename == "imu.h5":
            ids = metadata.get("device_ids", metadata.get("metadata", {}).get("device_ids", []))
            labels = metadata.get("preview_labels", [])
            for slot in range(data.shape[1]):
                # A USB packet has NaNs in other physical slots. Keep each
                # slot's own arrivals; preserve remaining channel-level NaNs.
                valid = np.isfinite(data[:, slot]).any(axis=1)
                result["imu"].append({"times": times[valid], "data": data[valid, slot],
                    "id": ids[slot] if slot < len(ids) else f"slot {slot}",
                    "label": labels[slot] if slot < len(labels) else f"IMU {slot + 1}",
                    "channels": channels, "units": units})
        else:
            result["encoder"] = {"times": times, "data": data, "channels": channels, "units": units}
    predictions, commands, gate_predictions = [], [], []
    gate_decisions, gate_transitions, gate_provenance_events = [], [], []
    command_context = []
    command_keys = set()
    command_indexes = {}
    # The log includes large model tensors: scan one line at a time and keep
    # only the few scalar fields used in the plots. Never read_text this file.
    with (root / "raw/online_events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith('{"kind":'):
                continue
            event = json.loads(line)
            kind = event.get("kind")
            payload = event["payload"]
            if kind == "prediction":
                at_value = payload.get("source_grid_time_ns", payload.get("time_ns"))
                if at_value is None:
                    continue
                at = int(at_value)
                if start <= at < stop:
                    values = [payload.get(key) for key in ("estimate_nm", "desired_left_nm", "desired_right_nm")]
                    predictions.append([(at - start) / 1e9,
                        *[np.nan if value is None else float(value) for value in values],
                        (int(payload["host_monotonic_ns"]) - start) / 1e9])
                    pre = [payload.get(key) for key in ("gate_pre_left_nm", "gate_pre_right_nm")]
                    post = [payload.get(key) for key in ("gate_post_left_nm", "gate_post_right_nm")]
                    gate_predictions.append([(at - start) / 1e9,
                        *[np.nan if value is None else float(value) for value in (*pre, *post)]])
                    status = payload.get("gate_status")
                    if isinstance(status, dict):
                        gate_transitions.append({"time_s": (int(payload["host_monotonic_ns"]) - start) / 1e9,
                            "kind": "prediction_status", "status": status})
            elif kind in {"sent_command", "command"}:
                # `sent_command` is the motor service's actual frame record.
                # The UI also writes an enriched `command` wrapper for the same
                # frame, so retain it only as a fallback if the actual event is absent.
                nested = payload.get("command") if isinstance(payload.get("command"), dict) else {}
                command = payload if payload.get("left_nm") is not None else nested
                at_value = command.get("host_monotonic_ns", payload.get("host_monotonic_ns"))
                if at_value is None:
                    continue
                at = int(at_value)
                if start <= at < stop:
                    if command.get("left_nm") is None or command.get("right_nm") is None:
                        continue
                    seq = command.get("seq", nested.get("seq"))
                    key = (None if seq is None else int(seq), at,
                           float(command["left_nm"]), float(command["right_nm"]))
                    context = payload.get("gate_context", command.get("gate_context"))
                    if not isinstance(context, dict):
                        context = {name.removeprefix("gate_"): payload.get(name)
                                   for name in ("gate_mode", "gate_sequence", "gate_source_grid_ns",
                                                "gate_generation", "gate_epoch", "gate_state",
                                                "gate_pre_left_nm", "gate_pre_right_nm",
                                                "gate_post_left_nm", "gate_post_right_nm")
                                   if payload.get(name) is not None}
                    item = {"time_s": (at - start) / 1e9, "host_monotonic_ns": at,
                            "kind": kind, "context": context}
                    if key not in command_keys:
                        command_keys.add(key)
                        command_indexes[key] = len(command_context)
                        commands.append([(at - start) / 1e9,
                            float(command["left_nm"]), float(command["right_nm"])])
                        command_context.append(item)
                    elif kind == "sent_command" and key in command_indexes:
                        # Prefer the motor's actual send record over the UI wrapper.
                        command_context[command_indexes[key]] = item
            elif kind in {"gate_decision", "gate_transition", "gate_provenance"}:
                at = int(event.get("host_monotonic_ns", payload.get("computed_at_ns",
                                                                      payload.get("host_monotonic_ns", start))))
                if start <= at < stop:
                    item = {"time_s": (at - start) / 1e9, "host_monotonic_ns": at,
                            "kind": kind, "payload": payload}
                    if kind == "gate_decision":
                        gate_decisions.append(item)
                    elif kind == "gate_transition":
                        gate_transitions.append(item)
                    else:
                        gate_provenance_events.append(item)
    result["prediction"] = np.asarray(predictions, dtype=np.float64).reshape(-1, 5)
    result["command"] = np.asarray(commands, dtype=np.float64).reshape(-1, 3)
    result["gate_prediction"] = np.asarray(gate_predictions, dtype=np.float64).reshape(-1, 5)
    result["gate_decisions"] = gate_decisions
    result["gate_transitions"] = gate_transitions
    result["gate_provenance_events"] = gate_provenance_events
    result["command_context"] = command_context
    metadata = session.get("metadata", {})
    gate_metadata = metadata.get("gate", {}) if isinstance(metadata, dict) else {}
    provenance = gate_metadata.get("provenance") if isinstance(gate_metadata, dict) else None
    if provenance is None and gate_provenance_events:
        provenance = gate_provenance_events[-1]["payload"]
    recorded_gate = bool(provenance or gate_decisions or gate_transitions or gate_provenance_events or
                         (isinstance(gate_metadata, dict) and gate_metadata.get("mode") not in (None, "none")) or
                         any(np.isfinite(row[1:]).any() for row in gate_predictions))
    states = []
    for item in gate_decisions + [entry for entry in gate_transitions if entry["kind"] != "prediction_status"]:
        body = item.get("payload", {})
        status = body.get("state") if isinstance(body, dict) else None
        if status is None and isinstance(body, dict):
            status = body.get("decision", {}).get("reason") if isinstance(body.get("decision"), dict) else None
        if status is not None:
            states.append(str(status))
    result["gate_summary"] = {
        "recorded": recorded_gate,
        "mode": gate_metadata.get("mode") if isinstance(gate_metadata, dict) else None,
        "label": ("未启用" if isinstance(gate_metadata, dict) and gate_metadata.get("mode") == "none" else
                  "已记录" if recorded_gate else "未记录"),
        "provenance": provenance,
        "decision_count": len(gate_decisions),
        "transition_count": len([item for item in gate_transitions if item["kind"] != "prediction_status"]),
        "states": states,
    }
    return result


def summary(data):
    return {"session": data["path"].name, "duration_s": data["duration"],
        "complete": data["session"]["complete"], "state": data["state"],
        "imu_rows_by_device": {item["id"]: len(item["times"]) for item in data["imu"]},
        "encoder_rows": 0 if data["encoder"] is None else len(data["encoder"]["times"]),
        "predictions_in_source_grid_interval": len(data["prediction"]),
        "sent_commands": len(data["command"]),
        "command_max_abs_nm": float(np.max(np.abs(data["command"][:, 1:]))) if len(data["command"]) else None,
        "gate": data.get("gate_summary", {"recorded": False, "label": "未记录"}),
        "gate_decision_events": len(data.get("gate_decisions", [])),
        "gate_transition_events": len(data.get("gate_transitions", [])),
        "timing": "model: source grid; command/feedback/IMU: actual host receive/send; origin: enable confirmation"}


def show_replay(data, *, snapshot=None, at=30):
    if snapshot:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QComboBox, QSlider)
    import pyqtgraph as pg
    from exo_collection.apps.collector.signal_previews import NoWheelViewBox

    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families() and Path("C:/Windows/Fonts/msyh.ttc").exists():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    app.setFont(QFont("Microsoft YaHei", 9))
    pg.setConfigOptions(antialias=False, background="#f8fafc", foreground="#24344b")

    class Replay(QWidget):
        def __init__(self):
            super().__init__()
            self.position, self.playing, self.last_wall = 0., True, time.perf_counter()
            self.traces, self.plots, self.cursors, self.imu_traces = [], [], [], []
            self.setWindowTitle(f"只读离线回放 · {data['path'].name}")
            self.resize(1740, 1000)
            layout = QVBoxLayout(self)
            info = QLabel(f"只读离线回放　｜　{data['path'].name}　｜　{data['duration']:.3f} s　｜　"
                f"{'完整' if data['session']['complete'] else '不完整'} / {data['state']}")
            info.setStyleSheet("font-size:17px;font-weight:600;color:#164e63;padding:5px")
            layout.addWidget(info)
            config = data["session"].get("metadata", {}).get("spec", {}).get("assistance", {})
            layout.addWidget(QLabel(f"体重 {config.get('mass_kg', '—')} kg　·　助力比例 {config.get('gain', '—')}　·　"
                f"低通 {config.get('lowpass_hz', '—')} Hz　·　延时 {config.get('delay_s', '—')} s　·　"
                f"双侧映射 {config.get('motor_distribution', '—')}"))
            layout.addWidget(QLabel("统一起点：真实使能确认；模型/期望辅助按源网格时刻，发送/反馈按实际时刻。"
                "原始信号独立计时；缺测保留 NaN；每页 10 秒，红线为播放位置；鼠标缩放已关闭。"))
            gate_info = data.get("gate_summary", {"label": "未记录"})
            gate_text = f"门控：{gate_info.get('label', '未记录')}"
            if gate_info.get("mode"):
                gate_text += f" · {gate_info['mode']}"
            if gate_info.get("recorded"):
                gate_text += (f" · 决策 {gate_info.get('decision_count', 0)} 条"
                              f" · 状态变化 {gate_info.get('transition_count', 0)} 条")
            layout.addWidget(QLabel(gate_text))
            controls = QHBoxLayout()
            self.button = QPushButton("暂停")
            self.button.clicked.connect(self.toggle)
            controls.addWidget(self.button)
            self.progress = QSlider(Qt.Orientation.Horizontal)
            self.progress.setRange(0, 100000)
            self.progress.sliderMoved.connect(lambda value: self.seek(value / 100000 * data["duration"]))
            controls.addWidget(self.progress, 1)
            self.time_label = QLabel()
            self.time_label.setMinimumWidth(170)
            controls.addWidget(self.time_label)
            self.speed = QComboBox()
            self.speed.addItems(["0.5×", "1×", "2×", "4×"])
            self.speed.setCurrentIndex(1)
            self.speed.currentIndexChanged.connect(lambda _index: setattr(self, "last_wall", time.perf_counter()))
            controls.addWidget(self.speed)
            controls.addWidget(QLabel("IMU 通道组"))
            self.group = QComboBox()
            self.group.addItems(["加速度 acc", "角速度 gyr", "磁场 mag", "姿态 roll / pitch / yaw"])
            self.group.currentIndexChanged.connect(self.change_group)
            controls.addWidget(self.group)
            layout.addLayout(controls)
            grid = QGridLayout()
            layout.addLayout(grid, 1)

            def plot(title, unit, row, col, span=1):
                widget = pg.PlotWidget(viewBox=NoWheelViewBox())
                widget.setTitle(title)
                widget.setLabel("left", unit)
                widget.setLabel("bottom", "使能后时间", units="s")
                widget.setMouseEnabled(x=False, y=False)
                widget.setMenuEnabled(False)
                widget.showGrid(x=True, y=True, alpha=.18)
                widget.addLegend(offset=(8, 8))
                cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#dc2626", width=2))
                widget.addItem(cursor)
                self.plots.append(widget)
                self.cursors.append(cursor)
                grid.addWidget(widget, row, col, 1, span)
                return widget

            def trace(widget, times, values, name, color, dashed=False):
                curve = widget.plot(name=name, pen=pg.mkPen(color, width=1.6,
                    style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine), connect="finite")
                entry = [curve, times, values]
                self.traces.append(entry)
                return entry

            names = {"imu_left_leg": "左腿 IMU", "imu_right_leg": "右腿 IMU", "imu_pelvis": "骨盆 IMU"}
            for slot, item in enumerate(data["imu"]):
                widget = plot(f"{names.get(item['label'], item['label'])} · {item['id']}", item["units"][0], 0, slot)
                entries = [trace(widget, item["times"], item["data"][:, axis], item["channels"][axis], color)
                    for axis, color in enumerate(("#dc2626", "#2563eb", "#059669"))]
                self.imu_traces.append((widget, item, entries))
            model = plot("关节估计 / 期望辅助 / 实际发送", "Nm", 1, 0, 3)
            prediction, command = data["prediction"], data["command"]
            for column, name, color, dashed in [(1, "模型右关节估计", "#334155", False)]:
                trace(model, prediction[:, 0], prediction[:, column], name, color, dashed)
            gate_trace = data.get("gate_prediction", np.empty((0, 5)))
            if data.get("gate_summary", {}).get("recorded") and len(gate_trace):
                for column, name, color, dashed in [(1, "门控前辅助左", "#7c3aed", True),
                    (2, "门控前辅助右", "#059669", True), (3, "门控后辅助左", "#8b5cf6", False),
                    (4, "门控后辅助右", "#10b981", False)]:
                    trace(model, gate_trace[:, 0], gate_trace[:, column], name, color, dashed)
            else:
                for column, name, color, dashed in [(2, "期望辅助左", "#7c3aed", True),
                    (3, "期望辅助右", "#059669", False)]:
                    trace(model, prediction[:, 0], prediction[:, column], name, color, dashed)
            for column, name, color, dashed in [(1, "实际发送左", "#f97316", True), (2, "实际发送右", "#2563eb", False)]:
                trace(model, command[:, 0], command[:, column], name, color, dashed)
            if data["encoder"] is not None:
                encoder = data["encoder"]
                for metric, title, unit in [(0, "电机反馈位置", "rad"), (1, "电机反馈速度", "rad/s"), (2, "电机反馈扭矩", "Nm")]:
                    widget = plot(title, unit, 2, metric)
                    for column, name, color in [(metric, "左", "#7c3aed"), (metric + 3, "右", "#059669")]:
                        trace(widget, encoder["times"], encoder["data"][:, column], name, color)
            for row, stretch in [(0, 3), (1, 4), (2, 3)]:
                grid.setRowStretch(row, stretch)
            self.timer = QTimer(self)
            self.timer.setInterval(40)
            self.timer.timeout.connect(self.advance)
            self.timer.start()
            self.render()

        def change_group(self, group):
            for widget, item, entries in self.imu_traces:
                legend = widget.plotItem.legend
                legend.clear()
                for axis, entry in enumerate(entries):
                    channel = group * 3 + axis
                    entry[2] = item["data"][:, channel]
                    legend.addItem(entry[0], item["channels"][channel])
                widget.setLabel("left", item["units"][group * 3])
            self.render()

        def toggle(self):
            self.playing = not self.playing and self.position < data["duration"]
            self.button.setText("暂停" if self.playing else "播放")
            self.last_wall = time.perf_counter()

        def seek(self, position):
            self.position = max(0., min(position, data["duration"]))
            self.last_wall = time.perf_counter()
            self.render()

        def advance(self):
            now = time.perf_counter()
            if self.playing and not self.progress.isSliderDown():
                self.position = min(data["duration"], self.position + (now - self.last_wall) * [.5, 1, 2, 4][self.speed.currentIndex()])
                if self.position >= data["duration"]:
                    self.playing = False
                    self.button.setText("播放")
                self.render()
            self.last_wall = now

        def render(self):
            # At an exact page boundary retain the completed ten-second page
            # until playback advances, so a snapshot at 30 s is meaningful.
            page = max(0, int(np.ceil(self.position / 10)) - 1) * 10
            for curve, times, values in self.traces:
                begin, end = np.searchsorted(times, [page, self.position], side="left")
                curve.setData(times[begin:end], values[begin:end])
            for widget, cursor in zip(self.plots, self.cursors):
                widget.setXRange(page, page + 10, padding=0)
                cursor.setValue(self.position)
            if not self.progress.isSliderDown():
                self.progress.setValue(round(self.position / data["duration"] * 100000))
            self.time_label.setText(f"{self.position:7.2f} / {data['duration']:.2f} s")

    window = Replay()
    if snapshot:
        window.show()
    else:
        window.showMaximized()
    if snapshot:
        window.playing = False
        window.button.setText("播放")
        window.seek(min(at, data["duration"]))
        def save_snapshot():
            destination = Path(snapshot).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(destination)):
                raise RuntimeError(f"Screenshot could not be saved: {destination}")
            print(str(destination))
            app.quit()
        QTimer.singleShot(250, save_snapshot)
    return app.exec()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--check", action="store_true", help="Read and summarize without constructing a GUI")
    parser.add_argument("--snapshot", type=Path, help="Render one offscreen screenshot and exit")
    parser.add_argument("--at", type=float, default=30, help="Screenshot position in seconds (default: 30)")
    args = parser.parse_args()
    path = args.session
    if path is None:
        available = sorted(path.parent.parent for path in (ROOT / "runtime_data/online_recordings").glob("*/.exo/online_session.json")
            if not path.parent.parent.name.endswith(".recording"))
        if not available:
            parser.error("No closed online recording found")
        path = available[-1]
    data = load_session(path)
    if args.check:
        print(json.dumps(summary(data), ensure_ascii=False, indent=2))
        return 0
    return show_replay(data, snapshot=args.snapshot, at=args.at)


if __name__ == "__main__":
    raise SystemExit(main())
