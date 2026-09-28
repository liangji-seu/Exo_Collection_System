"""Independent four-channel ultrasound / feature waterfall viewer.

This module deliberately does not run model inference.  It reads a finalized
``ultrasound.bin`` session, pairs the kth packet received on each channel, and
shows the exact per-frame 48-bin preprocessing next to the raw ADC waterfall.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.ndimage import uniform_filter1d

# Keep the viewer usable as a standalone file while reusing the project's
# strict indexed binary reader and raw Ethernet flag decoder.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from exo_collection.adapters.ultrasound.raw_ethernet import (  # noqa: E402
    decode_raw_ethernet_flags,
)
from exo_collection.readers.binary_block import BlockBinaryReader  # noqa: E402


@dataclass(frozen=True)
class SessionData:
    """All paired frames used by the viewer.

    ``raw_adc`` has shape ``(4, frame, 997)`` and ``features`` has shape
    ``(4, frame, 48)``.  The arrays intentionally retain every complete frame;
    no GUI-oriented even-index downsampling is applied.
    """

    session_dir: Path
    times_s: np.ndarray
    raw_adc: np.ndarray
    features: np.ndarray
    channels: tuple[str, ...]
    nominal_rate_hz: float


def compute_feature48(payload: np.ndarray) -> np.ndarray:
    """Apply the specified frame-local preprocessing to ADC payloads.

    Input may be ``(frame, 997)`` or ``(channel, frame, 997)``.  The returned
    last axis is the 48-dimensional mean feature, before any standardization.
    """

    values = np.asarray(payload, dtype=np.uint8).astype(np.float32)
    median = np.median(values, axis=-1, keepdims=True)
    envelope = uniform_filter1d(
        np.abs(values - median), size=20, axis=-1, mode="nearest"
    )
    logged = np.log1p(np.maximum(envelope, 0.0))[..., 20:977]
    parts = np.array_split(logged, 48, axis=-1)
    return np.stack([part.mean(axis=-1) for part in parts], axis=-1).astype(
        np.float32, copy=False
    )


def _find_session_dir(path: str | Path) -> Path:
    supplied = Path(path).expanduser().resolve()
    if supplied.is_file():
        if supplied.name.casefold() != "ultrasound.bin":
            raise ValueError("请选择 session 文件夹或 ultrasound.bin")
        return supplied.parent
    if not supplied.is_dir():
        raise FileNotFoundError(supplied)
    candidate = supplied / "ultrasound.bin"
    if not candidate.is_file():
        raise FileNotFoundError(f"目录中没有 ultrasound.bin：{supplied}")
    return supplied


def _read_complete_frames(
    session_dir: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], float]:
    """Read every complete kth-per-channel frame in two bounded passes."""

    data_path = session_dir / "ultrasound.bin"
    meta_path = session_dir / "ultrasound.meta.json"
    index_path = session_dir / "ultrasound.idx"
    if not meta_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("需要 ultrasound.meta.json 和 ultrasound.idx")
    with BlockBinaryReader(
        data_path,
        meta_path=meta_path,
        index_path=index_path,
        validate_crc=True,
        auto_rebuild_index=False,
    ) as reader:
        metadata = reader.metadata
        complete_wire = (
            str(metadata.get("raw_preservation", "")).strip().casefold()
            == "complete captured frame"
        )
        if not complete_wire:
            raise ValueError("当前独立查看器要求 raw_preservation=complete captured frame")
        depth = int(metadata.get("sample_shape", [0])[-1]) - 3
        if depth != 997:
            raise ValueError(f"期望 payload 长度 997，实际为 {depth}")

        counts = [0, 0, 0, 0]
        for record in reader.iter_blocks():
            if record.header.sample_count != 1 or len(record.data) != 1:
                raise ValueError("raw Ethernet block 必须恰好包含一个 packet")
            decoded = decode_raw_ethernet_flags(record.header.flags)
            packet = np.asarray(record.data[0])
            if packet.ndim != 1 or packet.size != 1000:
                raise ValueError("raw Ethernet packet 必须是一维 1000 字节帧")
            if int(packet[0]) != 0 or int(packet[1]) != decoded.channel + 1 or int(packet[-1]) != 255:
                raise ValueError("raw Ethernet packet 头/通道/尾标记不匹配")
            counts[decoded.channel] += 1
        frame_count = min(counts)
        if frame_count == 0:
            raise ValueError(f"没有形成完整四通道帧，各通道 packet 数：{counts}")

        raw = np.empty((4, frame_count, depth), dtype=np.uint8)
        arrival = np.empty((4, frame_count), dtype=np.int64)
        ordinal = [0, 0, 0, 0]
        for record in reader.iter_blocks():
            decoded = decode_raw_ethernet_flags(record.header.flags)
            channel = decoded.channel
            position = ordinal[channel]
            ordinal[channel] += 1
            if position >= frame_count:
                continue
            packet = np.asarray(record.data[0])
            raw[channel, position] = packet[2:-1]
            arrival[channel, position] = int(record.header.host_monotonic_ns)

        # Use the first packet actually seen in the session as t=0.  The
        # displayed timestamp for a paired frame is the latest of its four
        # channel arrivals, matching Data Studio's playback convention.
        times_s = (np.max(arrival, axis=0) - int(np.min(arrival))) / 1e9
        channels_meta = metadata.get("channels")
        if isinstance(channels_meta, list) and len(channels_meta) == 4:
            channels = tuple(f"ch_{int(v)}" if isinstance(v, (int, float)) else str(v) for v in channels_meta)
        else:
            channels = ("ch_1", "ch_2", "ch_3", "ch_4")
        rate = float(metadata.get("nominal_frame_rate_hz") or 20.0)
    return raw, times_s.astype(np.float64), channels, rate


def load_session(path: str | Path) -> SessionData:
    session_dir = _find_session_dir(path)
    raw, times_s, channels, rate = _read_complete_frames(session_dir)
    features = compute_feature48(raw)
    return SessionData(session_dir, times_s, raw, features, channels, rate)


def run_self_test(path: str | Path) -> str:
    data = load_session(path)
    expected = (4, data.raw_adc.shape[1], 997)
    if data.raw_adc.shape != expected or data.features.shape != (4, expected[1], 48):
        raise AssertionError((data.raw_adc.shape, data.features.shape))
    return (
        f"OK: frames={expected[1]}, raw={data.raw_adc.shape}, "
        f"features={data.features.shape}, t=[{data.times_s[0]:.6f}, {data.times_s[-1]:.6f}]s"
    )


def configure_font(app) -> None:
    """Provide an explicit Chinese font, including in offscreen Qt sessions."""
    from PySide6.QtGui import QFont, QFontDatabase
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 9))


def launch_viewer(data: SessionData) -> int:
    from PySide6.QtCore import QSignalBlocker, QTimer, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    import pyqtgraph as pg
    from exo_collection.apps.data_studio.local_dialogs import _SweepWaterfallPlot

    class Window(QMainWindow):
        def __init__(self, initial: SessionData) -> None:
            super().__init__()
            self.data = initial
            self.index = 0
            self.playing = False
            self._current_time = float(self.data.times_s[0])
            self._last_tick = 0.0
            self._cycle_start: float | None = None
            self.setWindowTitle("US 原始波形 / 48 段特征瀑布图")
            self.resize(1500, 900)
            self._plots: list[object] = []
            self._build()
            self._timer = QTimer(self)
            self._timer.setInterval(50)
            self._timer.timeout.connect(self._advance)
            self._show_frame(0)

        def _build(self) -> None:
            root = QWidget()
            outer = QVBoxLayout(root)
            toolbar = QHBoxLayout()
            choose = QPushButton("选择 session")
            choose.clicked.connect(self._choose)
            self.play = QPushButton("▶ 播放")
            self.play.clicked.connect(self._toggle)
            prev = QPushButton("上一帧")
            prev.clicked.connect(lambda: self._show_frame(self.index - 1, pause=True))
            nxt = QPushButton("下一帧")
            nxt.clicked.connect(lambda: self._show_frame(self.index + 1, pause=True))
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, max(0, self.data.raw_adc.shape[1] - 1))
            self.slider.valueChanged.connect(self._slider_changed)
            self.status = QLabel()
            toolbar.addWidget(choose)
            toolbar.addWidget(self.play)
            toolbar.addWidget(prev)
            toolbar.addWidget(nxt)
            toolbar.addWidget(self.slider, 1)
            toolbar.addWidget(self.status)
            outer.addLayout(toolbar)
            panels = QHBoxLayout()
            left = QGroupBox("左：每帧处理后的 48 段均值（未标准化）")
            right = QGroupBox("右：原始 US ADC payload（997 点）")
            left_grid, right_grid = QGridLayout(left), QGridLayout(right)
            self._plots.clear()
            for channel in range(4):
                label = self.data.channels[channel]
                lp = _SweepWaterfallPlot(
                    f"{label} · 48 段特征",
                    self.data.times_s,
                    np.asarray(self.data.features[channel]).T,
                    10.0,
                )
                lp.setLabel("left", "特征段")
                lp.setYRange(0.0, 48.0, padding=0.0)
                rp = _SweepWaterfallPlot(
                    f"{label} · 原始 ADC",
                    self.data.times_s,
                    np.asarray(self.data.raw_adc[channel]).T,
                    10.0,
                )
                rp.setLabel("left", "深度点")
                rp.setYRange(0.0, 997.0, padding=0.0)
                self._plots.extend((lp, rp))
                left_grid.addWidget(lp, channel, 0)
                right_grid.addWidget(rp, channel, 0)
            panels.addWidget(left, 1)
            panels.addWidget(right, 1)
            outer.addLayout(panels, 1)
            self.setCentralWidget(root)

        @staticmethod
        def _reset_plot(plot: object) -> None:
            """Clear a reused sweep canvas on seek or a new 10 s cycle."""
            canvas = getattr(plot, "_canvas", None)
            if canvas is not None:
                canvas.fill(np.nan)
            setattr(plot, "_last_current", None)
            setattr(plot, "_last_cycle_start", None)
            image = getattr(plot, "image", None)
            if image is not None and canvas is not None:
                image.setImage(canvas, autoLevels=False, levels=getattr(plot, "_levels", (0.0, 1.0)))

        def _render_time(self, current_time: float, index: int) -> None:
            current_time = float(current_time)
            cycle_start = float(np.floor(current_time / 10.0) * 10.0)
            if self._cycle_start != cycle_start or current_time < self._current_time:
                for plot in self._plots:
                    self._reset_plot(plot)
            self._cycle_start = cycle_start
            self._current_time = current_time
            with QSignalBlocker(self.slider):
                self.slider.setValue(int(index))
            for plot in self._plots:
                plot.update_time(current_time, cycle_start)
            self.index = int(index)
            self.status.setText(
                f"帧 {self.index + 1}/{self.data.raw_adc.shape[1]} · t={current_time:.3f}s"
            )

        def _show_frame(self, index: int, pause: bool = False) -> None:
            count = self.data.raw_adc.shape[1]
            if pause:
                self._set_playing(False)
            bounded = min(max(int(index), 0), count - 1)
            with QSignalBlocker(self.slider):
                self.slider.setValue(bounded)
            self._render_time(float(self.data.times_s[bounded]), bounded)

        def _slider_changed(self, index: int) -> None:
            self._show_frame(index, pause=True)

        def _set_playing(self, value: bool) -> None:
            self.playing = bool(value)
            if self.playing:
                import time
                self._last_tick = time.perf_counter()
                self._timer.start()
                self.play.setText("⏸ 暂停")
            else:
                self._timer.stop()
                self.play.setText("▶ 播放")

        def _toggle(self) -> None:
            self._set_playing(not self.playing)

        def _advance(self) -> None:
            import time
            now = time.perf_counter()
            elapsed = max(0.0, now - self._last_tick)
            self._last_tick = now
            target = self._current_time + elapsed
            if target >= float(self.data.times_s[-1]):
                self._show_frame(self.data.times_s.size - 1)
                self._set_playing(False)
                return
            index = int(np.searchsorted(self.data.times_s, target, side="right") - 1)
            index = max(0, min(index, self.data.times_s.size - 1))
            self._render_time(target, index)

        def _choose(self) -> None:
            from exo_collection.apps.us_session_selector import select_us_session
            from exo_collection.configuration import SharedAppSettings

            self._set_playing(False)
            selected = select_us_session(SharedAppSettings().data_root, self)
            if not selected:
                return
            try:
                self.data = load_session(selected)
                self._set_playing(False)
                self._cycle_start = None
                self._build()
                self._show_frame(0)
            except Exception as exc:  # GUI boundary: concise error is enough.
                self.status.setText(f"加载失败：{exc}")

        def keyPressEvent(self, event) -> None:
            if event.key() == Qt.Key.Key_Space:
                self._toggle()
            elif event.key() == Qt.Key.Key_Left:
                self._show_frame(self.index - 1, pause=True)
            elif event.key() == Qt.Key.Key_Right:
                self._show_frame(self.index + 1, pause=True)
            else:
                super().keyPressEvent(event)

    app = QApplication.instance() or QApplication(sys.argv)
    configure_font(app)
    window = Window(data)
    window.show()
    preview_path = os.environ.get("US_FEATURE_VIEWER_PREVIEW")
    if preview_path:
        preview_index = int(np.searchsorted(data.times_s, 5.0, side="right") - 1)
        window._show_frame(max(0, preview_index))
        QTimer.singleShot(250, lambda: (window.grab().save(preview_path), app.quit()))
    return int(app.exec())


__all__ = ["SessionData", "compute_feature48", "load_session", "launch_viewer", "run_self_test"]
