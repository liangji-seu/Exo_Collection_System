"""Exo Process 的批量解算核心。

``solve_one_session`` 复刻 run_calculate 的「自动同步 → 预处理 → OpenSim → 导出真值」
四步，但不经过 Qt Worker，直接调用 ``opensim_joint_moment_pipeline`` 的纯函数，因此
单 session 解算逻辑可脱离 UI 单测。``BatchWorker`` 是唯一接触 Qt 的薄壳，把 session
队列串行喂给 ``solve_one_session`` 并回发进度/结果信号。
"""

from __future__ import annotations

import json
import logging
import subprocess
from threading import Event, Thread
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal

from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path, pipeline_root
from exo_collection.apps.calculate.models import SessionRecord

_log = logging.getLogger(__name__)


class SessionSolveState(str, Enum):
    """一个 session 在批量解算流程中的展示状态。"""

    INCOMPLETE = "INCOMPLETE"              # 缺 c3d / txt
    MISSING_MODALITY = "MISSING_MODALITY"  # 齐全但缺 mocap.h5 / imu.h5
    UNSOLVED = "UNSOLVED"                  # 齐全，未解算
    SOLVED = "SOLVED"                      # 已有 ground_truth.csv
    RUNNING = "RUNNING"                    # 解算中
    QC_PASS = "QC_PASS"
    QC_WARN = "QC_WARN"
    QC_FAIL = "QC_FAIL"
    DONE = "DONE"                          # 本轮解算成功
    FAILED = "FAILED"                      # 解算失败
    SYNC_FAILED = "SYNC_FAILED"            # 同步失败（非 HIGH 置信度 / 无跺脚峰）
    CANCELLED = "CANCELLED"                # 被取消


class SolveCancelled(RuntimeError):
    """用户请求取消批量解算。"""


class SyncFailed(RuntimeError):
    """跺脚自动同步失败（非 HIGH 置信度 / 无跺脚峰），本 session 跳过不产出真值。"""


def _new_run_directory(parent: Path) -> Path:
    """在 ``parent`` 下生成不覆盖旧 run 的 ``run_<ts>_<shortid>``（同 run_calculate）。"""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = datetime.now().strftime("%f")[:4]
    candidate = parent / f"run_{stamp}_{short_id}"
    index = 1
    while candidate.exists():
        candidate = parent / f"run_{stamp}_{short_id}_{index}"
        index += 1
    return candidate


def _parse_event(line: str) -> dict[str, Any] | None:
    """把一行解析成 ``process_session.py`` 的 JSON-Lines 事件；非 JSON 返回 None。"""
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except ValueError:
        return None
    if isinstance(obj, dict) and "event" in obj:
        return obj
    return None


def _run_opensim(
    opensim_python: Path,
    script: Path,
    manifest_path: Path,
    cancel_file: Path,
    cancel_check: Callable[[], bool],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """启动 OpenSim 子进程跑 ``process_session.py``，返回 ``result`` 事件 payload。"""
    cmd = [
        str(opensim_python),
        str(script),
        "--manifest",
        str(manifest_path),
        "--cancel-file",
        str(cancel_file),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=str(script.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    result_payload: dict[str, Any] | None = None
    error_message: str | None = None
    cancelled = False
    assert process.stdout is not None
    # Cancellation must reach the child even while stdout is silent.
    finished = Event()

    def watch_cancel() -> None:
        while not finished.is_set():
            if cancel_check():
                try:
                    cancel_file.write_text("cancel", encoding="utf-8")
                    return
                except OSError:
                    _log.exception("写入 OpenSim 取消文件失败")
            finished.wait(0.1)

    watcher = Thread(target=watch_cancel, name="opensim-cancel", daemon=True)
    watcher.start()
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            event = _parse_event(line)
            if event is None:
                continue
            kind = event.get("event")
            if kind == "stage":
                progress(f"[{event.get('stage', '')}] {event.get('message', '')}".rstrip())
            elif kind == "log":
                progress(str(event.get("message", "")))
            elif kind == "result":
                result_payload = event
            elif kind == "cancelled":
                cancelled = True
            elif kind == "error":
                error_message = str(event.get("message", ""))

        exit_code = process.wait()
    finally:
        finished.set()
        watcher.join()
        process.stdout.close()
    if cancelled or cancel_check():
        raise SolveCancelled()
    if exit_code != 0:
        raise RuntimeError(error_message or f"OpenSim 子进程退出码 {exit_code}")
    if result_payload is None:
        raise RuntimeError("OpenSim 子进程未输出 result 事件")
    return result_payload


def _export_ground_truth(
    dynamic: SessionRecord,
    start_frame: int,
    run_dir: Path,
    payload: dict[str, Any],
) -> Path:
    """把右腿 IMU 12 通道 + 关节力矩对齐到 C3D 时间后写 ``ground_truth.csv``。"""
    import h5py

    from exo_collection.apps.calculate.ground_truth import (
        align_ground_truth,
        write_ground_truth_csv,
    )
    from exo_collection.apps.calculate.viewer import load_viewer_data
    from pipeline.synchronization.clock import (
        find_imu_sensor,
        imu_sensor_on_c3d_time,
        read_host_monotonic_ns,
    )

    viewer_dir = payload.get("viewer_dir")
    viewer_path = Path(viewer_dir) if viewer_dir else run_dir / "viewer"
    data = load_viewer_data(viewer_path)

    files = dynamic.files
    with h5py.File(files.mocap_h5_path, "r") as mocap_h5, h5py.File(
        files.imu_h5_path, "r"
    ) as imu_h5:
        mocap_host_ns = read_host_monotonic_ns(mocap_h5)
        if not 0 <= start_frame < mocap_host_ns.size:
            raise ValueError(f"起始帧 {start_frame} 超出 mocap.h5 范围")
        c3d_t0_host_ns = int(mocap_host_ns[start_frame])
        sensor_index, _ = find_imu_sensor(imu_h5, side="right")
        imu_time_s, imu_signal = imu_sensor_on_c3d_time(
            imu_h5, c3d_t0_host_ns, sensor_index=sensor_index, axis_slice=slice(0, 12)
        )

    time_s, imu_aligned, moments = align_ground_truth(
        data.time_s, data.moments, imu_time_s, imu_signal
    )
    out_path = dynamic.session_dir / "ground_truth.csv"
    write_ground_truth_csv(
        out_path, time_s, imu_aligned, moments, moment_names=data.moment_names
    )
    return out_path


def solve_one_session(
    dynamic: SessionRecord,
    static: SessionRecord,
    opensim_python: Path,
    generic_model: Path,
    mass_kg: float,
    height_m: float,
    *,
    cancel_check: Callable[[], bool],
    progress: Callable[[str], None],
    marker_cutoff_hz: float = 6.0,
    grf_cutoff_hz: float = 20.0,
    opensim_x_sign: float = -1.0,
    opensim_z_sign: float = -1.0,
) -> dict[str, Any]:
    """解算单个 session（同步 → 预处理 → OpenSim → 导出真值），返回结果字典。

    同步置信度非 HIGH 或无跺脚峰时抛 :class:`SyncFailed`；用户取消抛
    :class:`SolveCancelled`；其它异常（输入缺失 / 预处理 / OpenSim / 导出）原样上抛。
    """
    files = dynamic.files
    if not files.has_dynamic_inputs:
        raise ValueError("动态 Session 输入不齐全")
    static_c3d = static.files.c3d_path
    if static_c3d is None:
        raise ValueError("静态标定 Session 缺少 C3D")

    ensure_pipeline_on_path()

    # 1) 自动同步（跺脚 3 次）
    if cancel_check():
        raise SolveCancelled()
    progress("自动同步：C3D↔mocap.h5 匹配 + 跺脚峰对齐…")
    from pipeline.synchronization.sync import StompSyncError, run_auto_sync

    try:
        sync = run_auto_sync(
            files.c3d_path,  # type: ignore[arg-type]
            files.mocap_h5_path,  # type: ignore[arg-type]
            files.imu_h5_path,  # type: ignore[arg-type]
            files.txt_path,  # type: ignore[arg-type]
        )
    except StompSyncError as exc:
        raise SyncFailed(str(exc)) from exc
    confidence = str(sync.get("confidence", "LOW")).upper()
    if confidence != "HIGH":
        raise SyncFailed(f"同步置信度 {confidence}，需在 Calculate 中人工复核，本轮跳过")
    gaitway_offset_s = float(sync["gaitway_offset_s"])
    start_frame = int(sync["c3d_start_in_mocap_h5_frame"])
    progress(f"同步完成：offset {gaitway_offset_s:.6f} s（{confidence}）")

    # 2) 预处理（EXO 环境）
    if cancel_check():
        raise SolveCancelled()
    progress("预处理：读取坡度、修正力/COP、生成 TRC/GRF…")
    from pipeline.opensim_io.prep_session import prepare_session

    run_dir = _new_run_directory(dynamic.session_dir / "derived" / "opensim")
    summary = prepare_session(
        static_c3d_path=static_c3d,
        dynamic_c3d_path=files.c3d_path,  # type: ignore[arg-type]
        gaitway_txt_path=files.txt_path,  # type: ignore[arg-type]
        generic_model_path=generic_model,
        out_dir=run_dir,
        subject_id=dynamic.subject_code,
        mass_kg=mass_kg,
        height_m=height_m,
        gaitway_offset_s=gaitway_offset_s,
        marker_cutoff_hz=marker_cutoff_hz,
        grf_cutoff_hz=grf_cutoff_hz,
        opensim_x_sign=opensim_x_sign,
        opensim_z_sign=opensim_z_sign,
        sync_confidence=confidence,
        sync_quality={**sync, "method": "AUTO_HIGH"},
    )
    manifest_path = Path(summary["manifest_path"])

    # 3) OpenSim 子进程（opensim 环境）
    if cancel_check():
        raise SolveCancelled()
    progress("OpenSim 解算（Scale/IK/ID）…")
    cancel_file = run_dir / "cancel.flag"
    script = pipeline_root() / "scripts" / "process_session.py"
    payload = _run_opensim(
        opensim_python, script, manifest_path, cancel_file, cancel_check, progress
    )

    # 4) 导出真值 CSV 到 session 目录
    if cancel_check():
        raise SolveCancelled()
    progress("导出真值：IMU 12 通道 + 关节力矩 → ground_truth.csv…")
    out_path = _export_ground_truth(dynamic, start_frame, run_dir, payload)
    progress(f"已导出：{out_path}")

    qc = payload.get("qc") if isinstance(payload, dict) else None
    qc_status = qc.get("status") if isinstance(qc, dict) else None
    from exo_collection.apps.calculate.ground_truth_status import write_ground_truth_qc

    write_ground_truth_qc(out_path, qc_status, run_dir)
    progress(f"计算完成，QC：{qc_status or '未评估'}")
    return {
        "run_dir": run_dir,
        "viewer_dir": payload.get("viewer_dir"),
        "qc_status": qc_status,
        "sync_confidence": confidence,
        "out_path": out_path,
    }


def _qc_state(status: str | None) -> SessionSolveState:
    return {
        "PASS": SessionSolveState.QC_PASS,
        "WARN": SessionSolveState.QC_WARN,
        "FAIL": SessionSolveState.QC_FAIL,
    }.get(status, SessionSolveState.SOLVED)


def session_solve_status(record: SessionRecord) -> SessionSolveState:
    """只读判定一个 session 当前应展示的解算状态（不跑任何计算）。"""
    files = record.files
    complete = files.c3d_path is not None and files.txt_path is not None
    if not complete:
        return SessionSolveState.INCOMPLETE
    if (record.session_dir / "ground_truth.csv").is_file():
        from exo_collection.apps.calculate.ground_truth_status import read_ground_truth_qc

        return _qc_state(read_ground_truth_qc(record.session_dir))
    if files.mocap_h5_path is None or files.imu_h5_path is None:
        return SessionSolveState.MISSING_MODALITY
    return SessionSolveState.UNSOLVED


class _BatchSignals(QObject):
    session_started = Signal(str)             # session_name
    session_finished = Signal(str, str, str)  # name, SessionSolveState.value, out_path
    session_failed = Signal(str, str, str)    # name, SessionSolveState.value, message
    progress = Signal(str)
    all_done = Signal(int, int, int)          # total, ok, failed


class BatchWorker(QRunnable):
    """顺序解算一批 session；单个失败不中断整批，通过信号回发进度/结果。"""

    def __init__(
        self,
        sessions: list[SessionRecord],
        static: SessionRecord,
        opensim_python: Path,
        generic_model: Path,
        *,
        mass_kg: float | None = None,
        height_m: float | None = None,
    ) -> None:
        super().__init__()
        self._sessions = list(sessions)
        self._static = static
        self._opensim_python = Path(opensim_python)
        self._generic_model = Path(generic_model)
        self._mass_kg = mass_kg
        self._height_m = height_m
        self.signals = _BatchSignals()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def _cancel_check(self) -> bool:
        return self._cancel_requested

    def _patient_info(self, record: SessionRecord) -> tuple[float, float]:
        """优先用显式覆盖，否则从该 session 的测力台 TXT 头部读（失败回退默认）。"""
        mass_kg = self._mass_kg
        height_m = self._height_m
        if mass_kg is None or height_m is None:
            info: dict[str, Any] = {}
            if record.files.txt_path is not None:
                try:
                    ensure_pipeline_on_path()
                    from pipeline.gaitway import read_gaitway_patient_info

                    info = read_gaitway_patient_info(record.files.txt_path) or {}
                except Exception as exc:  # noqa: BLE001
                    _log.warning("读取测力台个人信息失败 %s: %s", record.session_name, exc)
            if mass_kg is None:
                weight = info.get("weight_kg")
                mass_kg = float(weight) if isinstance(weight, (int, float)) else 75.0
            if height_m is None:
                height = info.get("height_m")
                height_m = float(height) if isinstance(height, (int, float)) else 1.75
        return float(mass_kg), float(height_m)

    def run(self) -> None:
        total = len(self._sessions)
        ok = 0
        failed = 0
        for record in self._sessions:
            if self._cancel_requested:
                self.signals.progress.emit("已取消，停止后续 session。")
                break
            name = record.session_name
            self.signals.session_started.emit(name)
            self.signals.progress.emit(f"── {name} ──")
            mass_kg, height_m = self._patient_info(record)
            try:
                result = solve_one_session(
                    record,
                    self._static,
                    self._opensim_python,
                    self._generic_model,
                    mass_kg,
                    height_m,
                    cancel_check=self._cancel_check,
                    progress=self.signals.progress.emit,
                )
                ok += 1
                self.signals.session_finished.emit(
                    name, _qc_state(result.get("qc_status")).value, str(result["out_path"])
                )
            except SolveCancelled:
                self.signals.session_failed.emit(
                    name, SessionSolveState.CANCELLED.value, "已取消"
                )
                break
            except SyncFailed as exc:
                failed += 1
                self.signals.progress.emit(f"  同步失败：{exc}")
                self.signals.session_failed.emit(
                    name, SessionSolveState.SYNC_FAILED.value, str(exc)
                )
            except Exception as exc:  # noqa: BLE001 —— 线程边界，必须捕获
                failed += 1
                _log.exception("批量解算失败：%s", name)
                self.signals.progress.emit(f"  失败：{exc}")
                self.signals.session_failed.emit(
                    name, SessionSolveState.FAILED.value, str(exc)
                )
        self.signals.all_done.emit(total, ok, failed)


__all__ = [
    "BatchWorker",
    "SessionSolveState",
    "SolveCancelled",
    "SyncFailed",
    "session_solve_status",
    "solve_one_session",
]
