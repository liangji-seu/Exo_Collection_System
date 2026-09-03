"""Exo Calculate 的后台 Worker（QRunnable + 信号）。

所有大文件读取与 OpenSim 子进程调用都在后台线程进行，绝不阻塞 Qt 主线程。
计算函数（``run_auto_sync`` / ``read_c3d`` / ``read_gaitway_ascii`` …）在
``opensim_joint_moment_pipeline`` 里，只在 ``run()`` 内惰性 import，避免主
界面进程在 import 阶段就拉进 ezc3d / 大数组。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path

_log = logging.getLogger(__name__)


class _WorkerSignals(QObject):
    finished = Signal(object)   # 成功结果（类型由各 Worker 定义）
    failed = Signal(str)        # 失败消息
    progress = Signal(str)      # 进度 / 阶段消息
    result = Signal(object)     # 结构化结果（OpenSim 子进程 JSON-Lines ``result`` 事件）
    cancelled = Signal()        # OpenSim 子进程收到取消信号


@dataclass
class SyncDataBundle:
    """手动标定视图需要的全部原始信号（已映射到统一时间轴）。"""

    c3d_t0_host_ns: int
    c3d_h5_match_rms_mm: float
    c3d_h5_start_frame: int
    imu_sensor_label: str
    imu_time_s: np.ndarray          # 映射到 C3D 时间的 IMU 时间轴
    imu_acc: np.ndarray             # (n, 3) 右腿 IMU 加速度 XYZ
    imu_acc_norm: np.ndarray        # 加速度模长
    imu_envelope: np.ndarray        # 高通包络
    gaitway_time_s: np.ndarray
    gaitway_total_fz: np.ndarray
    gaitway_fz_left: np.ndarray
    gaitway_fz_right: np.ndarray
    marker_time_s: np.ndarray | None = None
    marker_acc_norm: np.ndarray | None = None
    marker_name: str | None = None


def _load_sync_data(
    c3d_path: Path,
    mocap_h5_path: Path,
    imu_h5_path: Path,
    gaitway_txt_path: Path,
    *,
    preferred_markers: list[str] | None = None,
) -> SyncDataBundle:
    """加载手动标定所需的信号（在后台线程调用）。

    复用同步引擎的 C3D↔mocap.h5 精确匹配得到 ``c3d_t0_host_ns``，从而把 IMU
    映射到 C3D 时间；Gaitway 保留自身时间轴，由用户在界面上对齐跺脚峰。
    """
    import h5py
    import json as _json

    ensure_pipeline_on_path()
    from pipeline.c3d.reader import read_c3d
    from pipeline.gaitway import read_gaitway_ascii
    from pipeline.synchronization.c3d_h5 import match_c3d_to_h5
    from pipeline.synchronization.clock import (
        find_imu_sensor,
        read_host_monotonic_ns,
    )
    from pipeline.synchronization.stomp import highpass_envelope

    c3d = read_c3d(c3d_path)
    gaitway = read_gaitway_ascii(gaitway_txt_path)

    with h5py.File(mocap_h5_path, "r") as mocap_h5, h5py.File(imu_h5_path, "r") as imu_h5:
        device = mocap_h5["metadata/device"][()]
        if isinstance(device, (bytes, bytearray)):
            device = _json.loads(device.decode("utf-8"))
        h5_names = [str(n) for n in device.get("marker_names", [])]
        h5_points = mocap_h5["samples/data"][:]
        match = match_c3d_to_h5(
            c3d.points_mm,
            c3d.point_labels,
            h5_points,
            h5_names,
            preferred_markers=preferred_markers,
        )
        mocap_host_ns = read_host_monotonic_ns(mocap_h5)
        c3d_t0_host_ns = int(mocap_host_ns[match.start_frame])

        sensor_index, sensor_label = find_imu_sensor(imu_h5, side="right")
        imu_host_ns = read_host_monotonic_ns(imu_h5)
        imu_time_s = (imu_host_ns - c3d_t0_host_ns) / 1e9
        acc = np.asarray(imu_h5["samples/data"][:, sensor_index, :3], dtype=np.float64)
        acc_norm = np.linalg.norm(acc, axis=1)
        # 采样率由映射后的时间轴估计
        if imu_time_s.size >= 2:
            rate = 1.0 / float(np.median(np.diff(imu_time_s)))
        else:
            rate = 100.0
        imu_envelope = highpass_envelope(acc_norm, rate)

    total_fz = gaitway.columns["GRFz vertical (N)"]
    fz_left = gaitway.columns["FzL(N)"]
    fz_right = gaitway.columns["FzR(N)"]

    # 第三证据：脚踝/脚跟 marker 加速度（可选，找到就用）。C3D label 可能带
    # subject 前缀（如 ``003_no_exo_dynamic:R.Ankle``），需规范化后匹配。
    marker_time_s = None
    marker_acc_norm = None
    marker_name = None
    from pipeline.synchronization.marker_names import normalize_marker_name

    label_to_index = {
        normalize_marker_name(label): idx for idx, label in enumerate(c3d.point_labels)
    }
    for candidate in ("R.Ankle", "R.Heel", "L.Ankle", "L.Heel"):
        if candidate in label_to_index:
            idx = label_to_index[candidate]
            pos = c3d.points_mm[:, idx, :]
            vel = np.gradient(pos, axis=0) * c3d.point_rate_hz
            accel = np.gradient(vel, axis=0) * c3d.point_rate_hz
            marker_acc_norm = np.linalg.norm(accel, axis=1)
            marker_time_s = c3d.time_s
            marker_name = candidate
            break

    return SyncDataBundle(
        c3d_t0_host_ns=c3d_t0_host_ns,
        c3d_h5_match_rms_mm=match.rms_mm,
        c3d_h5_start_frame=match.start_frame,
        imu_sensor_label=sensor_label,
        imu_time_s=imu_time_s,
        imu_acc=acc,
        imu_acc_norm=acc_norm,
        imu_envelope=imu_envelope,
        gaitway_time_s=gaitway.time_s,
        gaitway_total_fz=total_fz,
        gaitway_fz_left=fz_left,
        gaitway_fz_right=fz_right,
        marker_time_s=marker_time_s,
        marker_acc_norm=marker_acc_norm,
        marker_name=marker_name,
    )


class SyncWorker(QRunnable):
    """后台运行自动同步（``run_auto_sync``）。"""

    def __init__(
        self,
        c3d_path: Path,
        mocap_h5_path: Path,
        imu_h5_path: Path,
        gaitway_txt_path: Path,
        *,
        preferred_markers: list[str] | None = None,
        prominence: float = 0.05,
        final_adjustment_ms: float = 0.0,
    ) -> None:
        super().__init__()
        self._c3d = Path(c3d_path)
        self._mocap = Path(mocap_h5_path)
        self._imu = Path(imu_h5_path)
        self._gaitway = Path(gaitway_txt_path)
        self._preferred = preferred_markers
        self._prominence = prominence
        self._final_adjustment_ms = final_adjustment_ms
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            ensure_pipeline_on_path()
            from pipeline.synchronization.sync import run_auto_sync

            self.signals.progress.emit("自动同步：C3D↔mocap.h5 匹配…")
            result = run_auto_sync(
                self._c3d,
                self._mocap,
                self._imu,
                self._gaitway,
                preferred_markers=self._preferred,
                prominence=self._prominence,
                final_adjustment_ms=self._final_adjustment_ms,
            )
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 —— 线程边界，必须捕获
            _log.exception("自动同步失败")
            self.signals.failed.emit(str(exc))


class LoadSyncDataWorker(QRunnable):
    """后台加载手动标定所需的全部原始信号。"""

    def __init__(
        self,
        c3d_path: Path,
        mocap_h5_path: Path,
        imu_h5_path: Path,
        gaitway_txt_path: Path,
        *,
        preferred_markers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._c3d = Path(c3d_path)
        self._mocap = Path(mocap_h5_path)
        self._imu = Path(imu_h5_path)
        self._gaitway = Path(gaitway_txt_path)
        self._preferred = preferred_markers
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            ensure_pipeline_on_path()
            bundle = _load_sync_data(
                self._c3d,
                self._mocap,
                self._imu,
                self._gaitway,
                preferred_markers=self._preferred,
            )
            self.signals.finished.emit(bundle)
        except Exception as exc:  # noqa: BLE001
            _log.exception("加载手动标定数据失败")
            self.signals.failed.emit(str(exc))


class PrepWorker(QRunnable):
    """后台预处理：C3D/Gaitway → TRC/GRF/manifest（EXO 环境，不 import opensim）。

    产出写到 ``out_dir``（一个 ASCII 工作目录），返回 ``prepare_session`` 的摘要
    （含 ``manifest_path``）。与 OpenSim 解耦，先跑这步再跑 ``OpenSimProcessWorker``。
    """

    def __init__(
        self,
        *,
        static_c3d: Path,
        dynamic_c3d: Path,
        gaitway_txt: Path,
        generic_model: Path,
        out_dir: Path,
        subject_id: str,
        mass_kg: float,
        height_m: float,
        gaitway_offset_s: float,
        marker_cutoff_hz: float | None = None,
        grf_cutoff_hz: float | None = None,
        opensim_x_sign: float = -1.0,
        opensim_z_sign: float = -1.0,
        analysis_time_range_s: tuple[float, float] | None = None,
        static_time_range_s: tuple[float, float] | None = None,
        sync_confidence: str | None = None,
        sync_quality: dict[str, Any] | None = None,
        marker_adjustment_expert_confirmed: bool = False,
    ) -> None:
        super().__init__()
        self._static_c3d = Path(static_c3d)
        self._dynamic_c3d = Path(dynamic_c3d)
        self._gaitway_txt = Path(gaitway_txt)
        self._generic_model = Path(generic_model)
        self._out_dir = Path(out_dir)
        self._subject_id = subject_id
        self._mass_kg = mass_kg
        self._height_m = height_m
        self._gaitway_offset_s = gaitway_offset_s
        self._marker_cutoff = marker_cutoff_hz
        self._grf_cutoff = grf_cutoff_hz
        self._opensim_x_sign = opensim_x_sign
        self._opensim_z_sign = opensim_z_sign
        self._analysis_time_range = analysis_time_range_s
        self._static_time_range = static_time_range_s
        self._sync_confidence = sync_confidence
        self._sync_quality = sync_quality
        self._marker_adjustment_expert_confirmed = marker_adjustment_expert_confirmed
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            ensure_pipeline_on_path()
            from pipeline.opensim_io.prep_session import prepare_session

            self.signals.progress.emit("预处理：C3D→TRC、Gaitway→GRF、写 manifest…")
            summary = prepare_session(
                static_c3d_path=self._static_c3d,
                dynamic_c3d_path=self._dynamic_c3d,
                gaitway_txt_path=self._gaitway_txt,
                generic_model_path=self._generic_model,
                out_dir=self._out_dir,
                subject_id=self._subject_id,
                mass_kg=self._mass_kg,
                height_m=self._height_m,
                gaitway_offset_s=self._gaitway_offset_s,
                marker_cutoff_hz=self._marker_cutoff,
                grf_cutoff_hz=self._grf_cutoff,
                opensim_x_sign=self._opensim_x_sign,
                opensim_z_sign=self._opensim_z_sign,
                analysis_time_range_s=self._analysis_time_range,
                static_time_range_s=self._static_time_range,
                sync_confidence=self._sync_confidence,
                sync_quality=self._sync_quality,
                marker_adjustment_expert_confirmed=self._marker_adjustment_expert_confirmed,
            )
            self.signals.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001 —— 线程边界，必须捕获
            _log.exception("预处理失败")
            self.signals.failed.emit(str(exc))


def _finalize_outcome(
    exit_code: int,
    cancelled: bool,
    cancel_requested: bool,
    result_payload: dict[str, Any] | None,
    error_message: str | None,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """把子进程结束状态映射为 worker 结局（纯函数，可脱离子进程测试）。

    用户取消意图优先：即使子进程被强制 terminate（退出码非 0）、来不及输出
    cancelled 事件，只要父进程已请求取消就判 ``cancelled`` 而非 ``failed``
    （prompt6 §3.10 第 3 条）。
    """
    if cancel_requested or cancelled:
        return "cancelled", None, None
    if exit_code == 0:
        return "finished", result_payload or {"exit_code": 0}, None
    return "failed", None, error_message or f"OpenSim 子进程退出码 {exit_code}"


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


class OpenSimProcessWorker(QRunnable):
    """后台启动 OpenSim 子环境运行 ``process_session.py``。

    子进程按 JSON-Lines 输出进度/结果事件；本 Worker 解析后分别发 ``progress``、
    ``result``、``cancelled``、``failed``。OpenSim 只在子进程中 import。
    """

    def __init__(
        self,
        opensim_python: Path,
        script_path: Path,
        arguments: list[str],
    ) -> None:
        super().__init__()
        self._opensim_python = Path(opensim_python)
        self._script_path = Path(script_path)
        self._arguments = list(arguments)
        self.signals = _WorkerSignals()
        self._process: subprocess.Popen | None = None
        self._cancel_requested = False

    def run(self) -> None:
        cmd = [str(self._opensim_python), str(self._script_path), *self._arguments]
        self.signals.progress.emit("启动 OpenSim 子进程…")
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self._script_path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self.signals.failed.emit(f"无法启动 OpenSim 子进程：{exc}")
            return

        assert self._process.stdout is not None
        result_payload: dict[str, Any] | None = None
        error_message: str | None = None
        cancelled = False
        for line in self._process.stdout:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            event = _parse_event(line)
            if event is None:
                # OpenSim 原始输出（非 JSON）原样透传给进度日志
                self.signals.progress.emit(line)
                continue
            kind = event.get("event")
            if kind == "stage":
                stage = event.get("stage", "")
                message = event.get("message", "")
                self.signals.progress.emit(f"[{stage}] {message}".rstrip())
            elif kind == "log":
                self.signals.progress.emit(str(event.get("message", "")))
            elif kind == "result":
                result_payload = event
            elif kind == "cancelled":
                cancelled = True
            elif kind == "error":
                error_message = str(event.get("message", ""))
            # ``start`` 事件不单独展示

        exit_code = self._process.wait()
        outcome, payload, message = _finalize_outcome(
            exit_code, cancelled, self._cancel_requested, result_payload, error_message
        )
        if outcome == "cancelled":
            self.signals.cancelled.emit()
        elif outcome == "finished":
            self.signals.finished.emit(payload)
        else:
            self.signals.failed.emit(message)

    def cancel(self) -> None:
        """请求协作式取消：仅置位标记，等待子进程自行输出 cancelled 事件。

        调用方应在一段宽限期后再 ``terminate``/``kill``（超时兜底），父进程仍会
        因 ``_cancel_requested`` 为真而把结局标为 CANCELLED。
        """
        self._cancel_requested = True

    def terminate(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def kill(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()


__all__ = [
    "LoadSyncDataWorker",
    "OpenSimProcessWorker",
    "PrepWorker",
    "SyncDataBundle",
    "SyncWorker",
]
