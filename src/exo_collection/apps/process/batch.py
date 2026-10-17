"""Exo Process 的批量解算核心。

``solve_one_session`` 复刻 run_calculate 的「自动同步 → 预处理 → OpenSim → 导出真值」
四步，但不经过 Qt Worker，直接调用 ``opensim_joint_moment_pipeline`` 的纯函数，因此
单 session 解算逻辑可脱离 UI 单测。``BatchWorker`` 是唯一接触 Qt 的薄壳，把 session
队列串行喂给 ``solve_one_session`` 并回发进度/结果信号。
"""

from __future__ import annotations

import json
import hashlib
import math
import logging
import subprocess
from threading import Event, Thread
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal

from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path, pipeline_root
from exo_collection.apps.calculate.backpack_load import resolve_backpack_load
from exo_collection.apps.calculate.manual_review import clear_manual_review
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


class QCFailed(RuntimeError):
    """A completed run failed QC and must not replace the session training CSV."""
    def __init__(self, run_dir: Path, reasons: str = "") -> None:
        self.run_dir = run_dir
        super().__init__(f"QC FAIL：保留诊断 run {run_dir}，未覆盖 ground_truth.csv。{reasons}")


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
        export_validity_audit,
    )
    from exo_collection.apps.calculate.viewer import load_viewer_data
    from pipeline.synchronization.clock import (
        find_imu_sensor,
        imu_sensor_on_c3d_time,
        read_host_monotonic_ns,
    )

    viewer_dir = payload.get("viewer_dir")
    viewer_path = Path(viewer_dir) if viewer_dir else run_dir / "viewer"
    from exo_collection.apps.calculate.direction_policy import current_run_policy, load_run_manifest
    if not current_run_policy(load_run_manifest(run_dir)):
        raise ValueError("旧算法结果缺少当前外力/有效性策略，须重算后导出")
    data = load_viewer_data(viewer_path)
    if data.export_mask is None:
        raise ValueError("缺少逐帧审核/外力有效性依据")

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
        data.time_s, data.moments, imu_time_s, imu_signal, validity_mask=data.export_mask
    )
    out_path = dynamic.session_dir / "ground_truth.csv"
    write_ground_truth_csv(
        out_path, time_s, imu_aligned, moments, moment_names=data.moment_names
    )
    # Propagate export coverage into the final QC sidecar without changing CSV columns.
    payload["export_validity"] = export_validity_audit(data, time_s)
    return out_path


def load_confirmed_sync(dynamic: SessionRecord) -> dict[str, Any] | None:
    """Reuse only an applied/confirmed calibration whose four source hashes match.

    MANUAL_PAIRED historically enters SYNC_CONFIRMED immediately on Apply, so its
    persisted manual+n_pairs evidence is accepted even without confirmed_at_utc.
    Draft automatic MEDIUM/LOW candidates cannot pass this check.
    """
    ensure_pipeline_on_path()
    path = dynamic.session_dir / "derived" / "opensim" / "sync_calibration.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("dynamic_session_uuid") != dynamic.session_uuid or doc.get("trial_uuid") != dynamic.trial_uuid:
            return None
        files = dynamic.files
        inputs = {"c3d": files.c3d_path, "mocap_h5": files.mocap_h5_path,
                  "imu_h5": files.imu_h5_path, "gaitway_txt": files.txt_path}
        for key, filename in inputs.items():
            source = Path(filename)
            receipt = doc["inputs"][key]
            if Path(receipt["path"]).resolve() != source.resolve() or receipt.get("size_bytes") != source.stat().st_size:
                return None
            digest = hashlib.sha256()
            with source.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
            if receipt.get("sha256") != digest.hexdigest():
                return None
        result = doc["result"]
        method = str(doc.get("method", ""))
        manual_applied = (method == "MANUAL_PAIRED" and result.get("manual") is True
                          and int(result.get("n_pairs", 0)) >= 3)
        if not manual_applied and not doc.get("confirmed_at_utc"):
            return None
        if method not in {"MANUAL_PAIRED", "AUTO_HIGH", "EXPERT_FORCED"}:
            return None
        offset = float(result["gaitway_offset_s"])
        mad = result.get("mad_s")
        if manual_applied and mad is None:
            return None
        if not math.isfinite(offset) or (mad is not None and (not math.isfinite(float(mad)) or float(mad) < 0)):
            return None
        frame = result["c3d_start_in_mocap_h5_frame"]
        if isinstance(frame, bool) or not math.isfinite(float(frame)) or int(frame) != frame or int(frame) < 0:
            return None
        from pipeline.synchronization.c3d_h5 import is_reliable_match_rms
        if not is_reliable_match_rms(result.get("c3d_h5_match_rms_mm")):
            return None
        import h5py
        from pipeline.synchronization.clock import read_host_monotonic_ns
        with h5py.File(files.mocap_h5_path, "r") as mocap:
            if int(frame) >= len(read_host_monotonic_ns(mocap)):
                return None
        return {**result, "method": method, "calibration_reused": True,
                "calibration_source": str(path.resolve()), "confirmation_basis":
                "manual_applied" if manual_applied else "confirmed_at_utc",
                "confirmed_at_utc": doc.get("confirmed_at_utc"), "source_sha256_verified": True}
    except (OSError, ValueError, TypeError, KeyError):
        return None


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
    grf_cutoff_hz: float = 6.0,
    opensim_x_sign: float = -1.0,
    opensim_z_sign: float = -1.0,
    analysis_time_range_s: tuple[float, float] | None = None,
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
        sync = load_confirmed_sync(dynamic)
        if sync is None:
            sync = run_auto_sync(
                files.c3d_path,  # type: ignore[arg-type]
                files.mocap_h5_path,  # type: ignore[arg-type]
                files.imu_h5_path,  # type: ignore[arg-type]
                files.txt_path,  # type: ignore[arg-type]
            )
    except StompSyncError as exc:
        raise SyncFailed(str(exc)) from exc
    confidence = str(sync.get("confidence", "LOW")).upper()
    if confidence != "HIGH" and not sync.get("calibration_reused"):
        raise SyncFailed(f"同步置信度 {confidence}，需在 Calculate 中人工复核，本轮跳过")
    gaitway_offset_s = float(sync["gaitway_offset_s"])
    start_frame = int(sync["c3d_start_in_mocap_h5_frame"])
    progress("同步来源：" + ("复用已确认且来源匹配的标定" if sync.get("calibration_reused") else "自动 HIGH"))
    progress(f"同步完成：offset {gaitway_offset_s:.6f} s（{confidence}）")

    # 2) 预处理（EXO 环境）
    if cancel_check():
        raise SolveCancelled()
    progress("预处理：读取坡度、修正力/COP、生成 TRC/GRF…")
    from pipeline.opensim_io.prep_session import prepare_session

    run_dir = _new_run_directory(dynamic.session_dir / "derived" / "opensim")
    backpack = resolve_backpack_load(
        dynamic.condition_code,
        dynamic.condition_parameters,
        dynamic.session_dir,
    )
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
        sync_quality={**sync, "method": sync.get("method", "AUTO_HIGH")},
        analysis_time_range_s=analysis_time_range_s,
        quiet_standing=dynamic.is_quiet_standing,
        backpack=backpack,
        condition_code=dynamic.condition_code,
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

    qc = payload.get("qc") if isinstance(payload, dict) else None
    qc_status = qc.get("status") if isinstance(qc, dict) else None
    if qc_status == "FAIL":
        raise QCFailed(run_dir, str(qc.get("summary") or ""))

    # 4) 导出真值 CSV 到 session 目录
    if cancel_check():
        raise SolveCancelled()
    progress("导出真值：IMU 12 通道 + 关节力矩 → ground_truth.csv…")
    out_path = _export_ground_truth(dynamic, start_frame, run_dir, payload)
    progress(f"已导出：{out_path}")

    qc = payload.get("qc") if isinstance(payload, dict) else None
    qc_status = qc.get("status") if isinstance(qc, dict) else None
    from exo_collection.apps.calculate.ground_truth_status import write_ground_truth_qc

    write_ground_truth_qc(out_path, qc_status, run_dir, validity=payload.get("export_validity"))
    # 成功解算覆盖人工复核结论：清除「丢弃」标记（run_process 覆盖结果）。
    clear_manual_review(dynamic.session_dir)
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
        from exo_collection.apps.calculate.direction_policy import is_backward_code, export_is_current
        if not export_is_current(record.session_dir):
            return SessionSolveState.UNSOLVED
        from exo_collection.apps.calculate.ground_truth_status import read_ground_truth_qc

        status = read_ground_truth_qc(record.session_dir)
        if is_backward_code(record.condition_code) and status == "PASS":
            status = "WARN"
        return _qc_state(status)
    if files.mocap_h5_path is None or files.imu_h5_path is None:
        return SessionSolveState.MISSING_MODALITY
    return SessionSolveState.UNSOLVED


def erase_ground_truth(records: list[SessionRecord]) -> int:
    """删除每个 session 的 ``ground_truth.csv`` 及其 QC 附带文件，返回删除的 CSV 数。

    纯文件操作、不跑计算，可脱离 UI 单测。删除后 ``session_solve_status`` 会自动
    回退到「未解算」，便于重新解算。
    """
    removed = 0
    for record in records:
        csv_path = record.session_dir / "ground_truth.csv"
        qc_path = csv_path.with_suffix(".qc.json")
        if csv_path.is_file():
            try:
                csv_path.unlink()
                removed += 1
            except OSError as exc:
                _log.warning("删除 %s 失败：%s", csv_path, exc)
        if qc_path.is_file():
            try:
                qc_path.unlink()
            except OSError as exc:
                _log.warning("删除 %s 失败：%s", qc_path, exc)
    return removed


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
    "erase_ground_truth",
    "load_confirmed_sync",
    "QCFailed",
    "session_solve_status",
    "solve_one_session",
]
