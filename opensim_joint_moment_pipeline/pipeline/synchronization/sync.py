"""自动同步编排：C3D↔mocap.h5 → 主机时钟 → IMU↔Gaitway 跺脚。

把四个输入（``.c3d`` / ``mocap.h5`` / ``imu.h5`` / Gaitway ``.txt``）串成一条
时钟链，输出一个 JSON 可序列化的结果字典（含每步的数值、置信度与审计信息），
供 UI 与 CLI 共用。不 import Qt / matplotlib，保证能直接用在后台 worker。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..c3d.reader import read_c3d
from ..gaitway import read_gaitway_ascii
from .c3d_h5 import match_c3d_to_h5
from .clock import (
    clock_health,
    find_imu_sensor,
    imu_sample_rate_hz,
    imu_sensor_on_c3d_time,
    read_host_monotonic_ns,
)
from .stomp import highpass_envelope, pair_stomps_diagnosed, StompRejection


def _read_h5_marker_names(handle) -> list[str]:
    raw = handle["metadata/device"][()]
    if isinstance(raw, (bytes, bytearray)):
        raw = json.loads(raw.decode("utf-8"))
    return list(raw.get("marker_names", []))


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class StompSyncError(RuntimeError):
    """跺脚自动同步失败，携带可审计的结构化诊断（供 UI 展示并转入人工标定）。

    无跺脚的 Session 不能伪造成功：``diagnostics`` 里的两侧峰数、爆发段数与
    拒绝原因就是「确实没有足够同步动作」的数值证据（prompt6 §3.1 第 7 条）。
    """

    def __init__(self, message: str, diagnostics: dict) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _format_rejection(rej: StompRejection) -> str:
    """把 ``StompRejection`` 转成带数值的可读说明（单行，供 UI/日志展示）。"""
    def side(label: str, peak_count: int, burst_count: int, reason: str) -> str:
        return f"{label}：检测到 {peak_count} 个峰、选中 {burst_count} 个爆发段（原因：{reason}）"

    parts = [
        "跺脚峰不足 3 对，无法自动同步（需人工标定）",
        side("IMU", rej.imu_peak_count, rej.imu_burst_count, rej.imu_reason),
        side("Gaitway", rej.gaitway_peak_count, rej.gaitway_burst_count, rej.gaitway_reason),
    ]
    if rej.align_reason is not None:
        parts.append(rej.align_reason)
    return "；".join(parts)


def run_auto_sync(
    c3d_path: str | Path,
    mocap_h5_path: str | Path,
    imu_h5_path: str | Path,
    gaitway_txt_path: str | Path,
    *,
    final_adjustment_ms: float = 0.0,
    preferred_markers: list[str] | None = None,
    prominence: float = 0.05,
) -> dict:
    """运行自动同步，返回结果字典（失败抛异常；由调用方决定是否转人工）。

    返回字典关键字段与 ``sync_calibration.json`` 一致，可直接序列化。
    """
    c3d = read_c3d(c3d_path)
    gaitway = read_gaitway_ascii(gaitway_txt_path)

    with h5py.File(mocap_h5_path, "r") as mocap_h5, h5py.File(imu_h5_path, "r") as imu_h5:
        h5_names = _read_h5_marker_names(mocap_h5)
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
        mocap_health = clock_health(mocap_host_ns)
        mocap_period_ms = float(mocap_health.median_period_ns) / 1e6

        sensor_index, sensor_label = find_imu_sensor(imu_h5, side="right")
        imu_host_ns = read_host_monotonic_ns(imu_h5)
        imu_health = clock_health(imu_host_ns)
        imu_time_c3d = (imu_host_ns - c3d_t0_host_ns) / 1e9
        acc = np.asarray(imu_h5["samples/data"][:, sensor_index, :3], dtype=np.float64)
        imu_rate = imu_sample_rate_hz(imu_time_c3d)
        acc_norm = np.linalg.norm(acc, axis=1)
        imu_envelope = highpass_envelope(acc_norm, imu_rate)

    total_fz = gaitway.columns["GRFz vertical (N)"]
    force_rate = gaitway.sample_rate_hz
    force_envelope = highpass_envelope(total_fz, force_rate)

    alignment, rejection = pair_stomps_diagnosed(
        imu_time_c3d, imu_envelope, gaitway.time_s, force_envelope, prominence=prominence
    )
    if alignment is None:
        raise StompSyncError(
            _format_rejection(rejection),
            {
                "imu": {
                    "peak_count": rejection.imu_peak_count,
                    "burst_count": rejection.imu_burst_count,
                    "reason": rejection.imu_reason,
                },
                "gaitway": {
                    "peak_count": rejection.gaitway_peak_count,
                    "burst_count": rejection.gaitway_burst_count,
                    "reason": rejection.gaitway_reason,
                },
                "align_reason": rejection.align_reason,
            },
        )

    final_offset = alignment.median_offset_s + final_adjustment_ms / 1000.0
    peak_pairs = [
        {
            "index": k + 1,
            "imu_time_on_c3d_s": p.imu_time_s,
            "gaitway_time_s": p.gaitway_time_s,
            "offset_s": p.offset_s,
        }
        for k, p in enumerate(alignment.pairs)
    ]

    # C3D↔mocap.h5 匹配不是 exact+unique 时，不得直接贡献 HIGH 总置信度
    # （prompt6 §3.7 第 5 条）：跺脚 MAD 再小，匹配不唯一也最多 MEDIUM。
    confidence = alignment.confidence
    if confidence == "HIGH" and not (match.exact and match.unique):
        confidence = "MEDIUM"

    return {
        "c3d_start_in_mocap_h5_frame": match.start_frame,
        "c3d_h5_match_rms_mm": match.rms_mm,
        "c3d_h5_match_max_error_mm": match.max_error_mm,
        "c3d_h5_overlap_frames": match.overlap_frames,
        "c3d_h5_exact": match.exact,
        "c3d_h5_unique": match.unique,
        "c3d_h5_second_best_rms_mm": match.second_best_rms_mm,
        "c3d_h5_second_best_frame": match.second_best_frame,
        "c3d_h5_matched_markers": list(match.matched_markers),
        "c3d_t0_host_monotonic_ns": c3d_t0_host_ns,
        "mocap_h5_period_ms": mocap_period_ms,
        "mocap_h5_monotonic": mocap_health.monotonic,
        "mocap_h5_clock_gaps": mocap_health.n_gaps,
        "imu_sensor_index": sensor_index,
        "imu_sensor_label": sensor_label,
        "imu_sample_rate_hz": imu_rate,
        "imu_clock_monotonic": imu_health.monotonic,
        "imu_clock_gaps": imu_health.n_gaps,
        "gaitway_sample_rate_hz": force_rate,
        "imu_peak_times_on_c3d_s": [p.imu_time_s for p in alignment.pairs],
        "gaitway_peak_times_s": [p.gaitway_time_s for p in alignment.pairs],
        "peak_pairs": peak_pairs,
        "offsets_s": alignment.offsets_s.tolist(),
        "median_offset_s": alignment.median_offset_s,
        "mad_s": alignment.mad_s,
        "drift_ppm": alignment.drift_ppm,
        "scale_a": alignment.scale_a,
        "n_pairs": len(alignment.pairs),
        "confidence": confidence,
        "final_adjustment_ms": final_adjustment_ms,
        "gaitway_offset_s": final_offset,
    }


def _json_safe(obj: Any) -> Any:
    """递归把非有限浮点（inf/NaN）替换为 None，保证 ``allow_nan=False`` 可序列化。"""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_sync_calibration(
    out_dir: str | Path,
    result: dict,
    *,
    inputs: dict[str, str | Path],
    operator: str = "auto",
    note: str | None = None,
    method: str = "AUTO_HIGH",
    dynamic_session_uuid: str | None = None,
    trial_uuid: str | None = None,
    auto_candidate: dict | None = None,
    operator_type: str | None = None,
    confirmed_at: str | None = None,
    adjusted: bool = False,
) -> Path:
    """把同步结论 + 输入指纹写入 ``sync_calibration.json``（派生 sidecar）。

    输入变化（SHA-256 不同）会使旧标定失效，UI 据此提示重新标定。文档字段对齐
    prompt6 §3.2 第 7 条：session/trial UUID、四输入路径/大小/mtime/SHA-256、
    自动候选结果、最终采用结果、方法、峰对/offset/MAD/confidence/操作者说明、
    坐标与时间方向约定、schema 版本。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    inputs_meta: dict[str, Any] = {}
    for name, path in inputs.items():
        p = Path(path)
        try:
            stat = p.stat()
            size = int(stat.st_size)
            mtime = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()
        except OSError:
            size = None
            mtime = None
        inputs_meta[name] = {
            "path": str(p),
            "size_bytes": size,
            "mtime_utc": mtime,
            "sha256": _file_sha256(p) if p.is_file() else None,
        }

    document = {
        "schema_version": "1.1.0",
        "dynamic_session_uuid": dynamic_session_uuid,
        "trial_uuid": trial_uuid,
        "inputs": inputs_meta,
        "method": method,
        "operator": operator,
        "operator_type": operator_type,
        "confirmed_at_utc": confirmed_at,
        "adjusted": adjusted,
        "note": note,
        "auto_candidate": _json_safe(auto_candidate),
        "time_direction_convention": "t_gaitway = t_c3d + gaitway_offset_s",
        "coordinate_convention": "offset = t_gaitway - t_host",
        "result": _json_safe(result),
    }
    target = out / "sync_calibration.json"
    target.write_text(
        json.dumps(document, indent=2, allow_nan=False), encoding="utf-8"
    )
    return target


__all__ = ["StompSyncError", "run_auto_sync", "save_sync_calibration"]
