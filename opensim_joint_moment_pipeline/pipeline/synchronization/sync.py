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
    imu_sample_rate_hz,
    imu_sensor_candidates,
    read_host_monotonic_ns,
)
from .stomp import highpass_envelope, pair_stomps_diagnosed


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


def _format_multi_rejection(
    sensor_envs: list[dict[str, Any]],
    force_names: list[str],
    attempts: list[dict[str, Any]],
) -> str:
    """把多组合跺脚失败总结成带数值的可读说明（单行，供 UI/日志展示）。

    自动同步会逐一尝试「每个 IMU 传感器 × 每个测力台信号」的组合；全部失败时，
    这里先给一句总量概述，再展开历史默认组合（右腿 × 总垂直力）的峰数/爆发段数
    与拒绝原因，作为人工标定的数值证据。
    """
    head = (
        f"跺脚自动同步失败：已尝试 {len(sensor_envs)} 个 IMU 传感器 × "
        f"{len(force_names)} 个测力台信号组合，均无法找到 ≥3 对跺脚峰（需人工标定）"
    )
    detail = next(
        (
            a
            for a in attempts
            if a["imu"] == "imu_right_leg" and a["force"] == "GRFz vertical (N)"
        ),
        attempts[0] if attempts else None,
    )
    if detail is None:
        return head
    parts = [
        head,
        (
            f"参考组合 {detail['imu']}×{detail['force']}："
            f"IMU {detail['imu_peak_count']} 峰/{detail['imu_burst_count']} 段"
            f"（{detail['imu_reason']}）；"
            f"测力台 {detail['gaitway_peak_count']} 峰/{detail['gaitway_burst_count']} 段"
            f"（{detail['gaitway_reason']}）"
        ),
    ]
    if detail.get("align_reason"):
        parts.append(detail["align_reason"])
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

        # 枚举所有 IMU 传感器，各自算出「加速度模长 → 高通冲击包络」。MTw 直连
        # USB 的 samples/data 是多个独立设备流的并集（每行只属于一个传感器、其余
        # 槽位为 NaN），所以每个传感器只取自己的有效行；否则 100 Hz 的传感器会被
        # 混叠成 ~300 Hz 并误报丢帧（详见 clock.py 注释）。自动同步不再只依赖右腿，
        # 而是把所有可用传感器（左腿/右腿/盆骨…）都作为候选，由跺脚配对择优。
        imu_host_all_ns = read_host_monotonic_ns(imu_h5)
        imu_data = imu_h5["samples/data"]
        sensor_envs: list[dict[str, Any]] = []
        for idx, label in imu_sensor_candidates(imu_h5):
            acc = np.asarray(imu_data[:, idx, :3], dtype=np.float64)
            valid = np.isfinite(acc).all(axis=1)
            if int(valid.sum()) < 2:
                continue
            host_ns = imu_host_all_ns[valid]
            health = clock_health(host_ns)
            time_c3d = (host_ns - c3d_t0_host_ns) / 1e9
            rate = imu_sample_rate_hz(time_c3d)
            acc_norm = np.linalg.norm(acc[valid], axis=1)
            sensor_envs.append(
                {
                    "index": idx,
                    "label": label,
                    "time_c3d": time_c3d,
                    "envelope": highpass_envelope(acc_norm, rate),
                    "rate": rate,
                    "health": health,
                }
            )
        if not sensor_envs:
            raise ValueError("IMU 无有效传感器样本，无法同步")

    # 测力台候选信号：总垂直力是主信号；双侧 Fz 之和 / 单侧 Fz 作为备选（受试者
    # 可能以单脚为主跺脚，或个别导出的「总垂直力」列异常）。逐一与每个 IMU 传感器
    # 配对，按置信度 → 峰对数 → MAD 取最优；平手时优先历史默认「右腿 × 总垂直力」，
    # 保证结果可复现。
    force_rate = gaitway.sample_rate_hz
    force_candidates: list[tuple[str, np.ndarray]] = [
        ("GRFz vertical (N)", gaitway.columns["GRFz vertical (N)"]),
        ("FzL(N)+FzR(N)", gaitway.columns["FzL(N)"] + gaitway.columns["FzR(N)"]),
        ("FzL(N)", gaitway.columns["FzL(N)"]),
        ("FzR(N)", gaitway.columns["FzR(N)"]),
    ]
    force_envs = [
        (name, highpass_envelope(values, force_rate))
        for name, values in force_candidates
    ]

    conf_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    best: tuple[tuple[int, int, float, int], Any, dict[str, Any], str] | None = None
    attempts: list[dict[str, Any]] = []
    for senv in sensor_envs:
        for fname, fenv in force_envs:
            alignment, rejection = pair_stomps_diagnosed(
                senv["time_c3d"], senv["envelope"], gaitway.time_s, fenv,
                prominence=prominence,
            )
            if alignment is None:
                attempts.append(
                    {
                        "imu": senv["label"],
                        "force": fname,
                        "imu_peak_count": rejection.imu_peak_count,
                        "imu_burst_count": rejection.imu_burst_count,
                        "imu_reason": rejection.imu_reason,
                        "gaitway_peak_count": rejection.gaitway_peak_count,
                        "gaitway_burst_count": rejection.gaitway_burst_count,
                        "gaitway_reason": rejection.gaitway_reason,
                        "align_reason": rejection.align_reason,
                    }
                )
                continue
            canonical = (
                senv["label"] == "imu_right_leg" and fname == "GRFz vertical (N)"
            )
            key = (
                conf_rank[alignment.confidence],
                -len(alignment.pairs),
                alignment.mad_s,
                0 if canonical else 1,
            )
            if best is None or key < best[0]:
                best = (key, alignment, senv, fname)

    if best is None:
        raise StompSyncError(
            _format_multi_rejection(
                sensor_envs, [name for name, _ in force_candidates], attempts
            ),
            {
                "attempted_imu_sensors": [s["label"] for s in sensor_envs],
                "attempted_force_signals": [name for name, _ in force_candidates],
                "n_combinations": len(attempts),
                "combos": attempts,
            },
        )

    _, alignment, senv, force_signal = best
    sensor_index = senv["index"]
    sensor_label = senv["label"]
    imu_rate = senv["rate"]
    imu_health = senv["health"]

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
        "force_signal": force_signal,
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
