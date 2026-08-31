"""同步校验：点/模拟采样率对齐检查（不重采样）。"""

from __future__ import annotations

from ..preprocessing.synchronization import check_rate_alignment


def validate_sync(point_rate_hz: float, analog_rate_hz: float, n_frames: int) -> dict:
    info = check_rate_alignment(point_rate_hz, analog_rate_hz, n_frames)
    if info.ratio is None:
        return {"ok": False, "reason": "采样率缺失，无法判断同步", "sync": info}
    if info.is_aligned():
        return {"ok": True, "reason": "point/analog 同帧对齐（ratio=1）", "sync": info}
    if info.integer_ratio:
        return {"ok": True, "reason": f"整数倍关系 ratio={info.ratio:.3f}（需按配置重采样）",
                "sync": info}
    return {"ok": False, "reason": f"非整数倍 ratio={info.ratio:.4f}，需外部同步方式",
            "sync": info}


__all__ = ["validate_sync"]
