"""支撑相 mask / 分段 / CSV 写出。

由 :mod:`detect_contact` 与 :mod:`detect_single_support` 的结果生成：

1. ``support_phase.csv`` —— 每帧一行：time, right_contact, left_contact, phase, valid_for_id
2. ``segments.json``    —— 连续单支撑段的起止时间（供分段跑 ID）
3. 相位统计（各相占比、段数）

双支撑段（``valid_for_id=False``）第一版严格 mask 掉，不做左右力分解。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .detect_single_support import (
    PHASE_RIGHT_SS,
    PHASE_LEFT_SS,
    classify_phase,
    valid_for_id_mask,
)


@dataclass
class SupportMask:
    time_s: np.ndarray
    right_contact: np.ndarray
    left_contact: np.ndarray
    any_contact: np.ndarray
    phase: np.ndarray          # (n_frames,) object[str]
    valid_for_id: np.ndarray   # (n_frames,) bool
    segments: list[dict] = field(default_factory=list)

    def statistics(self) -> dict[str, Any]:
        n = len(self.phase)
        counts = {p: int((self.phase == p).sum()) for p in np.unique(self.phase)}
        total_valid = int(self.valid_for_id.sum())
        seg = self.segments
        return {
            "n_frames": n,
            "right_single_support_pct": 100.0 * counts.get(PHASE_RIGHT_SS, 0) / n,
            "left_single_support_pct": 100.0 * counts.get(PHASE_LEFT_SS, 0) / n,
            "double_support_pct": 100.0 * counts.get("DOUBLE_SUPPORT", 0) / n,
            "no_contact_pct": 100.0 * counts.get("NO_CONTACT", 0) / n,
            "unknown_pct": 100.0 * counts.get("UNKNOWN", 0) / n,
            "n_frames_valid_for_id": total_valid,
            "valid_for_id_pct": 100.0 * total_valid / n if n else 0.0,
            "n_segments": len(seg),
            "n_right_segments": sum(1 for s in seg if s["foot"] == "right"),
            "n_left_segments": sum(1 for s in seg if s["foot"] == "left"),
            "segments": seg,
        }


def extract_segments(
    time_s: np.ndarray,
    phase: np.ndarray,
    *,
    trim_boundary_ms: float = 20.0,
    min_segment_frames: int = 3,
) -> list[dict]:
    """把连续单支撑段拆成 ``{segment_id, foot, start_time, end_time, n_frames}``。

    每段首尾各裁 ``trim_boundary_ms``（默认 20 ms）以避开相位边界误差；
    裁剪后不足 ``min_segment_frames`` 帧的段丢弃。
    """
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.01
    trim_frames = int(round(trim_boundary_ms / 1000.0 / dt)) if dt > 0 else 0

    segments: list[dict] = []
    i = 0
    while i < len(phase):
        p = phase[i]
        if p not in (PHASE_RIGHT_SS, PHASE_LEFT_SS):
            i += 1
            continue
        j = i
        while j < len(phase) and phase[j] == p:
            j += 1
        foot = "right" if p == PHASE_RIGHT_SS else "left"
        a, b = i + trim_frames, j - trim_frames
        if b - a >= min_segment_frames:
            segments.append({
                "segment_id": f"{foot}_ss_{len(segments) + 1:03d}",
                "foot": foot,
                "start_time": float(time_s[a]),
                "end_time": float(time_s[b - 1]),
                "n_frames": b - a,
                "start_frame": a,
                "end_frame": b - 1,
            })
        i = j
    return segments


def build_support_mask(
    time_s: np.ndarray,
    right_contact: np.ndarray,
    left_contact: np.ndarray,
    any_contact: np.ndarray,
    *,
    trim_boundary_ms: float = 20.0,
) -> SupportMask:
    phase = classify_phase(right_contact, left_contact)
    valid = valid_for_id_mask(phase)
    segments = extract_segments(time_s, phase, trim_boundary_ms=trim_boundary_ms)
    return SupportMask(
        time_s=time_s,
        right_contact=np.asarray(right_contact),
        left_contact=np.asarray(left_contact),
        any_contact=np.asarray(any_contact),
        phase=phase,
        valid_for_id=valid,
        segments=segments,
    )


def write_support_csv(mask: SupportMask, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("time,right_contact,left_contact,phase,valid_for_id\n")
        for i in range(len(mask.time_s)):
            f.write(
                f"{mask.time_s[i]:.6f},"
                f"{int(mask.right_contact[i])},"
                f"{int(mask.left_contact[i])},"
                f"{mask.phase[i]},"
                f"{int(mask.valid_for_id[i])}\n"
            )


def write_segments_json(mask: SupportMask, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(mask.segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )


__all__ = [
    "SupportMask",
    "build_support_mask",
    "extract_segments",
    "write_support_csv",
    "write_segments_json",
]
