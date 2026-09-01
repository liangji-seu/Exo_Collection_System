"""支撑相分类：由左右脚接触布尔序列推导每帧的支撑相。

相位定义（严格，来自 prompt2 §8）：

- ``RIGHT_SINGLE_SUPPORT``：right_contact=True 且 left_contact=False
- ``LEFT_SINGLE_SUPPORT`` ：left_contact=True 且 right_contact=False
- ``DOUBLE_SUPPORT``      ：两者皆 True
- ``NO_CONTACT``          ：两者皆 False
- ``UNKNOWN``             ：数据不足（marker 缺失 / 条件冲突）

其中只有 ``RIGHT_SINGLE_SUPPORT`` / ``LEFT_SINGLE_SUPPORT`` 对 ID 可信
（单块测力台下双支撑无法分解左右力，第一版一律 mask 掉）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

PHASE_RIGHT_SS = "RIGHT_SINGLE_SUPPORT"
PHASE_LEFT_SS = "LEFT_SINGLE_SUPPORT"
PHASE_DOUBLE = "DOUBLE_SUPPORT"
PHASE_NO_CONTACT = "NO_CONTACT"
PHASE_UNKNOWN = "UNKNOWN"

# ID 可信的相位（单支撑）
VALID_FOR_ID = {PHASE_RIGHT_SS, PHASE_LEFT_SS}


def classify_phase(right_contact: Sequence[bool], left_contact: Sequence[bool]) -> np.ndarray:
    """返回 (n_frames,) 的相位字符串数组。

    任一帧的左右接触信号含 NaN（数据缺失）时标为 ``UNKNOWN``。
    """
    right = np.asarray(right_contact, dtype=object)
    left = np.asarray(left_contact, dtype=object)
    if right.shape != left.shape:
        raise ValueError(f"right {right.shape} != left {left.shape}")

    phase = np.empty(right.shape[0], dtype=object)
    for i in range(right.shape[0]):
        r, l = right[i], left[i]
        if r is None or l is None or (isinstance(r, float) and np.isnan(r)) or (
            isinstance(l, float) and np.isnan(l)
        ):
            phase[i] = PHASE_UNKNOWN
            continue
        r, l = bool(r), bool(l)
        if r and not l:
            phase[i] = PHASE_RIGHT_SS
        elif l and not r:
            phase[i] = PHASE_LEFT_SS
        elif r and l:
            phase[i] = PHASE_DOUBLE
        else:
            phase[i] = PHASE_NO_CONTACT
    return phase


def valid_for_id_mask(phase: Sequence[str]) -> np.ndarray:
    """返回每帧是否可用于 ID（True 仅当单支撑）。"""
    phase = np.asarray(phase, dtype=object)
    return np.fromiter((p in VALID_FOR_ID for p in phase), dtype=bool, count=len(phase))


__all__ = [
    "PHASE_RIGHT_SS",
    "PHASE_LEFT_SS",
    "PHASE_DOUBLE",
    "PHASE_NO_CONTACT",
    "PHASE_UNKNOWN",
    "VALID_FOR_ID",
    "classify_phase",
    "valid_for_id_mask",
]
