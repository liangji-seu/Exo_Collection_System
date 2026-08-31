"""Force / analog 通道识别与分类。

关键目标：判断 C3D 里保存的测力台数据是哪种模式：

- ``TOTAL_ONLY``   只有整条 Gaitway 的合力 Fx/Fy/Fz + COPx/COPy（+ 可能 Tz）
- ``LEFT_RIGHT``   已经分解为左右脚 FxL/FyL/... 与 FxR/FyR/...
- ``BOTH``         两种都有
- ``UNKNOWN``      无法识别

**严禁**提前假设是某一种：这决定下游能否做双侧 inverse dynamics。
如果只有 TOTAL，双支撑阶段无法唯一确定左右侧 external load，ID 必须 BLOCKING。
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .reader import C3dData

# 已知力/力矩/COP 分量的标准 token（小写，无左右标记）
_FORCE_COMPONENTS = {
    "fx": "Fx", "fy": "Fy", "fz": "Fz",
    "mx": "Mx", "my": "My", "mz": "Mz",
    "copx": "COPx", "copy": "COPy", "copz": "COPz",
    "tx": "Tx", "ty": "Ty", "tz": "Tz",
}

_FORCE_ONLY = {"Fx", "Fy", "Fz"}
_COP_ONLY = {"COPx", "COPy", "COPz"}
_MOMENT_ONLY = {"Mx", "My", "Mz", "Tx", "Ty", "Tz"}


@dataclass(frozen=True)
class ChannelClass:
    label: str
    kind: str          # "Fx"/"Fy"/.../"COPx"/.../"Tz"/... 或 "other"
    side: str          # "left" / "right" / "total" / "unknown"
    plate_index: int   # 尾随数字（1-based 力台编号），无则 0


def classify_channel(label: str) -> ChannelClass:
    """把单个 analog 通道名归类为 (kind, side, plate_index)。

    例：``Fx1`` -> (Fx, total, 1)；``FzR`` -> (Fz, right, 0)；``COPx2`` -> (COPx, total, 2)。
    """
    s = str(label).strip().lower()
    # 尾随数字 = 力台编号
    m = re.search(r"(\d+)\s*$", s)
    plate_index = int(m.group(1)) if m else 0
    base = s[: m.start()] if m else s

    # 左右标记：单独的前缀/后缀 l / r（不落在 component token 内）
    side = "total"
    if re.match(r"^[lr][_\- ]", base) or re.search(r"[_\- ][lr]$", base):
        side = "left" if base.lstrip("_ -")[0] == "l" else "right"
        base = re.sub(r"^[lr][_\- ]", "", base)
        base = re.sub(r"[_\- ][lr]$", "", base)

    # 去掉分隔符后精确匹配 component token
    token = re.sub(r"[\W_]", "", base)
    kind = _FORCE_COMPONENTS.get(token, "other")
    return ChannelClass(label, kind, side, plate_index)


def classify_channels(labels) -> list[ChannelClass]:
    return [classify_channel(lbl) for lbl in labels]


def detect_grf_mode(classes: list[ChannelClass]) -> str:
    """返回 ``TOTAL_ONLY`` / ``LEFT_RIGHT`` / ``BOTH`` / ``UNKNOWN``。"""
    kinds = {c.kind for c in classes}
    has_force = bool(kinds & _FORCE_ONLY)
    if not has_force:
        return "UNKNOWN"
    has_left = any(c.side == "left" for c in classes)
    has_right = any(c.side == "right" for c in classes)
    has_total = any(c.side == "total" for c in classes)
    if has_left and has_right and has_total:
        return "BOTH"
    if has_left and has_right:
        return "LEFT_RIGHT"
    if has_total and not has_left and not has_right:
        return "TOTAL_ONLY"
    return "UNKNOWN"


def force_channel_summary(data: C3dData) -> dict:
    """汇总 analog 通道：分类表 + GRF_MODE + 各力台通道映射。"""
    classes = classify_channels(data.analog_labels)
    grf_mode = detect_grf_mode(classes)
    channels = [
        {
            "label": c.label,
            "kind": c.kind,
            "side": c.side,
            "plate_index": c.plate_index,
        }
        for c in classes
    ]
    # 每个力台的通道（按 plate_index 分组）
    platforms = {}
    for c in classes:
        if c.plate_index:
            platforms.setdefault(c.plate_index, []).append(c.label)
    return {
        "grf_mode": grf_mode,
        "channels": channels,
        "force_components_found": sorted(k for k in {c.kind for c in classes} if k != "other"),
        "other_channels": [c.label for c in classes if c.kind == "other"],
        "platform_channels": platforms,
        "has_free_moment": any(c.kind == "Tz" for c in classes),
        "has_six_axis_moment": bool({c.kind for c in classes} & {"Mx", "My", "Mz"}),
    }


__all__ = ["ChannelClass", "classify_channel", "classify_channels",
           "detect_grf_mode", "force_channel_summary"]
