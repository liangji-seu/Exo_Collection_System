"""单位换算。核心是长度 mm↔m、力矩 Nmm↔Nm，用于 C3D(mm) → OpenSim(m) 转换。

原则：单位转换**必须显式**，不在坐标变换里偷偷进行。
"""

from __future__ import annotations

# 每个量纲 -> 基准单位(该量纲的 SI) 的换算因子
_FACTORS: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "mm": 1e-3, "cm": 1e-2},
    "force": {"N": 1.0, "kN": 1e3},
    "moment": {"Nm": 1.0, "Nmm": 1e-3},
    "angle": {"rad": 1.0, "deg": 3.141592653589793 / 180.0},
}

_DIMENSION_OF: dict[str, str] = {}
for _dim, _table in _FACTORS.items():
    for _unit in _table:
        _DIMENSION_OF[_unit] = _dim


def convert(value: float, src_unit: str, dst_unit: str) -> float:
    """把 ``value`` 从 ``src_unit`` 换算到 ``dst_unit``。"""
    src, dst = src_unit, dst_unit
    if src == dst:
        return value
    if src not in _DIMENSION_OF or dst not in _DIMENSION_OF:
        raise ValueError(f"未知单位：{src} / {dst}")
    if _DIMENSION_OF[src] != _DIMENSION_OF[dst]:
        raise ValueError(f"单位量纲不一致：{src}({_DIMENSION_OF[src]}) vs {dst}({_DIMENSION_OF[dst]})")
    to_si = _FACTORS[_DIMENSION_OF[src]][src]
    from_si = _FACTORS[_DIMENSION_OF[dst]][dst]
    return value * to_si / from_si


def mm_to_m(value: float) -> float:
    return convert(value, "mm", "m")


def m_to_mm(value: float) -> float:
    return convert(value, "m", "mm")


def nmm_to_nm(value: float) -> float:
    return convert(value, "Nmm", "Nm")


def is_length_unit(unit: str) -> bool:
    return _DIMENSION_OF.get(unit) == "length"


__all__ = ["convert", "mm_to_m", "m_to_mm", "nmm_to_nm", "is_length_unit"]
