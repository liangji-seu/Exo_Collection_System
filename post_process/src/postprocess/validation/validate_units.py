"""单位校验：确认 config 里的单位是可识别且量纲一致的。"""

from __future__ import annotations

from ..preprocessing.units import _DIMENSION_OF, _FACTORS


def validate_unit(unit: str) -> dict:
    if unit is None or unit == "":
        return {"ok": False, "reason": "单位为空"}
    if unit in _DIMENSION_OF:
        return {"ok": True, "reason": "", "dimension": _DIMENSION_OF[unit]}
    return {"ok": False, "reason": f"未知单位 '{unit}'，可用：{sorted(_FACTORS)}"}


def validate_unit_pair(src: str, dst: str) -> dict:
    s, d = validate_unit(src), validate_unit(dst)
    if not s["ok"]:
        return s
    if not d["ok"]:
        return d
    if s["dimension"] != d["dimension"]:
        return {"ok": False, "reason": f"量纲不一致：{src}({s['dimension']}) vs {dst}({d['dimension']})"}
    return {"ok": True, "reason": "", "dimension": s["dimension"]}


__all__ = ["validate_unit", "validate_unit_pair"]
