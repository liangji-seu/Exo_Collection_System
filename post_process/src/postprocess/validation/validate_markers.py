"""Marker 校验：检查 HH19 关键 marker 是否存在、缺失率是否可接受。"""

from __future__ import annotations

# HH19 动态 trial 应存在的关键真实 marker（短名，去掉 subject 前缀）
HH19_DYNAMIC_EXPECTED = [
    "R.ASIS", "L.ASIS",
    "R.Thigh", "L.Thigh",
    "R.Knee", "L.Knee",
    "R.Shank", "L.Shank",
    "R.Ankle", "L.Ankle",
    "R.Heel", "L.Heel",
    "R.Toe", "L.Toe",
]


def validate_marker_coverage(marker_rows: list[dict], *,
                             expected: list[str] | None = None,
                             missing_ratio_threshold: float = 0.5) -> dict:
    """marker_rows 每项含 label / short_name / missing_ratio。"""
    expected = expected or HH19_DYNAMIC_EXPECTED
    present = {m["short_name"] for m in marker_rows}
    missing = [m for m in expected if m not in present]
    high_missing = [
        m["short_name"] for m in marker_rows
        if m["short_name"] in expected and m["missing_ratio"] > missing_ratio_threshold
    ]
    ok = not missing and not high_missing
    return {
        "ok": ok,
        "missing_expected": missing,
        "high_missing_ratio": high_missing,
        "reason": "" if ok else f"缺失 marker: {missing}; 缺失率过高: {high_missing}",
    }


__all__ = ["HH19_DYNAMIC_EXPECTED", "validate_marker_coverage"]
