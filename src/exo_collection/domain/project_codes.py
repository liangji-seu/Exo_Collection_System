"""Stable project partitions shared by Collector, storage, and Data Studio."""

from __future__ import annotations

from typing import Final


PROJECT_CODE_TEST: Final = "T"
PROJECT_CODE_FORMAL_BASELINE: Final = "F_BASE"
PROJECT_CODE_FORMAL_STEADY: Final = "F_STEADY"
PROJECT_CODE_FORMAL_TRANSIENT: Final = "F_TRANSIENT"

# ``F`` remains readable for existing v1.0.0 datasets, but new Collector
# sessions use one of the three explicit formal partitions.
LEGACY_PROJECT_CODE_FORMAL: Final = "F"

SUPPORTED_PROJECT_CODES: Final = frozenset(
    {
        PROJECT_CODE_TEST,
        LEGACY_PROJECT_CODE_FORMAL,
        PROJECT_CODE_FORMAL_BASELINE,
        PROJECT_CODE_FORMAL_STEADY,
        PROJECT_CODE_FORMAL_TRANSIENT,
    }
)

COLLECTOR_PROJECTS: Final = (
    {"project_code": PROJECT_CODE_TEST, "project_name": "测试"},
    {
        "project_code": PROJECT_CODE_FORMAL_BASELINE,
        "project_name": "正式-基础",
    },
    {
        "project_code": PROJECT_CODE_FORMAL_STEADY,
        "project_name": "正式-稳态",
    },
    {
        "project_code": PROJECT_CODE_FORMAL_TRANSIENT,
        "project_name": "正式-非稳态",
    },
)

PROJECT_CONDITION_LEVELS: Final = {
    PROJECT_CODE_FORMAL_BASELINE: frozenset({"BASELINE"}),
    PROJECT_CODE_FORMAL_STEADY: frozenset({"STEADY_STATE"}),
    PROJECT_CODE_FORMAL_TRANSIENT: frozenset({"TRANSIENT"}),
}


def project_accepts_condition_level(
    project_code: str,
    condition_level: int | str | None,
) -> bool:
    """Return whether a condition belongs in the selected project.

    Test and legacy formal projects intentionally expose the complete protocol.
    """

    expected = PROJECT_CONDITION_LEVELS.get(project_code.strip().upper())
    if expected is None:
        return True
    return str(condition_level).strip().upper() in expected


__all__ = [
    "COLLECTOR_PROJECTS",
    "LEGACY_PROJECT_CODE_FORMAL",
    "PROJECT_CODE_FORMAL_BASELINE",
    "PROJECT_CODE_FORMAL_STEADY",
    "PROJECT_CODE_FORMAL_TRANSIENT",
    "PROJECT_CODE_TEST",
    "PROJECT_CONDITION_LEVELS",
    "SUPPORTED_PROJECT_CODES",
    "project_accepts_condition_level",
]
