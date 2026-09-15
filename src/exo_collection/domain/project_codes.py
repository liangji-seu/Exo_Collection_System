"""Stable project partitions shared by Collector, storage, and Data Studio."""

from __future__ import annotations

from typing import Final


PROJECT_CODE_TEST: Final = "T"
PROJECT_CODE_FORMAL_BASELINE: Final = "F_BASE"
PROJECT_CODE_FORMAL_STEADY: Final = "F_STEADY"
PROJECT_CODE_FORMAL_TRANSIENT: Final = "F_TRANSIENT"
PROJECT_CODE_FORMAL_SPECIAL: Final = "F_SPECIAL"

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
        PROJECT_CODE_FORMAL_SPECIAL,
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
    {
        "project_code": PROJECT_CODE_FORMAL_SPECIAL,
        "project_name": "正式-特殊",
    },
)

PROJECT_CONDITION_LEVELS: Final = {
    PROJECT_CODE_TEST: frozenset({"TEST"}),
    PROJECT_CODE_FORMAL_BASELINE: frozenset({"BASELINE"}),
    PROJECT_CODE_FORMAL_STEADY: frozenset({"STEADY_STATE"}),
    PROJECT_CODE_FORMAL_TRANSIENT: frozenset({"TRANSIENT"}),
    PROJECT_CODE_FORMAL_SPECIAL: frozenset({"SPECIAL"}),
}


def project_accepts_condition_level(
    project_code: str,
    condition_level: int | str | None,
) -> bool:
    """Return whether a condition belongs in the selected project.

    The test project exposes only ``TEST``-level conditions (free test and
    static calibration), while the legacy formal project (``F``) intentionally
    exposes the complete protocol.
    """

    expected = PROJECT_CONDITION_LEVELS.get(project_code.strip().upper())
    if expected is None:
        return True
    return str(condition_level).strip().upper() in expected


def project_for_condition_level(
    condition_level: int | str | None,
) -> dict[str, str] | None:
    """Return the ``{project_code, project_name}`` project owning a condition.

    Each ``condition_level`` maps to exactly one project (``PROJECT_CONDITION_LEVELS``
    is a bijection), so the project is fully determined by the selected condition.
    Returns ``None`` for unknown levels.
    """

    level = str(condition_level).strip().upper()
    for project in COLLECTOR_PROJECTS:
        levels = PROJECT_CONDITION_LEVELS.get(project["project_code"], frozenset())
        if level in levels:
            return dict(project)
    return None


__all__ = [
    "COLLECTOR_PROJECTS",
    "LEGACY_PROJECT_CODE_FORMAL",
    "PROJECT_CODE_FORMAL_BASELINE",
    "PROJECT_CODE_FORMAL_STEADY",
    "PROJECT_CODE_FORMAL_TRANSIENT",
    "PROJECT_CODE_FORMAL_SPECIAL",
    "PROJECT_CODE_TEST",
    "PROJECT_CONDITION_LEVELS",
    "SUPPORTED_PROJECT_CODES",
    "project_accepts_condition_level",
    "project_for_condition_level",
]
