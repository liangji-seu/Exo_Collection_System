"""Resolve the backpack load declared by a phase-4 walking condition."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


_LWALK_CODE = re.compile(
    r"^LWALK_(?:[A-Z0-9]+_)*(?P<load>[0-9]+(?:P[0-9]+)?KG)(?:_(?:NOEXO|EXO))?$",
    re.IGNORECASE,
)
_CONFIG_KEYS = {"com_m", "dimensions_xyz_m", "orientation_xyz_rad", "geometry_status"}
_DEFAULT_COM_M = [-0.12, 0.10, 0.0]
_DEFAULT_DIMENSIONS_M = [0.075, 0.10, 0.30]
_DEFAULT_ORIENTATION_RAD = [0.0, 0.0, 0.0]
_GEOMETRY_WARNING = (
    "Backpack geometry is approximate: the payload COM and dimensions assume "
    "that its weight is fixed in the bottom 10 cm band of a 50 x 30 x "
    "5-10 cm backpack."
)


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _load_from_code(token: str) -> float:
    return _finite_nonnegative(
        token[:-2].replace("P", ".").replace("p", "."), "condition load"
    )


def _vector3(value: Any, label: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a three-element vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{label} must contain finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain finite numbers") from exc
        if not math.isfinite(number) or (positive and number <= 0):
            qualifier = "positive finite numbers" if positive else "finite numbers"
            raise ValueError(f"{label} must contain {qualifier}")
        result.append(number)
    return result


def _geometry_override(
    session_dir: Path | None,
) -> tuple[dict[str, Any], str | None, str | None]:
    if session_dir is None:
        return {}, None, None
    path = Path(session_dir) / ".exo" / "backpack_config.json"
    if not path.is_file():
        return {}, None, None
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid backpack geometry config: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("backpack geometry config must be a JSON object")
    unknown = set(value) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown backpack geometry config keys: {sorted(unknown)}")

    override: dict[str, Any] = {}
    if "com_m" in value:
        override["com_m"] = _vector3(value["com_m"], "com_m")
    if "dimensions_xyz_m" in value:
        override["dimensions_xyz_m"] = _vector3(
            value["dimensions_xyz_m"], "dimensions_xyz_m", positive=True
        )
    if "orientation_xyz_rad" in value:
        override["orientation_xyz_rad"] = _vector3(
            value["orientation_xyz_rad"], "orientation_xyz_rad"
        )
    if "geometry_status" in value:
        if not isinstance(value["geometry_status"], str) or not value[
            "geometry_status"
        ].strip():
            raise ValueError("geometry_status must be a non-empty string")
        override["geometry_status"] = value["geometry_status"]
    return override, str(path), hashlib.sha256(raw).hexdigest()


def _result(
    condition_code: str,
    mass_kg: float,
    *,
    source: str,
    override: dict[str, Any] | None = None,
    config_path: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    override = override or {}
    enabled = mass_kg > 0
    geometry_status = override.get(
        "geometry_status", "assumed" if enabled else "not_applicable"
    )
    return {
        "enabled": enabled,
        "mass_kg": mass_kg,
        "source": source,
        "condition_code": condition_code,
        "geometry_status": geometry_status,
        "warnings": (
            [_GEOMETRY_WARNING]
            if enabled and geometry_status == "assumed"
            else []
        ),
        "config_path": config_path,
        "config_sha256": config_sha256,
        "spec": {
            "mode": "weld",
            "parent_body": "torso",
            "simulation_only": False,
            "empty_bag": {"mass_kg": 0.0},
            "payload": [
                {
                    "mass_kg": mass_kg,
                    "com_m": override.get("com_m", list(_DEFAULT_COM_M)),
                    "dimensions_xyz_m": override.get(
                        "dimensions_xyz_m", list(_DEFAULT_DIMENSIONS_M)
                    ),
                    "orientation_xyz_rad": override.get(
                        "orientation_xyz_rad", list(_DEFAULT_ORIENTATION_RAD)
                    ),
                }
            ],
        },
    }


def resolve_backpack_load(
    condition_code: str,
    parameters: dict,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve a nominal backpack mass without affecting legacy conditions."""
    code = condition_code.strip()
    if not code.upper().startswith("LWALK_"):
        return _result(condition_code, 0.0, source="NONE")

    match = _LWALK_CODE.fullmatch(code)
    if match is None:
        raise ValueError(f"malformed LWALK condition code: {condition_code}")

    # Validate the optional document before resolving load values so an invalid
    # override can never be partially processed.
    override, config_path, config_sha256 = _geometry_override(session_dir)
    code_mass = _load_from_code(match.group("load"))
    if "load_kg" in parameters:
        parameter_mass = _finite_nonnegative(parameters["load_kg"], "parameters.load_kg")
        if not math.isclose(parameter_mass, code_mass, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"backpack load mismatch: condition code={code_mass:g} kg, "
                f"parameters.load_kg={parameter_mass:g} kg"
            )
        mass_kg = parameter_mass
    else:
        mass_kg = code_mass

    return _result(
        condition_code,
        mass_kg,
        source="NOMINAL",
        override=override,
        config_path=config_path,
        config_sha256=config_sha256,
    )


__all__ = ["resolve_backpack_load"]
