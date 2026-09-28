"""Bridge resolved backpack settings into the inverse-dynamics model path.

This adapter intentionally depends only on the XML-based backpack generator;
importing it does not require an OpenSim Python installation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .backpack import BackpackSpec, aggregate_components, build_backpack_model


BACKPACK_MODEL_NAME = "hh19_static_calibrated_backpack.osim"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mass(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("resolved backpack mass_kg must be a finite non-negative number, not bool")
    try:
        mass = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved backpack mass_kg must be a finite non-negative number") from exc
    if not math.isfinite(mass) or mass < 0:
        raise ValueError("resolved backpack mass_kg must be a finite non-negative number")
    return mass


def _warnings(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("resolved backpack warnings must be a list")
    return copy.deepcopy(value)


def _receipt(
    human_model: Path,
    actual_model: Path,
    resolved: Mapping[str, Any] | None,
    *,
    enabled: bool,
    mass_kg: float,
    warnings: list[Any],
    generator_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    human_hash = _sha256(human_model)
    actual_hash = human_hash if actual_model == human_model else _sha256(actual_model)
    receipt: dict[str, Any] = {
        "enabled": enabled,
        "mass_kg": mass_kg,
        "source": None if resolved is None else copy.deepcopy(resolved.get("source")),
        "condition_code": None if resolved is None else copy.deepcopy(resolved.get("condition_code")),
        "geometry_status": None if resolved is None else copy.deepcopy(resolved.get("geometry_status")),
        "warnings": warnings,
        "human_model_path": str(human_model),
        "human_model_sha256": human_hash,
        "actual_model_path": str(actual_model),
        "actual_model_sha256": actual_hash,
        "generator_receipt": None if generator_receipt is None else copy.deepcopy(dict(generator_receipt)),
        "totals": {
            "resolved_mass_kg": mass_kg,
            "generated_backpack_mass_kg": (
                0.0 if generator_receipt is None else float(generator_receipt["total_backpack_mass_kg"])
            ),
        },
    }
    # Fail here rather than much later when the session metadata is written.
    try:
        json.dumps(receipt, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved backpack metadata must be JSON-serializable") from exc
    return receipt


def prepare_id_model(human_model: Path, resolved: dict[str, Any] | None) -> tuple[Path, dict[str, Any]]:
    """Return the model to use for ID and a serializable backpack receipt.

    ``None`` or an empty mapping is the legacy no-backpack path.  An explicitly
    disabled zero-mass resolution is also a no-op.  Positive mass is generated
    as a welded body beside the human model, leaving the human model untouched.
    """
    human_model = Path(human_model)
    if not human_model.is_file():
        raise FileNotFoundError(human_model)

    if resolved is None or resolved == {}:
        return human_model, _receipt(
            human_model,
            human_model,
            None,
            enabled=False,
            mass_kg=0.0,
            warnings=[],
            generator_receipt=None,
        )
    if not isinstance(resolved, dict):
        raise ValueError("resolved backpack settings must be a dict or None")

    enabled = resolved.get("enabled")
    if type(enabled) is not bool:
        raise ValueError("resolved backpack enabled must be bool")
    if "mass_kg" not in resolved:
        raise ValueError("resolved backpack settings must include mass_kg")
    mass_kg = _mass(resolved["mass_kg"])
    warnings = _warnings(resolved.get("warnings"))
    spec_payload = resolved.get("spec")

    if not enabled:
        if mass_kg != 0.0:
            raise ValueError("disabled backpack cannot have positive mass_kg")
        if spec_payload is not None:
            if not isinstance(spec_payload, Mapping):
                raise ValueError("resolved backpack spec must be a mapping")
            disabled_spec = BackpackSpec.from_mapping(spec_payload)
            disabled_total = aggregate_components((disabled_spec.empty_bag, *disabled_spec.payload)).mass_kg
            if disabled_total != 0.0:
                raise ValueError("disabled backpack cannot have a positive-mass spec")
        return human_model, _receipt(
            human_model,
            human_model,
            resolved,
            enabled=False,
            mass_kg=0.0,
            warnings=warnings,
            generator_receipt=None,
        )

    if mass_kg == 0.0:
        raise ValueError("enabled backpack must have positive mass_kg")
    if not isinstance(spec_payload, Mapping):
        raise ValueError("enabled backpack must include a spec mapping")

    spec_mapping = copy.deepcopy(dict(spec_payload))
    requested_mode = str(spec_mapping.get("mode", "weld")).lower()
    if requested_mode != "weld":
        raise ValueError("ID backpack model must use weld mode")
    spec_mapping["mode"] = "weld"
    spec = BackpackSpec.from_mapping(spec_mapping)
    spec_total = aggregate_components((spec.empty_bag, *spec.payload)).mass_kg
    if not math.isclose(spec_total, mass_kg, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"resolved backpack mass_kg ({mass_kg:.17g}) does not match spec total ({spec_total:.17g})"
        )

    output_model = human_model.parent / BACKPACK_MODEL_NAME
    generator_receipt = build_backpack_model(human_model, output_model, spec)
    generated_total = float(generator_receipt["total_backpack_mass_kg"])
    if not math.isclose(generated_total, mass_kg, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("backpack generator returned an unexpected total mass")
    return output_model, _receipt(
        human_model,
        output_model,
        resolved,
        enabled=True,
        mass_kg=mass_kg,
        warnings=warnings,
        generator_receipt=generator_receipt,
    )


__all__ = ["BACKPACK_MODEL_NAME", "prepare_id_model"]
