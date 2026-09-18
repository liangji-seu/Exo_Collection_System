"""Declarative model specification for the online-testing runtime.

``ModelSpec`` is the single source of truth that turns a "model" into something
the runtime can load and run.  It is deliberately **not** tied to any one model
architecture: the ``model.backend`` selects a loader (torch / callable / demo)
and the ``features`` / ``output`` / ``safety`` sections describe how real-time
subscriptions become model inputs and how model outputs become torque commands.

The schema is strict (``extra="forbid"``) so a bad spec fails loudly at load
time rather than halfway through an online session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"

# Hardware ceiling from the AK80-V3 motor controller (old_system/motor_controller.py).
TORQUE_MAX_NM = 9.0


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class ModelBackendSpec(_Strict):
    """How to instantiate the model object and load its weights."""

    backend: Literal["torch", "callable", "demo"]
    module_path: str | None = None
    class_name: str | None = None
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    weights_path: str | None = None
    device: str = "cpu"
    # Feature tensor layout expected by a torch forward(): "bct" = [B, C, T]
    # (channels-first, standard for 1-D conv / TCN over time), "btc" = [B, T, C].
    input_layout: Literal["bct", "btc"] = "bct"
    # Optional free-form hints for the demo backend (amplitude, frequency, ...).
    demo: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_class_for_loaded_backends(self) -> "ModelBackendSpec":
        if self.backend in ("torch", "callable"):
            if not self.module_path or not self.class_name:
                raise ValueError(
                    f"backend={self.backend!r} requires module_path and class_name"
                )
        return self


class FeatureInput(_Strict):
    """One modality's contribution to the model feature tensor."""

    modality: str
    # ``None`` means "use every channel the descriptor exposes".
    channels: list[str] | None = None
    # Optional override of the descriptor's nominal rate; defaults to descriptor.
    rate_hz: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _normalize_channels(self) -> "FeatureInput":
        if self.channels is not None:
            deduped: list[str] = []
            for channel in self.channels:
                channel = str(channel).strip()
                if not channel:
                    raise ValueError("feature channel names must be non-empty")
                if channel not in deduped:
                    deduped.append(channel)
            self.channels = deduped
        return self


class FeatureNormalization(_Strict):
    """Optional per-channel affine normalization applied after channel ordering."""

    mean: list[float] = Field(default_factory=list)
    std: list[float] = Field(default_factory=list)


class FeatureSpec(_Strict):
    control_rate_hz: float = Field(gt=0)
    window_s: float = Field(gt=0)
    inputs: list[FeatureInput] = Field(min_length=1)
    normalization: FeatureNormalization = Field(default_factory=FeatureNormalization)


class OutputSpec(_Strict):
    """How model outputs map onto bilateral motor torque."""

    kind: Literal["torque_left_right"] = "torque_left_right"
    # When the model emits a single scalar (right hip-flexion moment), map it to
    # one motor and drive the other to zero.
    single_output_maps_to: Literal["left_torque", "right_torque"] | None = None
    torque_limit_nm: float = Field(default=5.0, gt=0, le=TORQUE_MAX_NM)


class SafetySpec(_Strict):
    enable_required: bool = True
    zero_on_error: bool = True
    watchdog_s: float = Field(default=0.25, ge=0.01)


class ModelSpec(_Strict):
    """Top-level, strictly validated model specification."""

    schema_version: str = SCHEMA_VERSION
    model: ModelBackendSpec
    features: FeatureSpec
    output: OutputSpec = Field(default_factory=OutputSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)

    @model_validator(mode="after")
    def _check_schema_version(self) -> "ModelSpec":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        return self

    @property
    def feature_window_samples(self) -> int:
        """Control-grid length of one feature window (rounded up)."""
        return max(1, int(round(self.features.window_s * self.features.control_rate_hz)))


def load_model_spec(path: str | Path) -> ModelSpec:
    """Read and strictly validate a model spec JSON file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model spec JSON at {source}: {exc}") from exc
    return ModelSpec.model_validate(payload)


__all__ = [
    "SCHEMA_VERSION",
    "TORQUE_MAX_NM",
    "FeatureInput",
    "FeatureNormalization",
    "FeatureSpec",
    "ModelBackendSpec",
    "ModelSpec",
    "OutputSpec",
    "SafetySpec",
    "load_model_spec",
]
