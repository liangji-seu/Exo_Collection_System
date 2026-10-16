"""Unit tests for the ModelSpec declarative schema."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from exo_collection.apps.model_runtime.spec import ModelSpec, load_model_spec


def _valid_spec(**overrides) -> dict:
    spec: dict = {
        "schema_version": "1.0.0",
        "model": {"backend": "demo", "demo": {"amplitude": 1.0}},
        "features": {
            "control_rate_hz": 100.0,
            "window_s": 1.28,
            "inputs": [{"modality": "imu", "channels": None, "rate_hz": 120.0}],
        },
        "output": {"kind": "torque_left_right", "single_output_maps_to": "right_torque", "torque_limit_nm": 5.0},
        "safety": {"enable_required": True, "zero_on_error": True, "watchdog_s": 0.25},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in spec and isinstance(spec[key], dict):
            spec[key].update(value)
        else:
            spec[key] = value
    return spec


def test_valid_demo_spec() -> None:
    spec = ModelSpec.model_validate(_valid_spec())
    assert spec.model.backend == "demo"
    assert spec.output.single_output_maps_to == "right_torque"
    assert spec.feature_window_samples == 128


def test_valid_torch_spec() -> None:
    spec = ModelSpec.model_validate(
        _valid_spec(
            model={
                "backend": "torch",
                "module_path": "models/tcn.py",
                "class_name": "HipMomentTCN",
                "weights_path": "models/tcn.pt",
                "input_layout": "bct",
            }
        )
    )
    assert spec.model.input_layout == "bct"


@pytest.mark.parametrize(
    ("rate_hz", "window_s", "expected_frames"),
    [(100.0, 0.94, 94), (20.0, 2.0, 40)],
)
def test_enabled_imu_tcn_accepts_dynamic_training_windows(rate_hz, window_s, expected_frames):
    spec = ModelSpec.model_validate(
        _valid_spec(
            model={"backend": "torchscript", "weights_path": "model.ts", "input_layout": "btc"},
            features={"control_rate_hz": rate_hz, "window_s": window_s},
            output={"unit": "Nm/kg", "single_output_maps_to": "right_torque"},
            imu_tcn={"enabled": True},
        )
    )
    assert spec.feature_window_samples == expected_frames


def test_generic_window_keeps_rounding_compatibility():
    spec = ModelSpec.model_validate(
        _valid_spec(features={"control_rate_hz": 100.0, "window_s": 0.935})
    )
    assert spec.feature_window_samples == 94


def test_enabled_imu_tcn_requires_integer_positive_window():
    with pytest.raises(ValidationError, match="integer positive"):
        ModelSpec.model_validate(
            _valid_spec(
                model={
                    "backend": "torchscript",
                    "weights_path": "model.ts",
                    "input_layout": "btc",
                },
                features={"control_rate_hz": 100.0, "window_s": 0.935},
                output={"unit": "Nm/kg", "single_output_maps_to": "right_torque"},
                imu_tcn={"enabled": True},
            )
        )


def test_torch_requires_module_and_class() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(model={"backend": "torch"}))


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(bogus="nope"))


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(model={"backend": "onnx"}))


def test_non_positive_control_rate_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(features={"control_rate_hz": 0.0}))


@pytest.mark.parametrize("invalid", [18.01, -1, float("nan"), float("inf")])
def test_torque_limit_outside_firmware_protocol_rejected(invalid) -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(output={"torque_limit_nm": invalid}))


@pytest.mark.parametrize("limit", [0, 9, 12.5, 18])
def test_torque_limit_accepts_actual_protocol_range(limit):
    from exo_collection.domain.motor_protocol import TORQUE_MAX_NM
    from exo_collection.apps.model_runtime.spec import TORQUE_MAX_NM as spec_max
    from exo_collection.apps.model_runtime.torque import TORQUE_MAX_NM as torque_max
    assert TORQUE_MAX_NM == spec_max == torque_max == 18
    spec = ModelSpec.model_validate(_valid_spec(output={"torque_limit_nm": limit}))
    assert spec.output.torque_limit_nm == limit


def test_wrong_schema_version_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(schema_version="9.9.9"))


def test_channel_names_deduplicated() -> None:
    spec = ModelSpec.model_validate(
        _valid_spec(
            features={
                "control_rate_hz": 100.0,
                "window_s": 1.0,
                "inputs": [{"modality": "imu", "channels": ["a", "a", "b"], "rate_hz": 100.0}],
            }
        )
    )
    assert spec.features.inputs[0].channels == ["a", "b"]


def test_load_model_spec_from_file(tmp_path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_valid_spec()), encoding="utf-8")
    spec = load_model_spec(path)
    assert isinstance(spec, ModelSpec)


def test_load_model_spec_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_model_spec("definitely/not/here.json")


def test_motor_distribution_is_explicit_and_generic_default_remains_per_side():
    assert ModelSpec.model_validate(_valid_spec()).assistance.motor_distribution == "per_side"
    model = ModelSpec.model_validate(_valid_spec(assistance={"motor_distribution": "right_to_both_same_sign"}))
    assert model.assistance.motor_distribution == "right_to_both_same_sign"
    with pytest.raises(ValidationError, match="left-only"):
        ModelSpec.model_validate(_valid_spec(output={"single_output_maps_to": "left_torque"},
            assistance={"motor_distribution": "right_to_both_same_sign"}))
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(assistance={"motor_distribution": "guessed_direction"}))
