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


def test_torque_limit_above_hardware_max_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_valid_spec(output={"torque_limit_nm": 12.0}))


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
