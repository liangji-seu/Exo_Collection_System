"""Configured IMU roles remain authoritative across discovery and model defaults."""
from pathlib import Path

import pytest

from exo_collection.apps.model_runtime.model_bindings import bind_model_inputs
from exo_collection.apps.model_runtime.spec import load_model_spec


@pytest.fixture
def spec():
    return load_model_spec(Path(__file__).resolve().parents[2] /
        "runtime_data/models/imu_right_pelvis_torque_v1_20261010/model_spec.json")


def test_hardware_slots_override_model_and_ignore_discovery_order(spec):
    before = spec.model_dump()
    bound = bind_model_inputs(spec, profile_key="hardware",
        imu_parameters={"sensor_ids": ["", "R", "P"]},
        descriptor={"metadata": {"device_ids": ["P", "R"]}})
    assert (bound.imu_tcn.right_device_id, bound.imu_tcn.pelvis_device_id) == ("R", "P")
    assert spec.model_dump() == before
    assert bound is not spec


@pytest.mark.parametrize("slots", [["", "R", "R"], ["L", "", "P"],
    ["L", "R", ""], ["R", "R", "P"], ["R", "P"]])
def test_required_empty_or_duplicate_slots_fail_closed(spec, slots):
    with pytest.raises(ValueError):
        bind_model_inputs(spec, profile_key="hardware", imu_parameters={"sensor_ids": slots})


def test_simulated_config_uses_configured_roles(spec):
    result = bind_model_inputs(spec, profile_key="simulated",
        imu_parameters={"device_ids": ["LEFT", "RIGHT", "PELVIS"]},
        descriptor={"metadata": {"device_ids": ["PELVIS", "LEFT", "RIGHT"]}})
    assert (result.imu_tcn.right_device_id, result.imu_tcn.pelvis_device_id) == ("RIGHT", "PELVIS")


def test_missing_slots_only_allow_explicit_model_ids(spec):
    explicit = spec.model_copy(deep=True)
    explicit.imu_tcn.right_device_id, explicit.imu_tcn.pelvis_device_id = "R", "P"
    result = bind_model_inputs(explicit, profile_key="hardware", imu_parameters={"sensor_ids": []})
    assert result.imu_tcn.right_device_id == "R"
    explicit.imu_tcn.pelvis_device_id = None
    with pytest.raises(ValueError):
        bind_model_inputs(explicit, profile_key="hardware", imu_parameters={})


def test_descriptor_is_only_validation_and_generic_model_is_unchanged(spec):
    with pytest.raises(ValueError, match="不包含"):
        bind_model_inputs(spec, profile_key="hardware",
            imu_parameters={"sensor_ids": ["", "R", "P"]},
            descriptor={"metadata": {"device_ids": ["OTHER", "R"]}})
    generic = spec.model_copy(deep=True)
    generic.imu_tcn.enabled = False
    result = bind_model_inputs(generic, profile_key="hardware",
        imu_parameters={"sensor_ids": ["bad"]})
    assert result.model_dump() == generic.model_dump()
