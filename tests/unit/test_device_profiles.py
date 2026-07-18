from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from exo_collection.configuration.device_profiles import (
    HardwareDeviceProfileDocument,
    ImuSimulationParameters,
    SimulatedDeviceProfileDocument,
    default_simulated_device_profile_path,
    load_device_profile,
    load_simulated_device_profile,
)
from exo_collection.configuration.adapter_registry import build_adapters
from exo_collection.adapters.encoder.teensy_serial import TeensySerialEncoderAdapter
from exo_collection.adapters.imu.xsens_awinda import XsensAwindaImuAdapter
from exo_collection.adapters.sync_pulse.simulated import SimulatedSyncPulseAdapter
from exo_collection.adapters.ultrasound.elonxi import ElonxiUltrasoundAdapter
from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.orchestration.simulated import _make_adapters


def test_default_simulated_device_profile_is_typed_and_complete() -> None:
    path = default_simulated_device_profile_path()
    profile = load_simulated_device_profile()
    devices = profile.by_modality()

    assert path.name == "simulated.json"
    assert list(devices) == ["ultrasound", "imu", "encoder", "sync_pulse"]
    assert isinstance(devices["imu"].parameters, ImuSimulationParameters)
    assert devices["ultrasound"].writer == "block_binary"
    assert devices["ultrasound"].required
    assert devices["ultrasound"].parameters.frame_shape is None
    assert devices["ultrasound"].parameters.channel_count == 4
    assert devices["ultrasound"].parameters.samples_per_channel == 1000
    assert devices["ultrasound"].parameters.frame_rate_hz == 20
    assert devices["sync_pulse"].required
    assert devices["sync_pulse"].clock_domain == "sync_pulse_sim_clock"
    assert devices["sync_pulse"].parameters.pulse_width_s == 0.02
    assert devices["sync_pulse"].parameters.first_pulse_s == 0.25
    assert TrialRunRequest(data_root=path.parent).duration_s is None


def test_profile_rejects_unapproved_adapter_unknown_parameters_and_missing_modality(
    tmp_path: Path,
) -> None:
    payload = load_simulated_device_profile().model_dump(mode="json", by_alias=True)

    unapproved = json.loads(json.dumps(payload))
    unapproved["devices"][0]["adapter"] = "untrusted.module.ArbitraryAdapter"
    with pytest.raises(ValidationError, match="literal_error"):
        SimulatedDeviceProfileDocument.model_validate(unapproved)

    unknown_parameter = json.loads(json.dumps(payload))
    unknown_parameter["devices"][1]["parameters"]["made_up_setting"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SimulatedDeviceProfileDocument.model_validate(unknown_parameter)

    missing = json.loads(json.dumps(payload))
    missing["devices"].pop()
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValidationError, match="must define ultrasound"):
        load_simulated_device_profile(path)


def test_static_adapter_factory_applies_profile_then_request_overrides(tmp_path: Path) -> None:
    payload = load_simulated_device_profile().model_dump(mode="json", by_alias=True)
    imu = next(device for device in payload["devices"] if device["modality"] == "imu")
    imu["id"] = "imu_profile_device"
    imu["clock_domain"] = "imu_profile_clock"
    imu["required"] = False
    imu["parameters"].update(
        {"sample_rate_hz": 111.0, "samples_per_batch": 11, "queue_capacity": 9}
    )
    profile = SimulatedDeviceProfileDocument.model_validate(payload)
    request = TrialRunRequest(
        data_root=tmp_path,
        simulation={
            "imu": {
                "sample_rate_hz": 321.0,
                "queue_capacity": 7,
                "drop_every_n_batches": 5,
            }
        },
    )

    adapters = _make_adapters(request, profile)
    descriptor = adapters["imu"].descriptor()
    snapshot = adapters["imu"].configuration_snapshot()

    assert descriptor.device_id == "imu_profile_device"
    assert descriptor.clock_domain == "imu_profile_clock"
    assert descriptor.nominal_rate_hz == 321.0
    assert snapshot["samples_per_batch"] == 11
    assert snapshot["queue_capacity"] == 7
    assert snapshot["drop_every_n_batches"] == 5
    assert profile.by_modality()["imu"].required is False

    invalid_request = request.model_copy(
        update={"simulation": {"unregistered_modality": {"sample_rate_hz": 1}}}
    )
    with pytest.raises(ValueError, match="Unknown simulated modality override"):
        _make_adapters(invalid_request, profile)


def test_hardware_profile_is_strict_and_explicitly_has_simulated_sync() -> None:
    profile = load_device_profile("hardware")
    assert isinstance(profile, HardwareDeviceProfileDocument)
    assert profile.display_name == "真实三设备 + 模拟同步（台架验证）"
    assert profile.laboratory_sync_ready is False
    devices = profile.by_modality()
    assert [devices[name].simulated for name in devices] == [False, False, False, True]
    assert devices["ultrasound"].parameters.sdk_path is None
    assert devices["imu"].parameters.sensor_ids == ()
    assert devices["encoder"].parameters.port is None


def test_hardware_registry_constructs_without_loading_vendor_sdks() -> None:
    adapters = build_adapters(load_device_profile("hardware"))
    assert isinstance(adapters["ultrasound"], ElonxiUltrasoundAdapter)
    assert isinstance(adapters["imu"], XsensAwindaImuAdapter)
    assert isinstance(adapters["encoder"], TeensySerialEncoderAdapter)
    assert isinstance(adapters["sync_pulse"], SimulatedSyncPulseAdapter)
    assert adapters["ultrasound"].descriptor().metadata["simulated"] is False


def test_hardware_overrides_are_strictly_revalidated() -> None:
    profile = load_device_profile("hardware")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        build_adapters(profile, {"imu": {"invented": 1}})
    with pytest.raises(ValidationError, match="expected_device_count"):
        build_adapters(profile, {"imu": {"expected_device_count": 2}})
    with pytest.raises(ValidationError, match="channels"):
        build_adapters(profile, {"ultrasound": {"channels": [1, 2, 3, 9]}})
    with pytest.raises(ValueError, match="unknown device override modality"):
        build_adapters(profile, {"camera": {}})
