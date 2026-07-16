from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from exo_collection.configuration.device_profiles import (
    ImuSimulationParameters,
    SimulatedDeviceProfileDocument,
    default_simulated_device_profile_path,
    load_simulated_device_profile,
)
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
