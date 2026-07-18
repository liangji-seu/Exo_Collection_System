from __future__ import annotations

import json
from pathlib import Path

from exo_collection.adapters.encoder.simulated import SimulatedEncoderAdapter
from exo_collection.adapters.imu.simulated import SimulatedImuAdapter
from exo_collection.adapters.sync_pulse.simulated import SimulatedSyncPulseAdapter
from exo_collection.adapters.ultrasound.simulated import SimulatedUltrasoundAdapter
from exo_collection.configuration.device_profiles import DeviceProfileDocument
from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.orchestration.simulated import run_trial


def _hardware_protocol_fakes(
    _request: TrialRunRequest,
    profile: DeviceProfileDocument,
) -> dict[str, object]:
    """Exercise hardware-profile orchestration without importing vendor SDKs."""

    devices = profile.by_modality()
    return {
        "ultrasound": SimulatedUltrasoundAdapter(
            {
                "device_id": devices["ultrasound"].device_id,
                "clock_domain": devices["ultrasound"].clock_domain,
                "frame_rate_hz": 20.0,
                "channel_count": 4,
                "samples_per_channel": 1000,
            }
        ),
        "imu": SimulatedImuAdapter(
            {
                "device_id": devices["imu"].device_id,
                "clock_domain": devices["imu"].clock_domain,
                "device_ids": ("MTW_TRUNK", "MTW_LEFT", "MTW_RIGHT"),
                "sample_rate_hz": 200.0,
            }
        ),
        "encoder": SimulatedEncoderAdapter(
            {
                "device_id": devices["encoder"].device_id,
                "clock_domain": devices["encoder"].clock_domain,
                "sample_rate_hz": 100.0,
            }
        ),
        "sync_pulse": SimulatedSyncPulseAdapter(
            {
                "device_id": devices["sync_pulse"].device_id,
                "clock_domain": devices["sync_pulse"].clock_domain,
                "sample_rate_hz": 1000.0,
                "first_pulse_s": 0.05,
                "pulse_interval_s": 0.2,
                "pulse_width_s": 0.02,
            }
        ),
    }


def test_hardware_profile_reaches_immutable_package_with_injected_protocol_fakes(
    tmp_path: Path,
) -> None:
    result = run_trial(
        TrialRunRequest(
            data_root=tmp_path,
            device_profile_key="hardware",
            duration_s=0.2,
        ),
        adapter_factory=_hardware_protocol_fakes,  # type: ignore[arg-type]
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    devices = {item["modality"]: item for item in manifest["devices"]}
    modalities = {item["modality"]: item for item in manifest["modalities"]}
    assert devices["ultrasound"]["metadata"]["simulated"] is False
    assert devices["imu"]["metadata"]["simulated"] is False
    assert devices["encoder"]["metadata"]["simulated"] is False
    assert devices["sync_pulse"]["metadata"]["simulated"] is True
    assert modalities["ultrasound"]["frame_count"] > 0
    assert modalities["imu"]["sample_count"] > 0
    assert modalities["encoder"]["sample_count"] > 0
    assert result.manifest_path.is_file()
    assert not result.trial_directory.with_name(
        result.trial_directory.name + ".recording"
    ).exists()

    snapshot = json.loads(
        (result.trial_directory / "derived" / "configuration_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot["device_profile"]["key"] == "hardware"
    assert snapshot["device_profile"]["laboratory_sync_ready"] is False
