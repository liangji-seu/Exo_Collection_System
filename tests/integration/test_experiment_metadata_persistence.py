from __future__ import annotations

import json

import h5py

from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.orchestration.simulated import run_simulated_trial
from exo_collection.storage.manifest import load_manifest


def test_experiment_metadata_is_persisted_through_configuration_artifact(tmp_path) -> None:
    request = TrialRunRequest(
        data_root=tmp_path,
        duration_s=0.1,
        experiment_metadata={
            "subject": {"height_cm": 172.5, "weight_kg": 64.2},
            "ultrasound_probe": {
                "muscle": "gastrocnemius medialis",
                "laterality": "left",
                "longitudinal_position": "proximal",
                "channel_mapping": ["GM proximal", "GM mid", None, "GM distal"],
                "fixation_method": "elastic wrap",
                "strap_pressure": "scale mark 2",
                "probe_reapplied": True,
            },
            "measured_condition": {
                "treadmill_speed_mps": 0.75,
                "assist_level": 20,
                "load_kg": 3,
                "slope_deg": 0,
            },
            "trial_notes": "Probe was reapplied before this trial.",
        },
    )

    result = run_simulated_trial(request)
    snapshot_path = result.trial_directory / "derived/configuration_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    saved = snapshot["experiment_metadata"]

    assert saved == request.experiment_metadata.model_dump(mode="json")
    assert saved["subject"]["height_cm"] == 172.5
    assert saved["ultrasound_probe"]["channel_mapping"] == [
        "GM proximal",
        "GM mid",
        None,
        "GM distal",
    ]

    manifest = load_manifest(result.manifest_path)
    configuration_artifact = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.relative_path == "derived/configuration_snapshot.json"
    )
    assert configuration_artifact.immutable

    for modality in ("imu", "encoder", "sync_pulse"):
        with h5py.File(result.trial_directory / f"raw/{modality}.h5", "r") as file:
            embedded = json.loads(file["metadata/trial"].asstr()[()])
        assert embedded["project_uuid"] == str(request.project_uuid)
        assert embedded["subject_uuid"] == str(request.subject_uuid)
        assert embedded["session_uuid"] == str(request.session_uuid)
        assert embedded["trial_uuid"] == str(request.trial_uuid)
        assert embedded["project_code"] == "T"
        assert embedded["subject_code"] == "001"
        assert embedded["condition"] == manifest.condition.model_dump(mode="json")
        assert embedded["experiment_metadata"] == (
            request.experiment_metadata.model_dump(mode="json")
        )
