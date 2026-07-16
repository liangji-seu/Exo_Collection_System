from __future__ import annotations

import pytest
from pydantic import ValidationError

from exo_collection.orchestration.models import (
    TrialExperimentMetadata,
    TrialRunRequest,
)


def test_experiment_metadata_is_optional_and_blank_text_is_normalized(tmp_path) -> None:
    request = TrialRunRequest(
        data_root=tmp_path,
        experiment_metadata={
            "ultrasound_probe": {
                "muscle": "   ",
                "channel_mapping": [" vastus lateralis ", "", None, "  "],
            },
            "trial_notes": "  ",
        },
    )

    metadata = request.experiment_metadata
    assert metadata.subject.height_cm is None
    assert metadata.ultrasound_probe.muscle is None
    assert metadata.ultrasound_probe.channel_mapping == (
        "vastus lateralis",
        None,
        None,
        None,
    )
    assert metadata.trial_notes is None


def test_experiment_metadata_accepts_complete_structured_record() -> None:
    metadata = TrialExperimentMetadata.model_validate(
        {
            "subject": {
                "height_cm": 175.2,
                "weight_kg": 68.5,
                "leg_length_cm": 92.0,
                "sex": "male",
                "age_years": 24,
            },
            "ultrasound_probe": {
                "muscle": "vastus lateralis",
                "laterality": "right",
                "longitudinal_position": "middle",
                "channel_mapping": ["proximal", "middle-1", "middle-2", "distal"],
                "fixation_method": "elastic wrap",
                "strap_pressure": "scale mark 3",
                "probe_reapplied": False,
            },
            "measured_condition": {
                "treadmill_speed_mps": 0.8,
                "assist_level": 30,
                "load_kg": 5,
                "slope_deg": -5,
            },
            "trial_notes": "No visible probe slip.",
        }
    )

    assert metadata.subject.age_years == 24
    assert metadata.ultrasound_probe.channel_mapping[3] == "distal"
    assert metadata.measured_condition.slope_deg == -5


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("subject", "height_cm", 300),
        ("subject", "weight_kg", -1),
        ("subject", "leg_length_cm", 250),
        ("subject", "age_years", 121),
        ("measured_condition", "treadmill_speed_mps", -0.1),
        ("measured_condition", "assist_level", 101),
        ("measured_condition", "load_kg", 501),
        ("measured_condition", "slope_deg", 46),
    ],
)
def test_experiment_metadata_rejects_out_of_range_values(
    section: str, field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        TrialExperimentMetadata.model_validate({section: {field: value}})


def test_ultrasound_channel_mapping_requires_exactly_four_channels() -> None:
    with pytest.raises(ValidationError):
        TrialExperimentMetadata.model_validate(
            {"ultrasound_probe": {"channel_mapping": ["ch1", "ch2"]}}
        )
