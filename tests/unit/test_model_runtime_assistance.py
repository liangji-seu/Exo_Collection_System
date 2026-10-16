from __future__ import annotations

import math

import pytest

from exo_collection.apps.model_runtime.assistance import AssistancePipeline
from exo_collection.apps.model_runtime.spec import AssistanceSpec


def test_assistance_scales_mass_gain_sign_and_time_delay():
    pipeline = AssistancePipeline(
        AssistanceSpec(mass_kg=60.0, gain=0.5, lowpass_hz=0.0, delay_s=0.1, right_sign=-1)
    )
    first = pipeline.process(0, 2.0)
    second = pipeline.process(100_000_000, 2.0)
    assert first.desired_right_nm == 0.0  # delay is warming up
    assert second.estimate_nm == pytest.approx(120.0)
    assert second.desired_right_nm == pytest.approx(-60.0)


def test_lowpass_is_causal_and_reset_clears_delayed_nonzero_output():
    pipeline = AssistancePipeline(AssistanceSpec(mass_kg=50.0, gain=1.0, lowpass_hz=3.0))
    assert pipeline.process(0, 2.0).desired_right_nm == pytest.approx(100.0)
    filtered = pipeline.process(50_000_000, 0.0)
    assert 0.0 < filtered.desired_right_nm < 100.0
    pipeline.reset()
    assert pipeline.process(0, 0.0).desired_right_nm == 0.0


def test_nonfinite_estimate_is_immediate_zero():
    pipeline = AssistancePipeline(AssistanceSpec(mass_kg=60.0, gain=1.0))
    sample = pipeline.process(0, math.nan)
    assert sample.estimate_nm == 0.0
    assert sample.desired_right_nm == 0.0
