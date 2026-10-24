from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from exo_collection.apps.model_runtime.gates import load_gate
from exo_collection.apps.model_runtime.gates.contract import threshold_allows


APP = Path(__file__).resolve().parents[2]
WORKSPACE = APP.parent


def _import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sample_stream(n: int, seed: int = 71) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 100
    x = rng.normal(0, .15, size=(n, 12))
    for channel, phase, amp, freq in ((3, .1, 1.0, 1.1), (4, .4, .4, 1.1), (5, .9, .7, 1.1),
                                      (9, .5, .7, 1.1), (10, .8, .3, 1.1), (11, .2, .5, 1.1)):
        x[:, channel] += amp * np.sin(2 * np.pi * freq * t + phase)
    return x


def test_none_gate_emits_each_fifth_sample_and_reset_clears_sequence():
    gate = load_gate("none")
    emitted = [gate.step(i * 10_000_000, np.zeros(12)) for i in range(10)]
    decisions = [item for item in emitted if item]
    assert [item.source_grid_ns for item in decisions] == [40_000_000, 90_000_000]
    assert all(item.ready and item.allow and item.score is None for item in decisions)
    gate.reset()
    assert gate.step(0, np.zeros(12)) is None


def test_frozen_threshold_boundary_passes_equality_and_fails_nonfinite():
    threshold = 0.08490906655788422
    assert threshold_allows(threshold, threshold)
    assert not threshold_allows(np.nextafter(threshold, np.inf), threshold)
    assert not threshold_allows(float("nan"), threshold)
    assert not threshold_allows(float("inf"), threshold)


def test_ao_stream_matches_frozen_batch_features_and_scores():
    source = WORKSPACE / "paper_experiments" / "ao_research_loop_20261024" / "code" / "features.py"
    features_module = _import_file(source, "frozen_gate_ao_r_features_test")
    import json

    audit_path = WORKSPACE / "paper_experiments" / "ao_research_loop_20261024" / "round01_dev_attempt04" / "normalizer_audit.json"
    normalizer = json.loads(audit_path.read_text(encoding="utf-8"))["normalizer"]
    context = features_module.load_core_context(WORKSPACE)
    x = _sample_stream(1500)
    t = np.arange(len(x), dtype=np.float64) * .01
    q = np.ones(len(x), dtype=bool)
    core = context["module"].run_periodic(x, t, q, config=context["config"], normalizer=normalizer)
    endpoints = np.arange(4, len(x), 5, dtype=np.int64)
    features = features_module.b_adapter.features_from_core(x, q, endpoints, core)
    model = load_gate("ao")
    decisions = []
    observed_features = []
    for i, sample in enumerate(x):
        result = model.step(int(i * 10_000_000), sample)
        if result is not None:
            decisions.append(result)
            if result.ready:
                observed_features.append(model._features().copy())
    assert len(decisions) == len(endpoints)
    assert np.array_equal([item.ready for item in decisions], features["valid"]["AO"])
    ready = features["valid"]["AO"]
    np.testing.assert_allclose(np.asarray(observed_features), features["features"]["AO"][ready],
                               rtol=0, atol=1e-10)
    reference = model._model.predict_proba(features["features"]["AO"][ready])[:, model._ood_column]
    actual = np.asarray([item.raw_score for item in decisions if item.ready])
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)
    assert all(item.allow == (item.ready and item.raw_score <= item.threshold) for item in decisions)
    assert all(item.source_grid_ns == int(ep * 10_000_000) for item, ep in zip(decisions, endpoints))


def test_ao_invalid_sample_resets_contiguous_readiness():
    gate = load_gate("ao")
    x = _sample_stream(800)
    decisions = []
    for i, sample in enumerate(x):
        if i == 400:
            sample = sample.copy()
            sample[0] = np.nan
        item = gate.step(i * 10_000_000, sample)
        if item is not None:
            decisions.append(item)
    after_gap = [item for item in decisions if item.source_grid_ns >= 4_000_000_000]
    assert after_gap
    assert not after_gap[0].ready
    assert not any(item.ready for item in after_gap[:15])


def test_ao_rejects_old_base_normalizer_binding():
    import copy

    from exo_collection.apps.model_runtime.gates.loader import _read_manifest, _verify_ao_bindings

    manifest = copy.deepcopy(_read_manifest("ao.json"))
    manifest["normalizer"]["scales_rad_s"] = [1.599924691, 0.443698068]
    with pytest.raises(RuntimeError, match="R1 train-only audit"):
        _verify_ao_bindings(manifest)


def test_phase_gate_matches_frozen_runner_member_variance_and_median():
    import torch

    comparison = WORKSPACE / "paper_experiments" / "ao_gate_refinement_20261021" / "paper_comparison_20261022"
    runner = _import_file(comparison / "code" / "frozen_fit_runner.py", "frozen_gate_fit_runner_test")
    checkpoint_dir = comparison / "tcn_fit"
    models, scalers = [], []
    for seed in (17, 29, 43, 59, 71, 83, 97):
        ck = torch.load(checkpoint_dir / f"GaitPhase_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = runner.build_tcn(torch)
        model.load_state_dict(ck["model_state"], strict=True)
        models.append(model)
        scalers.append(ck["input_scaler"])
    x = _sample_stream(480, seed=11).astype(np.float32)
    endpoints = np.arange(4, len(x), 5, dtype=np.int64)
    valid = np.asarray([ep >= 99 for ep in endpoints])
    members = np.full((len(endpoints), 7), np.nan, dtype=np.float32)
    if valid.any():
        windows = runner.causal_windows(x, endpoints[valid])
        members[valid] = runner.predict_members(torch, models, scalers, windows, "cpu", batch_size=1)
    raw = np.full(len(endpoints), np.nan, dtype=np.float32)
    raw[valid] = runner.ensemble_variance(members[valid])
    median = runner.causal_median(raw, endpoints.astype(np.float64) * .01, valid, runner.FILTER_SECONDS)
    batched_members = np.full((len(endpoints), 7), np.nan, dtype=np.float32)
    if valid.any():
        batched_members[valid] = runner.predict_members(torch, models, scalers, windows, "cpu")
    batched_raw = np.full(len(endpoints), np.nan, dtype=np.float32)
    batched_raw[valid] = runner.ensemble_variance(batched_members[valid])
    batched_median = runner.causal_median(batched_raw, endpoints.astype(np.float64) * .01, valid,
                                          runner.FILTER_SECONDS)

    gate = load_gate("phase_ensemble")
    decisions = []
    for i, sample in enumerate(x):
        item = gate.step(i * 10_000_000, sample)
        if item is not None:
            decisions.append(item)
    assert len(decisions) == len(endpoints)
    actual_valid = np.asarray([item.ready for item in decisions])
    np.testing.assert_array_equal(actual_valid, valid)
    np.testing.assert_allclose([item.raw_score for item in decisions if item.ready], raw[valid], rtol=0, atol=2e-8)
    actual_median = np.asarray([item.score for item in decisions if item.ready])
    np.testing.assert_allclose(actual_median, median[valid], rtol=0, atol=2e-8,
                               err_msg=f"median mismatch indices {np.flatnonzero(np.abs(actual_median - median[valid]) > 2e-8).tolist()}")
    assert all(item.allow == (item.score <= item.threshold) for item in decisions if item.ready)
    assert all(not item.allow for item in decisions if not item.ready)
    # Frozen evaluation batches can differ from online singleton convolution by <1e-6;
    # verify that this floating-point variation does not change threshold decisions here.
    batch_error = np.max(np.abs(actual_median - batched_median[valid]))
    assert batch_error <= 1e-6
    assert np.array_equal(actual_median <= decisions[-1].threshold, batched_median[valid] <= decisions[-1].threshold)


def test_phase_gate_invalid_sample_clears_window_and_median():
    gate = load_gate("phase_ensemble")
    x = _sample_stream(400, seed=9).astype(np.float32)
    decisions = []
    for i, sample in enumerate(x):
        if i == 200:
            sample = sample.copy()
            sample[2] = np.nan
        item = gate.step(i * 10_000_000, sample)
        if item is not None:
            decisions.append(item)
    assert any(item.ready for item in decisions if item.source_grid_ns < 2_000_000_000)
    after_gap = [item for item in decisions if item.source_grid_ns >= 2_000_000_000]
    assert after_gap and not after_gap[0].ready
    assert not any(item.ready for item in after_gap[:20])


@pytest.mark.parametrize("mode", ["ao", "phase_ensemble"])
@pytest.mark.parametrize("anomaly", ["duplicate", "backward", "gap"])
def test_engines_reset_on_nonmonotonic_or_gapped_source_grid(mode, anomaly):
    x = _sample_stream(800, seed=19)
    anomaly_index = 500
    source_times = np.arange(len(x), dtype=np.int64) * 10_000_000
    if anomaly == "duplicate":
        source_times[anomaly_index] = source_times[anomaly_index - 1]
    elif anomaly == "backward":
        source_times[anomaly_index] = source_times[anomaly_index - 1] - 10_000_000
    else:
        source_times[anomaly_index:] += 30_000_000

    gate = load_gate(mode)
    decisions = []
    for timestamp, sample in zip(source_times, x):
        item = gate.step(int(timestamp), sample)
        if item is not None:
            decisions.append(item)
    decision_indices = [item.sequence * 5 - 1 for item in decisions]
    assert any(item.ready for item, index in zip(decisions, decision_indices) if index < anomaly_index)
    after_anomaly = [item for item, index in zip(decisions, decision_indices) if index >= anomaly_index]
    assert after_anomaly and not after_anomaly[0].ready
    assert not any(item.ready for item in after_anomaly[:20])


@pytest.mark.parametrize("mode", ["ao", "phase_ensemble"])
def test_engine_outputs_are_prefix_causal(mode):
    x = _sample_stream(520, seed=31)
    cutoff = 400

    def run(samples):
        gate = load_gate(mode)
        output = []
        for i, sample in enumerate(samples):
            item = gate.step(i * 10_000_000, sample)
            if item is not None:
                output.append(item)
        return output

    prefix = run(x[:cutoff])
    altered_tail = x.copy()
    altered_tail[cutoff:] = _sample_stream(len(x) - cutoff, seed=777) * 20
    complete = run(altered_tail)
    assert len(prefix) == cutoff // 5
    for left, right in zip(prefix, complete[:len(prefix)]):
        assert left == right


def test_phase_gate_rejects_tampered_asset_hash(tmp_path):
    from exo_collection.apps.model_runtime.gates.loader import _read_manifest
    from exo_collection.apps.model_runtime.gates.phase_ensemble import PhaseEnsembleGate

    manifest = _read_manifest("phase_ensemble.json")
    source = WORKSPACE / manifest["assets"]["directory"]
    for seed in (17, 29, 43, 59, 71, 83, 97):
        target = tmp_path / f"GaitPhase_seed{seed}.pt"
        target.write_bytes((source / target.name).read_bytes())
    target = tmp_path / "GaitPhase_seed17.pt"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        PhaseEnsembleGate(manifest, tmp_path)
