import importlib.util
import json
from pathlib import Path

import numpy as np


APP = Path(__file__).resolve().parents[2]
SOURCE = APP / "run_online_replay.py"
SPEC = importlib.util.spec_from_file_location("online_replay_for_gate_tests", SOURCE)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _session(tmp_path, events, *, metadata=None):
    (tmp_path / ".exo").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / ".exo/online_session.json").write_text(json.dumps({
        "start_host_monotonic_ns": 1_000_000_000,
        "stop_host_monotonic_ns": 2_000_000_000,
        "complete": True,
        "metadata": metadata or {},
    }), encoding="utf-8")
    (tmp_path / ".exo/manifest.json").write_text(json.dumps({"state": "closed"}), encoding="utf-8")
    with (tmp_path / "raw/online_events.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
    return replay.load_session(tmp_path)


def _event(kind, at, payload):
    return {"kind": kind, "host_monotonic_ns": at, "payload": payload}


def test_legacy_replay_preserves_shapes_and_marks_gate_unrecorded(tmp_path):
    prediction = {"time_ns": 1_200_000_000, "source_grid_time_ns": 1_200_000_000,
        "host_monotonic_ns": 1_205_000_000, "estimate_nm": 2.0,
        "desired_left_nm": 0.5, "desired_right_nm": 0.5}
    flat = {"host_monotonic_ns": 1_300_000_000, "seq": 11, "left_nm": 0.2, "right_nm": 0.3}
    wrapped = {"host_monotonic_ns": 1_300_000_000, "left_nm": 0.2, "right_nm": 0.3,
        "command": {**flat, "gate_context": None}}
    data = _session(tmp_path, [
        _event("prediction", 1_205_000_000, prediction),
        _event("sent_command", 1_300_000_000, flat),
        _event("command", 1_300_000_000, wrapped),
    ])

    assert data["prediction"].shape == (1, 5)
    assert data["command"].shape == (1, 3)
    assert data["gate_prediction"].shape == (1, 5)
    assert np.isnan(data["gate_prediction"][:, 1:]).all()
    assert data["gate_summary"]["label"] == "未记录"
    assert not data["gate_summary"]["recorded"]
    assert data["gate_decisions"] == [] and data["gate_transitions"] == []
    assert replay.summary(data)["sent_commands"] == 1


def test_new_gate_replay_separates_pre_post_and_deduplicates_commands(tmp_path):
    provenance = {"model_version": "frozen-ao", "threshold": 0.58, "assets": {"rf.joblib": "abc"}}
    prediction = {"time_ns": 1_400_000_000, "source_grid_time_ns": 1_400_000_000,
        "host_monotonic_ns": 1_405_000_000, "estimate_nm": 2.0,
        "desired_left_nm": 0.0, "desired_right_nm": 0.4,
        "gate_pre_left_nm": 0.2, "gate_pre_right_nm": 0.4,
        "gate_post_left_nm": 0.0, "gate_post_right_nm": 0.4,
        "gate_status": {"mode": "ao", "state": "allow", "sequence": 8, "generation": 3}}
    sent = {"host_monotonic_ns": 1_500_000_000, "seq": 17, "left_nm": 0.0, "right_nm": 0.4,
        "gate_context": {"mode": "ao", "sequence": 8, "source_grid_ns": 1_400_000_000,
                         "generation": 3, "state": "allow"}}
    wrapper = {"host_monotonic_ns": 1_500_000_000, "left_nm": 0.0, "right_nm": 0.4,
        "gate_mode": "ao", "gate_sequence": 8, "command": sent}
    data = _session(tmp_path, [
        _event("gate_decision", 999_999_999, {"ready": True, "allow": True}),
        _event("prediction", 1_405_000_000, prediction),
        _event("gate_decision", 1_450_000_000, {"ready": True, "allow": True, "reason": "allowed"}),
        _event("gate_transition", 1_460_000_000, {"state": "allow", "generation": 3}),
        _event("gate_transition", 2_000_000_000, {"state": "stale", "generation": 3}),
        _event("sent_command", 1_500_000_000, sent),
        _event("command", 1_500_000_000, wrapper),
    ], metadata={"gate": {"mode": "ao", "provenance": provenance}})

    assert data["prediction"].shape == (1, 5)
    assert data["command"].shape == (1, 3)
    np.testing.assert_allclose(data["gate_prediction"], [[0.4, 0.2, 0.4, 0.0, 0.4]])
    assert [event["kind"] for event in data["gate_decisions"]] == ["gate_decision"]
    assert [event["kind"] for event in data["gate_transitions"]] == ["prediction_status", "gate_transition"]
    assert data["gate_summary"]["recorded"]
    assert data["gate_summary"]["mode"] == "ao"
    assert data["gate_summary"]["provenance"] == provenance
    assert data["gate_summary"]["decision_count"] == 1
    assert data["gate_summary"]["transition_count"] == 1
    assert data["command_context"][0]["context"]["sequence"] == 8
    assert replay.summary(data)["sent_commands"] == 1


def test_mode_none_is_distinct_from_legacy_missing_gate_metadata(tmp_path):
    data = _session(tmp_path, [], metadata={"gate": {"mode": "none"}})
    assert data["gate_summary"]["label"] == "未启用"
    assert not data["gate_summary"]["recorded"]
