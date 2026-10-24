"""Read-only historical replay of the same torque model with three gate modes.

This reports algorithm output and compute cost, not real-time scheduling or
physical motor validation. No serial backend is constructed or enabled.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
import torch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "src"))

from exo_collection.apps.model_runtime.assistance import AssistancePipeline
from exo_collection.apps.model_runtime.spec import AssistanceSpec
from exo_collection.apps.model_runtime.gates import load_gate


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_grid(recording, seconds):
    session = json.loads((recording / ".exo/online_session.json").read_text(encoding="utf-8"))
    if not session.get("complete") or session.get("errors"):
        raise ValueError("Use a completed recording for this acceptance replay")
    start, stop = int(session["start_host_monotonic_ns"]), int(session["stop_host_monotonic_ns"])
    stop = min(stop, start + int(seconds * 1e9))
    with h5py.File(recording / "raw/imu.h5", "r") as f:
        stamps = np.asarray(f["samples/host_monotonic_ns"], dtype=np.int64)
        data = np.asarray(f["samples/data"], dtype=np.float32)
        metadata = json.loads(f["metadata/device"][()].decode("utf-8"))
    labels = metadata.get("preview_labels", metadata.get("metadata", {}).get("preview_labels", []))
    grid = np.arange(((start + 9_999_999) // 10_000_000) * 10_000_000, stop, 10_000_000, dtype=np.int64)
    chunks, good = [], np.ones(len(grid), dtype=bool)
    for name in ("imu_right_leg", "imu_pelvis"):
        slot = labels.index(name)
        # Each multiplexed packet contains one actual device and NaNs for others.
        present = np.isfinite(data[:, slot, :6]).all(axis=1) & (stamps >= start) & (stamps < stop)
        times, values = stamps[present], data[present, slot, :6]
        if not len(times) or np.any(np.diff(times) < 0):
            raise ValueError(f"Missing or unordered {name} stream")
        index = np.searchsorted(times, grid, side="right") - 1
        clipped = np.clip(index, 0, len(times) - 1)
        age = grid - times[clipped]
        good &= (index >= 0) & (age >= 0) & (age <= 50_000_000)
        chunks.append(values[clipped])
    return grid, np.concatenate(chunks, axis=1), good


def stats(values):
    a = np.asarray(values)
    return {"count": len(a), "mean_ms": float(np.mean(a)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(np.max(a))} if len(a) else {"count": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=APP / "runtime_data/online_recordings/20261016_210359_806362")
    parser.add_argument("--model", type=Path, default=APP / "runtime_data/models/imu_right_pelvis_torque_100hz_rf94_static_20261017/final_model.ts")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=APP / "runtime_data/online_qa/gate_integration_20261024")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with args.model.open("rb") as handle:
        model = torch.jit.load(handle, map_location="cpu").eval()
    grid, values, valid = read_grid(args.recording, args.seconds)
    predictions, torque_cost = {}, []
    with torch.inference_mode():
        for i in range(93, len(grid)):
            if not valid[i-93:i+1].all():
                continue
            t0 = perf_counter()
            predictions[i] = float(model(torch.from_numpy(values[i-93:i+1].copy()).unsqueeze(0)).reshape(-1)[0])
            torque_cost.append((perf_counter() - t0) * 1000)
    results = {}
    traces = {}
    config = AssistanceSpec(mass_kg=80.0, gain=0.2, lowpass_hz=3.0, delay_s=0.25,
                            motor_distribution="right_to_both_same_sign")
    for mode in ("none", "ao", "phase_ensemble"):
        engine, pipeline = load_gate(mode), AssistancePipeline(config)
        costs, rows = [], []
        decision = None
        with torch.inference_mode():
            for i, timestamp in enumerate(grid):
                t0 = perf_counter()
                new = engine.step(int(timestamp), values[i], valid=bool(valid[i]))
                costs.append((perf_counter() - t0) * 1000)
                if new is not None:
                    decision = new
                if i not in predictions:
                    continue
                sample = pipeline.process(int(timestamp), predictions[i])
                ready = decision is not None and decision.ready
                allowed = bool(ready and decision.allow and 0 <= timestamp - decision.source_grid_ns <= 100_000_000)
                rows.append({"source_grid_ns": int(timestamp), "mode": mode,
                             "prediction_nm_per_kg": predictions[i], "prediction_nm": sample.estimate_nm,
                             "pre_gate_nm": sample.desired_right_nm, "gate_ready": bool(ready),
                             "gate_allow": allowed, "decision": asdict(decision) if decision else None})
        # These are algorithm traces; command dispatch/recovery is tested separately.
        with (args.output / f"replay_{mode}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        results[mode] = {"samples": len(rows), "ready": sum(r["gate_ready"] for r in rows),
                         "allow": sum(r["gate_allow"] for r in rows), "compute": stats(costs),
                         "provenance": getattr(engine, "provenance", {})}
        traces[mode] = rows
    assert len({len(v) for v in traces.values()}) == 1
    for a, b, c in zip(*traces.values()):
        assert a["prediction_nm"] == b["prediction_nm"] == c["prediction_nm"]
        assert a["pre_gate_nm"] == b["pre_gate_nm"] == c["pre_gate_nm"]
    common = [i for i in range(len(traces["none"])) if all(v[i]["gate_ready"] for v in traces.values())]
    report = {"schema": "offline_gate_algorithm_replay_v1", "hardware_accessed": False,
              "scope": "algorithm replay; excludes worker deadlines, actual dispatch and physical safety",
              "recording": str(args.recording), "raw_sha256": sha256(args.recording / "raw/imu.h5"),
              "model": str(args.model), "model_sha256": sha256(args.model),
              "parameters": config.model_dump(), "grid_samples": len(grid), "valid_grid_samples": int(valid.sum()),
              "torque_compute": stats(torque_cost), "modes": results,
              "same_torque_and_pre_gate_assistance": True, "common_ready_samples": len(common),
              "common_ready_allow": {k: sum(v[i]["gate_allow"] for i in common) for k, v in traces.items()}}
    (args.output / "algorithm_replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
