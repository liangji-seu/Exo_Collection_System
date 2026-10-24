"""Replay recorded IMUs at wall-clock rate through the real controller, Mock only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from queue import Queue
import sys
import threading
from time import perf_counter_ns, sleep

import numpy as np
from PySide6.QtCore import QCoreApplication

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "src"))
from verify_online_gates import read_grid, sha256
from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.spec import load_model_spec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


def run_mode(mode, data, seconds, model_path):
    spec = load_model_spec(model_path)
    spec.imu_tcn.right_device_id, spec.imu_tcn.pelvis_device_id = "right", "pelvis"
    spec.assistance.mass_kg, spec.assistance.gain, spec.assistance.delay_s = 80.0, .2, .25
    runtime = build_model_runtime(spec, base_dir=model_path.parent)
    queue = Queue()
    hub = SubscriptionHub()
    channels = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
    hub.register_stream("imu", queue, channels=channels, sample_shape=(2, 6), rate_hz=100,
        descriptor={"channels": channels, "units": ("m/s2",)*3 + ("rad/s",)*3,
                    "sample_shape": (2, 6), "metadata": {"device_ids": ["right", "pelvis"]}})
    motor = MockTorqueOutput(torque_limit_nm=18.0, history_capacity=10000)
    samples, diagnostics = [], []
    controller = InferenceController(hub, spec, runtime, motor, gate_mode=mode,
        recording_callback=samples.append, diagnostic_callback=lambda k, v: diagnostics.append((k, v)),
        gate_recording_callback=lambda k, v: diagnostics.append((k, v)))
    stop = threading.Event()
    origin = (perf_counter_ns() // 10_000_000) * 10_000_000

    def feed(index, timestamp):
        values = data[index % len(data)].reshape(1, 2, 6).copy()
        queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="replay",
            host_monotonic_ns=timestamp, first_sample_index=index, sample_count=1,
            sequence_number=index, sample_rate_hz=100.0, data=values))

    # Seed a complete torque window; gates start their own fresh warmup.
    for i in range(101):
        feed(i, origin - (100-i)*10_000_000)
    hub.drain(max_per_stream=256)

    def feeder():
        i = 101
        while not stop.is_set():
            target = origin + (i-100)*10_000_000
            remaining = (target-perf_counter_ns())/1e9
            if remaining > 0:
                sleep(min(.001, remaining))
                continue
            feed(i, target)
            i += 1

    thread = threading.Thread(target=feeder, daemon=True)
    thread.start()
    start = perf_counter_ns()
    try:
        controller.set_enabled(True)
        controller.start()
        while perf_counter_ns()-start < seconds*1e9 and not controller.last_error:
            QCoreApplication.processEvents()
            sleep(.005)
    finally:
        controller.stop_and_wait(3000)
        stop.set()
        thread.join(1)
        QCoreApplication.processEvents()
    gate_states = [s.get("gate_status") for s in samples]
    states = Counter((asdict(s) if is_dataclass(s) else s or {}).get("state", "disabled") for s in gate_states)
    decisions = [v for k, v in diagnostics if k == "gate_decision"]
    history = motor.history
    result = {"mode": mode, "seconds": seconds, "predictions": len(samples),
              "prediction_hz": len(samples)/seconds, "last_error": controller.last_error,
              "gate_states": dict(states), "gate_decisions": len(decisions),
              "gate_ready_decisions": sum(bool(d.get("ready")) for d in decisions),
              "gate_allow_decisions": sum(bool(d.get("allow")) for d in decisions),
              "gate_drops": controller._gate_worker.dropped,
              "max_abs_command_nm": max((max(abs(c.left_nm), abs(c.right_nm)) for c in history), default=0),
              "nonzero_commands": sum(bool(c.left_nm or c.right_nm) for c in history),
              "bilateral_same_sign_and_value": all(c.left_nm == c.right_nm for c in history),
              "stopped": not controller.isRunning(), "motor_disabled": not motor.enabled,
              "last_command_zero": bool(history and history[-1].left_nm == history[-1].right_nm == 0),
              "provenance": controller.gate_provenance}
    output = APP / "runtime_data/online_qa/gate_integration_20261024"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"realtime_run_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / f"realtime_diagnostics_{mode}.jsonl").open("w", encoding="utf-8") as stream:
        for kind, payload in diagnostics:
            stream.write(json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False,
                default=lambda v: asdict(v) if is_dataclass(v) else str(v)) + "\n")
    assert result["last_error"] is None, result
    assert result["predictions"] >= seconds * spec.features.control_rate_hz * .90, result
    assert result["gate_drops"] == 0, result
    assert result["max_abs_command_nm"] <= 18.0, result
    assert result["bilateral_same_sign_and_value"], result
    assert result["stopped"] and result["motor_disabled"] and result["last_command_zero"], result
    if mode != "none":
        # A valid allow can still be holding zero while the post-reopen delay
        # epoch warms. Judge readiness from decisions, not that transient label.
        assert result["gate_ready_decisions"] > 0, result
        for command in history:
            if command.left_nm or command.right_nm:
                context = command.gate_context or {}
                assert context.get("output_allowed") is True, context
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--mode", choices=("all", "none", "ao", "phase_ensemble"), default="all")
    args = parser.parse_args()
    app = QCoreApplication.instance() or QCoreApplication([])
    recording = APP / "runtime_data/online_recordings/20261016_210359_806362"
    _, values, valid = read_grid(recording, max(20.0, args.seconds + 2.0))
    values = values[np.flatnonzero(valid)[0]:]
    model_path = APP / "runtime_data/models/imu_right_pelvis_torque_100hz_rf94_static_20261017/model_spec.json"
    modes = ("none", "ao", "phase_ensemble") if args.mode == "all" else (args.mode,)
    results = [run_mode(mode, values, args.seconds, model_path) for mode in modes]
    report = {"hardware_accessed": False, "backend": "MockTorqueOutput", "raw_sha256": sha256(recording / "raw/imu.h5"), "runs": results}
    output = APP / "runtime_data/online_qa/gate_integration_20261024"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"realtime_{args.mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
