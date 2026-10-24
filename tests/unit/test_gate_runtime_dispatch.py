from dataclasses import dataclass
import threading
import time
import types
import sys
from pathlib import Path

import numpy as np
from queue import Queue

from exo_collection.apps.model_runtime.gate_dispatch import GateDispatch
from exo_collection.apps.model_runtime.gate_worker import GateResult
from exo_collection.apps.model_runtime.assistance import AssistancePipeline
from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.inference import _quick_grid_retry_deadline
from exo_collection.apps.model_runtime.spec import AssistanceSpec
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


@dataclass(frozen=True)
class Decision:
    source_grid_ns: int
    sequence: int
    mode: str = "ao"
    ready: bool = True
    allow: bool = True
    score: float = .2
    threshold: float = .5
    reason: str = "below_threshold"


def result(source, computed, generation=7, decision=None):
    return GateResult(decision or Decision(source, source // 50_000_000),
                      computed, generation, 1.5)


def test_only_current_generation_completed_decisions_are_selectable():
    gate = GateDispatch("ao")
    gate.reset(7)
    gate.publish([result(900_000_000, 900_000_000, generation=6)])
    allowed, _, status = gate.decide(source_grid_ns=1_000_000_000,
                                     now_ns=1_000_000_000)
    assert not allowed
    assert status.state == "warming"

    gate.publish([result(950_000_000, 955_000_000)])
    allowed, epoch, status = gate.decide(source_grid_ns=1_000_000_000,
                                         now_ns=1_000_000_000)
    assert allowed
    assert epoch is not None
    assert status.sequence == 19
    assert status.state == "allow"


def test_reject_future_and_both_ttl_clocks_fail_closed():
    gate = GateDispatch("ao")
    gate.reset(2)
    gate.publish([result(990_000_000, 1_001_000_000, generation=2)])
    allowed, _, status = gate.decide(source_grid_ns=1_000_000_000,
                                     now_ns=1_000_000_000)
    assert not allowed
    assert status.state == "stale"

    gate.reset(2)
    rejected = Decision(990_000_000, 1, ready=True, allow=False,
                        score=.7, threshold=.5, reason="above_threshold")
    gate.publish([result(990_000_000, 990_000_000, 2, rejected)])
    allowed, _, status = gate.decide(source_grid_ns=1_000_000_000,
                                     now_ns=1_000_000_000)
    assert not allowed
    assert status.state == "rejected"

    gate.reset(2)
    gate.publish([result(900_000_000, 900_000_000, 2)])
    allowed, _, _ = gate.decide(source_grid_ns=920_000_000,
                                now_ns=1_000_000_001)
    assert not allowed  # endpoint-source age exceeds 100 ms


def test_unselected_or_future_result_cannot_extend_last_allowed_ttl():
    gate = GateDispatch("ao")
    gate.reset(3)
    old = result(1_000_000_000, 1_000_000_000, 3)
    gate.publish([old])
    assert gate.decide(source_grid_ns=1_000_000_000,
                       now_ns=1_000_000_000)[0]
    gate.publish([result(1_200_000_000, 1_200_000_000, 3)])
    assert gate.expire(1_100_000_001)
    assert gate.status.state == "stale"


def test_zero_delay_command_holds_latest_20hz_decision_for_100hz_ticks():
    gate = GateDispatch("ao")
    gate.reset(4)
    gate.publish([result(100_000_000, 101_000_000, generation=4)])
    for source in (110_000_000, 120_000_000):
        allowed, _, status = gate.decide(source_grid_ns=source, now_ns=source,
                                         candidate_source_ns=source)
        assert allowed
        assert status.source_grid_ns == 100_000_000


def test_quick_retry_polls_only_empty_grid_until_next_regular_deadline():
    regular = 10_000_000
    deadline = _quick_grid_retry_deadline(0, "no new source grid", regular)
    assert deadline == 1_000_000
    for _ in range(8):
        deadline = _quick_grid_retry_deadline(deadline, "no new source grid", regular)
    assert deadline == 9_000_000
    assert _quick_grid_retry_deadline(deadline, "no new source grid", regular) is None
    assert _quick_grid_retry_deadline(0, "warmup", regular) is None
    assert _quick_grid_retry_deadline(9_500_000, "no new source grid", regular) is None


def test_none_mode_preserves_unconditional_pass_through():
    gate = GateDispatch("none")
    gate.reset(1)
    allowed, epoch, status = gate.decide(source_grid_ns=0, now_ns=10**12,
                                         worker_closed="ignored")
    assert allowed and epoch is None
    assert status.state == "disabled"


def test_rejected_gate_interval_does_not_change_pregate_assistance_trajectory():
    config = AssistanceSpec(mass_kg=70, gain=.4, lowpass_hz=3, delay_s=.02)
    baseline = AssistancePipeline(config)
    gated = AssistancePipeline(config)
    values = [0.1, 0.3, -0.2, 0.4, 0.2, -0.1, 0.5]
    rejected = {2, 3, 4}
    baseline_pre, gated_pre, gated_post = [], [], []
    for index, estimate in enumerate(values):
        t_ns = index * 10_000_000
        reference = baseline.process(t_ns, estimate).desired_right_nm
        candidate = gated.process(t_ns, estimate).desired_right_nm
        baseline_pre.append(reference)
        gated_pre.append(candidate)
        gated_post.append(0.0 if index in rejected else candidate)
    assert gated_pre == baseline_pre
    assert all(gated_post[i] == 0 for i in rejected)
    assert gated_pre[5:] == baseline_pre[5:]


def test_watchdog_zeros_while_prediction_is_blocked(monkeypatch):
    class Engine:
        provenance = {"mode": "ao"}

        def reset(self):
            pass

        def step(self, source, values, valid=True):
            return None

    package_name = "exo_collection.apps.model_runtime.gates"
    package = types.ModuleType(package_name)
    package.load_gate = lambda mode, deployment_root=None: Engine()
    monkeypatch.setitem(sys.modules, package_name, package)

    clock = [1_000_000_000]
    spec = ModelSpec.model_validate({
        "schema_version": "1.0.0",
        "model": {"backend": "demo", "demo": {"amplitude": 1.0}},
        "features": {"control_rate_hz": 100.0, "window_s": .03,
                      "inputs": [{"modality": "imu", "channels": None}]},
        "output": {"kind": "torque_left_right"},
    })
    output = MockTorqueOutput(5.0, enable_required=False)
    controller = InferenceController(SubscriptionHub(), spec, build_model_runtime(spec),
                                     output, gate_mode="ao", clock_ns=lambda: clock[0],
                                     gate_process_isolated=False)
    started, release = threading.Event(), threading.Event()

    class BlockingRuntime:
        def predict(self, _features):
            started.set()
            release.wait(2)
            return np.array([0.0])

    def block_prediction():
        BlockingRuntime().predict(np.zeros((1,)))

    controller._generation = 1
    controller._enabled = True
    controller._gate_dispatch.reset(1)
    allowed_result = result(clock[0], clock[0], generation=1)
    controller._gate_dispatch.publish([allowed_result])
    assert controller._gate_dispatch.decide(source_grid_ns=clock[0], now_ns=clock[0])[0]
    output.send(2.0, 1.0)
    blocked = threading.Thread(target=block_prediction)
    blocked.start()
    assert started.wait(1)
    try:
        clock[0] += 100_000_001
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and output.last.left_nm != 0:
            time.sleep(.005)
        assert output.last.left_nm == output.last.right_nm == 0.0
        assert output.last.gate_context["state"] == "stale"
    finally:
        release.set()
        blocked.join(1)
        controller._gate_watchdog_stop.set()
        controller._gate_worker.close()


def test_mock_torque_copies_gate_context_at_command_boundary():
    from exo_collection.apps.model_runtime.torque import MockTorqueOutput

    output = MockTorqueOutput(5.0, enable_required=False)
    context = {"sequence": 8, "state": "allow", "post_left_nm": 1.25}
    output.send_with_gate_context(1.25, 0.0, context)
    context["sequence"] = 9
    output.send_with_gate_context(0.0, 0.0,
                                  {"sequence": 9, "state": "stale",
                                   "post_left_nm": 0.0})
    first, second = output.history
    assert first.gate_context["sequence"] == 8
    assert first.gate_context["post_left_nm"] == first.left_nm == 1.25
    assert second.gate_context["sequence"] == 9
    assert second.gate_context["state"] == "stale"


def test_display_only_run_feeds_gate_without_motor_enable(monkeypatch):
    from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
    from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor
    from exo_collection.apps.model_runtime.spec import load_model_spec

    class Decision:
        def __init__(self, source, seq):
            self.source_grid_ns = source
            self.sequence = seq
            self.ready = self.allow = True
            self.mode = "ao"
            self.score = .1
            self.threshold = .5
            self.reason = "below_threshold"

    class Engine:
        def __init__(self):
            self.seen = []

        def reset(self):
            pass

        def step(self, source, values, valid=True):
            self.seen.append(source)
            return Decision(source, len(self.seen)//5) if len(self.seen) % 5 == 0 else None

    package_name = "exo_collection.apps.model_runtime.gates"
    package = types.ModuleType(package_name)
    engine = Engine()
    package.load_gate = lambda mode, deployment_root=None: engine
    monkeypatch.setitem(sys.modules, package_name, package)

    project = Path(__file__).resolve().parents[2]
    bundle = project / "runtime_data/models/imu_right_pelvis_torque_100hz_rf94_static_20261017"
    spec = load_model_spec(bundle / "model_spec.json")
    spec.imu_tcn.right_device_id, spec.imu_tcn.pelvis_device_id = "right", "pelvis"
    now = [990_000_000]
    hub, queue = SubscriptionHub(), Queue()
    hub.register_stream("imu", queue, channels=IMU_CHANNELS, sample_shape=(2, 6), rate_hz=100,
        descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (2, 6),
                    "metadata": {"device_ids": ["right", "pelvis"]}})
    for index in range(100):
        at = index * 10_000_000
        data = np.full((1, 2, 6), index, dtype=np.float32)
        queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="imu",
            first_sample_index=index, sample_count=1, sequence_number=index,
            sample_rate_hz=100.0, host_monotonic_ns=at, data=data))

    class Runtime:
        def predict(self, _features):
            return np.array([.1])

    output = MockTorqueOutput(18.0, enable_required=True)
    controller = InferenceController(hub, spec, Runtime(), output,
        extractor=ImuTcnFeatureExtractor(spec.imu_tcn, clock_ns=lambda: now[0]),
        gate_mode="ao", gate_process_isolated=False, clock_ns=lambda: now[0])
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not controller._gate_worker.ready:
            time.sleep(.005)
        assert controller._gate_worker.ready
        controller._run_active = True
        controller._tick()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and len(engine.seen) < 94:
            time.sleep(.005)
        assert len(engine.seen) == 94
        assert not controller.enabled and not output.enabled
        assert output.last is not None and output.last.left_nm == output.last.right_nm == 0
    finally:
        controller.request_stop()
