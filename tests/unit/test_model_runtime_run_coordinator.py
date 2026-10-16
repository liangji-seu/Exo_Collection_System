"""Operator intent state transitions without Qt or a physical device."""
from types import SimpleNamespace

import pytest

from exo_collection.apps.model_runtime.run_coordinator import (
    OutputState, RunCoordinator, RunState,
)


class Output:
    def __init__(self):
        self.enabled = False
        self.fault = None
        self.connected = True
        self.closed = False
        self.enables = 0
        self.feedback = SimpleNamespace(host_monotonic_ns=100, error=0, state=0)
        self.commands = []

    def set_enabled(self, value):
        if value:
            self.enables += 1  # real service awaits new state=2
        else:
            self.enabled = False

    def zero(self):
        self.commands.append((0, 0))

    def send(self, left, right):
        self.commands.append((left, right))

    def close(self):
        self.closed = True


class Controller:
    def __init__(self, output):
        self.output = output
        self.started = 0
        self.enabled = False
        self.stop_requested = False
        self.exits = True

    def set_enabled(self, value):
        self.enabled = value
        self.output.set_enabled(value)

    def start(self):
        self.started += 1

    def request_stop(self):
        self.stop_requested = True
        self.set_enabled(False)

    def stop_and_wait(self, timeout_ms):
        self.request_stop()
        return self.exits

    def set_torque_output(self, output):
        self.set_enabled(False)
        self.output.zero()
        self.output = output
        self.set_enabled(False)


def prepared():
    run = RunCoordinator(clock=lambda: 100)
    output = Output()
    controller = Controller(output)
    run.prepare(controller, output)
    return run, controller, output


def test_load_does_not_start_and_run_without_motor_never_arms():
    run, controller, _ = prepared()
    assert run.run_state is RunState.READY and controller.started == 0
    run.start()
    run.poll_output(lambda: None)
    assert controller.started == 1 and run.running
    assert not controller.enabled
    assert run.output_state is OutputState.DETACHED


def test_connected_before_run_is_disarmed_then_one_request_waits_new_state2():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.poll_output(lambda: None)
    assert motor.enables == 0
    run.start()
    for _ in range(4):
        run.poll_output(lambda: None)
    assert motor.enables == 1
    assert run.output_state is OutputState.WAITING_CONFIRMATION
    motor.enabled = True  # bottom layer confirms its newer state=2 frame
    run.poll_output(lambda: None)
    assert run.output_state is OutputState.ENABLED


def test_connect_and_disconnect_while_running_preserves_prediction():
    run, controller, old_output = prepared()
    run.start()
    motor = Output()
    run.attach_motor(motor)
    assert old_output.closed
    run.poll_output(lambda: None)
    assert motor.enables == 1
    replacement = Output()
    run.detach_motor(replacement)
    assert run.running and controller.started == 1
    assert motor.closed and not motor.enabled
    assert controller.output is replacement and not controller.enabled


def test_unmet_gate_never_stops_prediction_or_consumes_enable():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.start()
    def invalid():
        raise ValueError("0 Nm")
    run.poll_output(invalid)
    assert run.running and motor.enables == 0 and "0 Nm" in run.reason


def test_fault_never_retries_or_clears_latch_even_on_new_run():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.start()
    run.poll_output(lambda: None)
    motor.fault = "timeout"
    run.poll_output(lambda: None)
    assert run.output_state is OutputState.FAULT
    motor.fault = None  # even a spurious external clear creates no intent
    for _ in range(4):
        run.poll_output(lambda: None)
    assert motor.enables == 1
    motor.fault = "latched"
    assert run.stop()
    run.release_controller()
    new_controller = Controller(motor)
    run.prepare(new_controller, motor)
    run.start()
    run.poll_output(lambda: None)
    assert motor.fault == "latched" and motor.enables == 1


def test_stop_cancels_intent_and_timeout_retains_controller():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.start()
    controller.exits = False
    assert not run.stop(timeout_ms=1)
    run.poll_output(lambda: None)
    assert run.controller is controller and controller.stop_requested
    assert run.run_state is RunState.STOPPING and motor.enables == 0
    with pytest.raises(RuntimeError):
        run.release_controller()
    controller.exits = True
    assert run.stop()
    assert run.release_controller() is controller


def test_failed_enable_is_consumed_once():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.start()
    original = controller.set_enabled
    def fail(value):
        if value:
            raise ValueError("refused")
        original(value)
    controller.set_enabled = fail
    run.poll_output(lambda: None)
    run.poll_output(lambda: None)
    assert run.running and run.output_state is OutputState.FAULT
    assert "refused" in run.reason and motor.enables == 0


def test_stop_attempts_every_cleanup_and_wait_after_errors():
    run, controller, output = prepared()
    calls = []
    def fail(label):
        def operation(*args, **kwargs):
            calls.append(label)
            raise IOError(label)
        return operation
    controller.request_stop = fail("request")
    output.set_enabled = fail("disable")
    output.zero = fail("zero")
    def wait(timeout_ms):
        calls.append("wait")
        return True
    controller.stop_and_wait = wait
    assert not run.stop()
    assert calls == ["request", "disable", "zero", "wait"]
    assert run.controller is controller and run.run_state is RunState.STOPPING
    assert "disable" in run.reason and "zero" in run.reason


def test_attach_failure_closes_new_motor_and_invalidates_run():
    run, controller, _ = prepared()
    run.start()
    motor = Output()
    def failed_swap(output):
        raise IOError("swap failed")
    controller.set_torque_output = failed_swap
    with pytest.raises(RuntimeError, match="swap failed"):
        run.attach_motor(motor)
    assert motor.closed and not motor.enabled
    assert not run.running and controller.stop_requested


def test_detach_failure_still_disables_and_closes_motor():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.start()
    def failed_swap(output):
        raise IOError("detach failed")
    controller.set_torque_output = failed_swap
    with pytest.raises(RuntimeError, match="detach failed"):
        run.detach_motor(Output())
    assert motor.closed and not motor.enabled and controller.stop_requested
    assert not run.running


def test_lost_enable_cleanup_failure_is_latched_without_timer_exception():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.start()
    run.poll_output(lambda: None)
    motor.enabled = True
    run.poll_output(lambda: None)
    assert run.output_state is OutputState.ENABLED
    motor.enabled = False
    def failed_disable(value):
        raise IOError("controller disable failed")
    controller.set_enabled = failed_disable
    run.poll_output(lambda: None)
    assert run.output_state is OutputState.FAULT
    assert "controller disable failed" in run.reason
    assert motor.commands[-1] == (0, 0)
    for _ in range(3):
        run.poll_output(lambda: None)
    assert motor.enables == 1


def test_reconnect_after_stopped_prediction_requires_a_fresh_run():
    run, controller, _ = prepared()
    motor = Output()
    run.attach_motor(motor)
    run.start()
    run.poll_output(lambda: None)
    motor.enabled = True
    run.poll_output(lambda: None)
    # An input fault stops/releases inference, independently of the COM port.
    assert run.stop()
    assert run.release_controller() is controller
    run.detach_motor(Output())
    replacement = Output()
    run.attach_motor(replacement)
    for _ in range(4):
        run.poll_output(lambda: None)
    assert not run.running and not replacement.enabled and replacement.enables == 0
    assert "预测未运行" in run.reason and "运行预测" in run.reason
    fresh = Controller(replacement)
    run.prepare(fresh, replacement)
    run.start()
    for _ in range(4):
        run.poll_output(lambda: None)
    assert run.running and replacement.enables == 1
    assert run.output_state is OutputState.WAITING_CONFIRMATION
