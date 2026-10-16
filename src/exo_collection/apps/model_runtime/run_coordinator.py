"""Application lifecycle and motor routing, independent of Qt widgets.

InferenceController owns inference and AssistancePipeline processing; the
serial service alone owns wire limits, feedback confirmation and fault latches.
This coordinator owns explicit operator intents and the attachment lifetime.
"""
from __future__ import annotations

from enum import Enum
from time import perf_counter_ns
from typing import Callable


class RunState(Enum):
    EMPTY = "empty"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class OutputState(Enum):
    DETACHED = "detached"
    DISARMED = "disarmed"
    ARM_INTENT = "arm_intent"
    WAITING_CONFIRMATION = "waiting_confirmation"
    ENABLED = "enabled"
    FAULT = "fault"


def _motor_fault_reason(motor) -> str:
    """Describe controller faults without conflating USB and CAN freshness."""
    fault = str(getattr(motor, "fault", "底层故障") or "底层故障")
    snapshot = getattr(motor, "fault_snapshot", None)
    try:
        snapshot = snapshot() if callable(snapshot) else snapshot
    except Exception:
        snapshot = None
    code = None
    if isinstance(snapshot, dict):
        code = snapshot.get("error")
        if code is None and isinstance(snapshot.get("feedback"), dict):
            code = snapshot["feedback"].get("error")
    if code == 253 or "253" in fault:
        return ("底层故障锁定：固件报告 CAN 反馈缺失（error=253）；"
                "主机 USB 状态帧年龄与 CAN 新鲜度分开记录，已保持失能。")
    if code == 254 or "254" in fault:
        return ("底层故障锁定：固件报告 PC 指令超时（error=254），"
                "已保持失能。")
    return f"底层故障锁定：{fault}；已保持失能。"


class RunCoordinator:
    def __init__(self, *, clock=perf_counter_ns):
        self.controller = None
        self.output = None
        self.motor = None
        self.run_state = RunState.EMPTY
        self.output_state = OutputState.DETACHED
        self.reason = "电机未连接 · 未下发"
        self._clock = clock

    @property
    def running(self):
        return self.run_state is RunState.RUNNING

    def prepare(self, controller, output):
        if self.controller is not None:
            raise RuntimeError("previous controller must be stopped before replacement")
        controller.set_enabled(False)
        self.controller, self.output = controller, output
        self.run_state = RunState.READY
        self._disarmed_state()

    def start(self):
        if self.running:
            return
        if self.run_state is not RunState.READY or self.controller is None:
            raise RuntimeError("prepare a fresh controller before Run")
        self.controller.set_enabled(False)
        self.controller.start()
        self.run_state = RunState.RUNNING
        self._new_arm_intent()

    @staticmethod
    def _attempt(errors, label, operation):
        try:
            return operation()
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            return None

    def _shutdown_output(self, output, errors, *, close=False):
        if output is None:
            return
        self._attempt(errors, "disable", lambda: output.set_enabled(False))
        self._attempt(errors, "zero", output.zero)
        if close:
            self._attempt(errors, "close", output.close)

    def stop(self, *, timeout_ms=3000):
        # Ineligible for callbacks before waiting, even if a blocked predict
        # delays exit. Every shutdown operation is attempted independently.
        self.run_state = RunState.STOPPING
        self._disarmed_state()
        errors = []
        controller = self.controller
        if controller is not None:
            self._attempt(errors, "request_stop", controller.request_stop)
        self._shutdown_output(self.output, errors)
        if self.motor is not self.output:
            self._shutdown_output(self.motor, errors)
        exited = True
        if controller is not None:
            exited = self._attempt(
                errors, "stop_and_wait",
                lambda: controller.stop_and_wait(timeout_ms=timeout_ms),
            )
            if exited is None:
                # request_stop may raise after invalidating the generation;
                # QThread.wait must still be attempted in that case.
                wait = getattr(controller, "wait", None)
                is_running = getattr(controller, "isRunning", None)
                if callable(wait) and callable(is_running):
                    exited = self._attempt(
                        errors, "wait", lambda: not is_running() or wait(timeout_ms),
                    )
        if not exited or errors:
            details = "; ".join(errors) if errors else "推理线程未退出"
            self.reason = f"停止未完成：{details}；保留控制器，禁止重载或关闭窗口。"
            return False
        self.run_state = RunState.STOPPED
        return True

    def release_controller(self):
        if self.run_state is RunState.STOPPING:
            raise RuntimeError(self.reason)
        controller = self.controller
        self.controller = None
        if self.output is not None and self.output is not self.motor:
            self.output.close()
        self.output = None
        return controller

    def attach_motor(self, motor):
        old_output = self.output
        try:
            motor.set_enabled(False)
            if self.controller is not None:
                recover = getattr(self.controller, "set_input_wait_recovery", None)
                if callable(recover):
                    recover(False)
                self.controller.set_torque_output(motor)
                self.output = motor
        except Exception as exc:
            # A sink swap can fail after partially attaching. Invalidate the
            # prediction session and close the newly opened serial resource.
            self.run_state = RunState.STOPPING
            errors = [f"attach: {exc}"]
            if self.controller is not None:
                self._attempt(errors, "request_stop", self.controller.request_stop)
            self._shutdown_output(motor, errors, close=True)
            self.reason = "; ".join(errors)
            raise RuntimeError(self.reason) from exc
        self.motor = motor
        if old_output is not None and old_output is not motor:
            try:
                old_output.close()
            except Exception as exc:
                # Closing an already detached display sink cannot arm hardware.
                self.reason = f"旧显示输出关闭失败：{exc}"
        self._new_arm_intent()

    def detach_motor(self, display_output):
        motor = self.motor
        if motor is None:
            return
        errors = []
        if self.controller is not None:
            try:
                self.controller.set_torque_output(display_output)
                self.output = display_output
            except Exception as exc:
                errors.append(f"detach: {exc}")
                self.run_state = RunState.STOPPING
                self._attempt(errors, "request_stop", self.controller.request_stop)
        elif display_output is not None:
            self._attempt(errors, "close display", display_output.close)
        self._shutdown_output(motor, errors, close=True)
        self.motor = None
        self.output_state = OutputState.DETACHED
        self.reason = "电机未连接 · 未下发"
        if errors:
            self.reason = "断开电机存在错误：" + "; ".join(errors)
            raise RuntimeError(self.reason)
        if self.controller is not None:
            recover = getattr(self.controller, "set_input_wait_recovery", None)
            if callable(recover):
                recover(True)

    def _disarmed_state(self):
        if self.motor is None:
            self.output_state = OutputState.DETACHED
            self.reason = "电机未连接 · 未下发"
        elif self.motor.fault:
            self.output_state = OutputState.FAULT
            self.reason = _motor_fault_reason(self.motor)
        else:
            self.output_state = OutputState.DISARMED
            self.reason = ("电机已连接 · 保持失能" if self.running else
                           "电机已连接 · 预测未运行；点击“运行预测”重新校验并启动")

    def _new_arm_intent(self):
        self._disarmed_state()
        if self.running and self.motor is not None and not self.motor.fault:
            self.output_state = OutputState.ARM_INTENT
            self.reason = "等待安全条件与新鲜反馈"

    def poll_output(self, validate: Callable[[], None]):
        """Consume at most one arm request per explicit Run/Connect edge.

        validate raises a readable reason for static real-input/direction/limit
        gates. No polling path clears a motor fault or creates a new intent.
        """
        motor, controller = self.motor, self.controller
        if motor is None or controller is None or not self.running:
            return
        if motor.fault:
            self.output_state = OutputState.FAULT
            self.reason = _motor_fault_reason(motor)
            return
        if self.output_state is OutputState.WAITING_CONFIRMATION:
            if motor.enabled:
                self.output_state = OutputState.ENABLED
                self.reason = "底层已确认使能"
            return
        if self.output_state is OutputState.ENABLED:
            if not motor.enabled:
                self.output_state = OutputState.FAULT
                errors = []
                self._attempt(errors, "disable controller", lambda: controller.set_enabled(False))
                self._shutdown_output(motor, errors)
                self.reason = "电机已失能；须重新运行或手动连接"
                if errors:
                    self.reason += "；" + "; ".join(errors)
            return
        if self.output_state is not OutputState.ARM_INTENT:
            return
        try:
            validate()
            feedback = motor.feedback
            if not motor.connected:
                raise ValueError("电机连接已断开")
            if feedback is None:
                raise ValueError("等待电机反馈")
            age_ns = self._clock() - feedback.host_monotonic_ns
            if not 0 <= age_ns <= getattr(motor, "feedback_timeout_ns", 250_000_000):
                raise ValueError("电机反馈过期")
            if feedback.error or feedback.state not in (0, 2):
                raise ValueError("电机反馈状态不允许使能")
        except Exception as exc:
            self.reason = f"保持失能：{exc}"
            return
        # Consume intent BEFORE crossing an external boundary. Failure cannot
        # become an automatic enable retry on the next timer invocation.
        self.output_state = OutputState.WAITING_CONFIRMATION
        self.reason = "已发出一次使能请求 · 等待底层新 state=2 确认"
        try:
            motor.send(0.0, 0.0)
            controller.set_enabled(True)
        except Exception as exc:
            errors = [str(exc)]
            self._attempt(errors, "disable controller", lambda: controller.set_enabled(False))
            self._shutdown_output(motor, errors)
            self.output_state = OutputState.FAULT
            self.reason = "使能请求失败：" + "; ".join(errors)
