"""Small model boundary: declared input contract and unit-aware output mapping.

The hub owns raw ring buffers, extractors own causal feature windows, and this
adapter validates the model boundary. No motor policy or UI state lives here.
"""
from __future__ import annotations

import numpy as np
from time import perf_counter_ns

from exo_collection.apps.model_runtime.backends import ModelRuntimeError
from exo_collection.apps.model_runtime.assistance import AssistancePipeline
from exo_collection.apps.model_runtime.features import DefaultWindowFeatureExtractor, FeatureExtractionError
from exo_collection.apps.model_runtime.spec import ModelSpec, OutputSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub


def _output_contract_description(spec: ModelSpec) -> str:
    estimate = ("模型仅估计右髋力矩；不产生左侧预测。" if spec.imu_tcn.enabled else
                "模型原始估计保留声明的左右输出映射。")
    distribution = ("右侧估计完成单位换算、滤波、助力比例、延时及方向处理后，同号同值分配左右电机（Nm）。"
                    if spec.assistance.motor_distribution == "right_to_both_same_sign" else
                    "辅助力矩按各侧估计分别处理并分配对应电机（Nm）。")
    return estimate + distribution


def _validate_motor_distribution(spec: ModelSpec, config=None) -> None:
    config = config or spec.assistance
    if (config.motor_distribution == "right_to_both_same_sign"
            and spec.output.single_output_maps_to == "left_torque"):
        raise ModelRuntimeError("right_to_both_same_sign requires a right-side output, not left_torque")


def map_output_values(output: np.ndarray, spec: OutputSpec) -> tuple[float, float]:
    """Map values in their declared unit; conversion to Nm belongs downstream."""
    if spec.kind != "torque_left_right":
        raise ModelRuntimeError(f"unsupported output kind: {spec.kind!r}")
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ModelRuntimeError("model returned a non-finite output")
    if values.size == 2:
        return float(values[0]), float(values[1])
    if values.size == 1:
        target = spec.single_output_maps_to or "right_torque"
        if target == "left_torque":
            return float(values[0]), 0.0
        if target == "right_torque":
            return 0.0, float(values[0])
        raise ModelRuntimeError(f"unknown single_output_maps_to: {target!r}")
    raise ModelRuntimeError(f"unsupported model output size: {values.size}")


class ModelInputAdapter:
    """Prepare/probe inputs without embedding a particular model in the loop."""

    def __init__(self, hub: SubscriptionHub, spec: ModelSpec, extractor, *, clock_ns=None) -> None:
        self.hub, self.spec, self.extractor = hub, spec, extractor
        self._clock_ns = clock_ns or perf_counter_ns
        self.last_reason: str | None = None
        self.last_input_expired = False
        self._source_times: dict[str, int] = {}

    def _freshness_errors(self, source_times: dict[str, int] | None = None) -> list[str]:
        # IMU freshness remains the extractor's stricter 50 ms contract. The
        # general model rule is the already-declared safety watchdog, not a
        # second hardcoded IMU-derived timing threshold.
        if self.spec.imu_tcn.enabled:
            return []
        now = int(self._clock_ns())
        limit_ns = int(self.spec.safety.watchdog_s * 1e9)
        errors = []
        for entry in self.spec.features.inputs:
            latest = self.hub.latest(entry.modality)
            time_ns = (source_times.get(entry.modality) if source_times is not None else
                       None if latest is None else int(latest[0]))
            if time_ns is not None and now - time_ns > limit_ns:
                errors.append(f"模态 {entry.modality} 输入已过期（超过 watchdog {self.spec.safety.watchdog_s:g} s）")
        return errors

    def result_expired_reason(self) -> str | None:
        errors = self._freshness_errors(self._source_times)
        return "; ".join(errors) if errors else None

    def contract(self) -> dict[str, object]:
        spec = self.spec
        if spec.imu_tcn.enabled:
            channels = [f"{device}:{axis}" for device in ("right", "pelvis")
                        for axis in ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")]
        else:
            channels = []
            for entry in spec.features.inputs:
                names = entry.channels
                if names is None:
                    try:
                        names = list(self.hub.flat_channels(entry.modality))
                    except KeyError:
                        names = ["<all descriptor channels>"]
                channels.extend(f"{entry.modality}:{name}" for name in names)
        return {
            "modalities": [entry.modality for entry in spec.features.inputs],
            "channels": channels,
            "window_samples": spec.feature_window_samples,
            "control_rate_hz": spec.features.control_rate_hz,
            "input_layout": spec.model.input_layout.upper(),
            "normalization": "embedded_scaler" if spec.imu_tcn.enabled else (
                "per_channel" if spec.features.normalization.mean or spec.features.normalization.std else "none"),
            "output_unit": spec.output.unit,
            "output_mapping": "single->" + (spec.output.single_output_maps_to or "right_torque") + "; pair->left,right",
            "motor_distribution": spec.assistance.motor_distribution,
            "contract_description": _output_contract_description(spec),
            "max_sample_age_s": spec.imu_tcn.max_sample_age_s if spec.imu_tcn.enabled else spec.safety.watchdog_s,
        }

    def preflight(self) -> dict[str, object]:
        probe = getattr(self.extractor, "preflight", None)
        try:
            # Third-party extractors can provide stricter checks. Without a
            # probe, inspect the standard declared source contract, never claim
            # ready simply because this is not the deployed IMU model.
            report = dict(probe(self.hub, self.spec.features) if callable(probe) else
                          DefaultWindowFeatureExtractor().preflight(self.hub, self.spec.features))
        except (FeatureExtractionError, KeyError, ValueError) as exc:
            report = {"ready": False, "errors": [str(exc)]}
        errors = list(report.get("errors", [])) + self._freshness_errors()
        report.update({"ready": bool(report.get("ready")) and not errors, "errors": errors})
        contract = self.contract()
        buffers = {}
        now_ns = int(self._clock_ns())
        for modality in contract["modalities"]:
            latest = self.hub.latest(modality)
            buffers[modality] = {"samples": self.hub.count(modality),
                                 "latest_ns": None if latest is None else int(latest[0]),
                                 "age_ms": None if latest is None else max(0, (now_ns-int(latest[0]))/1e6)}
        report.update({
            "mode": "imu_tcn" if self.spec.imu_tcn.enabled else "generic",
            "contract": contract,
            "contract_description": contract["contract_description"],
            "contract_summary": (f"{','.join(contract['modalities'])} | "
                                 f"{contract['window_samples']} 帧 × {len(contract['channels'])} 通道 | "
                                 f"{contract['control_rate_hz']:g} Hz | {contract['input_layout']} | "
                                 f"{contract['normalization']} | {contract['output_unit']} | "
                                 f"源年龄≤{contract['max_sample_age_s']:g} s"),
            "buffer_status": buffers,
        })
        if self.spec.imu_tcn.enabled:
            report.update({"right_device_id": self.spec.imu_tcn.right_device_id,
                           "pelvis_device_id": self.spec.imu_tcn.pelvis_device_id,
                           "alignment_mode": "none_independent_streams"})
        return report

    def extract(self) -> np.ndarray:
        self.last_reason = None
        self.last_input_expired = False
        errors = self._freshness_errors()
        if errors:
            self.last_reason = "; ".join(errors)
            self.last_input_expired = True
            return np.empty((0, 0), dtype=np.float32)
        source_times = {}
        for entry in self.spec.features.inputs:
            latest = self.hub.latest(entry.modality)
            if latest is not None:
                source_times[entry.modality] = int(latest[0])
        tensor = np.asarray(self.extractor.extract(self.hub, self.spec.features), dtype=np.float32)
        if tensor.size == 0:
            self.last_reason = getattr(self.extractor, "last_reason", None)
            return tensor
        channels = len(self.contract()["channels"])
        expected = (channels, self.spec.feature_window_samples)
        if tensor.shape != expected or not np.isfinite(tensor).all():
            raise FeatureExtractionError(f"model input must be finite [C,T]={expected}; got {tensor.shape}")
        self._source_times = source_times
        return tensor

class ModelOutputAdapter:
    """Keep native estimates distinct from the explicitly configured motor routing."""

    def __init__(self, spec: ModelSpec) -> None:
        _validate_motor_distribution(spec)
        self.spec = spec
        self.right = AssistancePipeline(spec.assistance)
        self.left = (None if spec.assistance.motor_distribution == "right_to_both_same_sign" else
                     AssistancePipeline(spec.assistance.model_copy(update={"right_sign": 1})))

    def configure(self, config) -> None:
        _validate_motor_distribution(self.spec, config)
        self.spec.assistance = config
        self.right.configure(config)
        self.left = (None if config.motor_distribution == "right_to_both_same_sign" else
                     AssistancePipeline(config.model_copy(update={"right_sign": 1})))

    def reset(self) -> None:
        self.right.reset()
        if self.left is not None:
            self.left.reset()

    def preflight(self) -> dict[str, object]:
        return self.right.preflight(self.spec.output.unit)

    def map_output(self, output: np.ndarray) -> tuple[float, float]:
        _validate_motor_distribution(self.spec)
        values = np.asarray(output).reshape(-1)
        if self.spec.imu_tcn.enabled and values.size != 1:
            raise ModelRuntimeError("IMU TCN must return one right hip estimate")
        return map_output_values(output, self.spec.output)

    def process(self, time_ns: int, left: float, right: float) -> dict[str, object]:
        _validate_motor_distribution(self.spec)
        unit = self.spec.output.unit
        right_sample = self.right.process(time_ns, right, unit=unit)
        imu = self.spec.imu_tcn.enabled
        distribution = self.spec.assistance.motor_distribution
        if distribution == "right_to_both_same_sign":
            # Copy the final Nm assistance exactly once. There is no second
            # filter/delay state and no second body-mass or sign multiplication.
            desired_left = right_sample.desired_right_nm
            ready = right_sample.assistance_ready
            estimate_left = (None if not ready else
                             left * self.spec.assistance.mass_kg if unit == "Nm/kg" else left)
        else:
            left_sample = self.left.process(time_ns, left, unit=unit)
            desired_left = 0.0 if imu else left_sample.desired_right_nm
            estimate_left = left_sample.estimate_nm
            ready = right_sample.assistance_ready and left_sample.assistance_ready
        return {
            "time_ns": int(time_ns), "unit": unit, "output_unit": unit,
            "raw_estimate_left": 0.0 if imu else left, "raw_estimate_right": right,
            "estimate_nm_per_kg": right_sample.estimate_nm_per_kg,
            "estimate_left_nm": 0.0 if imu else estimate_left,
            "estimate_nm": right_sample.estimate_nm,
            "desired_left_nm": desired_left,
            "desired_right_nm": right_sample.desired_right_nm,
            "assistance_ready": ready,
            "assistance_errors": list(right_sample.assistance_errors),
            "delayed_source_time_ns": right_sample.delayed_source_time_ns,
            "motor_distribution": distribution,
            "contract_description": _output_contract_description(self.spec),
        }


__all__ = ["ModelInputAdapter", "ModelOutputAdapter", "map_output_values"]
