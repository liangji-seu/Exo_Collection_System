"""Compatibility checks shared by the gate UI and inference preflight."""

from __future__ import annotations

from exo_collection.apps.model_runtime.spec import ModelSpec


def gate_compatibility_error(
    spec: ModelSpec,
    mode: str,
    *,
    has_imu_tcn_extractor: bool | None = None,
) -> str | None:
    """Return an actionable reason when a selected model cannot drive a gate."""
    if mode == "none":
        return None
    if (abs(float(spec.features.control_rate_hz) - 100.0) > 1e-6
            or int(spec.feature_window_samples) != 94):
        return "助力门控要求 100 Hz IMU 网格；请先选择 100 Hz / 94 点模型，或将门控设为不门控。"
    if has_imu_tcn_extractor is None:
        has_imu_tcn_extractor = bool(spec.imu_tcn.enabled)
    if not has_imu_tcn_extractor:
        return "助力门控要求右腿 + 骨盆 IMU 的 IMU-TCN 特征提取器；请先选择 IMU-TCN 模型，或将门控设为不门控。"
    return None


__all__ = ["gate_compatibility_error"]
