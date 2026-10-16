"""Bind model roles from configured device slots, never discovery order."""
from __future__ import annotations

from collections.abc import Mapping

from .spec import ModelSpec


def bind_model_inputs(spec: ModelSpec, *, profile_key: str, imu_parameters,
                      descriptor=None) -> ModelSpec:
    values = spec.model_dump()
    if not spec.imu_tcn.enabled:
        return ModelSpec.model_validate(values)
    parameters = (imu_parameters.model_dump() if hasattr(imu_parameters, "model_dump")
                  else dict(imu_parameters or {}))
    key = "sensor_ids" if profile_key == "hardware" else "device_ids"
    slots = parameters.get(key)
    binding = values["imu_tcn"]
    if slots:
        if not isinstance(slots, (tuple, list)) or len(slots) != 3:
            raise ValueError("IMU 配置需要左腿、右腿、骨盆三个位置槽。")
        ids = [str(value).strip() if value is not None else "" for value in slots]
        active = [value for value in ids if value]
        if len(active) != len(set(active)):
            raise ValueError("IMU 位置槽中的设备 ID 不可重复。")
        if not ids[1] or not ids[2]:
            raise ValueError("请在 IMU 设备设置填写右腿和骨盆 ID；左腿可留空。")
        binding["right_device_id"], binding["pelvis_device_id"] = ids[1:]
    required = [str(binding.get(name) or "").strip()
                for name in ("right_device_id", "pelvis_device_id")]
    if not all(required) or required[0] == required[1]:
        raise ValueError("模型需要不同的右腿与骨盆 IMU ID；请配置 IMU 设备位置槽。")
    binding["right_device_id"], binding["pelvis_device_id"] = required
    if isinstance(descriptor, Mapping):
        known = descriptor.get("metadata", {}).get("device_ids")
        if known and not set(required).issubset({str(value) for value in known}):
            raise ValueError("已连接 IMU 不包含配置的右腿或骨盆设备；请检查设备设置。")
    return ModelSpec.model_validate(values)
