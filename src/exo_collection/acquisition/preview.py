"""Pure conversion of raw modality batches into bounded UI preview events."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.domain.events import FrameBatch, SampleBatch


def build_preview_event(
    event: FrameBatch | SampleBatch,
    trial_uuid: UUID | str | None = None,
    *,
    extra_payload: dict[str, Any] | None = None,
) -> WorkerEvent:
    """Return the established, JSON-safe preview payload for one raw batch.

    The function is intentionally independent of Writers, Catalog and package
    layout so it is safe to use in both a recording worker and a preview-only
    device process.
    """

    values = np.asarray(event.data)
    if isinstance(event, FrameBatch):
        if values.ndim < 2 or values.shape[0] < 1:
            raise ValueError(f"invalid ultrasound frame batch shape: {values.shape}")
        source_frame = values[-1]
        frame = source_frame.astype(np.float32, copy=False)

        def downsample(signal: np.ndarray) -> np.ndarray:
            flattened = signal.reshape(-1)
            if flattened.size <= 512:
                return flattened
            indices = np.linspace(0, flattened.size - 1, 512, dtype=np.int64)
            return flattened[indices]

        is_multichannel_a_line = frame.ndim == 2 and 1 <= frame.shape[0] <= 16
        channels = (
            [downsample(frame[index]) for index in range(frame.shape[0])]
            if is_multichannel_a_line
            else [downsample(frame)]
        )
        source_channels = (
            [source_frame[index].reshape(-1) for index in range(source_frame.shape[0])]
            if is_multichannel_a_line
            else [source_frame.reshape(-1)]
        )

        def format_metrics(signal: np.ndarray) -> dict[str, Any]:
            count = max(1, int(signal.size))
            if np.issubdtype(signal.dtype, np.floating):
                finite = np.isfinite(signal)
                nonfinite_fraction = 1.0 - float(np.count_nonzero(finite)) / count
                finite_signal = signal[finite]
            else:
                nonfinite_fraction = 0.0
                finite_signal = signal
            zero_fraction = (
                float(np.count_nonzero(finite_signal == 0)) / count
                if finite_signal.size
                else 0.0
            )
            full_scale_fraction: float | None = None
            full_scale_value: int | float | None = None
            if np.issubdtype(signal.dtype, np.integer):
                full_scale_value = int(np.iinfo(signal.dtype).max)
                full_scale_fraction = (
                    float(np.count_nonzero(signal == full_scale_value)) / count
                )
            return {
                "dtype": str(signal.dtype),
                "zero_fraction": zero_fraction,
                "nonfinite_fraction": nonfinite_fraction,
                "full_scale_fraction": full_scale_fraction,
                "full_scale_value": full_scale_value,
                "all_zero": bool(signal.size and np.all(signal == 0)),
            }

        payload: dict[str, Any] = {
            "host_monotonic_ns": event.host_monotonic_ns,
            "values": channels[0].tolist(),
            "channels": [channel.tolist() for channel in channels],
            "channel_count": len(channels),
            "shape": [int(value) for value in frame.shape],
            "preview_sample_count": int(channels[0].size),
            "geometry": "a_line" if is_multichannel_a_line else "frame",
            "format_metrics": [format_metrics(channel) for channel in source_channels],
        }
    else:
        if values.ndim < 2 or values.shape[0] < 1:
            raise ValueError(f"invalid sample batch shape: {values.shape}")
        if event.modality == "imu":
            if values.ndim != 3:
                raise ValueError(f"invalid IMU batch shape: {values.shape}")
            labels = ("imu_trunk", "imu_left", "imu_right")
            channels = [
                values[:, device_index, 0].astype(float).tolist()
                for device_index in range(min(values.shape[1], len(labels)))
            ]
            payload = {
                "host_monotonic_ns": event.host_monotonic_ns,
                "values": channels[0] if channels else [],
                "channels": channels,
                "labels": list(labels[: len(channels)]),
                "channel": "acc_x",
                "channel_count": len(channels),
            }
        elif event.modality == "encoder":
            if values.ndim != 2 or values.shape[1] < 4:
                raise ValueError(f"invalid encoder batch shape: {values.shape}")
            labels = ("left_position", "right_position")
            channels = [
                values[:, 0].astype(float).tolist(),
                values[:, 3].astype(float).tolist(),
            ]
            payload = {
                "host_monotonic_ns": event.host_monotonic_ns,
                "values": channels[0],
                "channels": channels,
                "labels": list(labels),
                "channel": "position",
                "channel_count": len(channels),
            }
        else:
            signal = values[:, 0]
            rate = event.sample_rate_hz or 1.0
            x = (event.first_sample_index + np.arange(signal.size)) / rate
            payload = {
                "host_monotonic_ns": event.host_monotonic_ns,
                "x": x.astype(float).tolist(),
                "values": signal.astype(float).tolist(),
                "channel": "voltage",
            }
    if extra_payload:
        payload.update(extra_payload)
    return WorkerEvent(
        event_type=WorkerEventType.PREVIEW,
        trial_uuid=None if trial_uuid is None else str(trial_uuid),
        modality=event.modality,
        payload=payload,
    )


__all__ = ["build_preview_event"]
