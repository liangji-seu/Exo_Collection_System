"""Small dependency-free PNG quality previews from bounded acquisition history.

The history object is deliberately cheap to update: ultrasound is spatially
downsampled and retained in a fixed-length deque, while IMU/encoder curves are
kept in fixed-size point buffers.  PNG compression and plotting happen only
after the acquisition adapters and Writers have stopped.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np

from exo_collection.domain.events import FrameBatch, SampleBatch
from exo_collection.storage.layout import TrialLayout


US_PREVIEW_PATH = "reports/us_quality_preview.png"
SIGNAL_PREVIEW_PATH = "reports/imu_encoder_preview.png"
US_PREVIEW_SIZE = (1000, 720)
SIGNAL_PREVIEW_SIZE = (1000, 600)


@dataclass(frozen=True, slots=True)
class PreviewReportBundle:
    ultrasound_metadata: dict[str, Any]
    signal_metadata: dict[str, Any]
    soft_metrics: dict[str, Any]


class _BoundedSeries:
    def __init__(self, names: tuple[str, ...], max_points: int) -> None:
        self.names = names
        self.max_points = max_points
        self._segments: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._point_count = 0

    def append(self, x: np.ndarray, values: np.ndarray) -> None:
        x_array = np.asarray(x, dtype=np.float64).reshape(-1)
        value_array = np.asarray(values, dtype=np.float32)
        if value_array.ndim == 1:
            value_array = value_array[:, None]
        if (
            x_array.size == 0
            or value_array.shape != (x_array.size, len(self.names))
        ):
            return
        # One malformed or extremely large hardware batch must not defeat the
        # memory bound.  Evenly-spaced retention is sufficient for a QC plot.
        if x_array.size > self.max_points:
            selected = np.linspace(
                0, x_array.size - 1, self.max_points, dtype=np.int64
            )
            x_array = x_array[selected]
            value_array = value_array[selected]
        self._segments.append((x_array.copy(), value_array.copy()))
        self._point_count += x_array.size
        while self._point_count > self.max_points and self._segments:
            old_x, old_values = self._segments[0]
            excess = self._point_count - self.max_points
            if excess < old_x.size:
                self._segments[0] = (old_x[excess:], old_values[excess:])
                self._point_count -= excess
                break
            self._segments.popleft()
            self._point_count -= old_x.size

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._segments:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, len(self.names)), dtype=np.float32),
            )
        return (
            np.concatenate([item[0] for item in self._segments]),
            np.concatenate([item[1] for item in self._segments], axis=0),
        )


class BoundedPreviewHistory:
    """Retain only enough downsampled data for deterministic quality previews."""

    def __init__(
        self,
        *,
        max_ultrasound_frames: int = 160,
        ultrasound_depth_samples: int = 256,
        max_signal_points: int = 4000,
    ) -> None:
        if max_ultrasound_frames <= 0 or ultrasound_depth_samples <= 1:
            raise ValueError("ultrasound history bounds must be positive")
        if max_signal_points <= 1:
            raise ValueError("signal history bound must exceed one")
        self.max_ultrasound_frames = max_ultrasound_frames
        self.ultrasound_depth_samples = ultrasound_depth_samples
        self.max_signal_points = max_signal_points
        self._ultrasound: deque[tuple[int, np.ndarray]] = deque(
            maxlen=max_ultrasound_frames
        )
        self._raw_ultrasound: tuple[deque[tuple[int, np.ndarray]], ...] = tuple(
            deque(maxlen=max_ultrasound_frames) for _ in range(4)
        )
        self._ultrasound_capture_mode: str | None = None
        self._ultrasound_shape: tuple[int, int] | None = None
        self._ultrasound_dtype_min: float | None = None
        self._ultrasound_dtype_max: float | None = None
        self._ultrasound_frames_seen = 0
        self._imu = _BoundedSeries(("roll_deg", "pitch_deg"), max_signal_points)
        self._encoder = _BoundedSeries(
            ("left_position_rad", "right_position_rad"), max_signal_points
        )

    def capture(self, event: FrameBatch | SampleBatch) -> None:
        if isinstance(event, FrameBatch) and event.modality == "ultrasound":
            self._capture_ultrasound(event)
        elif isinstance(event, SampleBatch) and event.modality == "imu":
            self._capture_imu(event)
        elif isinstance(event, SampleBatch) and event.modality == "encoder":
            self._capture_encoder(event)

    def _capture_ultrasound(self, event: FrameBatch) -> None:
        batch = np.asarray(event.data)
        if batch.ndim < 2 or batch.shape[0] == 0:
            return
        if np.issubdtype(batch.dtype, np.integer):
            limits = np.iinfo(batch.dtype)
            self._ultrasound_dtype_min = float(limits.min)
            self._ultrasound_dtype_max = float(limits.max)
        if event.channel is not None:
            self._capture_raw_ultrasound_packet(event, batch)
            return
        if self._ultrasound_capture_mode not in (None, "device_synchronized_frames"):
            # A Trial must not combine packet-ordinal previews with real
            # device-synchronised frames. Raw data remains authoritative.
            return
        self._ultrasound_capture_mode = "device_synchronized_frames"
        for frame in batch:
            is_multichannel_a_line = (
                frame.ndim == 2 and 1 <= frame.shape[0] <= 16
            )
            channels = frame if is_multichannel_a_line else frame.reshape(1, -1)
            depth_indices = np.linspace(
                0,
                channels.shape[-1] - 1,
                min(self.ultrasound_depth_samples, channels.shape[-1]),
                dtype=np.int64,
            )
            reduced = np.asarray(channels[:, depth_indices], dtype=np.float32)
            shape = (int(reduced.shape[0]), int(reduced.shape[1]))
            if self._ultrasound_shape is None:
                self._ultrasound_shape = shape
            if shape != self._ultrasound_shape:
                # Device geometry changes mid-Trial are not combined into a
                # misleading image.  Raw data remains complete and auditable.
                continue
            self._ultrasound.append((event.host_monotonic_ns, reduced.copy()))
            self._ultrasound_frames_seen += 1

    def _capture_raw_ultrasound_packet(
        self, event: FrameBatch, batch: np.ndarray
    ) -> None:
        """Retain one Raw Ethernet A-line without inventing channel synchrony."""

        if self._ultrasound_capture_mode not in (None, "independent_channel_packets"):
            return
        channel = int(event.channel) if event.channel is not None else -1
        if channel not in range(4) or batch.ndim != 2 or batch.shape[0] != 1:
            return
        samples = batch[0]
        if (
            samples.size >= 3
            and int(samples[0]) == 0x00
            and int(samples[1]) == channel + 1
            and int(samples[-1]) == 0xFF
        ):
            samples = samples[2:-1]
        depth_indices = np.linspace(
            0,
            samples.shape[0] - 1,
            min(self.ultrasound_depth_samples, samples.shape[0]),
            dtype=np.int64,
        )
        reduced = np.asarray(samples[depth_indices], dtype=np.float32)
        shape = (4, int(reduced.shape[0]))
        if self._ultrasound_shape is None:
            self._ultrasound_shape = shape
        if shape != self._ultrasound_shape:
            return
        self._ultrasound_capture_mode = "independent_channel_packets"
        self._raw_ultrasound[channel].append(
            (event.host_monotonic_ns, reduced.copy())
        )
        self._ultrasound_frames_seen += 1

    @staticmethod
    def _sample_x(event: SampleBatch) -> np.ndarray:
        rate = float(event.sample_rate_hz or 1.0)
        return (
            event.first_sample_index + np.arange(event.sample_count, dtype=np.float64)
        ) / rate

    def _capture_imu(self, event: SampleBatch) -> None:
        values = np.asarray(event.data)
        if values.ndim != 3 or values.shape[0] == 0:
            return
        if values.shape[2] >= 11:
            selected = values[:, 0, (9, 10)]
        elif values.shape[2] >= 2:
            selected = values[:, 0, :2]
        else:
            selected = np.column_stack((values[:, 0, 0], values[:, 0, 0]))
        self._imu.append(self._sample_x(event), selected)

    def _capture_encoder(self, event: SampleBatch) -> None:
        values = np.asarray(event.data)
        if values.ndim != 2 or values.shape[0] == 0:
            return
        if values.shape[1] >= 4:
            selected = values[:, (0, 3)]
        elif values.shape[1] >= 2:
            selected = values[:, :2]
        else:
            selected = np.column_stack((values[:, 0], values[:, 0]))
        self._encoder.append(self._sample_x(event), selected)

    def ultrasound_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if self._ultrasound_capture_mode == "independent_channel_packets":
            retained = min(len(channel) for channel in self._raw_ultrasound)
            shape = self._ultrasound_shape or (
                4,
                self.ultrasound_depth_samples,
            )
            if retained == 0:
                # A partial set of channels is not padded with synthetic zero
                # data. It therefore cannot look like a complete acquisition.
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty((0, *shape), dtype=np.float32),
                )
            channel_tails = [
                list(channel)[-retained:] for channel in self._raw_ultrasound
            ]
            # Rows pair the nth retained arrival of each independent channel
            # only for a bounded QC image. They are not hardware-synchronised
            # frames; this is stated explicitly in the exported metrics.
            timestamps = np.asarray(
                [
                    max(channel_tails[channel][row][0] for channel in range(4))
                    for row in range(retained)
                ],
                dtype=np.int64,
            )
            frames = np.stack(
                [
                    np.stack(
                        [channel_tails[channel][row][1] for channel in range(4)]
                    )
                    for row in range(retained)
                ]
            )
            return timestamps, frames
        if not self._ultrasound:
            shape = self._ultrasound_shape or (1, self.ultrasound_depth_samples)
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, *shape), dtype=np.float32),
            )
        timestamps = np.asarray([item[0] for item in self._ultrasound], dtype=np.int64)
        frames = np.stack([item[1] for item in self._ultrasound])
        return timestamps, frames

    def ultrasound_soft_metrics(
        self, *, formal_t0_host_monotonic_ns: int | None = None
    ) -> dict[str, Any]:
        timestamps, frames = self.ultrasound_snapshot()
        includes_pretrigger = bool(
            timestamps.size
            and formal_t0_host_monotonic_ns is not None
            and int(timestamps.min()) < formal_t0_host_monotonic_ns
        )
        metrics: dict[str, Any] = {
            "metric_type": "soft_uncalibrated_preview_metrics",
            "source": "bounded_spatially_downsampled_acquisition_history",
            "hard_thresholds_applied": False,
            "frames_seen": self._ultrasound_frames_seen,
            "frames_retained": int(frames.shape[0]),
            "retention_capacity_frames": self.max_ultrasound_frames,
            "includes_pretrigger": includes_pretrigger,
        }
        if self._ultrasound_capture_mode == "independent_channel_packets":
            retained_per_channel = [
                len(channel) for channel in self._raw_ultrasound
            ]
            metrics.update(
                {
                    "alignment_semantics": (
                        "independent_channel_arrival_ordinal_for_qc_preview_only"
                    ),
                    "device_synchronized_frames": False,
                    "timestamp_semantics": (
                        "maximum_host_arrival_timestamp_per_qc_ordinal_row"
                    ),
                    "acquisition_unit": "raw_ethernet_channel_packet",
                    "expected_channel_count": 4,
                    "packets_retained_per_channel": retained_per_channel,
                }
            )
            all_raw_timestamps = [
                timestamp
                for channel in self._raw_ultrasound
                for timestamp, _ in channel
            ]
            metrics["includes_pretrigger"] = bool(
                all_raw_timestamps
                and formal_t0_host_monotonic_ns is not None
                and min(all_raw_timestamps) < formal_t0_host_monotonic_ns
            )
        else:
            metrics.update(
                {
                    "alignment_semantics": "device_synchronized_multichannel_frame",
                    "device_synchronized_frames": True,
                    "timestamp_semantics": "frame_batch_host_monotonic_timestamp",
                    "acquisition_unit": "multichannel_frame",
                }
            )
        if frames.size == 0:
            metrics.update(
                {
                    "channel_count": 0,
                    "depth_sample_count": 0,
                    "mean_intensity": None,
                    "standard_deviation": None,
                    "zero_fraction": None,
                    "saturation_fraction": None,
                    "channels": [],
                }
            )
            return metrics

        metrics["channel_count"] = int(frames.shape[1])
        metrics["depth_sample_count"] = int(frames.shape[2])
        metrics["mean_intensity"] = float(np.mean(frames, dtype=np.float64))
        metrics["standard_deviation"] = float(np.std(frames, dtype=np.float64))
        metrics["zero_fraction"] = float(np.mean(frames == 0))
        if (
            self._ultrasound_dtype_min is not None
            and self._ultrasound_dtype_max is not None
        ):
            saturated = (frames <= self._ultrasound_dtype_min) | (
                frames >= self._ultrasound_dtype_max
            )
            metrics["saturation_fraction"] = float(np.mean(saturated))
        else:
            metrics["saturation_fraction"] = None

        channel_metrics: list[dict[str, Any]] = []
        depth_denominator = max(1, frames.shape[2] - 1)
        for channel_index in range(frames.shape[1]):
            channel = frames[:, channel_index, :]
            peak_indices = np.argmax(channel, axis=1)
            peak_values = np.take_along_axis(
                channel, peak_indices[:, None], axis=1
            )[:, 0]
            normalized_depths = peak_indices / depth_denominator
            channel_metrics.append(
                {
                    "channel_index": channel_index,
                    "mean_intensity": float(np.mean(channel, dtype=np.float64)),
                    "standard_deviation": float(np.std(channel, dtype=np.float64)),
                    "peak_depth_normalized_mean": float(np.mean(normalized_depths)),
                    "peak_depth_normalized_standard_deviation": float(
                        np.std(normalized_depths)
                    ),
                    "peak_intensity_mean": float(
                        np.mean(peak_values, dtype=np.float64)
                    ),
                    "peak_intensity_standard_deviation": float(
                        np.std(peak_values, dtype=np.float64)
                    ),
                }
            )
        metrics["channels"] = channel_metrics
        return metrics

    def signal_soft_metrics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source": "bounded_downsampled_acquisition_history",
            "hard_thresholds_applied": False,
        }
        for modality, series in (("imu", self._imu), ("encoder", self._encoder)):
            x, values = series.snapshot()
            result[modality] = {
                "point_count": int(x.size),
                "retention_capacity_points": series.max_points,
                "channels": list(series.names),
                "time_span_s": float(x[-1] - x[0]) if x.size > 1 else 0.0,
                "minimum": (
                    [float(value) for value in np.min(values, axis=0)]
                    if values.size
                    else []
                ),
                "maximum": (
                    [float(value) for value in np.max(values, axis=0)]
                    if values.size
                    else []
                ),
            }
        return result

    def signal_snapshots(
        self,
    ) -> dict[str, tuple[tuple[str, ...], np.ndarray, np.ndarray]]:
        imu_x, imu_values = self._imu.snapshot()
        encoder_x, encoder_values = self._encoder.snapshot()
        return {
            "imu": (self._imu.names, imu_x, imu_values),
            "encoder": (self._encoder.names, encoder_x, encoder_values),
        }


def _draw_line(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    height, width = image.shape[:2]
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        double_error = 2 * error
        if double_error >= dy:
            error += dy
            x0 += sx
        if double_error <= dx:
            error += dx
            y0 += sy


def _draw_polyline(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    if x.size < 2:
        return
    for index in range(1, x.size):
        _draw_line(
            image,
            int(x[index - 1]),
            int(y[index - 1]),
            int(x[index]),
            int(y[index]),
            color,
        )


def _render_ultrasound(history: BoundedPreviewHistory) -> np.ndarray:
    width, height = US_PREVIEW_SIZE
    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    _, frames = history.ultrasound_snapshot()
    left, right, top, bottom = 55, 20, 20, 30
    if frames.size == 0:
        canvas[top : height - bottom, left : width - right] = (225, 230, 236)
        return canvas

    channel_count = frames.shape[1]
    gap = 12
    available_height = height - top - bottom - gap * (channel_count - 1)
    panel_height = max(1, available_height // channel_count)
    plot_width = width - left - right
    channel_colors = (
        (0, 188, 212),
        (255, 183, 77),
        (126, 87, 194),
        (102, 187, 106),
    )
    time_indices = np.linspace(
        0, frames.shape[0] - 1, plot_width, dtype=np.int64
    )
    depth_indices = np.linspace(
        0, frames.shape[2] - 1, panel_height, dtype=np.int64
    )
    for channel_index in range(channel_count):
        y0 = top + channel_index * (panel_height + gap)
        y1 = y0 + panel_height
        channel = frames[:, channel_index, :]
        sampled = channel[time_indices][:, depth_indices].T
        low, high = np.percentile(channel, (2.0, 98.0))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(channel)) if channel.size else 0.0
            high = low + 1.0
        normalized = np.clip((sampled - low) / (high - low), 0.0, 1.0)
        # A dark-blue to warm-white map preserves weak echoes while keeping
        # saturation immediately visible.
        rgb = np.empty((*normalized.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.asarray(12 + 243 * normalized, dtype=np.uint8)
        rgb[..., 1] = np.asarray(25 + 210 * normalized, dtype=np.uint8)
        rgb[..., 2] = np.asarray(55 + 145 * normalized, dtype=np.uint8)
        canvas[y0:y1, left : width - right] = rgb
        canvas[y0:y1, left - 7 : left - 2] = channel_colors[
            channel_index % len(channel_colors)
        ]
        peaks = np.argmax(channel, axis=1)
        peak_x = np.linspace(left, width - right - 1, peaks.size)
        peak_y = y0 + peaks * max(1, panel_height - 1) / max(
            1, frames.shape[2] - 1
        )
        _draw_polyline(
            canvas,
            peak_x.astype(np.int32),
            peak_y.astype(np.int32),
            (255, 70, 65),
        )
    return canvas


def _render_signal_panel(
    canvas: np.ndarray,
    bounds: tuple[int, int, int, int],
    x: np.ndarray,
    values: np.ndarray,
) -> None:
    x0, y0, x1, y1 = bounds
    canvas[y0:y1, x0:x1] = (251, 252, 253)
    for fraction in (0.25, 0.5, 0.75):
        grid_y = int(y0 + fraction * (y1 - y0))
        canvas[grid_y : grid_y + 1, x0:x1] = (218, 223, 230)
        grid_x = int(x0 + fraction * (x1 - x0))
        canvas[y0:y1, grid_x : grid_x + 1] = (228, 232, 237)
    if x.size < 2 or values.size == 0:
        return
    maximum_points = max(2, 2 * (x1 - x0))
    if x.size > maximum_points:
        selected = np.linspace(0, x.size - 1, maximum_points, dtype=np.int64)
        x = x[selected]
        values = values[selected]
    finite_x = np.isfinite(x)
    finite_values = np.all(np.isfinite(values), axis=1)
    keep = finite_x & finite_values
    x = x[keep]
    values = values[keep]
    if x.size < 2:
        return
    x_min, x_max = float(x.min()), float(x.max())
    if x_max <= x_min:
        x_max = x_min + 1.0
    value_min, value_max = np.percentile(values, (1.0, 99.0))
    if value_max <= value_min:
        value_max = value_min + 1.0
    plot_x = x0 + (x - x_min) * (x1 - x0 - 1) / (x_max - x_min)
    colors = ((26, 115, 232), (230, 81, 70), (52, 168, 83))
    for channel_index in range(values.shape[1]):
        plot_y = y1 - 1 - (values[:, channel_index] - value_min) * (
            y1 - y0 - 1
        ) / (value_max - value_min)
        _draw_polyline(
            canvas,
            plot_x.astype(np.int32),
            np.clip(plot_y, y0, y1 - 1).astype(np.int32),
            colors[channel_index % len(colors)],
        )


def _render_signals(history: BoundedPreviewHistory) -> np.ndarray:
    width, height = SIGNAL_PREVIEW_SIZE
    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    snapshots = history.signal_snapshots()
    left, right = 55, 20
    top, bottom, gap = 30, 30, 35
    panel_height = (height - top - bottom - gap) // 2
    for panel_index, modality in enumerate(("imu", "encoder")):
        _, x, values = snapshots[modality]
        y0 = top + panel_index * (panel_height + gap)
        _render_signal_panel(
            canvas,
            (left, y0, width - right, y0 + panel_height),
            x,
            values,
        )
        canvas[y0 : y0 + 5, left : left + 80] = (26, 115, 232)
        canvas[y0 : y0 + 5, left + 90 : left + 170] = (230, 81, 70)
    return canvas


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _write_rgb_png(
    path: Path,
    image: np.ndarray,
    *,
    description: dict[str, Any],
) -> None:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("PNG preview must be an RGB image")
    height, width = array.shape[:2]
    raw_rows = b"".join(b"\x00" + row.tobytes() for row in array)
    description_text = json.dumps(
        description, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("latin-1")
    content = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            _png_chunk(b"tEXt", b"Description\x00" + description_text),
            _png_chunk(b"IDAT", zlib.compress(raw_rows, level=6)),
            _png_chunk(b"IEND", b""),
        )
    )
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def publish_quality_preview_pngs(
    layout: TrialLayout,
    history: BoundedPreviewHistory,
    *,
    formal_t0_host_monotonic_ns: int,
    include_ultrasound: bool = True,
    include_signals: bool = True,
) -> PreviewReportBundle:
    """Render enabled PNGs after callers have stopped all Writers.

    The default preserves the original two-report contract.  Acquisition
    profiles that intentionally omit a modality can suppress its unrelated
    preview so every published file remains represented by the Manifest.
    """

    ultrasound_metrics = history.ultrasound_soft_metrics(
        formal_t0_host_monotonic_ns=formal_t0_host_monotonic_ns
    )
    signal_metrics = history.signal_soft_metrics()
    common = {
        "generated_after_acquisition_stop": True,
        "raw_file_scan_performed": False,
        "renderer": "exo-bounded-preview-png-1.0.0",
    }
    ultrasound_metadata = {
        **common,
        "width": US_PREVIEW_SIZE[0],
        "height": US_PREVIEW_SIZE[1],
        "soft_metrics": ultrasound_metrics,
    }
    signal_metadata = {
        **common,
        "width": SIGNAL_PREVIEW_SIZE[0],
        "height": SIGNAL_PREVIEW_SIZE[1],
        "soft_metrics": signal_metrics,
    }
    if include_ultrasound:
        us_partial = layout.partial_path(US_PREVIEW_PATH)
        _write_rgb_png(
            us_partial,
            _render_ultrasound(history),
            description=ultrasound_metadata,
        )
        layout.publish_partial(US_PREVIEW_PATH)
    if include_signals:
        signal_partial = layout.partial_path(SIGNAL_PREVIEW_PATH)
        _write_rgb_png(
            signal_partial,
            _render_signals(history),
            description=signal_metadata,
        )
        layout.publish_partial(SIGNAL_PREVIEW_PATH)
    return PreviewReportBundle(
        ultrasound_metadata=ultrasound_metadata,
        signal_metadata=signal_metadata,
        soft_metrics={
            "ultrasound": ultrasound_metrics,
            "imu_encoder": signal_metrics,
        },
    )


__all__ = [
    "BoundedPreviewHistory",
    "PreviewReportBundle",
    "SIGNAL_PREVIEW_PATH",
    "SIGNAL_PREVIEW_SIZE",
    "US_PREVIEW_PATH",
    "US_PREVIEW_SIZE",
    "publish_quality_preview_pngs",
]
