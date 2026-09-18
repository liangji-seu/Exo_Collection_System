"""Unit tests for the online-testing SubscriptionHub."""

from __future__ import annotations

from queue import Queue

import numpy as np
import pytest

from exo_collection.acquisition.recording_stream import (
    RecordedRawEvent,
    RecordingBoundary,
    RecordingBoundaryKind,
)
from exo_collection.apps.model_runtime.subscription import (
    SubscriptionHub,
    flatten_channel_names,
)
from exo_collection.domain.events import SampleBatch

_NS = 1_000_000_000


def _batch(modality: str, device_id: str, data: np.ndarray, t_ns: int, rate: float) -> SampleBatch:
    return SampleBatch(
        device_id=device_id,
        modality=modality,
        clock_domain=f"{modality}_clock",
        first_sample_index=0,
        sample_count=int(data.shape[0]),
        sequence_number=0,
        sample_rate_hz=rate,
        host_monotonic_ns=t_ns,
        data=data.astype(np.float32),
    )


# -- flatten_channel_names -------------------------------------------------


def test_flatten_identity() -> None:
    assert flatten_channel_names(("a", "b", "c"), (3,)) == ("a", "b", "c")


def test_flatten_imu_like_columns() -> None:
    # IMU: 12 axis names, sample_shape=(n_devices, 12).
    names = flatten_channel_names(("x", "y"), (2, 2))
    assert names == ("x[0]", "y[0]", "x[1]", "y[1]")


def test_flatten_ultrasound_rows() -> None:
    # Ultrasound a_line: channels=("ch_1","ch_2"), frame_shape=(2, 1000).
    names = flatten_channel_names(("ch_1", "ch_2"), (2, 1000))
    assert names[0] == "ch_1[0]"
    assert names[999] == "ch_1[999]"
    assert names[1000] == "ch_2[0]"
    assert len(names) == 2000


def test_flatten_frame_samples() -> None:
    # Ultrasound frame: channels=("amplitude",), frame_shape=(1000,).
    names = flatten_channel_names(("amplitude",), (1000,))
    assert names[0] == "amplitude[0]"
    assert len(names) == 1000


def test_flatten_generic_fallback() -> None:
    assert flatten_channel_names(("a", "b"), (5,)) == ("ch_0", "ch_1", "ch_2", "ch_3", "ch_4")


# -- SubscriptionHub -------------------------------------------------------


def test_register_and_drain_sample_batch() -> None:
    queue: Queue = Queue()
    hub = SubscriptionHub()
    hub.register_stream(
        "encoder",
        queue,
        channels=("a", "b", "c", "d", "e", "f"),
        sample_shape=(6,),
        rate_hz=100.0,
        device_id="enc",
    )
    queue.put(_batch("encoder", "enc", np.arange(18, dtype=np.float32).reshape(3, 6), 0, 100.0))
    consumed = hub.drain()
    assert consumed == 1
    assert hub.count("encoder") == 3
    assert hub.flat_channels("encoder") == ("a", "b", "c", "d", "e", "f")


def test_drain_skips_boundaries_and_forwards_raw_event() -> None:
    queue: Queue = Queue()
    hub = SubscriptionHub()
    hub.register_stream("encoder", queue, channels=("a",), sample_shape=(1,), rate_hz=100.0)
    queue.put(
        RecordingBoundary(
            kind=RecordingBoundaryKind.START,
            trial_uuid="00000000-0000-0000-0000-000000000000",
            modality="encoder",
            device_id="enc",
        )
    )
    raw = _batch("encoder", "enc", np.array([[1.0], [2.0]], dtype=np.float32), 0, 100.0)
    queue.put(RecordedRawEvent(trial_uuid="00000000-0000-0000-0000-000000000000", modality="encoder", device_id="enc", event=raw))
    queue.put(
        RecordingBoundary(
            kind=RecordingBoundaryKind.END,
            trial_uuid="00000000-0000-0000-0000-000000000000",
            modality="encoder",
            device_id="enc",
        )
    )
    consumed = hub.drain()
    assert consumed == 1  # boundary markers are consumed but not data
    assert hub.count("encoder") == 2
    assert hub.boundary("encoder") == "END"


def test_window_sliding() -> None:
    queue: Queue = Queue()
    hub = SubscriptionHub()
    hub.register_stream("encoder", queue, channels=("a", "b"), sample_shape=(2,), rate_hz=100.0)
    data = np.column_stack([np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32) + 10])
    queue.put(_batch("encoder", "enc", data, 0, 100.0))
    hub.drain()

    times, values = hub.window("encoder", 0.02)
    assert times.tolist() == [20_000_000, 30_000_000, 40_000_000]
    assert values.shape == (3, 2)
    assert values[0].tolist() == [2.0, 12.0]


def test_latest_returns_most_recent() -> None:
    queue: Queue = Queue()
    hub = SubscriptionHub()
    hub.register_stream("encoder", queue, channels=("a",), sample_shape=(1,), rate_hz=100.0)
    queue.put(_batch("encoder", "enc", np.array([[1.0], [2.0], [3.0]], dtype=np.float32), 0, 100.0))
    hub.drain()
    latest = hub.latest("encoder")
    assert latest is not None
    t_ns, row = latest
    assert t_ns == 20_000_000
    assert row.tolist() == [3.0]


def test_window_empty_and_clear() -> None:
    queue: Queue = Queue()
    hub = SubscriptionHub()
    hub.register_stream("encoder", queue, channels=("a",), sample_shape=(1,), rate_hz=100.0)
    times, values = hub.window("encoder", 1.0)
    assert times.size == 0
    assert values.shape == (0, 0)
    queue.put(_batch("encoder", "enc", np.array([[1.0], [2.0]], dtype=np.float32), 0, 100.0))
    hub.drain()
    hub.clear("encoder")
    assert hub.count("encoder") == 0


def test_unknown_modality_raises() -> None:
    hub = SubscriptionHub()
    with pytest.raises(KeyError):
        hub.window("missing", 1.0)
    with pytest.raises(KeyError):
        hub.subscribe("missing", lambda _e: None)


def test_subscribe_receives_events() -> None:
    queue: Queue = Queue()
    hub = SubscriptionHub()
    hub.register_stream("encoder", queue, channels=("a",), sample_shape=(1,), rate_hz=100.0)
    seen: list[str] = []
    hub.subscribe("encoder", lambda event: seen.append(event.modality))
    queue.put(_batch("encoder", "enc", np.array([[1.0]], dtype=np.float32), 0, 100.0))
    hub.drain()
    assert seen == ["encoder"]
