from __future__ import annotations

from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np

from exo_collection.adapters.base import StartToken, TrialContext
from exo_collection.adapters.xing_nokov import (
    XingNokovEmgAdapter,
    XingNokovForcePlateAdapter,
    XingNokovMocapAdapter,
)


class _FakeBackend:
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        self.callback: Callable[[Mapping[str, Any], int], None] | None = None
        self.started = False

    def connect(self, callback: Callable[[Mapping[str, Any], int], None]) -> None:
        self.callback = callback

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.callback = None

    def publish(self, payload: Mapping[str, Any], host_ns: int = 123_000_000) -> None:
        assert self.started and self.callback is not None
        self.callback(payload, host_ns)


def _start(adapter: Any) -> None:
    adapter.connect()
    adapter.prepare(TrialContext(trial_uuid=uuid4(), session_uuid=uuid4()))
    adapter.start(StartToken())


def test_mocap_adapter_copies_marker_sets_into_fixed_sample_geometry() -> None:
    backend = _FakeBackend(
        {
            "frame_rate_hz": 100.0,
            "marker_names": ["leg/hip", "leg/knee", "foot/heel"],
            "marker_sets": [
                {"name": "leg", "marker_names": ["hip", "knee"]},
                {"name": "foot", "marker_names": ["heel"]},
            ],
        }
    )
    adapter = XingNokovMocapAdapter(backend=backend)
    _start(adapter)
    backend.publish(
        {
            "frame_number": 42,
            "device_timestamp": 9876,
            "marker_sets": [
                {
                    "name": "leg",
                    "values": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                },
                {
                    "name": "foot",
                    "values": np.asarray([[7, 8, 9]], dtype=np.float32),
                },
            ],
        }
    )
    event = adapter.get_event(timeout=0.1)
    assert event is not None
    assert event.modality == "mocap"
    assert event.data.shape == (1, 3, 3)
    np.testing.assert_array_equal(event.data[0, 2], [7, 8, 9])
    assert adapter.descriptor().metadata["marker_names"] == [
        "leg/hip",
        "leg/knee",
        "foot/heel",
    ]
    adapter.stop()
    adapter.close()


def test_mocap_adapter_tracks_sustained_missing_markers() -> None:
    """缺失 marker（超大占位坐标）的持续时长被计入健康指标。"""
    backend = _FakeBackend(
        {
            "frame_rate_hz": 100.0,
            "marker_names": ["leg/hip", "leg/knee", "foot/heel"],
            "marker_sets": [
                {"name": "leg", "marker_names": ["hip", "knee"]},
                {"name": "foot", "marker_names": ["heel"]},
            ],
        }
    )
    adapter = XingNokovMocapAdapter(backend=backend)
    _start(adapter)

    present = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    missing_heel = np.asarray(
        [[99999999.0, 99999999.0, 99999999.0]], dtype=np.float32
    )
    recovered_heel = np.asarray([[7, 8, 9]], dtype=np.float32)

    # 帧 1（t=0）：heel 缺失 → 缺失 1 点，持续时长起始为 0。
    backend.publish(
        {
            "frame_number": 1,
            "device_timestamp": 1,
            "marker_sets": [
                {"name": "leg", "values": present},
                {"name": "foot", "values": missing_heel},
            ],
        },
        host_ns=0,
    )
    assert adapter.get_event(timeout=0.1) is not None
    metrics = adapter.health().metrics
    assert metrics["marker_missing_count"] == 1
    assert metrics["marker_missing_streak_s"] == 0.0

    # 帧 2（t=4s）：heel 仍缺失 → 已持续缺失 4 s。
    backend.publish(
        {
            "frame_number": 2,
            "device_timestamp": 2,
            "marker_sets": [
                {"name": "leg", "values": present},
                {"name": "foot", "values": missing_heel},
            ],
        },
        host_ns=4_000_000_000,
    )
    assert adapter.get_event(timeout=0.1) is not None
    metrics = adapter.health().metrics
    assert metrics["marker_missing_count"] == 1
    assert abs(metrics["marker_missing_streak_s"] - 4.0) < 0.01

    # 帧 3（t=5s）：heel 恢复 → 缺失归零。
    backend.publish(
        {
            "frame_number": 3,
            "device_timestamp": 3,
            "marker_sets": [
                {"name": "leg", "values": present},
                {"name": "foot", "values": recovered_heel},
            ],
        },
        host_ns=5_000_000_000,
    )
    assert adapter.get_event(timeout=0.1) is not None
    metrics = adapter.health().metrics
    assert metrics["marker_missing_count"] == 0
    assert metrics["marker_missing_streak_s"] == 0.0

    adapter.stop()
    adapter.close()


def test_emg_adapter_transports_sdk_subframes_as_sample_rows() -> None:
    backend = _FakeBackend({"sdk_version": "4.1.0.5645"})
    adapter = XingNokovEmgAdapter(
        {"channel_count": 3, "channel_names": ["RF", "BF", "TA"]},
        backend=backend,
    )
    _start(adapter)
    values = np.asarray(
        [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
        dtype=np.float32,
    )
    backend.publish(
        {
            "frame_number": 9,
            "device_timestamp": 4567,
            "values": values,
        }
    )
    event = adapter.get_event(timeout=0.1)
    assert event is not None
    assert event.modality == "emg"
    assert event.sample_count == 2
    assert event.data.shape == (2, 3)
    np.testing.assert_allclose(event.data, values)
    assert adapter.descriptor().channels == ("RF", "BF", "TA")
    adapter.stop()
    adapter.close()


def test_force_plate_adapter_transports_mocap_frame_analog_channels() -> None:
    backend = _FakeBackend({"sdk_version": "4.1.0.5645"})
    adapter = XingNokovForcePlateAdapter(backend=backend)
    _start(adapter)
    values = np.asarray([[12.0, 3.0, -500.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    backend.publish(
        {"frame_number": 17, "device_timestamp": 888, "values": values}
    )
    event = adapter.get_event(timeout=0.1)
    assert event is not None
    assert event.modality == "force_plate"
    assert event.sample_count == 1
    assert event.data.shape == (1, 6)
    np.testing.assert_allclose(event.data, values)
    assert adapter.descriptor().channels == ("fx", "fy", "fz", "mx", "my", "mz")
    assert adapter.descriptor().units == ("N", "N", "N", "N*m", "N*m", "N*m")
    adapter.stop()
    adapter.close()
