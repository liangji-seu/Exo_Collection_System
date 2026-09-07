from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from exo_collection.apps.data_studio.fullscreen_viewer import (
    FullscreenViewer,
    Mocap3DCanvas,
)
from exo_collection.apps.data_studio.local_tools import (
    MocapPlayback,
    SignalPlayback,
    TrialPlayback,
    _read_hdf5_mocap,
    _read_moment_csv,
)
from exo_collection.writers import Hdf5SignalWriter


def _write_mocap_h5(
    path: Path,
    n_markers: int,
    n_frames: int,
    *,
    names: tuple[str, ...] | None = None,
) -> None:
    names = names or tuple(f"M{index:02d}" for index in range(n_markers))
    with Hdf5SignalWriter(
        path,
        channels=names,
        units=("mm",) * n_markers,
        device_metadata={"device_id": "mocap_sim"},
        sample_shape=(n_markers, 3),
        nominal_rate_hz=100.0,
    ) as writer:
        writer.append(
            np.arange(n_frames * n_markers * 3, dtype=np.float32).reshape(
                n_frames, n_markers, 3
            ),
            sample_index=0,
            host_monotonic_ns=np.arange(n_frames, dtype=np.uint64) * 10_000_000
            + 900_000_000,
        )


def test_read_hdf5_mocap_returns_frames_markers_xyz_and_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mocap.h5"
    n_markers, n_frames = 16, 8
    _write_mocap_h5(path, n_markers, n_frames)

    mocap = _read_hdf5_mocap(
        path, formal_t0_ns=900_000_000, max_points=1000
    )

    assert mocap.positions.shape == (n_frames, n_markers, 3)
    assert mocap.marker_names == tuple(f"M{index:02d}" for index in range(n_markers))
    assert mocap.time_s.shape == (n_frames,)
    np.testing.assert_allclose(mocap.time_s, np.arange(n_frames) * 0.01)
    expected = np.arange(n_frames * n_markers * 3, dtype=np.float32).reshape(
        n_frames, n_markers, 3
    )
    np.testing.assert_allclose(np.asarray(mocap.positions), expected)


def test_read_hdf5_mocap_generates_names_when_metadata_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mocap_no_channels.h5"
    n_frames, n_markers = 4, 5
    with h5py.File(path, "w") as handle:
        handle.attrs["closed_cleanly"] = True
        samples = handle.create_group("samples")
        samples.create_dataset(
            "data", data=np.zeros((n_frames, n_markers, 3), dtype=np.float32)
        )
        samples.create_dataset(
            "host_monotonic_ns",
            data=(900_000_000 + np.arange(n_frames) * 10_000_000).astype(np.uint64),
        )

    mocap = _read_hdf5_mocap(path, formal_t0_ns=900_000_000, max_points=100)

    assert mocap.positions.shape == (n_frames, n_markers, 3)
    assert mocap.marker_names == tuple(
        f"marker_{index + 1:02d}" for index in range(n_markers)
    )


def test_read_hdf5_mocap_downsamples_frames_without_aliasing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mocap.h5"
    _write_mocap_h5(path, n_markers=3, n_frames=100)

    mocap = _read_hdf5_mocap(path, formal_t0_ns=900_000_000, max_points=20)

    assert mocap.time_s.size == 20
    assert mocap.positions.shape == (20, 3, 3)
    # Even index selection keeps the real frame time, monotonic in host order.
    assert bool(np.all(np.diff(mocap.time_s) > 0.0))


def test_read_moment_csv_parses_truth_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "time_s,knee_moment,ankle_moment\n"
        "0.0,1.0,2.0\n"
        "0.1,1.5,2.5\n"
        "0.2,2.0,3.0\n",
        encoding="utf-8-sig",
    )

    moment = _read_moment_csv(path)

    assert moment is not None
    assert moment.channels == ("knee_moment", "ankle_moment")
    assert moment.units == ("N·m", "N·m")
    np.testing.assert_allclose(moment.time_s, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(moment.values, [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])


def test_read_moment_csv_returns_none_when_absent_or_malformed(tmp_path: Path) -> None:
    assert _read_moment_csv(tmp_path / "missing.csv") is None

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("moment_a,moment_b\n1.0,2.0\n", encoding="utf-8")
    assert _read_moment_csv(bad_header) is None

    not_numeric = tmp_path / "not_numeric.csv"
    not_numeric.write_text("time_s,moment\n0.0,oops\n", encoding="utf-8")
    assert _read_moment_csv(not_numeric) is None


def test_fullscreen_viewer_registers_mocap_and_moment_docks() -> None:
    app = QApplication.instance() or QApplication(["test-fullscreen-viewer"])
    time_s = np.linspace(0.0, 10.0, 101, dtype=np.float64)
    mocap = MocapPlayback(
        time_s=np.linspace(0.0, 1.0, 10, dtype=np.float64),
        positions=np.zeros((10, 16, 3), dtype=np.float64),
        marker_names=tuple(f"marker_{index:02d}" for index in range(16)),
    )
    moment = SignalPlayback(
        time_s=time_s,
        values=np.zeros((time_s.size, 2), dtype=np.float64),
        channels=("knee_moment", "ankle_moment"),
        units=("N·m", "N·m"),
    )
    playback = TrialPlayback(
        manifest_path=Path("manifest.json"),
        trial_uuid="00000000-0000-0000-0000-000000000003",
        condition_code="WALK_LEVEL",
        formal_t0_host_monotonic_ns=0,
        ultrasound=None,
        imu=None,
        encoder=None,
        sync=None,
        sync_trigger_times_s=np.empty(0),
        mocap=mocap,
        moment=moment,
    )

    viewer = FullscreenViewer(playback)

    assert set(viewer.modalities) == {"mocap", "moment"}
    mocap_dock = viewer.dock_for("mocap")
    assert mocap_dock is not None
    assert isinstance(mocap_dock.widget(), Mocap3DCanvas)
    assert viewer.dock_for("moment") is not None

    viewer.set_playback_time(5.0)
    viewer.toggle_playback()
    viewer.toggle_playback()
    assert viewer._current_time > 0.0
    viewer.close()
    app.processEvents()
