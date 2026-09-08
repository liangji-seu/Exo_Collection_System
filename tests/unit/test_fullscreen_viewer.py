from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QVBoxLayout

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
from exo_collection.apps.data_studio.plots import TimeSeriesPlot
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
        "time_s,hip_flexion_r,hip_flexion_l\n"
        "0.0,1.0,2.0\n"
        "0.1,1.5,2.5\n"
        "0.2,2.0,3.0\n",
        encoding="utf-8-sig",
    )

    moment = _read_moment_csv(path)

    assert moment is not None
    assert moment.channels == ("hip_flexion_r", "hip_flexion_l")
    assert moment.units == ("N·m", "N·m")
    np.testing.assert_allclose(moment.time_s, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(moment.values, [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])


def test_read_moment_csv_keeps_only_hip_moments(tmp_path: Path) -> None:
    # ground_truth.csv 的表头是 time_s + 12 个 imu_* 特征 + 6 个关节力矩列；力矩
    # 真值面板只保留两个髋关节力矩，IMU 特征与膝/踝力矩都不展示。
    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "time_s,imu_acc_x,imu_acc_y,hip_flexion_r,hip_flexion_l,knee_angle_r\n"
        "0.0,1.0,2.0,3.0,4.0,5.0\n"
        "0.1,1.1,2.1,3.1,4.1,5.1\n",
        encoding="utf-8-sig",
    )

    moment = _read_moment_csv(path)

    assert moment is not None
    assert moment.channels == ("hip_flexion_r", "hip_flexion_l")
    np.testing.assert_allclose(moment.values, [[3.0, 4.0], [3.1, 4.1]])


def test_read_moment_csv_returns_none_when_only_imu_columns(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "time_s,imu_acc_x,imu_acc_y,imu_acc_z\n0.0,1.0,2.0,3.0\n",
        encoding="utf-8-sig",
    )
    assert _read_moment_csv(path) is None


def test_read_moment_csv_returns_none_without_hip_columns(tmp_path: Path) -> None:
    # 只有膝/踝力矩、没有髋关节力矩时，力矩真值面板无可展示内容。
    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "time_s,knee_angle_r,knee_angle_l,ankle_angle_r\n0.0,1.0,2.0,3.0\n",
        encoding="utf-8-sig",
    )
    assert _read_moment_csv(path) is None


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


def test_moment_panel_has_left_right_hip_toggles() -> None:
    app = QApplication.instance() or QApplication(["test-moment-toggles"])
    time_s = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    moment = SignalPlayback(
        time_s=time_s,
        values=np.column_stack([np.zeros(11), np.ones(11)]),
        channels=("hip_flexion_r", "hip_flexion_l"),
        units=("N·m", "N·m"),
    )
    playback = TrialPlayback(
        manifest_path=Path("manifest.json"),
        trial_uuid="00000000-0000-0000-0000-000000000004",
        condition_code="WALK_LEVEL",
        formal_t0_host_monotonic_ns=0,
        ultrasound=None,
        imu=None,
        encoder=None,
        sync=None,
        sync_trigger_times_s=np.empty(0),
        mocap=None,
        moment=moment,
    )

    viewer = FullscreenViewer(playback)
    dock = viewer.dock_for("moment")
    assert dock is not None
    panel = dock.widget()

    boxes = panel.findChildren(QCheckBox)
    labels = {box.text(): box for box in boxes}
    assert set(labels) == {"左髋", "右髋"}

    plot = panel.findChild(TimeSeriesPlot)
    assert plot is not None
    assert plot._curves[0].isVisible() and plot._curves[1].isVisible()

    # 通道顺序 hip_flexion_r(右)→index 0、hip_flexion_l(左)→index 1。
    labels["右髋"].setChecked(False)
    assert not plot._curves[0].isVisible()
    assert plot._curves[1].isVisible()

    labels["右髋"].setChecked(True)
    labels["左髋"].setChecked(False)
    assert plot._curves[0].isVisible()
    assert not plot._curves[1].isVisible()

    viewer.close()
    app.processEvents()


def test_mocap_canvas_fits_off_center_markers() -> None:
    app = QApplication.instance() or QApplication(["test-mocap-fit"])
    n_frames, n_markers = 3, 15
    # Real Nokov data sits far from the origin (≈ -4.6 m in X, +1.5 m in Y), so
    # the old raw-coordinate auto-fit left every marker outside the view.
    base = np.array([-4600.0, 1500.0, 0.0])
    rng = np.random.default_rng(0)
    offsets = rng.normal(0.0, 150.0, (n_frames, n_markers, 3))
    positions = base + offsets
    names = tuple(
        f"010_no_exo_dynamic/{name}"
        for name in (
            "R.ASIS", "L.ASIS", "V.Sacral", "R.Thigh", "R.Knee", "R.Shank",
            "R.Ankle", "R.Heel", "R.Toe", "L.Thigh", "L.Knee", "L.Shank",
            "L.Ankle", "L.Heel", "L.Toe",
        )
    )
    mocap = MocapPlayback(
        time_s=np.linspace(0.0, 1.0, n_frames, dtype=np.float64),
        positions=positions,
        marker_names=names,
    )

    canvas = Mocap3DCanvas(mocap)
    app.processEvents()

    projected = canvas._project(np.asarray(positions[0], dtype=np.float64))
    (xmin, xmax), (ymin, ymax) = canvas.viewRange()
    tol = 50.0  # mm; far smaller than the ~3 m offset the old bug produced
    assert projected[:, 0].min() >= xmin - tol
    assert projected[:, 0].max() <= xmax + tol
    assert projected[:, 1].min() >= ymin - tol
    assert projected[:, 1].max() <= ymax + tol

    # The lower-body marker set should connect into a skeleton, not just dots.
    assert len(canvas._segments) > 0
    canvas.close()
    app.processEvents()


def test_mocap_canvas_does_not_mirror_left_right() -> None:
    app = QApplication.instance() or QApplication(["test-mocap-lr"])
    # Nokov convention: the subject's RIGHT side sits at the more negative x
    # (R.ASIS < L.ASIS). The projection must keep the right marker on the
    # screen's right-hand side rather than mirroring left/right.
    positions = np.array(
        [[[-4600.0, 1500.0, 700.0], [-4400.0, 1500.0, 700.0]]],
        dtype=np.float64,
    )
    names = ("R.ASIS", "L.ASIS")
    mocap = MocapPlayback(
        time_s=np.array([0.0]),
        positions=positions,
        marker_names=names,
    )
    canvas = Mocap3DCanvas(mocap)
    canvas._azimuth = 0.0
    canvas._elevation = 0.0
    projected = canvas._project(positions[0])
    assert float(projected[0, 0]) > float(projected[1, 0])
    canvas.close()
    app.processEvents()


def test_imu_panel_lays_out_sensors_horizontally() -> None:
    app = QApplication.instance() or QApplication(["test-imu-layout"])
    channels = (
        "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
        "mag_x", "mag_y", "mag_z", "roll", "pitch", "yaw",
    )
    n = 120
    time_s = np.linspace(0.0, 1.2, n, dtype=np.float64)
    sensors = tuple(
        SignalPlayback(
            time_s=time_s,
            values=np.random.default_rng(index).normal(0.0, 1.0, (n, 12)),
            channels=channels,
            units=("",) * 12,
            sensor_labels=(f"IMU{index + 1}",),
        )
        for index in range(3)
    )
    playback = TrialPlayback(
        manifest_path=Path("manifest.json"),
        trial_uuid="00000000-0000-0000-0000-000000000009",
        condition_code="WALK_LEVEL",
        formal_t0_host_monotonic_ns=0,
        ultrasound=None,
        imu=None,
        encoder=None,
        sync=None,
        sync_trigger_times_s=np.empty(0),
        imu_sensors=sensors,
    )

    viewer = FullscreenViewer(playback)
    panel = viewer._build_imu_panel()

    layout = panel.layout()
    assert isinstance(layout, QHBoxLayout)
    blocks = [layout.itemAt(index).widget() for index in range(layout.count())]
    assert len(blocks) == 3
    for block in blocks:
        inner = block.layout()
        assert isinstance(inner, QVBoxLayout)
        assert inner.count() == 3
        for slot in range(inner.count()):
            assert isinstance(inner.itemAt(slot).widget(), TimeSeriesPlot)

    viewer.close()
    app.processEvents()
