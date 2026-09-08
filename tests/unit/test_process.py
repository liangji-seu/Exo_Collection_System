"""Exo Process 批量解算核心的纯逻辑单元测试。

不 import OpenSim、不跑真实解算；只验证状态判定、LOW 置信度跳过、输入门禁与
「同步 → 预处理 → OpenSim → 导出」的编排顺序。pipeline 的同步/预处理函数通过
monkeypatch 打桩，因此不会触碰真实 C3D / HDF5 数据。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 让测试能 monkeypatch pipeline 里的同步/预处理函数（不依赖 opensim）。
_REPO = Path(__file__).resolve().parents[2]
_PIPELINE = _REPO / "opensim_joint_moment_pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

import exo_collection.apps.process.batch as batch_module  # noqa: E402
import pipeline.opensim_io.prep_session as prep_module  # noqa: E402
import pipeline.synchronization.sync as sync_module  # noqa: E402
from exo_collection.apps.calculate.models import SessionFiles, SessionRecord  # noqa: E402
from exo_collection.apps.process.batch import (  # noqa: E402
    SessionSolveState,
    SolveCancelled,
    SyncFailed,
    session_solve_status,
    solve_one_session,
)


def _make_session(
    tmp_path: Path,
    *,
    files: SessionFiles | None = None,
    name: str = "session",
) -> SessionRecord:
    session_dir = tmp_path / name
    session_dir.mkdir(parents=True, exist_ok=True)
    return SessionRecord(
        manifest_path=session_dir / ".exo" / "manifest.json",
        session_dir=session_dir,
        session_name=name,
        subject_code="003",
        subject_uuid="subj-uuid",
        project_code="P",
        project_name="P",
        condition_code="WALK_STEADY_1P00",
        condition_name="WALK_STEADY_1P00",
        condition_level=None,
        repeat_index=1,
        trial_uuid="trial-uuid",
        session_uuid=f"session-uuid-{name}",
        state="FINALIZED",
        started_at_utc="2026-01-01T00:00:00",
        files=files if files is not None else SessionFiles(),
    )


def _complete_files() -> SessionFiles:
    return SessionFiles(
        c3d_path=Path("dyn.c3d"),
        txt_path=Path("dyn.txt"),
        mocap_h5_path=Path("mocap.h5"),
        imu_h5_path=Path("imu.h5"),
    )


def _static_session(tmp_path: Path) -> SessionRecord:
    return _make_session(
        tmp_path, files=SessionFiles(c3d_path=Path("static.c3d")), name="static"
    )


def _high_sync() -> dict:
    return {
        "gaitway_offset_s": 0.5,
        "confidence": "HIGH",
        "c3d_start_in_mocap_h5_frame": 100,
        "n_pairs": 3,
        "c3d_h5_matched_markers": ["m"] * 15,
        "c3d_h5_unique": True,
        "mad_s": 0.001,
        "mocap_h5_monotonic": True,
        "mocap_h5_clock_gaps": 0,
        "imu_clock_monotonic": True,
        "imu_clock_gaps": 0,
    }


# ── session_solve_status ────────────────────────────────────────


def test_status_incomplete_without_c3d_or_txt(tmp_path: Path) -> None:
    files = SessionFiles(c3d_path=Path("a.c3d"))  # 缺 txt
    assert session_solve_status(_make_session(tmp_path, files=files)) is (
        SessionSolveState.INCOMPLETE
    )


def test_status_solved_when_ground_truth_exists(tmp_path: Path) -> None:
    record = _make_session(tmp_path, files=_complete_files())
    (record.session_dir / "ground_truth.csv").write_text("time_s\n", encoding="utf-8")
    assert session_solve_status(record) is SessionSolveState.SOLVED


def test_status_missing_modality_without_mocap_or_imu(tmp_path: Path) -> None:
    files = SessionFiles(c3d_path=Path("a.c3d"), txt_path=Path("a.txt"))
    assert session_solve_status(_make_session(tmp_path, files=files)) is (
        SessionSolveState.MISSING_MODALITY
    )


def test_status_unsolved_when_complete(tmp_path: Path) -> None:
    assert session_solve_status(_make_session(tmp_path, files=_complete_files())) is (
        SessionSolveState.UNSOLVED
    )


# ── solve_one_session 输入门禁 / 取消 ───────────────────────────


def test_solve_one_session_missing_dynamic_inputs_raises(tmp_path: Path) -> None:
    files = SessionFiles(c3d_path=Path("a.c3d"), txt_path=Path("a.txt"))  # 缺 mocap/imu
    with pytest.raises(ValueError):
        solve_one_session(
            _make_session(tmp_path, files=files),
            _static_session(tmp_path / "st"),
            Path("python"),
            Path("model.osim"),
            75.0,
            1.75,
            cancel_check=lambda: False,
            progress=lambda _: None,
        )


def test_solve_one_session_cancelled_before_sync(tmp_path: Path) -> None:
    with pytest.raises(SolveCancelled):
        solve_one_session(
            _make_session(tmp_path, files=_complete_files()),
            _static_session(tmp_path / "st"),
            Path("python"),
            Path("model.osim"),
            75.0,
            1.75,
            cancel_check=lambda: True,
            progress=lambda _: None,
        )


# ── solve_one_session 同步置信度门禁 ─────────────────────────────


@pytest.mark.parametrize("confidence", ["LOW", "MEDIUM", "UNKNOWN", ""])
def test_solve_one_session_low_confidence_raises_sync_failed(
    tmp_path: Path, monkeypatch, confidence
) -> None:
    monkeypatch.setattr(
        sync_module,
        "run_auto_sync",
        lambda *a, **k: {**_high_sync(), "confidence": confidence},
    )
    with pytest.raises(SyncFailed):
        solve_one_session(
            _make_session(tmp_path, files=_complete_files()),
            _static_session(tmp_path / "st"),
            Path("python"),
            Path("model.osim"),
            75.0,
            1.75,
            cancel_check=lambda: False,
            progress=lambda _: None,
        )


def test_solve_one_session_stomp_error_raises_sync_failed(
    tmp_path: Path, monkeypatch
) -> None:
    def boom(*_a, **_k):
        raise sync_module.StompSyncError("跺脚峰不足 3 对", {})

    monkeypatch.setattr(sync_module, "run_auto_sync", boom)
    with pytest.raises(SyncFailed):
        solve_one_session(
            _make_session(tmp_path, files=_complete_files()),
            _static_session(tmp_path / "st"),
            Path("python"),
            Path("model.osim"),
            75.0,
            1.75,
            cancel_check=lambda: False,
            progress=lambda _: None,
        )


# ── solve_one_session 编排（happy path） ─────────────────────────


def test_solve_one_session_happy_path_returns_out_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sync_module, "run_auto_sync", lambda *a, **k: _high_sync())

    prep_out_dirs: list[Path] = []

    def fake_prepare(**kwargs) -> dict:
        from pipeline.qc.evaluate import _sync_checks
        checks = _sync_checks(kwargs["sync_quality"])
        assert not any(c["status"] == "FAIL" for c in checks)
        assert kwargs["sync_quality"]["n_pairs"] == 3
        out_dir = Path(kwargs["out_dir"])
        prep_out_dirs.append(out_dir)
        return {"manifest_path": str(out_dir / "manifest.json")}

    monkeypatch.setattr(prep_module, "prepare_session", fake_prepare)

    monkeypatch.setattr(
        batch_module,
        "_run_opensim",
        lambda *a, **k: {"viewer_dir": str(tmp_path / "viewer"), "qc": {"status": "PASS"}},
    )

    out_path = tmp_path / "ground_truth.csv"

    def fake_export(dynamic, start_frame, run_dir, payload) -> Path:
        assert start_frame == 100
        out_path.write_text("time_s\n", encoding="utf-8")
        return out_path

    monkeypatch.setattr(batch_module, "_export_ground_truth", fake_export)

    result = solve_one_session(
        _make_session(tmp_path, files=_complete_files()),
        _static_session(tmp_path / "st"),
        Path("python"),
        Path("model.osim"),
        75.0,
        1.75,
        cancel_check=lambda: False,
        progress=lambda _: None,
    )

    assert result["out_path"] == out_path
    assert result["sync_confidence"] == "HIGH"
    assert result["qc_status"] == "PASS"
    assert out_path.is_file()
    # 预处理写进 session 的 derived/opensim ASCII 工作目录，而非原始数据目录。
    assert len(prep_out_dirs) == 1
    assert prep_out_dirs[0].parts[-3:] == ("derived", "opensim", prep_out_dirs[0].name)
    assert prep_out_dirs[0].name.startswith("run_")


@pytest.mark.parametrize("qc", ["PASS", "WARN", "FAIL", None])
def test_qc_survives_reload_and_worker_signal(tmp_path, monkeypatch, qc):
    from exo_collection.apps.calculate.ground_truth_status import write_ground_truth_qc
    from exo_collection.apps.process.window import ProcessWindow
    record = _make_session(tmp_path, files=_complete_files())
    csv = record.session_dir / "ground_truth.csv"
    csv.write_text("time_s\n0\n", encoding="utf-8")
    write_ground_truth_qc(csv, qc, tmp_path)
    expected = {"PASS": SessionSolveState.QC_PASS, "WARN": SessionSolveState.QC_WARN,
                "FAIL": SessionSolveState.QC_FAIL}.get(qc, SessionSolveState.SOLVED)
    assert session_solve_status(record) == expected
    assert (qc or "未知") in ProcessWindow._solve_text(expected)[0]
    worker = batch_module.BatchWorker([record], _static_session(tmp_path / "st"),
                                     Path("python"), Path("model"), mass_kg=75, height_m=1.75)
    monkeypatch.setattr(batch_module, "solve_one_session", lambda *a, **k: {"qc_status": qc, "out_path": csv})
    results = []
    worker.signals.session_finished.connect(lambda *args: results.append(args))
    worker.run()
    assert results == [(record.session_name, expected.value, str(csv))]
    # A later CSV export must not inherit an old quality label.
    csv.write_text("replacement CSV", encoding="utf-8")
    assert session_solve_status(record) == SessionSolveState.SOLVED


def test_cancel_reaches_silent_subprocess(tmp_path):
    from threading import Event, Timer
    script = tmp_path / "silent.py"
    flag = tmp_path / "cancel.flag"
    script.write_text(
        "import sys,time\nfrom pathlib import Path\n"
        "flag=Path(sys.argv[sys.argv.index('--cancel-file')+1])\n"
        "deadline=time.monotonic()+5\n"
        "while not flag.exists() and time.monotonic()<deadline: time.sleep(.02)\n"
        "sys.exit(0 if flag.exists() else 9)\n", encoding="utf-8")
    cancelled = Event()
    timer = Timer(.15, cancelled.set)
    timer.start()
    try:
        with pytest.raises(SolveCancelled):
            batch_module._run_opensim(Path(sys.executable), script, tmp_path / "manifest.json",
                                      flag, cancelled.is_set, lambda _: None)
        assert flag.is_file(), "Cancellation was never delivered while stdout was silent"
    finally:
        timer.cancel()
        timer.join()


@pytest.mark.parametrize("qc", ["PASS", "WARN", "FAIL", None])
def test_data_studio_displays_persisted_qc(tmp_path, qc):
    from PySide6.QtWidgets import QApplication, QTreeWidgetItem
    from exo_collection.apps.calculate.ground_truth_status import write_ground_truth_qc
    from exo_collection.apps.data_studio.sync_data import check_trial_solved
    from exo_collection.apps.data_studio.window import DataStudioWindow
    app = QApplication.instance() or QApplication([])
    csv = tmp_path / "ground_truth.csv"
    csv.write_text("time_s\n0\n", encoding="utf-8")
    write_ground_truth_qc(csv, qc, tmp_path)
    manifest = tmp_path / ".exo" / "manifest.json"
    status = check_trial_solved(manifest)
    assert status.qc_status == qc
    tree = DataStudioWindow._attach_solved_status(
        [{"type": "trial", "manifest_path": str(manifest), "sync_cap_names": ["capture"]}],
        {str(manifest.resolve()): status})
    item = QTreeWidgetItem()
    DataStudioWindow._apply_sync_column(item, tree[0])
    assert f"QC {qc or '未知'}" in item.text(4)
    if qc != "PASS":
        assert item.foreground(4).color().name() != "#20a35a"
