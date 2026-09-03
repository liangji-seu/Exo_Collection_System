"""历史 run 管理与取消语义（prompt6 §3.10）单元测试。

- ``list_history_runs`` 列出 run_*、解析 QC/同步/静态 Session/分析区间；
- 输入文件变化后历史 run 标为 ``STALE_INPUTS``；
- 取消结局由 ``_finalize_outcome`` 判定：用户取消优先于子进程退出码。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exo_collection.apps.calculate.history import HistoryRun, list_history_runs
from exo_collection.apps.calculate.workers import _finalize_outcome


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _write_manifest(run_dir: Path, inputs: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "subject": {"id": "003", "mass_kg": 75.0, "height_m": 1.75},
                "inputs": inputs,
                "sync": {"method": "AUTO_HIGH", "gaitway_offset_s": 5.835, "confidence": "HIGH"},
                "analysis_time_range_s": [8.0, 20.0],
            }
        ),
        encoding="utf-8",
    )


def _write_result(run_dir: Path, status: str = "PASS") -> None:
    viewer = run_dir / "viewer"
    viewer.mkdir(exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "qc": {"status": status, "summary": "ok"},
                "viewer": {"viewer_dir": str(viewer), "n_frames": 1200},
                "files": {"viewer_dir": str(viewer)},
            }
        ),
        encoding="utf-8",
    )


def _make_run(tmp_path: Path, name: str = "run_20260903_143000_1234") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    return run_dir


def test_list_history_runs_completed(tmp_path: Path) -> None:
    static = tmp_path / "static.c3d"
    static.write_bytes(b"\x00" * 16)
    dyn = tmp_path / "dyn.c3d"
    dyn.write_bytes(b"\x01" * 32)
    txt = tmp_path / "trial.txt"
    txt.write_text("col\n1\n", encoding="utf-8")

    run_dir = _make_run(tmp_path)
    inputs = {
        "static_c3d": _fingerprint(static),
        "dynamic_c3d": _fingerprint(dyn),
        "gaitway_txt": _fingerprint(txt),
    }
    _write_manifest(run_dir, inputs)
    _write_result(run_dir, status="WARN")

    runs = list_history_runs(
        tmp_path,
        {"static_c3d": static, "dynamic_c3d": dyn, "gaitway_txt": txt},
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.state == "completed"
    assert run.qc_status == "WARN"
    assert run.sync_method == "AUTO_HIGH"
    assert run.sync_offset_s == 5.835
    assert run.analysis_window == (8.0, 20.0)
    assert run.static_session == tmp_path.name
    assert run.stale is False
    assert run.replayable


def test_history_run_stale_when_input_changes(tmp_path: Path) -> None:
    dyn = tmp_path / "dyn.c3d"
    dyn.write_bytes(b"\x01" * 32)
    txt = tmp_path / "trial.txt"
    txt.write_text("col\n1\n", encoding="utf-8")

    run_dir = _make_run(tmp_path)
    inputs = {
        "dynamic_c3d": _fingerprint(dyn),
        "gaitway_txt": _fingerprint(txt),
    }
    _write_manifest(run_dir, inputs)
    _write_result(run_dir)

    current = {"dynamic_c3d": dyn, "gaitway_txt": txt}
    assert not list_history_runs(tmp_path, current)[0].stale

    # 输入文件被替换（内容变化）→ 历史 run 标 STALE。
    dyn.write_bytes(b"\x02" * 32)
    run = list_history_runs(tmp_path, current)[0]
    assert run.stale is True
    assert "dynamic_c3d" in (run.stale_reason or "")


def test_history_run_cancelled_and_failed(tmp_path: Path) -> None:
    cancelled_dir = _make_run(tmp_path, "run_20260903_150000_0001")
    (cancelled_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (cancelled_dir / "cancel.flag").write_text("cancel", encoding="utf-8")

    failed_dir = _make_run(tmp_path, "run_20260903_150000_0002")
    (failed_dir / "manifest.json").write_text("{}", encoding="utf-8")

    runs = {r.run_id: r for r in list_history_runs(tmp_path)}
    assert runs["run_20260903_150000_0001"].state == "cancelled"
    assert runs["run_20260903_150000_0002"].state == "failed"
    assert runs["run_20260903_150000_0001"].qc_status is None


def test_list_history_runs_ignores_non_run_dirs(tmp_path: Path) -> None:
    (tmp_path / "other_dir").mkdir()
    (tmp_path / "stray_file.txt").write_text("x", encoding="utf-8")
    assert list_history_runs(tmp_path) == []


# --------------------------------------------------------------------------
# 取消语义：用户取消 → CANCELLED（不是 FAILED）
# --------------------------------------------------------------------------
def test_finalize_outcome_cancel_requested_wins_over_nonzero_exit() -> None:
    # 子进程被强制 terminate 退出码非 0，但用户已请求取消 → cancelled。
    outcome, _, _ = _finalize_outcome(1, False, True, None, None)
    assert outcome == "cancelled"


def test_finalize_outcome_child_cancelled_event() -> None:
    # 子进程协作式输出 cancelled 事件（退出码 2）→ cancelled。
    outcome, _, _ = _finalize_outcome(2, True, False, None, None)
    assert outcome == "cancelled"


def test_finalize_outcome_success_and_failure() -> None:
    outcome, payload, _ = _finalize_outcome(0, False, False, {"exit_code": 0}, None)
    assert outcome == "finished"
    assert payload == {"exit_code": 0}

    outcome, _, message = _finalize_outcome(1, False, False, None, "boom")
    assert outcome == "failed"
    assert message == "boom"


def test_history_run_dataclass_replayable() -> None:
    from exo_collection.apps.calculate.history import HistoryRun as HR

    run = HR(
        run_dir=Path("/tmp/run"), run_id="run_x", created_utc=None,
        state="completed", qc_status="PASS", static_session=None,
        sync_method=None, sync_offset_s=None, sync_confidence=None,
        analysis_window=None, stale=False, stale_reason=None,
        viewer_dir=None, n_frames=None,
    )
    assert run.replayable is False
