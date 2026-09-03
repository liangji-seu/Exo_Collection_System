from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from exo_collection.apps.data_studio.sync_data import (
    check_all_trial_sync,
    check_trial_sync_data,
    load_cap_names,
    sync_sidecar_files,
)


CAP_NAME = "subj001_condA_r1_abcd1234"
TRIAL_UUID = "00000000-0000-0000-0000-000000000001"


def _manifest_path(trial_root: Path) -> Path:
    (trial_root / ".exo").mkdir(parents=True, exist_ok=True)
    return trial_root / ".exo" / "manifest.json"


def _write_trigger(trial_root: Path, cap_name: str = CAP_NAME) -> None:
    raw = trial_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": "1.0.0",
        "trial_uuid": TRIAL_UUID,
        "sequence": 0,
        "kind": "capture_start",
        "capture_name": cap_name,
        "database_path": "C:/xingying/project",
        "notes": "",
        "description": "",
        "delay": "",
        "timecode": "",
        "packet_id": "",
        "host_monotonic_ns": 1000,
        "host_utc_ns": 2000,
    }
    (raw / "xingying_trigger.jsonl").write_text(
        json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_sync_manifest(trial_root: Path, cap_name: str) -> None:
    exo = trial_root / ".exo"
    exo.mkdir(parents=True, exist_ok=True)
    (exo / "sync_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "xingying_triggers": [{"capture_name": cap_name}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ── load_cap_names ──────────────────────────────────────────────


def test_load_cap_names_from_trigger(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    assert load_cap_names(trial_root) == (CAP_NAME,)


def test_load_cap_names_falls_back_to_sync_manifest(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_sync_manifest(trial_root, "another_cap")
    assert load_cap_names(trial_root) == ("another_cap",)


def test_load_cap_names_empty_when_no_source(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    assert load_cap_names(trial_root) == ()


def test_load_cap_names_dedupes(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_sync_manifest(trial_root, CAP_NAME)
    # sync_manifest with duplicate entries → deduped
    exo = trial_root / ".exo"
    (exo / "sync_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "xingying_triggers": [
                    {"capture_name": CAP_NAME},
                    {"capture_name": CAP_NAME},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert load_cap_names(trial_root) == (CAP_NAME,)


def test_load_cap_names_adds_take_stripped_base(tmp_path: Path) -> None:
    # XINGYING broadcasts the take-appended name (uuid8 + take index); the
    # operator-named .txt uses the bare uuid8 form.  Both must be registered.
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, f"{CAP_NAME}1")
    assert load_cap_names(trial_root) == (f"{CAP_NAME}1", CAP_NAME)


def test_load_cap_names_base_unchanged_when_uuid_ends_in_digit(tmp_path: Path) -> None:
    # uuid8 may end in a decimal digit; the bare form must not be truncated.
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)  # CAP_NAME ends in "abcd1234"
    assert load_cap_names(trial_root) == (CAP_NAME,)


# ── check_trial_sync_data ───────────────────────────────────────


def test_check_trial_sync_data_complete(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    (trial_root / f"{CAP_NAME}.c3d").write_text("x")
    (trial_root / f"{CAP_NAME}.txt").write_text("y")
    status = check_trial_sync_data(_manifest_path(trial_root))
    assert status.complete
    assert status.c3d_present and status.txt_present
    assert status.c3d_missing is None and status.txt_missing is None


def test_check_trial_sync_data_missing_c3d(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    (trial_root / f"{CAP_NAME}.txt").write_text("y")
    status = check_trial_sync_data(_manifest_path(trial_root))
    assert not status.complete
    assert status.c3d_missing == f"{CAP_NAME}.c3d"
    assert status.txt_present and status.txt_missing is None


def test_check_trial_sync_data_both_missing(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    status = check_trial_sync_data(_manifest_path(trial_root))
    assert not status.complete
    assert status.c3d_missing == f"{CAP_NAME}.c3d"
    assert status.txt_missing == f"{CAP_NAME}.txt"


def test_check_trial_sync_data_no_cap(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    status = check_trial_sync_data(_manifest_path(trial_root))
    assert not status.has_cap
    assert not status.complete
    assert status.c3d_missing is None and status.txt_missing is None


def test_check_trial_sync_data_matches_take_suffix(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    (trial_root / f"{CAP_NAME}_001.c3d").write_text("x")
    (trial_root / f"{CAP_NAME}_001.txt").write_text("y")
    status = check_trial_sync_data(_manifest_path(trial_root))
    assert status.complete


def test_check_trial_sync_data_matches_take_index_no_underscore(tmp_path: Path) -> None:
    # Real XINGYING naming: .c3d carries the take index (no underscore), the
    # operator-named .txt uses the bare uuid8 form.
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, f"{CAP_NAME}1")
    (trial_root / f"{CAP_NAME}1.c3d").write_text("x")
    (trial_root / f"{CAP_NAME}.txt").write_text("y")
    status = check_trial_sync_data(_manifest_path(trial_root))
    assert status.complete
    assert status.c3d_present and status.txt_present


def test_check_trial_sync_data_missing_labels_use_base_for_txt(tmp_path: Path) -> None:
    # The missing-txt label must point at the bare uuid8 name, not the take name.
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, f"{CAP_NAME}1")
    status = check_trial_sync_data(_manifest_path(trial_root))
    assert status.c3d_missing == f"{CAP_NAME}1.c3d"
    assert status.txt_missing == f"{CAP_NAME}.txt"


def test_check_all_trial_sync_only_finalized(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    (trial_root / f"{CAP_NAME}.c3d").write_text("x")
    (trial_root / f"{CAP_NAME}.txt").write_text("y")
    manifest_path = _manifest_path(trial_root)
    finalized = SimpleNamespace(state="FINALIZED", manifest_path=manifest_path)
    nonfinalized = SimpleNamespace(state="RECORDING", manifest_path=manifest_path)
    statuses = check_all_trial_sync([finalized, nonfinalized])
    assert len(statuses) == 1
    assert statuses[0].complete


# ── sync_sidecar_files ──────────────────────────────────────────


def test_sync_sidecar_files_copies_exact_match_recursively(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    scan = tmp_path / "scan"
    (scan / "nested").mkdir(parents=True)
    (scan / "nested" / f"{CAP_NAME}.c3d").write_text("x")

    result = sync_sidecar_files(scan, [_manifest_path(trial_root)], ".c3d")

    assert result.scanned_files == 1
    assert result.copied_files == 1
    assert result.matched_files == 1
    assert (trial_root / f"{CAP_NAME}.c3d").exists()


def test_sync_sidecar_files_copies_take_suffix(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / f"{CAP_NAME}_001.c3d").write_text("x")

    result = sync_sidecar_files(scan, [_manifest_path(trial_root)], ".c3d")

    assert result.copied_files == 1
    assert (trial_root / f"{CAP_NAME}_001.c3d").exists()


def test_sync_sidecar_files_copies_take_index_no_underscore(tmp_path: Path) -> None:
    # Recorded capture name carries the take index; the scan dir holds the bare
    # .txt name — the pair must still match and copy.
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, f"{CAP_NAME}1")
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / f"{CAP_NAME}.txt").write_text("x")

    result = sync_sidecar_files(scan, [_manifest_path(trial_root)], ".txt")

    assert result.matched_files == 1
    assert result.copied_files == 1
    assert result.unmatched_files == ()
    assert (trial_root / f"{CAP_NAME}.txt").exists()


def test_sync_sidecar_files_skips_existing(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    (trial_root / f"{CAP_NAME}.c3d").write_text("already-here")
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / f"{CAP_NAME}.c3d").write_text("new")

    result = sync_sidecar_files(scan, [_manifest_path(trial_root)], ".c3d")

    assert result.skipped_existing == 1
    assert result.copied_files == 0
    assert (trial_root / f"{CAP_NAME}.c3d").read_text() == "already-here"


def test_sync_sidecar_files_reports_unmatched_and_no_cap(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    no_cap = tmp_path / "no_cap"
    no_cap.mkdir()
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / "unrelated.c3d").write_text("x")

    result = sync_sidecar_files(
        scan,
        [_manifest_path(trial_root), _manifest_path(no_cap)],
        ".c3d",
    )

    assert result.scanned_files == 1
    assert result.copied_files == 0
    assert result.matched_files == 0
    assert result.unmatched_files == (str(scan / "unrelated.c3d"),)
    assert any(Path(p).name == "no_cap" for p in result.targets_without_cap)


def test_sync_sidecar_files_case_insensitive_suffix(tmp_path: Path) -> None:
    trial_root = tmp_path / "trial"
    trial_root.mkdir()
    _write_trigger(trial_root, CAP_NAME)
    scan = tmp_path / "scan"
    scan.mkdir()
    (scan / f"{CAP_NAME}.C3D").write_text("x")

    result = sync_sidecar_files(scan, [_manifest_path(trial_root)], ".c3d")

    assert result.copied_files == 1
    assert (trial_root / f"{CAP_NAME}.C3D").exists()
