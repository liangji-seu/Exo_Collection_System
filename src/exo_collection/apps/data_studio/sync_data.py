"""XINGYING ``.cap`` → ``.c3d``/``.txt`` 同步数据的检查与导入（非 Qt 后端）。

Collector 只把 XINGYING 动捕的 ``.cap`` 文件名记进 trial（``raw/xingying_trigger.jsonl``
与 ``.exo/sync_manifest.json`` 的 ``capture_name``），``.cap`` 本体留在 XINGYING 工程目录。
动捕导出的 ``.c3d`` 与测力台导出的 ``.txt`` 是事后外部产物，需要按 ``.cap`` 同名拷回 trial
的 session 文件夹。本模块负责：检查每个 trial 是否已补齐这两个文件，以及从用户选择的扫描
目录递归拷贝匹配文件到对应 session 文件夹。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from exo_collection.domain.states import TrialState
from exo_collection.domain.xingying_trigger import (
    XINGYING_TRIGGER_RELATIVE_PATH,
    load_xingying_trigger_events,
)

_log = logging.getLogger(__name__)

SYNC_MANIFEST_RELATIVE_PATH = Path(".exo") / "sync_manifest.json"

_SIDECAR_EXTENSIONS = (".c3d", ".txt")


@dataclass(frozen=True, slots=True)
class SyncDataStatus:
    """One trial 的 ``.c3d``/``.txt`` 同步数据检查结果（只读）。"""

    manifest_path: Path
    trial_root: Path
    cap_names: tuple[str, ...]
    c3d_present: bool
    txt_present: bool
    c3d_missing: str | None
    txt_missing: str | None

    @property
    def has_cap(self) -> bool:
        return bool(self.cap_names)

    @property
    def complete(self) -> bool:
        return self.has_cap and self.c3d_present and self.txt_present


@dataclass(frozen=True, slots=True)
class SyncCopyResult:
    """一次「同步动捕/测力台数据」拷贝的结果汇总。"""

    extension: str
    scanned_files: int
    matched_files: int
    copied_files: int
    skipped_existing: int
    target_count: int
    unmatched_files: tuple[str, ...] = ()
    targets_without_cap: tuple[str, ...] = ()


def _trial_root_from_manifest_path(manifest_path: Path) -> Path:
    """Return the trial root from a manifest.json path (current + legacy layout)."""
    parent = Path(manifest_path).parent.resolve()
    if parent.name == ".exo":
        return parent.parent.resolve()
    return parent


def _sidecar_files(trial_root: Path, extension: str) -> tuple[Path, ...]:
    """List files directly inside the session folder with the given extension."""
    if not trial_root.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in trial_root.iterdir()
            if path.is_file() and path.suffix.casefold() == extension.casefold()
        )
    )


def _stem_matches(stem: str, cap_name: str) -> bool:
    """Match a sidecar file stem to a recorded ``.cap`` capture name.

    XINGYING appends a take index to the actual ``.cap`` filename, so an exported
    ``.c3d``/``.txt`` may be named ``{cap}`` or ``{cap}_001``.  The capture name
    ends in a unique 8-hex UUID, so a prefix match cannot cross trials.
    """
    folded_stem = stem.casefold()
    folded_cap = cap_name.casefold()
    return folded_stem == folded_cap or folded_stem.startswith(folded_cap + "_")


_TAKE_SUFFIX_RE = re.compile(r"(.*_[0-9a-fA-F]{8})\d+")


def _base_capture_name(cap_name: str) -> str:
    """Strip XINGYING's trailing take index from a recorded capture name.

    The Collector sends ``{subject}_{condition}_r{repeat}_{uuid8}`` to XINGYING,
    but XINGYING appends a take number to the real ``.cap``/``.c3d`` name (e.g.
    ``..._248024cb1``).  The operator-named ``.txt`` keeps the bare ``uuid8`` form
    (``..._248024cb``), so both variants must map to the same trial.  The ``uuid8``
    is 8 hex chars and may itself end in a digit, so the take index is anchored on
    the 8-char uuid rather than on "trailing digits".
    """
    name = str(cap_name or "").strip()
    if not name:
        return name
    match = _TAKE_SUFFIX_RE.fullmatch(name)
    return match.group(1) if match else name


def load_cap_names(trial_root: Path) -> tuple[str, ...]:
    """Read the recorded XINGYING capture names for one finalized trial.

    Prefers ``raw/xingying_trigger.jsonl`` (canonical, validated); falls back to
    ``.exo/sync_manifest.json`` when the trigger artifact is absent or unreadable.
    Returns ``()`` when no capture name was recorded.
    """
    names: list[str] = []
    trigger_path = trial_root / XINGYING_TRIGGER_RELATIVE_PATH
    if trigger_path.is_file():
        try:
            names = [event.capture_name for event in load_xingying_trigger_events(trigger_path)]
        except Exception as exc:
            _log.warning("读取 xingying_trigger 失败 %s: %s", trigger_path, exc)
    if not names:
        sync_manifest_path = trial_root / SYNC_MANIFEST_RELATIVE_PATH
        if sync_manifest_path.is_file():
            try:
                document = json.loads(sync_manifest_path.read_text(encoding="utf-8"))
                triggers = document.get("xingying_triggers") or []
                if isinstance(triggers, list):
                    names = [
                        str(item.get("capture_name") or "")
                        for item in triggers
                        if isinstance(item, dict)
                    ]
            except Exception as exc:
                _log.warning("读取 sync_manifest 失败 %s: %s", sync_manifest_path, exc)

    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        clean = str(name).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
        # XINGYING appends a take index to the real capture name; the operator-
        # named .txt uses the bare form.  Register both so a .c3d (with take)
        # and a .txt (without take) resolve to the same trial.
        base = _base_capture_name(clean)
        if base and base != clean and base not in seen:
            seen.add(base)
            result.append(base)
    return tuple(result)


def check_trial_sync_data(manifest_path: Path) -> SyncDataStatus:
    """Check presence of ``{cap}.c3d`` and ``{cap}.txt`` inside the trial root."""
    trial_root = _trial_root_from_manifest_path(manifest_path)
    cap_names = load_cap_names(trial_root)
    if not cap_names:
        return SyncDataStatus(
            manifest_path=Path(manifest_path),
            trial_root=trial_root,
            cap_names=(),
            c3d_present=False,
            txt_present=False,
            c3d_missing=None,
            txt_missing=None,
        )
    c3d_files = _sidecar_files(trial_root, ".c3d")
    txt_files = _sidecar_files(trial_root, ".txt")
    c3d_present = any(
        _stem_matches(candidate.stem, cap_name)
        for cap_name in cap_names
        for candidate in c3d_files
    )
    txt_present = any(
        _stem_matches(candidate.stem, cap_name)
        for cap_name in cap_names
        for candidate in txt_files
    )
    primary = cap_names[0]
    primary_base = _base_capture_name(primary)
    return SyncDataStatus(
        manifest_path=Path(manifest_path),
        trial_root=trial_root,
        cap_names=cap_names,
        c3d_present=c3d_present,
        txt_present=txt_present,
        c3d_missing=None if c3d_present else f"{primary}.c3d",
        txt_missing=None if txt_present else f"{primary_base}.txt",
    )


def check_all_trial_sync(records: Iterable[Any]) -> tuple[SyncDataStatus, ...]:
    """Check sync data for every FINALIZED management record.

    Accepts ``TrialManagementRecord`` via duck typing so this module stays free of
    a management import.  A single trial's failure is logged and skipped.
    """
    statuses: list[SyncDataStatus] = []
    for record in records:
        if getattr(record, "state", None) != TrialState.FINALIZED.value:
            continue
        manifest_path = getattr(record, "manifest_path", None)
        if manifest_path is None:
            continue
        try:
            statuses.append(check_trial_sync_data(Path(manifest_path)))
        except Exception as exc:
            _log.warning("同步数据检查失败 %s: %s", manifest_path, exc)
    return tuple(statuses)


def sync_sidecar_files(
    scan_root: Path,
    manifest_paths: Iterable[Path],
    extension: str,
) -> SyncCopyResult:
    """Recursively find ``*{extension}`` files and copy stem matches into trial roots.

    The mapping from recorded cap name → trial root is rebuilt from each finalized
    trial's own ``.cap`` name, so the copy never relies on the caller having chosen
    the correct target directories.
    """
    if extension.casefold() not in {".c3d", ".txt"}:
        raise ValueError(f"不支持的同步文件扩展名: {extension}")

    cap_to_trial: dict[str, Path] = {}
    targets_without_cap: list[str] = []
    for manifest_path in manifest_paths:
        trial_root = _trial_root_from_manifest_path(Path(manifest_path))
        cap_names = load_cap_names(trial_root)
        if not cap_names:
            targets_without_cap.append(str(trial_root))
            continue
        for cap_name in cap_names:
            cap_to_trial.setdefault(cap_name.casefold(), trial_root)

    scan = Path(scan_root)
    if not scan.is_dir():
        raise ValueError(f"扫描目录不存在或不是目录: {scan}")

    scanned_files = 0
    matched_files = 0
    copied_files = 0
    skipped_existing = 0
    unmatched_files: list[str] = []

    for path in sorted(scan.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.casefold() != extension.casefold():
            continue
        scanned_files += 1
        folded_stem = path.stem.casefold()
        trial_root = cap_to_trial.get(folded_stem)
        if trial_root is None:
            for folded_cap, candidate_root in cap_to_trial.items():
                if folded_stem.startswith(folded_cap + "_"):
                    trial_root = candidate_root
                    break
        if trial_root is None:
            unmatched_files.append(str(path))
            continue
        matched_files += 1
        destination = trial_root / path.name
        if destination.exists():
            skipped_existing += 1
            continue
        shutil.copy2(path, destination)
        copied_files += 1

    return SyncCopyResult(
        extension=extension,
        scanned_files=scanned_files,
        matched_files=matched_files,
        copied_files=copied_files,
        skipped_existing=skipped_existing,
        target_count=len(cap_to_trial),
        unmatched_files=tuple(unmatched_files),
        targets_without_cap=tuple(targets_without_cap),
    )


__all__ = [
    "SYNC_MANIFEST_RELATIVE_PATH",
    "SyncCopyResult",
    "SyncDataStatus",
    "check_all_trial_sync",
    "check_trial_sync_data",
    "load_cap_names",
    "sync_sidecar_files",
]
