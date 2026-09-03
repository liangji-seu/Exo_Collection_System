"""历史 run 管理：列出当前 Session 的 ``run_*``，判定 STALE，供 UI 回放（prompt6 §3.10）。

纯逻辑、不 import Qt / OpenSim：直接读 ``derived/opensim/run_*/manifest.json`` 与
``result.json`` 两个小 JSON 文件，据此给出每个 run 的运行时间、静态 Session、
同步方法/offset、分析区间、QC 状态，并对照 ``manifest["inputs"]`` 里的输入指纹
（path/size/mtime_ns/sha256）判断输入是否已变化（``STALE_INPUTS``）。

旧 run 一律只读：本模块只读不写，绝不覆盖或删除历史 run（§3.10 第 5 条）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str | None:
    """全文件 SHA-256；文件不存在返回 None。"""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HistoryRun:
    """一次历史 run 的可展示摘要（JSON 可读，纯数据）。"""

    run_dir: Path
    run_id: str
    created_utc: str | None
    state: str                 # "completed" | "cancelled" | "failed"
    qc_status: str | None      # PASS / WARN / FAIL（仅 completed 时有意义）
    static_session: str | None
    sync_method: str | None
    sync_offset_s: float | None
    sync_confidence: str | None
    analysis_window: tuple[float, float] | None
    stale: bool
    stale_reason: str | None
    viewer_dir: Path | None
    n_frames: int | None

    @property
    def replayable(self) -> bool:
        return self.viewer_dir is not None and Path(self.viewer_dir).is_dir()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _run_created_utc(run_dir: Path) -> str | None:
    """从 ``run_YYYYMMDD_HHMMSS_ffff`` 目录名解析运行时间。"""
    name = run_dir.name
    stamp = name[len("run_"):] if name.startswith("run_") else ""
    try:
        return datetime.strptime(stamp[:15], "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return None


def _stale_check(
    recorded_inputs: dict[str, Any], current_inputs: dict[str, str | Path] | None
) -> tuple[bool, str | None]:
    """对照输入指纹判断是否 STALE。

    大小 + mtime_ns 一致视为未变（跳过全文件哈希，避免每次刷新都读大 C3D）；
    任一不一致才重算 SHA-256 做最终判定。
    """
    if current_inputs is None:
        return False, None
    reasons: list[str] = []
    for key, current in current_inputs.items():
        recorded = recorded_inputs.get(key) or {}
        rec_sha = recorded.get("sha256")
        if rec_sha is None:
            # 旧 manifest 无指纹，无法判定，不误标 stale。
            continue
        cur = Path(current)
        if not cur.is_file():
            reasons.append(f"{key} 缺失")
            continue
        try:
            stat = cur.stat()
            cur_size = int(stat.st_size)
            cur_mtime = int(stat.st_mtime_ns)
        except OSError:
            reasons.append(f"{key} 不可读")
            continue
        if recorded.get("size_bytes") == cur_size and recorded.get("mtime_ns") == cur_mtime:
            continue
        if _file_sha256(cur) != rec_sha:
            reasons.append(f"{key} 已变化")
    if reasons:
        return True, "；".join(reasons)
    return False, None


def _load_one_run(
    run_dir: Path, current_inputs: dict[str, str | Path] | None
) -> HistoryRun:
    manifest = _read_json(run_dir / "manifest.json")
    result = _read_json(run_dir / "result.json")
    cancel_flag = (run_dir / "cancel.flag").exists()

    if result is not None:
        state = "completed"
    elif cancel_flag:
        state = "cancelled"
    else:
        state = "failed"

    qc = result.get("qc") if result is not None else None
    qc_status = qc.get("status") if isinstance(qc, dict) else None

    sync = (manifest or {}).get("sync") or {}
    static_path = (((manifest or {}).get("inputs") or {}).get("static_c3d") or {}).get("path")
    static_session = Path(static_path).parent.name if static_path else None

    viewer_dir: Path | None = None
    if result is not None:
        files = result.get("files") or {}
        vd = files.get("viewer_dir") or (result.get("viewer") or {}).get("viewer_dir")
        if vd:
            viewer_dir = Path(vd)
    elif (run_dir / "viewer").is_dir():
        viewer_dir = run_dir / "viewer"

    analysis = (manifest or {}).get("analysis_time_range_s")
    analysis_window = tuple(analysis) if isinstance(analysis, (list, tuple)) and len(analysis) == 2 else None

    stale, stale_reason = _stale_check((manifest or {}).get("inputs") or {}, current_inputs)

    return HistoryRun(
        run_dir=run_dir,
        run_id=run_dir.name,
        created_utc=_run_created_utc(run_dir),
        state=state,
        qc_status=qc_status,
        static_session=static_session,
        sync_method=sync.get("method"),
        sync_offset_s=sync.get("gaitway_offset_s"),
        sync_confidence=sync.get("confidence"),
        analysis_window=analysis_window,
        stale=stale,
        stale_reason=stale_reason,
        viewer_dir=viewer_dir,
        n_frames=(result.get("viewer") or {}).get("n_frames") if result is not None else None,
    )


def list_history_runs(
    opensim_dir: str | Path,
    current_inputs: dict[str, str | Path] | None = None,
) -> list[HistoryRun]:
    """列出 ``opensim_dir`` 下的全部 ``run_*``，按运行时间倒序。

    ``current_inputs`` 的键与 ``manifest["inputs"]`` 一致（static_c3d /
    dynamic_c3d / gaitway_txt），用于 STALE 判定；传 None 则不判定。
    """
    root = Path(opensim_dir)
    if not root.is_dir():
        return []
    runs: list[HistoryRun] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        runs.append(_load_one_run(run_dir, current_inputs))
    runs.sort(key=lambda r: r.created_utc or "", reverse=True)
    return runs


__all__ = ["HistoryRun", "list_history_runs"]
