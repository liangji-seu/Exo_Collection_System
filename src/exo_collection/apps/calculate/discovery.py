"""Exo Calculate 的 Session 发现、静态推荐与只读输入检查。

复用 Data Studio 的存储布局与 Manifest（``exo_collection.storage``）以及
C3D/TXT 外部文件配对（``data_studio.sync_data``），不复制一套互不兼容的目录
解析。C3D/Gaitway 的数值读取复用 ``opensim_joint_moment_pipeline``。

目录深度不固定（受试者下可能出现 ``d1`` 等 day 子目录），因此一律以
``iter_finalized_manifest_paths`` 递归定位 ``.exo/manifest.json``，绝不按
固定层级硬编码。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from exo_collection.apps.calculate.models import (
    InputCheckReport,
    SessionFiles,
    SessionRecord,
)
from exo_collection.apps.data_studio.sync_data import load_cap_names
from exo_collection.storage.layout import iter_finalized_manifest_paths

_log = logging.getLogger(__name__)

_SIDECAR_SUFFIXES = (".c3d", ".txt")


def _trial_root_from_manifest_path(manifest_path: Path) -> Path:
    parent = Path(manifest_path).resolve().parent
    return parent.parent.resolve() if parent.name == ".exo" else parent


def _pick_sidecar(session_dir: Path, suffix: str, cap_names: tuple[str, ...]) -> Path | None:
    """在 session 目录内挑选匹配 capture 名的 ``.c3d``/``.txt``。"""
    candidates = sorted(
        p
        for p in session_dir.iterdir()
        if p.is_file() and p.suffix.casefold() == suffix.casefold()
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def matches(stem: str, cap: str) -> bool:
        folded_stem = stem.casefold()
        folded_cap = cap.casefold()
        # XINGYING 会追加 take 号，如 `{cap}_001`；cap 名以 8 位 hex 结尾，前缀
        # 匹配不会跨 trial。
        return folded_stem == folded_cap or folded_stem.startswith(folded_cap + "_")

    for cap in cap_names:
        for candidate in candidates:
            if matches(candidate.stem, cap):
                return candidate
    return candidates[0]


def _read_manifest_fields(manifest_path: Path) -> dict | None:
    try:
        document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _log.warning("读取 manifest 失败 %s: %s", manifest_path, exc)
        return None
    if not isinstance(document, dict):
        return None
    return document


def discover_sessions(data_root: str | Path) -> list[SessionRecord]:
    """扫描数据根下所有已最终化的 Session（按 subject + 时间排序）。"""
    records: list[SessionRecord] = []
    for manifest_path in iter_finalized_manifest_paths(data_root):
        document = _read_manifest_fields(manifest_path)
        if document is None:
            continue
        try:
            record = _record_from_document(manifest_path, document)
        except Exception as exc:  # 单个坏 manifest 不拖垮整次扫描
            _log.warning("解析 Session 失败 %s: %s", manifest_path, exc)
            continue
        if record is not None:
            records.append(record)
    records.sort(key=lambda r: (r.subject_code, r.project_code, r.condition_code,
                                r.repeat_index, r.started_at_utc))
    return records


def _record_from_document(manifest_path: Path, document: dict) -> SessionRecord | None:
    subject_code = str(document.get("subject_code") or "").strip()
    if not subject_code:
        return None
    session_dir = _trial_root_from_manifest_path(manifest_path)
    condition = document.get("condition") or {}
    parameters = condition.get("parameters") or {}
    cap_names = load_cap_names(session_dir)

    files = SessionFiles(
        c3d_path=_pick_sidecar(session_dir, ".c3d", cap_names),
        txt_path=_pick_sidecar(session_dir, ".txt", cap_names),
        mocap_h5_path=(session_dir / "mocap.h5") if (session_dir / "mocap.h5").is_file() else None,
        imu_h5_path=(session_dir / "imu.h5") if (session_dir / "imu.h5").is_file() else None,
    )
    timing = document.get("timing") or {}
    return SessionRecord(
        manifest_path=Path(manifest_path),
        session_dir=session_dir,
        session_name=session_dir.name,
        subject_code=subject_code,
        subject_uuid=str(document.get("subject_uuid") or ""),
        project_code=str(document.get("project_code") or ""),
        project_name=str(document.get("project_name") or ""),
        condition_code=str(condition.get("condition_code") or ""),
        condition_name=str(condition.get("condition_name") or ""),
        condition_level=condition.get("condition_level"),
        repeat_index=int(condition.get("repeat_index") or 1),
        trial_uuid=str(document.get("trial_uuid") or ""),
        session_uuid=str(document.get("session_uuid") or ""),
        state=str(document.get("state") or ""),
        started_at_utc=str(timing.get("started_at_utc") or ""),
        condition_parameters=dict(parameters),
        files=files,
    )


def recommend_static_for_subject(
    subject_code: str, sessions: list[SessionRecord]
) -> SessionRecord | None:
    """为某受试者推荐静态标定 Session。

    优先 ``STAND`` 工况、有 C3D、日期最近；宁可返回 ``None`` 也不在不同受试者间
    误用旧静态模型。
    """
    candidates = [
        s
        for s in sessions
        if s.subject_code == subject_code
        and s.is_stand
        and s.files.c3d_path is not None
    ]
    if not candidates:
        return None
    # 日期最近优先（started_at_utc 为 ISO 字符串，字典序即时间序）。
    candidates.sort(key=lambda s: s.started_at_utc, reverse=True)
    return candidates[0]


def recommend_static_session(
    dynamic: SessionRecord, sessions: list[SessionRecord]
) -> SessionRecord | None:
    """为动态 Session 推荐静态标定 Session（兼容旧接口）。"""
    return recommend_static_for_subject(dynamic.subject_code, sessions)


def check_inputs(
    dynamic: SessionRecord, static: SessionRecord | None
) -> InputCheckReport:
    """只读扫描动态/静态输入，返回可展示的检查报告（不 import OpenSim）。"""
    from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path

    ensure_pipeline_on_path()
    from pipeline.c3d.reader import read_c3d  # noqa: E402
    from pipeline.opensim_io.build_trc import extract_hh19  # noqa: E402

    problems: list[str] = []
    warnings: list[str] = []
    report_kwargs: dict = {
        "subject_code": dynamic.subject_code,
        "dynamic_session": dynamic.session_name,
        "static_session": static.session_name if static else None,
    }

    # 缺失输入
    missing = list(dynamic.files.missing())
    if missing:
        problems.append("动态 Session 缺失输入：" + ", ".join(missing))
    if static is None:
        warnings.append("未选择静态标定 Session，无法缩放模型")

    # 动态 C3D
    if dynamic.files.c3d_path is not None:
        try:
            data = read_c3d(dynamic.files.c3d_path)
            names, _ = extract_hh19(data)
            report_kwargs.update(
                dynamic_c3d_rate_hz=float(data.point_rate_hz),
                dynamic_c3d_duration_s=float(data.n_frames) / float(data.point_rate_hz),
                dynamic_hh19_markers=len(names),
            )
            if len(names) != 15:
                warnings.append(f"动态 HH19 实测 marker 数量为 {len(names)}（预期 15）")
        except Exception as exc:
            problems.append(f"动态 C3D 读取失败：{exc}")

    # 静态 C3D
    if static is not None and static.files.c3d_path is not None:
        try:
            data = read_c3d(static.files.c3d_path)
            names, _ = extract_hh19(data)
            report_kwargs.update(
                static_c3d_rate_hz=float(data.point_rate_hz),
                static_c3d_duration_s=float(data.n_frames) / float(data.point_rate_hz),
                static_hh19_markers=len(names),
            )
            if len(names) != 19:
                warnings.append(f"静态 HH19 实测 marker 数量为 {len(names)}（预期 19）")
        except Exception as exc:
            problems.append(f"静态 C3D 读取失败：{exc}")

    # Gaitway TXT
    if dynamic.files.txt_path is not None:
        try:
            from pipeline.gaitway import read_gaitway_ascii  # noqa: E402

            gaitway = read_gaitway_ascii(dynamic.files.txt_path)
            columns = set(gaitway.columns)
            bilateral = {"FzL(N)", "FzR(N)", "FyL(N)", "FyR(N)", "FxL(N)", "FxR(N)",
                         "CoPxL(m)", "CoPyL(m)", "CoPxR(m)", "CoPyR(m)"} <= columns
            report_kwargs.update(
                gaitway_rate_hz=gaitway.sample_rate_hz,
                gaitway_has_bilateral_columns=bilateral,
            )
            if not bilateral:
                problems.append("Gaitway TXT 缺少左右脚分解列（FzL/FzR/CoPx/CoPy）")
        except Exception as exc:
            problems.append(f"Gaitway TXT 读取失败：{exc}")

    # mocap.h5 / imu.h5 采样率（轻量读元数据）
    if dynamic.files.mocap_h5_path is not None:
        report_kwargs["mocap_h5_rate_hz"] = _h5_rate(dynamic.files.mocap_h5_path)
    if dynamic.files.imu_h5_path is not None:
        report_kwargs["imu_h5_rate_hz"] = _h5_rate(dynamic.files.imu_h5_path)

    return InputCheckReport(problems=tuple(problems), warnings=tuple(warnings), **report_kwargs)


def _h5_rate(path: Path) -> float | None:
    """轻量读取 HDF5 采样率（优先 metadata 标称值，退回 host_monotonic_ns 差分）。"""
    import h5py
    import numpy as np

    try:
        with h5py.File(path, "r") as handle:
            device = handle["metadata/device"][()]
            if isinstance(device, (bytes, bytearray)):
                device = json.loads(device.decode("utf-8"))
            nominal = device.get("frame_rate_hz") or device.get("nominal_rate_hz")
            if nominal:
                return float(nominal)
            times = handle["samples/host_monotonic_ns"][:2000].astype(np.float64)
            if times.size > 1:
                period_s = float(np.median(np.diff(times))) / 1e9
                return float(1.0 / period_s) if period_s > 0 else None
    except Exception as exc:
        _log.warning("读取 HDF5 采样率失败 %s: %s", path, exc)
    return None


__all__ = [
    "SessionFiles",
    "SessionRecord",
    "check_inputs",
    "discover_sessions",
    "recommend_static_for_subject",
    "recommend_static_session",
]
