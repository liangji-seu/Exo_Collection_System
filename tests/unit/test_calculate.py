"""Exo Calculate 的纯逻辑单元测试（状态机、DTO、设置持久化）。

不 import OpenSim、不读真实数据文件；只验证状态迁移、DTO 约定与设置回写。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from exo_collection.apps.calculate.controller import (  # noqa: E402
    CalculateController,
    SyncOutcome,
)
from exo_collection.apps.calculate.models import (  # noqa: E402
    InputCheckReport,
    ProcessingConfig,
    ProcessingState,
    SessionFiles,
    SessionRecord,
    SyncMethod,
    SyncPeakPair,
    SyncResult,
)
from exo_collection.apps.calculate.window import (  # noqa: E402
    CalculateWindow,
    _sync_result_from_raw,
)
from exo_collection.apps.calculate.workers import _parse_event  # noqa: E402
from exo_collection.configuration.app_settings import SharedAppSettings  # noqa: E402


def _file_settings(path: Path) -> SharedAppSettings:
    return SharedAppSettings(QSettings(str(path), QSettings.Format.IniFormat))


def _make_session(subject: str = "003", condition: str = "WALK_STEADY_1P00") -> SessionRecord:
    return SessionRecord(
        manifest_path=Path("/tmp") / "manifest.json",
        session_dir=Path("/tmp") / subject,
        session_name="session1_20260101_000000",
        subject_code=subject,
        subject_uuid="subj-uuid",
        project_code="F_STEADY",
        project_name="F_STEADY",
        condition_code=condition,
        condition_name=condition,
        condition_level=None,
        repeat_index=1,
        trial_uuid="trial-uuid",
        session_uuid="session-uuid",
        state="FINALIZED",
        started_at_utc="2026-01-01T00:00:00",
    )


# --------------------------------------------------------------------------
# 状态机：退出码 0 ≠ QC PASS
# --------------------------------------------------------------------------
def test_state_enum_distinguishes_process_success_from_qc() -> None:
    # 12 个状态齐全；FAILED（进程异常）与 COMPLETED_QC_*（QC 结论）不混用。
    expected = {
        "NOT_SCANNED", "INPUT_INVALID", "READY_FOR_SYNC", "SYNC_NEEDS_REVIEW",
        "SYNC_CONFIRMED", "PROCESSING", "COMPLETED_QC_PASS", "COMPLETED_QC_WARN",
        "COMPLETED_QC_FAIL", "CANCELLED", "FAILED", "STALE_INPUTS",
    }
    assert {s.value for s in ProcessingState} == expected


def test_controller_sync_state_flow() -> None:
    controller = CalculateController(Path("/tmp/data"))
    states: list[str] = []
    controller.state_changed.connect(states.append)

    dynamic = _make_session()
    controller.set_dynamic(dynamic)
    assert controller.state == ProcessingState.NOT_SCANNED

    controller.set_input_report(
        InputCheckReport(subject_code="003", dynamic_session="s", static_session=None)
    )
    assert controller.state == ProcessingState.READY_FOR_SYNC

    # 高可信自动同步 → 直接确认
    controller.set_sync(
        SyncResult(
            c3d_start_in_mocap_h5_frame=183,
            c3d_h5_match_rms_mm=0.0,
            c3d_h5_overlap_frames=5801,
            gaitway_offset_s=5.835,
            confidence="HIGH",
        )
    )
    assert controller.state == ProcessingState.SYNC_CONFIRMED

    # 低可信 → 需要人工复核
    controller.set_sync(
        SyncResult(
            c3d_start_in_mocap_h5_frame=183,
            c3d_h5_match_rms_mm=0.0,
            c3d_h5_overlap_frames=5801,
            gaitway_offset_s=5.835,
            confidence="LOW",
        )
    )
    assert controller.state == ProcessingState.SYNC_NEEDS_REVIEW


def test_controller_qc_status_is_not_process_failure() -> None:
    controller = CalculateController(Path("/tmp/data"))
    controller.mark_processing()
    assert controller.state == ProcessingState.PROCESSING
    controller.mark_completed("WARN")
    assert controller.state == ProcessingState.COMPLETED_QC_WARN
    # QC 通过不等于进程「成功退出」的 PASS；FAIL 是 QC 结论而非崩溃。
    controller.mark_completed("FAIL")
    assert controller.state == ProcessingState.COMPLETED_QC_FAIL
    controller.mark_failed()
    assert controller.state == ProcessingState.FAILED


def test_controller_switching_subject_clears_static() -> None:
    controller = CalculateController(Path("/tmp/data"))
    dynamic = _make_session(subject="001")
    controller.set_dynamic(dynamic)
    controller.set_static(_make_session(subject="001", condition="STAND"))
    assert controller.static is not None

    # 切到另一个受试者，旧静态模型不能误用。
    controller.set_dynamic(_make_session(subject="002"))
    assert controller.static is None


def test_sync_result_high_confidence_and_manual_flags() -> None:
    high = SyncResult(
        c3d_start_in_mocap_h5_frame=183,
        c3d_h5_match_rms_mm=0.0,
        c3d_h5_overlap_frames=5801,
        gaitway_offset_s=5.835,
        confidence="HIGH",
    )
    assert high.high_confidence
    manual = SyncResult(
        c3d_start_in_mocap_h5_frame=183,
        c3d_h5_match_rms_mm=0.0,
        c3d_h5_overlap_frames=5801,
        gaitway_offset_s=5.855,
        confidence="LOW",
        manual=True,
    )
    assert manual.manual and not manual.high_confidence


def test_sync_result_from_raw_converts_dict() -> None:
    raw = {
        "c3d_start_in_mocap_h5_frame": 183,
        "c3d_h5_match_rms_mm": 0.0,
        "c3d_h5_overlap_frames": 5801,
        "median_offset_s": 5.834848,
        "gaitway_offset_s": 5.854848,
        "confidence": "HIGH",
        "n_pairs": 5,
        "drift_ppm": None,
        "offsets_s": [5.804848, 5.834848, 5.836515, 5.843515, 5.810515],
        "peak_pairs": [
            {"index": 1, "imu_time_on_c3d_s": 9.727152, "gaitway_time_s": 15.532},
            {"index": 2, "imu_time_on_c3d_s": 10.452152, "gaitway_time_s": 16.287},
        ],
    }
    result = _sync_result_from_raw(raw, manual=False)
    assert result.gaitway_offset_s == pytest.approx(5.854848)
    assert result.median_offset_s == pytest.approx(5.834848)
    assert len(result.peak_pairs) == 2
    assert result.peak_pairs[0].offset_s == pytest.approx(15.532 - 9.727152)
    assert result.drift_ppm is None


def test_sync_outcome_from_dict_extracts_fields() -> None:
    outcome = SyncOutcome.from_dict(
        {
            "gaitway_offset_s": 5.854848,
            "confidence": "MEDIUM",
            "n_pairs": 4,
            "mad_s": 0.12,
            "drift_ppm": None,
        }
    )
    assert outcome.confidence == "MEDIUM"
    assert outcome.n_pairs == 4
    assert outcome.mad_s == pytest.approx(0.12)
    assert outcome.drift_ppm is None


# --------------------------------------------------------------------------
# §3.2 同步确认门禁：method / can_solve / confirm / expert-forced
# --------------------------------------------------------------------------
def test_sync_method_enum_values() -> None:
    assert {m.value for m in SyncMethod} == {
        "AUTO_HIGH", "MANUAL_PAIRED", "EXPERT_FORCED",
    }


def test_sync_result_is_expert_forced() -> None:
    auto = SyncResult(
        c3d_start_in_mocap_h5_frame=183,
        c3d_h5_match_rms_mm=0.0,
        c3d_h5_overlap_frames=5801,
        gaitway_offset_s=5.835,
        confidence="HIGH",
    )
    assert not auto.is_expert_forced
    forced = SyncResult(
        c3d_start_in_mocap_h5_frame=183,
        c3d_h5_match_rms_mm=0.0,
        c3d_h5_overlap_frames=5801,
        gaitway_offset_s=3.0,
        confidence="LOW",
        method=SyncMethod.EXPERT_FORCED,
    )
    assert forced.is_expert_forced


def test_controller_medium_auto_sync_is_not_solvable() -> None:
    controller = CalculateController(Path("/tmp/data"))
    controller.set_dynamic(_make_session())
    controller.set_input_report(
        InputCheckReport(subject_code="003", dynamic_session="s", static_session=None)
    )
    controller.set_sync(
        SyncResult(
            c3d_start_in_mocap_h5_frame=183,
            c3d_h5_match_rms_mm=0.3,
            c3d_h5_overlap_frames=5801,
            gaitway_offset_s=5.835,
            confidence="MEDIUM",
        )
    )
    assert controller.state == ProcessingState.SYNC_NEEDS_REVIEW
    assert not controller.can_solve

    controller.confirm_sync(operator_type="operator", note="人工复核通过")
    assert controller.state == ProcessingState.SYNC_CONFIRMED
    assert controller.can_solve
    assert controller.confirm_meta is not None
    assert controller.confirm_meta["operator_type"] == "operator"
    assert controller.confirm_meta["note"] == "人工复核通过"


def test_controller_expert_forced_is_solvable_and_not_pass() -> None:
    controller = CalculateController(Path("/tmp/data"))
    controller.set_dynamic(_make_session())
    controller.set_expert_forced(3.0, note="设备时钟无峰证据")
    assert controller.state == ProcessingState.SYNC_CONFIRMED
    assert controller.can_solve
    assert controller.sync_method is SyncMethod.EXPERT_FORCED
    assert controller.sync is not None and controller.sync.is_expert_forced
    assert controller.sync.confidence == "LOW"


def test_save_sync_calibration_schema_1_1_0(tmp_path: Path) -> None:
    from pipeline.synchronization.sync import save_sync_calibration

    c3d = tmp_path / "a.c3d"
    c3d.write_bytes(b"\x00" * 16)
    txt = tmp_path / "a.txt"
    txt.write_text("col\n1\n", encoding="utf-8")

    out = save_sync_calibration(
        tmp_path / "derived",
        {"gaitway_offset_s": 5.835, "confidence": "HIGH", "n_pairs": 5},
        inputs={"c3d": c3d, "gaitway_txt": txt},
        operator="auto",
        method="AUTO_HIGH",
        dynamic_session_uuid="session-uuid",
        trial_uuid="trial-uuid",
        auto_candidate={"c3d_start_in_mocap_h5_frame": 183},
    )

    import json

    doc = json.loads((out).read_text(encoding="utf-8"))
    assert doc["schema_version"] == "1.1.0"
    assert doc["method"] == "AUTO_HIGH"
    assert doc["dynamic_session_uuid"] == "session-uuid"
    assert doc["trial_uuid"] == "trial-uuid"
    assert doc["auto_candidate"]["c3d_start_in_mocap_h5_frame"] == 183
    assert doc["time_direction_convention"] == "t_gaitway = t_c3d + gaitway_offset_s"
    assert doc["coordinate_convention"] == "offset = t_gaitway - t_host"
    # 输入指纹：大小 + SHA-256，缺一不可（用于检测旧标定失效）。
    assert doc["inputs"]["c3d"]["size_bytes"] == 16
    assert len(doc["inputs"]["c3d"]["sha256"]) == 64
    assert doc["result"]["gaitway_offset_s"] == 5.835


# --------------------------------------------------------------------------
# §3.4 防止异步任务串 Session（operation token）
# --------------------------------------------------------------------------
def test_operation_token_switch_invalidates_stale() -> None:
    from exo_collection.apps.calculate.operation import OperationContext

    controller = CalculateController(Path("/tmp/data"))
    controller.set_dynamic(_make_session(subject="003"))
    controller.set_static(_make_session(subject="003", condition="STAND"))

    ctx = controller.begin_operation("sync")
    assert controller.is_current_operation(ctx)
    assert isinstance(ctx, OperationContext)
    assert ctx.kind == "sync"
    assert ctx.dynamic_session_uuid == "session-uuid"
    assert ctx.dynamic_trial_uuid == "trial-uuid"
    assert ctx.static_session_uuid == "session-uuid"
    assert ctx.operation_id >= 1

    # 切到另一个受试者 → 旧任务上下文立即失效。
    controller.set_dynamic(_make_session(subject="004"))
    assert not controller.is_current_operation(ctx)
    assert controller.active_operation is None

    # 新任务自增 op id，且只属于新 Session。
    ctx2 = controller.begin_operation("prep", run_dir=Path("/tmp/run_b"))
    assert ctx2.operation_id > ctx.operation_id
    assert controller.is_current_operation(ctx2)
    assert ctx2.dynamic_session_uuid == controller.dynamic.session_uuid
    assert ctx2.run_dir == Path("/tmp/run_b")

    controller.finish_operation(ctx2)
    assert controller.active_operation is None


def test_operation_finish_stale_does_not_clear_new() -> None:
    controller = CalculateController(Path("/tmp/data"))
    controller.set_dynamic(_make_session())
    a = controller.begin_operation("sync")
    b = controller.begin_operation("prep")
    # 过期的 finish 绝不能清掉更新任务的上下文。
    controller.finish_operation(a)
    assert controller.active_operation is b
    controller.finish_operation(b)
    assert controller.active_operation is None


def test_window_task_guard_and_input_lock(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    settings = _file_settings(tmp_path / "shared.ini")
    window = CalculateWindow(tmp_path, settings)
    controller = window._controller
    controller.set_dynamic(_make_session("003"))

    ctx = window._begin_task("sync")
    assert ctx is not None
    assert window._busy
    assert window._busy_kind == "sync"
    assert not window._selector.isEnabled()
    assert not window._choose_root_button.isEnabled()

    # 已有任务运行中，禁止再启动（防重复点击 §3.4 第 5 条）。
    assert window._begin_task("prep") is None

    # 模拟旧回调晚到：切 Session 后旧 ctx 被判过期丢弃。
    controller.set_dynamic(_make_session("004"))
    assert not window._is_current(ctx)

    window._end_task(ctx)
    assert not window._busy
    assert window._selector.isEnabled()
    assert window._choose_root_button.isEnabled()


# --------------------------------------------------------------------------
# 设置持久化：OpenSim 子环境
# --------------------------------------------------------------------------
def test_opensim_python_executable_round_trips(tmp_path: Path) -> None:
    settings = _file_settings(tmp_path / "shared.ini")
    assert settings.opensim_python_executable is None

    exe = tmp_path / "envs" / "opensim" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    settings.set_opensim_python_executable(exe)

    restored = _file_settings(tmp_path / "shared.ini")
    assert restored.opensim_python_executable == exe.resolve()

    restored.set_opensim_python_executable(None)
    assert _file_settings(tmp_path / "shared.ini").opensim_python_executable is None


# --------------------------------------------------------------------------
# 输入 DTO
# --------------------------------------------------------------------------
def test_session_files_missing_and_has_dynamic_inputs(tmp_path: Path) -> None:
    complete = SessionFiles(
        c3d_path=Path("a.c3d"),
        txt_path=Path("a.txt"),
        mocap_h5_path=Path("mocap.h5"),
        imu_h5_path=Path("imu.h5"),
    )
    assert complete.has_dynamic_inputs
    assert complete.missing() == ()

    partial = SessionFiles(c3d_path=Path("a.c3d"))
    assert not partial.has_dynamic_inputs
    assert "Gaitway TXT" in partial.missing()
    assert "mocap.h5" in partial.missing()


def test_session_record_is_stand_detection() -> None:
    assert _make_session(condition="STAND").is_stand
    assert not _make_session(condition="WALK_STEADY_1P00").is_stand


# --------------------------------------------------------------------------
# OpenSim 子进程 JSON-Lines 解析（_parse_event）
# --------------------------------------------------------------------------
def test_parse_event_stage() -> None:
    ev = _parse_event('{"event":"stage","stage":"ik","message":"动态 IK"}')
    assert ev == {"event": "stage", "stage": "ik", "message": "动态 IK"}


def test_parse_event_non_json_returns_none() -> None:
    assert _parse_event("[info] Running ScaleTool...") is None
    assert _parse_event("Some unstructured log line") is None


def test_parse_event_json_without_event_key_returns_none() -> None:
    assert _parse_event('{"foo": 1}') is None
    assert _parse_event("[1, 2, 3]") is None


def test_parse_event_blank_and_leading_whitespace() -> None:
    assert _parse_event("") is None
    assert _parse_event("   ") is None
    # 前导空白后紧跟 JSON 仍可解析（真实子进程输出可能带缩进）。
    ev = _parse_event('   {"event":"cancelled"}')
    assert ev == {"event": "cancelled"}


# --------------------------------------------------------------------------
# 初步 QC：App 采用子进程同一结论（§3.3），绝不另算一套不同结果
# --------------------------------------------------------------------------
def test_preliminary_qc_uses_subprocess_qc() -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(mass_kg=80.0)
    # 子进程已算好的版本化 qc（含同步/力覆盖）是权威结论；App 直接采用，
    # 即使本地 marker/ID 数值另会给出不同结论，也不另算一套。
    payload = {
        "marker_qc_overall": {"rms_mean_cm": 1.0},
        "id_qc": {"residual_force": {"rms_N": 50.0}},
        "qc": {"status": "WARN", "summary": "同步不唯一，已人工确认"},
    }
    status, detail = CalculateWindow._preliminary_qc(payload, config)
    assert status == "WARN"
    assert detail == "同步不唯一，已人工确认"


def test_preliminary_qc_falls_back_when_qc_missing() -> None:
    from types import SimpleNamespace

    config = SimpleNamespace(mass_kg=80.0)
    # 无 qc 块（旧结果/异常）：本地兜底重算，缺同步质量 → 保守判 FAIL，绝不 PASS。
    status, detail = CalculateWindow._preliminary_qc({}, config)
    assert status == "FAIL"
    assert "缺失" in detail


# --------------------------------------------------------------------------
# pipeline 路径解析：源码运行 vs PyInstaller 冻结（_MEIPASS）
# --------------------------------------------------------------------------
def test_pipeline_root_source_layout() -> None:
    import sys as _sys

    from exo_collection.apps.calculate._pipeline import pipeline_root

    # 源码运行：_pipeline.py 位于 <repo>/src/exo_collection/apps/calculate/，
    # parents[4] = 仓库根。
    root = pipeline_root()
    assert root.name == "opensim_joint_moment_pipeline"
    assert (root / "pipeline" / "__init__.py").is_file()
    assert (root / "scripts" / "process_session.py").is_file()


def test_pipeline_root_frozen_uses_meipass(tmp_path: Path, monkeypatch) -> None:
    import sys as _sys

    from exo_collection.apps.calculate import _pipeline

    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(_sys, "_MEIPASS", str(tmp_path), raising=False)

    root = _pipeline.pipeline_root()
    assert root == tmp_path / "opensim_joint_moment_pipeline"


# --------------------------------------------------------------------------
# 选择流程：受试者 → 自动静态绑定（STAND）→ 动态工况
# --------------------------------------------------------------------------
def _stand_session(subject: str = "003", date: str = "2026-09-01T00:00:00", c3d: bool = True):
    from dataclasses import replace

    from exo_collection.apps.calculate.models import SessionFiles

    files = SessionFiles(c3d_path=Path("static.c3d")) if c3d else SessionFiles()
    return replace(
        _make_session(subject=subject, condition="STAND"),
        started_at_utc=date,
        files=files,
    )


def test_recommend_static_for_subject_picks_most_recent() -> None:
    from exo_collection.apps.calculate.discovery import recommend_static_for_subject

    sessions = [
        _stand_session(date="2026-08-01T00:00:00"),
        _stand_session(date="2026-09-02T00:00:00"),
        _make_session(subject="003", condition="WALK_STEADY_1P00"),
    ]
    chosen = recommend_static_for_subject("003", sessions)
    assert chosen is not None
    assert chosen.is_stand
    assert chosen.started_at_utc.startswith("2026-09-02")


def test_recommend_static_for_subject_requires_c3d_and_same_subject() -> None:
    from exo_collection.apps.calculate.discovery import recommend_static_for_subject

    sessions = [
        _stand_session(subject="003", c3d=False),   # STAND 但缺 C3D
        _stand_session(subject="004"),              # 别的受试者
    ]
    assert recommend_static_for_subject("003", sessions) is None
    assert recommend_static_for_subject("005", sessions) is None


def test_session_selector_auto_binds_static_and_dynamic(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from exo_collection.apps.calculate import session_selector

    sessions = [
        _stand_session(date="2026-08-01T00:00:00"),
        _stand_session(date="2026-09-02T00:00:00"),
        _make_session(subject="003", condition="WALK_STEADY_1P00"),
    ]
    monkeypatch.setattr(session_selector, "discover_sessions", lambda root: sessions)

    selector = session_selector.SessionSelector(tmp_path)

    # 自动绑定：动态 = 非 STAND 工况，静态 = 最近且带 C3D 的 STAND。
    assert selector.current_dynamic() is not None
    assert not selector.current_dynamic().is_stand
    assert selector.current_static() is not None
    assert selector.current_static().is_stand
    assert selector.current_static().started_at_utc.startswith("2026-09-02")
