"""Exo Calculate 的纯逻辑单元测试（状态机、DTO、设置持久化）。

不 import OpenSim、不读真实数据文件；只验证状态迁移、DTO 约定与设置回写。
"""

from __future__ import annotations

import os
from dataclasses import replace
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
from exo_collection.apps.calculate.processing_view import ProcessingView  # noqa: E402


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


def test_transient_analysis_controls_require_current_c3d_window(tmp_path):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    view = ProcessingView(_file_settings(tmp_path / "transient.ini"))
    try:
        view.set_solve_enabled(True)
        view.set_condition("STAND_BWALK_STAND_NOEXO")
        assert not view._analysis_auto.isChecked()
        assert not view._analysis_auto.isEnabled()
        assert "推荐" not in view._analysis_auto.text()
        assert view._analysis_start.isEnabled() and view._analysis_end.isEnabled()
        assert view._static_auto.isChecked()
        assert not view._process_button.isEnabled()
        assert "第0帧" in view._analysis_hint.text()
        assert "不是TXT" in view._analysis_hint.text()
        assert "静止期" in view._analysis_hint.text()
        captured = []
        view.process_requested.connect(captured.append)
        view._request_process()
        assert not captured
        view._analysis_start.setValue(3)
        view._analysis_end.setValue(20)
        assert view._process_button.isEnabled()
        view.set_analysis_last_time(10)
        assert not view._process_button.isEnabled()
        view._analysis_end.setValue(9)
        view._request_process()
        assert captured[0].analysis_time_range_s == (3, 9)
        assert captured[0].static_time_range_s is None
        view.set_condition("DWALK_0P6_NOEXO")
        assert view._analysis_auto.isEnabled() and view._analysis_auto.isChecked()
        assert view._analysis_last_time_s is None
        assert view._analysis_start.value() == view._analysis_end.value() == 0
        assert view._process_button.isEnabled()
    finally:
        view.close()


@pytest.mark.parametrize("window", [None, (0, 0), (3, 1), (-1, 3), (0, float("nan")), (0, 31)])
def test_transient_rejects_invalid_analysis_windows(window):
    assert ProcessingView.analysis_window_error(window, required=True, last_time_s=30)
    assert ProcessingView.analysis_window_error((0, 30), required=True, last_time_s=30) is None


@pytest.mark.parametrize("failure", ["mark_failed", "mark_cancelled"])
@pytest.mark.parametrize("manual,confirm", [(True, False), (False, True), (False, False)])
def test_failed_processing_retry_keeps_confirmed_sync_gate(failure, manual, confirm):
    controller = CalculateController(Path("/tmp/data"))
    sync = SyncResult(c3d_start_in_mocap_h5_frame=0, c3d_h5_match_rms_mm=0,
                      c3d_h5_overlap_frames=100, gaitway_offset_s=-1.544952,
                      confidence="MEDIUM", manual=manual)
    controller.set_sync(sync)
    if confirm:
        controller.confirm_sync(note="reviewed")
    receipt = controller.confirm_meta
    if controller.can_solve:
        controller.mark_processing()
    getattr(controller, failure)()
    assert controller.can_solve == (manual or confirm)
    assert controller.sync is sync and controller.confirm_meta is receipt
    controller.set_dynamic(_make_session(condition="STAND_BWALK_STAND_NOEXO"))
    assert not controller.can_solve


def test_window_preflights_transient_bounds_before_worker_and_can_retry(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    window = CalculateWindow(tmp_path, _file_settings(tmp_path / "transient_window.ini"))
    try:
        files = SessionFiles(c3d_path=tmp_path / "trial.c3d", txt_path=tmp_path / "trial.txt",
                             mocap_h5_path=tmp_path / "mocap.h5", imu_h5_path=tmp_path / "imu.h5")
        dynamic = replace(_make_session(condition="STAND_BWALK_STAND_NOEXO"), session_dir=tmp_path / "session", files=files)
        static = replace(_make_session(condition="STATIC_CALIB"), files=SessionFiles(c3d_path=tmp_path / "static.c3d"))
        window._on_dynamic(dynamic)
        window._controller.set_static(static)
        sync = SyncResult(c3d_start_in_mocap_h5_frame=0, c3d_h5_match_rms_mm=0, c3d_h5_overlap_frames=100,
                          gaitway_offset_s=-1.544952, confidence="MEDIUM", manual=True)
        window._controller.set_sync(sync)
        monkeypatch.setattr(window._processing_view, "opensim_python", lambda: Path("python.exe"))
        monkeypatch.setattr(window, "_generic_model_path", lambda: tmp_path / "model.osim")
        report = InputCheckReport(subject_code="003", dynamic_session=dynamic.session_name, static_session=None,
                                  dynamic_c3d_duration_s=40, dynamic_c3d_rate_hz=100)
        calls = []
        def check(*args):
            calls.append(args)
            return report
        monkeypatch.setattr("exo_collection.apps.calculate.discovery.check_inputs", check)
        launched = []
        monkeypatch.setattr(window._thread_pool, "start", launched.append)
        invalid = ProcessingConfig(mass_kg=80, height_m=1.87, analysis_time_range_s=(0, 60))
        window._on_process_requested(invalid)
        assert len(calls) == 1 and not launched
        assert window._controller.run_dir is None and not window._busy
        assert window._controller.sync is sync and window._controller.can_solve
        assert window._processing_view._analysis_last_time_s == pytest.approx(39.99)
        view = window._processing_view
        view._analysis_start.setValue(5)
        view._analysis_end.setValue(30)
        valid = replace(invalid, analysis_time_range_s=(5, 30))
        window._on_process_requested(valid)
        assert len(launched) == 1
        ctx = window._controller.active_operation
        assert ctx is not None
        window._on_process_failed("controlled preprocessing failure", ctx)
        assert window._controller.state == ProcessingState.FAILED
        assert window._controller.can_solve and view._process_button.isEnabled()
        assert window._controller.sync is sync
        # Another trial clears both the previous 39.99s limit and manual input.
        window._on_dynamic(replace(dynamic, session_uuid="different", condition_code="DWALK_0P6_NOEXO"))
        assert view._analysis_last_time_s is None and view._analysis_auto.isChecked()
    finally:
        window.close()


def test_transient_metadata_check_cannot_continue_after_trial_switch(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    window = CalculateWindow(tmp_path, _file_settings(tmp_path / "switched_window.ini"))
    try:
        files = SessionFiles(c3d_path=tmp_path / "trial.c3d", txt_path=tmp_path / "trial.txt",
                             mocap_h5_path=tmp_path / "mocap.h5", imu_h5_path=tmp_path / "imu.h5")
        dynamic = replace(_make_session(condition="STAND_BWALK_STAND_NOEXO"), files=files)
        window._on_dynamic(dynamic)
        window._controller.set_static(replace(_make_session(condition="STATIC_CALIB"), files=files))
        window._controller.set_sync(SyncResult(c3d_start_in_mocap_h5_frame=0, c3d_h5_match_rms_mm=0,
            c3d_h5_overlap_frames=100, gaitway_offset_s=0, confidence="HIGH"))
        monkeypatch.setattr(window._processing_view, "opensim_python", lambda: Path("python.exe"))
        def check(*args):
            window._on_dynamic(replace(dynamic, session_uuid="new-session", condition_code="DWALK_0P6_NOEXO"))
            return InputCheckReport(subject_code="003", dynamic_session=dynamic.session_name, static_session=None,
                                    dynamic_c3d_duration_s=40, dynamic_c3d_rate_hz=100)
        monkeypatch.setattr("exo_collection.apps.calculate.discovery.check_inputs", check)
        window._on_process_requested(ProcessingConfig(mass_kg=80, height_m=1.87, analysis_time_range_s=(5, 30)))
        assert window._controller.run_dir is None and not window._busy
        assert "已丢弃旧" in window._processing_view._log.toPlainText()
        assert window._processing_view._analysis_last_time_s is None
    finally:
        window.close()


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
    controller.set_static(_make_session(subject="001", condition="STATIC_CALIB"))
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
    controller.set_static(_make_session(subject="003", condition="STATIC_CALIB"))

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


def test_imu_start_frame_rejects_bad_saved_match_without_fallback(tmp_path: Path) -> None:
    import json
    from dataclasses import replace
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    settings = _file_settings(tmp_path / "shared.ini")
    window = CalculateWindow(tmp_path, settings)
    session_dir = tmp_path / "session"
    dynamic = replace(_make_session(), session_dir=session_dir)
    window._controller.set_dynamic(dynamic)
    calibration = session_dir / "derived" / "opensim" / "sync_calibration.json"
    calibration.parent.mkdir(parents=True)

    # An explicitly bad result must stop lookup; a good auto_candidate cannot
    # hide it by serving as a silent fallback.
    calibration.write_text(
        json.dumps({
            "result": {"c3d_start_in_mocap_h5_frame": 11, "c3d_h5_match_rms_mm": 6.0},
            "auto_candidate": {"c3d_start_in_mocap_h5_frame": 12, "c3d_h5_match_rms_mm": 0.0},
        }),
        encoding="utf-8",
    )
    assert window._imu_start_frame() is None

    # A reliable saved result remains reusable.
    calibration.write_text(
        json.dumps({
            "result": {"c3d_start_in_mocap_h5_frame": 11, "c3d_h5_match_rms_mm": 0.0},
            "auto_candidate": {"c3d_start_in_mocap_h5_frame": 12, "c3d_h5_match_rms_mm": 6.0},
        }),
        encoding="utf-8",
    )
    assert window._imu_start_frame() == 11

    # Legacy blocks without an RMS field keep the historical behavior.
    calibration.write_text(
        json.dumps({"result": {"c3d_start_in_mocap_h5_frame": 13}}),
        encoding="utf-8",
    )
    assert window._imu_start_frame() == 13

    # The current in-memory sync path is guarded too.
    window._controller.set_sync(
        SyncResult(
            c3d_start_in_mocap_h5_frame=14,
            c3d_h5_match_rms_mm=6.0,
            c3d_h5_overlap_frames=100,
            gaitway_offset_s=0.0,
            confidence="LOW",
        )
    )
    assert window._imu_start_frame() is None
    window.close()


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
    from dataclasses import replace

    # 旧协议 STAND / 基础工况 STAND_30S 是正常动态工况，不再当作静态标定。
    assert not _make_session(condition="STAND").is_stand
    assert not _make_session(condition="STAND_30S_NOEXO").is_stand
    assert _make_session(condition="STAND_30S_NOEXO").is_quiet_standing
    assert _make_session(condition="STAND").is_quiet_standing
    # 新增开始/结束站立：仍是正常动态工况，且属于静止站立。
    for code in ("START_STAND_30S_NOEXO", "END_STAND_30S_NOEXO"):
        assert not _make_session(condition=code).is_stand
        assert _make_session(condition=code).is_quiet_standing
    # 新增开始/结束行走：动态工况，但不是静止站立。
    assert not _make_session(condition="START_WALK_1P0_30S_NOEXO").is_stand
    assert not _make_session(condition="START_WALK_1P0_30S_NOEXO").is_quiet_standing
    assert _make_session(condition="STATIC_CALIB").is_stand
    assert not _make_session(condition="STATIC_CALIB").is_quiet_standing
    assert not _make_session(condition="WALK_STEADY_1P00").is_stand
    # category 兜底：condition_code 不含 STATIC_CALIB，但 category 指明静态标定。
    by_category = replace(
        _make_session(condition="HELEN_HAYES"),
        condition_parameters={"category": "test_static_calibration"},
    )
    assert by_category.is_stand


def test_session_record_explicit_static_calibration_flag() -> None:
    from dataclasses import replace

    assert not _make_session(condition="STAND").is_explicit_static_calibration
    assert not _make_session(condition="STAND_30S_NOEXO").is_explicit_static_calibration
    assert not _make_session(condition="START_STAND_30S_NOEXO").is_explicit_static_calibration
    assert not _make_session(condition="END_STAND_30S_NOEXO").is_explicit_static_calibration
    assert _make_session(condition="STATIC_CALIB").is_explicit_static_calibration
    assert replace(
        _make_session(condition="HELEN_HAYES"),
        condition_parameters={"category": "test_static_calibration"},
    ).is_explicit_static_calibration


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
# 选择流程：受试者 → 自动静态绑定（STATIC_CALIB）→ 动态工况
# --------------------------------------------------------------------------
def _static_session(subject: str = "003", date: str = "2026-09-01T00:00:00", c3d: bool = True):
    from dataclasses import replace

    from exo_collection.apps.calculate.models import SessionFiles

    files = SessionFiles(c3d_path=Path("static.c3d")) if c3d else SessionFiles()
    return replace(
        _make_session(subject=subject, condition="STATIC_CALIB"),
        started_at_utc=date,
        files=files,
    )


def test_recommend_static_for_subject_picks_most_recent() -> None:
    from exo_collection.apps.calculate.discovery import recommend_static_for_subject

    sessions = [
        _static_session(date="2026-08-01T00:00:00"),
        _static_session(date="2026-09-02T00:00:00"),
        _make_session(subject="003", condition="WALK_STEADY_1P00"),
    ]
    chosen = recommend_static_for_subject("003", sessions)
    assert chosen is not None
    assert chosen.is_stand
    assert chosen.started_at_utc.startswith("2026-09-02")


def test_recommend_static_for_subject_requires_c3d_and_same_subject() -> None:
    from exo_collection.apps.calculate.discovery import recommend_static_for_subject

    sessions = [
        _static_session(subject="003", c3d=False),   # 静态标定但缺 C3D
        _static_session(subject="004"),              # 别的受试者
    ]
    assert recommend_static_for_subject("003", sessions) is None
    assert recommend_static_for_subject("005", sessions) is None


def test_recommend_static_ignores_legacy_stand() -> None:
    from dataclasses import replace

    from exo_collection.apps.calculate.discovery import recommend_static_for_subject
    from exo_collection.apps.calculate.models import SessionFiles

    # 旧协议 STAND / 基础工况 STAND_30S 不再是静态标定候选：只有显式 STATIC_CALIB
    # 会被推荐为静态模型。
    legacy_stand = replace(
        _make_session(subject="003", condition="STAND_30S_NOEXO"),
        started_at_utc="2026-09-02T00:00:00",
        files=SessionFiles(c3d_path=Path("stand.c3d")),
    )
    explicit = _static_session(date="2026-08-01T00:00:00")

    # 同时存在旧 STAND 与显式标定时，只有显式标定被推荐。
    chosen = recommend_static_for_subject("003", [legacy_stand, explicit])
    assert chosen is not None
    assert chosen.condition_code == "STATIC_CALIB"

    # 仅有旧 STAND 时不再返回静态标定（宁可 None 也不误用基线站立）。
    assert recommend_static_for_subject("003", [legacy_stand]) is None


def test_session_selector_auto_binds_static_and_dynamic(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from exo_collection.apps.calculate import session_selector

    sessions = [
        _static_session(date="2026-08-01T00:00:00"),
        _static_session(date="2026-09-02T00:00:00"),
        _make_session(subject="003", condition="WALK_STEADY_1P00"),
    ]
    monkeypatch.setattr(session_selector, "discover_sessions", lambda root: sessions)

    selector = session_selector.SessionSelector(tmp_path)

    # 自动绑定：动态 = 非静态工况，静态 = 最近且带 C3D 的静态标定。
    assert selector.current_dynamic() is not None
    assert not selector.current_dynamic().is_stand
    assert selector.current_static() is not None
    assert selector.current_static().is_stand
    assert selector.current_static().started_at_utc.startswith("2026-09-02")


def test_day_from_session_dir_parses_canonical_layout() -> None:
    from exo_collection.apps.calculate.discovery import day_from_session_dir

    assert day_from_session_dir(Path("data/001/d2/F_STEADY/WALK/session1")) == 2
    assert day_from_session_dir(Path("data/001/d10/F_STEADY/WALK/session1")) == 10


def test_day_from_session_dir_returns_none_without_day_dir() -> None:
    from exo_collection.apps.calculate.discovery import day_from_session_dir

    assert day_from_session_dir(Path("data/001/F_STEADY/WALK/session1")) is None


def test_distinct_days_sorted_with_none_last() -> None:
    from dataclasses import replace

    from exo_collection.apps.calculate.discovery import distinct_days

    records = [
        replace(_make_session(), day=2),
        replace(_make_session(), day=1),
        replace(_make_session(), day=None),
        replace(_make_session(), day=3),
    ]
    assert distinct_days(records) == [1, 2, 3, None]


def test_recommend_static_for_subject_day_scoped() -> None:
    from dataclasses import replace

    from exo_collection.apps.calculate.discovery import recommend_static_for_subject

    static_d1 = replace(_static_session(date="2026-08-01T00:00:00"), day=1)
    static_d2 = replace(_static_session(date="2026-09-02T00:00:00"), day=2)
    dynamic = replace(_make_session(subject="003", condition="WALK_STEADY_1P00"), day=2)
    sessions = [static_d1, static_d2, dynamic]

    assert recommend_static_for_subject("003", sessions, day=1) == static_d1
    assert recommend_static_for_subject("003", sessions, day=2) == static_d2
    assert recommend_static_for_subject("003", sessions, day=3) is None
    # 省略 day 时仍返回该受试者最近静态，不受天数过滤影响。
    assert recommend_static_for_subject("003", sessions) == static_d2


def test_session_selector_scopes_by_day(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from exo_collection.apps.calculate import session_selector

    sessions = [
        replace(_static_session(date="2026-08-01T00:00:00"), day=1),
        replace(_static_session(date="2026-09-02T00:00:00"), day=2),
        replace(_make_session(subject="003", condition="WALK_STEADY_1P00"), day=1),
        replace(_make_session(subject="003", condition="WALK_FAST_1P50"), day=2),
    ]
    monkeypatch.setattr(session_selector, "discover_sessions", lambda root: sessions)

    selector = session_selector.SessionSelector(tmp_path)

    # 默认选中 d1：动态/静态只含 d1。
    assert selector._day_combo.currentData() == 1
    assert selector.current_dynamic().condition_code == "WALK_STEADY_1P00"
    assert selector.current_static().day == 1

    # 切到 d2：动态/静态只含 d2。
    selector._day_combo.setCurrentIndex(1)
    assert selector._day_combo.currentData() == 2
    assert selector.current_dynamic().condition_code == "WALK_FAST_1P50"
    assert selector.current_static().day == 2
    selector.close()


def test_session_selector_discard_button_and_label(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from exo_collection.apps.calculate import session_selector
    from exo_collection.apps.calculate.manual_review import write_discard

    dynamic = replace(
        _make_session(subject="003", condition="WALK_STEADY_1P00"),
        session_dir=tmp_path / "003" / "walk",
    )
    monkeypatch.setattr(session_selector, "discover_sessions", lambda root: [dynamic])

    selector = session_selector.SessionSelector(tmp_path)

    # 丢弃按钮发出信号，携带当前动态 session。
    emitted: list[SessionRecord] = []
    selector.discard_requested.connect(emitted.append)
    selector._discard_button.click()
    assert emitted == [dynamic]

    # 丢弃后刷新标签，下拉显示「丢弃」。
    write_discard(dynamic.session_dir)
    selector.refresh_labels()
    assert "丢弃" in selector._dynamic_combo.currentText()
    selector.close()


def test_session_selector_restore_button_and_state(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from exo_collection.apps.calculate import session_selector
    from exo_collection.apps.calculate.manual_review import is_discarded, write_discard

    dynamic = replace(
        _make_session(subject="003", condition="WALK_STEADY_1P00"),
        session_dir=tmp_path / "003" / "walk",
    )
    monkeypatch.setattr(session_selector, "discover_sessions", lambda root: [dynamic])

    selector = session_selector.SessionSelector(tmp_path)

    # 未丢弃：丢弃可用、恢复禁用。
    assert selector._discard_button.isEnabled()
    assert not selector._restore_button.isEnabled()

    # 标记丢弃后刷新：丢弃禁用、恢复可用。
    write_discard(dynamic.session_dir)
    selector.refresh_labels()
    assert is_discarded(dynamic.session_dir)
    assert not selector._discard_button.isEnabled()
    assert selector._restore_button.isEnabled()

    # 恢复按钮发出信号，携带当前动态 session。
    emitted: list[SessionRecord] = []
    selector.restore_requested.connect(emitted.append)
    selector._restore_button.click()
    assert emitted == [dynamic]
    selector.close()


def test_grf_default_matches_kinematics_and_explicit_ui_setting_is_preserved(tmp_path):
    from PySide6.QtWidgets import QApplication
    from exo_collection.apps.calculate.workers import PrepWorker
    QApplication.instance() or QApplication([])
    config = ProcessingConfig(mass_kg=75, height_m=1.75)
    assert config.marker_cutoff_hz == config.grf_cutoff_hz == 6.0
    assert replace(config, grf_cutoff_hz=18.0).as_dict()["grf_cutoff_hz"] == 18.0
    view = ProcessingView(_file_settings(tmp_path / "grf_default.ini"))
    try:
        assert view._grf_cutoff.value() == 6.0
        assert "不保证" in view._grf_cutoff.toolTip()
        view.set_solve_enabled(True)
        emitted = []
        view.process_requested.connect(emitted.append)
        view._grf_cutoff.setValue(18.0)
        view._request_process()
        assert emitted[0].grf_cutoff_hz == 18.0
    finally:
        view.close()
    kwargs = dict(static_c3d=Path("static.c3d"), dynamic_c3d=Path("dynamic.c3d"),
                  gaitway_txt=Path("trial.txt"), generic_model=Path("model.osim"),
                  out_dir=tmp_path, subject_id="501", mass_kg=75, height_m=1.75, gaitway_offset_s=0.)
    assert PrepWorker(**kwargs)._grf_cutoff == 6.0
    assert PrepWorker(**kwargs, grf_cutoff_hz=18.0)._grf_cutoff == 18.0
