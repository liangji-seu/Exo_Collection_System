"""版本化 QC 规则（pipeline.qc.evaluate）的单元测试。

验证阈值分带、缺失判定、「退出码 0 ≠ QC PASS」的核心语义，以及 §3.3 把
同步质量纳入最终 QC 的状态规则（专家强制封顶 WARN、不唯一→FAIL、峰对<3→FAIL）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让测试能 import pipeline.qc（不依赖 opensim）
_REPO = Path(__file__).resolve().parents[2]
_PIPELINE = _REPO / "opensim_joint_moment_pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from pipeline.qc.evaluate import QC_SCHEMA_VERSION, evaluate_qc  # noqa: E402


def _id_qc(rms_n: float | None = 99.0, p95_n: float | None = 220.0) -> dict:
    return {"residual_force": {"rms_N": rms_n, "p95_N": p95_n}}


def _marker(rms_cm: float = 1.76) -> dict:
    return {"rms_mean_cm": rms_cm, "rms_p95_cm": 3.0, "max_marker_p95_cm": 4.0}


def _sync_ok(**overrides) -> dict:
    base = {
        "method": "AUTO_HIGH",
        "confidence": "HIGH",
        "gaitway_offset_s": 5.835,
        "n_pairs": 5,
        "mad_s": 0.01,
        "c3d_h5_common_markers": 15,
        "c3d_h5_rms_mm": 0.0,
        "c3d_h5_max_error_mm": 0.1,
        "c3d_h5_unique": True,
        "c3d_h5_exact": True,
        "mocap_clock_monotonic": True,
        "mocap_clock_gaps": 0,
        "imu_clock_monotonic": True,
        "imu_clock_gaps": 0,
    }
    base.update(overrides)
    return base


def _force_ok(**overrides) -> dict:
    base = {
        "n_valid_decomposed_frames": 5800,
        "n_right_contact_frames": 3200,
        "n_left_contact_frames": 3200,
    }
    base.update(overrides)
    return base


def _run(**kw) -> dict:
    params = dict(
        marker_qc_overall=_marker(),
        id_qc=_id_qc(99.0),
        mass_kg=80.0,
        sync=_sync_ok(),
        force=_force_ok(),
        dynamic_n_frames=6000,
    )
    params.update(kw)
    return evaluate_qc(**params)


def test_qc_pass_on_baseline() -> None:
    verdict = _run()
    assert verdict["schema_version"] == QC_SCHEMA_VERSION
    assert verdict["status"] == "PASS"


def test_qc_marker_bands() -> None:
    assert _run(marker_qc_overall={"rms_mean_cm": 1.0})["status"] == "PASS"
    assert _run(marker_qc_overall={"rms_mean_cm": 2.5})["status"] == "WARN"
    assert _run(marker_qc_overall={"rms_mean_cm": 5.0})["status"] == "FAIL"


def test_qc_residual_bands() -> None:
    # BW = 80 * 9.80665 ≈ 784.5 N；130 N ≈ 16.6% → WARN，300 N ≈ 38% → FAIL。
    assert _run(id_qc=_id_qc(130.0))["status"] == "WARN"
    assert _run(id_qc=_id_qc(300.0))["status"] == "FAIL"


def test_qc_missing_required_metrics_fail() -> None:
    verdict = evaluate_qc(marker_qc_overall={}, id_qc={}, mass_kg=80.0)
    assert verdict["status"] == "FAIL"
    assert "缺失" in verdict["summary"]


def test_qc_nan_treated_as_missing() -> None:
    import math

    verdict = evaluate_qc(marker_qc_overall={"rms_mean_cm": float("nan")},
                          id_qc=_id_qc(float("nan")), mass_kg=80.0)
    assert verdict["status"] == "FAIL"


def test_qc_checks_have_structure() -> None:
    verdict = _run()
    keys = {c["key"] for c in verdict["checks"]}
    assert "marker_rms_mean_cm" in keys
    assert "residual_force_rms_N" in keys
    # §3.3 同步质量进入 QC schema
    assert "sync_c3d_h5_unique" in keys
    assert "sync_n_pairs" in keys
    assert "force_coverage" in keys
    for c in verdict["checks"]:
        assert c["status"] in {"PASS", "WARN", "FAIL", "INFO"}


# --------------------------------------------------------------------------
# §3.3 同步质量状态规则
# --------------------------------------------------------------------------
def test_qc_expert_forced_caps_at_warn() -> None:
    # 其它指标全过，但专家强制 offset → 整份最多 WARN，绝不 PASS。
    verdict = _run(sync=_sync_ok(method="EXPERT_FORCED", confidence="LOW",
                                 n_pairs=0, mad_s=None, c3d_h5_unique=None))
    assert verdict["status"] == "WARN"


def test_qc_c3d_h5_not_unique_unconfirmed_fails() -> None:
    verdict = _run(sync=_sync_ok(c3d_h5_unique=False, method="AUTO_HIGH"))
    assert verdict["status"] == "FAIL"


def test_qc_c3d_h5_not_unique_manual_warns() -> None:
    verdict = _run(sync=_sync_ok(c3d_h5_unique=False, method="MANUAL_PAIRED"))
    assert verdict["status"] == "WARN"


def test_qc_few_pairs_fails() -> None:
    verdict = _run(sync=_sync_ok(n_pairs=2))
    assert verdict["status"] == "FAIL"


def test_qc_clock_not_monotonic_fails() -> None:
    verdict = _run(sync=_sync_ok(mocap_clock_monotonic=False, mocap_clock_gaps=3))
    assert verdict["status"] == "FAIL"


def test_qc_force_coverage_low_warns() -> None:
    verdict = _run(force=_force_ok(n_valid_decomposed_frames=4000))  # 4000/6000 = 0.67
    assert verdict["status"] == "WARN"


def test_qc_accepts_raw_run_auto_sync_keys() -> None:
    """直连 ``prepare_session(sync_quality=run_auto_sync(...))`` 时拿到的是
    ``run_auto_sync`` 的原始键（c3d_h5_matched_markers 列表 / c3d_h5_match_rms_mm /
    mocap_h5_*），也必须能正确判 QC，不得误报「公共 marker 缺失 / 时钟未评估」。

    真实数据验收回归：修复前 ``sync_c3d_h5_markers`` 因键名不匹配被判 FAIL、
    ``mocap 时钟`` 被判「未评估」。
    """
    raw = {
        "method": "AUTO_HIGH",
        "confidence": "HIGH",
        "gaitway_offset_s": 5.835,
        "n_pairs": 5,
        "mad_s": 0.01,
        "c3d_h5_matched_markers": ["R.ASIS", "L.ASIS", "R.Knee", "L.Knee", "R.Ankle",
                                    "L.Ankle", "R.Heel", "L.Heel", "R.Toe", "L.Toe"],
        "c3d_h5_match_rms_mm": 0.0,
        "c3d_h5_match_max_error_mm": 0.1,
        "c3d_h5_unique": True,
        "c3d_h5_exact": True,
        "mocap_h5_monotonic": True,
        "mocap_h5_clock_gaps": 0,
        "imu_clock_monotonic": True,
        "imu_clock_gaps": 0,
    }
    verdict = _run(sync=raw)
    # 基线各指标均在阈值内 → 应 PASS，且公共 marker 数被正确读成 10（列表长度）。
    assert verdict["status"] == "PASS"
    markers_check = next(c for c in verdict["checks"] if c["key"] == "sync_c3d_h5_markers")
    assert markers_check["value"] == 10.0
    assert markers_check["status"] == "PASS"
    # mocap 时钟不再「未评估」，而是「单调」。
    mocap_check = next(c for c in verdict["checks"] if c["key"] == "sync_mocap_clock")
    assert mocap_check["status"] == "PASS"
