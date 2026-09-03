"""版本化生物力学 QC 评估。

消费 ``marker_qc_overall``（marker 误差）、``id_qc``（残余力 + 髋力矩）、
``sync``（同步质量，prompt6 §3.3）与 ``force``（左右力覆盖，来自 manifest 的
``gaitway`` 块），输出：

    {
      "schema_version": "1.1.0",
      "status": "PASS" | "WARN" | "FAIL",
      "checks": [{"key","label","status","value","unit","threshold","detail"}, ...],
      "summary": "...",
    }

阈值是领域先验（marker RMS、残余力占体重比、同步唯一性/峰对/时钟），已在 003
基线数据上验证：marker rms_mean_cm ≈ 1.76 → PASS；残余力 rms ≈ 90~100 N
（约 12.7% BW）→ PASS；同步 C3D↔H5 精确且唯一、峰对数 ≥3 → PASS。

**同步结论规则（§3.3）**：

- 专家强制 offset（``method == EXPERT_FORCED``）：整份 QC 最多 ``WARN``，绝不
  ``PASS``；
- C3D↔H5 不唯一且无人工确认（``method`` 非 MANUAL_PAIRED / EXPERT_FORCED）：
  ``FAIL``；
- 跺脚峰对 <3 对且非专家强制：``FAIL``（普通模式禁止解算）；
- 同步 HIGH 且其它指标通过：允许 ``PASS``。

**明确不做的**：不把关节力矩本身设硬阈值判定 PASS/FAIL——力矩是结果而非 QC
门槛，只在 ``INFO`` 检查里展示量级供人工判断。
"""

from __future__ import annotations

import math
from typing import Any

QC_SCHEMA_VERSION = "1.1.0"

# 阈值（cm 与 体重占比）
MARKER_RMS_WARN_CM = 2.0
MARKER_RMS_FAIL_CM = 4.0
MARKER_MAX_WARN_CM = 6.0
RESIDUAL_WARN_FRAC = 0.15
RESIDUAL_FAIL_FRAC = 0.30
RESIDUAL_P95_WARN_FRAC = 0.40

# 同步质量阈值（版本化，进入审计结果；prompt6 §3.3）
SYNC_MIN_MARKERS = 3
SYNC_MAD_WARN_S = 0.05
SYNC_CLOCK_GAPS_WARN = 5

# 左右力有效覆盖比例阈值
FORCE_COVERAGE_WARN_FRAC = 0.80
FORCE_COVERAGE_FAIL_FRAC = 0.50

# 静态 marker 调整分级阈值（mm，进入审计结果；prompt6 §3.6）
# <30 常规 / 30~50 提醒 / 50~80 警告 / >80 默认阻止（专家确认后降为 WARN）。
MARKER_ADJUST_REMIND_MM = 30.0
MARKER_ADJUST_WARN_MM = 50.0
MARKER_ADJUST_FAIL_MM = 80.0

_G = 9.80665

# 判定优先级：FAIL > WARN > PASS > INFO（INFO 不影响结论）
_RANK = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}


def _check(key: str, label: str, value: float | None, status: str,
           unit: str, threshold: str, detail: str = "") -> dict[str, Any]:
    return {
        "key": key, "label": label, "status": status,
        "value": None if value is None else round(float(value), 4),
        "unit": unit, "threshold": threshold, "detail": detail,
    }


def _band(value: float | None, warn: float, fail: float) -> str:
    """把数值映射到 PASS/WARN/FAIL；None 或 NaN 视为 FAIL（必需指标缺失）。"""
    if value is None or math.isnan(float(value)):
        return "FAIL"
    if value > fail:
        return "FAIL"
    if value > warn:
        return "WARN"
    return "PASS"


def _sync_checks(sync: dict[str, Any] | None) -> list[dict[str, Any]]:
    """同步质量检查（prompt6 §3.3）。"""
    if not sync:
        return [_check("sync_info", "同步信息", None, "FAIL", "", "必需", "缺少同步质量信息")]

    method = str(sync.get("method") or "AUTO_HIGH")
    expert = method == "EXPERT_FORCED"
    manual = method == "MANUAL_PAIRED"
    human_confirmed = expert or manual

    # 兼容两种键名：App 侧 `_sync_quality()` 归一化后的键（c3d_h5_common_markers /
    # c3d_h5_rms_mm / mocap_clock_*），以及 `run_auto_sync` 的原始键
    # （c3d_h5_matched_markers / c3d_h5_match_rms_mm / mocap_h5_*）。真实数据验收
    # 直连 `prepare_session(sync_quality=run_auto_sync(...))` 时会拿到原始键，必须
    # 也能正确判 QC，否则「公共 marker 缺失 / 时钟未评估」误报 FAIL。
    n_markers = sync.get("c3d_h5_common_markers")
    if n_markers is None:
        matched = sync.get("c3d_h5_matched_markers")
        if isinstance(matched, (list, tuple, set)):
            n_markers = len(matched) or None

    rms = sync.get("c3d_h5_rms_mm")
    if rms is None:
        rms = sync.get("c3d_h5_match_rms_mm")
    max_err = sync.get("c3d_h5_max_error_mm")
    if max_err is None:
        max_err = sync.get("c3d_h5_match_max_error_mm")

    unique = sync.get("c3d_h5_unique")
    n_pairs = sync.get("n_pairs")
    mad = sync.get("mad_s")

    mocap_mono = sync.get("mocap_clock_monotonic")
    if mocap_mono is None:
        mocap_mono = sync.get("mocap_h5_monotonic")
    mocap_gaps = sync.get("mocap_clock_gaps")
    if mocap_gaps is None:
        mocap_gaps = sync.get("mocap_h5_clock_gaps")
    imu_mono = sync.get("imu_clock_monotonic")
    imu_gaps = sync.get("imu_clock_gaps")

    checks: list[dict[str, Any]] = []

    # 方法（专家强制 → 无条件 WARN 封顶）
    if expert:
        checks.append(_check("sync_method", "同步方法", None, "WARN", "",
                             "专家强制 offset：最多 WARN", "无峰证据，QC 绝不 PASS"))
    else:
        checks.append(_check("sync_method", "同步方法", None, "INFO", "",
                             "—", method))

    # C3D↔H5 公共 marker 数
    if n_markers is None:
        marker_status = "FAIL" if not human_confirmed else "WARN"
        marker_detail = "缺少自动匹配证据"
    elif n_markers < SYNC_MIN_MARKERS:
        marker_status = "FAIL"
        marker_detail = f"公共 marker 仅 {n_markers} 个"
    else:
        marker_status = "PASS"
        marker_detail = f"{n_markers} 个公共 marker"
    checks.append(_check("sync_c3d_h5_markers", "C3D↔H5 公共 marker",
                         n_markers, marker_status, "个", f"≥{SYNC_MIN_MARKERS}", marker_detail))

    # C3D↔H5 RMS / 最大误差（信息展示，唯一性单独判）
    checks.append(_check("sync_c3d_h5_rms_mm", "C3D↔H5 RMS", rms, "INFO", "mm", "—"))
    checks.append(_check("sync_c3d_h5_max_error_mm", "C3D↔H5 最大误差", max_err, "INFO", "mm", "—"))

    # 唯一性：不唯一且无人工确认 → FAIL；不唯一但人工确认 → WARN
    if unique is True:
        unique_status, unique_detail = "PASS", "精确且唯一"
    elif unique is False and human_confirmed:
        unique_status, unique_detail = "WARN", "不唯一，已人工确认"
    elif unique is False:
        unique_status, unique_detail = "FAIL", "不唯一且未人工确认"
    else:
        unique_status = "WARN" if human_confirmed else "FAIL"
        unique_detail = "唯一性未知（无自动匹配证据）"
    checks.append(_check("sync_c3d_h5_unique", "C3D↔H5 唯一性", None,
                         unique_status, "", "—", unique_detail))

    # 跺脚峰对（专家强制跳过）
    if expert:
        checks.append(_check("sync_n_pairs", "跺脚峰对", 0, "WARN", "对",
                             "专家强制：无峰证据", "跳过峰对校验"))
    elif n_pairs is None or n_pairs < 3:
        checks.append(_check("sync_n_pairs", "跺脚峰对", n_pairs, "FAIL", "对",
                             "≥3", "峰对不足 3 对，普通模式禁止解算"))
    else:
        checks.append(_check("sync_n_pairs", "跺脚峰对", n_pairs, "PASS", "对", "≥3"))

    # MAD
    if mad is None:
        mad_status = "INFO" if human_confirmed else "FAIL"
    elif mad > SYNC_MAD_WARN_S:
        mad_status = "WARN"
    else:
        mad_status = "PASS"
    checks.append(_check("sync_mad_s", "跺脚 MAD", mad, mad_status, "s",
                         f"≤{SYNC_MAD_WARN_S}"))

    # 时钟单调 + 间断数
    for prefix, label, mono, gaps in (
        ("sync_mocap_clock", "mocap 时钟", mocap_mono, mocap_gaps),
        ("sync_imu_clock", "IMU 时钟", imu_mono, imu_gaps),
    ):
        if mono is False:
            status, detail = "FAIL", "时钟非单调"
        elif gaps is None:
            status, detail = "INFO", "未评估"
        elif gaps > SYNC_CLOCK_GAPS_WARN:
            status, detail = "WARN", "间断偏多"
        else:
            status, detail = "PASS", "单调"
        checks.append(_check(prefix, label, gaps, status, "次",
                             f"gap≤{SYNC_CLOCK_GAPS_WARN}", detail))

    return checks


def _force_checks(force: dict[str, Any] | None, n_frames: int | None) -> list[dict[str, Any]]:
    """左右力有效覆盖比例（prompt6 §3.3）。"""
    if not force:
        return [_check("force_coverage", "左右力有效覆盖", None, "FAIL", "", "必需", "缺少力覆盖信息")]
    n_valid = force.get("n_valid_decomposed_frames")
    n_right = force.get("n_right_contact_frames")
    n_left = force.get("n_left_contact_frames")
    ratio = None
    if n_valid is not None and n_frames:
        ratio = float(n_valid) / float(n_frames)
    if ratio is None:
        status, detail = "FAIL", "缺少有效帧统计"
    elif ratio < FORCE_COVERAGE_FAIL_FRAC:
        status, detail = "FAIL", "有效覆盖过低"
    elif ratio < FORCE_COVERAGE_WARN_FRAC:
        status, detail = "WARN", "有效覆盖偏低"
    else:
        status, detail = "PASS", f"右 {n_right} / 左 {n_left} 接触帧"
    return [_check(
        "force_coverage", "左右力有效覆盖", ratio, status, "",
        f"≥{FORCE_COVERAGE_WARN_FRAC * 100:.0f}% 正常，<{FORCE_COVERAGE_FAIL_FRAC * 100:.0f}% 失败",
        detail,
    )]


def grade_marker_adjustments(refinement: dict[str, Any] | None) -> dict[str, Any]:
    """把 ``_refine_marker_locations`` 的报告逐点分级（prompt6 §3.6）。

    阈值集中在本模块常量，结果进入审计（result.json / qc_report.json），不在 UI
    里散落判断。``refinement`` 形如 ``{marker: {"adjustment_norm_mm": ..., "adjustment_mm": [...]}}``。
    """
    markers: list[dict[str, Any]] = []
    n_remind = n_warn = n_block = 0
    max_norm = None
    for name, entry in (refinement or {}).items():
        if not isinstance(entry, dict) or entry.get("updated") is not True:
            continue
        norm = entry.get("adjustment_norm_mm")
        if norm is None:
            continue
        norm = float(norm)
        if max_norm is None or norm > max_norm:
            max_norm = norm
        if norm > MARKER_ADJUST_FAIL_MM:
            grade, n_block = "BLOCK", n_block + 1
        elif norm > MARKER_ADJUST_WARN_MM:
            grade, n_warn = "WARN", n_warn + 1
        elif norm >= MARKER_ADJUST_REMIND_MM:
            grade, n_remind = "REMIND", n_remind + 1
        else:
            grade = "OK"
        markers.append({
            "marker": name,
            "adjustment_norm_mm": round(norm, 2),
            "adjustment_mm": [round(float(v), 2) for v in (entry.get("adjustment_mm") or [])],
            "grade": grade,
        })
    markers.sort(key=lambda m: -m["adjustment_norm_mm"])
    return {
        "markers": markers,
        "n_remind": n_remind,
        "n_warn": n_warn,
        "n_block": n_block,
        "max_adjustment_norm_mm": None if max_norm is None else round(max_norm, 2),
    }


def _marker_adjustment_checks(
    adjustment: dict[str, Any] | None, expert_confirmed: bool
) -> list[dict[str, Any]]:
    """静态 marker 调整分级 → QC 检查（prompt6 §3.6）。

    >80mm 默认阻止（FAIL）；专家确认后降为 WARN，绝不 PASS。
    """
    if not adjustment:
        return [_check("marker_adjustment", "静态 marker 调整", None, "INFO", "",
                       "—", "无 marker 调整报告")]
    max_norm = adjustment.get("max_adjustment_norm_mm")
    n_block = int(adjustment.get("n_block") or 0)
    n_warn = int(adjustment.get("n_warn") or 0)
    offenders = ", ".join(
        f"{m['marker']} {m['adjustment_norm_mm']}mm"
        for m in adjustment.get("markers", [])
        if m.get("adjustment_norm_mm", 0) > MARKER_ADJUST_WARN_MM
    ) or "无"

    if n_block > 0 and not expert_confirmed:
        status, detail = "FAIL", f"调整 >{MARKER_ADJUST_FAIL_MM:.0f}mm（{offenders}），需专家确认"
    elif n_block > 0:
        status, detail = "WARN", f"调整 >{MARKER_ADJUST_FAIL_MM:.0f}mm（{offenders}），专家已确认"
    elif n_warn > 0:
        status, detail = "WARN", f"调整 >{MARKER_ADJUST_WARN_MM:.0f}mm（{offenders}），检查贴点/名称/body归属"
    else:
        status, detail = "PASS", f"最大调整 {max_norm}mm"
    return [_check(
        "marker_adjustment", "静态 marker 调整", max_norm, status, "mm",
        f"≤{MARKER_ADJUST_WARN_MM:.0f} 正常，>{MARKER_ADJUST_FAIL_MM:.0f} 阻止",
        detail,
    )]


def evaluate_qc(
    *,
    marker_qc_overall: dict[str, Any] | None,
    id_qc: dict[str, Any] | None,
    mass_kg: float,
    sync: dict[str, Any] | None = None,
    force: dict[str, Any] | None = None,
    dynamic_n_frames: int | None = None,
    marker_adjustment: dict[str, Any] | None = None,
    marker_adjustment_expert_confirmed: bool = False,
) -> dict[str, Any]:
    """评估 QC。``marker_qc_overall`` / ``id_qc`` 来自 ``process_session.py``；
    ``sync`` / ``force`` 来自 ``manifest.json`` 的同步与力覆盖块（§3.3）；
    ``marker_adjustment`` 为 ``grade_marker_adjustments`` 的输出（§3.6）。"""
    marker = marker_qc_overall or {}
    idq = id_qc or {}
    residual = idq.get("residual_force") or {}
    body_weight = float(mass_kg) * _G if mass_kg and mass_kg > 0 else None

    rms_mean = marker.get("rms_mean_cm")
    rms_p95 = marker.get("rms_p95_cm")
    max_p95 = marker.get("max_marker_p95_cm")
    residual_rms = residual.get("rms_N")
    residual_p95 = residual.get("p95_N")

    residual_rms_frac = None
    residual_p95_frac = None
    if residual_rms is not None and body_weight:
        residual_rms_frac = residual_rms / body_weight
    if residual_p95 is not None and body_weight:
        residual_p95_frac = residual_p95 / body_weight

    checks: list[dict[str, Any]] = []

    # marker RMS（决定 FAIL/WARN 的核心指标）
    checks.append(_check(
        "marker_rms_mean_cm", "marker RMS 均值",
        rms_mean, _band(rms_mean, MARKER_RMS_WARN_CM, MARKER_RMS_FAIL_CM), "cm",
        f"≤{MARKER_RMS_WARN_CM} 正常，>{MARKER_RMS_FAIL_CM} 失败",
    ))
    checks.append(_check(
        "marker_rms_p95_cm", "marker RMS p95",
        rms_p95,
        "WARN" if rms_p95 is not None and rms_p95 > MARKER_MAX_WARN_CM else "PASS",
        "cm", f"≤{MARKER_MAX_WARN_CM}",
    ))
    checks.append(_check(
        "marker_max_p95_cm", "单 marker 峰值 p95",
        max_p95,
        "WARN" if max_p95 is not None and max_p95 > MARKER_MAX_WARN_CM else "PASS",
        "cm", f"≤{MARKER_MAX_WARN_CM}",
    ))

    # 残余力（占体重比）
    residual_status = "PASS"
    residual_threshold = "≤15% BW"
    if residual_rms is None:
        residual_status = "FAIL"
    elif residual_rms_frac is None:
        residual_status = "WARN"
        residual_threshold = "需要体重"
    else:
        residual_status = _band(residual_rms_frac, RESIDUAL_WARN_FRAC, RESIDUAL_FAIL_FRAC)
        residual_threshold = f"≤{RESIDUAL_WARN_FRAC * 100:.0f}% BW 正常，"
        residual_threshold += f">{RESIDUAL_FAIL_FRAC * 100:.0f}% BW 失败"
    checks.append(_check(
        "residual_force_rms_N", "残余力 RMS",
        residual_rms, residual_status, "N", residual_threshold,
        f"({residual_rms_frac * 100:.1f}% BW)" if residual_rms_frac is not None else "",
    ))
    checks.append(_check(
        "residual_force_p95_N", "残余力 p95",
        residual_p95,
        "WARN" if residual_p95_frac is not None and residual_p95_frac > RESIDUAL_P95_WARN_FRAC else "PASS",
        "N", f"≤{RESIDUAL_P95_WARN_FRAC * 100:.0f}% BW",
    ))

    # 髋力矩量级（仅展示，不做 PASS/FAIL 门槛）
    for side, label in (("r", "右髋屈伸"), ("l", "左髋屈伸")):
        entry = idq.get(f"hip_flexion_{side}") or {}
        checks.append(_check(
            f"hip_flexion_{side}_p95_Nm_per_kg", f"{label} p95",
            entry.get("p95_abs_Nm_per_kg"), "INFO", "Nm/kg", "—",
        ))

    # 同步质量 + 力覆盖（§3.3）
    checks.extend(_sync_checks(sync))
    checks.extend(_force_checks(force, dynamic_n_frames))
    # 静态 marker 调整分级（§3.6）
    checks.extend(_marker_adjustment_checks(marker_adjustment, marker_adjustment_expert_confirmed))

    status = "PASS"
    for c in checks:
        if _RANK[c["status"]] > _RANK[status]:
            status = c["status"]

    # 专家强制 offset：整份 QC 最多 WARN，绝不 PASS（§3.3 状态规则）。
    if sync and str(sync.get("method") or "") == "EXPERT_FORCED" and status == "PASS":
        status = "WARN"

    summary = _summarize(status, checks)
    return {
        "schema_version": QC_SCHEMA_VERSION,
        "status": status,
        "checks": checks,
        "summary": summary,
    }


def _summarize(status: str, checks: list[dict[str, Any]]) -> str:
    def _text(c: dict[str, Any]) -> str:
        if c["status"] == "FAIL" and c["value"] is None:
            return f"{c['label']}缺失"
        return f"{c['label']} {c['detail']}".strip()

    problems = [_text(c) for c in checks if c["status"] == "FAIL"]
    warnings = [_text(c) for c in checks if c["status"] == "WARN"]
    if problems:
        return "；".join(problems)
    if warnings:
        return "；".join(warnings)
    return "marker/残余力/同步/力覆盖均在阈值内"


__all__ = [
    "QC_SCHEMA_VERSION",
    "MARKER_ADJUST_REMIND_MM",
    "MARKER_ADJUST_WARN_MM",
    "MARKER_ADJUST_FAIL_MM",
    "grade_marker_adjustments",
    "evaluate_qc",
]
