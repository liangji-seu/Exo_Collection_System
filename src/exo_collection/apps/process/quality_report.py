"""按受试者/采集日汇总并绘制批量解算数据质量 PNG 报告。"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from PySide6.QtCore import QObject, QRect, QRunnable, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QImage, QPainter, QPen

from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path
from exo_collection.apps.calculate.models import SessionRecord


_G = 9.80665
_RANK = {"INFO": 0, "PASS": 1, "WARN": 2, "FAIL": 3, "NA": -1}
_COLORS = {
    "PASS": ("#dcfce7", "#16803c"),
    "WARN": ("#fff3c4", "#9a6700"),
    "FAIL": ("#fee4e2", "#b42318"),
    "INFO": ("#eaf2ff", "#175cd3"),
    "NA": ("#f2f4f7", "#667085"),
}
_FONT_FAMILY: str | None = None


def _report_font_family() -> str:
    """显式加载 Windows 中文字体，保证后台/offscreen 导出的 PNG 不出现方框。"""
    global _FONT_FAMILY
    if _FONT_FAMILY is not None:
        return _FONT_FAMILY
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                _FONT_FAMILY = families[0]
                return _FONT_FAMILY
    _FONT_FAMILY = "Microsoft YaHei UI"
    return _FONT_FAMILY


@dataclass(frozen=True, slots=True)
class QualityValue:
    text: str = "—"
    status: str = "NA"
    value: float | None = None


@dataclass(frozen=True, slots=True)
class SessionQualityRow:
    condition: str
    session: str
    kind: str
    solve_state: str
    qc: QualityValue
    marker_rms: QualityValue
    marker_peak: QualityValue
    residual: QualityValue
    force_coverage: QualityValue
    sync: QualityValue
    static_source: QualityValue
    hip_moment: QualityValue
    training: QualityValue


@dataclass(frozen=True, slots=True)
class DayQualityReport:
    subject: str
    day: str
    day_root: Path
    generated_at: str
    rows: tuple[SessionQualityRow, ...]
    sync_rechecked: bool


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _day_for_record(record: SessionRecord, data_root: Path) -> tuple[str, Path]:
    """返回 session 的 dX 标签及其目录；旧数据没有 dX 时归到“未分日”。"""
    try:
        relative = record.session_dir.resolve().relative_to(data_root.resolve())
        parts = relative.parts
    except (OSError, ValueError):
        parts = record.session_dir.parts

    try:
        subject_index = next(
            index for index, part in enumerate(parts) if part == record.subject_code
        )
    except StopIteration:
        return "未分日", record.session_dir

    if subject_index + 1 < len(parts) and re.fullmatch(
        r"d\d+", parts[subject_index + 1], flags=re.IGNORECASE
    ):
        day = parts[subject_index + 1]
        if record.session_dir.is_absolute():
            anchor_parts = record.session_dir.parts
            try:
                absolute_index = next(
                    index
                    for index, part in enumerate(anchor_parts)
                    if part == record.subject_code
                )
                return day, Path(*anchor_parts[: absolute_index + 2])
            except StopIteration:
                pass
        return day, data_root / record.subject_code / day
    return "未分日", data_root / record.subject_code


def group_subject_days(
    records: Iterable[SessionRecord], data_root: Path, subject: str
) -> dict[str, tuple[Path, list[SessionRecord]]]:
    grouped: dict[str, tuple[Path, list[SessionRecord]]] = {}
    for record in records:
        if record.subject_code != subject:
            continue
        day, root = _day_for_record(record, data_root)
        grouped.setdefault(day, (root, []))[1].append(record)
    for _, sessions in grouped.values():
        sessions.sort(
            key=lambda item: (
                item.condition_code or item.condition_name,
                item.repeat_index,
                item.session_name,
            )
        )
    return dict(
        sorted(
            grouped.items(),
            key=lambda item: (
                item[0] == "未分日",
                int(item[0][1:]) if re.fullmatch(r"d\d+", item[0], re.I) else 10**9,
                item[0],
            ),
        )
    )


def _ground_truth_run(record: SessionRecord) -> tuple[Path, str | None] | None:
    csv_path = record.session_dir / "ground_truth.csv"
    sidecar = csv_path.with_suffix(".qc.json")
    try:
        stat = csv_path.stat()
        document = _safe_json(sidecar)
        if document is None:
            return None
        if (document.get("csv_size"), document.get("csv_mtime_ns")) != (
            stat.st_size,
            stat.st_mtime_ns,
        ):
            return None
        run_dir = Path(str(document["run_dir"]))
        status = document.get("qc_status")
        return run_dir, status if status in {"PASS", "WARN", "FAIL"} else None
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _quality_from_check(
    checks: dict[str, dict[str, Any]], key: str, formatter: Callable[[float], str]
) -> QualityValue:
    check = checks.get(key)
    if not check:
        return QualityValue()
    value = check.get("value")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return QualityValue(status=str(check.get("status") or "NA"))
    if not math.isfinite(number):
        return QualityValue(status=str(check.get("status") or "NA"))
    return QualityValue(
        text=formatter(number),
        status=str(check.get("status") or "NA"),
        value=number,
    )


def _subject_from_static_path(path: str | None, data_root: Path) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(data_root.resolve())
        return relative.parts[0] if relative.parts else None
    except (OSError, ValueError):
        parts = candidate.parts
        for index, part in enumerate(parts[:-1]):
            if part.casefold() == data_root.name.casefold() and index + 1 < len(parts):
                return parts[index + 1]
    return None


def _stored_sync(checks: dict[str, dict[str, Any]]) -> QualityValue:
    relevant = [
        entry
        for key, entry in checks.items()
        if key.startswith("sync_") and entry.get("status") in _RANK
    ]
    if not relevant:
        return QualityValue("未评估", "WARN")
    status = max((str(item["status"]) for item in relevant), key=_RANK.get)
    pairs = checks.get("sync_n_pairs", {}).get("value")
    mad = checks.get("sync_mad_s", {}).get("value")
    details = []
    if pairs is not None:
        details.append(f"{float(pairs):.0f}对")
    if mad is not None:
        details.append(f"MAD {float(mad) * 1000:.1f}ms")
    return QualityValue(" · ".join(details) or "已读取旧QC", status)


def _fresh_sync(record: SessionRecord) -> QualityValue:
    files = record.files
    if not files.has_dynamic_inputs:
        return QualityValue("输入不齐全", "FAIL")
    ensure_pipeline_on_path()
    from pipeline.synchronization.sync import run_auto_sync

    try:
        result = run_auto_sync(
            files.c3d_path,
            files.mocap_h5_path,
            files.imu_h5_path,
            files.txt_path,
        )
    except Exception as exc:  # noqa: BLE001 -- 单个 session 失败必须进入报告
        reason = str(exc).replace("\n", " ")
        if len(reason) > 42:
            reason = reason[:39] + "…"
        return QualityValue(f"复核失败：{reason}", "FAIL")

    confidence = str(result.get("confidence") or "LOW").upper()
    pairs = int(result.get("n_pairs") or 0)
    imu_gaps = int(result.get("imu_clock_gaps") or 0)
    mocap_gaps = int(result.get("mocap_h5_clock_gaps") or 0)
    imu_rate = result.get("imu_sample_rate_hz")
    monotonic = bool(result.get("imu_clock_monotonic", False)) and bool(
        result.get("mocap_h5_monotonic", False)
    )
    if not monotonic or pairs < 3 or confidence == "LOW":
        status = "FAIL"
    elif confidence != "HIGH" or max(imu_gaps, mocap_gaps) > 5:
        status = "WARN"
    else:
        status = "PASS"
    rate_text = f"{float(imu_rate):.1f}Hz" if imu_rate is not None else "?Hz"
    return QualityValue(
        f"{confidence} · {pairs}对 · {rate_text} · gap {imu_gaps}/{mocap_gaps}",
        status,
        float(imu_gaps),
    )


def _worst_status(statuses: Iterable[str], default: str = "WARN") -> str:
    values = [status for status in statuses if status in _RANK and status != "INFO"]
    return max(values, key=_RANK.get) if values else default


def _row_from_record(
    record: SessionRecord,
    data_root: Path,
    *,
    recheck_sync: bool,
) -> SessionQualityRow:
    condition = record.condition_code or record.condition_name or "(未命名)"
    if record.is_stand:
        na = QualityValue()
        return SessionQualityRow(
            condition,
            record.session_name,
            "静态",
            "静态标定",
            na,
            na,
            na,
            na,
            na,
            na,
            QualityValue(record.subject_code, "INFO"),
            na,
            QualityValue("不生成力矩真值", "NA"),
        )

    ground_truth = record.session_dir / "ground_truth.csv"
    if not ground_truth.is_file():
        if record.files.c3d_path is None or record.files.txt_path is None:
            solve_state = "输入不齐全"
        elif record.files.mocap_h5_path is None or record.files.imu_h5_path is None:
            solve_state = "缺同步模态"
        else:
            solve_state = "未解算"
        na = QualityValue()
        return SessionQualityRow(
            condition,
            record.session_name,
            "动态",
            solve_state,
            QualityValue(solve_state, "INFO" if solve_state == "未解算" else "WARN"),
            na,
            na,
            na,
            na,
            na,
            na,
            na,
            QualityValue("解算后再判断", "NA"),
        )

    run_info = _ground_truth_run(record)
    if run_info is None:
        na = QualityValue()
        return SessionQualityRow(
            condition,
            record.session_name,
            "动态",
            "QC来源失效",
            QualityValue("QC未知", "WARN"),
            na,
            na,
            na,
            na,
            na,
            na,
            na,
            QualityValue("复核QC附带文件", "WARN"),
        )

    run_dir, sidecar_status = run_info
    qc_document = _safe_json(run_dir / "qc_report.json") or {}
    manifest = _safe_json(run_dir / "manifest.json") or {}
    check_list = qc_document.get("checks")
    checks = {
        str(item.get("key")): item
        for item in check_list
        if isinstance(item, dict) and item.get("key")
    } if isinstance(check_list, list) else {}

    marker_rms = _quality_from_check(
        checks, "marker_rms_mean_cm", lambda value: f"{value:.2f} cm"
    )
    marker_peak = _quality_from_check(
        checks, "marker_max_p95_cm", lambda value: f"{value:.2f} cm"
    )
    residual_n = _quality_from_check(
        checks, "residual_force_rms_N", lambda value: f"{value:.0f} N"
    )
    mass = (manifest.get("subject") or {}).get("mass_kg")
    residual_pct: float | None = None
    try:
        residual_pct = float(residual_n.value) / (float(mass) * _G) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    residual = QualityValue(
        (
            f"{residual_n.value:.0f} N / {residual_pct:.1f}%BW"
            if residual_n.value is not None and residual_pct is not None
            else residual_n.text
        ),
        residual_n.status,
        residual_pct,
    )
    coverage = _quality_from_check(
        checks, "force_coverage", lambda value: f"{value * 100:.1f}%"
    )
    right_hip = _quality_from_check(
        checks, "hip_flexion_r_p95_Nm_per_kg", lambda value: f"{value:.2f}"
    )
    left_hip = _quality_from_check(
        checks, "hip_flexion_l_p95_Nm_per_kg", lambda value: f"{value:.2f}"
    )
    hip_text = (
        f"右{right_hip.text} / 左{left_hip.text} Nm/kg"
        if right_hip.value is not None or left_hip.value is not None
        else "—"
    )
    hip = QualityValue(hip_text, "INFO", right_hip.value)

    sync = _fresh_sync(record) if recheck_sync else _stored_sync(checks)

    static_path = (((manifest.get("inputs") or {}).get("static_c3d") or {}).get("path"))
    static_subject = _subject_from_static_path(static_path, data_root)
    if static_subject is None:
        static = QualityValue("来源未知", "WARN")
    elif static_subject == record.subject_code:
        static = QualityValue(f"{static_subject} · 同受试者", "PASS")
    else:
        static = QualityValue(f"{static_subject} · 跨受试者", "FAIL")

    non_sync_statuses = [
        str(item.get("status"))
        for key, item in checks.items()
        if not key.startswith("sync_") and item.get("status") in {"PASS", "WARN", "FAIL"}
    ]
    if checks:
        effective_status = _worst_status([*non_sync_statuses, sync.status])
    else:
        effective_status = sidecar_status or "WARN"
    qc = QualityValue(effective_status, effective_status)

    reasons = []
    if effective_status == "FAIL":
        reasons.append("QC FAIL")
    if sync.status == "FAIL":
        reasons.append("同步复核失败")
    if static.status == "FAIL":
        reasons.append("跨受试者静态")
    if reasons:
        training = QualityValue("不可直接训练：" + "、".join(dict.fromkeys(reasons)), "FAIL")
    elif effective_status == "WARN" or static.status == "WARN":
        training = QualityValue("谨慎使用，先看黄色指标", "WARN")
    else:
        training = QualityValue("通过现有自动QC，可进入训练集", "PASS")

    return SessionQualityRow(
        condition,
        record.session_name,
        "动态",
        "已解算",
        qc,
        marker_rms,
        marker_peak,
        residual,
        coverage,
        sync,
        static,
        hip,
        training,
    )


def collect_day_report(
    records: Iterable[SessionRecord],
    data_root: Path,
    subject: str,
    day: str,
    day_root: Path,
    *,
    recheck_sync: bool = True,
    progress: Callable[[str], None] | None = None,
) -> DayQualityReport:
    records = list(records)
    rows: list[SessionQualityRow] = []
    for index, record in enumerate(records, start=1):
        if progress is not None:
            progress(f"质量报告 {subject}/{day}：分析 {index}/{len(records)} · {record.session_name}")
        rows.append(_row_from_record(record, data_root, recheck_sync=recheck_sync))
    return DayQualityReport(
        subject=subject,
        day=day,
        day_root=day_root,
        generated_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        rows=tuple(rows),
        sync_rechecked=recheck_sync,
    )


def _count_status(values: Iterable[QualityValue]) -> Counter[str]:
    return Counter(value.status for value in values if value.status in {"PASS", "WARN", "FAIL"})


def _numeric_summary(values: Iterable[QualityValue]) -> str:
    numbers = [float(value.value) for value in values if value.value is not None and math.isfinite(value.value)]
    if not numbers:
        return "—"
    return f"{min(numbers):.2f} / {statistics.median(numbers):.2f} / {max(numbers):.2f}"


def _status_text(counts: Counter[str]) -> str:
    return f"绿 {counts['PASS']}　黄 {counts['WARN']}　红 {counts['FAIL']}"


def _worst_from_counts(counts: Counter[str]) -> str:
    if counts["FAIL"]:
        return "FAIL"
    if counts["WARN"]:
        return "WARN"
    return "PASS" if counts["PASS"] else "NA"


class _ReportPainter:
    def __init__(self, image: QImage) -> None:
        self.image = image
        self.painter = QPainter(image)
        self.painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.margin = 56
        self.y = 48
        self.width = image.width() - 2 * self.margin
        self.font_family = _report_font_family()

    def finish(self) -> None:
        self.painter.end()

    def font(self, size: int, bold: bool = False) -> QFont:
        value = QFont(self.font_family, size)
        value.setBold(bold)
        return value

    def text(self, text: str, size: int, color: str = "#172033", *, bold: bool = False) -> None:
        self.painter.setFont(self.font(size, bold))
        self.painter.setPen(QColor(color))
        self.painter.drawText(self.margin, self.y, self.width, 46, Qt.AlignmentFlag.AlignLeft, text)

    def section(self, title: str) -> None:
        self.y += 28
        self.text(title, 17, bold=True)
        self.y += 44

    def badge(self, x: int, text: str, status: str, width: int) -> None:
        bg, fg = _COLORS.get(status, _COLORS["NA"])
        rect = QRect(x, self.y, width, 34)
        self.painter.setPen(Qt.PenStyle.NoPen)
        self.painter.setBrush(QColor(bg))
        self.painter.drawRoundedRect(rect, 10, 10)
        self.painter.setFont(self.font(10, True))
        self.painter.setPen(QColor(fg))
        self.painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def cards(self, cards: list[tuple[str, str, str, str]]) -> None:
        gap = 16
        card_w = (self.width - gap * 3) // 4
        card_h = 108
        for index, (label, value, note, status) in enumerate(cards):
            row, col = divmod(index, 4)
            x = self.margin + col * (card_w + gap)
            y = self.y + row * (card_h + gap)
            bg, fg = _COLORS.get(status, _COLORS["NA"])
            rect = QRect(x, y, card_w, card_h)
            self.painter.setPen(QPen(QColor("#dfe5ee"), 1))
            self.painter.setBrush(QColor("#ffffff"))
            self.painter.drawRoundedRect(rect, 12, 12)
            self.painter.fillRect(QRect(x, y, 7, card_h), QColor(fg))
            self.painter.setPen(QColor("#475467"))
            self.painter.setFont(self.font(10))
            self.painter.drawText(QRect(x + 20, y + 12, card_w - 32, 24), Qt.AlignmentFlag.AlignLeft, label)
            self.painter.setPen(QColor(fg))
            self.painter.setFont(self.font(22, True))
            self.painter.drawText(QRect(x + 20, y + 36, card_w - 32, 36), Qt.AlignmentFlag.AlignLeft, value)
            self.painter.setPen(QColor("#667085"))
            self.painter.setFont(self.font(9))
            self.painter.drawText(QRect(x + 20, y + 78, card_w - 32, 20), Qt.AlignmentFlag.AlignLeft, note)
        self.y += math.ceil(len(cards) / 4) * (card_h + gap)

    def notice(self, text: str, status: str = "INFO", height: int = 68) -> None:
        bg, fg = _COLORS.get(status, _COLORS["INFO"])
        rect = QRect(self.margin, self.y, self.width, height)
        self.painter.setPen(QPen(QColor(fg), 1))
        self.painter.setBrush(QColor(bg))
        self.painter.drawRoundedRect(rect, 10, 10)
        self.painter.setPen(QColor(fg))
        self.painter.setFont(self.font(11, True))
        self.painter.drawText(
            rect.adjusted(18, 10, -18, -10),
            Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )
        self.y += height

    def table(
        self,
        headers: list[str],
        widths: list[int],
        rows: list[list[tuple[str, str]]],
        *,
        row_height: int = 40,
    ) -> None:
        x0 = self.margin
        header_h = 44
        self.painter.setFont(self.font(10, True))
        x = x0
        for header, width in zip(headers, widths, strict=True):
            rect = QRect(x, self.y, width, header_h)
            self.painter.fillRect(rect, QColor("#26344d"))
            self.painter.setPen(QColor("#ffffff"))
            self.painter.drawText(rect.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, header)
            x += width
        self.y += header_h

        for row_index, row in enumerate(rows):
            x = x0
            for (value, status), width in zip(row, widths, strict=True):
                rect = QRect(x, self.y, width, row_height)
                bg, fg = _COLORS.get(status, ("#ffffff", "#344054"))
                if status == "PLAIN":
                    bg = "#ffffff" if row_index % 2 == 0 else "#f8fafc"
                    fg = "#344054"
                self.painter.fillRect(rect, QColor(bg))
                self.painter.setPen(QPen(QColor("#dfe5ee"), 1))
                self.painter.drawRect(rect)
                self.painter.setFont(self.font(9, status in {"PASS", "WARN", "FAIL"}))
                self.painter.setPen(QColor(fg))
                metrics = QFontMetrics(self.painter.font())
                shown = metrics.elidedText(value, Qt.TextElideMode.ElideRight, width - 14)
                self.painter.drawText(rect.adjusted(7, 0, -7, 0), Qt.AlignmentFlag.AlignVCenter, shown)
                x += width
            self.y += row_height


def _metric_rows(solved: list[SessionQualityRow]) -> list[list[tuple[str, str]]]:
    specifications = [
        ("Marker RMS 均值", [row.marker_rms for row in solved], "cm"),
        ("单 Marker 峰值 p95", [row.marker_peak for row in solved], "cm"),
        ("残余力 RMS / BW", [row.residual for row in solved], "%BW"),
        ("左右力有效覆盖", [row.force_coverage for row in solved], "比例"),
        ("独立 MTw 同步复核", [row.sync for row in solved], "—"),
        ("静态标定来源", [row.static_source for row in solved], "—"),
        ("右髋力矩 p95", [row.hip_moment for row in solved], "Nm/kg"),
    ]
    rendered = []
    for label, values, unit in specifications:
        counts = _count_status(values)
        status = _worst_from_counts(counts)
        summary = _numeric_summary(values)
        if label == "左右力有效覆盖" and summary != "—":
            numbers = [float(value.value) * 100 for value in values if value.value is not None]
            summary = f"{min(numbers):.1f} / {statistics.median(numbers):.1f} / {max(numbers):.1f}"
            unit = "%"
        rendered.append([
            (label, "PLAIN"),
            (_status_text(counts), status),
            (summary, status if summary != "—" else "NA"),
            (unit, "PLAIN"),
        ])
    return rendered


def _condition_rows(solved: list[SessionQualityRow]) -> list[list[tuple[str, str]]]:
    groups: dict[str, list[SessionQualityRow]] = defaultdict(list)
    for row in solved:
        groups[row.condition].append(row)
    output = []
    for condition, rows in sorted(groups.items()):
        qc_counts = _count_status(row.qc for row in rows)
        training_counts = _count_status(row.training for row in rows)
        marker = _numeric_summary(row.marker_rms for row in rows)
        residual = _numeric_summary(row.residual for row in rows)
        coverage_values = [row.force_coverage.value for row in rows if row.force_coverage.value is not None]
        coverage = (
            f"{statistics.median(coverage_values) * 100:.1f}%" if coverage_values else "—"
        )
        marker_values = [row.marker_rms.value for row in rows if row.marker_rms.value is not None]
        residual_values = [row.residual.value for row in rows if row.residual.value is not None]
        marker_median = statistics.median(marker_values) if marker_values else None
        residual_median = statistics.median(residual_values) if residual_values else None
        coverage_median = statistics.median(coverage_values) if coverage_values else None
        marker_status = (
            "PASS" if marker_median is not None and marker_median <= 2.0
            else "WARN" if marker_median is not None and marker_median <= 4.0
            else "FAIL" if marker_median is not None else "NA"
        )
        residual_status = (
            "PASS" if residual_median is not None and residual_median <= 15.0
            else "WARN" if residual_median is not None and residual_median <= 30.0
            else "FAIL" if residual_median is not None else "NA"
        )
        coverage_status = (
            "PASS" if coverage_median is not None and coverage_median >= 0.80
            else "WARN" if coverage_median is not None and coverage_median >= 0.50
            else "FAIL" if coverage_median is not None else "NA"
        )
        qc_status = _worst_from_counts(qc_counts)
        output.append([
            (condition, "PLAIN"),
            (str(len(rows)), "PLAIN"),
            (_status_text(qc_counts), qc_status),
            (marker.split(" / ")[1] + " cm" if " / " in marker else marker, marker_status),
            (residual.split(" / ")[1] + " %BW" if " / " in residual else residual, residual_status),
            (coverage, coverage_status),
            (_status_text(training_counts), _worst_from_counts(training_counts)),
        ])
    return output


def render_quality_report_png(report: DayQualityReport, output_path: Path) -> Path:
    """用 Qt 原生绘图输出可直接查看的长图 PNG。"""
    solved = [row for row in report.rows if row.solve_state == "已解算"]
    dynamic = [row for row in report.rows if row.kind == "动态"]
    static = [row for row in report.rows if row.kind == "静态"]
    qc_counts = _count_status(row.qc for row in solved)
    training_counts = _count_status(row.training for row in solved)
    unsolved = len(dynamic) - len(solved)
    cross_static = sum(row.static_source.status == "FAIL" for row in solved)
    sync_failed = sum(row.sync.status == "FAIL" for row in solved)

    metric_table = _metric_rows(solved)
    condition_table = _condition_rows(solved)
    session_row_h = 42
    height = (
        48 + 62 + 42 + 68 + 32 + 2 * 124 + 72
        + 44 + len(metric_table) * 40
        + 72 + 44 + len(condition_table) * 40
        + 72 + 44 + len(report.rows) * session_row_h
        + 150
    )
    image = QImage(2460, max(height, 1200), QImage.Format.Format_ARGB32)
    image.fill(QColor("#f4f7fb"))
    canvas = _ReportPainter(image)
    try:
        canvas.text(f"受试者 {report.subject} / {report.day} 数据质量报告", 25, bold=True)
        canvas.y += 48
        canvas.text(
            f"生成时间：{report.generated_at}　逐 session 指标来自各 ground_truth.csv 对应解算；"
            + ("同步已按三路独立 MTw 有效样本重新复核。" if report.sync_rechecked else "同步读取既有 QC。"),
            10,
            "#667085",
        )
        canvas.y += 42
        conclusion_status = "FAIL" if training_counts["FAIL"] else "WARN" if training_counts["WARN"] else "PASS"
        conclusion = (
            f"总体：已解算 {len(solved)}/{len(dynamic)} 个动态 session；"
            f"现有规则下可进入训练集 {training_counts['PASS']}，谨慎使用 {training_counts['WARN']}，"
            f"不可直接训练 {training_counts['FAIL']}。"
        )
        if cross_static:
            conclusion += f" 其中 {cross_static} 个使用了跨受试者静态标定。"
        canvas.notice(conclusion, conclusion_status)

        canvas.section("总体统计")
        canvas.cards([
            ("全部 session", str(len(report.rows)), f"动态 {len(dynamic)} / 静态 {len(static)}", "INFO"),
            ("已解算动态", str(len(solved)), f"未解算或不齐全 {unsolved}", "INFO"),
            ("有效 QC PASS", str(qc_counts["PASS"]), "已替换旧 MTw 时钟误判", "PASS"),
            ("有效 QC WARN", str(qc_counts["WARN"]), "存在黄色指标", "WARN"),
            ("有效 QC FAIL", str(qc_counts["FAIL"]), "至少一个红色指标", "FAIL"),
            ("同步复核失败", str(sync_failed), "需人工同步或重采", "FAIL" if sync_failed else "PASS"),
            ("跨受试者静态", str(cross_static), "不作为直接训练真值", "FAIL" if cross_static else "PASS"),
            ("可进入训练集", str(training_counts["PASS"]), "仍建议抽查波形", "PASS" if training_counts["PASS"] else "WARN"),
        ])

        canvas.section("指标汇总（最小值 / 中位数 / 最大值）")
        canvas.table(
            ["指标", "颜色统计", "数值统计", "单位"],
            [520, 520, 520, 280],
            metric_table,
        )

        canvas.section("按工况汇总（已解算动态 session）")
        canvas.table(
            ["工况", "数量", "有效 QC", "Marker RMS中位", "残余力中位", "力覆盖中位", "训练建议"],
            [430, 120, 350, 260, 260, 220, 400],
            condition_table or [[("暂无已解算动态 session", "NA"), ("—", "NA"), ("—", "NA"), ("—", "NA"), ("—", "NA"), ("—", "NA"), ("—", "NA")]],
        )

        canvas.section("逐 session 指标")
        session_rows: list[list[tuple[str, str]]] = []
        for row in report.rows:
            session_rows.append([
                (row.condition, "PLAIN"),
                (row.session, "PLAIN"),
                (row.kind, "PLAIN"),
                (row.qc.text if row.solve_state == "已解算" else row.solve_state, row.qc.status),
                (row.marker_rms.text, row.marker_rms.status),
                (row.marker_peak.text, row.marker_peak.status),
                (row.residual.text, row.residual.status),
                (row.force_coverage.text, row.force_coverage.status),
                (row.sync.text, row.sync.status),
                (row.static_source.text, row.static_source.status),
                (row.hip_moment.text, row.hip_moment.status),
                (row.training.text, row.training.status),
            ])
        canvas.table(
            ["工况", "Session", "类型", "有效QC", "Marker RMS", "Marker峰值", "残余力", "力覆盖", "同步复核", "静态来源", "髋力矩p95", "训练建议"],
            [260, 330, 90, 120, 130, 130, 190, 120, 300, 170, 250, 260],
            session_rows,
            row_height=session_row_h,
        )

        canvas.y += 28
        canvas.notice(
            "颜色说明：绿色=通过当前阈值；黄色=可保留但需谨慎；红色=必须复核或不能直接作为训练真值；"
            "灰色=未评估。髋力矩 p95 只展示量级，不参与 PASS/FAIL。QC 是自动筛查，不能替代波形抽查。",
            "INFO",
            82,
        )
    finally:
        canvas.finish()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(output_path), "PNG"):
        raise OSError(f"无法写入 PNG：{output_path}")
    return output_path


def export_subject_reports(
    records: Iterable[SessionRecord],
    data_root: Path,
    subject: str,
    *,
    recheck_sync: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    grouped = group_subject_days(records, data_root, subject)
    for day, (day_root, day_records) in grouped.items():
        report = collect_day_report(
            day_records,
            data_root,
            subject,
            day,
            day_root,
            recheck_sync=recheck_sync,
            progress=progress,
        )
        safe_day = re.sub(r"[^0-9A-Za-z_-]+", "_", day)
        output = day_root / f"quality_report_{subject}_{safe_day}.png"
        render_quality_report_png(report, output)
        outputs.append(output)
        if progress is not None:
            progress(f"已导出质量报告：{output}")
    return outputs


class _QualityReportSignals(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)


class QualityReportWorker(QRunnable):
    """后台生成报告，避免重新复核同步时阻塞 run_process 界面。"""

    def __init__(self, records: Iterable[SessionRecord], data_root: Path, subject: str) -> None:
        super().__init__()
        # Font registration belongs to the GUI thread. The worker only reuses
        # the resolved family when it later paints a QImage.
        _report_font_family()
        self._records = list(records)
        self._data_root = Path(data_root)
        self._subject = subject
        self.signals = _QualityReportSignals()

    @Slot()
    def run(self) -> None:
        try:
            outputs = export_subject_reports(
                self._records,
                self._data_root,
                self._subject,
                recheck_sync=True,
                progress=self.signals.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001 -- QRunnable 边界
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(outputs)


__all__ = [
    "DayQualityReport",
    "QualityReportWorker",
    "QualityValue",
    "SessionQualityRow",
    "collect_day_report",
    "export_subject_reports",
    "group_subject_days",
    "render_quality_report_png",
]
