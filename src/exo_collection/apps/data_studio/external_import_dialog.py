"""Parameter dialog for immutable external force-plate/mocap annex imports."""

from __future__ import annotations

from pathlib import Path
import re

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from exo_collection.external import ExternalImportRequest, ExternalModality


class ExternalImportDialog(QDialog):
    """Collect generic file and pulse mapping inputs without vendor assumptions."""

    def __init__(self, manifest_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.setObjectName("external_import_dialog")
        self.setWindowTitle("导入外部模态（不可变附录）")
        self.resize(720, 690)
        outer = QVBoxLayout(self)

        explanation = QLabel(
            "导入结果写入数据根目录 external_annexes，使用 Trial UUID 和基准 Manifest "
            "SHA-256 绑定；不会改写已最终化 Trial、Manifest 或任何原始数据。"
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        form = QFormLayout()
        manifest_edit = QLineEdit(str(self.manifest_path))
        manifest_edit.setReadOnly(True)
        manifest_edit.setObjectName("external_manifest_path")
        form.addRow("目标 Trial：", manifest_edit)

        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        self.source_edit = QLineEdit()
        self.source_edit.setObjectName("external_source_path")
        self.source_edit.setPlaceholderText("选择测力台、动作捕捉或其他外部导出文件")
        source_layout.addWidget(self.source_edit, 1)
        source_button = QPushButton("选择…")
        source_button.clicked.connect(self._choose_source)
        source_layout.addWidget(source_button)
        form.addRow("外部原文件：", source_row)

        self.modality_combo = QComboBox()
        self.modality_combo.setObjectName("external_modality")
        self.modality_combo.addItem("测力台", ExternalModality.FORCE_PLATE.value)
        self.modality_combo.addItem("动作捕捉", ExternalModality.MOCAP.value)
        self.modality_combo.addItem("其他", ExternalModality.OTHER.value)
        self.modality_combo.currentIndexChanged.connect(self._apply_modality)
        form.addRow("模态：", self.modality_combo)

        self.other_modality_edit = QLineEdit()
        self.other_modality_edit.setObjectName("external_other_modality")
        self.other_modality_edit.setPlaceholderText("选择“其他”时必填")
        form.addRow("其他模态名称：", self.other_modality_edit)

        self.source_system_edit = QLineEdit("manual_external_import")
        self.source_system_edit.setObjectName("external_source_system")
        self.source_system_edit.setPlaceholderText("设备/软件来源，不虚构厂商协议")
        form.addRow("来源系统：", self.source_system_edit)

        self.clock_domain_edit = QLineEdit("external_clock")
        self.clock_domain_edit.setObjectName("external_clock_domain")
        form.addRow("外部时钟域：", self.clock_domain_edit)

        self.time_unit_combo = QComboBox()
        self.time_unit_combo.setObjectName("external_time_unit")
        for label, value in (
            ("秒 (s)", "s"),
            ("毫秒 (ms)", "ms"),
            ("微秒 (us)", "us"),
            ("纳秒 (ns)", "ns"),
        ):
            self.time_unit_combo.addItem(label, value)
        form.addRow("脉冲时间单位：", self.time_unit_combo)
        outer.addLayout(form)

        pulse_group = QGroupBox("外部同步脉冲")
        pulse_form = QFormLayout(pulse_group)
        self.pulse_mode_combo = QComboBox()
        self.pulse_mode_combo.setObjectName("external_pulse_mode")
        self.pulse_mode_combo.addItem("手工输入时间序列", "manual")
        self.pulse_mode_combo.addItem("从 CSV 列读取", "csv")
        self.pulse_mode_combo.currentIndexChanged.connect(self._apply_pulse_mode)
        pulse_form.addRow("输入方式：", self.pulse_mode_combo)

        self.manual_pulses_edit = QPlainTextEdit()
        self.manual_pulses_edit.setObjectName("external_manual_pulses")
        self.manual_pulses_edit.setMaximumHeight(90)
        self.manual_pulses_edit.setPlaceholderText(
            "按严格递增顺序输入，例如：0.0, 1.0, 2.0；数量必须与 Trial 内部正式窗口脉冲一致"
        )
        pulse_form.addRow("脉冲时间：", self.manual_pulses_edit)

        csv_row = QWidget()
        csv_layout = QHBoxLayout(csv_row)
        csv_layout.setContentsMargins(0, 0, 0, 0)
        self.pulse_csv_edit = QLineEdit()
        self.pulse_csv_edit.setObjectName("external_pulse_csv_path")
        self.pulse_csv_edit.setPlaceholderText("可留空：直接从外部原文件读取 CSV")
        csv_layout.addWidget(self.pulse_csv_edit, 1)
        csv_button = QPushButton("选择…")
        csv_button.clicked.connect(self._choose_pulse_csv)
        csv_layout.addWidget(csv_button)
        pulse_form.addRow("脉冲 CSV：", csv_row)

        self.pulse_column_edit = QLineEdit()
        self.pulse_column_edit.setObjectName("external_pulse_csv_column")
        self.pulse_column_edit.setPlaceholderText("例如 trigger_time")
        pulse_form.addRow("时间列名：", self.pulse_column_edit)

        self.delimiter_edit = QLineEdit()
        self.delimiter_edit.setObjectName("external_csv_delimiter")
        self.delimiter_edit.setMaxLength(2)
        self.delimiter_edit.setPlaceholderText("留空自动识别；可输入 , ; | 或 \\t")
        pulse_form.addRow("CSV 分隔符：", self.delimiter_edit)

        self.encoding_combo = QComboBox()
        self.encoding_combo.setObjectName("external_csv_encoding")
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            self.encoding_combo.addItem(encoding, encoding)
        pulse_form.addRow("CSV 编码：", self.encoding_combo)
        outer.addWidget(pulse_group)

        warning = QLabel(
            "提示：同步映射只使用 Trial 正式记录窗口内保存的内部上升沿。若脉冲数量不一致，"
            "导入会失败且不会发布不完整附录。"
        )
        warning.setWordWrap(True)
        outer.addWidget(warning)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始导入")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._apply_modality()
        self._apply_pulse_mode()

    def _trial_root(self) -> Path:
        parent = self.manifest_path.parent
        if parent.name == ".exo":
            return parent.parent
        return parent

    def _choose_source(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择外部模态原文件",
            str(self._trial_root()),
            "所有文件 (*.*)",
        )
        if path:
            self.source_edit.setText(path)

    def _choose_pulse_csv(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择同步脉冲 CSV",
            str(self._trial_root()),
            "CSV/文本 (*.csv *.txt);;所有文件 (*.*)",
        )
        if path:
            self.pulse_csv_edit.setText(path)

    def _apply_modality(self) -> None:
        is_other = self.modality_combo.currentData() == ExternalModality.OTHER.value
        self.other_modality_edit.setEnabled(is_other)

    def _apply_pulse_mode(self) -> None:
        manual = self.pulse_mode_combo.currentData() == "manual"
        self.manual_pulses_edit.setEnabled(manual)
        self.pulse_csv_edit.setEnabled(not manual)
        self.pulse_column_edit.setEnabled(not manual)
        self.delimiter_edit.setEnabled(not manual)
        self.encoding_combo.setEnabled(not manual)

    @staticmethod
    def _parse_manual_pulses(text: str) -> list[float]:
        tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
        if not tokens:
            raise ValueError("请至少输入一个外部同步脉冲时间。")
        try:
            return [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("手工脉冲时间必须全部为数值。") from exc

    def take_request(self, dataset_root: str | Path) -> ExternalImportRequest:
        source_text = self.source_edit.text().strip()
        if not source_text:
            raise ValueError("请选择外部模态原文件。")
        modality = ExternalModality(str(self.modality_combo.currentData()))
        manual = self.pulse_mode_combo.currentData() == "manual"
        values: dict[str, object] = {
            "dataset_root": Path(dataset_root).expanduser().resolve(),
            "trial_manifest_path": self.manifest_path,
            "source_path": Path(source_text).expanduser(),
            "modality": modality,
            "source_system": self.source_system_edit.text().strip(),
            "other_modality_label": (
                self.other_modality_edit.text().strip() if modality is ExternalModality.OTHER else None
            ),
            "external_clock_domain": self.clock_domain_edit.text().strip(),
            "external_time_unit": str(self.time_unit_combo.currentData()),
            "csv_encoding": str(self.encoding_combo.currentData()),
        }
        if manual:
            values["external_pulse_times"] = self._parse_manual_pulses(
                self.manual_pulses_edit.toPlainText()
            )
        else:
            csv_text = self.pulse_csv_edit.text().strip()
            delimiter = self.delimiter_edit.text()
            if delimiter == r"\t":
                delimiter = "\t"
            values.update(
                {
                    "pulse_csv_path": Path(csv_text).expanduser() if csv_text else None,
                    "pulse_csv_column": self.pulse_column_edit.text().strip(),
                    "csv_delimiter": delimiter or None,
                }
            )
        return ExternalImportRequest.model_validate(values)


__all__ = ["ExternalImportDialog"]
