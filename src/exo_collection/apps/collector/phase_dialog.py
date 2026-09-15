"""期次配置对话框。

为 ``run_collector`` 提供「工况设置」界面，编辑三层结构：期次 → 主工况（4 类固定）
→ 详细工况（每条可勾选「增加穿戴」）。期次仅用于采集端下拉分组，不写入 manifest。
"""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.domain.condition_phases import (
    MAIN_CATEGORIES,
    MAIN_CATEGORY_KEYS,
    MAIN_CATEGORY_NAMES,
    PHASE_CONFIG_SCHEMA_VERSION,
    base_catalog,
)


class PhaseConfigDialog(QDialog):
    """编辑期次及每期各主工况下的详细工况（含穿戴开关）。"""

    def __init__(self, config: Mapping[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("工况设置（期次）")
        self.setMinimumSize(840, 600)
        self._catalog = base_catalog()
        self._validated_config: dict[str, Any] | None = None

        # 工作副本：期次列表，每期 {"name", "categories": {key: {"name", "details"}}}。
        self._phases: list[dict[str, Any]] = []
        phases_raw = config.get("phases") if isinstance(config, dict) else None
        if isinstance(phases_raw, list):
            for phase in phases_raw:
                if not isinstance(phase, dict):
                    continue
                name = str(phase.get("name") or "").strip()
                if not name:
                    continue
                self._phases.append(
                    {"name": name, "categories": self._normalize_categories(phase)}
                )

        self._build_ui()
        if self._phases:
            self._phase_list.setCurrentRow(0)
        else:
            self._sync_all_tabs()

    def _normalize_categories(self, phase: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        categories: dict[str, dict[str, Any]] = {}
        raw = phase.get("categories")
        if isinstance(raw, dict):
            for key in MAIN_CATEGORY_KEYS:
                cat = raw.get(key)
                details: list[dict[str, Any]] = []
                if isinstance(cat, dict):
                    for detail in cat.get("details") or []:
                        if not isinstance(detail, dict):
                            continue
                        code = str(detail.get("code") or "").strip().upper()
                        info = self._catalog.get(code)
                        if info is None:
                            continue
                        if info["has_noexo"]:
                            details.append({"code": code, "wear": bool(detail.get("wear"))})
                        elif info["has_exo"]:
                            details.append({"code": code, "wear": True})
                        else:
                            details.append({"code": code})
                categories[key] = {"name": MAIN_CATEGORY_NAMES[key], "details": details}
        else:
            for key in MAIN_CATEGORY_KEYS:
                categories[key] = {"name": MAIN_CATEGORY_NAMES[key], "details": []}
        return categories

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        body = QHBoxLayout()

        # 左：期次列表 + 管理按钮
        left = QVBoxLayout()
        left.addWidget(QLabel("期次："))
        self._phase_list = QListWidget()
        self._phase_list.setMinimumWidth(170)
        self._phase_list.setMaximumWidth(220)
        for phase in self._phases:
            self._phase_list.addItem(phase["name"])
        self._phase_list.currentRowChanged.connect(self._on_phase_changed)
        left.addWidget(self._phase_list, 1)

        self._add_phase_button = QPushButton("新增期次")
        self._add_phase_button.clicked.connect(self._add_phase)
        self._rename_phase_button = QPushButton("重命名")
        self._rename_phase_button.clicked.connect(self._rename_phase)
        self._remove_phase_button = QPushButton("删除期次")
        self._remove_phase_button.clicked.connect(self._remove_phase)
        phase_buttons = QHBoxLayout()
        phase_buttons.addWidget(self._add_phase_button)
        phase_buttons.addWidget(self._rename_phase_button)
        phase_buttons.addWidget(self._remove_phase_button)
        left.addLayout(phase_buttons)
        body.addLayout(left)

        # 右：四个主工况标签页
        self._tabs = QTabWidget()
        self._available_lists: dict[str, QListWidget] = {}
        self._included_tables: dict[str, QTableWidget] = {}
        for cat in MAIN_CATEGORIES:
            key = cat["key"]
            page = QWidget()
            self._tabs.addTab(page, cat["name"])
            self._build_category_page(page, key)
        body.addWidget(self._tabs, 1)

        outer.addLayout(body, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.accept)
        buttons.addWidget(cancel_button)
        buttons.addWidget(ok_button)
        outer.addLayout(buttons)

    def _build_category_page(self, page: QWidget, key: str) -> None:
        layout = QHBoxLayout(page)
        available = QListWidget()
        available.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["详细工况", "增加穿戴"])
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(1, 96)
        self._available_lists[key] = available
        self._included_tables[key] = table

        move_in = QPushButton("→ 纳入")
        move_in.clicked.connect(lambda *_: self._move_in(key))
        move_out = QPushButton("← 移出")
        move_out.clicked.connect(lambda *_: self._move_out(key))
        move_buttons = QVBoxLayout()
        move_buttons.addStretch(1)
        move_buttons.addWidget(move_in)
        move_buttons.addWidget(move_out)
        move_buttons.addStretch(1)

        layout.addLayout(self._list_column("未纳入工况", available), 1)
        layout.addLayout(move_buttons)
        layout.addLayout(self._list_column("已纳入工况", table), 1)

    @staticmethod
    def _list_column(title: str, widget: QWidget) -> QVBoxLayout:
        column = QVBoxLayout()
        column.addWidget(QLabel(title))
        column.addWidget(widget, 1)
        return column

    # ── 期次管理 ──────────────────────────────────────────────────────

    def _current_phase_index(self) -> int:
        return self._phase_list.currentRow()

    def _current_phase(self) -> dict[str, Any] | None:
        row = self._current_phase_index()
        if row < 0 or row >= len(self._phases):
            return None
        return self._phases[row]

    @Slot(int)
    def _on_phase_changed(self, *_args: object) -> None:
        self._sync_all_tabs()

    @Slot()
    def _add_phase(self) -> None:
        name, ok = QInputDialog.getText(
            self, "新增期次", "期次名称：", text=f"第{len(self._phases) + 1}期"
        )
        if not ok or not name.strip():
            return
        self._phases.append(
            {"name": name.strip(), "categories": self._empty_categories()}
        )
        self._phase_list.addItem(name.strip())
        self._phase_list.setCurrentRow(self._phase_list.count() - 1)

    @Slot()
    def _rename_phase(self) -> None:
        row = self._current_phase_index()
        if row < 0:
            return
        name, ok = QInputDialog.getText(
            self, "重命名期次", "期次名称：", text=self._phases[row]["name"]
        )
        if not ok or not name.strip():
            return
        self._phases[row]["name"] = name.strip()
        self._phase_list.item(row).setText(name.strip())

    @Slot()
    def _remove_phase(self) -> None:
        row = self._current_phase_index()
        if row < 0:
            return
        phase = self._phases[row]
        total = sum(
            len(cat["details"]) for cat in phase["categories"].values()
        )
        if total:
            reply = QMessageBox.question(
                self,
                "删除期次",
                f"期次「{phase['name']}」包含 {total} 个详细工况，确认删除？",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._phases.pop(row)
        self._phase_list.takeItem(row)
        self._sync_all_tabs()

    # ── 详细工况两栏同步 / 移动 ────────────────────────────────────────

    def _empty_categories(self) -> dict[str, dict[str, Any]]:
        return {
            key: {"name": MAIN_CATEGORY_NAMES[key], "details": []}
            for key in MAIN_CATEGORY_KEYS
        }

    def _sync_all_tabs(self) -> None:
        for key in MAIN_CATEGORY_KEYS:
            self._sync_tab(key)

    def _sync_tab(self, key: str) -> None:
        phase = self._current_phase()
        details = (phase["categories"][key]["details"] if phase else [])
        included = {detail["code"] for detail in details}

        available = [
            code
            for code, info in self._catalog.items()
            if info["category"] == key and code not in included
        ]
        self._fill_available(key, available)
        self._fill_included(key, details)

    def _fill_available(self, key: str, codes: list[str]) -> None:
        widget = self._available_lists[key]
        widget.clear()
        for code in codes:
            item = QListWidgetItem(self._catalog[code]["name"])
            item.setData(Qt.ItemDataRole.UserRole, code)
            item.setToolTip(code)
            widget.addItem(item)

    def _fill_included(self, key: str, details: list[dict[str, Any]]) -> None:
        table = self._included_tables[key]
        table.setRowCount(0)
        for row, detail in enumerate(details):
            code = detail["code"]
            info = self._catalog[code]
            table.insertRow(row)
            name_item = QTableWidgetItem(info["name"])
            name_item.setData(Qt.ItemDataRole.UserRole, code)
            name_item.setToolTip(code)
            name_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            table.setItem(row, 0, name_item)
            table.setCellWidget(row, 1, self._wear_widget(key, code, detail))

    def _wear_widget(self, key: str, code: str, detail: dict[str, Any]) -> QWidget:
        info = self._catalog[code]
        if not info["has_noexo"] and not info["has_exo"]:
            label = QLabel("—")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return label
        checkbox = QCheckBox()
        if info["has_noexo"] and info["has_exo"]:
            checkbox.setChecked(bool(detail.get("wear")))
            checkbox.setToolTip("勾选后同时纳入穿戴版本")
            checkbox.clicked.connect(
                lambda checked, k=key, c=code: self._set_wear(k, c, checked)
            )
        else:
            # 仅有穿戴版本，恒勾选且不可改。
            checkbox.setChecked(True)
            checkbox.setEnabled(False)
            checkbox.setToolTip("该工况仅有穿戴版本")
        return checkbox

    def _set_wear(self, key: str, code: str, checked: bool) -> None:
        phase = self._current_phase()
        if phase is None:
            return
        for detail in phase["categories"][key]["details"]:
            if detail["code"] == code:
                detail["wear"] = checked
                return

    @staticmethod
    def _selected_codes(widget: QListWidget) -> list[str]:
        return [
            widget.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(widget.count())
            if widget.item(index).isSelected()
        ]

    def _selected_table_codes(self, table: QTableWidget) -> list[str]:
        codes: list[str] = []
        for row in range(table.rowCount()):
            if table.item(row, 0).isSelected():
                codes.append(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        return codes

    @Slot()
    def _move_in(self, key: str) -> None:
        phase = self._current_phase()
        if phase is None:
            return
        details = phase["categories"][key]["details"]
        existing = {detail["code"] for detail in details}
        for code in self._selected_codes(self._available_lists[key]):
            if code in existing:
                continue
            info = self._catalog[code]
            if info["has_noexo"]:
                details.append({"code": code, "wear": False})
            elif info["has_exo"]:
                details.append({"code": code, "wear": True})
            else:
                details.append({"code": code})
            existing.add(code)
        self._sync_tab(key)

    @Slot()
    def _move_out(self, key: str) -> None:
        phase = self._current_phase()
        if phase is None:
            return
        remove = set(self._selected_table_codes(self._included_tables[key]))
        phase["categories"][key]["details"] = [
            detail
            for detail in phase["categories"][key]["details"]
            if detail["code"] not in remove
        ]
        self._sync_tab(key)

    # ── 保存 ──────────────────────────────────────────────────────────

    def _build_config(self) -> dict[str, Any]:
        if not self._phases:
            raise ValueError("至少需要保留一个期次")
        for phase in self._phases:
            if not phase["name"].strip():
                raise ValueError("期次名称不能为空")
        return {
            "schema_version": PHASE_CONFIG_SCHEMA_VERSION,
            "phases": [
                {
                    "name": phase["name"],
                    "categories": {
                        key: {
                            "name": phase["categories"][key]["name"],
                            "details": [
                                dict(detail)
                                for detail in phase["categories"][key]["details"]
                            ],
                        }
                        for key in MAIN_CATEGORY_KEYS
                    },
                }
                for phase in self._phases
            ],
        }

    @property
    def validated_config(self) -> dict[str, Any]:
        return self._validated_config or self._build_config()

    @Slot()
    def accept(self) -> None:
        try:
            self._validated_config = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "无法保存", str(exc))
            return
        super().accept()


__all__ = ["PhaseConfigDialog"]
