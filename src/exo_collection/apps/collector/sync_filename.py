"""同步文件名条：显示当前 trial 的录制文件名主干，供复制命名 txt。

动捕 ``.cap`` 与测力台 ``.txt``（gaitway3d 导出）必须同名，Data Studio 才能把它们
和 trial 对上。这个同名「主干」即 XINGYING 录制文件名（``CollectorWindow._build_xingying_capture_name``
生成的 ``{subject}_{condition}_r{repeat}_{uuid8}``）。

本控件嵌在 Collector 主界面左下角（左控制列底部），不是 dock：点「开始写盘」时由窗口
调用 :meth:`set_filename` 填入主干，操作员停止写盘后点「复制文件名」把名字写进剪贴板，
同时弹出一个 1 秒后自动消失的「复制成功」小提示。
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

_PLACEHOLDER = "开始写盘后生成"
_COPY_CONFIRM_MS = 1000


class SyncFilenameBar(QWidget):
    """一行「同步文件名」：文本框 + 一个「复制文件名」按钮。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sync_filename_bar")

        self._label = QLabel("同步文件名", self)
        self._label.setObjectName("sync_filename_label")

        self.line_edit = QLineEdit(self)
        self.line_edit.setObjectName("sync_filename_edit")
        self.line_edit.setReadOnly(True)
        self.line_edit.setPlaceholderText(_PLACEHOLDER)

        self.copy_stem_button = QPushButton("复制文件名", self)
        self.copy_stem_button.setObjectName("copy_sync_stem")
        self.copy_stem_button.setToolTip("复制文件名主干（不含扩展名）")
        self.copy_stem_button.setMinimumHeight(32)
        self.copy_stem_button.setMinimumWidth(96)
        self.copy_stem_button.clicked.connect(self._copy_stem)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._label)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.copy_stem_button)

        # 复制成功小弹窗：独立小窗，点击复制后显示 1 秒自动消失。
        self.copy_toast = QLabel(
            "复制成功",
            self,
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint,
        )
        self.copy_toast.setObjectName("copy_toast")
        self.copy_toast.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.copy_toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.copy_toast.setStyleSheet(
            "QLabel {"
            "background-color:#E6F4EA;"
            "color:#1F5D36;"
            "border:1px solid #79B78C;"
            "border-radius:6px;"
            "font-size:12px;"
            "font-weight:600;"
            "padding:6px 14px;"
            "}"
        )
        self.copy_toast.adjustSize()

        self._copy_toast_timer = QTimer(self)
        self._copy_toast_timer.setSingleShot(True)
        self._copy_toast_timer.timeout.connect(self.copy_toast.hide)

        self.set_filename(None)

    # ── public API ───────────────────────────────────────────────────────

    def set_filename(self, name: str | None) -> None:
        """填入当前 trial 的录制文件名主干；``None`` 表示尚无 trial。"""
        self._name = name.strip() if name else None
        if self._name:
            self.line_edit.setText(self._name)
            self.line_edit.setToolTip(self._name)
        else:
            self.line_edit.clear()
            self.line_edit.setToolTip("")
        self.copy_stem_button.setEnabled(self._name is not None)

    def filename(self) -> str | None:
        """当前显示的文件名主干（无 trial 时为 ``None``）。"""
        return self._name

    # ── copy ─────────────────────────────────────────────────────────────

    def _copy_stem(self) -> None:
        if not self._name:
            return
        QApplication.clipboard().setText(self._name)
        self._show_copy_toast()

    def _show_copy_toast(self) -> None:
        """在复制按钮上方弹出「复制成功」，1 秒后自动消失。"""
        top_left = self.copy_stem_button.mapToGlobal(
            self.copy_stem_button.rect().topLeft()
        )
        bottom_right = self.copy_stem_button.mapToGlobal(
            self.copy_stem_button.rect().bottomRight()
        )
        self.copy_toast.adjustSize()
        self.copy_toast.move(
            bottom_right.x() - self.copy_toast.width(),
            top_left.y() - self.copy_toast.height() - 6,
        )
        self.copy_toast.show()
        self.copy_toast.raise_()
        self._copy_toast_timer.start(_COPY_CONFIRM_MS)


__all__ = ["SyncFilenameBar"]
