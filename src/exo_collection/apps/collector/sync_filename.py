"""同步文件名条：显示当前 trial 的录制文件名主干，供复制命名 txt。

动捕 ``.cap`` 与测力台 ``.txt``（gaitway3d 导出）必须同名，Data Studio 才能把它们
和 trial 对上。这个同名「主干」即 XINGYING 录制文件名（``CollectorWindow._build_xingying_capture_name``
生成的 ``{subject}_{condition}_r{repeat}_{uuid8}``）。

本控件嵌在 Collector 主界面左下角（左控制列底部），不是 dock：点「开始写盘」时由窗口
调用 :meth:`set_filename` 填入主干，操作员停止写盘后点「复制主干」或「复制 .txt」把名字
写进剪贴板去命名 gaitway3d 导出的 txt。
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

_PLACEHOLDER = "开始写盘后生成"


class SyncFilenameBar(QWidget):
    """一行「同步文件名」：文本框 + 两个复制按钮。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sync_filename_bar")

        self._label = QLabel("同步文件名", self)
        self._label.setObjectName("sync_filename_label")

        self.line_edit = QLineEdit(self)
        self.line_edit.setObjectName("sync_filename_edit")
        self.line_edit.setReadOnly(True)
        self.line_edit.setPlaceholderText(_PLACEHOLDER)

        self.copy_stem_button = QPushButton("复制主干", self)
        self.copy_stem_button.setObjectName("copy_sync_stem")
        self.copy_stem_button.setToolTip("复制文件名主干（不含扩展名）")
        self.copy_stem_button.clicked.connect(self._copy_stem)

        self.copy_txt_button = QPushButton("复制 .txt", self)
        self.copy_txt_button.setObjectName("copy_sync_txt")
        self.copy_txt_button.setToolTip("复制带 .txt 后缀的完整文件名")
        self.copy_txt_button.clicked.connect(self._copy_txt)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._label)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.copy_stem_button)
        layout.addWidget(self.copy_txt_button)

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
        self.copy_txt_button.setEnabled(self._name is not None)

    def filename(self) -> str | None:
        """当前显示的文件名主干（无 trial 时为 ``None``）。"""
        return self._name

    # ── copy ─────────────────────────────────────────────────────────────

    def _copy_stem(self) -> None:
        if self._name:
            QApplication.clipboard().setText(self._name)

    def _copy_txt(self) -> None:
        if self._name:
            QApplication.clipboard().setText(f"{self._name}.txt")


__all__ = ["SyncFilenameBar"]
