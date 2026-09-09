"""Fixed modality-status overview strip: five large rounded blocks.

Each block maps to one acquisition modality and shows, at a glance, whether that
modality's data is currently being collected (green), interrupted/abnormal (red),
still connecting or waiting for data (yellow), or not connected at all (gray).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

# 方块状态 → (背景色, 前景色)。配色与 COLLECTOR_STYLESHEET 保持一致：
# 绿=#0f766e、红=#a53f3f、黄=#d97706、灰=#d3d0c7。
_BLOCK_PALETTE = {
    "off": ("#d3d0c7", "#5b6470"),
    "transition": ("#d97706", "#ffffff"),
    "ok": ("#0f766e", "#ffffff"),
    "bad": ("#a53f3f", "#ffffff"),
}

# 写盘指示用「录制红」#dc2626（更鲜亮），只作用于五个方块所在框的底色，
# 与单个方块的故障红 #a53f3f 明显区分；方块本身保持各自状态色不变。
_FRAME_BG_NORMAL = "#f5f2ea"
_FRAME_BG_RECORDING = "#dc2626"
_FRAME_BORDER_NORMAL = "#c8c5ba"
_FRAME_BORDER_RECORDING = "#dc2626"

_BLOCK_HEIGHT = 72
_BLOCK_RADIUS = 12


class ModalityStatusStrip(QFrame):
    """五个大圆角方块，用颜色表示各模态数据采集状态。

    ``modalities`` 是有序的 ``{modality_key: 显示名}`` 映射；方块的顺序与之一致。
    """

    def __init__(self, modalities: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("modality_status_strip")
        self._recording = False
        self._apply_frame_style()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._names: dict[str, str] = dict(modalities)
        self._blocks: dict[str, QLabel] = {}
        self._last: dict[str, tuple[str, str]] = {}
        for modality, name in self._names.items():
            block = QLabel()
            block.setObjectName(f"status_block_{modality}")
            block.setAlignment(Qt.AlignmentFlag.AlignCenter)
            block.setMinimumHeight(_BLOCK_HEIGHT)
            block.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._blocks[modality] = block
            layout.addWidget(block, 1)
            self.set_state(modality, "off", "未连接")

    def set_state(self, modality: str, state: str, status_text: str) -> None:
        """记录单个方块的期望状态并重绘；状态未变化时不重绘。

        写盘期间方块仍显示各自状态；写盘指示由所在框底色（``set_recording``）承担。
        """
        if modality not in self._blocks:
            return
        if self._last.get(modality) == (state, status_text):
            return
        self._last[modality] = (state, status_text)
        self._render(modality)

    def set_recording(self, recording: bool) -> None:
        """写盘期间把五个方块所在框的底色变为「录制红」；方块颜色不变。"""
        if recording == self._recording:
            return
        self._recording = recording
        self._apply_frame_style()

    def _apply_frame_style(self) -> None:
        background = _FRAME_BG_RECORDING if self._recording else _FRAME_BG_NORMAL
        border = _FRAME_BORDER_RECORDING if self._recording else _FRAME_BORDER_NORMAL
        self.setStyleSheet(
            "QFrame#modality_status_strip {"
            f" background: {background};"
            f" border: 1px solid {border};"
            " border-radius: 6px;"
            "}"
        )

    def _render(self, modality: str) -> None:
        block = self._blocks.get(modality)
        if block is None:
            return
        state, status_text = self._last.get(modality, ("off", "未连接"))
        background, foreground = _BLOCK_PALETTE.get(
            state, _BLOCK_PALETTE["off"]
        )
        name = self._names.get(modality, modality)
        block.setStyleSheet(
            f"QLabel {{ background: {background}; color: {foreground}; "
            f"border-radius: {_BLOCK_RADIUS}px; padding: 6px; }}"
        )
        block.setText(
            f"<div style='font-size:15px; font-weight:700;'>{name}</div>"
            f"<div style='font-size:12px; font-weight:500; margin-top:2px;'>"
            f"{status_text}</div>"
        )
        block.setToolTip(f"{name}：{status_text}")

    def reset(self) -> None:
        """把所有方块重置为灰色「未连接」。"""
        self.set_recording(False)
        for modality in self._blocks:
            self.set_state(modality, "off", "未连接")


__all__ = ["ModalityStatusStrip"]
