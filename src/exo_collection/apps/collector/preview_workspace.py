"""Dockable, persistent workspace for Collector modality previews."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QTimer, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QWidget,
)


class PreviewWorkspace(QMainWindow):
    """A small nested main window whose docks are bound by modality key.

    A dock is created once and can then be closed/re-added without destroying
    its plot buffers.  Closing a dock therefore affects presentation only; it
    never starts or stops a hardware preview worker.
    """

    layout_changed = Signal()
    focus_mode_requested = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("preview_workspace")
        self.setDockNestingEnabled(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._docks: dict[str, QDockWidget] = {}
        self._titles: dict[str, str] = {}
        self._stream_states: dict[str, str] = {}
        self._default_visible: tuple[str, ...] = ()
        self._restoring = False
        self._focus_mode = False

        self._add_menu = QMenu(self)
        toolbar = QToolBar("预览布局", self)
        toolbar.setObjectName("preview_workspace_toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        add_button = QToolButton(toolbar)
        add_button.setObjectName("add_preview_window")
        add_button.setText("＋ 添加窗口")
        add_button.setToolTip("添加或重新显示一个模态预览窗口")
        add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        add_button.setMenu(self._add_menu)
        toolbar.addWidget(add_button)

        reset_button = QPushButton("恢复默认布局", toolbar)
        reset_button.setObjectName("reset_preview_layout")
        reset_button.clicked.connect(self.reset_default_layout)
        toolbar.addWidget(reset_button)

        self._focus_button = QPushButton("放大预览区", toolbar)
        self._focus_button.setObjectName("toggle_preview_focus")
        self._focus_button.setCheckable(True)
        self._focus_button.toggled.connect(self._request_focus_mode)
        toolbar.addWidget(self._focus_button)

        toolbar.addSeparator()
        self._hint_action = QAction(
            "拖动标题栏停靠/浮动；拖动窗口边界调整大小",
            toolbar,
        )
        self._hint_action.setEnabled(False)
        toolbar.addAction(self._hint_action)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._layout_signal_timer = QTimer(self)
        self._layout_signal_timer.setSingleShot(True)
        self._layout_signal_timer.setInterval(150)
        self._layout_signal_timer.timeout.connect(self.layout_changed)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self._docks)

    def dock_for(self, modality: str) -> QDockWidget | None:
        return self._docks.get(modality)

    def register_panel(
        self,
        modality: str,
        title: str,
        widget: QWidget,
        *,
        visible_by_default: bool = True,
    ) -> QDockWidget:
        if modality in self._docks:
            raise ValueError(f"preview modality already registered: {modality}")

        dock = QDockWidget(f"{title} · 等待数据", self)
        dock.setObjectName(f"preview_dock_{modality}")
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        dock.setMinimumSize(180, 100)
        dock.setWidget(widget)
        self._docks[modality] = dock
        self._titles[modality] = title
        self._stream_states[modality] = "waiting"
        if visible_by_default:
            self._default_visible = (*self._default_visible, modality)

        add_action = QAction(title, self._add_menu)
        add_action.setObjectName(f"add_preview_{modality}")
        add_action.triggered.connect(
            lambda _checked=False, key=modality: self.show_panel(key)
        )
        self._add_menu.addAction(add_action)

        dock.dockLocationChanged.connect(self._queue_layout_changed)
        dock.topLevelChanged.connect(self._queue_layout_changed)
        dock.visibilityChanged.connect(self._queue_layout_changed)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
        dock.hide()
        return dock

    def set_stream_state(self, modality: str, state: str) -> None:
        """Update a dock's binding indicator without changing its visibility."""

        dock = self._docks.get(modality)
        title = self._titles.get(modality)
        if dock is None or title is None:
            return
        normalized = state.strip().lower()
        if self._stream_states.get(modality) == normalized:
            return
        self._stream_states[modality] = normalized
        suffix = {
            "live": "实时数据",
            "connected": "已连接",
            "disconnected": "未连接",
            "error": "数据异常",
        }.get(normalized, "等待数据")
        dock.setWindowTitle(f"{title} · {suffix}")

    def show_panel(self, modality: str) -> None:
        dock = self._docks.get(modality)
        if dock is None:
            raise KeyError(f"unknown preview modality: {modality}")
        dock.show()
        dock.raise_()
        self._queue_layout_changed()

    def hide_panel(self, modality: str) -> None:
        dock = self._docks.get(modality)
        if dock is not None:
            dock.hide()

    def visible_modalities(self) -> tuple[str, ...]:
        return tuple(key for key, dock in self._docks.items() if dock.isVisible())

    def save_layout(self, version: int = 1) -> QByteArray:
        return self.saveState(version)

    def restore_layout(self, state: QByteArray | bytes | None, version: int = 1) -> bool:
        if not state:
            self.reset_default_layout()
            return False
        payload = state if isinstance(state, QByteArray) else QByteArray(state)
        self._restoring = True
        try:
            restored = self.restoreState(payload, version)
        finally:
            self._restoring = False
        if not restored:
            self.reset_default_layout()
        return restored

    def reset_default_layout(self) -> None:
        """Create a predictable two-column dashboard from registered docks."""

        if not self._docks:
            return
        self._restoring = True
        try:
            for dock in self._docks.values():
                dock.setFloating(False)
                self.removeDockWidget(dock)
                dock.hide()

            ordered = list(self._docks)
            first = self._docks[ordered[0]]
            self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, first)

            left_anchor = first
            right_anchor: QDockWidget | None = None
            for index, modality in enumerate(ordered[1:], start=1):
                dock = self._docks[modality]
                self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
                if index == 1:
                    self.splitDockWidget(
                        first,
                        dock,
                        Qt.Orientation.Horizontal,
                    )
                    right_anchor = dock
                elif index % 2 == 0:
                    self.splitDockWidget(
                        left_anchor,
                        dock,
                        Qt.Orientation.Vertical,
                    )
                    left_anchor = dock
                else:
                    anchor = right_anchor if right_anchor is not None else first
                    self.splitDockWidget(
                        anchor,
                        dock,
                        Qt.Orientation.Vertical,
                    )
                    right_anchor = dock

            visible = set(self._default_visible)
            for modality, dock in self._docks.items():
                dock.setVisible(modality in visible)
        finally:
            self._restoring = False
        self._queue_layout_changed()

    def set_focus_mode(self, enabled: bool) -> None:
        self._focus_mode = bool(enabled)
        self._focus_button.blockSignals(True)
        self._focus_button.setChecked(self._focus_mode)
        self._focus_button.setText(
            "退出放大预览" if self._focus_mode else "放大预览区"
        )
        self._focus_button.blockSignals(False)

    def suspend_layout_tracking(self) -> None:
        """Prevent late dock signals from writing settings after shutdown."""

        self._layout_signal_timer.stop()
        self._restoring = True

    def _request_focus_mode(self, enabled: bool) -> None:
        self.set_focus_mode(enabled)
        self.focus_mode_requested.emit(enabled)

    def _queue_layout_changed(self, *_args: object) -> None:
        if not self._restoring:
            self._layout_signal_timer.start()
__all__ = ["PreviewWorkspace"]
