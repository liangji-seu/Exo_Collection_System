"""Visual theme for the Collector desktop application."""

from __future__ import annotations


COLLECTOR_STYLESHEET = """
QMainWindow {
    background: #f1f0eb;
}

QWidget {
    color: #2d3330;
}

QLabel#page_title {
    color: #26332f;
    font-size: 19px;
    font-weight: 700;
    padding: 2px 0;
}

QGroupBox {
    background: #fbfaf6;
    border: 1px solid #c8c5ba;
    border-radius: 6px;
    margin-top: 11px;
    padding-top: 6px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #9a5b13;
    background: #fbfaf6;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
    background: #fffefa;
    border: 1px solid #c8c5ba;
    border-radius: 4px;
    selection-background-color: #0f766e;
    selection-color: #ffffff;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 24px;
    padding: 2px 6px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border: 2px solid #0f766e;
}

QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    border: none;
    background: #e6e3da;
}

QPushButton {
    min-height: 28px;
    padding: 5px 11px;
    border: 1px solid #aaa69b;
    border-radius: 5px;
    background: #e6e3da;
    color: #2d3330;
    font-weight: 600;
}

QPushButton:hover {
    background: #d6d1c5;
    border-color: #77746c;
}

QPushButton:pressed {
    background: #c8c2b5;
}

QPushButton[buttonRole="connect"] {
    background: #0f766e;
    border-color: #115e59;
    color: #ffffff;
}

QPushButton[buttonRole="connect"]:hover {
    background: #14877d;
}

QPushButton[buttonRole="primary"] {
    background: #3f7d5b;
    border-color: #306247;
    color: #ffffff;
    font-weight: 700;
}

QPushButton[buttonRole="primary"]:hover {
    background: #4b916a;
}

QPushButton[buttonRole="disconnect"] {
    background: #b4533c;
    border-color: #8f3f2f;
    color: #ffffff;
}

QPushButton[buttonRole="disconnect"]:hover {
    background: #c9654c;
}

QPushButton[buttonRole="danger"] {
    background: #a53f3f;
    border-color: #843333;
    color: #ffffff;
    font-weight: 700;
}

QPushButton[buttonRole="danger"]:hover {
    background: #bb4b4b;
}

QPushButton[buttonRole="deviceConfig"] {
    min-height: 22px;
    padding: 2px 4px;
    border: none;
    background: transparent;
    color: #0f766e;
    text-align: left;
    text-decoration: underline;
    font-weight: 700;
}

QPushButton[buttonRole="deviceConfig"]:hover {
    background: #dcebe6;
    color: #115e59;
}

QPushButton:disabled {
    background: #e8e6df;
    border-color: #d3d0c7;
    color: #99968d;
}

QTableWidget {
    alternate-background-color: #f4f1e8;
    gridline-color: #d3d0c7;
}

QHeaderView::section {
    background: #e8e2d5;
    color: #3f433f;
    border: none;
    border-right: 1px solid #c8c5ba;
    border-bottom: 1px solid #c8c5ba;
    padding: 5px;
    font-weight: 700;
}

QPlainTextEdit#alerts {
    background: #232a27;
    color: #e9ede8;
    border-color: #4b5550;
    font-family: "Cascadia Mono", "Consolas", monospace;
    padding: 5px;
}

QScrollArea {
    background: transparent;
    border: none;
}

QMainWindow#preview_workspace {
    background: #e5e2d9;
    border: 1px solid #c8c5ba;
    border-radius: 6px;
}

QToolBar#preview_workspace_toolbar {
    background: #f5f2ea;
    border: none;
    border-bottom: 1px solid #c8c5ba;
    spacing: 6px;
    padding: 4px;
}

QDockWidget {
    color: #26332f;
    font-weight: 700;
}

QDockWidget::title {
    background: #dcebe6;
    border: 1px solid #a9c6bd;
    padding: 6px 8px;
    text-align: left;
}

QDockWidget > QWidget {
    background: #fbfaf6;
}

QScrollBar:vertical {
    width: 12px;
    margin: 0;
    background: #e6e3da;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    min-height: 32px;
    background: #aaa69b;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #77746c;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    height: 0;
    background: transparent;
}

QStatusBar {
    background: #e8e2d5;
    color: #3f433f;
    border-top: 1px solid #c8c5ba;
}

QToolTip {
    background: #26302c;
    color: #ffffff;
    border: 1px solid #59635e;
    padding: 4px;
}
"""
