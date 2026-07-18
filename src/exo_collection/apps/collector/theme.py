"""Visual theme for the Collector desktop application."""

from __future__ import annotations


COLLECTOR_STYLESHEET = """
QMainWindow {
    background: #eaf0f6;
}

QWidget {
    color: #1e293b;
}

QLabel#page_title {
    color: #0f2744;
    font-size: 19px;
    font-weight: 700;
    padding: 2px 0;
}

QGroupBox {
    background: #f8fafc;
    border: 1px solid #b9c7d8;
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
    color: #1d4ed8;
    background: #f8fafc;
}

QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTableWidget {
    background: #ffffff;
    border: 1px solid #b9c7d8;
    border-radius: 4px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}

QLineEdit, QComboBox, QSpinBox {
    min-height: 24px;
    padding: 2px 6px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {
    border: 2px solid #2563eb;
}

QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button {
    border: none;
    background: #e2e8f0;
}

QPushButton {
    min-height: 28px;
    padding: 5px 11px;
    border: 1px solid #94a3b8;
    border-radius: 5px;
    background: #e2e8f0;
    color: #1e293b;
    font-weight: 600;
}

QPushButton:hover {
    background: #cbd5e1;
    border-color: #64748b;
}

QPushButton:pressed {
    background: #b7c3d1;
}

QPushButton[buttonRole="connect"] {
    background: #1d4ed8;
    border-color: #1e40af;
    color: #ffffff;
}

QPushButton[buttonRole="connect"]:hover {
    background: #2563eb;
}

QPushButton[buttonRole="primary"] {
    background: #15803d;
    border-color: #166534;
    color: #ffffff;
    font-weight: 700;
}

QPushButton[buttonRole="primary"]:hover {
    background: #16a34a;
}

QPushButton[buttonRole="disconnect"] {
    background: #c2410c;
    border-color: #9a3412;
    color: #ffffff;
}

QPushButton[buttonRole="disconnect"]:hover {
    background: #ea580c;
}

QPushButton[buttonRole="danger"] {
    background: #b91c1c;
    border-color: #991b1b;
    color: #ffffff;
    font-weight: 700;
}

QPushButton[buttonRole="danger"]:hover {
    background: #dc2626;
}

QPushButton:disabled {
    background: #e5e7eb;
    border-color: #cbd5e1;
    color: #94a3b8;
}

QTableWidget {
    alternate-background-color: #eff6ff;
    gridline-color: #cbd5e1;
}

QHeaderView::section {
    background: #dbeafe;
    color: #1e3a5f;
    border: none;
    border-right: 1px solid #b9c7d8;
    border-bottom: 1px solid #b9c7d8;
    padding: 5px;
    font-weight: 700;
}

QPlainTextEdit#alerts {
    background: #0f172a;
    color: #dbeafe;
    border-color: #334155;
    font-family: "Cascadia Mono", "Consolas", monospace;
    padding: 5px;
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    width: 12px;
    margin: 0;
    background: #e2e8f0;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    min-height: 32px;
    background: #94a3b8;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #64748b;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    height: 0;
    background: transparent;
}

QStatusBar {
    background: #dbeafe;
    color: #1e3a5f;
    border-top: 1px solid #b9c7d8;
}

QToolTip {
    background: #0f172a;
    color: #ffffff;
    border: 1px solid #475569;
    padding: 4px;
}
"""

