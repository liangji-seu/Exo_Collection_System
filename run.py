"""Exo 采集系统统一启动器。

以后只需 ``python run.py``：弹出菜单选择「采集 / 归档 / 单个解算 / 批量解算」，
点哪个就在独立子进程中启动对应的 ``run_*.py``。子进程沿用当前解释器（EXO
环境）；Data Studio 的 SSH 依赖切换与 Calculate/Process 的 OpenSim 子环境由各自
入口脚本内部处理，启动器无需关心。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# 启动项：key → (显示名, 一句话说明, 入口脚本)。
_MODULES: dict[str, tuple[str, str, str]] = {
    "collector": ("采集", "实时多模态数据采集", "run_collector.py"),
    "data_studio": ("归档", "数据管理 · 归档 · 回放", "run_data_studio.py"),
    "calculate": ("单个解算", "标定 / 同步 / 解算 / 回放", "run_calculate.py"),
    "process": ("批量解算", "按受试者批量解算", "run_process.py"),
}

_LAUNCHER_STYLESHEET = """
QWidget#launcher {
    background: #f5f2ea;
}
QLabel#launcher_title {
    font-size: 20px;
    font-weight: 700;
    color: #1f2937;
}
QLabel#launcher_subtitle {
    font-size: 13px;
    color: #5b6470;
}
QLabel#launcher_footer {
    font-size: 12px;
    color: #0f766e;
}
QPushButton#launcher_button {
    background: #ffffff;
    border: 1px solid #d6d3c8;
    border-radius: 10px;
}
QPushButton#launcher_button:hover {
    border: 2px solid #0f766e;
    background: #eef7f5;
}
QPushButton#launcher_button:pressed {
    border: 2px solid #0f766e;
    background: #dcefeb;
}
QLabel#launcher_button_name {
    font-size: 18px;
    font-weight: 700;
    color: #0f766e;
    background: transparent;
}
QLabel#launcher_button_desc {
    font-size: 12px;
    color: #5b6470;
    background: transparent;
}
"""


def _make_button(name: str, desc: str) -> QPushButton:
    button = QPushButton()
    button.setObjectName("launcher_button")
    button.setMinimumHeight(72)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    layout = QVBoxLayout(button)
    layout.setContentsMargins(12, 8, 12, 8)
    layout.setSpacing(2)

    name_label = QLabel(name)
    name_label.setObjectName("launcher_button_name")
    name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    desc_label = QLabel(desc)
    desc_label.setObjectName("launcher_button_desc")
    desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    desc_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    layout.addWidget(name_label)
    layout.addWidget(desc_label)
    return button


class LauncherWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("launcher")
        self.setWindowTitle("Exo 外骨骼数据采集系统")
        self._footer: QLabel | None = None
        self._build_ui()
        self.resize(560, 330)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(10)

        title = QLabel("Exo 外骨骼数据采集系统")
        title.setObjectName("launcher_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("请选择要启动的模块")
        subtitle.setObjectName("launcher_subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        grid = QGridLayout()
        grid.setSpacing(10)
        for index, (key, (name, desc, _script)) in enumerate(_MODULES.items()):
            button = _make_button(name, desc)
            button.clicked.connect(lambda _=False, k=key, n=name: self._launch(k, n))
            row, col = divmod(index, 2)
            grid.addWidget(button, row, col)
        root.addLayout(grid)

        footer = QLabel("模块将在独立进程中启动，可连续打开多个。")
        footer.setObjectName("launcher_footer")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(footer)
        self._footer = footer

    def _launch(self, key: str, name: str) -> None:
        script = ROOT / _MODULES[key][2]
        try:
            subprocess.Popen(
                [sys.executable, str(script)],
                cwd=ROOT,
            )
        except OSError as exc:
            QMessageBox.critical(
                self,
                "启动失败",
                f"无法启动「{name}」：\n{exc}",
            )
            return
        if self._footer is not None:
            self._footer.setText(f"已启动：{name}")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Exo Launcher")
    app.setStyleSheet(_LAUNCHER_STYLESHEET)
    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
