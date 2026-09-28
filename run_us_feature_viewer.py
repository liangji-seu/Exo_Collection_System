"""Launch the independent US / 48-feature waterfall viewer."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _reexec_exo_python() -> None:
    """Use the project's EXO environment when the base interpreter lacks Qt."""

    try:
        import PySide6  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError:
        candidates = [
            Path(sys.executable).parent / "envs" / "EXO" / "python.exe",
            Path(r"E:\miniconda\envs\EXO\python.exe"),
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve() != Path(sys.executable).resolve():
                os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])
                return
        raise RuntimeError("缺少 PySide6/pyqtgraph，请使用 E:\\miniconda\\envs\\EXO\\python.exe")


def main() -> int:
    _reexec_exo_python()
    from us_feature_viewer import configure_font, launch_viewer, load_session, run_self_test

    if "--self-test" in sys.argv:
        args = [arg for arg in sys.argv[1:] if arg != "--self-test"]
        if not args:
            raise SystemExit("用法：python run_us_feature_viewer.py --self-test SESSION_DIR")
        print(run_self_test(args[0]))
        return 0
    if len(sys.argv) > 1:
        data = load_session(sys.argv[1])
    else:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)
        configure_font(app)
        from exo_collection.apps.us_session_selector import select_us_session
        from exo_collection.configuration import SharedAppSettings

        selected = select_us_session(SharedAppSettings().data_root)
        if not selected:
            return 0
        data = load_session(selected)
    return launch_viewer(data)


if __name__ == "__main__":
    raise SystemExit(main())
