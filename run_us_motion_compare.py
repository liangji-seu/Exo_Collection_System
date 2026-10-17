"""Launcher for the standalone ultrasound raw motion comparison tool."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


EXO_PYTHON = Path(r"E:\miniconda\envs\EXO\python.exe")
SYSTEM_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SYSTEM_DIR.parent


def _ensure_runtime() -> None:
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import PySide6  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError as exc:
        current = Path(sys.executable).resolve()
        if EXO_PYTHON.is_file() and EXO_PYTHON != current:
            os.execv(str(EXO_PYTHON), [str(EXO_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit("缺少运行依赖 PySide6/pyqtgraph/numpy/scipy，请使用 E:\\miniconda\\envs\\EXO\\python.exe 运行。") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较四种超声区域移动方法的原始 ADC 回放")
    parser.add_argument("path", nargs="?", help="session 文件夹或 ultrasound.bin")
    parser.add_argument("--session", help="session 文件夹或 ultrasound.bin")
    parser.add_argument("--channel", type=int, help="初始物理通道")
    parser.add_argument("--start", type=float, default=None, help="开始秒")
    parser.add_argument("--end", type=float, default=None, help="结束秒")
    parser.add_argument("--analyze", action="store_true", help="加载后自动计算，结果准备好后暂停")
    parser.add_argument("--smoke-test", action="store_true", help="离屏运行可控交互核验")
    parser.add_argument("--screenshot", help="smoke test 截图路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.path and args.session:
        raise SystemExit("位置参数与 --session 只能使用一个")
    if args.start is not None and args.start < 0:
        raise SystemExit("--start 不能小于 0")
    if args.end is not None and args.end < 0:
        raise SystemExit("--end 不能小于 0")
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _ensure_runtime()
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    from Exo_Collection_System.us_motion_compare_viewer import launch_viewer, run_smoke_test

    if args.smoke_test:
        return run_smoke_test(args.screenshot)
    return launch_viewer(args.session or args.path, args.channel, args.start, args.end, args.analyze)


if __name__ == "__main__":
    raise SystemExit(main())
