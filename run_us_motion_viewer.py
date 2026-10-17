"""Command-line launcher for the read-only ultrasound motion player."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


EXO_PYTHON = Path(r"E:\miniconda\envs\EXO\python.exe")


def _ensure_runtime() -> None:
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import PySide6  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError as exc:
        current = Path(sys.executable).resolve()
        if EXO_PYTHON.is_file() and EXO_PYTHON.resolve() != current:
            os.execv(
                str(EXO_PYTHON),
                [str(EXO_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
        raise SystemExit(
            "缺少运行依赖 PySide6/pyqtgraph/numpy/scipy。"
            "请使用 E:\\miniconda\\envs\\EXO\\python.exe 运行；程序不会自动安装依赖。"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回放历史超声 session 并显示区域移动方向")
    parser.add_argument("path", nargs="?", help="session 文件夹或 ultrasound.bin")
    parser.add_argument("--session", dest="session", help="session 文件夹或 ultrasound.bin")
    parser.add_argument("--channel", type=int, choices=(1, 2, 3, 4), help="初始物理通道")
    parser.add_argument("--time", type=float, default=None, help="初始时间（秒，定位到最近帧）")
    parser.add_argument("--play", action="store_true", help="加载完成后自动开始按真实时间播放")
    parser.add_argument("--smoke-test", action="store_true", help="离屏运行自动交互核验")
    parser.add_argument("--screenshot", help="smoke test 截图输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.path and args.session:
        raise SystemExit("位置参数与 --session 只能使用一个")
    if args.time is not None and args.time < 0:
        raise SystemExit("--time 不能小于 0")
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _ensure_runtime()
    from us_motion_viewer import launch_viewer, run_smoke_test

    if args.smoke_test:
        return run_smoke_test(args.screenshot, args.session or args.path)
    return launch_viewer(
        args.session or args.path,
        args.channel,
        args.time,
        autoplay=args.play,
    )


if __name__ == "__main__":
    raise SystemExit(main())
