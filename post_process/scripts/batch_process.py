#!/usr/bin/env python
"""批量 inspection：遍历 dataset_root 下所有 .c3d，逐个跑 inspection。

用法::

    python scripts/batch_process.py --dataset_root <目录> [--out <输出根>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from postprocess.c3d.inspect_c3d import run_inspection  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="批量 C3D inspection")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = Path(args.dataset_root)
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")

    c3d_files = sorted(root.rglob("*.c3d"))
    if not c3d_files:
        print(f"未找到 .c3d 文件：{root}")
        return 1

    print(f"找到 {len(c3d_files)} 个 c3d 文件")
    for c3d in c3d_files:
        out = (args.out or root / "inspections") / c3d.stem
        try:
            rep = run_inspection(c3d, out)
            print(f"  [OK] {c3d.name}  GRF_MODE={rep['force']['grf_mode']}  "
                  f"markers={rep['point']['n_points']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {c3d.name}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
