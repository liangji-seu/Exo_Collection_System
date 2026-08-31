#!/usr/bin/env python
"""C3D inspection（无 OpenSim / 无标定矩阵也可运行）。

用法::

    python scripts/inspect_trial.py --c3d xxx.c3d [--out 输出目录]

输出：inspection_report.json + inspection_report.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from postprocess.c3d.inspect_c3d import run_inspection  # noqa: E402


def _summary(rep: dict) -> str:
    p, a, f, sy = rep["point"], rep["analog"], rep["force"], rep["sync"]
    subj = rep["subjects"]
    lines = [
        "=" * 55,
        f"文件：{rep['path']}",
        f"软件：{rep['software']} {rep['software_version']}",
        "-" * 55,
        f"[POINT]  rate={p['rate_hz']} Hz  单位={p['units']}  点数={p['n_points']}  "
        f"帧={p['n_frames']}  时长≈{p['duration_s']:.2f}s",
        f"[ANALOG] rate={a['rate_hz']} Hz  通道数={a['n_channels']}",
        f"  通道：{a['channel_names']}",
        f"[FORCE]  GRF_MODE = {f['grf_mode']}",
        f"  分量：{f['force_components_found']}  Tz存在={f['has_free_moment']}  "
        f"Mx/My/Mz存在={f['has_six_axis_moment']}",
        f"[SYNC]   point/analog ratio={sy['ratio']}  整数倍={sy['integer_ratio']}",
        f"[MARKER] 分类：{rep['marker_class_counts']}",
    ]
    for s in subj:
        lines.append(f"[SUBJECT] {s['name']}  static={s['is_static']}  "
                     f"markers={s['n_markers']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="C3D inspection")
    parser.add_argument("--c3d", required=True, help="c3d 文件路径")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出目录（默认与 c3d 同级的 inspection/）")
    args = parser.parse_args(argv)

    c3d_path = Path(args.c3d)
    if not c3d_path.is_file():
        parser.error(f"c3d 不存在：{c3d_path}")
    out = args.out or c3d_path.parent / "inspection"
    rep = run_inspection(c3d_path, out)
    print(_summary(rep))
    print("-" * 55)
    print(f"已写出：{out / 'inspection_report.json'}")
    print(f"        {out / 'inspection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
