"""单 trial C3D inspection 命令行入口。

用法：:

    python scripts/inspect_trial.py --c3d data/c3d/001_WALK_LEVEL_r1_5432165b1.c3d \
        --out outputs/inspection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 让 `pipeline` 包可导入（脚本位于 opensim_pipeline/scripts/）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.c3d.inspect_c3d import inspect, build_markdown  # noqa: E402
from pipeline.c3d.reader import read_c3d  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="C3D inspection")
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--out", default="outputs/inspection")
    args = ap.parse_args()

    data = read_c3d(args.c3d)
    rep = inspect(data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inspection_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "inspection_report.md").write_text(build_markdown(rep), encoding="utf-8")

    print(build_markdown(rep))
    print(f"\n[written] {out / 'inspection_report.json'}")
    print(f"[written] {out / 'inspection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
