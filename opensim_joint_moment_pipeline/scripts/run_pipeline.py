"""Pipeline 命令行入口。

用法：:

    python scripts/run_pipeline.py --config configs/subject_001.yaml \
        --out outputs/subject_001/walk_level
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.pipeline import load_config, build_preflight, run_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenSim single-support pipeline")
    ap.add_argument("--config", default="configs/subject_001.yaml")
    ap.add_argument("--out", default="outputs/subject_001/walk_level")
    args = ap.parse_args()

    config = load_config(args.config)
    out = Path(args.out)

    print(build_preflight(config).render())

    result = run_pipeline(config, out)
    (out / "pipeline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 52)
    print("Pipeline result")
    print("=" * 52)
    print(json.dumps(result["stages"], ensure_ascii=False, indent=2, default=str))
    print(f"\n[output dir] {out}")
    print(f"[max executable stage] {result['max_executable_stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
