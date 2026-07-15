"""Command-line vertical validation for one simulated Trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .models import TrialRunRequest
from .simulated import run_simulated_trial


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and finalize one simulated multimodal Trial")
    parser.add_argument("--data-root", type=Path, default=Path("runtime_data"))
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--project", default="Exoskeleton Study")
    parser.add_argument("--subject", default="SIM-001")
    parser.add_argument("--operator", default="simulator")
    parser.add_argument("--condition", default="WALK_LEVEL")
    parser.add_argument("--repeat", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = TrialRunRequest(
        data_root=args.data_root,
        duration_s=args.duration,
        project_name=args.project,
        subject_code=args.subject,
        operator=args.operator,
        condition_code=args.condition,
        condition_name=args.condition.replace("_", " ").title(),
        repeat_index=args.repeat,
    )
    result = run_simulated_trial(request)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

