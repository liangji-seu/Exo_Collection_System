#!/usr/bin/env python
"""流水线入口：preflight + 部分执行。

用法::

    python scripts/run_pipeline.py --config configs/S01_walk_075.yaml

流程：
1. 读 config
2. 若 static C3D 可用，先跑 inspection
3. validate_config → 打印 preflight 表 + Maximum executable stage
4. 只执行到 Maximum executable stage（下游 BLOCKING 不会让整个程序失败）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("需要 PyYAML：pip install pyyaml")


from postprocess.c3d.inspect_c3d import inspect, read_c3d  # noqa: E402
from postprocess.validation.validate_config import validate_config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OpenSim 逆动力学流水线")
    parser.add_argument("--config", required=True, help="YAML config 路径")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        parser.error(f"config 不存在：{cfg_path}")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 1. 若 static c3d 存在，跑 inspection
    inspection = None
    static = config.get("files", {}).get("static_c3d")
    if static and not str(static).strip().upper().startswith("TODO") and Path(static).is_file():
        data = read_c3d(static)
        inspection = inspect(data)
        print(f"[inspect] 已读取 static C3D：{static}")
        print(f"          GRF_MODE = {inspection['force']['grf_mode']}")

    # 2. 校验 + preflight
    pre = validate_config(config, inspection=inspection)
    print()
    print(pre.report())

    # 3. 部分执行：这里只执行到 inspection 这一级（其余 stage 由各 wrapper 按需调用）
    max_stage = pre.max_executable_stage()
    print()
    print(f"本次最多执行到：{max_stage}")
    if max_stage in ("NONE", "C3D_INSPECTION"):
        print("（Scale/IK/ID 均被 BLOCKING，先补齐 config 中标定的字段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
