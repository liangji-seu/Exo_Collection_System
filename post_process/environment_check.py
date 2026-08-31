"""检测 post_process 流水线依赖，重点区分 OpenSim 绑定是否可用。

用法::

    python environment_check.py

说明：
- 没有 OpenSim 绑定时，C3D inspection / 坐标变换 / TRC / MOT 写出仍然可用，
  只有 Scale / IK / ID 三个 stage 会报 BLOCKING。
- 不要假设 ``pip install opensim`` 一定能装：OpenSim 4.x 的 Python 绑定是
  官方 conda 包（``conda install -c opensim-org opensim``），与 conda 的
  Python 版本严格绑定。
"""

from __future__ import annotations

import platform
import sys

_REQUIRED = ["numpy", "scipy", "pandas", "ezc3d", "yaml", "matplotlib"]
_OPTIONAL = ["opensim"]


def _probe(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
    except Exception as exc:  # noqa: BLE001 - 我们只关心能不能 import
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(mod, "__version__", getattr(mod, "version", ""))
    return True, str(version) if version else "(no __version__)"


def main() -> int:
    lines = ["=" * 60, "post_process 环境检测", "=" * 60]
    lines.append(f"python: {sys.version.split()[0]}  ({platform.python_implementation()})")
    lines.append(f"os:     {platform.system()} {platform.release()}  "
                 f"arch={platform.machine()}")
    lines.append(f"conda:  {'yes (active)' if sys.prefix.lower().find('conda') >= 0 or 'CONDA_PREFIX' in __import__('os').environ else 'unknown/not active'}")
    lines.append("")

    lines.append("-- 必需依赖 --")
    all_ok = True
    for name in _REQUIRED:
        ok, version = _probe(name)
        lines.append(f"  [{'OK' if ok else 'MISSING'}] {name:12s} {version}")
        all_ok = all_ok and ok

    lines.append("")
    lines.append("-- OpenSim 绑定（可选） --")
    ok, version = _probe("opensim")
    lines.append(f"  [{'OK' if ok else 'MISSING'}] opensim   {version}")
    if not ok:
        lines.append("")
        lines.append("  OpenSim Python 绑定未安装。安装方式（官方）：")
        lines.append("    conda install -c opensim-org opensim")
        lines.append("  注意：OpenSim 绑定与特定 Python 版本绑定，通常需要专用 conda 环境；")
        lines.append("  若当前环境无法安装，可把上游阶段（inspection/预处理/TRC/MOT）")
        lines.append("  与 OpenSim 阶段拆分到不同环境执行。")
        lines.append("  → 未安装时 pipeline 最大可执行 stage = MARKER_PREPROCESSING（Scale/IK/ID 全部 BLOCKING）。")

    lines.append("")
    lines.append("-- 结论 --")
    if not all_ok:
        lines.append("  必需依赖缺失，请先 `pip install -r requirements.txt`。")
    lines.append("  OpenSim 可用" if ok else "  OpenSim 不可用（上游阶段仍可开发/测试）。")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
