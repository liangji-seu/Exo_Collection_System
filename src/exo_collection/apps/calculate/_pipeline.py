"""把 ``opensim_joint_moment_pipeline`` 挂到 ``sys.path`` 的小工具。

该目录不是可安装的 Python 包（顶层没有 ``__init__.py``），其可 import 单元是
``opensim_joint_moment_pipeline/pipeline``（包名 ``pipeline``）。既有脚本通过
``sys.path.insert(0, parents[1])`` 找到它；Exo Calculate 复用同样的约定，只
把路径推导集中到这一处，避免在 UI / worker 里散落相对路径。

``ensure_pipeline_on_path`` 幂等，可在计算函数内部惰性调用（不 import Qt、不
import opensim），因此 discovery / worker 都能安全使用。
"""

from __future__ import annotations

import sys
from pathlib import Path


def pipeline_root() -> Path:
    """返回 ``opensim_joint_moment_pipeline`` 目录（不一定存在时也返回路径）。

    冻结（PyInstaller）时，pipeline 被 bundle 到 ``<bundle>/opensim_joint_moment_pipeline``；
    非冻结时，本文件位于 ``<repo>/src/exo_collection/apps/calculate/_pipeline.py``，
    ``parents[4]`` 即仓库根 ``Exo_Collection_System``。
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / "opensim_joint_moment_pipeline"
    return Path(__file__).resolve().parents[4] / "opensim_joint_moment_pipeline"


def ensure_pipeline_on_path() -> Path:
    """把 pipeline 目录加入 ``sys.path``（幂等），返回该目录路径。"""
    root = pipeline_root()
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    return root
