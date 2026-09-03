"""OpenSim 子环境 ``python.exe`` 的自动发现与校验。

主界面进程（EXO 环境，NumPy 1.x）绝不 ``import opensim``；它只在受控子进程中
启动这个 ``python.exe`` 执行 Scale/IK/ID。本模块只做**路径发现**（扫描常见
conda 安装位置）与**轻量校验**（启动子进程 ``import opensim``，不 import 到本进程），
结果按当前 Windows 用户持久化到共享设置。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_CANDIDATE_DIRS = (
    Path("E:/miniconda/envs"),
    Path("C:/miniconda3/envs"),
    Path("C:/ProgramData/miniconda3/envs"),
    Path.home() / "miniconda3" / "envs",
    Path.home() / "anaconda3" / "envs",
    Path.home() / "Miniconda3" / "envs",
    Path.home() / "Anaconda3" / "envs",
)


@dataclass(frozen=True)
class OpenSimEnvInfo:
    executable: Path
    version: str | None


def _python_executable(env_dir: Path) -> Path:
    candidate = env_dir / "python.exe" if os.name == "nt" else env_dir / "bin" / "python"
    return candidate


def discover_opensim_python() -> list[OpenSimEnvInfo]:
    """扫描常见 conda 安装位置，返回能 ``import opensim`` 的环境列表。

    不修改本进程的 ``sys.path``，校验完全在子进程中进行。
    """
    found: list[OpenSimEnvInfo] = []
    seen: set[Path] = set()
    for base in _CANDIDATE_DIRS:
        if not base.is_dir():
            continue
        for env_dir in sorted(base.iterdir()):
            executable = _python_executable(env_dir)
            resolved = executable.resolve()
            if not resolved.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            info = _probe(executable)
            if info is not None:
                found.append(info)
    return found


def _probe(executable: Path) -> OpenSimEnvInfo | None:
    try:
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                "import opensim; print(opensim.GetVersionAndDate())",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    version = completed.stdout.strip().splitlines()[-1].strip() if completed.stdout.strip() else None
    return OpenSimEnvInfo(executable=executable.resolve(), version=version)


def validate_opensim_python(executable: str | Path) -> OpenSimEnvInfo | None:
    """校验用户选择的 ``python.exe`` 是否真的能 import opensim。"""
    path = Path(executable).expanduser().resolve()
    if not path.is_file():
        return None
    return _probe(path)


def pick_default_opensim_python(
    discovered: list[OpenSimEnvInfo],
) -> OpenSimEnvInfo | None:
    """从发现结果里挑一个默认项（优先名字含 ``opensim``）。"""
    if not discovered:
        return None
    for info in discovered:
        if "opensim" in info.executable.parts[-2].casefold():
            return info
    return discovered[0]


__all__ = [
    "OpenSimEnvInfo",
    "discover_opensim_python",
    "pick_default_opensim_python",
    "validate_opensim_python",
]
