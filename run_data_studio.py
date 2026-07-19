"""一键启动 Exo Data Studio（数据管理端）。

Windows 上存在多个 Python 时，``python`` 可能指向缺少 SSH 依赖的
Conda 解释器。本入口会优先继续使用当前解释器；只有当前解释器无法
导入 Paramiko/SCP、而 ``py -3.11`` 对应解释器完整时，才自动重启到
那个 CPython 3.11。正常使用仍然只需 ``python run_data_studio.py``。
"""

import os
from pathlib import Path
import subprocess
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
_NETWORK_RUNTIME_CHECK = "import paramiko, scp, PySide6, sqlalchemy, h5py"


def _network_runtime_available(executable: str) -> bool:
    completed = subprocess.run(
        [executable, "-c", _NETWORK_RUNTIME_CHECK],
        cwd=SOURCE_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _system_python311() -> str | None:
    if os.name != "nt":
        return None
    completed = subprocess.run(
        [
            "py",
            "-3.11",
            "-c",
            f"{_NETWORK_RUNTIME_CHECK}; import sys; print(sys.executable)",
        ],
        cwd=SOURCE_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _relaunch_with_complete_runtime() -> int | None:
    if _network_runtime_available(sys.executable):
        return None
    candidate = _system_python311()
    if not candidate or Path(candidate).resolve() == Path(sys.executable).resolve():
        return None
    print(
        "当前 Python 缺少 Data Studio SSH 依赖，自动切换到："
        f"{candidate}",
        flush=True,
    )
    return subprocess.call([candidate, str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    relaunched_exit_code = _relaunch_with_complete_runtime()
    if relaunched_exit_code is not None:
        raise SystemExit(relaunched_exit_code)
    sys.path.insert(0, str(SOURCE_ROOT))
    from exo_collection.apps.data_studio.main import main

    raise SystemExit(main())
