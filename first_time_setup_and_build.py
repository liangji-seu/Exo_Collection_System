"""Zero-argument first setup and build using system CPython 3.11 x64.

This intentionally does not create or activate a virtual environment. Run the
offline hardware-runtime installer before this script when the frozen build
must include the official Xsens binding.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(label: str, arguments: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    completed = subprocess.run(arguments, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}"
        )


def _validate_interpreter() -> None:
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") * 8 != 64:
        raise RuntimeError(
            "需要 64 位 CPython 3.11；当前解释器为 "
            f"{sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit): "
            f"{sys.executable}"
        )
    if sys.prefix != sys.base_prefix:
        raise RuntimeError(
            "当前命令位于虚拟环境中。请退出虚拟环境后，使用系统 Python 执行："
            "python first_time_setup_and_build.py"
        )


def _verify_hardware_build_dependencies() -> None:
    check = (
        "import serial, scapy, xsensdeviceapi; "
        "from scapy.all import conf; "
        "assert conf.use_pcap, "
        "'Npcap/WinPcap API-compatible capture backend is unavailable'; "
        "print('HARDWARE_BUILD_DEPENDENCIES_OK')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", check], cwd=ROOT, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "真实硬件构建依赖不完整。请先解压并运行 SDK_Transfer 中最新的 "
            "INSTALL_HARDWARE_RUNTIME.cmd，同时安装 Npcap 并勾选 "
            "WinPcap API-compatible Mode，然后重新运行本脚本。"
        )


def main() -> int:
    _validate_interpreter()
    print(f"System Python: {sys.executable}")
    print("不会创建虚拟环境；依赖将安装到当前 Windows 用户的 Python 目录。")
    _run("检查 pip", [sys.executable, "-m", "pip", "--version"])
    _run(
        "更新构建工具",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "--upgrade",
            "pip",
            "setuptools>=69",
            "wheel",
        ],
    )
    _run(
        "安装应用、测试、打包和开源硬件依赖",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "--upgrade",
            "--no-build-isolation",
            "-e",
            f"{ROOT}[dev,packaging,hardware]",
        ],
    )
    _verify_hardware_build_dependencies()
    _run("运行完整测试", [sys.executable, "-m", "pytest", "-q"])
    _run("构建两个桌面应用", [sys.executable, str(ROOT / "build_exe.py")])
    print("\nFIRST_TIME_SETUP_AND_BUILD_OK")
    print(f"Collector: {ROOT / 'dist' / 'ExoCollector.exe'}")
    print(f"Data Studio: {ROOT / 'dist' / 'ExoDataStudio.exe'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n首次安装/构建失败：{type(exc).__name__}: {exc}")
        print("修复上面的错误后，直接重新运行同一个零参数脚本即可。")
        raise SystemExit(1) from exc
