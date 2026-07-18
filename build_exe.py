"""一键编译 ExoCollector.exe 和 ExoDataStudio.exe"""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC_DIR = ROOT / "packaging"
COLLECTOR_SPEC = SPEC_DIR / "collector.spec"
DATA_STUDIO_SPEC = SPEC_DIR / "data_studio.spec"


def run_pyinstaller(spec: Path, label: str) -> None:
    print(f"[1/2] 正在编译 {label} ...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"[ERROR] {label} 编译失败 (exit code {result.returncode})")
        sys.exit(result.returncode)
    exe = ROOT / "dist" / f"{label}.exe"
    if exe.is_file():
        print(f"        -> {exe}")
    else:
        print(f"[WARN] 未找到预期的输出文件: {exe}")


def main() -> int:
    COLLECTOR_SPEC.resolve(strict=True)
    DATA_STUDIO_SPEC.resolve(strict=True)

    run_pyinstaller(COLLECTOR_SPEC, "ExoCollector")
    run_pyinstaller(DATA_STUDIO_SPEC, "ExoDataStudio")

    print()
    print("编译完成。产物在 dist/ 目录下:")
    for exe in sorted((ROOT / "dist").glob("*.exe")):
        print(f"  {exe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
