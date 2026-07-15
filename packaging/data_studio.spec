# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import json
import subprocess

project_root = Path(SPECPATH).parent
source_root = project_root / "src"
build_info = project_root / "build" / "build-info.json"
build_info.parent.mkdir(parents=True, exist_ok=True)
try:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    ).stdout.strip()
except (OSError, subprocess.SubprocessError):
    git_commit = "unknown-local-build"
build_info.write_text(json.dumps({"git_commit": git_commit}) + "\n", encoding="utf-8")

a = Analysis(
    [str(source_root / "exo_collection/apps/data_studio/main.py")],
    pathex=[str(source_root)],
    binaries=[],
    datas=[
        (str(project_root / "config"), "config"),
        (str(project_root / "schemas"), "schemas"),
        (str(build_info), "exo_collection"),
        (str(source_root / "exo_collection/catalog/migrations"), "exo_collection/catalog/migrations"),
    ],
    hiddenimports=[
        "h5py",
        "numpy",
        "pyqtgraph",
        "sqlalchemy.dialects.sqlite",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ExoDataStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
