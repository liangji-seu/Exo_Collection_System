# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project_root = Path(SPECPATH).parent
source_root = project_root / "src"

a = Analysis(
    [str(source_root / "exo_collection/apps/collector/main.py")],
    pathex=[str(source_root)],
    binaries=[],
    datas=[
        (str(project_root / "config"), "config"),
        (str(project_root / "schemas"), "schemas"),
        (str(source_root / "exo_collection/catalog/migrations"), "exo_collection/catalog/migrations"),
    ],
    hiddenimports=[
        "h5py",
        "numpy",
        "pyqtgraph",
        "sqlalchemy.dialects.sqlite",
        "exo_collection.orchestration.simulated",
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
    name="ExoCollector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

