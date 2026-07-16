# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import json
import os
import subprocess
from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH).parent
source_root = project_root / "src"
build_info = project_root / "build" / "build-info.json"
build_info.parent.mkdir(parents=True, exist_ok=True)
git_commit = os.environ.get("EXO_BUILD_GIT_COMMIT", "").strip()
git_dirty = os.environ.get("EXO_BUILD_GIT_DIRTY", "").strip().lower()
application_version = os.environ.get("EXO_BUILD_APP_VERSION", "").strip()
if not git_commit:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
        git_dirty = str(
            bool(
                subprocess.run(
                    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                    cwd=project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    shell=False,
                ).stdout.strip()
            )
        ).lower()
    except (OSError, subprocess.SubprocessError):
        git_commit = "unknown-local-build"
        git_dirty = "unknown"
build_info.write_text(
    json.dumps(
        {
            "application_version": application_version or "unknown",
            "git_commit": git_commit,
            "git_worktree_dirty": (
                git_dirty == "true" if git_dirty in {"true", "false"} else None
            ),
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

# Data Studio owns several spawn-process tools.  Some import their operation
# only inside the child entry point, and the SSH/SCP backend imports Paramiko
# lazily so credentials never touch startup configuration.  Explicitly collect
# these modules for frozen-child imports.
hiddenimports = sorted(
    set(
        [
            "h5py",
            "numpy",
            "pyqtgraph",
            "scp",
            "sqlalchemy.dialects.sqlite",
            "multiprocessing.popen_spawn_win32",
            "exo_collection.apps.data_studio.local_tools",
            "exo_collection.apps.data_studio.process_workers",
            "exo_collection.apps.data_studio.recovery_service",
            "exo_collection.apps.data_studio.upload",
            "exo_collection.external.importer",
            "exo_collection.storage.recovery_manager",
        ]
        + collect_submodules("exo_collection.apps.data_studio")
        + collect_submodules("exo_collection.external")
        + collect_submodules("exo_collection.readers")
        + collect_submodules("exo_collection.storage")
        + collect_submodules("paramiko")
    )
)

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
    hiddenimports=hiddenimports,
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
