# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import importlib.util
import json
import os
import subprocess
from PyInstaller.utils.hooks import collect_submodules


def optional_submodules(package):
    """Collect a hardware package only when it is installed for this build."""

    return collect_submodules(package) if importlib.util.find_spec(package) else []

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

# Windows ``spawn`` workers import orchestration and writer implementations in
# the child process.  Keep those modules explicit so a frozen executable is
# validated against the same process boundaries used by the source build.
hiddenimports = sorted(
    set(
        [
            "h5py",
            "numpy",
            "pyqtgraph",
            "sqlalchemy.dialects.sqlite",
            "multiprocessing.popen_spawn_win32",
            "exo_collection.acquisition.workers",
            "exo_collection.apps.collector.preflight",
            "exo_collection.orchestration.simulated",
            "exo_collection.writers.block_binary_process",
        ]
        + collect_submodules("exo_collection.adapters")
        + collect_submodules("exo_collection.quality")
        + collect_submodules("exo_collection.reporting")
        + collect_submodules("exo_collection.writers")
        + optional_submodules("serial")
        + optional_submodules("zeroconf")
        + optional_submodules("pythonnet")
        + optional_submodules("clr")
        + optional_submodules("xsensdeviceapi")
    )
)

a = Analysis(
    [str(source_root / "exo_collection/apps/collector/main.py")],
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
    name="ExoCollector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
