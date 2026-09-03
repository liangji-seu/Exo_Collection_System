# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import json
import os
import subprocess
from PyInstaller.utils.hooks import collect_submodules


def _collect_tree(src: Path, dest_prefix: str, excludes: set[str]) -> list[tuple[str, str]]:
    """把一个目录递归地收集成 ``datas`` 的 (src, dest_dir) 二元组列表。

    PyInstaller 对「源为单个文件」的 datas 会**自动**把 ``basename(src)`` 接到
    dest 目录后面（见 ``PyInstaller/building/utils.py::format_binaries_and_datas``），
    所以这里的 dest 必须给**目录**而不是完整目标文件路径，否则文件名会重复一层。
    """
    pairs: list[tuple[str, str]] = []
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        if any(part in excludes for part in rel.parts):
            continue
        dest_dir = Path(dest_prefix) / rel.parent
        pairs.append((str(path), str(dest_dir)))
    return pairs

project_root = Path(SPECPATH).parent
source_root = project_root / "src"
pipeline_root = project_root / "opensim_joint_moment_pipeline"
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

# Exo Calculate 主进程只 import pipeline 的**数据/同步侧**（ezc3d/scipy/h5py/
# pandas），绝不 import opensim。pipeline 通过 ``ensure_pipeline_on_path()`` 在
# 运行时加入 sys.path，所以这里把它的**源码**当作数据文件打进 bundle，而不是
# 让 PyInstaller 静态分析它（否则会误把 opensim 依赖拖进主进程）。
#
# 同一份源码还供 opensim 子进程（opensim 环境 python.exe）在磁盘上运行——
# ``process_session.py`` 与它 import 的 ``run_precision_opensim`` 必须原样落盘。
hiddenimports = sorted(
    set(
        [
            "h5py",
            "numpy",
            "pyqtgraph",
            "scipy",
            "pandas",
            "ezc3d",
            "multiprocessing.popen_spawn_win32",
        ]
        + collect_submodules("exo_collection.apps.calculate")
    )
)

a = Analysis(
    [str(source_root / "exo_collection/apps/calculate/main.py")],
    pathex=[str(source_root)],
    binaries=[],
    datas=[
        (str(project_root / "config"), "config"),
        (str(project_root / "schemas"), "schemas"),
        (str(build_info), "exo_collection"),
        (str(source_root / "exo_collection/catalog/migrations"), "exo_collection/catalog/migrations"),
        # pipeline 源码（EXO 侧 import + opensim 子进程 import 共用），排除 pyc。
        *_collect_tree(
            pipeline_root / "pipeline",
            "opensim_joint_moment_pipeline/pipeline",
            {"__pycache__"},
        ),
        # opensim 子进程脚本（process_session 及其依赖的 run_precision_opensim）。
        *_collect_tree(
            pipeline_root / "scripts",
            "opensim_joint_moment_pipeline/scripts",
            {"__pycache__"},
        ),
        # gait2392 通用模型（Scale 起点）；排除 opensim.log 运行日志。
        *_collect_tree(
            pipeline_root / "data" / "models" / "gait2392",
            "opensim_joint_moment_pipeline/data/models/gait2392",
            {"opensim.log", "__pycache__"},
        ),
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
    name="ExoCalculate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
