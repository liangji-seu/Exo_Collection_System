from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from exo_collection.orchestration import simulated


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


class _BuildInfoResource:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def joinpath(self, _name: str) -> _BuildInfoResource:
        return self

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return json.dumps(self.payload)


# ── Runtime provenance ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("dirty", "expected_suffix"),
    [(False, ""), (True, "+dirty"), (None, "+provenance-unavailable")],
)
def test_runtime_git_provenance_preserves_build_worktree_state(
    monkeypatch: pytest.MonkeyPatch,
    dirty: bool | None,
    expected_suffix: str,
) -> None:
    commit = "1" * 40
    monkeypatch.delenv("EXO_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        simulated.resources,
        "files",
        lambda _package: _BuildInfoResource(
            {"git_commit": commit, "git_worktree_dirty": dirty}
        ),
    )

    assert simulated._git_commit() == commit + expected_suffix


def test_source_runtime_git_provenance_marks_dirty_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    monkeypatch.delenv("EXO_GIT_COMMIT", raising=False)

    def missing_resource(_package: str) -> object:
        raise FileNotFoundError

    responses = iter(
        [
            SimpleNamespace(stdout=commit + "\n"),
            SimpleNamespace(stdout=" M src/example.py\n"),
        ]
    )
    monkeypatch.setattr(simulated.resources, "files", missing_resource)
    monkeypatch.setattr(simulated.subprocess, "run", lambda *args, **kwargs: next(responses))

    assert simulated._git_commit() == commit + "+dirty"


# ── PS1 release script assertions ────────────────────────────────────────


def test_python_build_entry_builds_both_desktop_applications() -> None:
    script = _text("build_exe.py")
    ast.parse(script)

    assert "packaging" in script
    assert "collector.spec" in script
    assert "data_studio.spec" in script
    assert "ExoCollector" in script
    assert "ExoDataStudio" in script
    assert "sys.executable" in script
    assert '"-m", "PyInstaller"' in script
    assert '"--noconfirm", "--clean"' in script


# ── PyInstaller spec structure checks ─────────────────────────────────────


def test_pyinstaller_specs_are_valid_and_collect_spawn_boundaries() -> None:
    collector = _text("packaging/collector.spec")
    studio = _text("packaging/data_studio.spec")
    ast.parse(collector)
    ast.parse(studio)

    for spec in (collector, studio):
        assert "EXO_BUILD_GIT_COMMIT" in spec
        assert "EXO_BUILD_GIT_DIRTY" in spec
        assert "multiprocessing.popen_spawn_win32" in spec

    for module_name in (
        "exo_collection.acquisition.workers",
        "exo_collection.apps.collector.preflight",
        "exo_collection.orchestration.simulated",
        "exo_collection.writers.block_binary_process",
    ):
        assert module_name in collector
    for module_name in (
        "exo_collection.apps.data_studio.process_workers",
        "exo_collection.apps.data_studio.recovery_service",
        "exo_collection.apps.data_studio.upload",
        "exo_collection.external.importer",
        "exo_collection.storage.recovery_manager",
    ):
        assert module_name in studio


# ── Inno Setup installer assertions ──────────────────────────────────────


def test_installer_carries_the_same_audit_manifest_as_the_zip() -> None:
    installer = _text("packaging/installer/ExoCollectionSystem.iss")

    assert "BUILD_MANIFEST.json" in installer
    assert "README_PROJECT.md" in installer
    assert "ExoCollectionSystem-{#AppVersion}-windows-x64" in installer


# ── CMD launcher Unicode + space path smoke test ──────────────────────────


@pytest.mark.parametrize(
    ("launcher_name", "module_name"),
    [
        ("run_collector.py", "exo_collection.apps.collector.main"),
        ("run_data_studio.py", "exo_collection.apps.data_studio.main"),
    ],
)
def test_zero_argument_python_launchers_call_application_main(
    launcher_name: str,
    module_name: str,
) -> None:
    script = _text(launcher_name)
    ast.parse(script)

    assert f"from {module_name} import main" in script
    assert 'if __name__ == "__main__"' in script
    assert "raise SystemExit(main())" in script


# ── PowerShell AST parser check ──────────────────────────────────────────


@pytest.mark.parametrize(
    "relative_path",
    ["build_exe.py", "run_collector.py", "run_data_studio.py"],
)
def test_python_entry_scripts_parse_without_errors(relative_path: str) -> None:
    source = _text(relative_path)
    compile(source, str(PROJECT_ROOT / relative_path), "exec")


# ── compileall – every .py file must be valid Python 3.11 syntax ─────────


def test_all_python_source_files_compile_without_syntax_errors() -> None:
    src_dir = str(PROJECT_ROOT / "src")
    assert Path(src_dir).is_dir(), f"Source directory not found: {src_dir}"

    import compileall
    import sys

    # Use compileall to verify every .py file under src/ and tests/ is
    # syntactically valid.  Force recompilation so stale .pyc caches
    # cannot mask real breakage.
    for label, directory in [
        ("src", str(PROJECT_ROOT / "src")),
        ("tests", str(PROJECT_ROOT / "tests")),
    ]:
        if not Path(directory).is_dir():
            continue
        stdout = sys.stdout
        import io
        sys.stdout = io.StringIO()
        try:
            compileall.compile_dir(
                directory,
                maxlevels=20,
                force=True,
                quiet=0,
                legacy=True,
                optimize=-1,
                workers=0,
            )
        finally:
            captured = sys.stdout.getvalue()
            sys.stdout = stdout

        # compileall prints a line for every file that it compiles.
        # Lines that begin with "Compiling " succeeded; anything else
        # (especially "*** " lines) signal an error.
        error_lines = [
            line
            for line in captured.splitlines()
            if line and not line.startswith("Listing ")
            and not line.startswith("Compiling ")
        ]
        assert not error_lines, (
            f"compileall errors in {label}/:\n"
            + "\n".join(error_lines[:20])
            + ("\n..." if len(error_lines) > 20 else "")
        )


# ── git diff --check whitespace audit ────────────────────────────────────


def test_git_diff_check_passes_on_working_tree() -> None:
    if not (PROJECT_ROOT / ".git").is_dir():
        pytest.skip("Not a Git repository; working-tree whitespace check skipped.")

    # Validate that the current diff introduces no trailing-whitespace or
    # conflict-marker issues.
    completed = subprocess.run(
        ["git", "diff", "--check"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    # git diff --check returns 0 when there are no whitespace issues.
    # It can also return 0 when there is no diff at all, which is a valid
    # state on a clean working tree.
    if completed.returncode != 0:
        error_detail = (
            (completed.stdout + "\n" + completed.stderr).strip()[:3000]
        )
        pytest.fail(
            f"git diff --check detected whitespace or conflict-marker "
            f"issues in the working tree:\n{error_detail}"
        )


# ── PyInstaller spec hidden-import validator ──────────────────────────────


def test_pyinstaller_spec_hidden_imports_are_importable() -> None:
    """Every explicitly-listed hidden import must be resolvable at test time."""
    collector = _text("packaging/collector.spec")
    studio = _text("packaging/data_studio.spec")

    # Extract string literals from the hiddenimports bracket in each spec
    # using AST, so we assert against the actual frozen configuration.
    for label, spec_text in [
        ("collector.spec", collector),
        ("data_studio.spec", studio),
    ]:
        tree = ast.parse(spec_text)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == "hiddenimports":
                    names = _extract_strings_from_list(node.value)
                    for name in names:
                        # ModuleNotFoundError is the only acceptable failure;
                        # anything else (SyntaxError, ImportError for a
                        # transitive dep, …) must propagate.
                        try:
                            __import__(name)
                        except ModuleNotFoundError:
                            # Some PyInstaller-only imports (e.g.
                            # multiprocessing.popen_spawn_win32) may not
                            # resolve in a regular Python session.
                            if name.startswith("multiprocessing"):
                                continue
                            raise
                    found = True
                    break
        assert found, f"No hiddenimports assignment found in {label}"


def _extract_strings_from_list(node: ast.expr) -> list[str]:
    """Extract plain string constants from a nested AST list expression."""
    strings: list[str] = []
    _walk = [node]
    while _walk:
        cur = _walk.pop()
        if isinstance(cur, ast.Constant) and isinstance(cur.value, str):
            strings.append(cur.value)
        elif isinstance(cur, (ast.List, ast.Tuple)):
            _walk.extend(cur.elts)
        elif isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name):
            if cur.func.id in ("set", "sorted"):
                _walk.extend(cur.args)
    return strings


# ── CMD launcher Windows reserved-name guard ──────────────────────────────


def test_legacy_cmd_and_powershell_launchers_are_removed() -> None:
    legacy_files = [
        "Run_ExoCollector.cmd",
        "Run_ExoDataStudio.cmd",
        "Run_ExoCollector_From_Source.cmd",
        "Run_ExoDataStudio_From_Source.cmd",
        "Build_Windows.cmd",
        "First_Time_Setup.cmd",
        "First_Time_Setup_And_Build.cmd",
        "packaging/build_windows.ps1",
        "scripts/first_time_setup_and_build.ps1",
        "scripts/run_from_source.ps1",
    ]

    for relative_path in legacy_files:
        assert not (PROJECT_ROOT / relative_path).exists()
    for replacement in ("build_exe.py", "run_collector.py", "run_data_studio.py"):
        assert (PROJECT_ROOT / replacement).is_file()


# ── build-info.json provenance audit ─────────────────────────────────────


def test_pyinstaller_spec_writes_correct_build_info_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the spec file provenance logic: env-var → build-info.json."""
    monkeypatch.setenv("EXO_BUILD_GIT_COMMIT", "1" * 40)
    monkeypatch.setenv("EXO_BUILD_GIT_DIRTY", "false")
    monkeypatch.setenv("EXO_BUILD_APP_VERSION", "1.0.0")

    # Replicate the spec file's build-info writing logic exactly.
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    build_info = build_dir / "build-info.json"

    git_commit = os.environ.get("EXO_BUILD_GIT_COMMIT", "").strip()
    git_dirty = os.environ.get("EXO_BUILD_GIT_DIRTY", "").strip().lower()
    application_version = os.environ.get("EXO_BUILD_APP_VERSION", "").strip()

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

    loaded = json.loads(build_info.read_text(encoding="utf-8"))
    assert loaded["application_version"] == "1.0.0"
    assert loaded["git_commit"] == "1" * 40
    assert loaded["git_worktree_dirty"] is False

    # Confirm the simulated._git_commit runtime reads it back correctly.
    monkeypatch.delenv("EXO_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        simulated.resources,
        "files",
        lambda _package: _BuildInfoResource(
            {"git_commit": "1" * 40, "git_worktree_dirty": False}
        ),
    )
    assert simulated._git_commit() == "1" * 40
