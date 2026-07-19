"""Streaming file integrity helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading a large artifact in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_manifest(
    trial_directory: str | Path,
    relative_paths: Iterable[str | Path],
    destination: str | Path | None = None,
) -> Path:
    """Write a deterministic sha256sum-compatible manifest atomically."""

    root = Path(trial_directory).resolve()
    output = Path(destination) if destination is not None else root / "checksums.sha256"
    if not output.is_absolute():
        output = root / output
    partial = output.with_name(output.name + ".partial")
    lines: list[str] = []
    for relative in sorted(Path(item).as_posix() for item in relative_paths):
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Artifact escapes Trial directory: {relative}")
        lines.append(f"{sha256_file(path)}  {relative}\n")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        stream.writelines(lines)
        stream.flush()
    partial.replace(output)
    return output


def verify_checksum_manifest(
    path: str | Path,
    *,
    trial_root: str | Path | None = None,
) -> dict[str, bool]:
    checksum_path = Path(path)
    root = (
        Path(trial_root).resolve()
        if trial_root is not None
        else checksum_path.parent.resolve()
    )
    results: dict[str, bool] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64:
            raise ValueError(f"Invalid checksum line: {line!r}")
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ValueError(f"Checksum path escapes Trial directory: {relative}")
        results[relative] = candidate.is_file() and sha256_file(candidate) == expected.lower()
    return results

