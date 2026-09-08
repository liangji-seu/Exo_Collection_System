"""QC provenance for the CSV actually exported, independent of later runs."""

import json
from pathlib import Path


def write_ground_truth_qc(csv_path: Path, status: str | None, run_dir: Path) -> None:
    stat = csv_path.stat()
    destination = csv_path.with_suffix(".qc.json")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "qc_status": status,
        "run_dir": str(run_dir.resolve()),
        "csv_size": stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
    }), encoding="utf-8")
    temporary.replace(destination)


def read_ground_truth_qc(session_dir: Path) -> str | None:
    try:
        csv_path = session_dir / "ground_truth.csv"
        stat = csv_path.stat()
        document = json.loads(csv_path.with_suffix(".qc.json").read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return None
        if (document.get("csv_size"), document.get("csv_mtime_ns")) != (stat.st_size, stat.st_mtime_ns):
            return None
        status = document.get("qc_status")
        return status if status in ("PASS", "WARN", "FAIL") else None
    except (OSError, ValueError):
        return None
