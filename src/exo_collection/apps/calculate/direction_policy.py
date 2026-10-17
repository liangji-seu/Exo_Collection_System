"""Check provenance before reusing or exporting backward-walking results."""
import json
from pathlib import Path
from ._pipeline import ensure_pipeline_on_path


GRF_INPUT_POLICY_VERSION = "continuous_native_wrench_v2"
EXPORT_VALIDITY_VERSION = "bilateral_analysis_v1"


def current_grf_policy(manifest: dict | None) -> bool:
    return bool(manifest and (manifest.get("processing") or {}).get("grf_input_policy_version") == GRF_INPUT_POLICY_VERSION)


def current_run_policy(manifest: dict | None) -> bool:
    if not current_grf_policy(manifest):
        return False
    processing = (manifest or {}).get("processing") or {}
    code = processing.get("condition_code", "")
    return not is_backward_code(code) or current_direction_policy(manifest)


def is_backward_code(code: str) -> bool:
    return "BWALK" in code.upper().split("_")


def current_direction_policy(manifest: dict | None) -> bool:
    ensure_pipeline_on_path()
    from pipeline.opensim_io.walking_direction import DIRECTION_POLICY_VERSION
    return bool(manifest and (manifest.get("processing") or {}).get("direction_policy_version") == DIRECTION_POLICY_VERSION
                and (manifest.get("gaitway") or {}).get("anatomical_mapping"))


def load_run_manifest(run_dir: Path | None) -> dict | None:
    try:
        return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if run_dir else None
    except (OSError, ValueError):
        return None


def backward_export_is_current(session_dir: Path) -> bool:
    try:
        receipt = json.loads((session_dir / "ground_truth.qc.json").read_text(encoding="utf-8"))
        csv = (session_dir / "ground_truth.csv").stat()
        if (receipt.get("csv_size"), receipt.get("csv_mtime_ns")) != (csv.st_size, csv.st_mtime_ns):
            return False
        return (receipt.get("export_validity_version") == EXPORT_VALIDITY_VERSION
                and current_run_policy(load_run_manifest(Path(receipt["run_dir"]))))
    except (OSError, ValueError, KeyError):
        return False


# Historical public name kept for callers; policy now covers all walking directions.
export_is_current = backward_export_is_current
