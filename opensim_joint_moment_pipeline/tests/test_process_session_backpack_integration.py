from __future__ import annotations

import ast
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "process_session.py"


def _load_helpers():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_add_backpack_qc_audit", "_add_backpack_viewer_audit"}
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"json": json, "Path": Path}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace


def _receipt(*, enabled=True, mass=5.0, geometry="assumed"):
    return {
        "enabled": enabled,
        "mass_kg": mass,
        "source": "nominal",
        "geometry_status": geometry,
        "human_model_path": "human.osim",
        "human_model_sha256": "human-hash",
        "actual_model_path": "loaded.osim",
        "actual_model_sha256": "loaded-hash",
        "generator_receipt": {"total_backpack_mass_kg": mass},
        "totals": {"resolved_mass_kg": mass, "generated_backpack_mass_kg": mass},
        "warnings": [],
    }


def test_assumed_geometry_adds_warn_without_downgrading_fail():
    add_qc = _load_helpers()["_add_backpack_qc_audit"]
    receipt = _receipt()
    failed = {"status": "FAIL", "summary": "existing failure", "checks": []}

    add_qc(failed, receipt)

    assert failed["status"] == "FAIL"
    assert failed["summary"] == "existing failure"
    assert failed["checks"][-1]["key"] == "backpack_geometry_assumed"
    assert "nominal" in failed["checks"][-1]["detail"]
    assert failed["backpack"] is receipt


def test_disabled_zero_adds_audit_but_no_qc_warning():
    add_qc = _load_helpers()["_add_backpack_qc_audit"]
    receipt = _receipt(enabled=False, mass=0.0, geometry="not_needed")
    report = {"status": "PASS", "summary": "ok", "checks": []}

    add_qc(report, receipt)

    assert report["status"] == "PASS"
    assert report["checks"] == []
    assert report["backpack"] is receipt


def test_viewer_metadata_records_backpack_receipt(tmp_path):
    add_viewer = _load_helpers()["_add_backpack_viewer_audit"]
    receipt = _receipt()
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "viewer_meta.json").write_text('{"n_frames": 10}', encoding="utf-8")
    summary = {"viewer_dir": str(viewer), "n_frames": 10}

    add_viewer(summary, viewer, receipt)

    assert summary["backpack"] is receipt
    meta = json.loads((viewer / "viewer_meta.json").read_text(encoding="utf-8"))
    assert meta["backpack"]["actual_model_sha256"] == "loaded-hash"


def test_id_model_is_inserted_after_dynamic_ik_and_used_only_downstream():
    source = SCRIPT.read_text(encoding="utf-8")
    dynamic_ik = source.index('final_ik = Path("hh19_static_calibrated_ik.mot")')
    prepare = source.index("id_model, backpack_receipt = prepare_id_model")
    id_setup = source.index("_id_setup_xml(", prepare)

    assert dynamic_ik < prepare < id_setup
    assert "id_model.name, final_ik.name" in source[id_setup:id_setup + 300]
    assert "marker_qc = _marker_qc(final_model" in source
    assert "model_path=Path(id_model.name)" in source
    assert '"human_model": str(out / final_model.name)' in source
