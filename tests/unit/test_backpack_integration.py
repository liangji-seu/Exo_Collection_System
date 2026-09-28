from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parents[2]
_PIPELINE = _REPO / "opensim_joint_moment_pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

import pipeline.opensim_io.prep_session as prep_module  # noqa: E402
from exo_collection.apps.calculate.workers import PrepWorker  # noqa: E402


def test_prep_worker_resolves_and_passes_backpack(tmp_path, monkeypatch):
    received: dict = {}

    def fake_prepare(**kwargs):
        received.update(kwargs)
        return {"manifest_path": str(tmp_path / "manifest.json")}

    monkeypatch.setattr(prep_module, "prepare_session", fake_prepare)
    worker = PrepWorker(
        static_c3d=Path("static.c3d"),
        dynamic_c3d=Path("dynamic.c3d"),
        gaitway_txt=Path("gaitway.txt"),
        generic_model=Path("model.osim"),
        out_dir=tmp_path,
        subject_id="003",
        mass_kg=75.0,
        height_m=1.75,
        gaitway_offset_s=0.5,
        condition_code="LWALK_LEVEL_2P5KG_EXO",
        condition_parameters={"load_kg": 2.5},
        session_dir=tmp_path,
    )
    failures: list[str] = []
    worker.signals.failed.connect(failures.append)

    worker.run()

    assert failures == []
    assert received["mass_kg"] == 75.0
    assert received["backpack"]["mass_kg"] == 2.5
    assert received["backpack"]["spec"]["parent_body"] == "torso"


@pytest.mark.parametrize("backpack", [None, {"enabled": True, "mass_kg": 5.0}])
def test_prepare_session_optionally_writes_backpack_manifest(
    tmp_path, monkeypatch, backpack
):
    data = SimpleNamespace(
        time_s=np.array([0.0, 1.0, 2.0]),
        n_frames=3,
        point_rate_hz=100.0,
    )
    gaitway = SimpleNamespace(
        time_s=np.array([0.0, 1.0, 2.0]),
        columns={"FzR(N)": np.ones(3), "FzL(N)": np.ones(3)},
    )
    window = SimpleNamespace(
        to_dict=lambda: {
            "start_s": 0.0,
            "end_s": 1.0,
            "duration_s": 1.0,
            "method": "test",
        }
    )
    steady = SimpleNamespace(
        start_s=0.0,
        end_s=1.0,
        to_dict=lambda: {"start_s": 0.0, "end_s": 1.0, "method": "test"},
    )
    monkeypatch.setattr(prep_module, "read_c3d", lambda _: data)
    monkeypatch.setattr(prep_module, "read_gaitway_ascii", lambda _: gaitway)
    monkeypatch.setattr(prep_module, "select_static_window", lambda _: window)
    monkeypatch.setattr(prep_module, "detect_steady_walking", lambda *a: steady)
    monkeypatch.setattr(
        prep_module,
        "build_trc",
        lambda *a, **k: (["marker"], np.zeros((3, 1, 3))),
    )
    monkeypatch.setattr(
        prep_module,
        "build_bilateral_grf",
        lambda *a, **k: ({}, np.ones(3, dtype=bool), {}),
    )
    monkeypatch.setattr(prep_module, "write_grf_mot", lambda *a, **k: None)
    monkeypatch.setattr(prep_module, "write_external_loads_xml", lambda *a, **k: None)
    monkeypatch.setattr(prep_module.shutil, "copyfile", lambda *a, **k: None)
    monkeypatch.setattr(prep_module.np, "save", lambda *a, **k: None)

    kwargs = {
        "static_c3d_path": Path("static.c3d"),
        "dynamic_c3d_path": Path("dynamic.c3d"),
        "gaitway_txt_path": Path("gaitway.txt"),
        "generic_model_path": Path("model.osim"),
        "out_dir": tmp_path / ("with" if backpack else "without"),
        "subject_id": "003",
        "mass_kg": 75.0,
        "height_m": 1.75,
        "gaitway_offset_s": 0.0,
    }
    if backpack is not None:
        kwargs["backpack"] = backpack

    summary = prep_module.prepare_session(**kwargs)
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["subject"]["mass_kg"] == 75.0
    if backpack is None:
        assert "backpack" not in manifest
    else:
        assert manifest["backpack"] == backpack
