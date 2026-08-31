import numpy as np

from postprocess.blocking import Status
from postprocess.validation.validate_config import validate_config


def _base_config(**overrides):
    cfg = {
        "subject": {"id": "S01", "mass_kg": "TODO_BLOCKING", "height_m": None},
        "files": {"static_c3d": "TODO_BLOCKING", "dynamic_c3d": "TODO_BLOCKING",
                  "generic_model": {"path": "TODO_BLOCKING"}},
        "marker": {"protocol": "Helen_Hayes_19", "input_unit": "TODO_BLOCKING",
                   "use_virtual_sacral": "TODO_BLOCKING", "mapping": {}},
        "transforms": {
            "forceplate_to_mocap": {"status": "TODO_BLOCKING_CALIBRATION",
                                    "rotation_matrix": None, "translation_m": None},
            "mocap_to_opensim": {"status": "TODO_BLOCKING_CONFIRM_AXES",
                                 "rotation_matrix": None},
        },
        "force": {"convention": "TODO_BLOCKING"},
    }
    cfg.update(overrides)
    return cfg


def test_all_todo_is_blocking():
    pre = validate_config(_base_config())
    assert pre.get("subject_mass").status is Status.BLOCKING
    assert pre.get("forceplate_to_mocap_transform").status is Status.BLOCKING
    assert pre.get("grf_force_convention").status is Status.BLOCKING
    assert pre.max_executable_stage() in ("NONE", "C3D_INSPECTION")


def test_filled_mass_and_transform_ready():
    cfg = _base_config(
        subject={"id": "S01", "mass_kg": 75.0, "height_m": 1.75},
        transforms={
            "forceplate_to_mocap": {"status": "calibrated",
                                    "rotation_matrix": np.eye(3).tolist(),
                                    "translation_m": [0.0, 0.0, 0.0]},
            "mocap_to_opensim": {"status": "confirmed",
                                 "rotation_matrix": np.eye(3).tolist()},
        },
        force={"convention": "ground_on_foot"},
    )
    pre = validate_config(cfg)
    assert pre.get("subject_mass").status is Status.READY
    assert pre.get("forceplate_to_mocap_transform").status is Status.READY
    assert pre.get("mocap_to_opensim_transform").status is Status.READY
    assert pre.get("grf_force_convention").status is Status.READY
