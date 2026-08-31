"""IKTool setup XML 模板写出。

marker weights 与 final coordinate 选择是 TODO（见 MISSING_INFORMATION.md）。
第一版只落模板，IK 执行走 postprocess.opensim_pipeline.inverse_kinematics。
"""

from __future__ import annotations

from pathlib import Path


def write_ik_setup(path: str | Path, **kwargs) -> None:
    model = kwargs.get("model", "scaled_model.osim")
    trc = kwargs.get("trc", "dynamic.trc")
    out = kwargs.get("output_motion", "ik.mot")
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<OpenSimDocument Version=\"40000\">\n"
        "  <IKTool name=\"ik\">\n"
        f"    <model_file>{model}</model_file>\n"
        f"    <marker_file>{trc}</marker_file>\n"
        "    <!-- TODO: IKTaskSet marker weights (DEFAULT_INITIAL_VALUE) -->\n"
        f"    <output_motion_file>{out}</output_motion_file>\n"
        "  </IKTool>\n"
        "</OpenSimDocument>\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(xml, encoding="utf-8")


__all__ = ["write_ik_setup"]
