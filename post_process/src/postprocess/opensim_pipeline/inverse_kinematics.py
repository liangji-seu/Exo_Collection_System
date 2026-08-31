"""IKTool wrapper（骨架）。

输入：scaled_model.osim + dynamic.trc。
输出：ik.mot + ik_marker_errors（见 qc.ik_qc）。

BLOCKING：marker weights（TODO，模板用 DEFAULT_INITIAL_VALUE）、
动态 trial 里虚拟关节中心默认不参与（见 config）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._bindings import PipelineBlockingError, require_opensim


def run_ik(config: dict[str, Any], *, model: str | Path = "scaled_model.osim",
           trc: str | Path = "dynamic.trc", out_motion: str | Path = "ik.mot") -> dict:
    osim = require_opensim()

    if not Path(model).is_file():
        raise PipelineBlockingError(f"scaled model 缺失：{model}")
    if not Path(trc).is_file():
        raise PipelineBlockingError(f"dynamic trc 缺失：{trc}")

    tool = osim.InverseKinematicsTool()
    tool.setModelFileName(str(model))
    tool.setMarkerDataFileName(str(trc))
    tool.setOutputMotionFileName(str(out_motion))
    # TODO: IKTaskSet marker weights —— config.ik.marker_weights（DEFAULT_INITIAL_VALUE）
    tool.run()
    return {"ok": True, "output_motion": str(out_motion)}


__all__ = ["run_ik"]
