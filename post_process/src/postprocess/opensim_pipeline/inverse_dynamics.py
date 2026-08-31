"""IDTool wrapper（骨架）。

输入：scaled_model.osim + ik.mot + external_loads.xml + grf.mot。
输出：inverse_dynamics.sto。

第一版 RRA 默认关闭。GRF 必须已转换到 OpenSim ground 且左右脚分解完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._bindings import PipelineBlockingError, require_opensim


def run_id(config: dict[str, Any], *, model: str | Path = "scaled_model.osim",
           ik_mot: str | Path = "ik.mot", ext_loads: str | Path = "external_loads.xml",
           out: str | Path = "inverse_dynamics.sto") -> dict:
    osim = require_opensim()

    for f in (model, ik_mot, ext_loads):
        if not Path(f).is_file():
            raise PipelineBlockingError(f"缺失输入：{f}")

    tool = osim.InverseDynamicsTool()
    tool.setModelFileName(str(model))
    tool.setCoordinatesFileName(str(ik_mot))
    tool.setExternalLoadsFileName(str(ext_loads))
    tool.setOutputGenForceFileName(str(out))
    tool.run()
    return {"ok": True, "output": str(out)}


__all__ = ["run_id"]
