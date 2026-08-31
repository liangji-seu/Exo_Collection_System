"""ScaleTool wrapper（骨架）。

输入：generic_model + static.trc + subject.mass + HH19 measurement。
输出：scaled_model.osim + scale_report.json。

BLOCKING 项（见 docs/MISSING_INFORMATION.md）：
- gait2392 generic osim 路径
- HH19 → gait2392 的 measurement / marker mapping（不可仅凭名字相似推导）
- 受试者体重
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._bindings import PipelineBlockingError, require_opensim


def run_scale(config: dict[str, Any], *, static_trc: str | Path | None = None,
              out_model: str | Path = "scaled_model.osim") -> dict:
    osim = require_opensim()

    generic = config.get("files", {}).get("generic_model", {}).get("path")
    if not generic or not Path(generic).is_file():
        raise PipelineBlockingError("generic model (gait2392) 缺失，无法 Scale")

    static = static_trc or config.get("files", {}).get("static_c3d")
    if static is None:
        raise PipelineBlockingError("static.trc 缺失，无法 Scale")

    mass = config.get("subject", {}).get("mass_kg")
    if mass is None:
        raise PipelineBlockingError("受试者体重缺失，无法 Scale")

    # 骨架：真正执行需要 HH19 measurement 映射（BLOCKING）。这里先构建工具对象，
    # 等 measurement 映射确认后填入 MeasurementSet。
    tool = osim.ScaleTool()
    tool.getGenericModelMaker().setModelFileName(str(generic))
    tool.getModelScaler().setMarkerFileName(str(static))
    tool.setSubjectMass(float(mass))
    tool.setOutputModelFileName(str(out_model))
    # TODO_BLOCKING: tool.getModelScaler().setMeasurementSet(...)  —— 待映射确认
    tool.run()
    return {
        "ok": True,
        "output_model": str(out_model),
        "note": "Scale 已执行（MeasurementSet 映射仍未确认，结果待校验）",
    }


__all__ = ["run_scale"]
