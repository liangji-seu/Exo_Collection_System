"""ScaleTool setup XML 模板写出。

注意：这里只生成模板骨架，HH19 → gait2392 的 measurement / marker 映射是
BLOCKING（见 docs/MISSING_INFORMATION.md）。第一版以模板形式落盘，等模型与
marker mapping 确认后再填实。真正的 Scale 执行建议走 OpenSim 绑定
（postprocess.opensim_pipeline.scale），而不是手写 XML。
"""

from __future__ import annotations

from pathlib import Path


def write_scale_setup(path: str | Path, **kwargs) -> None:
    generic = kwargs.get("generic_model", "TODO_BLOCKING_generic_model_path")
    marker_file = kwargs.get("static_trc", "TODO_BLOCKING_static.trc")
    out_model = kwargs.get("output_model", "scaled_model.osim")
    mass_kg = kwargs.get("mass_kg", "TODO_BLOCKING")
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<OpenSimDocument Version=\"40000\">\n"
        "  <ScaleTool name=\"scale\">\n"
        f"    <mass>{mass_kg}</mass>\n"
        f"    <generic_model_maker><model_file>{generic}</model_file></generic_model_maker>\n"
        "    <!-- TODO_BLOCKING: MeasurementSet（HH19 → gait2392 body scaling 映射未确认） -->\n"
        "    <ModelScaler>\n"
        "      <scaling_order>\n"
        "        <!-- TODO_BLOCKING: scaling order -->\n"
        "      </scaling_order>\n"
        f"      <marker_file>{marker_file}</marker_file>\n"
        "    </ModelScaler>\n"
        f"    <output_model_file>{out_model}</output_model_file>\n"
        "  </ScaleTool>\n"
        "</OpenSimDocument>\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(xml, encoding="utf-8")


__all__ = ["write_scale_setup"]
