"""IDTool setup XML 模板写出。ID 执行走 postprocess.opensim_pipeline.inverse_dynamics。
"""

from __future__ import annotations

from pathlib import Path


def write_id_setup(path: str | Path, **kwargs) -> None:
    model = kwargs.get("model", "scaled_model.osim")
    coord = kwargs.get("coordinates_file", "ik.mot")
    ext_loads = kwargs.get("external_loads", "external_loads.xml")
    out = kwargs.get("output", "inverse_dynamics.sto")
    xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<OpenSimDocument Version=\"40000\">\n"
        "  <InverseDynamicsTool name=\"id\">\n"
        f"    <model_file>{model}</model_file>\n"
        f"    <coordinates_file>{coord}</coordinates_file>\n"
        f"    <external_loads_file>{ext_loads}</external_loads_file>\n"
        f"    <output_gen_force_file>{out}</output_gen_force_file>\n"
        "  </InverseDynamicsTool>\n"
        "</OpenSimDocument>\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(xml, encoding="utf-8")


__all__ = ["write_id_setup"]
