"""OpenSim 下游编排：自定义 HH19 MarkerSet → Scale(含 MarkerPlacer) → IK → ID。

**只在 opensim 环境运行**（numpy 2.x + opensim 4.6）。

流程：
1. 给通用 gait2392 模型加 19 个 HH19 marker → ``*_markers.osim``；
2. ScaleTool（内部先 MarkerPlacer 用 static trial 精定位 marker，再按测量缩放段长/质量）
   → ``*_scaled.osim``；
3. IKTool（scaled 模型 + 动态 TRC）→ ``*_ik.mot``；
4. IDTool（scaled 模型 + ik.mot + ExternalLoads）→ ``*_id.mot``（关节力矩）。

所有 setup 用字符串生成后写盘，再 `Tool(path).run()`，与官方工作流一致、可读可查。

**Helen Hayes 关键差异**：静态 trial 19 个 marker（含 medial 膝/踝，用于定关节中心），
动态 trial 只有 15 个（medial 摘除）。因此 MarkerPlacer 的 IK task 按**静态 TRC**实际
marker 过滤，主 IK 的 task 按**动态 TRC**实际 marker 过滤——不在 TRC 里的 marker 不建 task。
"""

from __future__ import annotations

import os
from pathlib import Path

import opensim as osim

from .hh19_markers import (
    HH19_MARKERS,
    IK_MARKER_WEIGHTS,
    SCALE_MEASUREMENTS,
    LOCKED_COORDINATES,
)

# gait2392 无躯干 marker，锁定腰椎（骨盆-躯干）自由度在默认位形
_LOCKED_EXTRA = ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]


# --------------------------------------------------------------------------- #
# marker 添加
# --------------------------------------------------------------------------- #
def add_hh19_markers(model: osim.Model) -> None:
    ms = model.getMarkerSet()
    for name, (body_name, loc) in HH19_MARKERS.items():
        mk = osim.Marker()
        mk.setName(name)
        mk.set_location(osim.Vec3(loc[0], loc[1], loc[2]))
        # 必须用 setParentFrameName 且填**绝对路径**（/bodyset/pelvis）：
        # - 传 Body 对象给构造器 / connectSocket_parent_frame 只连了对象引用，
        #   printToXML 时 connectee path 为空 → 重载报 “parent_frame unspecified”；
        # - 填裸名 'pelvis' 序列化后重载时找不到该路径。
        body = model.getBodySet().get(body_name)
        mk.setParentFrameName(body.getAbsolutePathString())
        ms.adoptAndAppend(mk)
    model.finalizeConnections()


# --------------------------------------------------------------------------- #
# TRC 元信息（不依赖 opensim，直接解析我们 write_trc 写出的格式）
# --------------------------------------------------------------------------- #
def _trc_info(path: str) -> tuple[list[str], tuple[float, float]]:
    """返回 (marker_names, (t0, t1))。marker 名在第 4 行（每名占 3 列），时间在第 2 列。"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header = lines[3].split("\t")
    markers = [header[i] for i in range(2, len(header), 3) if header[i]]
    # 头部 5 行（PathFileType / DataRate / 数值 / marker 名 / 坐标标签），数据从第 6 行起
    t0 = float(lines[5].split("\t")[1])
    t1 = float(lines[-1].split("\t")[1])
    return markers, (t0, t1)


# --------------------------------------------------------------------------- #
# setup XML 生成
# --------------------------------------------------------------------------- #
def _marker_task_xml(name: str, weight: float) -> str:
    return (
        f'\t\t\t\t<IKMarkerTask name="{name}">\n'
        f'\t\t\t\t\t<apply>true</apply>\n'
        f'\t\t\t\t\t<weight>{weight}</weight>\n'
        f'\t\t\t\t</IKMarkerTask>\n'
    )


def _coordinate_task_xml(name: str) -> str:
    return (
        f'\t\t\t\t<IKCoordinateTask name="{name}">\n'
        f'\t\t\t\t\t<apply>true</apply>\n'
        f'\t\t\t\t\t<weight>1000</weight>\n'
        f'\t\t\t\t\t<value_type>default_value</value_type>\n'
        f'\t\t\t\t\t<value>0</value>\n'
        f'\t\t\t\t</IKCoordinateTask>\n'
    )


def _ik_task_set_xml(weights: dict[str, float], locked: list[str]) -> str:
    tasks = "".join(_marker_task_xml(n, w) for n, w in weights.items())
    tasks += "".join(_coordinate_task_xml(c) for c in locked)
    return (
        '\t\t<IKTaskSet name="hh19_IK">\n'
        '\t\t\t<objects>\n' + tasks + '\t\t\t</objects>\n'
        '\t\t\t<groups/>\n'
        '\t\t</IKTaskSet>\n'
    )


def _measurement_xml(name: str, pairs: list[tuple[str, str]], bodies: list[str]) -> str:
    pair_xml = ""
    for a, b in pairs:
        pair_xml += (
            '\t\t\t\t\t<MarkerPair name="">\n'
            f'\t\t\t\t\t\t<markers> {a} {b} </markers>\n'
            '\t\t\t\t\t</MarkerPair>\n'
        )
    body_xml = "".join(
        f'\t\t\t\t\t<BodyScale name="{bn}">\n\t\t\t\t\t\t<axes> X Y Z </axes>\n\t\t\t\t\t</BodyScale>\n'
        for bn in bodies
    )
    return (
        f'\t\t\t\t<Measurement name="{name}">\n'
        '\t\t\t\t\t<apply>true</apply>\n'
        '\t\t\t\t\t<MarkerPairSet name="">\n\t\t\t\t\t\t<objects>\n'
        + pair_xml +
        '\t\t\t\t\t\t</objects>\n\t\t\t\t\t\t<groups/>\n\t\t\t\t\t</MarkerPairSet>\n'
        '\t\t\t\t\t<BodyScaleSet name="">\n\t\t\t\t\t\t<objects>\n'
        + body_xml +
        '\t\t\t\t\t\t</objects>\n\t\t\t\t\t\t<groups/>\n\t\t\t\t\t</BodyScaleSet>\n'
        '\t\t\t\t</Measurement>\n'
    )


def _scale_setup_xml(
    model_file: str,
    static_trc: str,
    static_markers: list[str],
    scaled_only: str,
    out_model: str,
    out_motion: str,
    mass_kg: float,
    t0: float,
    t1: float,
) -> str:
    measurements = "".join(
        _measurement_xml(name, pairs, bodies)
        for name, (pairs, bodies) in SCALE_MEASUREMENTS.items()
    )
    # MarkerPlacer 内部 IK：关节中心类 marker 高权重，thigh/shank 低权重；只建静态 TRC 里有的
    placer_weights = {
        n: (1000.0 if "Thigh" not in n and "Shank" not in n else 1.0)
        for n in HH19_MARKERS
        if n in static_markers
    }
    placer_tasks = _ik_task_set_xml(placer_weights, LOCKED_COORDINATES + _LOCKED_EXTRA)

    # ModelScaler 与 MarkerPlacer 必须写**不同**文件（官方约定）：
    #   ModelScaler -> *_scaledOnly.osim（只缩放段长/质量）
    #   MarkerPlacer -> *_scaled.osim（读 scaledOnly，再把 marker 精定位后写出）
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40000">\n'
        '\t<ScaleTool name="hh19_scale">\n'
        f'\t\t<mass> {mass_kg} </mass>\n'
        '\t\t<GenericModelMaker name="">\n'
        f'\t\t\t<model_file> {model_file} </model_file>\n'
        '\t\t</GenericModelMaker>\n'
        '\t\t<ModelScaler name="">\n'
        '\t\t\t<apply>true</apply>\n'
        '\t\t\t<scaling_order> measurements </scaling_order>\n'
        '\t\t\t<MeasurementSet name="hh19">\n'
        '\t\t\t\t<objects>\n' + measurements + '\t\t\t\t</objects>\n'
        '\t\t\t\t<groups/>\n'
        '\t\t\t</MeasurementSet>\n'
        f'\t\t\t<marker_file> {static_trc} </marker_file>\n'
        f'\t\t\t<time_range> {t0} {t1} </time_range>\n'
        '\t\t\t<preserve_mass_distribution> true </preserve_mass_distribution>\n'
        f'\t\t\t<output_model_file> {scaled_only} </output_model_file>\n'
        '\t\t\t<output_scale_file> Unassigned </output_scale_file>\n'
        '\t\t</ModelScaler>\n'
        '\t\t<MarkerPlacer name="">\n'
        # MarkerPlacer 关闭：其内部 model write 在 Windows 上抛 Xml::writeToFile
        # 空错误（ModelScaler 同目录写却正常）。而默认 HH19 偏移在 ModelScaler 缩放后
        # 与静态 trial 已吻合到 RMS≈0.05mm，精定位收益可忽略，直接用 scaledOnly 跑 IK/ID。
        '\t\t\t<apply>false</apply>\n'
        + placer_tasks +
        f'\t\t\t<marker_file> {static_trc} </marker_file>\n'
        '\t\t\t<coordinate_file> </coordinate_file>\n'
        f'\t\t\t<time_range> {t0} {t1} </time_range>\n'
        f'\t\t\t<output_motion_file> {out_motion} </output_motion_file>\n'
        f'\t\t\t<output_model_file> {out_model} </output_model_file>\n'
        '\t\t\t<output_marker_file> Unassigned </output_marker_file>\n'
        '\t\t\t<max_marker_movement> -1 </max_marker_movement>\n'
        '\t\t</MarkerPlacer>\n'
        '\t</ScaleTool>\n'
        '</OpenSimDocument>\n'
    )


def _ik_setup_xml(
    model_file: str,
    dynamic_trc: str,
    dynamic_markers: list[str],
    out_mot: str,
    t0: float,
    t1: float,
) -> str:
    # 主 IK：只用动态 TRC 里实际存在的 marker
    weights = {n: w for n, w in IK_MARKER_WEIGHTS.items() if n in dynamic_markers}
    tasks = _ik_task_set_xml(weights, LOCKED_COORDINATES + _LOCKED_EXTRA)
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40000">\n'
        '\t<InverseKinematicsTool name="hh19_ik">\n'
        '\t\t<results_directory> ./ </results_directory>\n'
        f'\t\t<model_file> {model_file} </model_file>\n'
        '\t\t<constraint_weight> 20 </constraint_weight>\n'
        '\t\t<accuracy> 1e-05 </accuracy>\n'
        + tasks +
        f'\t\t<marker_file> {dynamic_trc} </marker_file>\n'
        '\t\t<coordinate_file> </coordinate_file>\n'
        f'\t\t<time_range> {t0} {t1} </time_range>\n'
        '\t\t<report_errors> true </report_errors>\n'
        f'\t\t<output_motion_file> {out_mot} </output_motion_file>\n'
        '\t</InverseKinematicsTool>\n'
        '</OpenSimDocument>\n'
    )


def _id_setup_xml(
    model_file: str,
    ik_mot: str,
    external_loads: str,
    out_mot: str,
    t0: float,
    t1: float,
) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40000">\n'
        '\t<InverseDynamicsTool name="hh19_id">\n'
        '\t\t<results_directory> ./ </results_directory>\n'
        f'\t\t<model_file> {model_file} </model_file>\n'
        f'\t\t<time_range> {t0} {t1} </time_range>\n'
        '\t\t<forces_to_exclude> Muscles </forces_to_exclude>\n'
        f'\t\t<external_loads_file> {external_loads} </external_loads_file>\n'
        f'\t\t<coordinates_file> {ik_mot} </coordinates_file>\n'
        '\t\t<lowpass_cutoff_frequency_for_coordinates> 6 </lowpass_cutoff_frequency_for_coordinates>\n'
        f'\t\t<output_gen_force_file> {out_mot} </output_gen_force_file>\n'
        '\t</InverseDynamicsTool>\n'
        '</OpenSimDocument>\n'
    )


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def run_scale_ik_id(
    generic_model: str | Path,
    static_trc: str | Path,
    dynamic_trc: str | Path,
    external_loads: str | Path,
    out_dir: str | Path,
    *,
    mass_kg: float = 80.0,
    static_time: tuple[float, float] | None = None,
    dyn_time: tuple[float, float] | None = None,
) -> dict:
    # 关键约束：OpenSim 的 C++ 层打不开含中文的**绝对**路径（与 ezc3d 同款坑），
    # 但**相对 ASCII 路径**可以。而 setup 文件里的相对路径是**相对于 setup 文件所在目录**
    # 解析的。因此：所有 OpenSim 要消费的文件统一写进 `out` 目录，setup XML 里只用
    # **裸文件名**引用；Python 侧读文件则用完整相对路径（相对 cwd）。
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generic_model = str(Path(generic_model))
    static_trc = str(Path(static_trc))
    dynamic_trc = str(Path(dynamic_trc))
    external_loads = str(Path(external_loads))

    # 落盘文件名（setup XML 内以裸名引用，相对 out 目录）
    markers_model = out / "hh19_markers.osim"
    scaled_only = out / "hh19_scaledOnly.osim"
    scaled_model = out / "hh19_scaled.osim"
    static_mot = out / "hh19_static_pose.mot"
    ik_mot = out / "hh19_ik.mot"
    id_mot = out / "hh19_id.mot"

    static_markers, st_range = _trc_info(static_trc)
    dyn_markers, dyn_range = _trc_info(dynamic_trc)
    if static_time is None:
        static_time = st_range
    if dyn_time is None:
        dyn_time = dyn_range

    # 1. 加 marker → markers 模型（写进 out，供 setup 以裸名引用）
    #    此刻 cwd 仍是 opensim_pipeline，generic_model 是相对 cwd 的 ASCII 路径；
    #    含中文的绝对路径 OpenSim C++ 层打不开，必须在这里就加载通用模型。
    model = osim.Model(generic_model)
    add_hh19_markers(model)
    model.printToXML(str(markers_model))

    # 关键：不同 Tool 解析相对路径的基准不一致——ScaleTool 相对 setup 文件目录，
    # IK/ID 相对 cwd。把 cwd 切到 out 目录后二者基准重合，裸文件名对所有 Tool 唯一命中。
    _orig_cwd = os.getcwd()
    os.chdir(out)

    try:
        # 2. Scale（ModelScaler 用静态 TRC；MarkerPlacer 关闭，见 _scale_setup_xml）
        with open("scale_setup.xml", "w", encoding="utf-8") as f:
            f.write(_scale_setup_xml(
                markers_model.name, Path(static_trc).name, static_markers,
                scaled_only.name, scaled_model.name, static_mot.name, mass_kg,
                static_time[0], static_time[1]))
        osim.ScaleTool("scale_setup.xml").run()

        # 3. IK（动态 TRC，scaledOnly 模型）
        with open("ik_setup.xml", "w", encoding="utf-8") as f:
            f.write(_ik_setup_xml(
                scaled_only.name, Path(dynamic_trc).name, dyn_markers,
                ik_mot.name, dyn_time[0], dyn_time[1]))
        osim.InverseKinematicsTool("ik_setup.xml").run()

        # 4. ID
        with open("id_setup.xml", "w", encoding="utf-8") as f:
            f.write(_id_setup_xml(
                scaled_only.name, ik_mot.name, Path(external_loads).name,
                id_mot.name, dyn_time[0], dyn_time[1]))
        osim.InverseDynamicsTool("id_setup.xml").run()
    finally:
        os.chdir(_orig_cwd)

    return {
        "markers_model": str(markers_model),
        "scaled_model": str(scaled_only),
        "ik_mot": str(ik_mot),
        "id_mot": str(id_mot),
        "static_markers": static_markers,
        "dynamic_markers": dyn_markers,
    }


__all__ = ["add_hh19_markers", "run_scale_ik_id"]
