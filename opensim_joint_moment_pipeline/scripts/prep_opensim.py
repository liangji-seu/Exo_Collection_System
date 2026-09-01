"""OpenSim 预处理（**EXO 环境**：numpy 1.x + ezc3d）。

把 C3D 一次性转成 OpenSim 下游可直接消费的中间产物，并写 ``manifest.json``：

    static.trc           静态 trial（19 marker，OpenSim ground 帧，mm）
    dynamic.trc          动态 trial（15 marker，medial 已剔除，OpenSim ground 帧，mm）
    grf.mot              地面反力（单支撑分配，OpenSim ground 帧，SI）
    external_loads.xml   ExternalLoads（左右 calcn）
    manifest.json        供 opensim 环境 ``scripts/run_opensim.py`` 读取的路径/质量/时间窗

用法：:

    python scripts/prep_opensim.py --config configs/subject_001.yaml \
        --out outputs/subject_001/walk_level/opensim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.pipeline import load_config  # noqa: E402
from pipeline.c3d.reader import read_c3d  # noqa: E402
from pipeline.gait.detect_contact import detect_contacts  # noqa: E402
from pipeline.opensim_io.build_trc import build_trc  # noqa: E402
from pipeline.opensim_io.grf import build_grf  # noqa: E402
from pipeline.transforms import R_MOCAP_TO_OPENSIM  # noqa: E402


def _fp_to_mocap(config: dict) -> np.ndarray:
    m = config["transforms"]["forceplate_to_mocap"]["rotation_matrix"]
    return np.asarray(m, dtype=np.float64)


def _trim_boolean_runs(mask: np.ndarray, trim_frames: int) -> np.ndarray:
    """Remove ``trim_frames`` from both ends of every true run."""
    mask = np.asarray(mask, dtype=bool)
    if trim_frames <= 0:
        return mask.copy()
    out = np.zeros_like(mask)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if end - start > 2 * trim_frames:
            out[start + trim_frames:end - trim_frames] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenSim 预处理（EXO 环境）")
    ap.add_argument("--config", default="configs/subject_001.yaml")
    ap.add_argument("--out", default="outputs/subject_001/walk_level/opensim")
    args = ap.parse_args()

    config = load_config(args.config)
    files = config["files"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    static_path = Path(files["static_c3d"])
    dynamic_path = Path(files["dynamic_c3d"])
    generic_model = Path(files["generic_model"])
    mass_kg = float(config["subject"]["mass_kg"])
    R_fp = _fp_to_mocap(config)
    filtering = config.get("filtering", {})
    filtering_enabled = bool(filtering.get("enabled", False))
    marker_cutoff = float(filtering["marker_cutoff_hz"]) if filtering_enabled else None
    grf_cutoff = float(filtering["grf_cutoff_hz"]) if filtering_enabled else None
    force_advance_ms = float(config.get("synchronization", {}).get("force_advance_ms", 0.0))
    force_direction = config.get("force_direction_correction", {})
    force_x_sign = float(force_direction.get("opensim_x_sign", 1.0))
    force_z_sign = float(force_direction.get("opensim_z_sign", 1.0))

    static_data = read_c3d(static_path)
    dynamic_data = read_c3d(dynamic_path)

    # 1. 接触检测（动态，供 GRF 单支撑分配）
    ss_cfg = config.get("single_support", {})
    contact = detect_contacts(
        dynamic_data,
        vertical_axis=None,
        force_threshold_N=float(ss_cfg.get("vertical_force_threshold_N", 50.0)),
        foot_height_threshold_mm=float(ss_cfg.get("foot_height_threshold_mm", 30.0)),
    )

    # 单支撑有效性掩码：单块跑台只能拿到 total 合力，双支撑期无法拆左右脚，
    # 因此 ID 结果只在单支撑期有效（支撑脚 = 该脚）。保存供 run_opensim 摘要时过滤。
    right_single = contact.right_contact & ~contact.left_contact
    left_single = contact.left_contact & ~contact.right_contact
    trim_boundary_ms = float(ss_cfg.get("trim_boundary_ms", 20.0))
    trim_frames = int(round(trim_boundary_ms / 1000.0 * dynamic_data.point_rate_hz))
    right_valid = _trim_boolean_runs(right_single, trim_frames)
    left_valid = _trim_boolean_runs(left_single, trim_frames)
    np.save(out / "support_mask.npy", np.column_stack([right_valid, left_valid]))

    # 2. 静态 / 动态 TRC（OpenSim ground 帧）
    static_names, _ = build_trc(
        static_data, out / "static.trc", opensim_frame=True, cutoff_hz=marker_cutoff)
    dyn_names, _ = build_trc(
        dynamic_data, out / "dynamic.trc", opensim_frame=True, cutoff_hz=marker_cutoff)

    # 3. GRF + ExternalLoads
    grf = build_grf(
        dynamic_data,
        contact.right_contact,
        contact.left_contact,
        R_fp,
        out,
        mot_name="grf.mot",
        include_free_moment=False,
        cutoff_hz=grf_cutoff,
        advance_ms=force_advance_ms,
        force_x_sign=force_x_sign,
        force_z_sign=force_z_sign,
    )

    # 4. manifest（供 opensim 环境）
    manifest = {
        "subject": {
            "id": config["subject"]["id"],
            "mass_kg": mass_kg,
        },
        "generic_model": str(generic_model),
        "static_trc": str(out / "static.trc"),
        "dynamic_trc": str(out / "dynamic.trc"),
        "external_loads": grf["xml"],
        "grf_mot": grf["mot"],
        "support_mask": str(out / "support_mask.npy"),
        "analysis_time_range_s": config.get("analysis", {}).get("time_range_s"),
        "processing": {
            "marker_cutoff_hz": marker_cutoff,
            "grf_cutoff_hz": grf_cutoff,
            "force_advance_ms": force_advance_ms,
            "opensim_force_x_sign": force_x_sign,
            "opensim_force_z_sign": force_z_sign,
            "support_trim_boundary_ms": trim_boundary_ms,
        },
        "out_dir": str(out),
        "mocap_to_opensim_R": R_MOCAP_TO_OPENSIM.tolist(),
        "contact": {
            "vertical_axis": contact.vertical_axis,
            "ground_z_mm": contact.ground_z_mm,
            "n_right_contact": int(contact.right_contact.sum()),
            "n_left_contact": int(contact.left_contact.sum()),
            "n_right_single": int((contact.right_contact & ~contact.left_contact).sum()),
            "n_left_single": int((contact.left_contact & ~contact.right_contact).sum()),
            "n_right_valid_trimmed": int(right_valid.sum()),
            "n_left_valid_trimmed": int(left_valid.sum()),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 52)
    print("OpenSim preprocessing done")
    print("=" * 52)
    print(f"[static TRC]  {len(static_names)} markers -> {manifest['static_trc']}")
    print(f"[dynamic TRC] {len(dyn_names)} markers -> {manifest['dynamic_trc']}")
    print(f"[GRF]         {manifest['grf_mot']}")
    print(f"[loads]       {manifest['external_loads']}")
    print(f"[manifest]    {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
