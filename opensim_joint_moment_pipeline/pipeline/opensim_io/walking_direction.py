"""Body heading and anatomical load mapping, independent of moment signs.

All geometry is in the calibrated laboratory OpenSim ground frame. A constant
proper rotation can face the model along +X; it does not change joint-coordinate
moment signs, reverse belt forces, or validate the vendor's decomposition.
"""
from __future__ import annotations

import numpy as np

HALF_TURN = np.diag([-1.0, 1.0, -1.0])
DIRECTION_POLICY_VERSION = "body_heading_anatomical_mapping_v1"


def is_backward_trial(condition_code: str) -> bool:
    return "BWALK" in condition_code.upper().split("_")


def body_heading(names, points_mm, selected=None, *, required=False):
    """Choose identity/180 degrees only, leaving ordinary forward yaw intact."""
    if selected is None:
        selected = np.ones(len(points_mm), dtype=bool)
    scores = []
    counts = []
    consistency = []
    for side in ("R", "L"):
        if any(f"{side}.{part}" not in names for part in ("Heel", "Toe")):
            break
        vector = points_mm[:, names.index(f"{side}.Toe")] - points_mm[:, names.index(f"{side}.Heel")]
        length = np.linalg.norm(vector[:, [0, 2]], axis=1)
        valid = selected & np.isfinite(vector).all(axis=1) & (length > 100)
        score = vector[valid, 0] / length[valid]
        scores.append(float(np.median(score)) if len(score) else 0.0)
        counts.append(int(valid.sum()))
        consistency.append(float(np.mean(score > 0.7 if scores[-1] > 0 else score < -0.7)) if len(score) else 0.0)
    reliable = len(scores) == 2 and min(counts) >= 30 and min(consistency) >= 0.9 and (
        min(scores) > 0.7 or max(scores) < -0.7
    )
    if required and not reliable:
        raise ValueError("倒走身体朝向无法可靠判定或窗口内包含转身：请复核双脚 Heel/Toe marker，并选取身体朝向固定的分析窗口")
    reversed_heading = reliable and max(scores) < -0.7
    rotation = HALF_TURN.copy() if reversed_heading else np.eye(3)
    return rotation, {
        "heading": "reverse" if reversed_heading else ("forward" if reliable else "undetermined"),
        "heel_to_toe_x_cosine_median": scores,
        "n_valid_frames_per_foot": counts,
        "orientation_consistency_fraction": consistency,
        "min_consistency_fraction": 0.9,
        "min_frames": 30, "min_abs_cosine": 0.7,
        "rotation_ground_to_model": rotation.tolist(),
        "yaw_normalization_deg": 180 if reversed_heading else 0,
    }


def _distance_to_foot(point, heel, toe):
    # Horizontal distance to the Heel-Toe segment (not nearest foot centroid).
    p, h, t = point[:, [0, 2]], heel[:, [0, 2]], toe[:, [0, 2]]
    segment = t - h
    denominator = np.sum(segment * segment, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.clip(np.sum((p - h) * segment, axis=1) / denominator, 0, 1)
    return np.linalg.norm(p - h - fraction[:, None] * segment, axis=1)


def anatomical_load_mapping(feet, names, points_mm, selected, *, force_threshold_N=50.0):
    """Fail closed unless BOTH native sides agree on a stable foot mapping.

    The evidence unit is one contiguous single-support bout; CI resamples bouts,
    not correlated 100 Hz frames. This only checks allocation geometry.
    """
    evidence = {}
    decisions = []
    thresholds = {"min_bouts_per_native_side": 3, "min_frames_per_bout": 5,
                  "min_frame_margin_m": 0.02, "min_bout_agreement": 0.8,
                  "min_margin_ci_lower_m": 0.02, "max_selected_distance_m": 0.15}
    contacts = [np.asarray(foot.get("contact", np.linalg.norm(foot["force"], axis=1) > force_threshold_N)) for foot in feet]
    for index, source in enumerate(("R", "L")):
        point = feet[index]["point"]
        distances = []
        for side in ("R", "L"):
            keys = [f"{side}.Heel", f"{side}.Toe"]
            if any(key not in names for key in keys):
                raise ValueError("倒走左右载荷验证缺少双脚 Heel/Toe marker")
            distances.append(_distance_to_foot(point, points_mm[:, names.index(keys[0])] / 1000,
                                                points_mm[:, names.index(keys[1])] / 1000))
        own, other = distances[index], distances[1 - index]
        single = selected & contacts[index] & ~contacts[1 - index]
        finite = np.isfinite(own) & np.isfinite(other)
        edges = np.flatnonzero(np.diff(np.r_[False, single, False]))
        margins, own_bouts, other_bouts = [], [], []
        for start, stop in zip(edges[::2], edges[1::2]):
            valid = finite[start:stop]
            if valid.sum() < thresholds["min_frames_per_bout"]:
                continue
            o, a = own[start:stop][valid], other[start:stop][valid]
            margins.append(float(np.median(a - o)))
            own_bouts.append(float(np.median(o)))
            other_bouts.append(float(np.median(a)))
        margin = np.asarray(margins)
        identity = bool(len(margin) and np.median(margin) > 0)
        signed = margin if identity else -margin
        rng = np.random.default_rng(501)
        ci = (np.percentile(np.median(rng.choice(signed, (1000, len(signed)), replace=True), axis=1), [2.5, 97.5])
              if len(signed) else np.array([np.nan, np.nan]))
        agreement = float(np.mean(signed > thresholds["min_frame_margin_m"])) if len(signed) else 0.0
        distance = float(np.median(own_bouts if identity else other_bouts)) if len(signed) else None
        accepted = (len(signed) >= thresholds["min_bouts_per_native_side"]
                    and agreement >= thresholds["min_bout_agreement"]
                    and ci[0] > thresholds["min_margin_ci_lower_m"]
                    and distance < thresholds["max_selected_distance_m"])
        evidence[source] = {
            "n_single_support_frames": int((single & finite).sum()), "n_bouts": len(margin),
            "mapping": "identity" if identity else "swap", "accepted": bool(accepted),
            "own_foot_distance_median_m": float(np.median(own_bouts)) if own_bouts else None,
            "other_foot_distance_median_m": float(np.median(other_bouts)) if other_bouts else None,
            "selected_margin_bootstrap95ci_m": ci.tolist() if len(signed) else None,
            "bout_agreement": agreement,
        }
        decisions.append(identity if accepted else None)
    audit = {"method": "single_support_cop_to_heel_toe_segment", "thresholds": thresholds,
             "evidence": evidence, "double_support_decomposition_validated": False}
    if None in decisions or decisions[0] != decisions[1]:
        raise ValueError("倒走 Gaitway 左右分力映射未通过单支撑几何验证；请核对同步/足marker/COP并选择含足够步数的稳定窗口。证据=" + str(audit))
    swap = not decisions[0]
    if swap:
        feet = [dict(feet[1], name="right"), dict(feet[0], name="left")]
    audit["native_to_anatomical"] = {"R": "left" if swap else "right", "L": "right" if swap else "left"}
    return feet, audit


def rotate_loads(feet, rotation):
    """Apply the SAME proper rotation to force, COP and free torque."""
    if np.array_equal(rotation, np.eye(3)):
        return feet
    return [dict(foot, **{key: foot[key] @ rotation.T for key in ("force", "point", "torque")}) for foot in feet]
