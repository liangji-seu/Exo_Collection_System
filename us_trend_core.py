"""Conservative multi-scale ultrasound displacement trend estimation.

The module deliberately keeps the motion result separate from the screening
mask.  A displacement can be numerically estimated in a weak echo region, but
it is not presented as reliable evidence unless the local and bidirectional
checks pass.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d


N_SAMPLES = 957
DEPTH = np.arange(20, 977, dtype=np.float64)


DEFAULT_CONFIG = {
    "n_samples": N_SAMPLES,
    "support_start": 128,
    "support_stop": 829,
    "depth_start_adc": 20,
    "window_sizes": (49, 99, 197),
    "window_weights": (0.2, 0.5, 0.3),
    "shift_min": -24,
    "shift_max": 24,
    "node_step": 8,
    "node_delta_max": 2,
    "second_order_penalty": 0.16,
    "slope_penalty": 0.012,
    "subpixel_limit": 0.4,
    "depth_smoothing_sigma": 2.0,
    "spacing_window": 97,
    "texture_threshold": 0.15,
    "ncc_threshold": 0.70,
    "reverse_ncc_threshold": 0.70,
    "cycle_threshold": 2.0,
    "peak_margin_threshold": 0.025,
    "near_zero_direction": 1.0,
    "unchanged_spacing": 0.010,
    "min_region_length": 64,
    "causal_tau": 0.15,
}


def _as_profile(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (N_SAMPLES,):
        raise ValueError(f"{name} 必须是 957 点一维轮廓")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须全部为有限值")
    return array


def _local_zncc(current: np.ndarray, past: np.ndarray, center: int, shift: int, width: int) -> float:
    half = width // 2
    x = current[center - half : center + half + 1]
    y_center = center + shift
    y = past[y_center - half : y_center + half + 1]
    if x.size != width or y.size != width:
        return -1.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


def _score_nodes(current: np.ndarray, past: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    widths = DEFAULT_CONFIG["window_sizes"]
    weights = DEFAULT_CONFIG["window_weights"]
    shifts = np.arange(DEFAULT_CONFIG["shift_min"], DEFAULT_CONFIG["shift_max"] + 1)
    node_grid = nodes[:, None, None]
    shift_grid = shifts[None, :, None]
    result = np.zeros((len(nodes), len(shifts)), dtype=np.float64)
    for width, weight in zip(widths, weights):
        offsets = np.arange(-(width // 2), width // 2 + 1, dtype=np.int64)[None, None, :]
        x = current[node_grid + offsets]
        y = past[node_grid + shift_grid + offsets]
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
        numerator = np.sum(x * y, axis=-1)
        denominator = np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))
        result += weight * np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    return np.clip(result, -1.0, 1.0)


def _second_order_path(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select a smooth path with |first difference| <= 2.

    The state is (previous shift, current shift), so the transition cost can
    penalize the second difference directly.  The small first difference cost
    breaks ties in favour of a locally flat trend without suppressing a
    gradual compression or expansion.
    """
    shifts = np.arange(DEFAULT_CONFIG["shift_min"], DEFAULT_CONFIG["shift_max"] + 1, dtype=np.int64)
    n_nodes, n_shifts = scores.shape
    if n_nodes == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    unary = -np.nan_to_num(scores, nan=-1.0)
    if n_nodes == 1:
        choice = int(np.argmin(unary[0]))
        return np.array([float(shifts[choice])]), np.array([float(scores[0, choice])])

    inf = np.inf
    pair_cost = np.full((n_shifts, n_shifts), inf, dtype=np.float64)
    for p in range(n_shifts):
        for c in range(n_shifts):
            delta = shifts[c] - shifts[p]
            if abs(delta) <= DEFAULT_CONFIG["node_delta_max"]:
                pair_cost[p, c] = (
                    unary[0, p]
                    + unary[1, c]
                    + DEFAULT_CONFIG["slope_penalty"] * float(delta * delta)
                )
    # Back pointers contain the shift index at the preceding node.
    back = np.full((n_nodes, n_shifts, n_shifts), -1, dtype=np.int16)
    costs = pair_cost
    for k in range(2, n_nodes):
        next_cost = np.full_like(costs, inf)
        for current in range(n_shifts):
            valid_previous = np.abs(shifts - shifts[current]) <= DEFAULT_CONFIG["node_delta_max"]
            previous_cost = costs[:, current]
            if not np.isfinite(previous_cost[valid_previous]).any():
                continue
            for nxt in range(max(0, current - 2), min(n_shifts, current + 3)):
                curvature = shifts[nxt] - 2 * shifts[current] + shifts
                candidates = (
                    previous_cost
                    + unary[k, nxt]
                    + DEFAULT_CONFIG["slope_penalty"] * float((shifts[nxt] - shifts[current]) ** 2)
                    + DEFAULT_CONFIG["second_order_penalty"] * curvature * curvature
                )
                candidates[~valid_previous] = inf
                previous = int(np.argmin(candidates))
                if np.isfinite(candidates[previous]):
                    next_cost[current, nxt] = candidates[previous]
                    back[k, current, nxt] = previous
        costs = next_cost
    previous, current = np.unravel_index(np.argmin(costs), costs.shape)
    path = np.empty(n_nodes, dtype=np.float64)
    path[-2:] = shifts[[previous, current]]
    for k in range(n_nodes - 1, 1, -1):
        previous_previous = int(back[k, previous, current])
        if previous_previous < 0:
            # This only occurs for a pathological all-invalid score matrix.
            previous_previous = previous
        path[k - 2] = shifts[previous_previous]
        current, previous = previous, previous_previous
    chosen = np.rint(path).astype(int) - DEFAULT_CONFIG["shift_min"]
    chosen = np.clip(chosen, 0, n_shifts - 1)
    return path, scores[np.arange(n_nodes), chosen]


def _subpixel_refine(scores: np.ndarray, path: np.ndarray) -> np.ndarray:
    shifts = np.arange(DEFAULT_CONFIG["shift_min"], DEFAULT_CONFIG["shift_max"] + 1)
    refined = path.astype(np.float64).copy()
    limit = DEFAULT_CONFIG["subpixel_limit"]
    for i, value in enumerate(path):
        j = int(np.rint(value) - shifts[0])
        if not 1 <= j < len(shifts) - 1:
            continue
        y0, y1, y2 = scores[i, j - 1], scores[i, j], scores[i, j + 1]
        denominator = y0 - 2.0 * y1 + y2
        if denominator < -1e-10 and y1 >= y0 and y1 >= y2:
            offset = 0.5 * (y0 - y2) / denominator
            refined[i] = value + float(np.clip(offset, -limit, limit))
    return refined


def _pava_non_decreasing(values: np.ndarray) -> np.ndarray:
    means: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        means.append(float(value))
        weights.append(1.0)
        starts.append(index)
        ends.append(index)
        while len(means) >= 2 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            means[-2] = (weights[-2] * means[-2] + weights[-1] * means[-1]) / total
            weights[-2] = total
            ends[-2] = ends[-1]
            means.pop()
            weights.pop()
            starts.pop()
            ends.pop()
    output = np.empty(len(values), dtype=np.float64)
    for mean, start, end in zip(means, starts, ends):
        output[start : end + 1] = mean
    return output


def _monotonic_depth_projection(displacement: np.ndarray) -> np.ndarray:
    coordinate = np.arange(len(displacement), dtype=np.float64) + displacement
    epsilon = 1e-4
    adjusted = _pava_non_decreasing(coordinate - epsilon * np.arange(len(coordinate)))
    return adjusted + epsilon * np.arange(len(coordinate)) - np.arange(len(coordinate))


def _interpolate_path(nodes: np.ndarray, path: np.ndarray) -> np.ndarray:
    full = np.interp(np.arange(N_SAMPLES), nodes, path)
    # A short Gaussian pass removes node quantisation while preserving the
    # broad trend.  The projection prevents a crossing of z+v.
    full = gaussian_filter1d(full, DEFAULT_CONFIG["depth_smoothing_sigma"], mode="nearest")
    return _monotonic_depth_projection(full)


def _nearest_node_values(nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(np.arange(N_SAMPLES), nodes, values)


def _final_warp_ncc(current: np.ndarray, past: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Old-baseline-compatible 49-point ZNCC after a full-field warp."""
    start, stop = DEFAULT_CONFIG["support_start"], DEFAULT_CONFIG["support_stop"]
    centers = np.arange(start, stop, dtype=np.int64)
    result = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    z = np.arange(N_SAMPLES, dtype=np.float64)
    warped = np.interp(z + displacement, z, past, left=np.nan, right=np.nan)
    half = 24
    offsets = np.arange(-half, half + 1, dtype=np.int64)[None, :]
    x = current[centers[:, None] + offsets]
    y = warped[centers[:, None] + offsets]
    x = x - np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    values = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-12)
    result[start:stop] = np.clip(values, -1.0, 1.0)
    return result


def _local_texture(profile: np.ndarray) -> np.ndarray:
    texture = np.zeros(N_SAMPLES, dtype=np.float64)
    for index in range(DEFAULT_CONFIG["support_start"], DEFAULT_CONFIG["support_stop"]):
        half = 24
        patch = profile[index - half : index + half + 1]
        texture[index] = np.std(patch) / (np.std(profile) + 1e-12)
    return texture


def _point_mapping_derivative(backward: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    derivative = np.gradient(np.asarray(backward, dtype=np.float64))
    finite = np.isfinite(backward) & np.isfinite(derivative)
    result[finite] = 1.0 + derivative[finite]
    if valid is not None:
        result[~np.asarray(valid, dtype=bool)] = np.nan
    return result


def _relative_spacing(backward: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    half = DEFAULT_CONFIG["spacing_window"] // 2
    # Fixed 97-point local least-squares slope; it suppresses integer-node
    # staircasing while keeping the definition in the current coordinate.
    centers = np.arange(DEFAULT_CONFIG["support_start"] + half,
                        DEFAULT_CONFIG["support_stop"] - half)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(backward, dtype=np.float64), 2 * half + 1)
    selected = windows[centers - half]
    valid_windows = np.isfinite(selected).all(axis=1)
    centered = selected - np.mean(selected, axis=1, keepdims=True)
    denominator = float(np.sum(offsets * offsets))
    slope = np.sum(centered * offsets[None, :], axis=1) / denominator
    mapping = 1.0 + slope
    good = valid_windows & (mapping > 1e-6)
    result[centers[good]] = 1.0 / mapping[good] - 1.0
    if valid is not None:
        result[~valid] = np.nan
    return result


def _classify_segment(direction: np.ndarray, spacing: np.ndarray, start: int, stop: int) -> tuple[str, str]:
    d = direction[start:stop]
    s = spacing[start:stop]
    d = d[np.isfinite(d)]
    s = s[np.isfinite(s)]
    if not len(d):
        return "unknown", "unknown"
    median_d = float(np.median(d))
    median_s = float(np.median(s)) if len(s) else np.nan
    direction_label = (
        "left" if median_d <= -DEFAULT_CONFIG["near_zero_direction"]
        else "right" if median_d >= DEFAULT_CONFIG["near_zero_direction"]
        else "near_zero"
    )
    if not np.isfinite(median_s) or abs(median_s) < DEFAULT_CONFIG["unchanged_spacing"]:
        deformation = "unchanged"
    elif median_s < 0:
        deformation = "compress"
    else:
        deformation = "expand"
    return direction_label, deformation


def _build_regions(direction: np.ndarray, spacing: np.ndarray, screen: np.ndarray) -> list[dict[str, Any]]:
    start0, stop0 = DEFAULT_CONFIG["support_start"], DEFAULT_CONFIG["support_stop"]
    # Smooth the classification features before binning.  This allows a
    # genuine left/right or compress/expand change to form separate regions,
    # while preventing one noisy sample from creating a new interval.
    mask = np.asarray(screen[start0:stop0], dtype=bool)
    smooth_d = gaussian_filter1d(np.nan_to_num(direction, nan=0.0), 8.0, mode="nearest")
    smooth_s = gaussian_filter1d(np.nan_to_num(spacing, nan=0.0), DEFAULT_CONFIG["spacing_window"] / 2.0, mode="nearest")
    categories: list[tuple[str, str]] = []
    for index in range(start0, stop0):
        d = smooth_d[index]
        s = smooth_s[index]
        label = "left" if d <= -DEFAULT_CONFIG["near_zero_direction"] else "right" if d >= DEFAULT_CONFIG["near_zero_direction"] else "near_zero"
        deformation = "unchanged" if abs(s) < DEFAULT_CONFIG["unchanged_spacing"] else "compress" if s < 0 else "expand"
        categories.append((label, deformation))
    runs: list[tuple[int, int, tuple[str, str]]] = []
    first = 0
    for index in range(1, len(categories) + 1):
        if index == len(categories) or categories[index] != categories[first]:
            start, stop = start0 + first, start0 + index
            runs.append((start, stop, categories[first]))
            first = index
    # Runs shorter than the minimum effective interval are treated as unknown.
    # A trend region may contain isolated unscreened points, but it must have
    # substantial screened coverage; the mask itself is never filled.
    merged: list[tuple[int, int, tuple[str, str]]] = []
    for start, stop, category in runs:
        coverage = float(np.mean(screen[start:stop]))
        if stop - start < DEFAULT_CONFIG["min_region_length"] or coverage < 0.60:
            category = ("unknown", "unknown")
        if merged and category == merged[-1][2]:
            merged[-1] = (merged[-1][0], stop, category)
        else:
            merged.append((start, stop, category))
    regions: list[dict[str, Any]] = []
    for start, stop, (label, deformation) in merged:
        if stop - start < DEFAULT_CONFIG["min_region_length"]:
            continue
        values_d = direction[start:stop][screen[start:stop]]
        values_s = spacing[start:stop][screen[start:stop]]
        regions.append({
            "start_index": int(start),
            "stop_index": int(stop),
            "depth_start": int(DEPTH[start]),
            "depth_stop": int(DEPTH[stop - 1]),
            "label": label,
            "deformation": deformation,
            "median_direction": float(np.median(values_d)) if len(values_d) else float("nan"),
            "median_spacing": float(np.median(values_s)) if len(values_s) else float("nan"),
        })
    return regions


def _path_once(current: np.ndarray, past: np.ndarray) -> dict[str, np.ndarray]:
    start, stop = DEFAULT_CONFIG["support_start"], DEFAULT_CONFIG["support_stop"]
    nodes = np.arange(start, stop, DEFAULT_CONFIG["node_step"], dtype=int)
    scores = _score_nodes(current, past, nodes)
    path, chosen_scores = _second_order_path(scores)
    path = _subpixel_refine(scores, path)
    backward = _interpolate_path(nodes, path)
    selected = np.clip(np.rint(path).astype(int) - DEFAULT_CONFIG["shift_min"], 0, scores.shape[1] - 1)
    margin = np.empty(len(nodes), dtype=np.float64)
    for row, col in enumerate(selected):
        other = np.delete(scores[row], np.arange(max(0, col - 2), min(scores.shape[1], col + 3)))
        margin[row] = scores[row, col] - np.max(other) if len(other) else 0.0
    return {
        "nodes": nodes,
        "backward": backward,
        "node_score": _nearest_node_values(nodes, chosen_scores),
        "final_score": _final_warp_ncc(current, past, backward),
        "peak_margin": _nearest_node_values(nodes, margin),
    }


def _copy_result(result: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in result.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
    return copied


def estimate_trend_pair(current: Any, past: Any) -> dict[str, Any]:
    """Estimate one current/past pair on the conservative 128:829 support."""
    started = time.perf_counter()
    current = _as_profile(current, "current")
    past = _as_profile(past, "past")
    forward = _path_once(current, past)
    reverse = _path_once(past, current)
    backward = forward["backward"]
    direction = -backward
    reverse_at_mapped = np.interp(np.arange(N_SAMPLES) + backward, np.arange(N_SAMPLES), reverse["backward"], left=np.nan, right=np.nan)
    cycle = backward + reverse_at_mapped
    match_ncc = forward["final_score"]
    reverse_ncc = np.interp(
        np.arange(N_SAMPLES) + backward,
        np.arange(N_SAMPLES),
        reverse["final_score"],
        left=np.nan,
        right=np.nan,
    )
    texture = _local_texture(current)
    support = np.zeros(N_SAMPLES, dtype=bool)
    support[DEFAULT_CONFIG["support_start"] : DEFAULT_CONFIG["support_stop"]] = True
    screen = (
        support
        & (texture >= DEFAULT_CONFIG["texture_threshold"])
        & (match_ncc >= DEFAULT_CONFIG["ncc_threshold"])
        & (reverse_ncc >= DEFAULT_CONFIG["reverse_ncc_threshold"])
        & (np.abs(cycle) <= DEFAULT_CONFIG["cycle_threshold"])
        & (forward["peak_margin"] >= DEFAULT_CONFIG["peak_margin_threshold"])
    )
    spacing = _relative_spacing(backward, support)
    mapping_derivative = _point_mapping_derivative(backward, support)
    result = {
        "direction": direction.astype(np.float32),
        "backward": backward.astype(np.float32),
        "relative_spacing": spacing.astype(np.float32),
        "point_mapping_derivative": mapping_derivative.astype(np.float32),
        "screen_pass": screen,
        "support": support,
        "regions": _build_regions(direction, spacing, screen),
        "compute_ms": float((time.perf_counter() - started) * 1000.0),
        "match_ncc": match_ncc.astype(np.float32),
        "reverse_match_ncc": reverse_ncc.astype(np.float32),
        "cycle_residual": cycle.astype(np.float32),
        "texture_ratio": texture.astype(np.float32),
        "peak_margin": forward["peak_margin"].astype(np.float32),
        "current": current.astype(np.float32),
        "past": past.astype(np.float32),
        "depth": DEPTH.copy(),
    }
    return result


@dataclass
class CausalTrend:
    """Causal temporal smoothing of displacement fields.

    One scalar alpha is applied to the full supported displacement field.
    Current-frame screening is retained and, when profiles are available,
    repeated after the final 49-point warp NCC.  Depth spacing is always
    recomputed from the filtered field.
    """

    tau: float = DEFAULT_CONFIG["causal_tau"]
    _last_time: float | None = None
    _last_result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau 必须为正数")

    def reset(self) -> None:
        self._last_time = None
        self._last_result = None

    @staticmethod
    def _resolve_pair(pair: Any) -> dict[str, Any]:
        if isinstance(pair, dict) and ("backward" in pair or "direction" in pair):
            result = _copy_result(pair)
            if "backward" not in result:
                result["backward"] = -np.asarray(result["direction"], dtype=np.float32)
            if "direction" not in result:
                result["direction"] = -np.asarray(result["backward"], dtype=np.float32)
            if "support" not in result:
                result["support"] = np.zeros(N_SAMPLES, dtype=bool)
                result["support"][DEFAULT_CONFIG["support_start"] : DEFAULT_CONFIG["support_stop"]] = True
            if "screen_pass" not in result:
                result["screen_pass"] = np.asarray(result["support"], dtype=bool).copy()
            if "relative_spacing" not in result:
                result["relative_spacing"] = _relative_spacing(
                    np.asarray(result["backward"], dtype=np.float64), np.asarray(result["support"], dtype=bool)
                ).astype(np.float32)
            if "point_mapping_derivative" not in result:
                result["point_mapping_derivative"] = _point_mapping_derivative(
                    np.asarray(result["backward"], dtype=np.float64), np.asarray(result["support"], dtype=bool)
                ).astype(np.float32)
            if "regions" not in result:
                result["regions"] = _build_regions(
                    np.asarray(result["direction"], dtype=np.float64),
                    np.asarray(result["relative_spacing"], dtype=np.float64),
                    np.asarray(result["screen_pass"], dtype=bool),
                )
            return result
        if isinstance(pair, dict) and "current" in pair and "past" in pair:
            return estimate_trend_pair(pair["current"], pair["past"])
        if isinstance(pair, (tuple, list)) and len(pair) == 2:
            return estimate_trend_pair(pair[0], pair[1])
        raise TypeError("pair 必须是 (current, past) 或含 current/past 的字典")

    def update(self, pair: Any, time_value: float) -> dict[str, Any]:
        now = float(time_value)
        if not np.isfinite(now):
            raise ValueError("time 必须为有限数")
        current = self._resolve_pair(pair)
        reset = self._last_time is None or now <= self._last_time or now - self._last_time > 0.075 + 1e-12
        if reset or self._last_result is None:
            filtered = _copy_result(current)
            alpha = 1.0
        else:
            dt = max(now - self._last_time, 1e-6)
            alpha = float(1.0 - np.exp(-dt / self.tau))
            previous = self._last_result
            filtered = _copy_result(current)
            old = np.asarray(previous["backward"], dtype=np.float64)
            new = np.asarray(current["backward"], dtype=np.float64)
            current_mask = np.asarray(current["screen_pass"], dtype=bool)
            # One scalar alpha is applied to the entire displacement field.
            observed = np.asarray(current["support"], dtype=bool) & np.isfinite(old) & np.isfinite(new)
            smoothed = new.copy()
            smoothed[observed] = (1.0 - alpha) * old[observed] + alpha * new[observed]
            filtered["backward"] = smoothed.astype(np.float32)
            filtered["direction"] = (-smoothed).astype(np.float32)
            filtered["screen_pass"] = current_mask.copy()
            if "current" in current and "past" in current:
                final_match = _final_warp_ncc(
                    np.asarray(current["current"], dtype=np.float64),
                    np.asarray(current["past"], dtype=np.float64),
                    smoothed,
                )
                filtered["match_ncc"] = final_match.astype(np.float32)
                filtered["screen_pass"] &= np.isfinite(final_match) & (final_match >= DEFAULT_CONFIG["ncc_threshold"])
            filtered["relative_spacing"] = _relative_spacing(smoothed, np.asarray(filtered["support"], dtype=bool)).astype(np.float32)
            filtered["point_mapping_derivative"] = _point_mapping_derivative(
                smoothed, np.asarray(filtered["support"], dtype=bool)
            ).astype(np.float32)
            filtered["regions"] = _build_regions(filtered["direction"], filtered["relative_spacing"], filtered["screen_pass"])
        filtered["screen_pass"] = np.asarray(filtered["screen_pass"], dtype=bool) & np.asarray(current["screen_pass"], dtype=bool)
        filtered["direction"] = -np.asarray(filtered["backward"], dtype=np.float32)
        filtered["relative_spacing"] = _relative_spacing(np.asarray(filtered["backward"], dtype=np.float64), np.asarray(filtered["support"], dtype=bool)).astype(np.float32)
        filtered["point_mapping_derivative"] = _point_mapping_derivative(
            np.asarray(filtered["backward"], dtype=np.float64), np.asarray(filtered["support"], dtype=bool)
        ).astype(np.float32)
        filtered["regions"] = _build_regions(filtered["direction"], filtered["relative_spacing"], filtered["screen_pass"])
        self._last_time = now
        self._last_result = _copy_result(filtered)
        output = _copy_result(filtered)
        output.update({"reset": bool(reset), "alpha": alpha, "time": now})
        return output


__all__ = ["DEFAULT_CONFIG", "CausalTrend", "estimate_trend_pair"]
