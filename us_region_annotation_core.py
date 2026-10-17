"""Small, dependency-light core for qualitative ultrasound region annotation.

The annotation model stores time segments and, inside each segment, depth cuts
drawn from the left time edge to the right time edge.  Depth increases towards
the bottom of the image.  Rasterization treats depth intervals as
top-inclusive/bottom-exclusive, except that the global bottom depth is
included in the last interval so the complete image rectangle is covered.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


LABELS = {
    -1: "未标注",
    0: "无明显变化",
    1: "收缩区域",
    2: "伸展区域",
    3: "不确定",
}

_SCHEMA_VERSION = 1
_SEMANTICS = "visual_qualitative_unspecified"
_SOURCE_IDENTITY_FIELDS = ("binary_path", "size_bytes", "mtime_ns")


def _bounds(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numbers") from exc
    if not math.isfinite(low) or not math.isfinite(high) or not low < high:
        raise ValueError(f"{name} must be finite and increasing")
    return low, high


def _index(index: Any, size: int, name: str = "index") -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise IndexError(f"{name} must be an integer")
    index = int(index)
    if not 0 <= index < size:
        raise IndexError(f"{name} out of range")
    return index


def _cut_pair(d_left: Any, d_right: Any, depth_bounds: Any) -> list[float]:
    low, high = _bounds(depth_bounds, "depth_bounds")
    try:
        left, right = float(d_left), float(d_right)
    except (TypeError, ValueError) as exc:
        raise ValueError("cut depths must be numbers") from exc
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("cut depths must be finite")
    if not low < left < high or not low < right < high:
        raise ValueError("cut endpoints must be strictly inside depth_bounds")
    return [left, right]


def _validate_cuts(cuts: Any, depth_bounds: Any) -> list[list[float]]:
    low, high = _bounds(depth_bounds, "depth_bounds")
    if not isinstance(cuts, list):
        raise ValueError("cuts must be a list")
    normalised: list[list[float]] = []
    previous_left = previous_right = None
    for cut in cuts:
        if not isinstance(cut, (list, tuple)) or len(cut) != 2:
            raise ValueError("each cut must have two endpoints")
        pair = _cut_pair(cut[0], cut[1], (low, high))
        if previous_left is not None and (
            pair[0] <= previous_left or pair[1] <= previous_right
        ):
            raise ValueError("cuts cross, touch, or are out of depth order")
        previous_left, previous_right = pair
        normalised.append(pair)
    return normalised


def _segment_shape(segment: Any) -> tuple[float, float, list[list[float]], list[int]]:
    if not isinstance(segment, dict):
        raise ValueError("segment must be a dictionary")
    for key in ("id", "start", "end", "cuts", "labels"):
        if key not in segment:
            raise ValueError(f"segment is missing {key}")
    if not isinstance(segment["id"], str) or not segment["id"]:
        raise ValueError("segment id must be a non-empty string")
    try:
        start, end = float(segment["start"]), float(segment["end"])
    except (TypeError, ValueError) as exc:
        raise ValueError("segment times must be numbers") from exc
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("segment must have a finite, non-zero time interval")
    cuts = segment["cuts"]
    labels = segment["labels"]
    if not isinstance(labels, list) or len(labels) != len(cuts) + 1:
        raise ValueError("labels must contain one entry per depth block")
    clean_labels: list[int] = []
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError("labels must be integer LABELS keys")
        label = int(label)
        if label not in LABELS:
            raise ValueError(f"unknown label {label}")
        clean_labels.append(label)
    return start, end, cuts, clean_labels


def _validate_segment(segment: dict[str, Any], depth_bounds: Any) -> None:
    start, end, cuts, labels = _segment_shape(segment)
    _validate_cuts(cuts, depth_bounds)
    if len(labels) != len(cuts) + 1:
        raise ValueError("labels/cuts size mismatch")


def new_document(
    source: dict, time_bounds: tuple, depth_bounds: tuple, channels: list
) -> dict:
    """Create an empty version-one annotation document."""
    if not isinstance(source, dict):
        raise TypeError("source must be a dictionary")
    time_low, time_high = _bounds(time_bounds, "time_bounds")
    depth_low, depth_high = _bounds(depth_bounds, "depth_bounds")
    if not isinstance(channels, list):
        raise ValueError("channels must be a list")
    if any(not isinstance(channel, str) or not channel for channel in channels):
        raise ValueError("channel names must be non-empty strings")
    if len(set(channels)) != len(channels):
        raise ValueError("channel names must be unique")
    return {
        "schema_version": _SCHEMA_VERSION,
        "source": copy.deepcopy(source),
        "time_bounds": [time_low, time_high],
        "depth_bounds": [depth_low, depth_high],
        "label_semantics": _SEMANTICS,
        "channels": {channel: [] for channel in channels},
    }


def add_segment(doc: dict, channel: str, start: float, end: float) -> dict:
    """Add a non-overlapping half-open time segment and return it."""
    validate_document(doc)
    if not isinstance(channel, str) or channel not in doc["channels"]:
        raise KeyError(f"unknown channel {channel!r}")
    time_low, time_high = _bounds(doc["time_bounds"], "time_bounds")
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("segment times must be numbers") from exc
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("segment times must be finite")
    if not time_low <= start < end <= time_high:
        raise ValueError("segment must be a non-zero interval inside time_bounds")
    segments = doc["channels"][channel]
    for existing in segments:
        if start < existing["end"] and end > existing["start"]:
            raise ValueError("segment overlaps an existing segment")
    ids = {
        segment["id"]
        for values in doc["channels"].values()
        for segment in values
    }
    serial = 1
    while f"segment-{serial}" in ids:
        serial += 1
    segment = {"id": f"segment-{serial}", "start": start, "end": end, "cuts": [], "labels": [-1]}
    segments.append(segment)
    segments.sort(key=lambda item: (item["start"], item["end"], item["id"]))
    return segment


def add_cut(segment: dict, d_left: float, d_right: float, depth_bounds: tuple) -> None:
    """Insert a depth cut; the split blocks created by it become unlabelled."""
    _segment_shape(segment)
    candidate = [list(cut) for cut in segment["cuts"]]
    candidate.append(_cut_pair(d_left, d_right, depth_bounds))
    candidate.sort(key=lambda pair: (pair[0], pair[1]))
    _validate_cuts(candidate, depth_bounds)
    old_labels = [int(value) for value in segment["labels"]]
    insertion = candidate.index(_cut_pair(d_left, d_right, depth_bounds))
    segment["cuts"] = candidate
    segment["labels"] = old_labels[:insertion] + [-1, -1] + old_labels[insertion + 1 :]


def move_cut(
    segment: dict, index: int, d_left: float, d_right: float, depth_bounds: tuple
) -> None:
    """Move one cut without allowing it to cross or touch any other cut."""
    _segment_shape(segment)
    index = _index(index, len(segment["cuts"]), "cut index")
    candidate = [list(cut) for cut in segment["cuts"]]
    candidate[index] = _cut_pair(d_left, d_right, depth_bounds)
    _validate_cuts(candidate, depth_bounds)
    segment["cuts"] = candidate
    segment["labels"][index] = -1
    segment["labels"][index + 1] = -1


def remove_cut(segment: dict, index: int) -> None:
    """Remove one cut and leave the newly merged depth block unlabelled."""
    _segment_shape(segment)
    index = _index(index, len(segment["cuts"]), "cut index")
    segment["cuts"].pop(index)
    old_labels = [int(value) for value in segment["labels"]]
    segment["labels"] = old_labels[:index] + [-1] + old_labels[index + 2 :]


def region_polygons(segment: dict, depth_bounds: tuple) -> list[np.ndarray]:
    """Return depth-block quadrilaterals as ``(x_seconds, y_depth)`` arrays."""
    start, end, cuts, labels = _segment_shape(segment)
    clean_cuts = _validate_cuts(cuts, depth_bounds)
    low, high = _bounds(depth_bounds, "depth_bounds")
    polygons: list[np.ndarray] = []
    for index in range(len(labels)):
        if index == 0:
            top_left = top_right = low
        else:
            top_left, top_right = clean_cuts[index - 1]
        if index == len(clean_cuts):
            bottom_left = bottom_right = high
        else:
            bottom_left, bottom_right = clean_cuts[index]
        polygons.append(
            np.asarray(
                [[start, top_left], [end, top_right], [end, bottom_right], [start, bottom_left]],
                dtype=float,
            )
        )
    return polygons


def set_label(segment: dict, index: int, label: int) -> None:
    _segment_shape(segment)
    index = _index(index, len(segment["labels"]), "label index")
    if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
        raise ValueError("label must be an integer LABELS key")
    label = int(label)
    if label not in LABELS:
        raise ValueError(f"unknown label {label}")
    segment["labels"][index] = label


def validate_document(doc: dict) -> None:
    if not isinstance(doc, dict):
        raise ValueError("document must be a dictionary")
    if doc.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported annotation schema version")
    if not isinstance(doc.get("source"), dict):
        raise ValueError("source must be a dictionary")
    time_bounds = _bounds(doc.get("time_bounds"), "time_bounds")
    depth_bounds = _bounds(doc.get("depth_bounds"), "depth_bounds")
    if doc.get("label_semantics") != _SEMANTICS:
        raise ValueError("unsupported label semantics")
    channels = doc.get("channels")
    if not isinstance(channels, dict):
        raise ValueError("channels must be a dictionary")
    ids: set[str] = set()
    for channel, segments in channels.items():
        if not isinstance(channel, str) or not channel or not isinstance(segments, list):
            raise ValueError("channels must map names to segment lists")
        previous_start = previous_end = None
        for segment in segments:
            _validate_segment(segment, depth_bounds)
            start, end = float(segment["start"]), float(segment["end"])
            if not time_bounds[0] <= start < end <= time_bounds[1]:
                raise ValueError("segment must lie inside time_bounds")
            if segment["id"] in ids:
                raise ValueError("segment ids must be unique")
            ids.add(segment["id"])
            if previous_start is not None and (
                start < previous_start or (start == previous_start and end < previous_end)
            ):
                raise ValueError("segments must be sorted by time")
            if previous_end is not None and start < previous_end:
                raise ValueError("segments overlap")
            previous_start, previous_end = start, end


def save_document(doc: dict, path) -> None:
    """Validate and atomically replace a JSON annotation file."""
    validate_document(doc)
    destination = Path(path)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(doc, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_document(path, expected_source=None) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        doc = json.load(stream)
    validate_document(doc)
    if expected_source is not None:
        if not isinstance(expected_source, dict):
            raise TypeError("expected_source must be a dictionary")
        actual_source = doc["source"]
        for field in _SOURCE_IDENTITY_FIELDS:
            if field not in actual_source or field not in expected_source:
                continue
            actual, expected = actual_source[field], expected_source[field]
            if field == "binary_path":
                actual, expected = os.fspath(actual), os.fspath(expected)
            if actual != expected:
                raise ValueError(f"source identity mismatch for {field}")
    return doc


def rasterize(doc: dict, channel: str, times, depth, valid=None) -> np.ndarray:
    """Rasterize labels to int8, with ``-1`` for gaps/invalid/uncertain cells.

    Time intervals are half-open.  A segment ending exactly at the document's
    global time endpoint also covers that endpoint.  At a sloped cut, the
    deeper block owns the cut row (``searchsorted(..., side='right')``).
    """
    validate_document(doc)
    if channel not in doc["channels"]:
        raise KeyError(f"unknown channel {channel!r}")
    times_array = np.asarray(times, dtype=float)
    depth_array = np.asarray(depth, dtype=float)
    if times_array.ndim != 1 or depth_array.ndim != 1:
        raise ValueError("times and depth must be one-dimensional")
    result = np.full((times_array.size, depth_array.size), -1, dtype=np.int8)
    valid_array = None
    if valid is not None:
        valid_array = np.asarray(valid, dtype=bool)
        if valid_array.shape not in (times_array.shape, result.shape):
            raise ValueError("valid must have shape (T,) or (T, D)")
    time_low, global_end = _bounds(doc["time_bounds"], "time_bounds")
    depth_low, depth_high = _bounds(doc["depth_bounds"], "depth_bounds")
    segments = doc["channels"][channel]
    for row, time_value in enumerate(times_array):
        if not math.isfinite(float(time_value)):
            continue
        if valid_array is not None and valid_array.ndim == 1 and not valid_array[row]:
            continue
        selected = None
        for segment in segments:
            is_inside = segment["start"] <= time_value < segment["end"]
            is_final_endpoint = (
                time_value == global_end and segment["end"] == global_end
            )
            if is_inside or is_final_endpoint:
                selected = segment
                break
        if selected is None:
            continue
        x = float(time_value)
        if x == selected["end"]:
            fraction = 1.0
        else:
            fraction = (x - selected["start"]) / (selected["end"] - selected["start"])
        cut_values = np.asarray(
            [cut[0] + fraction * (cut[1] - cut[0]) for cut in selected["cuts"]],
            dtype=float,
        )
        inside_depth = np.isfinite(depth_array) & (depth_array >= depth_low) & (depth_array <= depth_high)
        if cut_values.size:
            block_indices = np.searchsorted(cut_values, depth_array, side="right")
        else:
            block_indices = np.zeros(depth_array.shape, dtype=int)
        labels = np.asarray(selected["labels"], dtype=np.int8)
        row_values = labels[block_indices]
        row_values = np.where(row_values == 3, -1, row_values).astype(np.int8)
        row_values[~inside_depth] = -1
        if valid_array is not None and valid_array.ndim == 2:
            row_values = np.where(valid_array[row], row_values, -1).astype(np.int8)
        result[row] = row_values
    return result


__all__ = [
    "LABELS",
    "new_document",
    "add_segment",
    "add_cut",
    "move_cut",
    "remove_cut",
    "region_polygons",
    "set_label",
    "validate_document",
    "save_document",
    "load_document",
    "rasterize",
]
