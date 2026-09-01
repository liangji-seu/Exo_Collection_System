"""Export OpenSim IK marker and body-origin trajectories for the inline 3-D QC view."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import opensim as osim


BODY_NAMES = [
    "pelvis", "torso",
    "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
]

SKELETON_CONNECTIONS = [
    [0, 1],
    [0, 2], [2, 3], [3, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10], [10, 11],
]


def read_mot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    end = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    columns = lines[end + 1].split()
    values = np.loadtxt(lines[end + 2 :])
    return columns, values


def vec3(value: osim.Vec3) -> list[float]:
    return [round(float(value.get(i)), 4) for i in range(3)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--ik", required=True, type=Path)
    parser.add_argument("--visualization", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    html = args.visualization.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="exo-sync-data">(.*?)</script>', html, re.DOTALL)
    if not match:
        raise RuntimeError("exo-sync-data block not found")
    visual = json.loads(match.group(1))
    sample_times = np.asarray(visual["times"], dtype=float)

    columns, values = read_mot(args.ik)
    ik_time = values[:, columns.index("time")]
    column_lookup = {name: i for i, name in enumerate(columns)}

    model_path = args.model.resolve()
    output_path = args.out.resolve()
    previous_cwd = Path.cwd()
    os.chdir(model_path.parent)
    model = osim.Model(model_path.name)
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    markers = model.getMarkerSet()
    bodies = model.getBodySet()
    marker_names = [markers.get(i).getName() for i in range(markers.getSize())]

    marker_frames: list[list[list[float]] | None] = []
    body_frames: list[list[list[float]] | None] = []
    for sample_time in sample_times:
        if sample_time < ik_time[0] - 0.006 or sample_time > ik_time[-1] + 0.006:
            marker_frames.append(None)
            body_frames.append(None)
            continue
        row = int(np.argmin(np.abs(ik_time - sample_time)))
        if abs(float(ik_time[row]) - float(sample_time)) > 0.006:
            marker_frames.append(None)
            body_frames.append(None)
            continue

        state.setTime(float(ik_time[row]))
        for i in range(coordinates.getSize()):
            coordinate = coordinates.get(i)
            name = coordinate.getName()
            if name not in column_lookup:
                continue
            value = float(values[row, column_lookup[name]])
            if coordinate.getMotionType() == osim.Coordinate.Rotational:
                value = math.radians(value)
            coordinate.setValue(state, value, False)
        model.realizePosition(state)

        marker_frames.append([vec3(markers.get(i).getLocationInGround(state)) for i in range(markers.getSize())])
        body_frames.append([vec3(bodies.get(name).getPositionInGround(state)) for name in BODY_NAMES])

    output = {
        "markerNames": marker_names,
        "markerFrames": marker_frames,
        "bodyNames": BODY_NAMES,
        "bodyFrames": body_frames,
        "bodyConnections": SKELETON_CONNECTIONS,
        "timeRange": [float(ik_time[0]), float(ik_time[-1])],
    }
    os.chdir(previous_cwd)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
