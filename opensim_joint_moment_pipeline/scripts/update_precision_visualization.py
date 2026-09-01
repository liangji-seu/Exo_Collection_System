from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_mot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    columns = lines[header_index + 1].split()
    values = np.loadtxt(lines[header_index + 2 :])
    return columns, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", required=True, type=Path)
    parser.add_argument("--id", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--lag-ms", required=True, type=int)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()

    source = args.html.read_text(encoding="utf-8")
    match = re.search(
        r'(<script type="application/json" id="exo-sync-data">)(.*?)(</script>)',
        source,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError("exo-sync-data JSON block was not found")
    data = json.loads(match.group(2))

    columns, values = read_mot(args.id)
    time = values[:, columns.index("time")]
    hip_r = values[:, columns.index("hip_flexion_r_moment")]
    hip_l = values[:, columns.index("hip_flexion_l_moment")]
    support = np.load(args.mask).astype(bool)
    if support.shape != (len(time), 2):
        raise ValueError(f"Unexpected support mask shape: {support.shape}")

    sampled_r: list[float | None] = []
    sampled_l: list[float | None] = []
    for sample_time in data["times"]:
        index = int(np.argmin(np.abs(time - sample_time)))
        close = abs(float(time[index]) - float(sample_time)) <= 0.006
        sampled_r.append(round(float(hip_r[index]), 2) if close and support[index, 0] else None)
        sampled_l.append(round(float(hip_l[index]), 2) if close and support[index, 1] else None)

    data["hipMomentR"] = sampled_r
    data["hipMomentL"] = sampled_l
    data["hipMomentLagMs"] = args.lag_ms
    data["hipMomentWindow"] = [round(float(time[0]), 2), round(float(time[-1]), 2)]
    data["hipMomentMethod"] = "precision-single-support"

    # The audit found both horizontal components had the opposite sign in the
    # earlier preview. Make this idempotent so the visual can be regenerated.
    if not data.get("forceDirectionCorrected", False):
        data["forceFull"] = [
            [-float(vector[0]), float(vector[1]), -float(vector[2])]
            for vector in data["forceFull"]
        ]
    data["forceDirectionCorrected"] = True
    if args.overlay:
        overlay = json.loads(args.overlay.read_text(encoding="utf-8"))
        data["opensimMarkerNames"] = overlay["markerNames"]
        data["opensimMarkerFrames"] = overlay["markerFrames"]
        data["opensimBodyNames"] = overlay["bodyNames"]
        data["opensimBodyFrames"] = overlay["bodyFrames"]
        data["opensimBodyConnections"] = overlay["bodyConnections"]
        data["opensimTimeRange"] = overlay["timeRange"]

    replacement = match.group(1) + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + match.group(3)
    source = source[: match.start()] + replacement + source[match.end() :]
    source = source.replace("提前量：<output id=\"exo-delay-value\" class=\"tabular-nums\">150 ms", "提前量：<output id=\"exo-delay-value\" class=\"tabular-nums\">160 ms")
    source = source.replace('id="exo-delay" type="range" min="0" max="220" value="150"', 'id="exo-delay" type="range" min="0" max="220" value="160"')
    source = source.replace("Force/COP 提前 150 ms", "Force/COP 提前 160 ms")
    source = source.replace("髋关节屈伸力矩（测力台提前 150 ms）", "髋关节屈伸力矩（精度修正版：提前 160 ms）")
    source = source.replace(
        "仅显示单支撑有效帧；双支撑期留空。",
        "仅显示 12–30 s 内、去除支撑切换边缘后的单支撑有效帧；双支撑期留空。",
    )
    args.html.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
