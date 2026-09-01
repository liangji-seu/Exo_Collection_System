"""HH19 marker 协议 → gait2392 模型的自定义 MarkerSet（D 方案）。

不硬凑 gait2392 官方 41-marker 命名，而是让模型直接拥有与我们 C3D 一致的
19 个 marker。每个 marker 的 body 归属按解剖定义，初始位置取 gait2392 官方
Scale_MarkerSet 里最接近的同类 marker（MarkerPlacer 会再用 static trial 精确
重定位，故初始值只需「大致对、body 对」即可）。

我们的 19 个真实 marker（左右对称）：
    骨盆：  V.Sacral / L.ASIS / R.ASIS
    大腿：  L.Thigh / R.Thigh
    膝：    L.Knee / L.Knee.Medial / R.Knee / R.Knee.Medial
    小腿：  L.Shank / R.Shank
    踝：    L.Ankle / L.Ankle.Medial / R.Ankle / R.Ankle.Medial
    足：    L.Heel / L.Toe / R.Heel / R.Toe
"""

from __future__ import annotations

# name -> (body, [x, y, z] 初始位置，body 局部系)
# 位置参照 gait2392_Scale_MarkerSet.xml：
#   Knee/Ankle 用 Lat/Med 的官方位置；Thigh/Shank 取官方三 marker 的中位；Toe 取 Med/Lat 中点。
HH19_MARKERS: dict[str, tuple[str, list[float]]] = {
    # 骨盆
    "V.Sacral":     ("pelvis",  [-0.160,  0.040,  0.000]),
    "R.ASIS":       ("pelvis",  [ 0.020,  0.030,  0.128]),
    "L.ASIS":       ("pelvis",  [ 0.020,  0.030, -0.128]),
    # 大腿（单 marker，取 Upper/Front/Rear 中位，偏外侧面）
    "R.Thigh":      ("femur_r", [ 0.036, -0.250,  0.043]),
    "L.Thigh":      ("femur_l", [ 0.036, -0.250, -0.043]),
    # 膝（lateral=外髁=官方 .Lat，medial=.Med）
    "R.Knee":       ("femur_r", [ 0.000, -0.404,  0.050]),
    "L.Knee":       ("femur_l", [ 0.000, -0.404, -0.050]),
    "R.Knee.Medial":("femur_r", [ 0.000, -0.404, -0.050]),
    "L.Knee.Medial":("femur_l", [ 0.000, -0.404,  0.050]),
    # 小腿
    "R.Shank":      ("tibia_r", [ 0.025, -0.092,  0.033]),
    "L.Shank":      ("tibia_l", [ 0.025, -0.092, -0.033]),
    # 踝
    "R.Ankle":      ("tibia_r", [-0.005, -0.410,  0.053]),
    "L.Ankle":      ("tibia_l", [-0.005, -0.410, -0.053]),
    "R.Ankle.Medial":("tibia_r",[ 0.006, -0.389, -0.038]),
    "L.Ankle.Medial":("tibia_l",[ 0.006, -0.389,  0.038]),
    # 足
    "R.Heel":       ("calcn_r", [-0.020,  0.020,  0.000]),
    "L.Heel":       ("calcn_l", [-0.020,  0.020,  0.000]),
    "R.Toe":        ("calcn_r", [ 0.190,  0.005,  0.010]),
    "L.Toe":        ("calcn_l", [ 0.190,  0.005, -0.010]),
}

# IK 权重（参照官方：骨盆/足 = 10，segment 中段 = 1；我们额外用膝/踝成对 marker 钉关节中心，给 5）
IK_MARKER_WEIGHTS: dict[str, float] = {
    "V.Sacral": 10.0, "R.ASIS": 10.0, "L.ASIS": 10.0,
    "R.Thigh": 1.0, "L.Thigh": 1.0,
    "R.Knee": 5.0, "L.Knee": 5.0, "R.Knee.Medial": 5.0, "L.Knee.Medial": 5.0,
    "R.Shank": 1.0, "L.Shank": 1.0,
    "R.Ankle": 5.0, "L.Ankle": 5.0, "R.Ankle.Medial": 5.0, "L.Ankle.Medial": 5.0,
    "R.Heel": 10.0, "L.Heel": 10.0, "R.Toe": 10.0, "L.Toe": 10.0,
}

# Scale 测量对（marker 对 -> 缩放 body）。
# torso 因无 Top.Head 不测量；其余 4 段映射到我们的同名 marker。
SCALE_MEASUREMENTS: dict[str, tuple[list[tuple[str, str]], list[str]]] = {
    "pelvis": ([("R.ASIS", "L.ASIS")], ["pelvis"]),
    "thigh": ([("R.ASIS", "R.Knee"), ("L.ASIS", "L.Knee")],
              ["femur_r", "femur_l"]),
    "shank": ([("R.Knee", "R.Ankle"), ("L.Knee", "L.Ankle")],
              ["tibia_r", "tibia_l", "talus_r", "talus_l"]),
    "foot": ([("R.Heel", "R.Toe"), ("L.Heel", "L.Toe")],
             ["calcn_r", "calcn_l", "toes_r", "toes_l"]),
}

# IK 中锁定为 0（default_value）的坐标（官方做法，sub-talar/mtp 无法用 marker 追踪）
LOCKED_COORDINATES: list[str] = [
    "subtalar_angle_r", "mtp_angle_r", "subtalar_angle_l", "mtp_angle_l",
]


__all__ = [
    "HH19_MARKERS",
    "IK_MARKER_WEIGHTS",
    "SCALE_MEASUREMENTS",
    "LOCKED_COORDINATES",
]
