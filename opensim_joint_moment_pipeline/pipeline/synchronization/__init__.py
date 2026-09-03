"""时间同步：C3D↔mocap.h5 精确匹配、H5 主机时钟、IMU↔Gaitway 跺脚配对。"""

from .c3d_h5 import C3dH5Match, match_c3d_to_h5
from .clock import (
    ClockHealth,
    clock_health,
    find_imu_sensor,
    imu_sample_rate_hz,
    imu_sensor_on_c3d_time,
    read_host_monotonic_ns,
)
from .marker_names import build_marker_index, normalize_marker_name
from .stomp import (
    StompAlignment,
    StompPair,
    StompRejection,
    detect_impact,
    diagnose_impacts,
    highpass_envelope,
    pair_stomps,
    pair_stomps_diagnosed,
)
from .sync import StompSyncError, run_auto_sync, save_sync_calibration

__all__ = [
    "C3dH5Match",
    "ClockHealth",
    "StompAlignment",
    "StompPair",
    "StompRejection",
    "StompSyncError",
    "build_marker_index",
    "clock_health",
    "detect_impact",
    "diagnose_impacts",
    "find_imu_sensor",
    "highpass_envelope",
    "imu_sample_rate_hz",
    "imu_sensor_on_c3d_time",
    "match_c3d_to_h5",
    "normalize_marker_name",
    "pair_stomps",
    "pair_stomps_diagnosed",
    "read_host_monotonic_ns",
    "run_auto_sync",
    "save_sync_calibration",
]
