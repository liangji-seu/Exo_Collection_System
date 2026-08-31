import numpy as np
import tempfile
from pathlib import Path

from postprocess.opensim_io.write_grf_mot import write_grf_mot


def test_write_grf_mot_columns():
    time = np.array([0.0, 0.01])
    n = 2
    feet = [
        {"name": "left", "force": np.zeros((n, 3)),
         "point": np.zeros((n, 3)), "torque": np.zeros((n, 3))},
        {"name": "right", "force": np.zeros((n, 3)),
         "point": np.zeros((n, 3)), "torque": np.zeros((n, 3))},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "grf.mot"
        write_grf_mot(p, time, feet)
        lines = p.read_text(encoding="utf-8").splitlines()
        assert "endheader" in lines
        # 列名行 = time + 2 只脚 * 9 列 = 19 列
        # 头部共 6 行（name/version/nRows/nColumns/inDegrees/endheader），列名在第 7 行（索引 6）
        col_line = lines[6]
        cols = col_line.split("\t")
        assert len(cols) == 19
        assert cols[0] == "time"
        assert "1_ground_force_vx" in cols
        assert "2_ground_torque_z" in cols
        # 数据行：6 头部 + 1 列名 + n 数据
        assert len(lines) == 6 + 1 + n
