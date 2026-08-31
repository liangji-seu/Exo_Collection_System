import numpy as np
import tempfile
from pathlib import Path

from postprocess.opensim_io.write_trc import write_trc


def test_write_trc_structure():
    time = np.array([0.0, 0.01, 0.02])
    names = ["R.ASIS", "L.ASIS"]
    data = np.zeros((3, 2, 3))
    data[:, 0, :] = [1, 2, 3]
    data[:, 1, :] = [4, 5, 6]

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "trial.trc"
        write_trc(p, time, names, data, rate_hz=100.0)
        lines = p.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("PathFileType")
        assert "NumMarkers" in lines[1]
        # 3 个数据行 + 5 个头部行（PathFileType / DataRate 标签 / DataRate 值 /
        # marker 名行 / 坐标标签行）
        assert len(lines) == 8
        # 第一个数据行（索引 5）有 1(frame#) + 1(time) + 2*3 = 8 列
        assert len(lines[5].split("\t")) == 8
        assert lines[5].split("\t")[0] == "1"
