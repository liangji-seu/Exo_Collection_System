"""gaitway ASCII 头部受试者信息读取（身高/体重自动预填）的单元测试。

真实 gaitway 导出的 ``Patient height (m)`` 字段实际以**厘米**存储（187.000 = 1.87 m），
但标签误写为 (m)。这里锁定：> 3.0 m 一律按厘米 ÷ 100 归一，避免把身高填成 187 m。
"""

from __future__ import annotations

from pathlib import Path

from pipeline.gaitway import read_gaitway_patient_info

_HEADER = """Gaitway-3D ASCII DataFile version\t6
File date\t02-09-2026
Patient name\tji liang
Patient sex\tM
Patient birth date\t27/04/2000
Patient weight (kg)\t80.000
Patient height (m)\t187.000
Sample rate (Hz)\t1000
Time (s)\tGRFz vertical (N)
0.0000\t793.61
0.0010\t793.64
"""


def _write(tmp_path: Path, text: str = _HEADER) -> Path:
    path = tmp_path / "gaitway.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_weight_and_normalizes_height_cm_to_m(tmp_path: Path) -> None:
    info = read_gaitway_patient_info(_write(tmp_path))
    assert info["weight_kg"] == 80.0
    # 187 cm（误标 m）→ 1.87 m。
    assert info["height_m"] == 1.87


def test_height_already_in_meters_is_kept(tmp_path: Path) -> None:
    text = _HEADER.replace("Patient height (m)\t187.000", "Patient height (m)\t1.87")
    info = read_gaitway_patient_info(_write(tmp_path, text))
    assert info["height_m"] == 1.87


def test_reads_name_sex_birth_date(tmp_path: Path) -> None:
    info = read_gaitway_patient_info(_write(tmp_path))
    assert info["name"] == "ji liang"
    assert info["sex"] == "M"
    assert info["birth_date"] == "27/04/2000"


def test_missing_patient_fields_absent(tmp_path: Path) -> None:
    lines = [
        ln for ln in _HEADER.splitlines()
        if not ln.startswith(("Patient weight", "Patient height"))
    ]
    info = read_gaitway_patient_info(_write(tmp_path, "\n".join(lines) + "\n"))
    assert "weight_kg" not in info
    assert "height_m" not in info


def test_non_gaitway_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "not_gaitway.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")
    assert read_gaitway_patient_info(path) == {}
