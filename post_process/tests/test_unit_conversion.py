from postprocess.preprocessing.units import convert, mm_to_m, nmm_to_nm


def test_mm_to_m():
    assert mm_to_m(1000.0) == 1.0


def test_nmm_to_nm():
    assert nmm_to_nm(1000.0) == 1.0


def test_convert_roundtrip():
    assert convert(convert(3.7, "mm", "m"), "m", "mm") == 3.7


def test_convert_dimension_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        convert(1.0, "mm", "N")
