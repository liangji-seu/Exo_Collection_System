from __future__ import annotations

import numpy as np
import pytest

from us_region_annotation_core import (
    LABELS,
    add_cut,
    add_segment,
    load_document,
    move_cut,
    new_document,
    rasterize,
    region_polygons,
    remove_cut,
    save_document,
    set_label,
    validate_document,
)


def make_doc() -> dict:
    return new_document(
        {"binary_path": "scan.bin", "size_bytes": 12, "mtime_ns": 34},
        (0, 4),
        (0, 50),
        ["US1"],
    )


def test_segments_allow_touching_boundaries_but_reject_overlap() -> None:
    doc = make_doc()
    first = add_segment(doc, "US1", 0, 2)
    second = add_segment(doc, "US1", 2, 4)
    assert first["labels"] == [-1] and second["labels"] == [-1]
    assert [segment["start"] for segment in doc["channels"]["US1"]] == [0.0, 2.0]
    assert first["id"] != second["id"]
    with pytest.raises(ValueError):
        add_segment(doc, "US1", 1.99, 2.5)


def test_validation_rejects_segment_outside_global_time_bounds() -> None:
    doc = make_doc()
    segment = add_segment(doc, "US1", 0, 2)
    segment["start"] = -0.01
    with pytest.raises(ValueError, match="time_bounds"):
        validate_document(doc)


def test_crossing_cuts_are_rejected() -> None:
    doc = make_doc()
    segment = add_segment(doc, "US1", 0, 2)
    add_cut(segment, 15, 25, (0, 50))
    with pytest.raises(ValueError):
        add_cut(segment, 30, 10, (0, 50))
    add_cut(segment, 30, 35, (0, 50))
    with pytest.raises(ValueError):
        move_cut(segment, 0, 40, 20, (0, 50))
    assert segment["cuts"] == [[15.0, 25.0], [30.0, 35.0]]


def test_split_and_merge_do_not_leave_old_labels() -> None:
    doc = make_doc()
    segment = add_segment(doc, "US1", 0, 2)
    set_label(segment, 0, 1)
    add_cut(segment, 10, 12, (0, 50))
    set_label(segment, 0, 1)
    set_label(segment, 1, 2)
    add_cut(segment, 30, 32, (0, 50))
    assert segment["labels"] == [1, -1, -1]
    remove_cut(segment, 0)
    assert segment["labels"] == [-1, -1]
    set_label(segment, 0, 2)
    set_label(segment, 1, 1)
    remove_cut(segment, 0)
    assert segment["cuts"] == []
    assert segment["labels"] == [-1]


def test_roundtrip_and_source_identity_mismatch(tmp_path) -> None:
    doc = make_doc()
    segment = add_segment(doc, "US1", 0, 2)
    add_cut(segment, 20, 22, (0, 50))
    set_label(segment, 0, 1)
    path = tmp_path / "annotation.json"
    save_document(doc, path)
    loaded = load_document(path, expected_source={"binary_path": "scan.bin", "size_bytes": 12})
    assert loaded == doc
    with pytest.raises(ValueError, match="size_bytes"):
        load_document(path, expected_source={"size_bytes": 13})


def test_rasterize_marks_gaps_invalid_and_uncertain_as_minus_one() -> None:
    doc = make_doc()
    first = add_segment(doc, "US1", 0, 1)
    add_cut(first, 20, 20, (0, 50))
    set_label(first, 0, 1)
    set_label(first, 1, 3)
    second = add_segment(doc, "US1", 2, 4)
    set_label(second, 0, 2)
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 4.0])
    depth = np.array([0.0, 10.0, 20.0, 30.0, 50.0])
    result = rasterize(doc, "US1", times, depth, valid=[True, True, True, True, True, False])
    assert result.dtype == np.int8
    assert result[0].tolist() == [1, 1, -1, -1, -1]
    assert np.all(result[2] == -1)  # segment end and the following gap
    assert np.all(result[3] == -1)
    assert result[4].tolist() == [2, 2, 2, 2, 2]
    assert np.all(result[5] == -1)  # valid=False, despite the final endpoint rule


def test_region_polygons_are_time_depth_quadrilaterals() -> None:
    doc = make_doc()
    segment = add_segment(doc, "US1", 1, 3)
    add_cut(segment, 10, 20, (0, 50))
    polygons = region_polygons(segment, (0, 50))
    assert len(polygons) == 2
    assert all(isinstance(polygon, np.ndarray) and polygon.shape == (4, 2) for polygon in polygons)
    np.testing.assert_allclose(polygons[0], [[1, 0], [3, 0], [3, 20], [1, 10]])
    assert LABELS[3] == "不确定"
