from __future__ import annotations

from uuid import uuid4

from exo_collection.orchestration.models import TrialRunRequest


def test_default_request_uses_test_partition_and_stable_subject_identity(tmp_path) -> None:
    first = TrialRunRequest(data_root=tmp_path)
    second = TrialRunRequest(data_root=tmp_path)

    assert first.project_code == "T"
    assert first.project_name == "测试"
    assert first.subject_code == "001"
    assert first.duration_s is None
    assert first.project_uuid == second.project_uuid
    assert first.subject_uuid == second.subject_uuid
    assert first.session_uuid != second.session_uuid
    assert first.trial_uuid != second.trial_uuid


def test_formal_and_test_partitions_have_distinct_stable_project_ids(tmp_path) -> None:
    formal = TrialRunRequest(
        data_root=tmp_path,
        project_code="F",
        project_name="正式",
        subject_code="001",
    )
    test = TrialRunRequest(
        data_root=tmp_path,
        project_code="T",
        project_name="测试",
        subject_code="001",
    )

    assert formal.project_uuid != test.project_uuid
    assert formal.subject_uuid != test.subject_uuid


def test_explicit_hierarchy_uuids_are_never_rewritten(tmp_path) -> None:
    project_uuid = uuid4()
    subject_uuid = uuid4()

    request = TrialRunRequest(
        data_root=tmp_path,
        project_uuid=project_uuid,
        subject_uuid=subject_uuid,
    )

    assert request.project_uuid == project_uuid
    assert request.subject_uuid == subject_uuid
