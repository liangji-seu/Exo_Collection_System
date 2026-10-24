"""Catalog inspection must remain cheap and describe actual complete deployments."""
import builtins
import json
from pathlib import Path

from exo_collection.apps.model_runtime.model_catalog import discover_local_models, read_model_entry


DEPLOYMENT = Path(__file__).resolve().parents[2] / "runtime_data/models/imu_right_pelvis_torque_100hz_rf94_static_20261017"


def deployment(root, name, *, backend="torchscript", weights=True, metadata=None):
    directory = root / name
    directory.mkdir()
    spec = json.loads((DEPLOYMENT / "model_spec.json").read_text(encoding="utf-8"))
    if backend == "demo":
        spec["model"] = {"backend": "demo"}
        spec["imu_tcn"]["enabled"] = False
    (directory / "model_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    if weights:
        (directory / "final_model.ts").write_bytes(b"placeholder-for-file-check-only")
    if metadata is not None:
        (directory / "model_display.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_installed_model_has_the_real_user_facing_name_without_importing_torch(monkeypatch):
    original_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        assert name.split(".")[0] != "torch", "Discovery must not import a model backend"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    catalog = discover_local_models(DEPLOYMENT.parent)
    entries = {entry.spec_path.parent.name: entry for entry in catalog.entries}
    static = entries[DEPLOYMENT.name]
    assert static.name == "右髋估计·双侧同号助力 · IMU-TCN 静止优化版（100 Hz / 94帧）"
    assert "静止工况增强训练版" in static.description
    assert len(entries) == 1
    assert not catalog.issues


def test_catalog_hides_retired_models_but_discovers_new_deployments(tmp_path):
    for retired in (
        "imu_right_pelvis_torque_v1_20261010",
        "imu_right_pelvis_torque_v2_20261016",
        "imu_right_pelvis_torque_100hz_rf94_20261016",
    ):
        deployment(tmp_path, retired)
    deployment(tmp_path, "future_model")
    catalog = discover_local_models(tmp_path)
    assert [entry.spec_path.parent.name for entry in catalog.entries] == ["future_model"]


def test_catalog_excludes_missing_invalid_and_demo_deployments(tmp_path):
    deployment(tmp_path, "complete", metadata={"name": "已训练模型"})
    deployment(tmp_path, "missing_weights", weights=False)
    deployment(tmp_path, "example", backend="demo")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "model_spec.json").write_text("{}", encoding="utf-8")
    (tmp_path / "missing_spec").mkdir()
    catalog = discover_local_models(tmp_path)
    assert [entry.name for entry in catalog.entries] == ["已训练模型"]
    assert len(catalog.issues) == 4
    assert any("文件缺失" in reason for _, reason in catalog.issues)
    assert any("模拟示例" in reason for _, reason in catalog.issues)


def test_metadata_free_model_uses_contract_description_and_custom_metadata_is_supported(tmp_path):
    directory = deployment(tmp_path, "unfriendly_experiment_name")
    path = directory / "model_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["assistance"]["motor_distribution"] = "per_side"
    path.write_text(json.dumps(spec), encoding="utf-8")
    entry = read_model_entry(directory / "model_spec.json")
    assert entry.name == "右髋力矩估计 · IMU-TCN"
    assert "右腿 + 骨盆 IMU" in entry.description
    assert "Nm/kg" in entry.description
    assert "unfriendly_experiment_name" not in entry.name
    assert str(directory) in entry.tooltip


def test_empty_or_missing_directory_has_no_fake_models(tmp_path):
    assert discover_local_models(tmp_path).entries == ()
    assert discover_local_models(tmp_path / "absent").entries == ()


def test_generic_single_side_output_does_not_invent_a_hip_target(tmp_path):
    directory = deployment(tmp_path, "generic")
    path = directory / "model_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["imu_tcn"]["enabled"] = False
    path.write_text(json.dumps(spec), encoding="utf-8")
    entry = read_model_entry(path)
    assert "右侧力矩" in entry.name
    assert "髋" not in entry.name + entry.description


def test_metadata_free_shared_model_describes_one_estimate_and_two_motor_outputs(tmp_path):
    directory = deployment(tmp_path, "shared")
    entry = read_model_entry(directory / "model_spec.json")
    assert entry.name == "右髋估计·双侧同号助力 · IMU-TCN"
    assert "输出：右髋力矩 (Nm/kg)" in entry.description
    assert "共享右髋预测，左右同号同 Nm" in entry.description
