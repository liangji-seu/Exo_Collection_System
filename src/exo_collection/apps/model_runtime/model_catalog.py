"""Cheap local deployment discovery: JSON and file checks, no model imports."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .spec import ModelSpec, load_model_spec


# These deployments were retired from the online app.  Ignore them even when
# another workstation still has a local copy; other future deployments remain
# discoverable.
RETIRED_ONLINE_DEPLOYMENTS = frozenset({
    "imu_right_pelvis_torque_v1_20261010",
    "imu_right_pelvis_torque_v2_20261016",
    "imu_right_pelvis_torque_100hz_rf94_20261016",
})


@dataclass(frozen=True)
class ModelCatalogEntry:
    spec_path: Path
    name: str
    description: str

    @property
    def tooltip(self) -> str:
        return f"{self.description}\n{self.spec_path}"


@dataclass(frozen=True)
class ModelCatalog:
    entries: tuple[ModelCatalogEntry, ...]
    issues: tuple[tuple[Path, str], ...]


def _default_display(spec: ModelSpec) -> tuple[str, str]:
    modality_names = {"imu": "IMU", "ultrasound": "超声", "emg": "肌电",
                      "encoder": "编码器", "mocap": "动捕"}
    inputs = " + ".join(modality_names.get(item.modality, item.modality)
                        for item in spec.features.inputs)
    side = spec.output.single_output_maps_to
    target = "右侧力矩" if side == "right_torque" else "左侧力矩" if side == "left_torque" else "双侧力矩"
    if spec.imu_tcn.enabled:
        inputs, target, architecture = "右腿 + 骨盆 IMU", "右髋力矩", "IMU-TCN"
    else:
        architecture = f"{inputs} {spec.model.backend}"
    if getattr(spec.assistance, "motor_distribution", "per_side") == "right_to_both_same_sign":
        estimate_target = "右髋" if spec.imu_tcn.enabled else target
        source_target = "右髋" if spec.imu_tcn.enabled else "右侧"
        return (f"{estimate_target}估计·双侧同号助力 · {architecture}",
                f"输入：{inputs} ｜输出：{target} ({spec.output.unit}) ｜助力：共享{source_target}预测，左右同号同 Nm")
    return (f"{target}估计 · {architecture}",
            f"输入：{inputs} ｜输出：{target} ({spec.output.unit})")


def read_model_entry(path: Path, *, require_files: bool = True) -> ModelCatalogEntry:
    path = Path(path).resolve()
    spec = load_model_spec(path)
    if require_files:
        if spec.model.backend == "demo":
            raise ValueError("模拟示例不是已部署训练模型")
        if spec.model.backend == "torch" and not spec.model.weights_path:
            raise ValueError("训练模型缺少权重路径")
        for field in ("module_path", "weights_path"):
            value = getattr(spec.model, field)
            if value:
                required = Path(value)
                if not required.is_absolute():
                    required = path.parent / required
                if not required.is_file():
                    raise ValueError(f"模型文件缺失：{required.name}")
    name, description = _default_display(spec)
    metadata_path = path.parent / "model_display.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("模型显示信息必须是 JSON 对象")
        for key in ("name", "description"):
            if key in metadata and (not isinstance(metadata[key], str) or not metadata[key].strip()):
                raise ValueError(f"模型显示信息 {key} 必须是非空文本")
        name = metadata.get("name", name).strip()
        description = metadata.get("description", description).strip()
    return ModelCatalogEntry(path, name, description)


def discover_local_models(directory: Path) -> ModelCatalog:
    entries, issues = [], []
    directory = Path(directory)
    for deployment in sorted(directory.iterdir()) if directory.is_dir() else ():
        if not deployment.is_dir():
            continue
        if deployment.name in RETIRED_ONLINE_DEPLOYMENTS:
            continue
        path = deployment / "model_spec.json"
        if not path.is_file():
            issues.append((path, "部署目录缺少 model_spec.json"))
            continue
        try:
            entries.append(read_model_entry(path))
        except Exception as exc:
            issues.append((path, str(exc)))
    return ModelCatalog(tuple(entries), tuple(issues))
