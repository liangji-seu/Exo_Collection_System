"""Build a rigid backpack attachment for an OpenSim 4 model.

The module deliberately has no OpenSim Python dependency.  It computes the
rigid-body aggregate with NumPy and edits the OpenSim XML with the standard
library.  A component ``com_m`` is expressed in the selected parent body's
frame.  An explicit ``inertia_kg_m2`` is already expressed in that frame and
is about the component COM.  For a rectangular component, the inertia is
computed in its local frame and rotated with ``orientation_xyz_rad`` using
fixed-axis X-then-Y-then-Z rotations, ``R = Rz(z) @ Ry(y) @ Rx(x)`` (local
vectors to parent vectors).

The input must always be the original scaled model.  Generated models carry
a small sidecar marker and are rejected as inputs, preventing accidental
mass accumulation when a run is repeated.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-12
GENERATOR = "opensim_pipeline.backpack"
MODEL_MARKER = b"opensim_pipeline.backpack generated"


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return arr


def _inertia_matrix(value: Any, name: str = "inertia_kg_m2") -> np.ndarray:
    """Parse OpenSim's [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] or a 3x3 matrix."""
    arr = np.asarray(value, dtype=float)
    if arr.shape == (6,):
        out = np.array(
            [[arr[0], arr[3], arr[4]], [arr[3], arr[1], arr[5]], [arr[4], arr[5], arr[2]]],
            dtype=float,
        )
    elif arr.shape == (3, 3):
        out = (arr + arr.T) / 2.0
        if not np.allclose(arr, arr.T, atol=1e-10, rtol=0):
            raise ValueError(f"{name} must be symmetric")
    else:
        raise ValueError(f"{name} must have six OpenSim values or be a 3x3 matrix")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    principal = np.linalg.eigvalsh(out)
    if principal[0] < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    if principal[2] > principal[0] + principal[1] + 1e-10:
        raise ValueError(f"{name} violates the rigid-body principal-inertia triangle inequality")
    return out


def _opensim_inertia(matrix: np.ndarray) -> list[float]:
    return [float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]),
            float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[1, 2])]


def _parallel_axis(mass: float, displacement: np.ndarray) -> np.ndarray:
    return mass * (float(displacement @ displacement) * np.eye(3) - np.outer(displacement, displacement))


def rotation_xyz_rad(angles: Sequence[float]) -> np.ndarray:
    """Return local-to-parent R for fixed-axis X, then Y, then Z rotations."""
    x, y, z = _vec3(angles, "orientation_xyz_rad")
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def cuboid_inertia(mass_kg: float, dimensions_xyz_m: Sequence[float]) -> np.ndarray:
    """Inertia of a uniform cuboid about its COM in its local xyz frame."""
    if not np.isfinite(mass_kg) or mass_kg < 0:
        raise ValueError("mass_kg must be finite and non-negative")
    d = _vec3(dimensions_xyz_m, "dimensions_xyz_m")
    if np.any(d <= 0):
        raise ValueError("dimensions_xyz_m must be strictly positive")
    x, y, z = d
    return np.diag([mass_kg * (y * y + z * z) / 12.0,
                    mass_kg * (x * x + z * z) / 12.0,
                    mass_kg * (x * x + y * y) / 12.0])


@dataclass(frozen=True)
class Component:
    name: str
    mass_kg: float
    com_m: np.ndarray
    inertia_parent_kg_m2: np.ndarray

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default_name: str) -> "Component":
        name = str(data.get("name", default_name))
        mass = float(data.get("mass_kg", 0.0))
        if not np.isfinite(mass) or mass < 0:
            raise ValueError(f"{name}: mass_kg must be finite and non-negative")
        com = _vec3(data.get("com_m", [0.0, 0.0, 0.0]), f"{name}.com_m")
        has_dims = "dimensions_xyz_m" in data
        has_inertia = "inertia_kg_m2" in data
        if has_dims and has_inertia:
            raise ValueError(f"{name}: provide dimensions_xyz_m or inertia_kg_m2, not both")
        if has_dims:
            local = cuboid_inertia(mass, data["dimensions_xyz_m"])
            orientation = data.get("orientation_xyz_rad", [0.0, 0.0, 0.0])
            R = rotation_xyz_rad(orientation)
            inertia = R @ local @ R.T
        elif has_inertia:
            # Explicit inertia is intentionally already in the parent frame.
            inertia = _inertia_matrix(data["inertia_kg_m2"], f"{name}.inertia_kg_m2")
        else:
            inertia = np.zeros((3, 3), dtype=float)
        if mass <= EPS and (np.max(np.abs(inertia)) > 1e-10):
            raise ValueError(f"{name}: zero mass cannot have non-zero inertia")
        return cls(name, mass, com, inertia)


@dataclass(frozen=True)
class BackpackSpec:
    parent_body: str = "torso"
    mode: str = "weld"
    empty_bag: Component = field(default_factory=lambda: Component("empty_bag", 0.0, np.zeros(3), np.zeros((3, 3))))
    payload: tuple[Component, ...] = ()
    simulation_only: bool = True
    _source_mapping: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BackpackSpec":
        parent = str(data.get("parent_body", "torso"))
        mode = str(data.get("mode", "weld")).lower()
        if mode not in {"weld", "merge"}:
            raise ValueError("mode must be 'weld' or 'merge'")
        empty = Component.from_mapping(data.get("empty_bag", {}), "empty_bag")
        payload_data = data.get("payload", data.get("payload_components", []))
        if not isinstance(payload_data, Iterable) or isinstance(payload_data, (str, bytes, Mapping)):
            raise ValueError("payload must be a list of component objects")
        payload = tuple(Component.from_mapping(x, f"payload_{i + 1}") for i, x in enumerate(payload_data))
        names = [empty.name, *(x.name for x in payload)]
        if len(set(names)) != len(names):
            raise ValueError("component names must be unique")
        return cls(parent, mode, empty, payload, bool(data.get("simulation_only", True)), copy.deepcopy(dict(data)))

    def as_mapping(self) -> dict[str, Any]:
        if self._source_mapping:
            return copy.deepcopy(self._source_mapping)
        def component(c: Component) -> dict[str, Any]:
            return {"name": c.name, "mass_kg": c.mass_kg, "com_m": c.com_m.tolist(),
                    "inertia_kg_m2": _opensim_inertia(c.inertia_parent_kg_m2)}
        return {"parent_body": self.parent_body, "mode": self.mode, "simulation_only": self.simulation_only,
                "empty_bag": component(self.empty_bag), "payload": [component(x) for x in self.payload]}


@dataclass(frozen=True)
class Aggregate:
    mass_kg: float
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray


def aggregate_components(components: Iterable[Component]) -> Aggregate:
    comps = tuple(components)
    total = float(sum(c.mass_kg for c in comps))
    if total <= EPS:
        return Aggregate(0.0, np.zeros(3), np.zeros((3, 3)))
    com = sum((c.mass_kg * c.com_m for c in comps), np.zeros(3)) / total
    inertia = sum((c.inertia_parent_kg_m2 + _parallel_axis(c.mass_kg, c.com_m - com) for c in comps), np.zeros((3, 3)))
    return Aggregate(total, com, (inertia + inertia.T) / 2.0)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children_named(parent: ET.Element, name: str) -> list[ET.Element]:
    return [x for x in list(parent) if _local(x.tag) == name]


def _find_set(root: ET.Element, name: str) -> ET.Element:
    for x in root.iter():
        if _local(x.tag) == name:
            return x
    raise ValueError(f"OpenSim XML has no {name}")


def _objects(set_element: ET.Element) -> ET.Element:
    for child in list(set_element):
        if _local(child.tag) == "objects":
            return child
    return ET.SubElement(set_element, "objects")


def _body(objects: ET.Element, name: str) -> ET.Element:
    for body in list(objects):
        if _local(body.tag) == "Body" and body.get("name") == name:
            return body
    raise ValueError(f"OpenSim XML has no Body named {name!r}")


def _property(body: ET.Element, tag: str, default: Sequence[float]) -> ET.Element:
    for child in list(body):
        if _local(child.tag) == tag:
            return child
    child = ET.Element(tag)
    child.text = " ".join(str(float(x)) for x in default)
    body.append(child)
    return child


def _parse_numbers(element: ET.Element, n: int, name: str) -> np.ndarray:
    try:
        arr = np.fromstring(element.text or "", sep=" ", dtype=float)
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc
    if arr.size != n or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain {n} finite values")
    return arr


def _body_mass_props(body: ET.Element) -> tuple[float, np.ndarray, np.ndarray]:
    mass_node = next((x for x in list(body) if _local(x.tag) == "mass"), None)
    com_node = next((x for x in list(body) if _local(x.tag) == "mass_center"), None)
    inertia_node = next((x for x in list(body) if _local(x.tag) == "inertia"), None)
    if mass_node is None or com_node is None or inertia_node is None:
        raise ValueError(f"Body {body.get('name')!r} lacks mass, mass_center, or inertia")
    try:
        mass = float((mass_node.text or "").strip())
    except ValueError as exc:
        raise ValueError(f"Body {body.get('name')!r} has invalid mass") from exc
    if not np.isfinite(mass) or mass < 0:
        raise ValueError(f"Body {body.get('name')!r} has invalid mass")
    com = _parse_numbers(com_node, 3, "mass_center")
    inertia = _inertia_matrix(_parse_numbers(inertia_node, 6, "inertia"), "inertia")
    return mass, com, inertia


def _set_property(body: ET.Element, tag: str, text: str) -> None:
    node = next((x for x in list(body) if _local(x.tag) == tag), None)
    if node is None:
        node = ET.SubElement(body, tag)
    node.text = text


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.17g}" for v in values)


def _input_guard(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("output must differ from source; never edit the scaled model in place")
    try:
        # A marker in the model itself protects against a copied/renamed model
        # or a sidecar that was moved separately.
        if MODEL_MARKER in source.read_bytes():
            raise ValueError("source appears to be a generated backpack model; use the original scaled model")
    except FileNotFoundError:
        pass
    sidecar = source.with_suffix(".json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("generator") == GENERATOR:
            raise ValueError("source appears to be a generated backpack model; use the original scaled model")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_generation_marker(path: Path) -> None:
    data = path.read_bytes()
    marker = b"<!-- " + MODEL_MARKER + b"; use the original scaled model as input -->"
    if marker in data:
        return
    declaration_end = data.find(b"?>")
    if declaration_end >= 0:
        data = data[: declaration_end + 2] + b"\n" + marker + data[declaration_end + 2:]
    else:
        data = marker + b"\n" + data
    path.write_bytes(data)


def build_backpack_model(source_model: str | Path, output_model: str | Path, spec: BackpackSpec | Mapping[str, Any], sidecar: str | Path | None = None) -> dict[str, Any]:
    """Create a weld or merged model and return the sidecar metadata."""
    source, output = Path(source_model), Path(output_model)
    _input_guard(source, output)
    if not source.is_file():
        raise FileNotFoundError(source)
    sidecar_path = Path(sidecar) if sidecar is not None else output.with_suffix(".json")
    if sidecar_path.resolve() in {source.resolve(), output.resolve()}:
        raise ValueError("sidecar must be a separate JSON file; it cannot overwrite source or output model")
    if not isinstance(spec, BackpackSpec):
        spec = BackpackSpec.from_mapping(spec)
    aggregate = aggregate_components((spec.empty_bag, *spec.payload))
    before_after: dict[str, float | None] = {}
    if aggregate.mass_kg <= EPS:
        # Read-only inspection supplies useful numeric before/after masses;
        # the subsequent copy still preserves the original model bytes.
        try:
            baseline_tree = ET.parse(source)
            baseline_root = baseline_tree.getroot()
            baseline_objects = _objects(_find_set(baseline_root, "BodySet"))
            baseline_parent = _body(baseline_objects, spec.parent_body)
            baseline_mass, _, _ = _body_mass_props(baseline_parent)
            before_after[spec.parent_body] = baseline_mass
            before_after[spec.parent_body + "_after"] = baseline_mass
        except ET.ParseError as exc:
            raise ValueError(f"cannot parse OpenSim XML: {source}") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
    else:
        try:
            tree = ET.parse(source)
        except ET.ParseError as exc:
            raise ValueError(f"cannot parse OpenSim XML: {source}") from exc
        root = tree.getroot()
        body_set = _find_set(root, "BodySet")
        body_objects = _objects(body_set)
        parent = _body(body_objects, spec.parent_body)
        existing_body_names = {x.get("name") for x in list(body_objects) if _local(x.tag) == "Body"}
        if "backpack" in existing_body_names:
            raise ValueError("model already contains a Body named 'backpack'; start from the original scaled model")
        parent_mass, parent_com, parent_inertia = _body_mass_props(parent)
        before_after[spec.parent_body] = parent_mass
        if spec.mode == "merge":
            new_mass = parent_mass + aggregate.mass_kg
            new_com = (parent_mass * parent_com + aggregate.mass_kg * aggregate.com_m) / new_mass
            new_inertia = (parent_inertia + _parallel_axis(parent_mass, parent_com - new_com) +
                           aggregate.inertia_kg_m2 + _parallel_axis(aggregate.mass_kg, aggregate.com_m - new_com))
            _set_property(parent, "mass", f"{new_mass:.17g}")
            _set_property(parent, "mass_center", _fmt(new_com))
            _set_property(parent, "inertia", _fmt(_opensim_inertia((new_inertia + new_inertia.T) / 2)))
            before_after[spec.parent_body + "_after"] = new_mass
        else:
            child_name = "backpack"
            child = ET.Element("Body", {"name": child_name})
            ET.SubElement(child, "mass").text = f"{aggregate.mass_kg:.17g}"
            ET.SubElement(child, "mass_center").text = _fmt(aggregate.com_m)
            ET.SubElement(child, "inertia").text = _fmt(_opensim_inertia(aggregate.inertia_kg_m2))
            body_objects.append(child)
            joint_set = _find_set(root, "JointSet")
            joint_objects = _objects(joint_set)
            joint_name = "backpack_weld"
            joint_names = {x.get("name") for x in list(joint_objects)}
            if joint_name in joint_names:
                raise ValueError("model already contains a joint named 'backpack_weld'; start from the original scaled model")
            joint = ET.Element("WeldJoint", {"name": joint_name})
            # Absolute component paths avoid resolution against the JointSet
            # when the model is loaded by OpenSim 4.x.
            ET.SubElement(joint, "socket_parent_frame").text = f"/bodyset/{spec.parent_body}"
            ET.SubElement(joint, "socket_child_frame").text = f"/bodyset/{child_name}"
            joint_objects.append(joint)
            before_after[child_name] = aggregate.mass_kg
        output.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output, encoding="utf-8", xml_declaration=True)
        _write_generation_marker(output)

    meta = {
        "generator": GENERATOR,
        "schema": 1,
        "simulation_only": bool(spec.simulation_only),
        "source_model": str(source),
        "input_sha256": _sha256(source),
        "output_model": str(output),
        "mode": spec.mode,
        "parent_body": spec.parent_body,
        "parameters": spec.as_mapping(),
        "total_backpack_mass_kg": aggregate.mass_kg,
        "backpack_com_parent_m": aggregate.com_m.tolist(),
        "backpack_inertia_about_com_parent_kg_m2": _opensim_inertia(aggregate.inertia_kg_m2),
        "before_after_body_masses_kg": before_after,
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


__all__ = ["Aggregate", "BackpackSpec", "Component", "aggregate_components", "build_backpack_model", "cuboid_inertia", "rotation_xyz_rad"]
