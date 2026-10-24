from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from .contract import GateDecision, GateEngine


def _app_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _workspace_root() -> Path:
    return _app_root().parent


def _read_manifest(name: str) -> dict:
    path = _app_root() / "config" / "gate_deployments" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(name: str) -> Path:
    return _app_root() / "config" / "gate_deployments" / name


def _workspace_asset(relative: str) -> Path:
    path = (_workspace_root() / relative).resolve()
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"gate deployment asset is missing: {path}")
    return path


def _verify_sources(manifest: dict) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, expected in manifest.get("source_sha256", {}).items():
        path = _workspace_asset(relative)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"gate source hash mismatch: {path}")
        verified[relative] = actual
    return verified


def _load_source_json(relative: str) -> dict:
    return json.loads(_workspace_asset(relative).read_text(encoding="utf-8-sig"))


def _verify_ao_bindings(manifest: dict) -> None:
    root = "paper_experiments/ao_research_loop_20261024/round02_dev_readout/"
    r1 = "paper_experiments/ao_gate_refinement_20261021/"
    r = "paper_experiments/ao_research_loop_20261024/"
    model_manifest = _load_source_json(root + "model_manifest.json")
    freeze = _load_source_json("paper_experiments/ao_research_loop_20261024/MODEL_SELECTION_FREEZE.json")
    core = _load_source_json(r1 + "core_config_normalizer.json")
    run_lock = _load_source_json(root + "R2_RUN_LOCK.json")
    normalizer_audit_path = r + "round01_dev_attempt04/normalizer_audit.json"
    normalizer_audit = _load_source_json(normalizer_audit_path)["normalizer"]
    if manifest["asset"]["sha256"] != model_manifest["rf_models"]["AO"]["sha256"]:
        raise RuntimeError("AO deployment asset does not match the frozen RF-full model manifest")
    if float(manifest["threshold"]) != float(model_manifest["rf_thresholds"]["AO"]["q99"]):
        raise RuntimeError("AO deployment threshold does not match frozen RF-full AO q99")
    selected = freeze["chosen_for_highlight"]["supervised_AO"]
    if (selected.get("readout"), selected.get("group"), selected.get("budget")) != ("RF_full", "AO", "supervised"):
        raise RuntimeError("AO deployment does not match the frozen supervised_AO selection")
    if manifest["config"] != core["config"]:
        raise RuntimeError("AO runtime config differs from the frozen AO core config")
    for key in ("schema", "scales_rad_s", "axes_xyz", "axis_signs"):
        if manifest["normalizer"].get(key) != normalizer_audit.get(key):
            raise RuntimeError(f"AO runtime normalizer differs from R1 train-only audit field {key}")
    feature_hash = manifest["source_sha256"].get(r + "code/features.py")
    core_hash = manifest["source_sha256"].get(r + "code/adaptive_periodic_mh.py")
    audit_hash = manifest["source_sha256"].get(normalizer_audit_path)
    if feature_hash != run_lock.get("features_sha256") or core_hash != run_lock.get("new_core_sha256"):
        raise RuntimeError("AO feature/core source hashes differ from the R2 run lock")
    if audit_hash != run_lock.get("r1_source", {}).get("normalizer_audit_sha256"):
        raise RuntimeError("AO normalizer audit hash differs from the R2 run lock")


def _verify_phase_bindings(manifest: dict) -> None:
    comparison = "paper_experiments/ao_gate_refinement_20261021/paper_comparison_20261022/"
    bundle = _load_source_json(comparison + "tcn_fit/approved_bundle.json")
    lock = _load_source_json("paper_experiments/ao_research_loop_20261024/paper_baseline_revision03/PAPER_EVAL_LOCK.json")
    hashes = manifest["assets"]["checkpoints"]
    bundle_hashes = bundle["artifacts_sha256"]
    for seed, expected in hashes.items():
        if expected != bundle_hashes.get(f"GaitPhase_seed{seed}.pt"):
            raise RuntimeError(f"phase deployment hash differs from approved bundle for seed {seed}")
    thresholds = lock["calibration"]["thresholds_raw_median"]
    if (float(manifest["raw_threshold"]) != float(thresholds["raw_q99"])
            or float(manifest["threshold"]) != float(thresholds["median_q99"])):
        raise RuntimeError("phase deployment thresholds differ from the frozen calibration lock")
    runner_hash = manifest["source_sha256"].get(comparison + "code/frozen_fit_runner.py")
    if bundle.get("runner_code_sha256") != runner_hash:
        raise RuntimeError("phase deployment runner hash differs from approved bundle")


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(manifest_name: str, manifest: dict, source_hashes: dict[str, str], assets: dict[str, str]) -> dict:
    path = _manifest_path(manifest_name)
    provenance = {
        "mode": manifest["mode"],
        "model_version": manifest["model_version"],
        "deployment_manifest_path": str(path),
        "deployment_manifest_sha256": sha256_file(path),
        "threshold_name": manifest.get("threshold_name", "threshold"),
        "threshold": float(manifest["threshold"]),
        "sources": source_hashes,
        "assets": assets,
        "input": manifest.get("input"),
    }
    if "raw_threshold" in manifest:
        provenance["raw_threshold"] = float(manifest["raw_threshold"])
    if "config" in manifest:
        provenance["config_sha256"] = _json_sha256(manifest["config"])
    if "normalizer" in manifest:
        provenance["normalizer_sha256"] = _json_sha256(manifest["normalizer"])
    return provenance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NoGate:
    mode = "none"

    def __init__(self) -> None:
        self._ticks = 0
        self._sequence = 0
        self.provenance = {"mode": self.mode, "model_version": "none"}

    def reset(self) -> None:
        self._ticks = 0
        self._sequence = 0

    def step(self, source_grid_ns: int, values: Sequence[float], valid: bool = True) -> GateDecision | None:
        if len(values) != 12:
            raise ValueError("gate input must contain 12 right-leg + pelvis channels")
        self._ticks += 1
        if self._ticks % 5:
            return None
        self._sequence += 1
        return GateDecision(True, True, None, None, None, "disabled", int(source_grid_ns), self._sequence, self.mode)


def load_gate(mode: str, *, deployment_root: str | Path | None = None) -> GateEngine:
    """Load a verified frozen gate. Assets default to the adjacent research workspace."""
    if mode == "none":
        return NoGate()
    if mode == "ao":
        manifest = _read_manifest("ao.json")
        sources = _verify_sources(manifest)
        _verify_ao_bindings(manifest)
        asset = (Path(deployment_root).resolve() / Path(manifest["asset"]["path"]).name
                 if deployment_root is not None else _workspace_asset(manifest["asset"]["path"]))
        actual_asset = sha256_file(asset)
        if actual_asset != manifest["asset"]["sha256"]:
            raise RuntimeError(f"AO gate model hash mismatch: {asset}")
        from .ao import AOGate

        engine = AOGate(manifest, asset)
        engine.provenance = _provenance("ao.json", manifest, sources, {asset.name: actual_asset})
        return engine
    if mode == "phase_ensemble":
        manifest = _read_manifest("phase_ensemble.json")
        sources = _verify_sources(manifest)
        _verify_phase_bindings(manifest)
        directory = (Path(deployment_root).resolve() if deployment_root is not None
                     else _workspace_asset(manifest["assets"]["directory"]))
        from .phase_ensemble import PhaseEnsembleGate

        engine = PhaseEnsembleGate(manifest, directory)
        engine.provenance = _provenance("phase_ensemble.json", manifest, sources, engine.asset_hashes)
        provenance = dict(engine.provenance)
        provenance["scaler_hashes"] = dict(engine.scaler_hashes)
        engine.provenance = provenance
        return engine
    raise ValueError(f"unknown gate mode: {mode!r}")
