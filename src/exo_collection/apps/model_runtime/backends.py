"""Pluggable model runtimes for the online-testing framework.

The runtime never hardcodes a model architecture.  A ``ModelRuntime`` wraps one
of three backends selected by ``ModelSpec.model.backend``:

- ``torch``     — lazy-import PyTorch, load a class from a module file, restore
                  weights, and call it in eval mode (primary backend).
- ``callable``  — load any Python class exposing ``predict(ndarray)``; zero
                  dependency beyond numpy (fallback / rapid prototyping).
- ``demo``      — a deterministic, input-driven scalar built into the framework
                  so the whole pipeline and its tests run without torch.

Contract: ``predict`` receives a ``[C, T]`` float32 feature window and returns a
flat ``[out_features]`` float32 vector.  A runtime reports ``output_size`` so the
caller can map scalars to motors via ``OutputSpec``.
"""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from exo_collection.apps.model_runtime.spec import ModelBackendSpec, ModelSpec


class ModelRuntimeError(RuntimeError):
    """Raised when a model cannot be loaded or invoked."""


class ModelRuntime(ABC):
    """A loaded model plus its feature/output contract."""

    @abstractmethod
    def load(self, model_spec: ModelBackendSpec, *, base_dir: Path | None = None) -> None:
        """Instantiate the model object and restore any weights."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Map a ``[C, T]`` feature window to a flat ``[out_features]`` vector."""

    @property
    @abstractmethod
    def output_size(self) -> int: ...

    @property
    @abstractmethod
    def backend(self) -> str: ...

    def unload(self) -> None:
        """Release any heavyweight resources (best-effort, default no-op)."""


def _resolve_path(value: str | None, base_dir: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _load_class_from_path(module_path: Path, class_name: str) -> type:
    """Import a class from an arbitrary ``.py`` file without a package install."""
    if not module_path.is_file():
        raise ModelRuntimeError(f"model module not found: {module_path}")
    module_name = f"_exo_model_runtime_{abs(hash(str(module_path)))}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise ModelRuntimeError(f"cannot build import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        # Allow intra-module relative imports and side-imports of sibling helpers.
        parent_dir = str(module_path.parent)
        added = parent_dir not in sys.path
        if added:
            sys.path.insert(0, parent_dir)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            if added:
                try:
                    sys.path.remove(parent_dir)
                except ValueError:
                    pass
    except ModelRuntimeError:
        raise
    except Exception as exc:
        raise ModelRuntimeError(f"failed to import {module_path}: {exc}") from exc
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        raise ModelRuntimeError(
            f"module {module_path} has no class named {class_name!r}"
        )
    return cls


class TorchModelRuntime(ModelRuntime):
    """Load a PyTorch module from a source file and restore weights."""

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._layout: str = "bct"

    @property
    def backend(self) -> str:
        return "torch"

    @property
    def output_size(self) -> int:
        return 1  # unknown until a forward pass; callers infer from prediction

    def load(self, model_spec: ModelBackendSpec, *, base_dir: Path | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ModelRuntimeError(
                "PyTorch is not installed in this environment; use the "
                "'callable' or 'demo' backend instead."
            ) from exc

        module_path = _resolve_path(model_spec.module_path, base_dir)
        weights_path = _resolve_path(model_spec.weights_path, base_dir)
        if module_path is None:
            raise ModelRuntimeError("torch backend requires module_path")
        cls = _load_class_from_path(module_path, model_spec.class_name or "")
        try:
            model = cls(**dict(model_spec.init_kwargs))
        except Exception as exc:
            raise ModelRuntimeError(
                f"failed to instantiate {model_spec.class_name}: {exc}"
            ) from exc

        if weights_path is not None:
            if not weights_path.is_file():
                raise ModelRuntimeError(f"weights file not found: {weights_path}")
            try:
                state = torch.load(str(weights_path), map_location=model_spec.device)
            except Exception as exc:
                raise ModelRuntimeError(f"failed to read weights {weights_path}: {exc}") from exc
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                model.load_state_dict(state)
            except Exception as exc:
                raise ModelRuntimeError(f"failed to restore weights: {exc}") from exc

        try:
            model.eval()
            model.to(model_spec.device)
        except Exception as exc:
            raise ModelRuntimeError(f"failed to move model to {model_spec.device}: {exc}") from exc

        self._model = model
        self._device = model_spec.device
        self._layout = model_spec.input_layout

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelRuntimeError("model not loaded")
        import torch

        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2:
            raise ModelRuntimeError(f"expected [C, T] features, got shape {arr.shape}")
        tensor = torch.from_numpy(arr).to(self._device)
        if self._layout == "bct":
            tensor = tensor.unsqueeze(0)  # [1, C, T]
        else:
            tensor = tensor.permute(1, 0).unsqueeze(0)  # [1, T, C]
        with torch.no_grad():
            output = self._model(tensor)
        out = output.detach().cpu().numpy()
        return _flatten_output(out)

    def unload(self) -> None:
        self._model = None


class CallableModelRuntime(ModelRuntime):
    """Load any Python class exposing ``predict(ndarray) -> ndarray``."""

    def __init__(self) -> None:
        self._obj: Any = None

    @property
    def backend(self) -> str:
        return "callable"

    @property
    def output_size(self) -> int:
        return 1

    def load(self, model_spec: ModelBackendSpec, *, base_dir: Path | None = None) -> None:
        module_path = _resolve_path(model_spec.module_path, base_dir)
        if module_path is None:
            raise ModelRuntimeError("callable backend requires module_path")
        cls = _load_class_from_path(module_path, model_spec.class_name or "")
        try:
            obj = cls(**dict(model_spec.init_kwargs))
        except Exception as exc:
            raise ModelRuntimeError(
                f"failed to instantiate {model_spec.class_name}: {exc}"
            ) from exc
        if not callable(getattr(obj, "predict", None)):
            raise ModelRuntimeError(
                f"{model_spec.class_name} must expose a predict(ndarray) method"
            )
        self._obj = obj

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._obj is None:
            raise ModelRuntimeError("model not loaded")
        arr = np.asarray(features, dtype=np.float32)
        try:
            output = self._obj.predict(arr)
        except Exception as exc:
            raise ModelRuntimeError(f"predict failed: {exc}") from exc
        return _flatten_output(np.asarray(output, dtype=np.float32))

    def unload(self) -> None:
        self._obj = None


class DemoModelRuntime(ModelRuntime):
    """Deterministic, input-driven scalar for framework + E2E testing."""

    def __init__(self) -> None:
        self._amplitude = 1.0
        self._scale = 1.0

    @property
    def backend(self) -> str:
        return "demo"

    @property
    def output_size(self) -> int:
        return 1

    def load(self, model_spec: ModelBackendSpec, *, base_dir: Path | None = None) -> None:
        demo = dict(model_spec.demo or {})
        self._amplitude = float(demo.get("amplitude", 1.0))
        self._scale = float(demo.get("scale", 1.0)) or 1.0

    def predict(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        # A bounded, input-driven value: amplitude * tanh(mean / scale).
        if arr.size == 0:
            mean = 0.0
        else:
            mean = float(np.nanmean(arr))
        value = self._amplitude * float(np.tanh(mean / self._scale))
        return np.asarray([value], dtype=np.float32)


def _flatten_output(output: np.ndarray) -> np.ndarray:
    out = np.asarray(output, dtype=np.float32)
    flat = out.reshape(-1)
    if flat.size == 0:
        raise ModelRuntimeError("model returned an empty output")
    # Normalize NaN / inf to zero so the safety layer never forwards garbage.
    flat = np.where(np.isfinite(flat), flat, 0.0)
    return flat


def build_model_runtime(spec: ModelSpec, *, base_dir: Path | None = None) -> ModelRuntime:
    """Construct and load the runtime selected by ``spec.model.backend``."""
    backend = spec.model.backend
    if backend == "torch":
        runtime: ModelRuntime = TorchModelRuntime()
    elif backend == "callable":
        runtime = CallableModelRuntime()
    elif backend == "demo":
        runtime = DemoModelRuntime()
    else:
        raise ModelRuntimeError(f"unknown backend: {backend!r}")
    runtime.load(spec.model, base_dir=base_dir)
    return runtime


__all__ = [
    "CallableModelRuntime",
    "DemoModelRuntime",
    "ModelRuntime",
    "ModelRuntimeError",
    "TorchModelRuntime",
    "build_model_runtime",
]
