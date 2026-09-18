"""Unit tests for the pluggable model runtimes (callable / demo / torch)."""

from __future__ import annotations

import numpy as np
import pytest

from exo_collection.apps.model_runtime.backends import (
    CallableModelRuntime,
    DemoModelRuntime,
    ModelRuntimeError,
    build_model_runtime,
)
from exo_collection.apps.model_runtime.spec import ModelBackendSpec, ModelSpec

_CALLABLE_SRC = '''
import numpy as np

class SumOverTime:
    def __init__(self, scale=1.0):
        self.scale = scale

    def predict(self, features):
        return np.asarray(features, dtype=np.float32).sum(axis=1) * self.scale
'''


def _spec(backend: str, **kwargs) -> ModelSpec:
    model = {"backend": backend, **kwargs}
    return ModelSpec.model_validate(
        {
            "schema_version": "1.0.0",
            "model": model,
            "features": {
                "control_rate_hz": 100.0,
                "window_s": 0.1,
                "inputs": [{"modality": "imu", "channels": None}],
            },
            "output": {"kind": "torque_left_right"},
        }
    )


def test_demo_runtime_deterministic_and_bounded(tmp_path) -> None:
    runtime = DemoModelRuntime()
    runtime.load(ModelBackendSpec(backend="demo", demo={"amplitude": 2.0, "scale": 1.0}))
    features = np.full((2, 8), 0.5, dtype=np.float32)
    out = runtime.predict(features)
    assert out.shape == (1,)
    assert -2.0 <= out[0] <= 2.0
    assert runtime.predict(features)[0] == pytest.approx(out[0])
    assert runtime.backend == "demo"


def test_callable_runtime(tmp_path) -> None:
    module = tmp_path / "m.py"
    module.write_text(_CALLABLE_SRC, encoding="utf-8")
    runtime = CallableModelRuntime()
    runtime.load(
        ModelBackendSpec(
            backend="callable",
            module_path=str(module),
            class_name="SumOverTime",
            init_kwargs={"scale": 2.0},
        )
    )
    features = np.ones((3, 5), dtype=np.float32)
    out = runtime.predict(features)
    assert out.shape == (3,)
    assert out.tolist() == [10.0, 10.0, 10.0]


def test_callable_missing_predict_rejected(tmp_path) -> None:
    module = tmp_path / "bad.py"
    module.write_text("class NoPredict:\n    pass\n", encoding="utf-8")
    runtime = CallableModelRuntime()
    with pytest.raises(ModelRuntimeError):
        runtime.load(
            ModelBackendSpec(backend="callable", module_path=str(module), class_name="NoPredict")
        )


def test_missing_module_raises(tmp_path) -> None:
    runtime = CallableModelRuntime()
    with pytest.raises(ModelRuntimeError):
        runtime.load(
            ModelBackendSpec(
                backend="callable",
                module_path=str(tmp_path / "nope.py"),
                class_name="X",
            )
        )


def test_build_model_runtime_demo(tmp_path) -> None:
    runtime = build_model_runtime(_spec("demo"))
    assert isinstance(runtime, DemoModelRuntime)


def test_predict_before_load_raises() -> None:
    runtime = DemoModelRuntime()
    # Demo is load-less, so this exercises the guard on a loader-backed runtime.
    callable_runtime = CallableModelRuntime()
    with pytest.raises(ModelRuntimeError):
        callable_runtime.predict(np.zeros((2, 4), dtype=np.float32))


_TORCH_SRC = '''
import torch

class TinyTCN(torch.nn.Module):
    def __init__(self, in_channels=2, window_len=8, out_features=1):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_features, kernel_size=1)

    def forward(self, x):
        return self.conv(x).mean(dim=-1)
'''


def test_torch_runtime(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    module = tmp_path / "tcn.py"
    module.write_text(_TORCH_SRC, encoding="utf-8")

    # Build the same tiny net and save real weights for the runtime to restore.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tcn_fixture", str(module))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model = fixture.TinyTCN(in_channels=2, window_len=8, out_features=1)
    weights = tmp_path / "tcn.pt"
    torch.save(model.state_dict(), str(weights))

    runtime = build_model_runtime(
        _spec(
            "torch",
            module_path=str(module),
            class_name="TinyTCN",
            init_kwargs={"in_channels": 2, "window_len": 8, "out_features": 1},
            weights_path=str(weights),
            device="cpu",
            input_layout="bct",
        )
    )
    features = np.ones((2, 8), dtype=np.float32)
    out = runtime.predict(features)
    assert out.shape == (1,)
    assert np.isfinite(out).all()
