from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .contract import GateDecision, threshold_allows
from .loader import sha256_file


class _CausalLayer(nn.Module):
    def __init__(self, in_channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(in_channels, 30, kernel_size=20, dilation=self.dilation, padding=0)
        self.bn = nn.BatchNorm1d(30)

    def forward(self, x):
        x = nn.functional.pad(x, (19 * self.dilation, 0))
        return self.bn(nn.functional.relu(self.conv(x)))


class _GaitPhaseTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_CausalLayer(12 if i == 0 else 30, d) for i, d in enumerate((1, 2, 4))])
        self.head = nn.Linear(30, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, :, -1]).squeeze(-1)


class PhaseEnsembleGate:
    """Seven frozen phase models with the paper's sample variance and causal median."""

    mode = "phase_ensemble"
    _SEEDS = (17, 29, 43, 59, 71, 83, 97)
    _WINDOW = 100
    _STRIDE = 5
    _GAP_NS = 15_000_000
    _MEDIAN_SECONDS = 0.5

    def __init__(self, manifest: dict, checkpoint_dir: Path) -> None:
        self._manifest = manifest
        self._threshold = float(manifest["threshold"])
        self._raw_threshold = float(manifest["raw_threshold"])
        if int(manifest["window_samples"]) != self._WINDOW or int(manifest["decision_stride_samples"]) != self._STRIDE:
            raise RuntimeError("phase gate deployment window or stride differs from frozen definition")
        self._models: list[nn.Module] = []
        self._scalers: list[tuple[np.ndarray, np.ndarray]] = []
        self.asset_hashes: dict[str, str] = {}
        self.scaler_hashes: dict[str, str] = {}
        self._torch = torch
        for seed in self._SEEDS:
            path = checkpoint_dir / f"GaitPhase_seed{seed}.pt"
            expected = manifest["assets"]["checkpoints"][str(seed)]
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"phase checkpoint hash mismatch for seed {seed}: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("family") != "GaitPhase" or int(checkpoint.get("seed", -1)) != seed:
                raise RuntimeError(f"phase checkpoint identity mismatch for seed {seed}")
            scaler = checkpoint.get("input_scaler") or checkpoint.get("training_info", {}).get("input_scaler")
            if not scaler or np.asarray(scaler.get("mean")).shape != (12,) or np.asarray(scaler.get("scale")).shape != (12,):
                raise RuntimeError(f"phase checkpoint has invalid input scaler for seed {seed}")
            mean = np.asarray(scaler["mean"], dtype=np.float32)
            scale = np.asarray(scaler["scale"], dtype=np.float32)
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise RuntimeError(f"phase checkpoint scaler is non-finite for seed {seed}")
            model = _GaitPhaseTCN()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            self._models.append(model)
            self._scalers.append((mean, scale))
            self.asset_hashes[path.name] = actual
            self.scaler_hashes[str(seed)] = hashlib.sha256(mean.tobytes() + scale.tobytes()).hexdigest()
        self.reset()

    def reset(self) -> None:
        self._samples: deque[np.ndarray] = deque(maxlen=self._WINDOW)
        self._scores: deque[tuple[int, float]] = deque()
        self._last_grid_ns: int | None = None
        self._first_grid_ns: int | None = None
        self._ticks = 0
        self._sequence = 0

    def _raw_score(self) -> float:
        window = np.asarray(self._samples, dtype=np.float32)
        member_scores = np.empty(7, dtype=np.float32)
        with torch.inference_mode():
            for i, (model, (mean, scale)) in enumerate(zip(self._models, self._scalers)):
                transformed = (window.reshape(-1, 12) - mean) / scale
                transformed = transformed.reshape(1, self._WINDOW, 12).transpose(0, 2, 1).copy()
                xb = torch.from_numpy(transformed)
                member_scores[i] = float(model(xb).cpu().numpy()[0])
        # Keep the original runner's float32 result after a float64 ddof=1 variance.
        return float(np.var(member_scores.astype(np.float64), ddof=1).astype(np.float32))

    def _filtered_score(self, time_s: float, raw_score: float) -> float:
        self._scores.append((time_s, raw_score))
        while self._scores and time_s - self._scores[0][0] > self._MEDIAN_SECONDS:
            self._scores.popleft()
        return float(np.median(np.asarray([item[1] for item in self._scores], dtype=np.float64)).astype(np.float32))

    def step(self, source_grid_ns: int, values: Sequence[float], valid: bool = True) -> GateDecision | None:
        x = np.asarray(values, dtype=np.float32)
        if x.shape != (12,):
            raise ValueError("gate input must be a 12-value right-leg + pelvis sample")
        grid_ns = int(source_grid_ns)
        if self._first_grid_ns is None:
            self._first_grid_ns = grid_ns
        gap = self._last_grid_ns is not None and (
            grid_ns <= self._last_grid_ns or grid_ns - self._last_grid_ns > self._GAP_NS
        )
        good = bool(valid and np.isfinite(x).all() and not gap)
        if not good:
            self._samples.clear()
            self._scores.clear()
        if good:
            self._samples.append(x.copy())
        self._last_grid_ns = grid_ns
        self._ticks += 1
        if self._ticks % self._STRIDE:
            return None
        self._sequence += 1
        if len(self._samples) < self._WINDOW:
            reason = "invalid_input" if not good else "warming_up"
            return GateDecision(False, False, None, None, self._threshold, reason,
                                grid_ns, self._sequence, self.mode)
        raw_score = self._raw_score()
        if not np.isfinite(raw_score):
            self._scores.clear()
            return GateDecision(False, False, None, raw_score, self._threshold, "non_finite_score",
                                grid_ns, self._sequence, self.mode)
        # Match the frozen resampled-time vector, which is arange(N) * 0.01.
        # A relative tick clock also avoids loss of precision from absolute monotonic ns.
        median_score = self._filtered_score((self._ticks - 1) * 0.01, raw_score)
        if not np.isfinite(median_score):
            return GateDecision(False, False, None, raw_score, self._threshold, "non_finite_score",
                                grid_ns, self._sequence, self.mode)
        allow = threshold_allows(median_score, self._threshold)
        reason = "allowed" if allow else "rejected"
        return GateDecision(True, allow, median_score, raw_score, self._threshold, reason,
                            grid_ns, self._sequence, self.mode)
