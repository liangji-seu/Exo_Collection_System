from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from .contract import GateDecision, threshold_allows


def _logv(value: float) -> float:
    return float(np.log(max(value, 1e-8))) if np.isfinite(value) else float("nan")


def _anchor(buf: np.ndarray, grid: np.ndarray, dt: float) -> tuple[float, float, float]:
    z = np.asarray(buf, dtype=float)
    if len(z) < 20:
        return float("nan"), 0.0, float("nan")
    z = z - z.mean(axis=0, keepdims=True)
    energy = float(np.mean(np.sum(z * z, axis=1)))
    if energy < 1e-10:
        return float("nan"), 0.0, 0.0
    n = len(z)
    tt = (np.arange(n) - n + 1) * dt
    w = 2 * np.pi * np.asarray(grid, float)[:, None] * tt[None, :]
    ss, cc = np.sin(w), np.cos(w)
    cs = (2.0 / n) * (ss @ z)
    cc0 = (2.0 / n) * (cc @ z)
    scores = np.sum(cs * cs + cc0 * cc0, axis=1)
    k = int(np.argmax(scores))
    frequency = float(grid[k])
    ordered = np.sort(scores)
    peak_ratio = float(scores[k] / (ordered[-2] + 1e-12)) if len(scores) > 1 else 1.0
    band_ratio = float(scores[k] / (energy + 1e-12))
    evidence = max(0.0, min(1.0, (peak_ratio - 1) / max(peak_ratio, 1e-9)))
    return frequency, evidence, band_ratio


class AOGate:
    """Persistent 100 Hz port of the frozen vector-anchor AO24 feature route."""

    mode = "ao"

    def __init__(self, manifest: dict, asset_path: Path) -> None:
        self._manifest = manifest
        self._config = manifest["config"]
        normalizer = manifest["normalizer"]
        self._scales = np.asarray(normalizer["scales_rad_s"], dtype=float)
        self._axes = np.asarray(normalizer["axes_xyz"], dtype=int)
        self._signs = np.asarray(normalizer["axis_signs"], dtype=float)
        self._grid = np.asarray(self._config["anchor_grid_hz"], dtype=float)
        self._threshold = float(manifest["threshold"])
        self._model = joblib.load(asset_path)
        classes = np.asarray(self._model.classes_)
        matches = np.flatnonzero(classes == 1)
        if not len(matches):
            matches = np.flatnonzero(classes.astype(str) == "OOD")
        if len(matches) != 1:
            raise RuntimeError(f"AO RF model has no unique OOD class: {classes!r}")
        self._ood_column = int(matches[0])
        if int(manifest["feature_count"]) != 24 or int(getattr(self._model, "n_features_in_", -1)) != 24:
            raise RuntimeError("AO RF model feature width does not match frozen AO24 manifest")
        self._ticks = 0
        self._sequence = 0
        self.reset()

    def reset(self) -> None:
        c = self._config
        self._phi = np.zeros(2, dtype=float)
        self._omega = np.full(2, 2 * np.pi * float(c["seed_frequency_hz"]))
        self._a = np.zeros((2, int(c["harmonics"]), 3), dtype=float)
        self._d = np.zeros_like(self._a)
        self._b = np.zeros((2, 3), dtype=float)
        self._hist: list[deque[np.ndarray]] = [deque(maxlen=int(c["anchor_window"])) for _ in range(2)]
        self._continuity = np.zeros(2, dtype=float)
        self._active = False
        self._count = 0
        self._prev_t: float | None = None
        self._prev_grid_ns: int | None = None
        self._first_grid_ns: int | None = None
        self._last_anchor = np.full(2, np.nan)
        self._last_evidence = np.zeros(2)
        self._last_band = np.full(2, np.nan)
        self._rings: deque[dict] = deque(maxlen=100)
        self._ticks = 0
        self._sequence = 0

    def _reset_segment(self) -> None:
        self._active = False
        self._count = 0
        self._hist = [deque(maxlen=int(self._config["anchor_window"])) for _ in range(2)]
        self._continuity[:] = 0
        self._last_anchor[:] = np.nan
        self._last_evidence[:] = 0
        self._last_band[:] = np.nan
        self._rings.clear()

    def _consume(self, grid_ns: int, x: np.ndarray, valid: bool, evaluate: bool) -> tuple[bool, float | None, str]:
        c = self._config
        if self._first_grid_ns is None:
            self._first_grid_ns = grid_ns
        t = (grid_ns - self._first_grid_ns) * 1e-9
        finite = bool(np.isfinite(x).all())
        ok = bool(valid and finite)
        is_gap = self._prev_grid_ns is not None and (
            grid_ns <= self._prev_grid_ns or (grid_ns - self._prev_grid_ns) * 1e-9 > float(c["gap_reset_s"])
        )
        if not ok or (self._active and (self._prev_t is None or t <= self._prev_t or t - self._prev_t > float(c["gap_reset_s"]))):
            self._reset_segment()
            if not ok:
                self._prev_grid_ns = grid_ns
                self._prev_t = None
                return False, None, "invalid_input"
        # Invalid duplicate/out-of-order timestamps reset the old stream and start
        # a fresh segment at the current finite sample, matching the frozen core.
        if is_gap:
            self._reset_segment()
        gyro = np.stack((x[3:6], x[9:12]), axis=0)
        y = gyro / self._scales[:, None]
        h = int(c["harmonics"])
        hg = np.arange(1, h + 1, dtype=float)
        core_valid = False
        error = np.full(2, np.nan)
        reconstruction = np.full((2, 3), np.nan)
        motion = np.full(2, np.nan)
        coherence = np.zeros(2)
        continuity = np.full(2, np.nan)
        frequency = np.full(2, np.nan)
        clipped = np.zeros(2, dtype=float)
        band = np.full(2, np.nan)
        evidence = np.full(2, np.nan)
        anchor = np.full(2, np.nan)

        if not self._active:
            self._active = True
            self._count = 0
            self._phi[:] = 0
            self._omega[:] = 2 * np.pi * float(c["seed_frequency_hz"])
            self._a.fill(0)
            self._d.fill(0)
            self._b[:] = y
            reconstruction[:] = y * self._scales[:, None]
            self._prev_t = t
            self._prev_grid_ns = grid_ns
            self._count = 1
            for j in range(2):
                self._hist[j].append(y[j].copy())
        else:
            pg = float(c["phase_gain"])
            fg = float(c["frequency_gain"])
            cg = float(c["coefficient_gain"])
            for j in range(2):
                oldphi = self._phi[j]
                oldw = self._omega[j]
                olda = self._a[j].copy()
                oldd = self._d[j].copy()
                oldb = self._b[j].copy()
                s = np.sin(hg * oldphi)
                co = np.cos(hg * oldphi)
                pred = oldb + np.sum(olda * s[:, None] + oldd * co[:, None], axis=0)
                err = y[j] - pred
                reconstruction[j] = pred * self._scales[j]
                error[j] = np.linalg.norm(err)
                frequency[j] = oldw / (2 * np.pi)
                deriv = np.sum(hg[:, None] * (olda * co[:, None] - oldd * s[:, None]), axis=0)
                den = float(np.dot(deriv, deriv) + .02)
                phase_innov = float(np.dot(err, deriv) / den)
                self._phi[j] = oldphi + float(c["dt"]) * (oldw + pg * np.clip(phase_innov, -3, 3))
                omega_free = oldw + float(c["dt"]) * fg * np.clip(phase_innov, -2, 2)
                self._hist[j].append(y[j].copy())
                anchor_refreshed = False
                if len(self._hist[j]) >= int(c["anchor_window"]) and self._count % int(c["anchor_step"]) == 0:
                    a_hz, ev, br = _anchor(np.asarray(self._hist[j]), self._grid, float(c["dt"]))
                    self._last_anchor[j], self._last_evidence[j], self._last_band[j] = a_hz, ev, br
                    anchor_refreshed = True
                a_hz = float(self._last_anchor[j])
                ev = float(self._last_evidence[j])
                br = float(self._last_band[j])
                anchor[j], evidence[j], band[j] = a_hz, ev, br
                if anchor_refreshed and np.isfinite(a_hz):
                    blend = float(c["anchor_blend"])
                    omega_free = (1 - blend) * omega_free + blend * 2 * np.pi * a_hz
                lo, hi = c["frequency_bounds_hz"]
                unclipped = omega_free
                self._omega[j] = np.clip(unclipped, 2 * np.pi * float(lo), 2 * np.pi * float(hi))
                clipped[j] = float(self._omega[j] != unclipped)
                self._a[j] = olda + float(c["dt"]) * cg * err[None, :] * s[:, None]
                self._d[j] = oldd + float(c["dt"]) * cg * err[None, :] * co[:, None]
                self._b[j] = oldb + float(c["dt"]) * float(c["bias_gain"]) * err
                motion[j] = float(np.sqrt(np.mean(np.sum(np.asarray(self._hist[j]) ** 2, axis=1))))
                locked = (np.isfinite(a_hz) and ev > .02 and np.isfinite(br) and br > .10
                          and motion[j] > float(c["activity_floor"]) and not clipped[j])
                self._continuity[j] = min(1., self._continuity[j] + float(c["dt"])) if locked else max(0., self._continuity[j] - float(c["dt"]) / .5)
                continuity[j] = self._continuity[j]
                coherence[j] = np.exp(-abs(oldw / (2 * np.pi) - a_hz) / .25) if np.isfinite(a_hz) else 0.
            self._count += 1
            self._prev_t = t
            self._prev_grid_ns = grid_ns
            core_valid = self._count >= int(c["warmup"])

        self._rings.append({"gyro": gyro.copy(), "valid": core_valid, "error": error,
                            "reconstruction": reconstruction, "motion": motion,
                            "coherence": coherence, "continuity": continuity,
                            "frequency": frequency, "clipped": clipped,
                            "band": band, "evidence": evidence, "anchor": anchor})
        if not evaluate:
            return core_valid, None, "pending"
        feature = self._features()
        if feature is None:
            return False, None, "warming_up"
        raw_score = float(self._model.predict_proba(feature.reshape(1, 24))[0, self._ood_column])
        if not np.isfinite(raw_score):
            return False, raw_score, "non_finite_score"
        return True, raw_score, "rejected" if raw_score > self._threshold else "allowed"

    def _features(self) -> np.ndarray | None:
        if len(self._rings) < 100:
            return None
        rows = list(self._rings)
        if not all(row["valid"] for row in rows):
            return None
        gyro = np.stack([row["gyro"] for row in rows])
        activity_right = _logv(float(np.sqrt(np.mean(np.sum(gyro[:, 0] ** 2, axis=1)))))
        activity_pelvis = _logv(float(np.sqrt(np.mean(np.sum(gyro[:, 1] ** 2, axis=1)))))
        activity = np.array([activity_right, activity_pelvis, activity_right - activity_pelvis])
        error = np.stack([row["error"] for row in rows])
        reconstruction = np.stack([row["reconstruction"] for row in rows])
        motion = np.stack([row["motion"] for row in rows])
        error_rms = np.sqrt(np.mean(error ** 2, axis=0))
        reconstruction_rms = np.sqrt(np.mean(np.sum(reconstruction ** 2, axis=2), axis=0))
        band = rows[-1]["band"]
        evidence = rows[-1]["evidence"]
        anchor = rows[-1]["anchor"]
        coherence = rows[-1]["coherence"]
        continuity = rows[-1]["continuity"]
        frequency = rows[-1]["frequency"]
        clipped = rows[-1]["clipped"]
        values = np.r_[
            activity,
            _logv(float(band[0])), _logv(float(band[1])), evidence,
            anchor, anchor[0] - anchor[1],
            _logv(float(error_rms[0])), _logv(float(error_rms[1])),
            _logv(float(error_rms[0] / max(float(motion[-1, 0]), 1e-8))),
            _logv(float(error_rms[1] / max(float(motion[-1, 1]), 1e-8))),
            coherence, continuity, frequency, clipped,
            _logv(float(reconstruction_rms[0])), _logv(float(reconstruction_rms[1])),
        ]
        if values.shape != (24,) or not np.isfinite(values).all():
            return None
        return values

    def step(self, source_grid_ns: int, values: Sequence[float], valid: bool = True) -> GateDecision | None:
        x = np.asarray(values, dtype=float)
        if x.shape != (12,):
            raise ValueError("gate input must be a 12-value right-leg + pelvis sample")
        evaluate = (self._ticks + 1) % 5 == 0
        self._ticks += 1
        ready, score, reason = self._consume(int(source_grid_ns), x, bool(valid), evaluate)
        if self._ticks % 5:
            return None
        self._sequence += 1
        return GateDecision(ready, bool(ready and score is not None and threshold_allows(score, self._threshold)),
                            score if ready else None, score, self._threshold, reason,
                            int(source_grid_ns), self._sequence, self.mode)
