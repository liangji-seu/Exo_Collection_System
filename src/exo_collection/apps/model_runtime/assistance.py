"""Causal middle-layer conversion from model estimates to right assistance."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from exo_collection.apps.model_runtime.spec import AssistanceSpec

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class AssistanceSample:
    time_ns: int
    estimate_nm_per_kg: float | None
    estimate_nm: float | None
    desired_right_nm: float | None
    assistance_ready: bool = True
    assistance_errors: tuple[str, ...] = ()
    delayed_source_time_ns: int | None = None


class AssistancePipeline:
    """One-pole causal low-pass, gain/mass scaling, then time delay."""

    def __init__(self, config: AssistanceSpec | None = None) -> None:
        self.config = config or AssistanceSpec()
        self.reset()

    def configure(self, config: AssistanceSpec) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._filtered: float | None = None
        self._last_time_ns: int | None = None
        self._history: deque[tuple[int, float]] = deque()

    def preflight(self, unit: str) -> dict[str, object]:
        errors = []
        if unit not in {"Nm", "Nm/kg"}:
            errors.append(f"不支持的模型输出单位：{unit}")
        elif unit == "Nm/kg" and self.config.mass_kg <= 0:
            errors.append("模型输出为 Nm/kg；请填写正体重（kg）后再使能助力")
        return {"ready": not errors, "errors": errors}

    def process(
        self,
        time_ns: int,
        estimate: float,
        *,
        unit: str = "Nm/kg",
    ) -> AssistanceSample:
        value = float(estimate)
        if not math.isfinite(value):
            self.reset()
            return AssistanceSample(int(time_ns), 0.0, 0.0, 0.0)
        t_ns = int(time_ns)
        report = self.preflight(unit)
        if not report["ready"]:
            self.reset()
            return AssistanceSample(t_ns, value if unit == "Nm/kg" else None,
                                    None, None, False, tuple(report["errors"]))
        if self._last_time_ns is not None and t_ns <= self._last_time_ns:
            # A source grid is strictly monotonic; do not create a new output
            # from a repeated or reordered endpoint.
            return AssistanceSample(t_ns, value if unit == "Nm/kg" else None, 0.0, 0.0)
        if self._filtered is None:
            filtered = value
        else:
            dt_s = max(0.0, (t_ns - int(self._last_time_ns)) / _NS_PER_S)
            if self.config.lowpass_hz <= 0:
                filtered = value
            else:
                tau = 1.0 / (2.0 * math.pi * float(self.config.lowpass_hz))
                alpha = 1.0 if tau <= 0 else dt_s / (tau + dt_s)
                filtered = self._filtered + alpha * (value - self._filtered)
        self._filtered = float(filtered)
        self._last_time_ns = t_ns
        self._history.append((t_ns, float(filtered)))
        target_ns = t_ns - int(round(float(self.config.delay_s) * _NS_PER_S))
        delayed_sample = self._delayed_value(target_ns)
        delayed = None if delayed_sample is None else delayed_sample[1]
        delayed_source_ns = None if delayed_sample is None else delayed_sample[0]
        # Drop samples that can no longer be selected, retaining one boundary.
        while len(self._history) > 1 and self._history[1][0] <= target_ns:
            self._history.popleft()
        estimate_nm = value * float(self.config.mass_kg) if unit == "Nm/kg" else value
        if delayed is None:
            desired = 0.0
        else:
            scale = float(self.config.mass_kg) if unit == "Nm/kg" else 1.0
            desired = float(self.config.right_sign) * scale * float(self.config.gain) * delayed
        if not math.isfinite(desired) or not math.isfinite(estimate_nm):
            self.reset()
            desired = 0.0
            estimate_nm = 0.0
        return AssistanceSample(t_ns, value if unit == "Nm/kg" else None, estimate_nm,
                                desired, delayed_source_time_ns=delayed_source_ns)

    def _delayed_value(self, target_ns: int) -> tuple[int, float] | None:
        if not self._history:
            return None
        if target_ns < self._history[0][0]:
            return None
        selected = self._history[0]
        for t_ns, value in self._history:
            if t_ns > target_ns:
                break
            selected = (t_ns, value)
        return int(selected[0]), float(selected[1])


__all__ = ["AssistancePipeline", "AssistanceSample"]
