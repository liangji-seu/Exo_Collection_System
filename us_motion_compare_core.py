"""Read-only backend for comparing four ultrasound motion estimators.

The public API is intentionally small so a desktop UI can be built without
knowing about the frozen replay implementation::

    session = load_recording(path, progress=..., cancel=...)
    engine = ComparisonEngine()
    result = engine.analyze(session, channel=3, start=20, end=20.3)

``result`` contains only the requested display interval.  A one-second
causal warmup is evaluated internally so that the nine-frame models and the
CNN9 region tracker have the same state they have in the reviewed replay.
No result or source data is written to disk.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .us_motion_core import LoadCancelled, MotionSession, load_session


_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
_COARSE = _REPO / "paper_experiments" / "aus_selfsupervised_motion_20261013" / "coarse_cnn_pilot"
if str(_COARSE) not in sys.path:
    sys.path.insert(0, str(_COARSE))

try:
    from evaluate import METHODS, ReplayConfig, evaluate_frame
    from region_tracker import RegionTracker, find_large_regions
    from run_evaluation import load_checkpoint
except ImportError:  # pragma: no cover - package import when the pilot is imported as a package
    from ..paper_experiments.aus_selfsupervised_motion_20261013.coarse_cnn_pilot.evaluate import (  # type: ignore
        METHODS, ReplayConfig, evaluate_frame,
    )
    from ..paper_experiments.aus_selfsupervised_motion_20261013.coarse_cnn_pilot.region_tracker import (  # type: ignore
        RegionTracker, find_large_regions,
    )
    from ..paper_experiments.aus_selfsupervised_motion_20261013.coarse_cnn_pilot.run_evaluation import (  # type: ignore
        load_checkpoint,
    )


# Keep the names and order stable for the UI.  Values are deliberately plain
# Chinese labels so callers do not need to translate display text themselves.
METHOD_LABELS: dict[str, str] = {
    "ncc": "局部相关",
    "traditional12": "连续控制点拟合",
    "cnn2": "两帧匹配+CNN",
    "cnn9": "九帧匹配+CNN",
}


class CancelledError(RuntimeError):
    """Raised when a caller cancels loading or analysis."""


def load_recording(path: str | Path, progress: Callable[[int, str], Any] | None = None,
                   cancel: Callable[[], bool] | None = None) -> MotionSession:
    """Load one recording through the existing read-only session loader."""
    try:
        return load_session(path, progress=progress, cancel=cancel)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"录音文件不存在：{path}") from exc
    except LoadCancelled as exc:
        raise CancelledError("已取消加载") from exc


def _default_checkpoints() -> tuple[Path, Path]:
    output = _COARSE / "outputs_matching"
    return output / "matching_h2_best.pt", output / "matching_h9_best.pt"


class ComparisonEngine:
    """Lazy model holder and read-only four-method comparison engine."""

    def __init__(self, checkpoint2: str | Path | None = None,
                 checkpoint9: str | Path | None = None,
                 config: ReplayConfig | None = None, **kwargs: Any) -> None:
        # The aliases make the API convenient for UI code while retaining the
        # short checkpoint2/checkpoint9 names used by the replay scripts.
        checkpoint2 = kwargs.pop("model2_path", checkpoint2)
        checkpoint9 = kwargs.pop("model9_path", checkpoint9)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"未知 ComparisonEngine 参数：{unknown}")
        default2, default9 = _default_checkpoints()
        self.checkpoint2 = Path(checkpoint2) if checkpoint2 is not None else default2
        self.checkpoint9 = Path(checkpoint9) if checkpoint9 is not None else default9
        self.config = config if config is not None else ReplayConfig()
        self._model2: Any | None = None
        self._model9: Any | None = None

    @property
    def models_loaded(self) -> bool:
        return self._model2 is not None and self._model9 is not None

    def _ensure_models(self) -> tuple[Any, Any]:
        if self._model2 is not None and self._model9 is not None:
            return self._model2, self._model9
        missing = [str(p) for p in (self.checkpoint2, self.checkpoint9) if not p.is_file()]
        if missing:
            raise FileNotFoundError("模型文件不存在：" + "；".join(missing))
        try:
            import torch
            torch.set_num_threads(4)
            model2 = load_checkpoint(self.checkpoint2)
            model9 = load_checkpoint(self.checkpoint9)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"模型加载失败：{exc}") from exc
        self._model2, self._model9 = model2, model9
        return model2, model9

    @staticmethod
    def _check_cancel(cancel: Callable[[], bool] | None) -> None:
        if cancel is not None and cancel():
            raise CancelledError("已取消分析")

    @staticmethod
    def _progress(progress: Callable[[int, str], Any] | None, percent: float, message: str) -> None:
        if progress is not None:
            progress(max(0, min(100, int(percent))), message)

    def analyze(self, session: MotionSession, channel: int, start: float, end: float,
                progress: Callable[[int, str], Any] | None = None,
                cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Analyze an inclusive time interval for one physical channel.

        The returned arrays are NumPy arrays and have one row per selected
        20-Hz display frame.  ``spacing`` is the fractional spacing change;
        callers that want percent may multiply it by 100.
        """
        try:
            start_value, end_value = float(start), float(end)
        except (TypeError, ValueError) as exc:
            raise ValueError("时间区间必须是有限数值") from exc
        if not np.isfinite(start_value) or not np.isfinite(end_value):
            raise ValueError("时间区间必须是有限数值")
        if start_value > end_value:
            raise ValueError("开始时间不能晚于结束时间")
        if channel not in session.channels:
            raise ValueError(f"不存在物理通道 {channel}；可用通道：{session.channels}")

        times_all = np.asarray(session.times, dtype=np.float64)
        display_idx = np.flatnonzero((times_all >= start_value - 1e-9) &
                                     (times_all <= end_value + 1e-9))
        if display_idx.size == 0:
            raise ValueError(f"时间区间没有数据帧：{start_value}..{end_value}")

        self._check_cancel(cancel)
        # Both CNN methods are part of the fixed four-method result, so model
        # loading is done before any frame is evaluated and errors are clear.
        model2, model9 = self._ensure_models()
        c = session.channels.index(channel)
        profiles = np.asarray(session.profiles[:, c], dtype=np.float32)
        valid = np.asarray(session.valid[:, c], dtype=bool)
        gap_all = session.metadata.get("gap_segment")
        gap = np.asarray(gap_all)[:, c] if gap_all is not None else None
        warm_idx = np.flatnonzero((times_all >= start_value - 1.0 - 1e-9) &
                                  (times_all <= end_value + 1e-9))
        display_set = set(int(i) for i in display_idx)
        tracker = RegionTracker(min_points=self.config.min_region_points,
                                near_zero=self.config.near_zero_points,
                                boundary_confirm_frames=self.config.direction_confirmation_frames,
                                max_gap_seconds=self.config.max_gap_seconds)
        frames: list[dict[str, Any]] = []
        total = max(1, len(warm_idx))
        self._progress(progress, 0, "准备比较四种方法")
        for row, index in enumerate(warm_idx.tolist()):
            self._check_cancel(cancel)
            frame = evaluate_frame(
                profiles, int(index), times_all, valid=valid, gap_segment=gap,
                model2=model2, model9=model9, tracker=tracker, track_cnn9=True,
                config=self.config,
                input_transform=self._normalize_window,
            )
            frame["methods"]["cnn9"]["_regions_payload"] = frame.get("regions") or {
                "candidates": [], "regions": []
            }
            shared_evidence = (np.isfinite(profiles[index]) & np.isfinite(profiles[index - 1])
                               if index >= 1 and frame["shared_valid"] else np.zeros(957, bool))
            # Regions for NCC/traditional12/CNN2 are intentionally current
            # frame only.  CNN9 is the sole causally tracked method.
            for method in ("ncc", "traditional12", "cnn2"):
                result = frame["methods"][method]
                found = find_large_regions(
                    result["direction"], result["spacing"],
                    evidence_mask=shared_evidence,
                    screen_mask=result["screen"], min_points=self.config.min_region_points,
                )
                result["_regions"] = {"candidates": found["candidates"],
                                      "regions": found["filtered"]}
            if index in display_set:
                frames.append(frame)
            self._progress(progress, 100.0 * (row + 1) / total,
                           f"分析帧 {row + 1}/{total}")
        self._check_cancel(cancel)

        n = len(frames)
        raw_all = getattr(session, "raw_frames", None)
        if raw_all is None:
            raw = np.full((n, 997), np.nan, dtype=np.float32)
        else:
            raw = np.asarray(raw_all[display_idx, c], dtype=np.float32).copy()
        methods: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            method_frames = [f["methods"][method] for f in frames]
            regions: list[dict[str, Any]] = []
            for result in method_frames:
                if method == "cnn9":
                    tracked = result.get("_regions_payload")
                    # Set below from evaluate_frame's tracker payload.  Keep
                    # unknown warmup frames as an explicit empty payload.
                    payload = tracked or {"candidates": [], "regions": []}
                    regions.append({"candidates": payload.get("candidates", []),
                                    "regions": payload.get("regions", [])})
                else:
                    payload = result.get("_regions", {"candidates": [], "regions": []})
                    regions.append({"candidates": payload["candidates"],
                                    "regions": payload["regions"]})
            methods[method] = {
                "direction": np.stack([r["direction"] for r in method_frames]).astype(np.float32),
                "spacing": np.stack([r["spacing"] for r in method_frames]).astype(np.float32),
                "screen": np.stack([r["screen"] for r in method_frames]).astype(bool),
                "regions": regions,
            }
        self._progress(progress, 100, "比较完成")
        metadata = dict(session.metadata)
        metadata.update({
            "methods": dict(METHOD_LABELS),
            "frame_period_seconds": self.config.pair_interval_seconds,
            "common_support": [self.config.common_start, self.config.common_stop],
            "history_frames": 9,
            "warmup_seconds": 1.0,
            "region_tracking": {
                "cnn9": "因果跟踪，连续3帧确认方向",
                "ncc": "未跟踪；当前帧 candidates/filtered",
                "traditional12": "未跟踪；当前帧 candidates/filtered",
                "cnn2": "未跟踪；当前帧 candidates/filtered",
            },
            "region_tracking_method": "cnn9",
            "region_untracked_methods": ["ncc", "traditional12", "cnn2"],
        })
        return {
            "times": times_all[display_idx].copy(),
            "raw": raw,
            "depth": np.asarray(session.depth, dtype=np.int64).copy(),
            "methods": methods,
            "session_path": str(session.path),
            "channel": int(channel),
            "start": start_value,
            "end": end_value,
            "metadata": metadata,
        }

    @staticmethod
    def _normalize_window(history: np.ndarray) -> np.ndarray:
        # Import lazily with the same module path used by the reviewed runner.
        try:
            from synthetic import normalize_window
        except ImportError:  # pragma: no cover
            from ..paper_experiments.aus_selfsupervised_motion_20261013.coarse_cnn_pilot.synthetic import normalize_window  # type: ignore
        normalized, _, _ = normalize_window(np.asarray(history, dtype=np.float32))
        return normalized


__all__ = ["METHOD_LABELS", "CancelledError", "load_recording", "ComparisonEngine"]
