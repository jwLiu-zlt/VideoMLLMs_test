from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List

import numpy as np

from .embedding import cosine_distance, l2_normalize
from utils.adaptive_threshold import StableAdaptiveThreshold


class Signal(str, Enum):
    SILENCE = "SILENCE"
    WAIT = "WAIT"
    SUSPICIOUS = "SUSPICIOUS"


@dataclass
class PrototypeResult:
    clip_id: int
    start_time: float
    end_time: float
    change: float | None
    deviation: float | None
    offset: float | None
    threshold: float | None
    decision: Signal
    prototype_ready: bool


class PrototypeGate:
    def __init__(
        self,
        warmup_clips: int = 5,
        alpha: float = 0.9,
        tau_silence: float | None = None,
        tau_suspicious: float | None = None,
        tau_change_low: float | None = None,
        tau_change_high: float | None = None,
        default_tau_silence: float = 0.03,
        default_tau_suspicious: float = 0.08,
        default_tau_change_low: float = 0.01,
        default_tau_change_high: float = 0.013,
        tau_margin: float = 1e-3,
        init_var_threshold: float = 0.35,
        init_change_threshold: float = 0.45,
        max_wait: int = 3,
        adaptive_window_size: int = 32,
        adaptive_alpha_high: float = 3.0,
        adaptive_min_thr: float = 0.02,
        adaptive_max_thr: float = 0.25,
        adaptive_warmup: int = 8,
        adaptive_min_interval: int = 5,
        adaptive_confirm_hits: int = 1,
        change_weight: float = 1.0,
        update_threshold_ratio: float = 0.7,
    ) -> None:
        if change_weight < 0:
            raise ValueError("change_weight must be non-negative")
        if not 0.0 <= update_threshold_ratio <= 1.0:
            raise ValueError("update_threshold_ratio must be in [0, 1]")

        self.warmup_clips = warmup_clips
        self.alpha = alpha
        self.tau_silence = tau_silence
        self.tau_suspicious = tau_suspicious
        self.tau_change_low = tau_change_low
        self.tau_change_high = tau_change_high
        self.default_tau_silence = default_tau_silence
        self.default_tau_suspicious = default_tau_suspicious
        self.default_tau_change_low = default_tau_change_low
        self.default_tau_change_high = default_tau_change_high
        self.tau_margin = tau_margin
        self.init_var_threshold = init_var_threshold
        self.init_change_threshold = init_change_threshold
        self.max_wait = max_wait
        self.change_weight = change_weight
        self.update_threshold_ratio = update_threshold_ratio
        self.prototype: np.ndarray | None = None
        self.buffer: Deque[np.ndarray] = deque(maxlen=warmup_clips)
        self.wait_count = 0
        self.threshold = StableAdaptiveThreshold(
            window_size=adaptive_window_size,
            alpha_high=adaptive_alpha_high,
            min_thr=adaptive_min_thr,
            max_thr=adaptive_max_thr,
            warmup=adaptive_warmup,
            min_interval=adaptive_min_interval,
            confirm_hits=adaptive_confirm_hits,
        )

    @property
    def prototype_ready(self) -> bool:
        return self.prototype is not None

    def effective_thresholds(self) -> dict[str, float]:
        threshold = self.threshold.compute_thresholds() if self.threshold.window else self.threshold.max_thr
        return {
            "tau_silence": threshold,
            "tau_suspicious": threshold,
            "tau_change_low": self.tau_change_low,
            "tau_change_high": self.tau_change_high,
        }

    def run(self, clips, embeddings: np.ndarray) -> List[PrototypeResult]:
        results: List[PrototypeResult] = []
        previous: np.ndarray | None = None

        for clip, embedding in zip(clips, embeddings):
            z = l2_normalize(embedding)
            change = cosine_distance(z, previous)

            if not self.prototype_ready:
                self.buffer.append(z)
                self._try_initialize()
                decision = Signal.WAIT
                deviation = cosine_distance(z, self.prototype) if self.prototype_ready else None
                offset = None
                threshold = None
            else:
                deviation = cosine_distance(z, self.prototype)
                offset = self._combined_offset(deviation, change)
                threshold_result = self.threshold.update(offset)
                threshold = threshold_result["threshold"]
                if threshold_result["trigger"]:
                    decision = Signal.SUSPICIOUS
                elif len(self.threshold.window) < self.threshold.warmup:
                    decision = Signal.WAIT
                elif threshold_result.get("over_threshold"):
                    decision = Signal.WAIT
                else:
                    decision = Signal.SILENCE

                if decision == Signal.WAIT:
                    self.wait_count += 1
                    if self.wait_count > self.max_wait:
                        decision = Signal.SUSPICIOUS
                else:
                    self.wait_count = 0

                if (
                    decision == Signal.SILENCE
                    and threshold is not None
                    and offset < self.update_threshold_ratio * threshold
                ):
                    self.prototype = l2_normalize(self.alpha * self.prototype + (1.0 - self.alpha) * z)

            results.append(
                PrototypeResult(
                    clip_id=clip.clip_id,
                    start_time=clip.start_time,
                    end_time=clip.end_time,
                    change=change,
                    deviation=deviation,
                    offset=offset,
                    threshold=threshold,
                    decision=decision,
                    prototype_ready=self.prototype_ready,
                )
            )
            previous = z

        return results

    def _try_initialize(self) -> None:
        if self.prototype_ready or len(self.buffer) < self.warmup_clips:
            return
        vectors = np.stack(list(self.buffer), axis=0)
        center = l2_normalize(vectors.mean(axis=0))
        variance = float(np.mean([cosine_distance(v, center) for v in vectors]))
        changes = [cosine_distance(vectors[i], vectors[i - 1]) for i in range(1, len(vectors))]
        avg_change = float(np.mean(changes)) if changes else 0.0
        if variance < self.init_var_threshold and avg_change < self.init_change_threshold:
            self.prototype = center

    def _combined_offset(self, deviation: float | None, change: float | None) -> float:
        deviation_value = 0.0 if deviation is None else deviation
        change_value = 0.0 if change is None else self.change_weight * change
        return float(max(deviation_value, change_value))
