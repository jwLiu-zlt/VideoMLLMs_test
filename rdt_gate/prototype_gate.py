from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List

import numpy as np

from .embedding import cosine_distance, l2_normalize


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
    decision: Signal
    prototype_ready: bool


class PrototypeGate:
    def __init__(
        self,
        warmup_clips: int = 5,
        alpha: float = 0.9,
        tau_silence: float = 0.20,
        tau_suspicious: float = 0.50,
        tau_change_low: float = 0.20,
        tau_change_high: float = 0.50,
        init_var_threshold: float = 0.35,
        init_change_threshold: float = 0.45,
        max_wait: int = 3,
    ) -> None:
        self.warmup_clips = warmup_clips
        self.alpha = alpha
        self.tau_silence = tau_silence
        self.tau_suspicious = tau_suspicious
        self.tau_change_low = tau_change_low
        self.tau_change_high = tau_change_high
        self.init_var_threshold = init_var_threshold
        self.init_change_threshold = init_change_threshold
        self.max_wait = max_wait
        self.prototype: np.ndarray | None = None
        self.buffer: Deque[np.ndarray] = deque(maxlen=warmup_clips)
        self.wait_count = 0

    @property
    def prototype_ready(self) -> bool:
        return self.prototype is not None

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
            else:
                deviation = cosine_distance(z, self.prototype)
                if (
                    deviation is not None
                    and change is not None
                    and deviation < self.tau_silence
                    and change < self.tau_change_low
                ):
                    decision = Signal.SILENCE
                elif (
                    deviation is not None
                    and change is not None
                    and (deviation > self.tau_suspicious or change > self.tau_change_high)
                ):
                    decision = Signal.SUSPICIOUS
                else:
                    decision = Signal.WAIT

                if decision == Signal.WAIT:
                    self.wait_count += 1
                    if self.wait_count > self.max_wait:
                        decision = Signal.SUSPICIOUS
                else:
                    self.wait_count = 0

                if decision == Signal.SILENCE:
                    self.prototype = l2_normalize(self.alpha * self.prototype + (1.0 - self.alpha) * z)

            results.append(
                PrototypeResult(
                    clip_id=clip.clip_id,
                    start_time=clip.start_time,
                    end_time=clip.end_time,
                    change=change,
                    deviation=deviation,
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
