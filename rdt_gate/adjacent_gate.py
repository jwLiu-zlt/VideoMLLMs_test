from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .embedding import cosine_distance, l2_normalize
from .prototype_gate import Signal


@dataclass
class AdjacentResult:
    clip_id: int
    start_time: float
    end_time: float
    change: float | None
    decision: Signal


def run_adjacent_gate(
    clips,
    embeddings: np.ndarray,
    adj_tau_silence: float = 0.20,
    adj_tau_suspicious: float = 0.45,
) -> List[AdjacentResult]:
    results: List[AdjacentResult] = []
    previous: np.ndarray | None = None

    for clip, embedding in zip(clips, embeddings):
        z = l2_normalize(embedding)
        change = cosine_distance(z, previous)
        if previous is None:
            decision = Signal.WAIT
        elif change is not None and change < adj_tau_silence:
            decision = Signal.SILENCE
        elif change is not None and change > adj_tau_suspicious:
            decision = Signal.SUSPICIOUS
        else:
            decision = Signal.WAIT

        results.append(
            AdjacentResult(
                clip_id=clip.clip_id,
                start_time=clip.start_time,
                end_time=clip.end_time,
                change=change,
                decision=decision,
            )
        )
        previous = z

    return results
