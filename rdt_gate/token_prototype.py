from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .embedding import l2_normalize
from .prototype_gate import PrototypeGate, PrototypeResult, Signal


@dataclass
class DeviationJudgment:
    frame_id: int
    is_deviated: bool
    deviation_level: str
    trigger_slow_path: bool
    reason: str


@dataclass
class FrameTokenPrototype:
    frame_embeddings: np.ndarray
    prototype_vector: np.ndarray | None
    results: List[PrototypeResult]
    judgments: List[DeviationJudgment]


@dataclass
class _FrameItem:
    clip_id: int
    start_time: float
    end_time: float


def tokens_to_frame_embeddings(
    frame_tokens: Sequence[torch.Tensor],
    pool: str = "mean",
    exclude_cls: bool = True,
) -> np.ndarray:
    """Pool per-frame visual tokens into one normalized vector per frame.

    Input shape per frame is usually:
    - simple backend: [num_tiles, 1 + patch_tokens, dim]
    - LiveStar backend: [num_tiles, merged_tokens, dim]

    The output can be fed directly to PrototypeGate.
    """
    embeddings = []
    for tokens in frame_tokens:
        if tokens.ndim != 3:
            raise ValueError(f"Expected frame tokens with shape [tiles, tokens, dim], got {tuple(tokens.shape)}")
        values = tokens.detach().float().cpu()
        if pool == "cls":
            pooled = values[:, 0, :].mean(dim=0)
        elif pool == "mean":
            if exclude_cls and values.shape[1] > 1:
                values = values[:, 1:, :]
            pooled = values.mean(dim=(0, 1))
        else:
            raise ValueError(f"Unsupported token pool strategy: {pool}")
        embeddings.append(l2_normalize(pooled.numpy()))

    if not embeddings:
        raise ValueError("Cannot build prototype embeddings from an empty token sequence.")
    return np.stack(embeddings, axis=0)


def build_frame_token_prototype(
    frame_tokens: Sequence[torch.Tensor],
    frame_times: Sequence[float],
    sample_fps: float,
    pool: str = "mean",
    exclude_cls: bool = True,
    warmup_frames: int = 4,
    alpha: float = 0.9,
    tau_silence: float = 0.03,
    tau_suspicious: float = 0.08,
    tau_change_low: float = 0.01,
    tau_change_high: float = 0.013,
    init_var_threshold: float = 0.10,
    init_change_threshold: float = 0.10,
    max_wait: int = 99,
) -> FrameTokenPrototype:
    frame_embeddings = tokens_to_frame_embeddings(frame_tokens, pool=pool, exclude_cls=exclude_cls)
    frame_items = _make_frame_items(frame_times, sample_fps)

    gate = PrototypeGate(
        warmup_clips=warmup_frames,
        alpha=alpha,
        tau_silence=tau_silence,
        tau_suspicious=tau_suspicious,
        tau_change_low=tau_change_low,
        tau_change_high=tau_change_high,
        init_var_threshold=init_var_threshold,
        init_change_threshold=init_change_threshold,
        max_wait=max_wait,
    )
    results = gate.run(frame_items, frame_embeddings)
    judgments = judge_deviation_results(
        results,
        tau_silence=tau_silence,
        tau_suspicious=tau_suspicious,
        tau_change_low=tau_change_low,
        tau_change_high=tau_change_high,
    )
    prototype_vector = None if gate.prototype is None else gate.prototype.copy()
    return FrameTokenPrototype(
        frame_embeddings=frame_embeddings,
        prototype_vector=prototype_vector,
        results=results,
        judgments=judgments,
    )


def judge_deviation_results(
    results: Sequence[PrototypeResult],
    tau_silence: float,
    tau_suspicious: float,
    tau_change_low: float,
    tau_change_high: float,
) -> List[DeviationJudgment]:
    return [
        judge_deviation(
            row,
            tau_silence=tau_silence,
            tau_suspicious=tau_suspicious,
            tau_change_low=tau_change_low,
            tau_change_high=tau_change_high,
        )
        for row in results
    ]


def judge_deviation(
    row: PrototypeResult,
    tau_silence: float,
    tau_suspicious: float,
    tau_change_low: float,
    tau_change_high: float,
) -> DeviationJudgment:
    """Make the deviation decision explicit and explainable.

    ``is_deviated`` is strictly about distance from the routine prototype.
    ``trigger_slow_path`` mirrors the gate's final SUSPICIOUS decision, which
    can be caused by prototype deviation, adjacent change, or wait timeout.
    """
    reasons = []
    deviation = row.deviation
    change = row.change

    if not row.prototype_ready or deviation is None:
        deviation_level = "not_ready"
        is_deviated = False
        reasons.append("prototype_not_ready")
    elif deviation >= tau_suspicious:
        deviation_level = "deviated"
        is_deviated = True
        reasons.append("deviation_over_tau_suspicious")
    elif deviation <= tau_silence:
        deviation_level = "normal"
        is_deviated = False
        reasons.append("deviation_below_tau_silence")
    else:
        deviation_level = "borderline"
        is_deviated = False
        reasons.append("deviation_between_thresholds")

    if change is None:
        reasons.append("change_not_available")
    elif change >= tau_change_high:
        reasons.append("change_over_tau_high")
    elif change <= tau_change_low:
        reasons.append("change_below_tau_low")
    else:
        reasons.append("change_between_thresholds")

    trigger_slow_path = row.decision == Signal.SUSPICIOUS
    if trigger_slow_path and not is_deviated and not any("change_over" in reason for reason in reasons):
        reasons.append("suspicious_by_wait_or_gate_policy")

    return DeviationJudgment(
        frame_id=row.clip_id,
        is_deviated=is_deviated,
        deviation_level=deviation_level,
        trigger_slow_path=trigger_slow_path,
        reason=";".join(reasons),
    )


def _make_frame_items(frame_times: Sequence[float], sample_fps: float) -> List[_FrameItem]:
    duration = 1.0 / sample_fps if sample_fps > 0 else 0.0
    items = []
    for index, start_time in enumerate(frame_times):
        if index + 1 < len(frame_times):
            end_time = float(frame_times[index + 1])
        else:
            end_time = float(start_time + duration)
        items.append(_FrameItem(index, float(start_time), end_time))
    return items
