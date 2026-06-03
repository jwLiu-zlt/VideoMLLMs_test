from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence

import numpy as np

from .decision_schema import GateDecision, Signal
from .embedding import cosine_distance, l2_normalize
from .token_aggregator import AggregatedTokenItem


@dataclass
class PrototypeSlot:
    """A single routine prototype in the memory bank."""

    vector: np.ndarray
    count: int
    last_update_id: int
    stable_score: float = 1.0


class PrototypeBankGate:
    """Multi-prototype routine gate with adaptive thresholds.

    Warmup items initialize a small prototype bank. During online inference,
    each item is compared against the nearest prototype. Only high-confidence
    SILENCE items update the bank, which keeps suspicious content from
    polluting routine memory.
    """

    def __init__(
        self,
        warmup_items: int = 8,
        max_prototypes: int = 4,
        alpha: float = 0.9,
        init_cluster_threshold: float = 0.08,
        tau_silence: float | None = None,
        tau_suspicious: float | None = None,
        tau_change_low: float | None = None,
        tau_change_high: float | None = None,
        default_tau_silence: float = 0.03,
        default_tau_suspicious: float = 0.08,
        default_tau_change_low: float = 0.01,
        default_tau_change_high: float = 0.013,
        max_wait: int = 5,
        cooldown_items: int = 3,
    ) -> None:
        if warmup_items <= 0:
            raise ValueError("warmup_items must be positive.")
        if max_prototypes <= 0:
            raise ValueError("max_prototypes must be positive.")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1).")

        self.warmup_items = warmup_items
        self.max_prototypes = max_prototypes
        self.alpha = alpha
        self.init_cluster_threshold = init_cluster_threshold
        self.max_wait = max_wait
        self.cooldown_items = cooldown_items

        self.tau_silence = tau_silence
        self.tau_suspicious = tau_suspicious
        self.tau_change_low = tau_change_low
        self.tau_change_high = tau_change_high

        self.default_tau_silence = default_tau_silence
        self.default_tau_suspicious = default_tau_suspicious
        self.default_tau_change_low = default_tau_change_low
        self.default_tau_change_high = default_tau_change_high

        self.prototype_bank: list[PrototypeSlot] = []
        self.buffer: Deque[np.ndarray] = deque(maxlen=warmup_items)
        self.wait_count = 0
        self.cooldown_remaining = 0

    @property
    def prototype_ready(self) -> bool:
        return bool(self.prototype_bank)

    @property
    def prototype_vectors(self) -> np.ndarray | None:
        if not self.prototype_bank:
            return None
        return np.stack([slot.vector.copy() for slot in self.prototype_bank], axis=0)

    def run(self, items: Sequence[AggregatedTokenItem]) -> list[GateDecision]:
        results: list[GateDecision] = []
        previous: np.ndarray | None = None
        for item in items:
            decision = self.step(item, previous)
            results.append(decision)
            previous = l2_normalize(item.embedding)
        return results

    def step(self, item: AggregatedTokenItem, previous: np.ndarray | None = None) -> GateDecision:
        z = l2_normalize(item.embedding)
        change = cosine_distance(z, previous)

        if not self.prototype_ready:
            self.buffer.append(z)
            self._try_initialize()
            matched_id = None
            deviation = None
            if self.prototype_ready:
                matched_id, deviation = self._match_nearest_prototype(z)
            return GateDecision(
                item_id=item.item_id,
                start_time=item.start_time,
                end_time=item.end_time,
                change=change,
                min_deviation=deviation,
                matched_prototype_id=matched_id,
                decision=Signal.WAIT,
                prototype_ready=self.prototype_ready,
                trigger_slow_path=False,
                deviation_level="warmup_ready" if self.prototype_ready else "not_ready",
                reason="prototype_initialized" if self.prototype_ready else "prototype_not_ready",
                evidence=item.evidence,
            )

        matched_id, deviation = self._match_nearest_prototype(z)
        decision, level, reason = self._decide(deviation, change)
        trigger_slow_path = decision == Signal.SUSPICIOUS

        if decision == Signal.SUSPICIOUS:
            self.cooldown_remaining = self.cooldown_items
            self.wait_count = 0
        elif decision == Signal.WAIT:
            self.wait_count += 1
            if self.wait_count > self.max_wait:
                decision = Signal.SUSPICIOUS
                trigger_slow_path = True
                reason += ";wait_timeout"
                self.cooldown_remaining = self.cooldown_items
        else:
            self.wait_count = 0

        self._maybe_update(matched_id, z, deviation, decision, item.item_id)

        return GateDecision(
            item_id=item.item_id,
            start_time=item.start_time,
            end_time=item.end_time,
            change=change,
            min_deviation=deviation,
            matched_prototype_id=matched_id,
            decision=decision,
            prototype_ready=True,
            trigger_slow_path=trigger_slow_path,
            deviation_level=level,
            reason=reason,
            evidence=item.evidence,
        )

    def _try_initialize(self) -> None:
        if self.prototype_ready or len(self.buffer) < self.warmup_items:
            return

        vectors = [l2_normalize(v) for v in self.buffer]
        self._calibrate_thresholds(vectors)
        for item_id, vector in enumerate(vectors):
            self._add_or_merge_initial_prototype(vector, item_id)

    def _add_or_merge_initial_prototype(self, vector: np.ndarray, item_id: int) -> None:
        if not self.prototype_bank:
            self.prototype_bank.append(PrototypeSlot(vector=vector, count=1, last_update_id=item_id))
            return

        matched_id, deviation = self._match_nearest_prototype(vector)
        if deviation <= self.init_cluster_threshold or len(self.prototype_bank) >= self.max_prototypes:
            slot = self.prototype_bank[matched_id]
            weight = 1.0 / float(slot.count + 1)
            slot.vector = l2_normalize((1.0 - weight) * slot.vector + weight * vector)
            slot.count += 1
            slot.last_update_id = item_id
        else:
            self.prototype_bank.append(PrototypeSlot(vector=vector, count=1, last_update_id=item_id))

    def _calibrate_thresholds(self, vectors: list[np.ndarray]) -> None:
        center = l2_normalize(np.stack(vectors, axis=0).mean(axis=0))
        deviations = np.asarray([cosine_distance(v, center) or 0.0 for v in vectors], dtype=np.float32)
        changes = np.asarray(
            [cosine_distance(vectors[i], vectors[i - 1]) or 0.0 for i in range(1, len(vectors))],
            dtype=np.float32,
        )

        if self.tau_silence is None:
            self.tau_silence = self._safe_quantile(deviations, 0.75, self.default_tau_silence)
        if self.tau_suspicious is None:
            self.tau_suspicious = self._safe_quantile(deviations, 0.95, self.default_tau_suspicious) + 1e-3
        if self.tau_change_low is None:
            self.tau_change_low = self._safe_quantile(changes, 0.50, self.default_tau_change_low)
        if self.tau_change_high is None:
            self.tau_change_high = self._safe_quantile(changes, 0.95, self.default_tau_change_high) + 1e-3

        if self.tau_suspicious <= self.tau_silence:
            self.tau_suspicious = self.tau_silence + 1e-3
        if self.tau_change_high <= self.tau_change_low:
            self.tau_change_high = self.tau_change_low + 1e-3

    def _safe_quantile(self, values: np.ndarray, q: float, default: float) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return default
        value = float(np.quantile(values, q))
        return value if np.isfinite(value) else default

    def _match_nearest_prototype(self, z: np.ndarray) -> tuple[int, float]:
        distances = [cosine_distance(z, slot.vector) or 0.0 for slot in self.prototype_bank]
        matched_id = int(np.argmin(distances))
        return matched_id, float(distances[matched_id])

    def _decide(self, deviation: float, change: float | None) -> tuple[Signal, str, str]:
        assert self.tau_silence is not None
        assert self.tau_suspicious is not None
        assert self.tau_change_low is not None
        assert self.tau_change_high is not None

        reasons: list[str] = []
        if deviation <= self.tau_silence:
            level = "normal"
            reasons.append("deviation_below_tau_silence")
        elif deviation >= self.tau_suspicious:
            level = "deviated"
            reasons.append("deviation_over_tau_suspicious")
        else:
            level = "borderline"
            reasons.append("deviation_between_thresholds")

        if change is None:
            reasons.append("change_not_available")
        elif change >= self.tau_change_high:
            reasons.append("change_over_tau_high")
        elif change <= self.tau_change_low:
            reasons.append("change_below_tau_low")
        else:
            reasons.append("change_between_thresholds")

        if deviation >= self.tau_suspicious or (change is not None and change >= self.tau_change_high):
            return Signal.SUSPICIOUS, level, ";".join(reasons)
        if deviation <= self.tau_silence and (change is None or change <= self.tau_change_low):
            return Signal.SILENCE, level, ";".join(reasons)
        return Signal.WAIT, level, ";".join(reasons)

    def _maybe_update(
        self,
        matched_id: int,
        z: np.ndarray,
        deviation: float,
        decision: Signal,
        item_id: int,
    ) -> None:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return
        if decision != Signal.SILENCE:
            return
        if self.tau_silence is None or deviation > self.tau_silence:
            return

        if deviation <= self.tau_silence * 0.5:
            weight = 1.0 - self.alpha
        else:
            weight = 0.5 * (1.0 - self.alpha)
        if weight <= 0.0:
            return

        slot = self.prototype_bank[matched_id]
        slot.vector = l2_normalize((1.0 - weight) * slot.vector + weight * z)
        slot.count += 1
        slot.last_update_id = item_id
