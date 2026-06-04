from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


class StableAdaptiveThreshold:
    def __init__(
        self,
        window_size: int = 32,
        alpha_high: float = 3.0,
        min_thr: float = 0.02,
        max_thr: float = 0.25,
        warmup: int = 8,
        min_interval: int = 5,
        confirm_hits: int = 1,
        eps: float = 1e-6,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if warmup <= 0:
            raise ValueError("warmup must be positive")
        if warmup > window_size:
            raise ValueError("warmup must be <= window_size")
        if min_thr >= max_thr:
            raise ValueError("min_thr must be smaller than max_thr")
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if confirm_hits <= 0:
            raise ValueError("confirm_hits must be positive")

        self.window_size = window_size
        self.alpha_high = alpha_high
        self.min_thr = min_thr
        self.max_thr = max_thr
        self.warmup = warmup
        self.min_interval = min_interval
        self.confirm_hits = confirm_hits
        self.eps = eps

        self.window: Deque[float] = deque(maxlen=window_size)
        self.t = 0
        self.last_trigger_t = -10**9
        self.over_threshold_count = 0

    def reset(self) -> None:
        self.window.clear()
        self.t = 0
        self.last_trigger_t = -10**9
        self.over_threshold_count = 0

    def compute_thresholds(self) -> float:
        values = np.asarray(self.window, dtype=np.float32)

        median = np.median(values)
        mad = np.median(np.abs(values - median)) + self.eps

        threshold = median + self.alpha_high * mad
        threshold = float(np.clip(threshold, self.min_thr, self.max_thr))

        return threshold

    def update(self, offset: float) -> dict:
        self.t += 1
        offset = float(offset)
        self.window.append(offset)

        if len(self.window) < self.warmup:
            return {
                "trigger": False,
                "state": "stable",
                "threshold": self.max_thr,
                "offset": offset,
                "over_threshold": False,
                "confirmed": False,
                "cooldown": False,
                "t": self.t,
            }

        threshold = self.compute_thresholds()

        over_threshold = offset > threshold
        if over_threshold:
            self.over_threshold_count += 1
        else:
            self.over_threshold_count = 0

        confirmed = self.over_threshold_count >= self.confirm_hits
        can_trigger = (self.t - self.last_trigger_t) >= self.min_interval
        trigger = over_threshold and confirmed and can_trigger
        if trigger:
            self.last_trigger_t = self.t
            self.over_threshold_count = 0

        return {
            "trigger": trigger,
            "state": "stable",
            "threshold": threshold,
            "offset": offset,
            "over_threshold": over_threshold,
            "confirmed": confirmed,
            "cooldown": over_threshold and not can_trigger,
            "t": self.t,
        }
