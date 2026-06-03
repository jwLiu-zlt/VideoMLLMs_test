from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Signal(str, Enum):
    """Prototype gate output signal.

    SILENCE means the current item is close to routine memory.
    WAIT means the gate needs more context before making a strong decision.
    SUSPICIOUS means the current item should enter the slow path.
    """

    SILENCE = "SILENCE"
    WAIT = "WAIT"
    SUSPICIOUS = "SUSPICIOUS"


@dataclass
class GateDecision:
    """Unified decision record for CSV, reports, and slow-path routing.

    Keep this structure stable so downstream code does not need to rebuild
    ad-hoc dictionaries for every output surface.
    """

    item_id: int
    start_time: float
    end_time: float
    change: float | None
    min_deviation: float | None
    matched_prototype_id: int | None
    decision: Signal
    prototype_ready: bool
    trigger_slow_path: bool
    deviation_level: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    slow_path_text: str | None = None
