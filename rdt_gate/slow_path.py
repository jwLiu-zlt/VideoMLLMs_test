from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .decision_schema import GateDecision, Signal


@dataclass
class SlowPathRequest:
    """Small, explicit request passed to a slow-path model runner."""

    video_path: str
    frame_indices: list[int]
    frame_times: list[float]
    prompt: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlowPathResult:
    item_id: int
    success: bool
    text: str | None
    warning: str | None = None


def should_trigger_slow_path(decision: GateDecision) -> bool:
    return decision.decision == Signal.SUSPICIOUS or decision.trigger_slow_path


def build_slow_path_prompt(decision: GateDecision) -> str:
    return (
        "你是一个流式视频理解助手。当前片段被原型向量 gate 判断为可能偏离常规模式。"
        "请结合当前帧内容，简洁说明发生了什么变化，是否值得响应。\n"
        f"Gate reason: {decision.reason}\n"
        f"Deviation: {decision.min_deviation}\n"
        f"Matched prototype: {decision.matched_prototype_id}\n"
    )


def run_slow_path_safely(request: SlowPathRequest, runner: Callable[[SlowPathRequest], str]) -> SlowPathResult:
    """Run a slow-path model without breaking the fast gate on failure."""

    item_id = int(request.evidence.get("item_id", -1))
    try:
        return SlowPathResult(item_id=item_id, success=True, text=runner(request))
    except Exception as exc:
        return SlowPathResult(item_id=item_id, success=False, text=None, warning=f"slow_path_failed: {exc}")
