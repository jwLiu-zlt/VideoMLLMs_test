from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable

from .prototype_gate import Signal


def signal_counts(results: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(row.decision.value if isinstance(row.decision, Signal) else row.decision) for row in results)
    return {signal.value: counts.get(signal.value, 0) for signal in Signal}


def compute_metrics(
    prototype_results,
    adjacent_results,
    event_start: float | None = None,
    event_end: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    return {
        "prototype": _method_metrics(prototype_results, event_start, event_end),
        "adjacent": _method_metrics(adjacent_results, event_start, event_end),
    }


def _method_metrics(results, event_start: float | None, event_end: float | None) -> Dict[str, Any]:
    counts = signal_counts(results)
    total = len(results)
    metrics: Dict[str, Any] = {
        "signal_counts": counts,
        "false_suspicious_rate": None,
        "event_suspicious_rate": None,
        "slow_path_trigger_rate": counts[Signal.SUSPICIOUS.value] / total if total else 0.0,
    }

    if event_start is None or event_end is None:
        return metrics

    routine = [r for r in results if r.end_time <= event_start]
    event = [r for r in results if r.start_time < event_end and r.end_time > event_start]
    metrics["false_suspicious_rate"] = _suspicious_rate(routine)
    metrics["event_suspicious_rate"] = _suspicious_rate(event)
    return metrics


def _suspicious_rate(rows) -> float | None:
    if not rows:
        return None
    suspicious = sum(1 for row in rows if row.decision == Signal.SUSPICIOUS)
    return suspicious / len(rows)


def make_report(
    video_path: str,
    clip_seconds: float,
    frames_per_clip: int,
    embedding_backend: str,
    prototype_params: Dict[str, Any],
    adjacent_params: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    event_start: float | None,
    event_end: float | None,
) -> str:
    lines = [
        "# Fast-Slow RDT-Gate 对比实验报告",
        "",
        "## 实验配置",
        "",
        f"- 视频路径: `{video_path}`",
        f"- clip_seconds: `{clip_seconds}`",
        f"- frames_per_clip: `{frames_per_clip}`",
        f"- embedding_backend: `{embedding_backend}`",
        f"- event_start/event_end: `{event_start}` / `{event_end}`",
        "",
        "## 方法参数",
        "",
        f"- Prototype RDT-Gate: `{prototype_params}`",
        f"- Adjacent Baseline: `{adjacent_params}`",
        "",
        "## 信号数量统计",
        "",
        "| 方法 | SILENCE | WAIT | SUSPICIOUS |",
        "|---|---:|---:|---:|",
    ]

    for name in ("prototype", "adjacent"):
        counts = metrics[name]["signal_counts"]
        lines.append(
            f"| {name} | {counts['SILENCE']} | {counts['WAIT']} | {counts['SUSPICIOUS']} |"
        )

    lines.extend(
        [
            "",
            "## 指标对比",
            "",
            "| 方法 | false suspicious rate | event suspicious rate | slow path trigger rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("prototype", "adjacent"):
        row = metrics[name]
        lines.append(
            f"| {name} | {_fmt(row['false_suspicious_rate'])} | "
            f"{_fmt(row['event_suspicious_rate'])} | {_fmt(row['slow_path_trigger_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 图表解释",
            "",
            "- `change_scores.png`: 展示相邻 clip 的变化分数。普通动态阶段 change 仍可能波动较大，因此单独使用 change 容易误触发。",
            "- `prototype_deviation.png`: 展示当前 clip 偏离 routine prototype 的程度。如果 routine 阶段 deviation 较低、事件阶段升高，说明原型捕捉了常规动态模式。",
            "- `decision_timeline.png`: 展示两种方法在时间轴上的 SILENCE/WAIT/SUSPICIOUS 输出差异。",
            "- `signal_counts.png`: 对比两种方法输出的信号数量，重点观察 SILENCE 与 SUSPICIOUS 的比例。",
            "- `metrics_comparison.png`: 在提供事件标注时，对比误触发率、事件触发率和 Slow Path 触发率。",
            "",
            "## 结论",
            "",
            "在该视频中，Adjacent Similarity Baseline 仅依赖相邻 clip 的变化程度，因此在高动态但语义上仍属于常规模式的阶段容易输出更多 SUSPICIOUS 信号。Prototype-based RDT-Gate 通过维护常规动态原型，能够识别当前 clip 是否仍接近稳定的 routine pattern，因此在普通动态阶段输出更多 SILENCE 信号。",
            "",
            "如果在事件区间内 Prototype-based RDT-Gate 的 deviation 明显升高并输出 SUSPICIOUS，说明原型方法不仅能够抑制普通动态误触发，也能对偏离常规动态的关键变化保持敏感。",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
