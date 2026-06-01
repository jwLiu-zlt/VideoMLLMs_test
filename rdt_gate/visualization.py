from __future__ import annotations

import os
import tempfile
from typing import Dict

_cache_root = tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_root, "rdt_gate_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_cache_root, "rdt_gate_cache"))
os.environ.setdefault("FONTCONFIG_CACHE", os.path.join(_cache_root, "rdt_gate_fontconfig"))
for cache_dir in (
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    os.environ["FONTCONFIG_CACHE"],
):
    os.makedirs(cache_dir, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from .prototype_gate import Signal


SIGNAL_COLOR = {
    Signal.SILENCE.value: "#2ca02c",
    Signal.WAIT.value: "#ffbf00",
    Signal.SUSPICIOUS.value: "#d62728",
}


def plot_all(
    output_dir: str,
    prototype_results,
    adjacent_results,
    metrics: Dict,
    tau_silence: float,
    tau_suspicious: float,
    event_start: float | None,
    event_end: float | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _plot_change_scores(output_dir, prototype_results, event_start, event_end)
    _plot_deviation(output_dir, prototype_results, tau_silence, tau_suspicious, event_start, event_end)
    _plot_decision_timeline(output_dir, prototype_results, adjacent_results, event_start, event_end)
    _plot_signal_counts(output_dir, metrics)
    if event_start is not None and event_end is not None:
        _plot_metrics(output_dir, metrics)


def _event_span(event_start, event_end) -> None:
    if event_start is not None and event_end is not None:
        plt.axvspan(event_start, event_end, color="red", alpha=0.14, label="event")


def _plot_change_scores(output_dir, prototype_results, event_start, event_end) -> None:
    times = [r.start_time for r in prototype_results]
    change = [np.nan if r.change is None else r.change for r in prototype_results]
    plt.figure(figsize=(10, 4))
    plt.plot(times, change, marker="o", linewidth=1.6, label="change score")
    _event_span(event_start, event_end)
    plt.xlabel("Time (s)")
    plt.ylabel("1 - cosine(z_t, z_t-1)")
    plt.title("Adjacent Change Score Timeline")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "change_scores.png"), dpi=160)
    plt.close()


def _plot_deviation(output_dir, prototype_results, tau_silence, tau_suspicious, event_start, event_end) -> None:
    times = [r.start_time for r in prototype_results]
    deviation = [np.nan if r.deviation is None else r.deviation for r in prototype_results]
    plt.figure(figsize=(10, 4))
    plt.plot(times, deviation, marker="o", linewidth=1.6, label="prototype deviation")
    plt.axhline(tau_silence, color="#2ca02c", linestyle="--", label="tau_silence")
    plt.axhline(tau_suspicious, color="#d62728", linestyle="--", label="tau_suspicious")
    _event_span(event_start, event_end)
    plt.xlabel("Time (s)")
    plt.ylabel("1 - cosine(z_t, P_routine)")
    plt.title("Prototype Deviation Timeline")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prototype_deviation.png"), dpi=160)
    plt.close()


def _plot_decision_timeline(output_dir, prototype_results, adjacent_results, event_start, event_end) -> None:
    plt.figure(figsize=(10, 3.8))
    rows = [("Prototype RDT-Gate", prototype_results, 1), ("Adjacent Baseline", adjacent_results, 0)]
    for label, results, y in rows:
        for result in results:
            decision = result.decision.value
            width = max(0.05, result.end_time - result.start_time)
            plt.barh(y, width, left=result.start_time, height=0.35, color=SIGNAL_COLOR[decision])
    _event_span(event_start, event_end)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=signal) for signal, color in SIGNAL_COLOR.items()]
    plt.yticks([0, 1], ["Adjacent Baseline", "Prototype RDT-Gate"])
    plt.xlabel("Time (s)")
    plt.title("Decision Timeline")
    plt.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_timeline.png"), dpi=160)
    plt.close()


def _plot_signal_counts(output_dir, metrics) -> None:
    labels = [s.value for s in Signal]
    x = np.arange(len(labels))
    width = 0.36
    proto = [metrics["prototype"]["signal_counts"][label] for label in labels]
    adj = [metrics["adjacent"]["signal_counts"][label] for label in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, proto, width, label="Prototype RDT-Gate")
    plt.bar(x + width / 2, adj, width, label="Adjacent Baseline")
    plt.xticks(x, labels)
    plt.ylabel("Clip count")
    plt.title("Signal Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_counts.png"), dpi=160)
    plt.close()


def _plot_metrics(output_dir, metrics) -> None:
    labels = ["false_suspicious_rate", "event_suspicious_rate", "slow_path_trigger_rate"]
    x = np.arange(len(labels))
    width = 0.36
    proto = [metrics["prototype"][label] or 0.0 for label in labels]
    adj = [metrics["adjacent"][label] or 0.0 for label in labels]
    plt.figure(figsize=(9, 4))
    plt.bar(x - width / 2, proto, width, label="Prototype RDT-Gate")
    plt.bar(x + width / 2, adj, width, label="Adjacent Baseline")
    plt.xticks(x, labels, rotation=12)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=160)
    plt.close()
