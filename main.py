#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

from rdt_gate.adjacent_gate import run_adjacent_gate
from rdt_gate.embedding import extract_embeddings, make_synthetic_embeddings
from rdt_gate.evaluation import compute_metrics, make_report
from rdt_gate.prototype_gate import PrototypeGate
from rdt_gate.video_utils import load_video_clips
from rdt_gate.visualization import plot_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-Slow RDT-Gate vs adjacent similarity demo.")
    parser.add_argument("--video_path", default=None, help="Input video path.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for CSV, plots, and report.")
    parser.add_argument("--clip_seconds", type=float, default=1.0)
    parser.add_argument("--frames_per_clip", type=int, default=8)
    parser.add_argument("--embedding_backend", default="simple", choices=["simple", "clip"])
    parser.add_argument("--warmup_clips", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--tau_silence", type=float, default=0.20)
    parser.add_argument("--tau_suspicious", type=float, default=0.50)
    parser.add_argument("--tau_change_low", type=float, default=0.20)
    parser.add_argument("--tau_change_high", type=float, default=0.50)
    parser.add_argument("--adj_tau_silence", type=float, default=0.20)
    parser.add_argument("--adj_tau_suspicious", type=float, default=0.45)
    parser.add_argument("--init_var_threshold", type=float, default=0.35)
    parser.add_argument("--init_change_threshold", type=float, default=0.45)
    parser.add_argument("--max_wait", type=int, default=3)
    parser.add_argument("--event_start", type=float, default=None)
    parser.add_argument("--event_end", type=float, default=None)
    parser.add_argument("--use_synthetic_demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_synthetic_demo:
        clips, embeddings = make_synthetic_embeddings()
        video_label = "synthetic_demo"
        if args.event_start is None:
            args.event_start = 20.0
        if args.event_end is None:
            args.event_end = 25.0
    else:
        if not args.video_path:
            raise SystemExit("Please provide --video_path or use --use_synthetic_demo.")
        clips = load_video_clips(args.video_path, args.clip_seconds, args.frames_per_clip)
        embeddings = extract_embeddings(clips, args.embedding_backend)
        video_label = args.video_path

    prototype_params = {
        "warmup_clips": args.warmup_clips,
        "alpha": args.alpha,
        "tau_silence": args.tau_silence,
        "tau_suspicious": args.tau_suspicious,
        "tau_change_low": args.tau_change_low,
        "tau_change_high": args.tau_change_high,
        "init_var_threshold": args.init_var_threshold,
        "init_change_threshold": args.init_change_threshold,
        "max_wait": args.max_wait,
    }
    adjacent_params = {
        "adj_tau_silence": args.adj_tau_silence,
        "adj_tau_suspicious": args.adj_tau_suspicious,
    }

    prototype_gate = PrototypeGate(**prototype_params)
    prototype_results = prototype_gate.run(clips, embeddings)
    adjacent_results = run_adjacent_gate(
        clips,
        embeddings,
        adj_tau_silence=args.adj_tau_silence,
        adj_tau_suspicious=args.adj_tau_suspicious,
    )
    metrics = compute_metrics(prototype_results, adjacent_results, args.event_start, args.event_end)

    _write_prototype_csv(os.path.join(args.output_dir, "prototype_results.csv"), prototype_results)
    _write_adjacent_csv(os.path.join(args.output_dir, "adjacent_results.csv"), adjacent_results)
    _write_json(os.path.join(args.output_dir, "metrics.json"), metrics)
    plot_all(
        args.output_dir,
        prototype_results,
        adjacent_results,
        metrics,
        args.tau_silence,
        args.tau_suspicious,
        args.event_start,
        args.event_end,
    )
    report = make_report(
        video_label,
        args.clip_seconds,
        args.frames_per_clip,
        args.embedding_backend,
        prototype_params,
        adjacent_params,
        metrics,
        args.event_start,
        args.event_end,
    )
    with open(os.path.join(args.output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Processed {len(clips)} clips. Results saved to: {args.output_dir}")


def _write_prototype_csv(path: str, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_id", "start_time", "end_time", "change", "deviation", "decision", "prototype_ready"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "start_time": f"{row.start_time:.4f}",
                    "end_time": f"{row.end_time:.4f}",
                    "change": _fmt(row.change),
                    "deviation": _fmt(row.deviation),
                    "decision": row.decision.value,
                    "prototype_ready": row.prototype_ready,
                }
            )


def _write_adjacent_csv(path: str, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "start_time", "end_time", "change", "decision"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "start_time": f"{row.start_time:.4f}",
                    "end_time": f"{row.end_time:.4f}",
                    "change": _fmt(row.change),
                    "decision": row.decision.value,
                }
            )


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
