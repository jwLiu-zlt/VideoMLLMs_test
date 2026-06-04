#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

from rdt_gate.adjacent_gate import run_adjacent_gate
from rdt_gate.embedding import extract_embeddings, make_synthetic_embeddings
from rdt_gate.everos_memory import make_everos_session_id, sync_experiment_run
from rdt_gate.evaluation import compute_metrics, make_report
from rdt_gate.llm_slow_path import LLMSlowPathResult, make_llm_runner, run_llm_slow_path
from rdt_gate.prototype_gate import PrototypeGate
from rdt_gate.video_utils import load_video_clips
from rdt_gate.visualization import plot_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-Slow RDT-Gate demo.")
    parser.add_argument("--config", default=None, help="YAML config file. CLI args override config values.")
    parser.add_argument("--video_path", default="data/1.mp4", help="Input video path.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for CSV, plots, and report. Defaults to outputs/<video_name>/.",
    )
    parser.add_argument("--clip_seconds", type=float, default=0.5)
    parser.add_argument("--frames_per_clip", type=int, default=8)
    parser.add_argument("--embedding_backend", default="simple", choices=["simple", "clip"])
    parser.add_argument("--warmup_clips", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--tau_silence", type=float, default=None)
    parser.add_argument("--tau_suspicious", type=float, default=None)
    parser.add_argument("--tau_change_low", type=float, default=None)
    parser.add_argument("--tau_change_high", type=float, default=None)
    parser.add_argument("--adjacent_enable", action="store_true", help="Run the adjacent similarity baseline.")
    parser.add_argument("--adj_tau_silence", type=float, default=0.01)
    parser.add_argument("--adj_tau_suspicious", type=float, default=0.013)
    parser.add_argument("--init_var_threshold", type=float, default=0.10)
    parser.add_argument("--init_change_threshold", type=float, default=0.10)
    parser.add_argument("--max_wait", type=int, default=99)
    parser.add_argument("--adaptive_window_size", type=int, default=32)
    parser.add_argument("--adaptive_alpha_high", type=float, default=3.0)
    parser.add_argument("--adaptive_min_thr", type=float, default=0.02)
    parser.add_argument("--adaptive_max_thr", type=float, default=0.25)
    parser.add_argument("--adaptive_warmup", type=int, default=8)
    parser.add_argument("--adaptive_min_interval", type=int, default=5)
    parser.add_argument("--adaptive_confirm_hits", type=int, default=2)
    parser.add_argument("--change_weight", type=float, default=1.0)
    parser.add_argument("--update_threshold_ratio", type=float, default=0.7)
    parser.add_argument("--event_start", type=float, default=None)
    parser.add_argument("--event_end", type=float, default=None)
    parser.add_argument(
        "--event_config",
        default="event_annotations.json",
        help="JSON file mapping each video to its event_start/event_end annotation.",
    )
    parser.add_argument("--use_synthetic_demo", action="store_true")
    parser.add_argument("--everos_enable", action="store_true", help="Store this run in EverOS memory.")
    parser.add_argument(
        "--everos_user_id",
        default=os.environ.get("EVEROS_USER_ID", "damo_0526_user"),
        help="EverOS user_id owner for experiment memories.",
    )
    parser.add_argument("--everos_session_id", default=None, help="EverOS session_id for this experiment run.")
    parser.add_argument(
        "--everos_search",
        default=None,
        help="Optional query to retrieve prior EverOS context before saving this run.",
    )
    parser.add_argument("--everos_top_k", type=int, default=5, help="Top K memories for --everos_search.")
    parser.add_argument("--everos_no_flush", action="store_true", help="Do not flush EverOS agent memory after save.")
    parser.add_argument("--llm_enable", action="store_true", help="Run an LLM slow path after gate decisions.")
    parser.add_argument("--llm_backend", default="mock", choices=["mock", "hf"])
    parser.add_argument("--llm_model_path", default=None, help="Local HuggingFace model path/name for --llm_backend hf.")
    parser.add_argument("--llm_device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--llm_max_new_tokens", type=int, default=128)
    parser.add_argument("--llm_trigger_mode", default="suspicious", choices=["suspicious", "all"])
    parser.add_argument(
        "--llm_prompt_template",
        default=(
            "你是一个流式视频理解助手。视频: {video_label}\n"
            "当前 clip={clip_id}, 时间={start_time}-{end_time}s。\n"
            "Gate decision={decision}, change={change}, deviation={deviation}, "
            "offset={offset}, threshold={threshold}。\n"
            "请根据这些 gate 信号判断是否值得响应，并用一句话说明原因。"
        ),
        help=(
            "Prompt template. Available fields: video_label, clip_id, start_time, end_time, "
            "decision, change, deviation, offset, threshold."
        ),
    )
    args = parser.parse_args()
    if args.config:
        args = _merge_yaml_config(args, parser)
    return args


def _merge_yaml_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    config = _load_simple_yaml(args.config)
    defaults = parser.parse_args([])
    merged = vars(args).copy()
    for key, value in config.items():
        if not hasattr(args, key):
            raise SystemExit(f"Unknown config key in {args.config}: {key}")
        if getattr(args, key) == getattr(defaults, key):
            merged[key] = value
    return argparse.Namespace(**merged)


def _load_simple_yaml(path: str) -> dict[str, Any]:
    """Load the small YAML subset used by this demo's configs."""
    if not os.path.exists(path):
        raise SystemExit(f"Config file not found: {path}")

    config: dict[str, Any] = {}
    section_stack: list[tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if ":" not in stripped:
                raise SystemExit(f"Invalid YAML line {line_number} in {path}: {raw_line.rstrip()}")

            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            while section_stack and indent <= section_stack[-1][0]:
                section_stack.pop()

            if raw_value == "":
                section_stack.append((indent, key))
                continue

            full_key = _config_key([section for _, section in section_stack], key)
            config[full_key] = _parse_yaml_scalar(raw_value)

    return config


def _config_key(sections: list[str], key: str) -> str:
    if not sections:
        return key
    section = sections[-1]
    if section in {"prototype", "adaptive", "adjacent", "event", "everos", "data"}:
        return key
    return "_".join(sections + [key])


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = parse_args()
    _resolve_event_annotation(args)
    _resolve_output_dir(args)
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
        "adaptive_window_size": args.adaptive_window_size,
        "adaptive_alpha_high": args.adaptive_alpha_high,
        "adaptive_min_thr": args.adaptive_min_thr,
        "adaptive_max_thr": args.adaptive_max_thr,
        "adaptive_warmup": args.adaptive_warmup,
        "adaptive_min_interval": args.adaptive_min_interval,
        "adaptive_confirm_hits": args.adaptive_confirm_hits,
        "change_weight": args.change_weight,
        "update_threshold_ratio": args.update_threshold_ratio,
    }
    adjacent_params = {
        "adjacent_enable": args.adjacent_enable,
        "adj_tau_silence": args.adj_tau_silence,
        "adj_tau_suspicious": args.adj_tau_suspicious,
    }

    prototype_gate = PrototypeGate(**prototype_params)
    prototype_results = prototype_gate.run(clips, embeddings)
    effective_thresholds = prototype_gate.effective_thresholds()
    prototype_params.update(effective_thresholds)
    adjacent_results = None
    if args.adjacent_enable:
        adjacent_results = run_adjacent_gate(
            clips,
            embeddings,
            adj_tau_silence=args.adj_tau_silence,
            adj_tau_suspicious=args.adj_tau_suspicious,
        )
    llm_results: list[LLMSlowPathResult] = []
    if args.llm_enable:
        llm_runner = make_llm_runner(
            backend=args.llm_backend,
            model_path=args.llm_model_path,
            device=args.llm_device,
            max_new_tokens=args.llm_max_new_tokens,
        )
        llm_results = run_llm_slow_path(
            clips=clips,
            gate_results=prototype_results,
            runner=llm_runner,
            prompt_template=args.llm_prompt_template,
            video_label=video_label,
            trigger_mode=args.llm_trigger_mode,
        )
    metrics = compute_metrics(prototype_results, adjacent_results, args.event_start, args.event_end)

    _write_prototype_csv(os.path.join(args.output_dir, "prototype_results.csv"), prototype_results)
    adjacent_csv = os.path.join(args.output_dir, "adjacent_results.csv")
    if args.adjacent_enable and adjacent_results is not None:
        _write_adjacent_csv(adjacent_csv, adjacent_results)
    elif os.path.exists(adjacent_csv):
        os.remove(adjacent_csv)
    if args.llm_enable:
        _write_llm_csv(os.path.join(args.output_dir, "llm_results.csv"), llm_results)
        _write_json(
            os.path.join(args.output_dir, "llm_results.json"),
            [_llm_result_to_dict(row) for row in llm_results],
        )
    _write_json(os.path.join(args.output_dir, "metrics.json"), metrics)
    plot_all(
        args.output_dir,
        prototype_results,
        adjacent_results,
        metrics,
        effective_thresholds["tau_silence"],
        effective_thresholds["tau_suspicious"],
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
    _write_run_config(os.path.join(args.output_dir, "config.yaml"), args, prototype_params, adjacent_params)

    if args.everos_enable:
        everos_status = _sync_everos(args, video_label, prototype_params, adjacent_params, metrics, report)
        _write_json(os.path.join(args.output_dir, "everos_status.json"), everos_status)

    print(f"Processed {len(clips)} clips. Results saved to: {args.output_dir}")


def _resolve_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir:
        return
    if args.use_synthetic_demo:
        name = "synthetic_demo"
    else:
        name = _output_name_for_video(args.video_path)
    args.output_dir = os.path.join("outputs", name)


def _output_name_for_video(video_path: str) -> str:
    base = os.path.basename(video_path)
    stem, ext = os.path.splitext(base)
    raw_name = f"{stem}{ext.lstrip('.')}" if ext else stem
    parts = []
    for char in raw_name.lower():
        parts.append(char if char.isalnum() else "_")
    name = "_".join(part for part in "".join(parts).split("_") if part)
    return name or "video"


def _resolve_event_annotation(args: argparse.Namespace) -> None:
    if args.event_start is not None or args.event_end is not None:
        if args.event_start is None or args.event_end is None:
            raise SystemExit("Please provide both --event_start and --event_end, or neither.")
        if args.event_end <= args.event_start:
            raise SystemExit("--event_end must be greater than --event_start.")
        return

    if args.use_synthetic_demo:
        args.event_start = 20.0
        args.event_end = 25.0
        return

    annotation = _load_video_event_annotation(args.event_config, args.video_path)
    if annotation is None:
        print(
            "No event annotation found for this video. "
            "Event metrics will be null; pass --event_start/--event_end or update "
            f"{args.event_config} to evaluate event recall."
        )
        return

    args.event_start = annotation["event_start"]
    args.event_end = annotation["event_end"]
    print(
        f"Loaded event annotation for {args.video_path}: "
        f"{args.event_start:.4f}s - {args.event_end:.4f}s"
    )


def _load_video_event_annotation(config_path: str, video_path: str) -> dict[str, float] | None:
    if not config_path or not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    candidates = [
        video_path,
        os.path.normpath(video_path),
        os.path.basename(video_path),
        os.path.splitext(os.path.basename(video_path))[0],
        os.path.abspath(video_path),
    ]
    for key in candidates:
        if key in annotations:
            return _parse_event_annotation(annotations[key], key)
    return None


def _parse_event_annotation(value: Any, key: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise SystemExit(f"Invalid event annotation for {key}: expected an object.")
    if "event_start" not in value or "event_end" not in value:
        raise SystemExit(f"Invalid event annotation for {key}: missing event_start/event_end.")
    event_start = float(value["event_start"])
    event_end = float(value["event_end"])
    if event_end <= event_start:
        raise SystemExit(f"Invalid event annotation for {key}: event_end must be greater than event_start.")
    return {"event_start": event_start, "event_end": event_end}


def _write_prototype_csv(path: str, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "start_time",
                "end_time",
                "change",
                "deviation",
                "offset",
                "threshold",
                "decision",
                "prototype_ready",
            ],
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
                    "offset": _fmt(row.offset),
                    "threshold": _fmt(row.threshold),
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


def _write_llm_csv(path: str, rows: list[LLMSlowPathResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "start_time",
                "end_time",
                "decision",
                "success",
                "prompt",
                "response",
                "warning",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_llm_result_to_dict(row))


def _llm_result_to_dict(row: LLMSlowPathResult) -> dict[str, Any]:
    return {
        "clip_id": row.clip_id,
        "start_time": f"{row.start_time:.4f}",
        "end_time": f"{row.end_time:.4f}",
        "decision": row.decision,
        "success": row.success,
        "prompt": row.prompt,
        "response": row.response or "",
        "warning": row.warning or "",
    }


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_run_config(
    path: str,
    args: argparse.Namespace,
    prototype_params: dict[str, Any],
    adjacent_params: dict[str, Any],
) -> None:
    payload = {
        "config": args.config,
        "data": {
            "video_path": args.video_path,
            "output_dir": args.output_dir,
            "clip_seconds": args.clip_seconds,
            "frames_per_clip": args.frames_per_clip,
            "embedding_backend": args.embedding_backend,
            "use_synthetic_demo": args.use_synthetic_demo,
        },
        "prototype": prototype_params,
        "adjacent": adjacent_params,
        "event": {
            "event_start": args.event_start,
            "event_end": args.event_end,
            "event_config": args.event_config,
        },
        "everos": {
            "everos_enable": args.everos_enable,
            "everos_user_id": args.everos_user_id,
            "everos_session_id": args.everos_session_id,
            "everos_search": args.everos_search,
            "everos_top_k": args.everos_top_k,
            "everos_no_flush": args.everos_no_flush,
        },
        "llm": {
            "enable": args.llm_enable,
            "backend": args.llm_backend,
            "model_path": args.llm_model_path,
            "device": args.llm_device,
            "max_new_tokens": args.llm_max_new_tokens,
            "trigger_mode": args.llm_trigger_mode,
            "prompt_template": args.llm_prompt_template,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        _dump_simple_yaml(payload, f)


def _dump_simple_yaml(value: Any, f, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                f.write(f"{prefix}{key}:\n")
                _dump_simple_yaml(item, f, indent + 2)
            else:
                f.write(f"{prefix}{key}: {_format_yaml_scalar(item)}\n")
        return
    f.write(f"{prefix}{_format_yaml_scalar(value)}\n")


def _format_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#{}[]&,*?|-<>=!%@\\\"'"):
        return json.dumps(text, ensure_ascii=False)
    return text


def _sync_everos(
    args: argparse.Namespace,
    video_label: str,
    prototype_params: dict[str, Any],
    adjacent_params: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    report: str,
) -> dict[str, Any]:
    session_id = args.everos_session_id or make_everos_session_id(video_label, args.output_dir)
    try:
        status = sync_experiment_run(
            user_id=args.everos_user_id,
            session_id=session_id,
            video_label=video_label,
            output_dir=args.output_dir,
            clip_seconds=args.clip_seconds,
            frames_per_clip=args.frames_per_clip,
            embedding_backend=args.embedding_backend,
            prototype_params=prototype_params,
            adjacent_params=adjacent_params,
            metrics=metrics,
            report=report,
            search_query=args.everos_search,
            search_top_k=args.everos_top_k,
            flush=not args.everos_no_flush,
        )
        print(f"EverOS memory saved for user_id={args.everos_user_id}, session_id={session_id}.")
        return status
    except Exception as exc:
        message = f"EverOS sync failed: {exc}"
        print(message)
        return {
            "enabled": True,
            "user_id": args.everos_user_id,
            "session_id": session_id,
            "saved": False,
            "flushed": False,
            "warnings": [message],
        }


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
