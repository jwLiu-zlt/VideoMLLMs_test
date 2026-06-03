#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rdt_gate.livestar_tokens import encode_video_to_vit_tokens, encode_video_with_livestar_model
from rdt_gate.token_prototype import build_frame_token_prototype, build_frame_token_prototype_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: sample video frames and encode them into LiveStar-style visual tokens."
    )
    parser.add_argument("--video_path", default="data/1.mp4")
    parser.add_argument(
        "--backend",
        choices=["simple", "livestar"],
        default="simple",
        help="simple is local and runnable; livestar requires a complete LiveStar checkpoint.",
    )
    parser.add_argument("--model_path", default="../LiveStar/inference")
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--max_num", type=int, default=1)
    parser.add_argument("--no_thumbnail", action="store_true", help="Disable LiveStar thumbnail tile.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_path", default="outputs/vit_tokens_1fps.pt")
    parser.add_argument("--prototype_enable", action="store_true", help="Build a routine prototype vector from tokens.")
    parser.add_argument("--prototype_gate", choices=["bank", "single"], default="bank")
    parser.add_argument("--prototype_pool", choices=["mean", "cls", "tile_mean"], default="tile_mean")
    parser.add_argument("--prototype_keep_cls", action="store_true", help="Keep CLS token when mean-pooling tokens.")
    parser.add_argument("--clip_window", type=int, default=1)
    parser.add_argument("--max_prototypes", type=int, default=4)
    parser.add_argument("--init_cluster_threshold", type=float, default=0.08)
    parser.add_argument("--cooldown_items", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--tau_silence", type=float, default=None)
    parser.add_argument("--tau_suspicious", type=float, default=None)
    parser.add_argument("--tau_change_low", type=float, default=None)
    parser.add_argument("--tau_change_high", type=float, default=None)
    parser.add_argument("--init_var_threshold", type=float, default=0.10)
    parser.add_argument("--init_change_threshold", type=float, default=0.10)
    parser.add_argument("--max_wait", type=int, default=99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "livestar":
        result = encode_video_with_livestar_model(
            video_path=args.video_path,
            model_path=args.model_path,
            sample_fps=args.sample_fps,
            image_size=args.image_size,
            max_num=args.max_num,
            use_thumbnail=not args.no_thumbnail,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        result = encode_video_to_vit_tokens(
            video_path=args.video_path,
            sample_fps=args.sample_fps,
            image_size=args.image_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            max_num=args.max_num,
            use_thumbnail=not args.no_thumbnail,
            device=args.device,
        )

    prototype = None
    if args.prototype_enable:
        if args.prototype_gate == "bank":
            prototype = build_frame_token_prototype_bank(
                frame_tokens=result.frame_tokens,
                frame_times=result.inputs.frame_times,
                sample_fps=args.sample_fps,
                pool_mode=args.prototype_pool,
                exclude_cls=not args.prototype_keep_cls,
                clip_window=args.clip_window,
                warmup_frames=args.warmup_frames,
                max_prototypes=args.max_prototypes,
                alpha=args.alpha,
                init_cluster_threshold=args.init_cluster_threshold,
                tau_silence=args.tau_silence,
                tau_suspicious=args.tau_suspicious,
                tau_change_low=args.tau_change_low,
                tau_change_high=args.tau_change_high,
                max_wait=args.max_wait,
                cooldown_items=args.cooldown_items,
            )
        else:
            legacy_pool = "mean" if args.prototype_pool == "tile_mean" else args.prototype_pool
            prototype = build_frame_token_prototype(
                frame_tokens=result.frame_tokens,
                frame_times=result.inputs.frame_times,
                sample_fps=args.sample_fps,
                pool=legacy_pool,
                exclude_cls=not args.prototype_keep_cls,
                warmup_frames=args.warmup_frames,
                alpha=args.alpha,
                tau_silence=args.tau_silence or 0.03,
                tau_suspicious=args.tau_suspicious or 0.08,
                tau_change_low=args.tau_change_low or 0.01,
                tau_change_high=args.tau_change_high or 0.013,
                init_var_threshold=args.init_var_threshold,
                init_change_threshold=args.init_change_threshold,
                max_wait=args.max_wait,
            )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    payload = {
        "video_path": args.video_path,
        "backend": result.backend,
        "model_path": args.model_path if args.backend == "livestar" else None,
        "sample_fps": args.sample_fps,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "use_thumbnail": not args.no_thumbnail,
        "frame_times": result.inputs.frame_times,
        "num_patches_list": result.inputs.num_patches_list,
        "pixel_values_shape": tuple(result.inputs.pixel_values.shape),
        "tokens": result.tokens,
        "tokens_shape": tuple(result.tokens.shape),
    }
    if prototype is not None:
        payload.update(
            {
                "frame_embeddings": torch.from_numpy(prototype.frame_embeddings),
                "frame_embeddings_shape": tuple(prototype.frame_embeddings.shape),
                "prototype_vector": None
                if _prototype_vectors(prototype) is None
                else torch.from_numpy(_prototype_vectors(prototype)),
                "prototype_vector_shape": None
                if _prototype_vectors(prototype) is None
                else tuple(_prototype_vectors(prototype).shape),
                "prototype_gate": args.prototype_gate,
                "prototype_pool": args.prototype_pool,
                "prototype_keep_cls": args.prototype_keep_cls,
                "clip_window": args.clip_window,
                "deviation_judgments": _serialize_prototype_results(prototype),
            }
        )

    torch.save(
        payload,
        args.save_path,
    )
    if prototype is not None:
        csv_path = _prototype_csv_path(args.save_path)
        _write_prototype_csv(csv_path, prototype)

    print(f"video_path: {args.video_path}")
    print(f"backend: {result.backend}")
    if args.backend == "livestar":
        print(f"model_path: {args.model_path}")
    print(f"sample_fps: {args.sample_fps}")
    print(f"sampled_frames: {len(result.inputs.frame_times)}")
    print(f"frame_times: {[round(t, 3) for t in result.inputs.frame_times]}")
    print(f"num_patches_list: {result.inputs.num_patches_list}")
    print(f"pixel_values shape: {tuple(result.inputs.pixel_values.shape)}")
    print(f"vit tokens shape: {tuple(result.tokens.shape)}")
    print(f"tokens per image tile: {result.tokens.shape[1]}")
    if prototype is not None:
        print(f"frame embeddings shape: {tuple(prototype.frame_embeddings.shape)}")
        print(
            "prototype vector shape: "
            f"{None if _prototype_vectors(prototype) is None else tuple(_prototype_vectors(prototype).shape)}"
        )
        print(f"prototype ready: {_prototype_vectors(prototype) is not None}")
        print(f"prototype gate: {args.prototype_gate}")
        print(f"prototype csv: {_prototype_csv_path(args.save_path)}")
    print(f"saved: {args.save_path}")


def _prototype_csv_path(save_path: str) -> str:
    root, _ = os.path.splitext(save_path)
    return root + "_prototype.csv"


def _write_prototype_csv(path: str, prototype) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "start_time",
                "end_time",
                "change",
                "deviation",
                "matched_prototype_id",
                "decision",
                "prototype_ready",
                "is_deviated",
                "deviation_level",
                "trigger_slow_path",
                "reason",
            ],
        )
        writer.writeheader()
        for row in _iter_csv_rows(prototype):
            writer.writerow(
                {
                    "frame_id": row["frame_id"],
                    "start_time": f"{row['start_time']:.4f}",
                    "end_time": f"{row['end_time']:.4f}",
                    "change": _fmt(row["change"]),
                    "deviation": _fmt(row["deviation"]),
                    "matched_prototype_id": "" if row["matched_prototype_id"] is None else row["matched_prototype_id"],
                    "decision": row["decision"],
                    "prototype_ready": row["prototype_ready"],
                    "is_deviated": row["is_deviated"],
                    "deviation_level": row["deviation_level"],
                    "trigger_slow_path": row["trigger_slow_path"],
                    "reason": row["reason"],
                }
            )


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _prototype_vectors(prototype):
    if hasattr(prototype, "prototype_vectors"):
        return prototype.prototype_vectors
    if prototype.prototype_vector is None:
        return None
    return prototype.prototype_vector


def _serialize_prototype_results(prototype):
    return list(_iter_csv_rows(prototype))


def _iter_csv_rows(prototype):
    if hasattr(prototype, "prototype_vectors"):
        for row in prototype.results:
            yield {
                "frame_id": row.item_id,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "change": row.change,
                "deviation": row.min_deviation,
                "matched_prototype_id": row.matched_prototype_id,
                "decision": row.decision.value,
                "prototype_ready": row.prototype_ready,
                "is_deviated": row.deviation_level == "deviated",
                "deviation_level": row.deviation_level,
                "trigger_slow_path": row.trigger_slow_path,
                "reason": row.reason,
            }
        return

    for row, judgment in zip(prototype.results, prototype.judgments):
        yield {
            "frame_id": row.clip_id,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "change": row.change,
            "deviation": row.deviation,
            "matched_prototype_id": None,
            "decision": row.decision.value,
            "prototype_ready": row.prototype_ready,
            "is_deviated": judgment.is_deviated,
            "deviation_level": judgment.deviation_level,
            "trigger_slow_path": judgment.trigger_slow_path,
            "reason": judgment.reason,
        }


if __name__ == "__main__":
    main()
