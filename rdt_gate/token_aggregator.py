from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .embedding import l2_normalize


@dataclass
class AggregatedTokenItem:
    """One normalized item consumed by the prototype gate.

    The item can represent a single frame or a short clip. Evidence keeps
    frame/tile/token metadata for later explanation when the slow path fires.
    """

    item_id: int
    start_time: float
    end_time: float
    embedding: np.ndarray
    frame_ids: list[int]
    evidence: dict[str, Any] = field(default_factory=dict)


def aggregate_frame_tokens(
    frame_tokens: Sequence[torch.Tensor],
    frame_times: Sequence[float],
    sample_fps: float,
    pool_mode: str = "tile_mean",
    clip_window: int = 1,
    exclude_cls: bool = True,
) -> list[AggregatedTokenItem]:
    """Aggregate LiveStar-style visual tokens into prototype embeddings.

    Args:
        frame_tokens: Each tensor must be [num_tiles, num_tokens, dim].
        frame_times: Seconds for each sampled frame.
        sample_fps: Used to infer the last item's end time.
        pool_mode: mean, tile_mean, or cls.
        clip_window: Number of consecutive frames per output item.
        exclude_cls: When pooling patch tokens, ignore token 0 if present.

    Returns:
        A list of L2-normalized AggregatedTokenItem records.
    """

    if not frame_tokens:
        raise ValueError("frame_tokens is empty, cannot aggregate prototype embeddings.")
    if len(frame_tokens) != len(frame_times):
        raise ValueError(f"frame_tokens length {len(frame_tokens)} != frame_times length {len(frame_times)}")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")
    if clip_window <= 0:
        raise ValueError("clip_window must be positive.")

    frame_embeddings: list[np.ndarray] = []
    frame_evidence: list[dict[str, Any]] = []

    for frame_id, tokens in enumerate(frame_tokens):
        if tokens.ndim != 3:
            raise ValueError(f"Expected frame_tokens[{frame_id}] shape [tiles, tokens, dim], got {tuple(tokens.shape)}")

        values = tokens.detach().float().cpu()
        pooled, evidence = _pool_one_frame(values, pool_mode=pool_mode, exclude_cls=exclude_cls)
        evidence.update(
            {
                "frame_id": frame_id,
                "num_tiles": int(values.shape[0]),
                "num_tokens": int(values.shape[1]),
                "dim": int(values.shape[2]),
            }
        )
        frame_embeddings.append(l2_normalize(pooled.numpy().astype(np.float32)))
        frame_evidence.append(evidence)

    items: list[AggregatedTokenItem] = []
    duration = 1.0 / sample_fps
    for start in range(0, len(frame_embeddings), clip_window):
        end = min(start + clip_window, len(frame_embeddings))
        clip_vectors = np.stack(frame_embeddings[start:end], axis=0)
        clip_embedding = l2_normalize(clip_vectors.mean(axis=0))
        start_time = float(frame_times[start])
        end_time = float(frame_times[end]) if end < len(frame_times) else float(frame_times[end - 1] + duration)

        items.append(
            AggregatedTokenItem(
                item_id=len(items),
                start_time=start_time,
                end_time=end_time,
                embedding=clip_embedding,
                frame_ids=list(range(start, end)),
                evidence={
                    "pool_mode": pool_mode,
                    "clip_window": clip_window,
                    "frames": frame_evidence[start:end],
                },
            )
        )

    return items


def embeddings_from_items(items: Sequence[AggregatedTokenItem]) -> np.ndarray:
    """Return a dense [num_items, dim] matrix from aggregated items."""

    if not items:
        raise ValueError("items is empty, cannot build embedding matrix.")
    return np.stack([l2_normalize(item.embedding) for item in items], axis=0)


def _pool_one_frame(values: torch.Tensor, pool_mode: str, exclude_cls: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    if pool_mode == "cls":
        return values[:, 0, :].mean(dim=0), {"pool_mode": "cls"}

    token_values = values
    if exclude_cls and values.shape[1] > 1:
        token_values = values[:, 1:, :]

    if pool_mode == "mean":
        return token_values.mean(dim=(0, 1)), {"pool_mode": "mean", "exclude_cls": exclude_cls}

    if pool_mode == "tile_mean":
        tile_embeds = token_values.mean(dim=1)
        return tile_embeds.mean(dim=0), {"pool_mode": "tile_mean", "exclude_cls": exclude_cls}

    raise ValueError(f"Unsupported pool_mode: {pool_mode}")
