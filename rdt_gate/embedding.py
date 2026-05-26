from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .video_utils import VideoClip


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


def cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(1.0 - np.dot(l2_normalize(a), l2_normalize(b)))


class SimpleClipEmbedder:
    """CPU-only visual embedding based on appearance, texture, and motion."""

    def __init__(self, resize: Tuple[int, int] = (224, 224)) -> None:
        self.resize = resize

    def embed_clip(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("Cannot embed an empty clip.")

        resized = [cv2.resize(frame, self.resize) for frame in frames]
        features: List[np.ndarray] = []

        color_features = [self._color_histogram(frame) for frame in resized]
        edge_features = [self._edge_histogram(frame) for frame in resized]
        gray_maps = [self._coarse_gray_map(frame) for frame in resized]

        features.append(0.55 * np.mean(color_features, axis=0))
        features.append(0.45 * np.mean(edge_features, axis=0))
        features.append(0.70 * np.mean(gray_maps, axis=0))
        features.append(0.85 * self._motion_features(resized))

        return l2_normalize(np.concatenate(features).astype(np.float32))

    def _color_histogram(self, frame: np.ndarray, bins: int = 32) -> np.ndarray:
        hist_parts = []
        for channel in range(3):
            hist = cv2.calcHist([frame], [channel], None, [bins], [0, 256]).flatten()
            hist_parts.append(hist / (hist.sum() + 1e-8))
        return np.concatenate(hist_parts).astype(np.float32)

    def _edge_histogram(self, frame: np.ndarray, bins: int = 32) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        hist = cv2.calcHist([edges], [0], None, [bins], [0, 256]).flatten()
        return (hist / (hist.sum() + 1e-8)).astype(np.float32)

    def _coarse_gray_map(self, frame: np.ndarray, size: Tuple[int, int] = (16, 12)) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).flatten()

    def _motion_features(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if len(frames) < 2:
            return np.zeros(32 + 5 + 16 * 12, dtype=np.float32)

        motion_values = []
        motion_maps = []
        prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for frame in frames[1:]:
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.abs(current - prev) / 255.0
            motion_values.append(float(diff.mean()))
            motion_maps.append(cv2.resize(diff, (16, 12), interpolation=cv2.INTER_AREA))
            prev = current

        values = np.asarray(motion_values, dtype=np.float32)
        hist = np.histogram(values, bins=32, range=(0.0, 1.0))[0].astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        stats = np.array(
            [
                values.mean(),
                values.std(),
                values.max(),
                np.percentile(values, 75),
                values[-1],
            ],
            dtype=np.float32,
        )
        motion_map = np.mean(motion_maps, axis=0).astype(np.float32).flatten()
        return np.concatenate([hist, stats, motion_map])


def extract_embeddings(clips: Iterable[VideoClip], backend: str = "simple") -> np.ndarray:
    if backend != "simple":
        raise NotImplementedError(
            f"embedding_backend={backend!r} is not implemented in the initial version. Use 'simple'."
        )
    embedder = SimpleClipEmbedder()
    return np.stack([embedder.embed_clip(clip.frames) for clip in clips], axis=0)


def make_synthetic_embeddings(
    num_clips: int = 30,
    dim: int = 128,
    seed: int = 7,
) -> tuple[list[VideoClip], np.ndarray]:
    rng = np.random.default_rng(seed)
    routine_center = l2_normalize(rng.normal(size=dim))
    raw_event = rng.normal(size=dim)
    event_direction = l2_normalize(raw_event - np.dot(raw_event, routine_center) * routine_center)
    recovery_center = l2_normalize(0.80 * routine_center + 0.20 * event_direction)

    embeddings = []
    clips = []
    for clip_id in range(num_clips):
        if clip_id < 20:
            center = routine_center
            noise_scale = 0.025 if clip_id % 5 else 0.045
        elif clip_id < 25:
            center = l2_normalize(0.15 * routine_center + 0.85 * event_direction)
            noise_scale = 0.030
        else:
            center = recovery_center
            noise_scale = 0.025

        embeddings.append(l2_normalize(center + noise_scale * rng.normal(size=dim)))
        clips.append(VideoClip(clip_id, float(clip_id), float(clip_id + 1), []))

    return clips, np.stack(embeddings, axis=0)
