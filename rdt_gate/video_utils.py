from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class VideoClip:
    clip_id: int
    start_time: float
    end_time: float
    frames: List[np.ndarray]


def load_video_clips(
    video_path: str,
    clip_seconds: float = 1.0,
    frames_per_clip: int = 8,
) -> List[VideoClip]:
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be positive.")
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError(f"Cannot read valid FPS/frame count from video: {video_path}")

    clip_frame_count = max(1, int(round(clip_seconds * fps)))
    clips: List[VideoClip] = []
    clip_id = 0

    for start_frame in range(0, frame_count, clip_frame_count):
        end_frame = min(start_frame + clip_frame_count, frame_count)
        if end_frame <= start_frame:
            continue

        indices = np.linspace(
            start_frame,
            end_frame - 1,
            num=min(frames_per_clip, end_frame - start_frame),
            dtype=int,
        )
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)

        if frames:
            clips.append(
                VideoClip(
                    clip_id=clip_id,
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    frames=frames,
                )
            )
            clip_id += 1

    cap.release()
    if not clips:
        raise ValueError(f"No clips could be extracted from video: {video_path}")
    return clips
