from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class VideoTokenInputs:
    pixel_values: torch.Tensor
    num_patches_list: List[int]
    frame_times: List[float]


@dataclass
class VideoTokenOutputs:
    inputs: VideoTokenInputs
    tokens: torch.Tensor
    frame_tokens: List[torch.Tensor]
    backend: str = "simple"


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """LiveStar-style dynamic tiling before ViT encoding."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    tiles_per_row = target_width // image_size
    for i in range(blocks):
        box = (
            (i % tiles_per_row) * image_size,
            (i // tiles_per_row) * image_size,
            ((i % tiles_per_row) + 1) * image_size,
            ((i // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_to_tensor(
    image: Image.Image,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def sample_video_frames(
    video_path: str,
    sample_fps: float = 1.0,
) -> tuple[List[Image.Image], List[float]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")

    try:
        from decord import VideoReader, cpu
    except ImportError:
        return _sample_video_frames_cv2(video_path, sample_fps)

    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_count = len(video_reader)
    fps = float(video_reader.get_avg_fps())
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Cannot read valid FPS/frame count from video: {video_path}")

    interval = max(1, int(fps / sample_fps))
    frame_indices = list(range(0, frame_count, interval))
    batch = video_reader.get_batch(frame_indices).asnumpy()

    frames = [Image.fromarray(batch[i]).convert("RGB") for i in range(batch.shape[0])]
    frame_times = [frame_idx / fps for frame_idx in frame_indices]
    if not frames:
        raise ValueError(f"No frames sampled from video: {video_path}")
    return frames, frame_times


def _sample_video_frames_cv2(video_path: str, sample_fps: float) -> tuple[List[Image.Image], List[float]]:
    """OpenCV fallback for environments without decord.

    decord is preferred because it matches LiveStar, but OpenCV keeps the demo
    runnable in lightweight environments.
    """

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Video loading requires decord or opencv-python. Install one of them to read videos."
        ) from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Cannot read valid FPS/frame count from video: {video_path}")

    interval = max(1, int(fps / sample_fps))
    frames: List[Image.Image] = []
    frame_times: List[float] = []

    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).convert("RGB"))
            frame_times.append(frame_idx / fps)
        frame_idx += 1

    capture.release()
    if not frames:
        raise ValueError(f"No frames sampled from video: {video_path}")
    return frames, frame_times


def load_video_token_inputs(
    video_path: str,
    sample_fps: float = 1.0,
    image_size: int = 448,
    max_num: int = 1,
    use_thumbnail: bool = True,
) -> VideoTokenInputs:
    frames, frame_times = sample_video_frames(video_path, sample_fps=sample_fps)

    pixel_values_list = []
    num_patches_list: List[int] = []
    for frame in frames:
        tiles = dynamic_preprocess(
            frame,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
            max_num=max_num,
        )
        pixel_values = torch.stack([image_to_tensor(tile) for tile in tiles])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    return VideoTokenInputs(
        pixel_values=torch.cat(pixel_values_list, dim=0),
        num_patches_list=num_patches_list,
        frame_times=frame_times,
    )


class SimpleViTPatchTokenizer(nn.Module):
    """A small ViT-style patch tokenizer compatible with LiveStar preprocessing.

    This is not a trained InternViT replacement. It is a local, runnable encoder
    that turns each image tile into patch tokens with the same core operation:
    Conv2d(kernel=patch_size, stride=patch_size).
    """

    def __init__(
        self,
        image_size: int = 448,
        patch_size: int = 14,
        embed_dim: int = 1024,
        include_cls_token: bool = True,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.include_cls_token = include_cls_token
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + int(include_cls_token), embed_dim)
        )
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        if self.include_cls_token:
            class_embeds = self.class_embedding.expand(pixel_values.shape[0], 1, -1)
            patch_embeds = torch.cat([class_embeds, patch_embeds], dim=1)
        return patch_embeds + self.position_embedding.to(patch_embeds.dtype)


@torch.no_grad()
def encode_video_to_vit_tokens(
    video_path: str,
    sample_fps: float = 1.0,
    image_size: int = 448,
    patch_size: int = 14,
    embed_dim: int = 1024,
    max_num: int = 1,
    use_thumbnail: bool = True,
    device: str = "cpu",
) -> VideoTokenOutputs:
    inputs = load_video_token_inputs(
        video_path=video_path,
        sample_fps=sample_fps,
        image_size=image_size,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    encoder = SimpleViTPatchTokenizer(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
    ).to(device)
    pixel_values = inputs.pixel_values.to(device)
    tokens = encoder(pixel_values).cpu()

    frame_tokens = []
    cursor = 0
    for num_patches in inputs.num_patches_list:
        frame_tokens.append(tokens[cursor : cursor + num_patches])
        cursor += num_patches

    return VideoTokenOutputs(inputs=inputs, tokens=tokens, frame_tokens=frame_tokens, backend="simple")


@torch.no_grad()
def encode_video_with_livestar_model(
    video_path: str,
    model_path: str,
    sample_fps: float = 1.0,
    image_size: int = 448,
    max_num: int = 1,
    use_thumbnail: bool = True,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> VideoTokenOutputs:
    """Encode sampled video frames with LiveStar's real visual encoder.

    This mirrors LiveStar/inference/demo_ui.py::load_video for preprocessing and
    LiveStar/inference/modeling_livestar_chat.py::extract_feature for token
    extraction. It requires a complete LiveStar/InternVideo checkpoint at
    ``model_path``.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "LiveStar model encoding requires transformers. Install LiveStar "
            "dependencies or use the default local patch-token backend."
        ) from exc

    inputs = load_video_token_inputs(
        video_path=video_path,
        sample_fps=sample_fps,
        image_size=image_size,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    torch_dtype = _resolve_dtype(dtype, device)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    pixel_values = inputs.pixel_values.to(device=device, dtype=torch_dtype)
    if not hasattr(model, "extract_feature"):
        raise AttributeError("Loaded model does not expose LiveStar extract_feature(pixel_values).")
    tokens = model.extract_feature(pixel_values).detach().cpu()

    frame_tokens = []
    cursor = 0
    for num_patches in inputs.num_patches_list:
        frame_tokens.append(tokens[cursor : cursor + num_patches])
        cursor += num_patches

    return VideoTokenOutputs(inputs=inputs, tokens=tokens, frame_tokens=frame_tokens, backend="livestar")


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
