# 原型向量驱动的 LiveStar 融合项目技术文档

## 1. 项目定位

本项目的主线不是复刻 LiveStar，而是以“原型向量”为核心，构建一个面向流式视频理解的 Fast-Slow Gate 系统。

核心目标：

1. 使用原型向量维护视频中的常规模式，也就是 routine pattern。
2. 对每个新进入的视频片段或帧，判断它是否偏离 routine pattern。
3. 只有在明显偏离、变化较大、或持续不确定时，才调用 LiveStar 这类大模型慢路径。
4. 借鉴 LiveStar 的帧处理、dynamic tiling、视觉 token 提取、`<IMG_CONTEXT>` 图文融合和响应-静默解码思想。
5. 保证工程逻辑清楚、鲁棒性强、容易调参、容易扩展。

最终系统应该支持：

- 普通视频流：稳定输出 `SILENCE / WAIT / SUSPICIOUS`。
- LiveStar 后端：把关键片段送入大模型解释。
- 原型向量后端：独立运行，不依赖大模型也能做触发判断。
- 可解释输出：说明为什么触发慢路径，例如偏离哪个 prototype、哪个 frame 或 tile 最异常。

## 2. 总体架构

推荐架构如下：

```text
video stream
  -> frame sampler
  -> LiveStar-style dynamic tiling
  -> visual token encoder
       -> simple backend 或 LiveStar backend
  -> token aggregator
  -> prototype bank gate
       -> SILENCE
       -> WAIT
       -> SUSPICIOUS
  -> slow path router
       -> 不触发：继续缓存
       -> 触发：调用 LiveStar chat / 大模型解释
  -> result logger
  -> report / csv / visualization
```

关键设计原则：

- 原型向量是主控模块。
- LiveStar 是可选慢路径和高质量视觉 token 来源。
- 原型判断必须能够在没有完整 LiveStar checkpoint 时运行。
- 所有模块都要能单独测试，避免大模型依赖导致整体不可跑。

## 3. 推荐目录结构

```text
damo_0526/
  main.py
  requirements.txt
  README.md
  docs/
    prototype_livestar_project_design.md
  rdt_gate/
    __init__.py
    embedding.py
    video_utils.py
    livestar_tokens.py
    token_prototype.py
    prototype_gate.py
    prototype_bank_gate.py
    token_aggregator.py
    slow_path.py
    decision_schema.py
    visualization.py
    evaluation.py
  test/
    damo1.py
    test_token_aggregator.py
    test_prototype_bank_gate.py
  outputs/
```

其中新增模块建议如下：

- `prototype_bank_gate.py`：多原型向量 gate，是项目核心。
- `token_aggregator.py`：把 LiveStar visual tokens 聚合成 frame/clip embedding。
- `slow_path.py`：统一封装 LiveStar 或其他大模型调用。
- `decision_schema.py`：统一定义 gate 输出结构，保证 CSV、日志、慢路径都能复用。

## 4. 数据流设计

### 4.1 输入视频到帧

借鉴 LiveStar 的 `load_video()`：

1. 使用 `decord.VideoReader` 读取视频。
2. 读取 `fps` 和 `frame_count`。
3. 根据 `sample_fps` 均匀抽帧。
4. 每帧记录时间戳。
5. 对异常情况做保护：
   - 视频路径不存在，直接抛出清楚错误。
   - fps 小于等于 0，抛出错误。
   - 没抽到帧，抛出错误。
   - `sample_fps <= 0`，拒绝运行。

推荐接口：

```python
def sample_video_frames(
    video_path: str,
    sample_fps: float = 1.0,
) -> tuple[list[Image.Image], list[float]]:
    """读取视频并按固定 fps 抽帧。

    中文注释建议：
    - frame_times 与 frames 一一对应。
    - 不要在这里做模型推理，只负责稳定读帧。
    - 所有异常都要带上 video_path，方便排查。
    """
```

### 4.2 帧到 LiveStar-style tiles

借鉴 LiveStar 的 `dynamic_preprocess()`：

1. 根据原图宽高比选择最接近的 tile 网格。
2. resize 后切成 `image_size x image_size` tiles。
3. 可选加入 thumbnail。
4. 每个 tile 做 ImageNet normalize。

推荐保持当前项目 [rdt_gate/livestar_tokens.py](/Users/guosiqi/vippython/MLLMs/damo_0526/rdt_gate/livestar_tokens.py:1) 的实现，并增强注释。

重要参数：

- `image_size=448`
- `max_num=1`：默认一帧一个 tile，省算力。
- `use_thumbnail=True`：多 tile 时保留全局上下文。

### 4.3 tiles 到 visual tokens

支持两个后端：

1. `simple backend`
   - 用本地 ViT-style patch tokenizer。
   - 不需要完整 LiveStar 权重。
   - 用于测试和原型验证。

2. `livestar backend`
   - 调用 LiveStar 的 `model.extract_feature(pixel_values)`。
   - 输出经过 ViT、pixel shuffle、ToMe merge、MLP 后的视觉 token。
   - 更适合真实推理和最终系统。

推荐统一输出：

```python
@dataclass
class VideoTokenOutputs:
    inputs: VideoTokenInputs
    tokens: torch.Tensor
    frame_tokens: list[torch.Tensor]
    backend: str
```

`frame_tokens` 的语义：

```text
frame_tokens[i].shape = [num_tiles, num_tokens, dim]
```

如果是 LiveStar backend：

- `num_tokens` 通常是 16。
- `dim` 通常是 4096，因为已经投影到 LLM hidden size。

如果是 simple backend：

- `num_tokens` 可能是 `1 + patch_tokens`。
- `dim` 是本地 tokenizer 的 embedding dim。

## 5. 原型向量核心优化

当前项目已有单原型 `PrototypeGate`。建议升级为多原型 `PrototypeBankGate`。

### 5.1 为什么不用单一 prototype

单原型的问题：

- 正常状态可能不止一种。
- 高动态但语义正常的阶段会被误判。
- EMA 更新容易被异常污染。
- 阈值固定，换视频容易失效。

因此新的原型向量应该是：

```text
prototype_bank = [
  routine_static_prototype,
  routine_motion_prototype,
  routine_scene_prototype,
  routine_semantic_prototype,
  ...
]
```

判断当前 embedding `z` 是否异常：

```python
deviation = min(cosine_distance(z, p) for p in prototype_bank)
matched_prototype_id = argmin(...)
```

### 5.2 PrototypeBankGate 输入输出

推荐输入：

```python
z: np.ndarray
timestamp: float
change: float | None
metadata: dict
```

推荐输出：

```python
@dataclass
class GateDecision:
    item_id: int
    start_time: float
    end_time: float
    change: float | None
    min_deviation: float | None
    matched_prototype_id: int | None
    decision: Signal
    prototype_ready: bool
    trigger_slow_path: bool
    deviation_level: str
    reason: str
```

### 5.3 初始化策略

旧版逻辑是 warmup 后直接 mean。

新版推荐：

1. 收集 warmup embeddings。
2. 计算相邻 change。
3. 过滤明显抖动帧。
4. 如果 warmup 内变化很小，初始化一个 prototype。
5. 如果 warmup 内有多种稳定模式，使用轻量聚类初始化多个 prototype。

不建议一开始引入复杂聚类库。可以用简单在线策略：

```text
for z in warmup_vectors:
  如果没有 prototype:
    建第一个 prototype
  否则:
    找最近 prototype
    如果距离小于 init_cluster_threshold:
      加入该 prototype 的缓存
    否则:
      新建 prototype，但数量不能超过 max_prototypes
```

每个 prototype 维护：

```python
@dataclass
class PrototypeSlot:
    vector: np.ndarray
    count: int
    last_update_id: int
    stable_score: float
```

### 5.4 自适应阈值

固定阈值容易失败。推荐 warmup 后自动估计：

```python
tau_silence = quantile(warmup_deviations, 0.75)
tau_suspicious = quantile(warmup_deviations, 0.95) + margin
tau_change_low = quantile(warmup_changes, 0.50)
tau_change_high = quantile(warmup_changes, 0.95) + margin
```

为了鲁棒性：

- 如果 warmup 样本太少，回退默认阈值。
- 如果 quantile 得到 NaN，回退默认阈值。
- 如果 `tau_suspicious <= tau_silence`，强制拉开间隔。

推荐参数：

```python
min_tau_gap = 1e-3
default_tau_silence = 0.03
default_tau_suspicious = 0.08
default_tau_change_low = 0.01
default_tau_change_high = 0.013
```

### 5.5 保守更新策略

不能只要 `SILENCE` 就直接更新。推荐：

```text
SILENCE 且 deviation 很低：
  正常 EMA 更新

SILENCE 但 deviation 接近边界：
  小权重 EMA 更新

WAIT：
  不更新主 prototype，只放入 pending buffer

SUSPICIOUS：
  不更新，并进入 cooldown

cooldown 内：
  禁止更新，防止异常污染 routine
```

推荐更新公式：

```python
weight = update_weight_by_confidence(deviation)
new_vector = l2_normalize((1 - weight) * old_vector + weight * z)
```

其中：

```python
if deviation <= tau_silence * 0.5:
    weight = 1.0 - alpha
elif deviation <= tau_silence:
    weight = 0.5 * (1.0 - alpha)
else:
    weight = 0.0
```

## 6. Token 聚合策略

当前项目的 `tokens_to_frame_embeddings()` 是 mean pooling。建议升级为 `token_aggregator.py`。

### 6.1 聚合模式

支持四种模式：

1. `mean`
   - 对所有 token 平均。
   - 简单稳定，适合 baseline。

2. `tile_mean`
   - 先每个 tile 内平均，再对 tile 平均。
   - 保留 tile 层级，避免大 tile 数视频权重异常。

3. `max_deviation`
   - 每帧保留与 prototype bank 偏离最大的 tile/token 信息。
   - 适合做异常定位。

4. `temporal_clip`
   - 多帧组成一个 clip embedding。
   - 适合减少逐帧噪声。

推荐默认：

```python
pool_mode = "tile_mean"
clip_window = 4
```

### 6.2 推荐接口

```python
@dataclass
class AggregatedTokenItem:
    item_id: int
    start_time: float
    end_time: float
    embedding: np.ndarray
    frame_ids: list[int]
    evidence: dict
```

```python
def aggregate_frame_tokens(
    frame_tokens: Sequence[torch.Tensor],
    frame_times: Sequence[float],
    sample_fps: float,
    pool_mode: str = "tile_mean",
    clip_window: int = 1,
) -> list[AggregatedTokenItem]:
    """把 LiveStar visual tokens 聚合成原型向量可用的 embedding。

    中文注释建议：
    - 这里不做 gate 判断，只负责 token 到 embedding。
    - 每个 embedding 都必须 l2 normalize。
    - evidence 用于后续解释异常位置。
    """
```

## 7. Slow Path 设计

慢路径负责调用 LiveStar 或其他大模型解释视频片段。

### 7.1 触发条件

触发慢路径的条件：

- `decision == SUSPICIOUS`
- `change > tau_change_high`
- `min_deviation > tau_suspicious`
- `WAIT` 连续超过 `max_wait`
- 用户强制开启 debug 模式

### 7.2 慢路径输入

不要把全部历史都扔给大模型。推荐只送：

- 当前触发帧。
- 触发前后若干帧。
- 与 prototype 偏离最大的 tile 或 frame。
- 最近一次稳定描述，如果有。

推荐结构：

```python
@dataclass
class SlowPathRequest:
    video_path: str
    frame_indices: list[int]
    frame_times: list[float]
    prompt: str
    reason: str
    evidence: dict
```

### 7.3 借鉴 LiveStar 的响应-静默逻辑

可以把原型 gate 和 LiveStar SVeD 组合：

```text
prototype gate 先判断是否值得进入慢路径
  如果 SILENCE：不调用 LiveStar
  如果 WAIT：累计上下文
  如果 SUSPICIOUS：调用 LiveStar

LiveStar 内部再用 perplexity 验证
  如果旧答案仍成立：可以不重新生成
  如果旧答案不成立：生成新解释
```

融合后决策：

```text
final_trigger =
  prototype_suspicious
  OR high_adjacent_change
  OR livestar_ppl_over_threshold
```

## 8. 运行主流程

推荐 `main.py` 主流程：

```python
def main():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 读取视频并抽帧
    frames, frame_times = sample_video_frames(args.video_path, args.sample_fps)

    # 3. 编码视觉 token
    token_outputs = encode_video_tokens(
        video_path=args.video_path,
        backend=args.token_backend,
        model_path=args.livestar_model_path,
    )

    # 4. token 聚合成 embedding
    items = aggregate_frame_tokens(
        token_outputs.frame_tokens,
        token_outputs.inputs.frame_times,
        sample_fps=args.sample_fps,
        pool_mode=args.pool_mode,
        clip_window=args.clip_window,
    )

    # 5. 原型向量 gate
    gate = PrototypeBankGate(...)
    decisions = gate.run(items)

    # 6. 慢路径
    if args.enable_slow_path:
        slow_path_results = run_slow_path_if_needed(decisions, token_outputs, args)

    # 7. 保存 CSV / JSON / 图表
    write_outputs(decisions, slow_path_results, args.output_dir)
```

## 9. 鲁棒性要求

### 9.1 文件和路径

- 所有路径都要用 `os.path.exists()` 检查。
- LiveStar model path 不存在时，如果 `backend=livestar`，直接报清楚错误。
- 如果 `backend=simple`，不需要 LiveStar 权重。

### 9.2 数值稳定

- 所有 embedding 必须 `float32`。
- cosine distance 前必须 L2 normalize。
- normalize 时加 `1e-8`。
- 阈值计算时过滤 NaN 和 None。
- 空列表直接报错，不进入模型。

### 9.3 大模型调用

- CUDA 不可用时，允许切换 CPU，但要提示速度很慢。
- dtype 不支持时回退 float32。
- LiveStar `extract_feature` 不存在时，抛出明确 `AttributeError`。
- 慢路径失败不能中断整个 gate，应该记录 warning 并继续输出 prototype 结果。

### 9.4 冷启动

prototype 未 ready 时：

- 输出 `WAIT`。
- reason 写 `prototype_not_ready`。
- 不触发慢路径，除非 `force_slow_path_on_warmup=True`。

### 9.5 异常污染防护

- SUSPICIOUS 后进入 cooldown。
- cooldown 内禁止更新 prototype。
- WAIT 不更新主 prototype。
- 只有高置信 SILENCE 才更新。

## 10. CSV 和日志输出

推荐输出字段：

```text
item_id
start_time
end_time
change
min_deviation
matched_prototype_id
decision
prototype_ready
trigger_slow_path
deviation_level
reason
top_frame_id
top_tile_id
top_token_id
slow_path_text
```

其中：

- `reason` 用分号拼接，例如：
  - `prototype_not_ready;change_not_available`
  - `deviation_over_tau_suspicious;trigger_slow_path`
  - `matched_routine_prototype;deviation_below_tau_silence`
- `slow_path_text` 如果没有触发慢路径，留空。

## 11. 推荐命令

### 11.1 仅使用 simple backend

```bash
python main.py \
  --video_path data/1.mp4 \
  --token_backend simple \
  --sample_fps 1 \
  --pool_mode tile_mean \
  --clip_window 4 \
  --output_dir outputs_proto_bank
```

### 11.2 使用 LiveStar backend

```bash
python main.py \
  --video_path data/1.mp4 \
  --token_backend livestar \
  --livestar_model_path ../LiveStar/inference \
  --device cuda \
  --dtype bfloat16 \
  --sample_fps 1 \
  --pool_mode tile_mean \
  --clip_window 4 \
  --output_dir outputs_livestar_proto
```

### 11.3 开启慢路径

```bash
python main.py \
  --video_path data/1.mp4 \
  --token_backend livestar \
  --livestar_model_path ../LiveStar/inference \
  --enable_slow_path \
  --output_dir outputs_slow_path
```

## 12. 和 LiveStar 的融合边界

推荐借鉴 LiveStar 的部分：

- `decord.VideoReader` 抽帧。
- `dynamic_preprocess()`。
- ImageNet normalize。
- `model.extract_feature(pixel_values)`。
- `<image>` 到 `<IMG_CONTEXT>` 的视觉 token 替换思路。
- `check_answer` perplexity 验证。
- streaming KV cache 接口。

不建议直接照搬的部分：

- 不要把 response-silence 完全交给 perplexity。
- 不要每一帧都调用 LiveStar chat。
- 不要只靠 EOS 或大模型输出决定是否响应。
- 不要让大模型慢路径控制 prototype 更新。

本项目应该保持：

```text
prototype vector 是主控
LiveStar 是视觉 token 来源和慢路径解释器
```

## 13. 开发顺序

建议按以下顺序实现：

1. 新增 `decision_schema.py`，统一输出结构。
2. 新增 `token_aggregator.py`，支持 `mean` 和 `tile_mean`。
3. 新增 `prototype_bank_gate.py`，先支持多 prototype 和自适应阈值。
4. 把 `test/damo1.py` 改成可以使用新 gate。
5. 给 `main.py` 增加 `token_backend` 入口。
6. 新增 CSV 输出字段。
7. 增加 slow path，但默认关闭。
8. 最后接 LiveStar perplexity 验证。

## 14. 最小可行版本

第一版不需要一步到位。最小可行版本：

- simple backend 可跑。
- LiveStar backend 可选。
- `PrototypeBankGate` 支持：
  - warmup 初始化
  - 多原型匹配
  - 自适应阈值
  - 保守更新
- `token_aggregator` 支持：
  - `mean`
  - `tile_mean`
- 输出完整 CSV。

之后再加：

- top abnormal tile/token evidence。
- LiveStar slow path。
- KV cache。
- 更复杂的 prototype 聚类。

## 15. 关键结论

这个融合项目的核心不是“用 LiveStar 替代原型向量”，而是：

```text
用 LiveStar 的视觉 token 提升原型向量质量，
用原型向量减少 LiveStar 慢路径调用，
用 LiveStar 慢路径解释原型向量发现的异常。
```

最终系统应该形成三层能力：

1. Fast Gate：原型向量快速判断是否偏离 routine。
2. Evidence Layer：指出异常来自哪一帧、哪一个 tile、哪一组 token。
3. Slow Reasoner：必要时调用 LiveStar 生成自然语言解释。

这样既保留你的原型向量主线，又能自然融合 LiveStar 的视频处理和大模型理解能力。

## 16. 关键代码骨架

这一节用于直接指导生成完整项目代码。实现时可以按下面的接口拆文件，先保证 simple backend 可跑，再逐步接 LiveStar backend。

### 16.1 `decision_schema.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Signal(str, Enum):
    """Gate 输出信号。

    SILENCE：当前片段接近 routine，不需要慢路径。
    WAIT：当前片段不够确定，继续累计上下文。
    SUSPICIOUS：当前片段明显偏离 routine，建议触发慢路径。
    """

    SILENCE = "SILENCE"
    WAIT = "WAIT"
    SUSPICIOUS = "SUSPICIOUS"


@dataclass
class GateDecision:
    """原型向量 gate 的统一输出结构。

    这个结构后续会同时用于 CSV、JSON、可视化和 slow path router。
    字段要尽量稳定，避免不同模块各自拼字典导致后续维护困难。
    """

    item_id: int
    start_time: float
    end_time: float
    change: float | None
    min_deviation: float | None
    matched_prototype_id: int | None
    decision: Signal
    prototype_ready: bool
    trigger_slow_path: bool
    deviation_level: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    slow_path_text: str | None = None
```

### 16.2 `token_aggregator.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .embedding import l2_normalize


@dataclass
class AggregatedTokenItem:
    """聚合后的原型向量输入单元。

    一个 item 可以是一帧，也可以是多个连续帧组成的 clip。
    embedding 是原型 gate 的直接输入，必须已经 L2 normalize。
    evidence 保存帧、tile、token 的索引信息，方便触发后解释。
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
) -> list[AggregatedTokenItem]:
    """把 LiveStar-style visual tokens 聚合成原型向量输入。

    参数说明：
    - frame_tokens[i] 形状应为 [num_tiles, num_tokens, dim]。
    - frame_times[i] 是第 i 帧的秒级时间戳。
    - clip_window=1 表示逐帧判断，clip_window>1 表示多帧平滑。

    鲁棒性要求：
    - 空 frame_tokens 直接报错。
    - frame_tokens 和 frame_times 长度不一致直接报错。
    - 每个 tensor 维度必须是 3，否则报错。
    - 所有输出 embedding 都必须是 float32 且 L2 normalize。
    """

    if not frame_tokens:
        raise ValueError("frame_tokens is empty, cannot aggregate prototype embeddings.")
    if len(frame_tokens) != len(frame_times):
        raise ValueError(
            f"frame_tokens length {len(frame_tokens)} != frame_times length {len(frame_times)}"
        )
    if clip_window <= 0:
        raise ValueError("clip_window must be positive.")

    frame_embeddings: list[np.ndarray] = []
    frame_evidence: list[dict[str, Any]] = []

    for frame_id, tokens in enumerate(frame_tokens):
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected frame_tokens[{frame_id}] shape [tiles, tokens, dim], got {tuple(tokens.shape)}"
            )

        values = tokens.detach().float().cpu()

        if pool_mode == "mean":
            # 最简单的全局平均，适合作为 baseline。
            pooled = values.mean(dim=(0, 1))
            evidence = {"pool_mode": "mean", "frame_id": frame_id}
        elif pool_mode == "tile_mean":
            # 先 tile 内平均，再 frame 内平均，避免多 tile 帧权重异常。
            tile_embeds = values.mean(dim=1)
            pooled = tile_embeds.mean(dim=0)
            evidence = {
                "pool_mode": "tile_mean",
                "frame_id": frame_id,
                "num_tiles": int(values.shape[0]),
                "num_tokens": int(values.shape[1]),
            }
        else:
            raise ValueError(f"Unsupported pool_mode: {pool_mode}")

        frame_embeddings.append(l2_normalize(pooled.numpy().astype(np.float32)))
        frame_evidence.append(evidence)

    items: list[AggregatedTokenItem] = []
    duration = 1.0 / sample_fps if sample_fps > 0 else 0.0

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
```

### 16.3 `prototype_bank_gate.py`

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence

import numpy as np

from .decision_schema import GateDecision, Signal
from .embedding import cosine_distance, l2_normalize
from .token_aggregator import AggregatedTokenItem


@dataclass
class PrototypeSlot:
    """一个 routine prototype 槽位。

    vector：当前原型向量。
    count：被稳定样本更新过多少次。
    last_update_id：最近一次更新来自哪个 item。
    stable_score：预留字段，后续可用于删除不稳定 prototype。
    """

    vector: np.ndarray
    count: int
    last_update_id: int
    stable_score: float = 1.0


class PrototypeBankGate:
    """多原型向量 Gate。

    这是项目主控模块：
    - warmup 阶段建立 routine prototype bank。
    - online 阶段判断当前 embedding 是否偏离所有 routine prototypes。
    - 只允许高置信 SILENCE 更新 prototype，防止异常污染。
    """

    def __init__(
        self,
        warmup_items: int = 8,
        max_prototypes: int = 4,
        alpha: float = 0.9,
        init_cluster_threshold: float = 0.08,
        tau_silence: float | None = None,
        tau_suspicious: float | None = None,
        tau_change_low: float | None = None,
        tau_change_high: float | None = None,
        default_tau_silence: float = 0.03,
        default_tau_suspicious: float = 0.08,
        default_tau_change_low: float = 0.01,
        default_tau_change_high: float = 0.013,
        max_wait: int = 5,
        cooldown_items: int = 3,
    ) -> None:
        self.warmup_items = warmup_items
        self.max_prototypes = max_prototypes
        self.alpha = alpha
        self.init_cluster_threshold = init_cluster_threshold
        self.max_wait = max_wait
        self.cooldown_items = cooldown_items

        self.tau_silence = tau_silence
        self.tau_suspicious = tau_suspicious
        self.tau_change_low = tau_change_low
        self.tau_change_high = tau_change_high

        self.default_tau_silence = default_tau_silence
        self.default_tau_suspicious = default_tau_suspicious
        self.default_tau_change_low = default_tau_change_low
        self.default_tau_change_high = default_tau_change_high

        self.prototype_bank: list[PrototypeSlot] = []
        self.buffer: Deque[np.ndarray] = deque(maxlen=warmup_items)
        self.wait_count = 0
        self.cooldown_remaining = 0

    @property
    def prototype_ready(self) -> bool:
        return len(self.prototype_bank) > 0

    def run(self, items: Sequence[AggregatedTokenItem]) -> list[GateDecision]:
        """批量运行 gate，适合离线 demo 和测试。

        在线场景也可以逐个 item 调用 step()。
        """

        results: list[GateDecision] = []
        previous: np.ndarray | None = None
        for item in items:
            decision = self.step(item, previous)
            results.append(decision)
            previous = l2_normalize(item.embedding)
        return results

    def step(self, item: AggregatedTokenItem, previous: np.ndarray | None = None) -> GateDecision:
        """处理单个 frame/clip item。

        这里要保证任何情况下都返回 GateDecision，而不是半路吞异常。
        真正不可恢复的输入问题，比如维度错误，应在上游 aggregator 阶段处理。
        """

        z = l2_normalize(item.embedding)
        change = cosine_distance(z, previous)

        if not self.prototype_ready:
            self.buffer.append(z)
            self._try_initialize()
            return GateDecision(
                item_id=item.item_id,
                start_time=item.start_time,
                end_time=item.end_time,
                change=change,
                min_deviation=None,
                matched_prototype_id=None,
                decision=Signal.WAIT,
                prototype_ready=self.prototype_ready,
                trigger_slow_path=False,
                deviation_level="not_ready",
                reason="prototype_not_ready",
                evidence=item.evidence,
            )

        matched_id, deviation = self._match_nearest_prototype(z)
        decision, level, reason = self._decide(deviation, change)
        trigger_slow_path = decision == Signal.SUSPICIOUS

        if decision == Signal.SUSPICIOUS:
            self.cooldown_remaining = self.cooldown_items
            self.wait_count = 0
        elif decision == Signal.WAIT:
            self.wait_count += 1
            if self.wait_count > self.max_wait:
                decision = Signal.SUSPICIOUS
                trigger_slow_path = True
                reason += ";wait_timeout"
                self.cooldown_remaining = self.cooldown_items
        else:
            self.wait_count = 0

        self._maybe_update(matched_id, z, deviation, decision, item.item_id)

        return GateDecision(
            item_id=item.item_id,
            start_time=item.start_time,
            end_time=item.end_time,
            change=change,
            min_deviation=deviation,
            matched_prototype_id=matched_id,
            decision=decision,
            prototype_ready=True,
            trigger_slow_path=trigger_slow_path,
            deviation_level=level,
            reason=reason,
            evidence=item.evidence,
        )

    def _try_initialize(self) -> None:
        if self.prototype_ready or len(self.buffer) < self.warmup_items:
            return

        vectors = [l2_normalize(v) for v in self.buffer]
        self._calibrate_thresholds(vectors)

        for idx, vector in enumerate(vectors):
            self._add_or_merge_initial_prototype(vector, idx)

    def _add_or_merge_initial_prototype(self, vector: np.ndarray, item_id: int) -> None:
        if not self.prototype_bank:
            self.prototype_bank.append(PrototypeSlot(vector=vector, count=1, last_update_id=item_id))
            return

        matched_id, deviation = self._match_nearest_prototype(vector)
        if deviation <= self.init_cluster_threshold or len(self.prototype_bank) >= self.max_prototypes:
            slot = self.prototype_bank[matched_id]
            weight = 1.0 / float(slot.count + 1)
            slot.vector = l2_normalize((1.0 - weight) * slot.vector + weight * vector)
            slot.count += 1
            slot.last_update_id = item_id
        else:
            self.prototype_bank.append(PrototypeSlot(vector=vector, count=1, last_update_id=item_id))

    def _calibrate_thresholds(self, vectors: list[np.ndarray]) -> None:
        """用 warmup 数据自动估计阈值。

        如果用户显式传了阈值，就尊重用户设置。
        如果 warmup 数据异常，就回退默认值。
        """

        center = l2_normalize(np.stack(vectors, axis=0).mean(axis=0))
        deviations = np.asarray([cosine_distance(v, center) or 0.0 for v in vectors], dtype=np.float32)
        changes = np.asarray(
            [cosine_distance(vectors[i], vectors[i - 1]) or 0.0 for i in range(1, len(vectors))],
            dtype=np.float32,
        )

        self.tau_silence = self.tau_silence or self._safe_quantile(deviations, 0.75, self.default_tau_silence)
        self.tau_suspicious = self.tau_suspicious or (
            self._safe_quantile(deviations, 0.95, self.default_tau_suspicious) + 1e-3
        )
        self.tau_change_low = self.tau_change_low or self._safe_quantile(changes, 0.50, self.default_tau_change_low)
        self.tau_change_high = self.tau_change_high or (
            self._safe_quantile(changes, 0.95, self.default_tau_change_high) + 1e-3
        )

        if self.tau_suspicious <= self.tau_silence:
            self.tau_suspicious = self.tau_silence + 1e-3
        if self.tau_change_high <= self.tau_change_low:
            self.tau_change_high = self.tau_change_low + 1e-3

    def _safe_quantile(self, values: np.ndarray, q: float, default: float) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return default
        return float(np.quantile(values, q))

    def _match_nearest_prototype(self, z: np.ndarray) -> tuple[int, float]:
        distances = [cosine_distance(z, slot.vector) or 0.0 for slot in self.prototype_bank]
        matched_id = int(np.argmin(distances))
        return matched_id, float(distances[matched_id])

    def _decide(self, deviation: float, change: float | None) -> tuple[Signal, str, str]:
        assert self.tau_silence is not None
        assert self.tau_suspicious is not None
        assert self.tau_change_low is not None
        assert self.tau_change_high is not None

        reasons: list[str] = []
        if deviation <= self.tau_silence:
            level = "normal"
            reasons.append("deviation_below_tau_silence")
        elif deviation >= self.tau_suspicious:
            level = "deviated"
            reasons.append("deviation_over_tau_suspicious")
        else:
            level = "borderline"
            reasons.append("deviation_between_thresholds")

        if change is None:
            reasons.append("change_not_available")
        elif change >= self.tau_change_high:
            reasons.append("change_over_tau_high")
        elif change <= self.tau_change_low:
            reasons.append("change_below_tau_low")
        else:
            reasons.append("change_between_thresholds")

        if deviation >= self.tau_suspicious or (change is not None and change >= self.tau_change_high):
            return Signal.SUSPICIOUS, level, ";".join(reasons)
        if deviation <= self.tau_silence and (change is None or change <= self.tau_change_low):
            return Signal.SILENCE, level, ";".join(reasons)
        return Signal.WAIT, level, ";".join(reasons)

    def _maybe_update(
        self,
        matched_id: int,
        z: np.ndarray,
        deviation: float,
        decision: Signal,
        item_id: int,
    ) -> None:
        """保守更新 prototype，防止异常污染 routine memory。"""

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return
        if decision != Signal.SILENCE:
            return
        if self.tau_silence is None or deviation > self.tau_silence:
            return

        if deviation <= self.tau_silence * 0.5:
            weight = 1.0 - self.alpha
        else:
            weight = 0.5 * (1.0 - self.alpha)

        if weight <= 0:
            return

        slot = self.prototype_bank[matched_id]
        slot.vector = l2_normalize((1.0 - weight) * slot.vector + weight * z)
        slot.count += 1
        slot.last_update_id = item_id
```

### 16.4 `slow_path.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .decision_schema import GateDecision, Signal


@dataclass
class SlowPathRequest:
    """慢路径请求。

    只放必要上下文，避免把完整视频历史都扔给大模型。
    """

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
    """判断是否触发慢路径。

    这里故意只依赖 GateDecision，不直接依赖模型或视频对象，
    这样方便单元测试。
    """

    return decision.decision == Signal.SUSPICIOUS or decision.trigger_slow_path


def build_slow_path_prompt(decision: GateDecision) -> str:
    """根据原型 gate 的证据构造大模型 prompt。"""

    return (
        "你是一个流式视频理解助手。"
        "当前片段被原型向量 gate 判断为可能偏离常规模式。"
        "请结合当前帧内容，简洁说明发生了什么变化，是否值得响应。\n"
        f"Gate reason: {decision.reason}\n"
        f"Deviation: {decision.min_deviation}\n"
        f"Matched prototype: {decision.matched_prototype_id}\n"
    )


def run_slow_path_safely(request: SlowPathRequest, runner) -> SlowPathResult:
    """安全调用慢路径。

    runner 可以是 LiveStar runner，也可以是其他大模型 runner。
    任何慢路径异常都不应该中断 prototype gate 主流程。
    """

    try:
        text = runner(request)
        return SlowPathResult(item_id=request.evidence.get("item_id", -1), success=True, text=text)
    except Exception as exc:
        return SlowPathResult(
            item_id=request.evidence.get("item_id", -1),
            success=False,
            text=None,
            warning=f"slow_path_failed: {exc}",
        )
```

## 17. 生成完整项目时的注意事项

如果后续让代码生成器或另一个 agent 按本文档生成完整项目，必须遵守：

1. 优先保证 `simple backend` 可运行。
2. `livestar backend` 必须是可选依赖，不能让没有 CUDA 或没有权重的环境直接崩。
3. 所有新增函数都要有中文注释，特别是原型更新、阈值校准、慢路径触发原因。
4. 所有 CSV 字段都来自 `GateDecision`，不要散落多个临时字典。
5. 任何慢路径错误只写 warning，不中断主流程。
6. 不要删除当前已有的 `PrototypeGate`，新版 `PrototypeBankGate` 先作为增强模块并行存在。
7. 测试最少覆盖：
   - 空 token 报错。
   - warmup 后 prototype ready。
   - 正常样本输出 SILENCE。
   - 大偏离样本输出 SUSPICIOUS。
   - cooldown 期间不更新 prototype。
