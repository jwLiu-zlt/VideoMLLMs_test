# RDT-Gate Demo

这是一个 proof-of-concept 项目，用来验证流式视频触发策略：

- Prototype-based RDT-Gate：维护常规动态原型，判断当前 clip 是否偏离 routine pattern。

目标是验证：RDT-Gate 是否能在高动态但低信息增益的普通阶段保持 `SILENCE`，并在偏离 routine pattern 时输出 `SUSPICIOUS`。

## 安装

```bash
pip install -r requirements.txt
```

默认 `simple` embedding 使用 OpenCV 手工特征，不需要下载大模型。

## 默认运行

```bash
python main.py
```

当前默认配置针对 `data/1.mp4`，结果写入 `outputs/1mp4/`。事件标注不再硬编码到命令行默认值里；程序会优先从 `event_annotations.json` 按视频路径读取。

也可以使用 YAML 配置运行：

```bash
python main.py --config configs/default.yaml
python main.py --config configs/bap1.yaml
python main.py --config configs/lna5.yaml
```

命令行参数会覆盖 YAML 中的同名配置，例如：

```bash
python main.py --config configs/lna5.yaml --adaptive_alpha_high 2.0 --output_dir outputs/lna5_alpha2
```

每次运行都会把实际生效配置保存到输出目录的 `config.yaml`。

## 接入 LLM 慢路径

默认不会调用大语言模型。打开后，主流程先跑 RDT-Gate，只把触发条件满足的 clip 交给 LLM，并输出：

- `llm_results.csv`
- `llm_results.json`

先用 mock 后端跑通链路：

```bash
python main.py --config configs/default.yaml --llm_enable --llm_backend mock
```

接本地 HuggingFace 文本生成模型：

```bash
python main.py \
  --config configs/default.yaml \
  --llm_enable \
  --llm_backend hf \
  --llm_model_path /path/to/local/llm \
  --llm_device auto
```

如果环境里没有 `transformers`，先安装：

```bash
pip install transformers accelerate
```

当前接入的是文本 LLM slow path：输入是 gate 的 `decision/change/deviation/offset/threshold` 等结构化信号，默认只在 `SUSPICIOUS` clip 上调用。之后要换成 Qwen2.5-VL、Video-LLaVA 或 LiveStar，只需要在 `rdt_gate/llm_slow_path.py` 里新增 runner，把 `LLMSlowPathRequest.clip.frames` 传给视觉语言模型。

默认参数：

```bash
--video_path data/1.mp4
--clip_seconds 0.5
--frames_per_clip 8
--warmup_clips 4
--tau_silence None
--tau_suspicious None
--tau_change_low None
--tau_change_high None
--adjacent_enable false
--adj_tau_silence 0.01
--adj_tau_suspicious 0.013
--init_var_threshold 0.10
--init_change_threshold 0.10
--max_wait 99
--adaptive_window_size 32
--adaptive_alpha_high 3.0
--adaptive_min_thr 0.02
--adaptive_max_thr 0.25
--adaptive_warmup 8
--adaptive_min_interval 5
--adaptive_confirm_hits 2
--change_weight 1.0
--update_threshold_ratio 0.7
```

Prototype RDT-Gate 默认使用自适应阈值模块：

```text
Sliding Window + Median/MAD + Minimum Trigger Interval + Threshold Clipping
```

它接收融合后的 offset：

```text
offset = max(deviation, change_weight * change)
```

阈值由最近窗口内的 offset 中位数和 MAD 计算，并裁剪到 `[adaptive_min_thr, adaptive_max_thr]`，避免阈值过低导致连续误触发。Prototype 只在高置信 `SILENCE` 时更新：

```text
offset < update_threshold_ratio * threshold
```

这样可以减少事件片段被误判为 SILENCE 后污染 routine prototype。

## 事件标注

事件区间用于计算 `false_suspicious_rate` 和 `event_suspicious_rate`。每个视频必须使用自己的标注，不能复用其它场景的默认区间。

推荐写入 `event_annotations.json`：

```json
{
  "data/1.mp4": {
    "event_start": 4.0,
    "event_end": 5.5,
    "note": "Default annotation from the original README."
  }
}
```

也可以在命令行显式传入：

```bash
python main.py --video_path data/2.mp4 --event_start 6.0 --event_end 8.5
```

如果某个视频没有标注，事件相关指标会输出 `null`，并且不会生成 `metrics_comparison.png`，避免把错误区间当作真实评估。

## 运行其它真实视频

```bash
python main.py --video_path data/2.mp4
```

结果会自动写入 `outputs/2mp4/`。如果 `event_annotations.json` 中没有 `data/2.mp4` 的事件时间，事件相关指标会保持 `null`。补齐标注后再用于评估。

## 运行 Synthetic Demo

```bash
python main.py --use_synthetic_demo
```

synthetic demo 会生成 30 个 clip embedding，其中 20-24 秒默认是异常事件区间，结果写入 `outputs/synthetic_demo/`。

## LiveStar 视频 token Demo

LiveStar 中对应链路是：

- `LiveStar/inference/demo_ui.py::load_video`: 使用 `decord.VideoReader` 抽帧，将每帧转成 PIL 图像，做 dynamic tiling，并按 ImageNet mean/std 归一化成 `pixel_values`。
- `LiveStar/inference/modeling_livestar_chat.py::extract_feature`: 调用 `self.vision_model(pixel_values)`，去掉 CLS token，经过 pixel shuffle、token merge 和 `mlp1`，得到喂给语言模型的视觉 token。

本项目已把这条链路转成 `rdt_gate/livestar_tokens.py`，并提供测试 demo：

```bash
python test/damo1.py \
  --video_path data/1.mp4 \
  --backend simple \
  --sample_fps 1 \
  --save_path outputs/vit_tokens_1fps.pt
```

默认 `simple` backend 不需要完整 LiveStar 大模型，它复刻 LiveStar 的视频抽帧、dynamic tiling、归一化流程，并用一个本地 ViT-style patch tokenizer 生成 token，适合验证输入视频到 token 的数据形状。

如果你已经有完整的 LiveStar checkpoint，可以使用真实视觉编码器：

```bash
python test/damo1.py \
  --video_path data/1.mp4 \
  --backend livestar \
  --model_path ../LiveStar/inference \
  --device cuda \
  --dtype bfloat16 \
  --save_path outputs/livestar_tokens_1fps.pt
```

输出 `.pt` 文件包含 `frame_times`、`num_patches_list`、`pixel_values_shape`、`tokens` 和 `tokens_shape`。

如果要基于视觉 token 生成 routine prototype 向量：

```bash
python test/damo1.py \
  --video_path data/1.mp4 \
  --backend simple \
  --sample_fps 1 \
  --prototype_enable \
  --save_path outputs/token_prototype_demo.pt
```

这会额外保存：

- `frame_embeddings`: 每帧 token 池化后的归一化向量。
- `prototype_vector`: 默认是多原型 bank，shape 为 `[num_prototypes, dim]`；如果使用 `--prototype_gate single`，则为旧版单原型向量。
- `outputs/token_prototype_demo_prototype.csv`: 每帧的 change、deviation、decision、is_deviated、deviation_level、trigger_slow_path 和 reason。

默认 `--prototype_gate bank` 会启用优化后的 PrototypeBankGate：

- warmup 后自动估计 silence / suspicious / change 阈值。
- 维护多个 routine prototype，避免单中心原型无法覆盖多种正常状态。
- 只用高置信 `SILENCE` 样本保守更新，`SUSPICIOUS` 后进入 cooldown，避免异常污染 routine memory。
- CSV 额外包含 `matched_prototype_id`，方便解释当前片段匹配到哪个常规模式。

可选参数示例：

```bash
python test/damo1.py \
  --video_path data/1.mp4 \
  --backend simple \
  --sample_fps 1 \
  --prototype_enable \
  --prototype_gate bank \
  --prototype_pool tile_mean \
  --clip_window 2 \
  --warmup_frames 4 \
  --max_prototypes 4 \
  --save_path outputs/token_prototype_bank_demo.pt
```

如果需要回到旧版单原型逻辑：

```bash
python test/damo1.py \
  --video_path data/1.mp4 \
  --backend simple \
  --sample_fps 1 \
  --prototype_enable \
  --prototype_gate single \
  --prototype_pool mean \
  --save_path outputs/token_prototype_single_demo.pt
```

## EverOS 记忆集成

EverOS 可选启用，用于把每次实验的配置、指标和报告保存为 agent memory，方便之后检索历史阈值调参经验。API key 只从环境变量读取，不要写入代码或提交到仓库。

```bash
export EVEROS_API_KEY="your_api_key"
python -m pip install -i https://pypi.org/simple everos  # 当前镜像没有 everos 时使用
python main.py --everos_enable --everos_user_id damo_0526_user
```

检索历史实验上下文并写入 `everos_context.md`：

```bash
python main.py \
  --everos_enable \
  --everos_user_id damo_0526_user \
  --everos_search "prior RDT-Gate threshold tuning for high dynamic video"
```

EverOS 官方 SDK 需要 Python 3.9+；如果当前环境低于 3.9，项目会自动使用标准库 HTTP fallback 调用 EverOS API。

## 输出文件

- `outputs/<video_name>/prototype_results.csv`: RDT-Gate 每个 clip 的 change、deviation、decision。
- `outputs/<video_name>/adjacent_results.csv`: 只有传入 `--adjacent_enable` 时生成，相邻基线每个 clip 的 change、decision。
- `outputs/<video_name>/metrics.json`: 信号数量、误触发率、事件触发率和 Slow Path 触发率。
- `outputs/<video_name>/change_scores.png`: 相邻 change 分数时间线。
- `outputs/<video_name>/prototype_deviation.png`: prototype deviation 时间线。
- `outputs/<video_name>/decision_timeline.png`: RDT-Gate 决策时间线；启用 adjacent 时会额外显示 baseline。
- `outputs/<video_name>/signal_counts.png`: 信号数量柱状图。
- `outputs/<video_name>/metrics_comparison.png`: 有事件标注时生成的关键指标图。
- `outputs/<video_name>/report.md`: 自动实验报告。
- `outputs/<video_name>/everos_status.json`: 启用 EverOS 时生成，记录保存状态、session_id 和检索数量。
- `outputs/<video_name>/everos_context.md`: 传入 `--everos_search` 时生成，记录检索到的历史记忆上下文。

默认情况下，每个视频会写入自己的子目录，例如 `outputs/1mp4/`、`outputs/2mp4/`、`outputs/3mp4/`。如果显式传入 `--output_dir`，则使用传入路径。

## 如何理解结果

`change_scores.png` 反映相邻 clip 差异；在高动态普通阶段它可能仍然较高。

`prototype_deviation.png` 反映当前 clip 是否偏离常规动态原型。理想情况下，routine 阶段 deviation 较低，事件阶段 deviation 升高。

`decision_timeline.png` 和 `signal_counts.png` 用于观察 RDT-Gate 是否在普通动态阶段输出更多 `SILENCE`，并在事件阶段保持 `SUSPICIOUS` 敏感性。
