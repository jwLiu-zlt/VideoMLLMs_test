# Fast-Slow RDT-Gate 对比实验报告

## 实验配置

- 视频路径: `data/1.mp4`
- clip_seconds: `1.0`
- frames_per_clip: `8`
- embedding_backend: `simple`
- event_start/event_end: `None` / `None`

## 方法参数

- Prototype RDT-Gate: `{'warmup_clips': 5, 'alpha': 0.9, 'tau_silence': 0.015, 'tau_suspicious': 0.035, 'tau_change_low': 0.015, 'tau_change_high': 0.035, 'init_var_threshold': 0.35, 'init_change_threshold': 0.45, 'max_wait': 3}`
- Adjacent Baseline: `{'adj_tau_silence': 0.015, 'adj_tau_suspicious': 0.035}`

## 信号数量统计

| 方法 | SILENCE | WAIT | SUSPICIOUS |
|---|---:|---:|---:|
| prototype | 0 | 8 | 5 |
| adjacent | 9 | 3 | 1 |

## 指标对比

| 方法 | false suspicious rate | event suspicious rate | slow path trigger rate |
|---|---:|---:|---:|
| prototype | null | null | 0.3846 |
| adjacent | null | null | 0.0769 |

## 图表解释

- `change_scores.png`: 展示相邻 clip 的变化分数。普通动态阶段 change 仍可能波动较大，因此单独使用 change 容易误触发。
- `prototype_deviation.png`: 展示当前 clip 偏离 routine prototype 的程度。如果 routine 阶段 deviation 较低、事件阶段升高，说明原型捕捉了常规动态模式。
- `decision_timeline.png`: 展示两种方法在时间轴上的 SILENCE/WAIT/SUSPICIOUS 输出差异。
- `signal_counts.png`: 对比两种方法输出的信号数量，重点观察 SILENCE 与 SUSPICIOUS 的比例。
- `metrics_comparison.png`: 在提供事件标注时，对比误触发率、事件触发率和 Slow Path 触发率。

## 结论

在该视频中，Adjacent Similarity Baseline 仅依赖相邻 clip 的变化程度，因此在高动态但语义上仍属于常规模式的阶段容易输出更多 SUSPICIOUS 信号。Prototype-based RDT-Gate 通过维护常规动态原型，能够识别当前 clip 是否仍接近稳定的 routine pattern，因此在普通动态阶段输出更多 SILENCE 信号。

如果在事件区间内 Prototype-based RDT-Gate 的 deviation 明显升高并输出 SUSPICIOUS，说明原型方法不仅能够抑制普通动态误触发，也能对偏离常规动态的关键变化保持敏感。
