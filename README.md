# RDT-Gate Demo

这是一个 proof-of-concept 项目，用来对比两种流式视频触发策略：

- Prototype-based RDT-Gate：维护常规动态原型，判断当前 clip 是否偏离 routine pattern。
- Adjacent Similarity Baseline：只比较当前 clip 与上一 clip 的相似度。

目标是验证：在高动态但低信息增益的普通阶段，相邻相似度方法容易误触发；RDT-Gate 更适合输出 `SILENCE / WAIT / SUSPICIOUS`。

## 安装

```bash
pip install -r requirements.txt
```

默认 `simple` embedding 使用 OpenCV 手工特征，不需要下载大模型。

## 默认运行

```bash
python main.py
```

当前默认配置针对 `data/1.mp4`，事件标注为 `4.0s - 5.5s`，结果写入 `outputs/`。

默认参数：

```bash
--video_path data/1.mp4
--output_dir outputs
--clip_seconds 0.5
--frames_per_clip 8
--warmup_clips 4
--tau_silence 0.03
--tau_suspicious 0.08
--tau_change_low 0.01
--tau_change_high 0.013
--adj_tau_silence 0.01
--adj_tau_suspicious 0.013
--init_var_threshold 0.10
--init_change_threshold 0.10
--max_wait 99
--event_start 4.0
--event_end 5.5
```

## 运行其它真实视频

```bash
python main.py --video_path data/2.mp4 --output_dir outputs_2mp4
```

如果事件时间不同，需要同步传入新的标注和必要阈值。

## 运行 Synthetic Demo

```bash
python main.py --use_synthetic_demo --output_dir outputs_synthetic
```

synthetic demo 会生成 30 个 clip embedding，其中 20-24 秒默认是异常事件区间。

## 输出文件

- `outputs/prototype_results.csv`: RDT-Gate 每个 clip 的 change、deviation、decision。
- `outputs/adjacent_results.csv`: 相邻基线每个 clip 的 change、decision。
- `outputs/metrics.json`: 信号数量、误触发率、事件触发率和 Slow Path 触发率。
- `outputs/change_scores.png`: 相邻 change 分数时间线。
- `outputs/prototype_deviation.png`: prototype deviation 时间线。
- `outputs/decision_timeline.png`: 两种方法的决策时间线。
- `outputs/signal_counts.png`: 信号数量柱状图。
- `outputs/metrics_comparison.png`: 有事件标注时生成的关键指标图。
- `outputs/report.md`: 自动实验报告。

## 如何理解结果

`change_scores.png` 反映相邻 clip 差异；在高动态普通阶段它可能仍然较高。

`prototype_deviation.png` 反映当前 clip 是否偏离常规动态原型。理想情况下，routine 阶段 deviation 较低，事件阶段 deviation 升高。

`decision_timeline.png` 和 `signal_counts.png` 用于观察 RDT-Gate 是否在普通动态阶段输出更多 `SILENCE`，并在事件阶段保持 `SUSPICIOUS` 敏感性。
