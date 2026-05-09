# BFNet Synthetic Experiment Summary

记录时间：2026-05-09

本文档记录 synthetic dataset A 上 BFNet / jSSA 的历史测试结果。部分 1024、2048 和权重消融文件已经删除，下面的数值来自当时的评估输出和训练记录。

## 误差口径

所有机制误差均使用论文中的 componentwise 等效节面口径：

- 分别构造预测机制和真值机制的两组等效节面；
- 计算 4 种组合下 strike / dip / rake 的误差；
- strike 与 rake 使用周期最小角差；
- dip 使用绝对差；
- 每个角度分别取 4 组组合中的最小值；
- 表中格式均为 `mean / median`，单位为度。

## 当前保留文件

当前只保留 5000 tau_center 数据集及两套最终模型：

- `synthetic/waveforms_a_5000`
- `synthetic/bfnet_samples_a_5000_tau_center`
- `model/bfnet_synthetic_a_5000_tau_center_paper.pt/.ckpt/.json`
- `model/bfnet_synthetic_a_5000_tau_center_aux002.pt/.ckpt/.json`
- `result/bfnet_synthetic_a_5000_tau_center_paper_metrics.json`
- `result/bfnet_synthetic_a_5000_tau_center_paper_predictions.csv`
- `result/bfnet_synthetic_a_5000_tau_center_paper_train_log.csv`
- `result/bfnet_synthetic_a_5000_tau_center_aux002_metrics.json`
- `result/bfnet_synthetic_a_5000_tau_center_aux002_predictions.csv`
- `result/bfnet_synthetic_a_5000_tau_center_aux002_train_log.csv`

## 主要实验结果

| 数据集 / 版本 | 状态 | best val loss | BFNet strike | BFNet dip | BFNet rake | jSSA strike | jSSA dip | jSSA rake |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 origin-slice paper | 已删除 | 0.09144 | 15.94 / 9.59 | 4.48 / 3.05 | 19.27 / 10.96 | 7.94 / 5.11 | 3.82 / 3.09 | 19.16 / 14.32 |
| 1024 tau_center paper | 已删除 | 0.08737 | 13.76 / 8.09 | 3.87 / 2.49 | 20.54 / 10.89 | 7.90 / 5.04 | 3.68 / 2.96 | 18.84 / 14.21 |
| 2048 tau_center paper, old run | 已删除 | 0.07018 | 9.23 / 4.67 | 2.84 / 2.03 | 14.26 / 9.35 | 8.20 / 5.05 | 3.82 / 3.11 | 18.86 / 14.90 |
| 2048 tau_center paper, rebuilt run | 已删除 | 0.06915 | 12.87 / 6.02 | 3.38 / 2.42 | 19.13 / 10.48 | 8.20 / 5.05 | 3.82 / 3.11 | 18.86 / 14.90 |
| 2048 paper -> aux002, alpha=0.02 | 已删除 | 未保留 | 9.57 / 4.56 | 2.86 / 2.03 | 15.39 / 8.55 | 8.20 / 5.05 | 3.82 / 3.11 | 18.86 / 14.90 |
| 2048 paper -> aux005, alpha=0.05 | 已删除 | 未保留 | 8.72 / 4.22 | 2.63 / 1.92 | 13.68 / 7.77 | 8.20 / 5.05 | 3.82 / 3.11 | 18.86 / 14.90 |
| 5000 tau_center paper | 保留 | 0.05737 | 9.55 / 4.79 | 2.77 / 1.95 | 15.72 / 9.33 | 7.91 / 4.90 | 3.76 / 3.02 | 18.42 / 14.08 |
| 5000 paper -> aux001, alpha=0.01 | 已删除 | 0.04946 | 8.19 / 3.97 | 2.45 / 1.78 | 14.26 / 8.57 | 7.91 / 4.90 | 3.76 / 3.02 | 18.42 / 14.08 |
| 5000 paper -> aux002, alpha=0.02 | 保留 | 0.05362 | 6.95 / 3.48 | 2.26 / 1.60 | 12.93 / 8.11 | 7.91 / 4.90 | 3.76 / 3.02 | 18.42 / 14.08 |
| 5000 paper -> aux005, alpha=0.05 | 已删除 | 约 0.05816 | 7.38 / 3.64 | 2.23 / 1.62 | 13.80 / 8.15 | 7.91 / 4.90 | 3.76 / 3.02 | 18.42 / 14.08 |

## 长尾统计

### 5000 paper

- BFNet strike > 45 deg: 202 / 5000
- BFNet strike > 60 deg: 113 / 5000
- BFNet rake > 45 deg: 323 / 5000
- BFNet rake > 60 deg: 240 / 5000

### 5000 aux002, alpha=0.02

- BFNet strike > 45 deg: 114 / 5000
- BFNet strike > 60 deg: 62 / 5000
- BFNet rake > 45 deg: 189 / 5000
- BFNet rake > 60 deg: 148 / 5000

### 5000 aux005, alpha=0.05

- BFNet strike > 45 deg: 122 / 5000
- BFNet strike > 60 deg: 78 / 5000
- BFNet rake > 45 deg: 237 / 5000
- BFNet rake > 60 deg: 191 / 5000

## 数据集与切片版本说明

### origin-slice

旧版本做法：直接使用真值 `origin_sample` 对应的亮度场切片作为 BFNet 输入。

特点：

- 仍然是 jSSA 亮度场，不是直接把波形送入 BFNet；
- 但没有搜索峰值时刻 tau*；
- 与论文中 `B(tau*)` 的描述不完全一致。

### tau_center

当前主版本做法：

- 在 `origin_sample +/- 40` 个采样点内搜索 tau；
- 选 tau* 时只在中心 `3 x 3 x 3` 空间小邻域内寻找最大亮度；
- 保存完整六维 `B(tau*)`，形状为 `(8, 8, 8, 24, 7, 24)`；
- 机制网格为 strike 24 点、dip 7 点、rake 24 点；
- 样本存储为 `float16`。

这个做法避免了全空间边界假峰把 tau* 拉偏，同时保留完整机制空间输入，更适合监督训练。

## 关键诊断结论

1. 1024 到 2048 时，增加数据量能明显改善 BFNet 机制误差。
2. 2048 到 5000 时，val loss 继续降低，但 paper 版机制角度误差没有同步明显下降。
3. 原因是训练目标是 moment tensor MSE，而评估是 strike / dip / rake 角度误差；二者在等效节面、低/高 dip、rake 周期边界附近并不完全一致。
4. BFNet paper 版主要问题是 strike/rake 长尾，而不是 dip。
5. 5000 paper 中，BFNet 的 dip 已经明显优于 jSSA，但 strike mean 被长尾拖住。
6. 等效 sin/cos 辅助损失可以明显压制 strike/rake 长尾。
7. 在 5000 上，`alpha=0.02` 是当前最佳权重；`alpha=0.05` 在 2048 上更好，但在 5000 上不如 `alpha=0.02`。

## 推荐工作流

### 论文版 BFNet 基线

先按论文版训练：

```powershell
python train_bfnet_synthetic.py --samples-csv synthetic/bfnet_samples_a_5000_tau_center/samples.csv --output model/bfnet_synthetic_a_5000_tau_center_paper.pt --checkpoint model/bfnet_synthetic_a_5000_tau_center_paper.ckpt --log-csv result/



bfnet_synthetic_a_5000_tau_center_paper_train_log.csv --stage1-momentum 0.9 --num-workers 0
```

### 改进版 BFNet

在论文版模型基础上做 aux 微调：

```powershell
python train_bfnet_synthetic.py --samples-csv synthetic/bfnet_samples_a_5000_tau_center/samples.csv --resume model/bfnet_synthetic_a_5000_tau_center_paper.pt --output model/bfnet_synthetic_a_5000_tau_center_aux002.pt --checkpoint model/bfnet_synthetic_a_5000_tau_center_aux002.ckpt --log-csv result/bfnet_synthetic_a_5000_tau_center_aux002_train_log.csv --stage1-epochs 0 --stage2-epochs 40 --stage2-lr 5e-5 --stage2-min-lr 5e-6 --stage2-batch-size 8 --equiv-sincos-loss-weight 0.02 --num-workers 0
```

### 评估命令

```powershell
python evaluate_bfnet_synthetic.py --samples-csv synthetic/bfnet_samples_a_5000_tau_center/samples.csv --model model/bfnet_synthetic_a_5000_tau_center_aux002.pt --predictions-csv result/bfnet_synthetic_a_5000_tau_center_aux002_predictions.csv --metrics-json result/bfnet_synthetic_a_5000_tau_center_aux002_metrics.json --batch-size 4 --num-workers 0
```

## 当前最终结论

当前最好的模型是：

```text
model/bfnet_synthetic_a_5000_tau_center_aux002.pt
```

最终推荐结果：

| 方法 | Strike mean / median | Dip mean / median | Rake mean / median |
|---|---:|---:|---:|
| BFNet aux002 | 6.95 / 3.48 | 2.26 / 1.60 | 12.93 / 8.11 |
| jSSA | 7.91 / 4.90 | 3.76 / 3.02 | 18.42 / 14.08 |

该结果中，BFNet 的 strike、dip、rake 三个参数的 mean 和 median 均优于 jSSA。
