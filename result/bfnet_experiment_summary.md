# BFNet 多层速度模型实验总结

记录时间：2026-05-16

本阶段实验围绕多层速度模型下的 jSSA 与 BFNet 机制反演展开，核心问题是：在亮度场计算中，将辐射强度方向从几何直达方向改为多层射线追踪得到的震源出射方向（source-takeoff）后，是否能稳定改善 jSSA 与 BFNet 的机制误差。

表中误差均为 `mean / median`，单位为度。机制误差采用 `componentwise equivalent-plane` 口径：预测机制与真实机制分别展开为等价节面，对 strike、dip、rake 三个分量分别取等价节面组合中的最小误差。

## 实验设置

数据集 A 使用 5000 个多层射线追踪合成事件，台站为放射状分布：

- 波形：`synthetic/waveforms_ml_ray_5000`
- geometry 输入：`synthetic/bfnet_samples_ml_ray_5000_geom`
- source-takeoff 输入：`synthetic/bfnet_samples_ml_ray_5000_source`
- 台站：`conf/station_synth_dataset_a.xyz`

数据集 B 使用 5000 个多层射线追踪合成事件，台站为 8x8 正方形分布：

- 波形：`synthetic/waveforms_ml_ray_5000_b`
- geometry 输入：`synthetic/bfnet_samples_ml_ray_5000_b_geom`
- source-takeoff 输入：`synthetic/bfnet_samples_ml_ray_5000_b_source`
- 台站：`conf/station_synth_dataset_b.xyz`

两套数据均使用多层速度模型 `conf/vel_synth_jssa.txt`。geometry 与 source-takeoff 的唯一区别是理论辐射强度方向：geometry 使用震源到台站的几何直达方向，source-takeoff 使用 `generate_tt()` 输出的 `incident` 和 `azimuth`，结合多层速度模型换算震源端出射方向。

## 亮度场生成

BFNet 输入来自 jSSA 最终亮度场。每个事件单独构建局部搜索网格，空间网格为 `8 x 8 x 8`，机制网格为 `24 x 7 x 24`。亮度场计算在给定射线追踪走时表、理论辐射强度表和观测波形的基础上，对空间网格、机制网格和 tau 采样时刻进行对齐叠加。

统一使用的关键设置为：

- `--tt-model raytrace`
- `--tau-search-radius-samples 40`
- `--tau-pick-mode center`
- `--tau-pick-spatial-radius 1`
- `--brightness-backend torch`
- `--dtype float16`

tau-center 策略用于在事件中心附近选择 tau，减弱边界假峰对亮度场切片的影响。它不是本阶段的方法创新点，只作为稳定构建 BFNet 输入的工程设置。

## 训练策略

最终保留三阶段训练策略：

```text
stage1: SGD, 100 epochs, aux=0.02
stage2: AdamW, 100 epochs, aux=0.05
finetune: AdamW, 50 epochs, aux=0.05, low lr
```

其中 aux 指等价节面 sin/cos 辅助损失权重。stage1 先训练并保存 validation best；stage2 从 stage1 best 开始；finetune 从 stage2 best 开始，用更低学习率继续精修。对照实验表明，单纯将 stage2 拉长到 150 轮不如 `stage2=100 + low-lr finetune=50` 稳定。

## 数据集 A 结果

| 输入方向 | 方法 | Strike | Dip | Rake | 三者误差和 |
|---|---|---:|---:|---:|---:|
| geometry | jSSA | 7.83 / 4.46 | 3.72 / 3.03 | 17.75 / 13.19 | 29.30 / 20.68 |
| geometry | BFNet 三阶段 | 6.23 / 2.76 | 2.02 / 1.42 | 12.36 / 7.29 | 20.61 / 11.47 |
| source-takeoff | jSSA | 7.02 / 4.14 | 3.34 / 2.51 | 15.02 / 10.64 | 25.38 / 17.29 |
| source-takeoff | BFNet 三阶段 | 5.91 / 3.00 | 1.96 / 1.37 | 11.19 / 6.87 | 19.06 / 11.24 |

数据集 A 上可以得到三个结论：

1. source-takeoff 相较 geometry 的 jSSA 在 strike、dip、rake 的均值和中位数上均有改进。
2. BFNet 相较各自对应的 jSSA 在 strike、dip、rake 的均值和中位数上均有改进。
3. source-takeoff BFNet 相较 geometry BFNet 整体更优，mean 总误差和 median 总误差均更低。

## 数据集 B 结果

| 输入方向 | 方法 | Strike | Dip | Rake | 三者误差和 |
|---|---|---:|---:|---:|---:|
| geometry | jSSA | 9.26 / 5.33 | 3.73 / 2.97 | 16.56 / 11.58 | 29.55 / 19.88 |
| geometry | BFNet 三阶段 | 6.34 / 3.31 | 2.05 / 1.45 | 10.66 / 6.46 | 19.06 / 11.23 |
| source-takeoff | jSSA | 8.77 / 5.01 | 3.71 / 2.70 | 14.48 / 9.14 | 26.96 / 16.85 |
| source-takeoff | BFNet 三阶段 | 6.41 / 3.48 | 2.01 / 1.38 | 11.21 / 6.30 | 19.63 / 11.15 |

数据集 B 上同样可以得到前两个结论：

1. source-takeoff 相较 geometry 的 jSSA 在 strike、dip、rake 的均值和中位数上均有改进。
2. BFNet 相较各自对应的 jSSA 在 strike、dip、rake 的均值和中位数上均有改进。

对于 source-takeoff BFNet 相较 geometry BFNet，数据集 B 的结果更接近：source-takeoff 的 median 总误差略低，且 dip 与 rake median 更优；geometry 的 mean 总误差略低，主要来自 strike mean 和 rake mean。因而更严谨的表述是：source-takeoff 对 jSSA 的改进最稳定；BFNet 对 jSSA 的改进也稳定；source-takeoff 对 BFNet 的进一步提升在数据集 A 更明显，在数据集 B 主要体现在中位数和部分分量上。

## 跨台站分布模型泛化测试

为检验模型泛化能力，使用数据集 A 上训练得到的 `seed20260506` 三阶段 BFNet 模型，直接评估 500 个 dataset B 事件，不在 dataset B 上重新训练。该实验与上一节的 dataset B 重新训练实验不同，主要用于考察 A 台站分布训练出的模型能否迁移到 8x8 正方形台站分布。当前 500B 验证集使用 `seed=20260523` 重新生成。

| 输入方向 | 方法 | Strike | Dip | Rake | 三者误差和 |
|---|---|---:|---:|---:|---:|
| geometry | jSSA | 9.69 / 5.55 | 3.92 / 3.30 | 17.00 / 11.91 | 30.61 / 20.75 |
| geometry | BFNet 三阶段 | 9.25 / 4.43 | 2.90 / 1.97 | 14.75 / 9.07 | 26.90 / 15.47 |
| source-takeoff | jSSA | 9.14 / 5.40 | 3.78 / 2.79 | 16.23 / 10.52 | 29.16 / 18.72 |
| source-takeoff | BFNet 三阶段 | 8.69 / 4.05 | 2.74 / 2.22 | 13.19 / 7.64 | 24.61 / 13.91 |

该泛化测试表明：source-takeoff jSSA 仍优于 geometry jSSA；geometry BFNet 和 source-takeoff BFNet 相较各自对应的 jSSA，在 strike、dip、rake 的均值和中位数上均有改进，没有出现分量例外。整体上，source-takeoff BFNet 的 mean 总误差和 median 总误差均低于 geometry BFNet。

## 综合结论

两个台站分布下，source-takeoff 辐射方向均能稳定改善 jSSA 机制误差，说明多层介质中使用震源出射方向计算辐射强度比几何直达方向更符合合成数据的物理生成机制。

BFNet 在 geometry 和 source-takeoff 两套输入下均显著优于各自 jSSA，说明神经网络能够进一步校正 jSSA 亮度场中的机制误差。三阶段训练中，额外低学习率 finetune 能进一步降低机制误差，优于单纯拉长 stage2 训练。

source-takeoff BFNet 相较 geometry BFNet 的优势具有数据集依赖性：在数据集 A 上整体优势明确；在数据集 B 上整体表现接近，source-takeoff 的中位数总误差略优，而 geometry 的均值总误差略优。这说明 source-takeoff 的物理修正对传统 jSSA 的增益最稳定，对 BFNet 的增益还受到台站分布、长尾样本和训练目标的共同影响。

## 保留文件

数据集 A 最终模型与结果：

- `model/seed20260506_final/`
- `result/seed20260506_final/`

数据集 B 最终模型与结果：

- `model/dataset_b_5000/`
- `result/dataset_b_5000/`

跨台站分布模型泛化测试：

- `result/generalization_500_b/`

主数据集 jSSA 基准：

- `result/dataset_a_5000/jssa_ml_ray_5000_geom_vs_source_metrics.json`
- `result/dataset_a_5000/jssa_ml_ray_5000_geom_vs_source_paired_errors.csv`

辅助诊断：

- `result/diagnostics/station_azimuth_strike_diagnostics.json`
- `result/diagnostics/station_azimuth_strike_diagnostics.csv`

