# Source-side takeoff 辐射强度修正实验总结

记录时间：2026-05-16  
整理更新：2026-06-11

本阶段实验围绕层状速度模型下的 jSSA 与 BFNet 震源机制反演展开，核心问题是：在理论辐射强度计算中，将传统的 **geometric source-receiver direction** 替换为由多层射线追踪得到的 **source-side takeoff direction** 后，是否能够改善 jSSA 机制反演结果，并进一步影响 BFNet 的机制预测表现。

表中误差均为 `mean / median`，单位为度。机制误差采用 `componentwise equivalent-plane` 口径计算，即考虑断层面与辅助面的等价性，对预测机制和真实机制的等价节面组合分别计算 strike、dip、rake 误差，并取对应分量的最小值。

## 1. 实验数据与输入构建

### 1.1 合成数据集

Dataset A 使用 5000 个多层射线追踪合成事件，台站为放射状分布：

- 波形：`synthetic/waveforms_ml_ray_5000`
- geometric source-receiver 输入：`synthetic/bfnet_samples_ml_ray_5000_geom`
- source-side takeoff 输入：`synthetic/bfnet_samples_ml_ray_5000_source`
- 台站文件：`conf/station_synth_dataset_a.xyz`

Dataset B 使用正方形网格台站分布，用于检验方法在不同观测几何下的适用性：

- 台站文件：`conf/station_synth_dataset_b.xyz`
- 当前保留的跨台站泛化子集：`synthetic/waveforms_ml_ray_500_b`
- geometric source-receiver 输入：`synthetic/bfnet_samples_ml_ray_500_b_geom`
- source-side takeoff 输入：`synthetic/bfnet_samples_ml_ray_500_b_source`

两套数据均使用层状速度模型 `conf/vel_synth_jssa.txt`。两种输入的唯一区别是理论辐射强度所使用的传播方向：geometric source-receiver 使用候选震源到台站的几何直达方向；source-side takeoff 使用多层射线追踪输出的接收端入射角和方位角，并结合射线参数守恒关系换算震源端出射方向。

### 1.2 jSSA 亮度场构建

BFNet 输入来自 jSSA 机制亮度场。每个事件在局部空间搜索网格内计算亮度场，空间网格为 `8 x 8 x 8`，机制网格为 `24 x 7 x 24`。在给定走时表、理论辐射强度表和观测波形的条件下，jSSA 对空间网格、机制网格和 tau 采样时刻进行联合扫描。

统一使用的关键设置为：

- `--tt-model raytrace`
- `--tau-search-radius-samples 40`
- `--tau-pick-mode center`
- `--tau-pick-spatial-radius 1`
- `--brightness-backend torch`
- `--dtype float16`

tau-center 策略用于在事件中心附近选择 tau，减弱边界假峰对亮度场切片的影响。该策略不是本文方法创新点，只作为稳定构建 BFNet 输入的工程设置。

### 1.3 BFNet 训练策略

最终保留三阶段训练策略：

```text
stage1: SGD, 100 epochs, aux=0.02
stage2: AdamW, 100 epochs, aux=0.05
finetune: AdamW, 50 epochs, aux=0.05, low lr
```

其中 `aux` 表示等价节面 sin/cos 辅助损失权重。stage1 先训练并保存 validation best；stage2 从 stage1 best 开始；finetune 从 stage2 best 开始，用更低学习率继续精修。对照实验表明，单纯将 stage2 拉长到 150 轮不如 `stage2=100 + low-lr finetune=50` 稳定。

## 2. Dataset A 结果

| 输入方向 | 方法 | Strike | Dip | Rake | 三者误差和 |
|---|---|---:|---:|---:|---:|
| Geometric source-receiver | jSSA | 7.83 / 4.46 | 3.72 / 3.03 | 17.75 / 13.19 | 29.30 / 20.68 |
| Geometric source-receiver | BFNet 三阶段 | 6.23 / 2.76 | 2.02 / 1.42 | 12.36 / 7.29 | 20.61 / 11.47 |
| Source-side takeoff | jSSA | 7.02 / 4.14 | 3.34 / 2.51 | 15.02 / 10.64 | 25.38 / 17.29 |
| Source-side takeoff | BFNet 三阶段 | 5.91 / 3.00 | 1.96 / 1.37 | 11.19 / 6.87 | 19.06 / 11.24 |

Dataset A 上可以得到三个结论：

1. Source-side takeoff 相较 geometric source-receiver 能降低 jSSA 的 strike、dip 和 rake 误差。
2. BFNet 相较各自对应的 jSSA 结果进一步降低机制误差，说明完整亮度场信息对机制预测有效。
3. Source-side takeoff BFNet 的总误差均值和中位数均低于 geometric source-receiver BFNet，说明修正后的亮度场也能改善 BFNet 输入质量。

## 3. Dataset B 结果

| 输入方向 | 方法 | Strike | Dip | Rake | 三者误差和 |
|---|---|---:|---:|---:|---:|
| Geometric source-receiver | jSSA | 9.26 / 5.33 | 3.73 / 2.97 | 16.56 / 11.58 | 29.55 / 19.88 |
| Geometric source-receiver | BFNet 三阶段 | 6.34 / 3.31 | 2.05 / 1.45 | 10.66 / 6.46 | 19.06 / 11.23 |
| Source-side takeoff | jSSA | 8.77 / 5.01 | 3.71 / 2.70 | 14.48 / 9.14 | 26.96 / 16.85 |
| Source-side takeoff | BFNet 三阶段 | 6.41 / 3.48 | 2.01 / 1.38 | 11.21 / 6.30 | 19.63 / 11.15 |

Dataset B 上，source-side takeoff 对 jSSA 的改进仍然稳定，三项机制参数的均值和中位数总体降低。BFNet 相较各自对应的 jSSA 也均有明显改进。对于 BFNet 两种输入之间的比较，Dataset B 的结果更接近：source-side takeoff 的总误差中位数略低，dip 与 rake 中位数更优；geometric source-receiver 的总误差均值略低，主要受 strike mean 和 rake mean 影响。因此更严谨的表述是：source-side takeoff 对 jSSA 的增益最稳定；对 BFNet 的进一步增益在 Dataset A 更明显，在 Dataset B 主要体现在中位数和部分机制分量上。

## 4. 跨台站分布泛化测试

为检验 BFNet 的跨台站分布泛化能力，使用 Dataset A 上训练得到的三阶段 BFNet 模型，直接评估 Dataset B 的 500 个事件子集，不在 Dataset B 上重新训练。该实验用于考察模型从放射状台站分布迁移到 `8 x 8` 正方形台站分布时，source-side takeoff 修正是否仍然有效。

| 输入方向 | 方法 | Strike | Dip | Rake | 三者误差和 |
|---|---|---:|---:|---:|---:|
| Geometric source-receiver | jSSA | 9.69 / 5.55 | 3.92 / 3.30 | 17.00 / 11.91 | 30.61 / 20.75 |
| Geometric source-receiver | BFNet 三阶段 | 9.25 / 4.43 | 2.90 / 1.97 | 14.75 / 9.07 | 26.90 / 15.47 |
| Source-side takeoff | jSSA | 9.14 / 5.40 | 3.78 / 2.79 | 16.23 / 10.52 | 29.16 / 18.72 |
| Source-side takeoff | BFNet 三阶段 | 8.69 / 4.05 | 2.74 / 2.22 | 13.19 / 7.64 | 24.61 / 13.91 |

跨台站泛化测试表明，source-side takeoff jSSA 仍优于 geometric source-receiver jSSA；两种 BFNet 均相较各自对应的 jSSA 降低机制误差。整体上，source-side takeoff BFNet 的总误差均值和中位数均低于 geometric source-receiver BFNet，说明该修正在台站分布改变时仍具有一定泛化价值。

## 5. 真实工区无标签验证

为检验 source-side takeoff 修正在真实数据中的适用性，使用宁夏煤层气压裂监测项目的连续波形进行 jSSA 对比实验。真实工区缺少可靠的 strike、dip、rake 真值标签，因此该实验采用台站留出验证和机制响应集中程度作为无标签评价指标。

真实工区验证保持波形、台站、速度模型、搜索网格、走时表和 jSSA 处理流程一致，仅改变理论辐射强度方向：

- Geometric source-receiver：使用震源到台站的几何直达方向计算理论辐射强度。
- Source-side takeoff：使用多层速度模型射线追踪得到的接收端入射角和方位角，换算震源端出射方向后计算理论辐射强度。

台站留出验证的做法是：每次随机留出 25% 台站，不参与 jSSA 机制反演；使用剩余台站得到最优机制后，再检验该机制对留出台站观测响应的解释能力。当前验证使用 50 个事件、3 次留出划分，共 150 组成对比较。

| 指标 | 含义 | Geometric source-receiver | Source-side takeoff | Source-side takeoff 更优次数 |
|---|---|---:|---:|---:|
| 台站残差 | 留出台站预测响应与观测响应的归一化残差，越低越好 | 0.9132 / 0.9322 | 0.8990 / 0.9242 | 102 / 150 |
| 振幅相关性 | 留出台站预测响应与观测响应的振幅相关性，越高越好 | -0.0212 / -0.0841 | -0.0091 / -0.0707 | 80 / 150 |
| 机制熵 | 机制亮度场在候选机制网格上的分散程度，越低表示机制响应越集中 | 0.9817 / 0.9823 | 0.9811 / 0.9809 | 127 / 150 |

表中数值为 `mean / median`。结果显示，source-side takeoff 在台站残差、振幅相关性和机制熵三个代理指标上均呈现改善趋势。其中，source-side takeoff 在 102/150 组留出验证中获得更低台站残差，在 127/150 组中获得更低机制熵，说明该修正倾向于提高留出台站解释能力，并使真实工区 jSSA 机制响应更加集中。振幅相关性的改善幅度较小，但成对比较中仍有 80/150 组 source-side takeoff 更优。

## 6. 综合结论

两个台站分布下，source-side takeoff 辐射方向均能稳定改善 jSSA 机制误差，说明在层状速度模型中，使用震源端出射方向计算理论辐射强度比使用几何直达方向更符合射线传播过程。

BFNet 在 geometric source-receiver 和 source-side takeoff 两套输入下均显著优于各自对应的 jSSA，说明神经网络能够进一步学习亮度场结构与机制参数之间的映射关系。Source-side takeoff 对 BFNet 的增益具有一定数据集依赖性：在 Dataset A 上整体优势明确，在 Dataset B 上整体表现接近，但在跨台站泛化测试中仍表现出更低的总误差。

真实工区无标签验证进一步表明，source-side takeoff 会改变 jSSA 的机制响应，并在台站留出残差、振幅相关性和机制熵等代理指标上呈现改善趋势。该结果与合成实验中的误差降低趋势相互补充，说明该修正方法具有嵌入现有 jSSA-BFNet 流程的应用潜力。

## 7. 结果文件组织

Dataset A 最终结果：

- `result/dataset_a_final/`
- `result/dataset_a_final/bfnet_geometric_source_receiver/`
- `result/dataset_a_final/bfnet_source_side_takeoff/`
- `result/dataset_a_final/jssa_geometric_vs_source_takeoff/`
- `result/dataset_a_final/bfnet_geometric_vs_source_takeoff_paired_stats.json`

Dataset B 最终结果：

- `result/dataset_b_final/`
- `result/dataset_b_final/bfnet_geometric_source_receiver/`
- `result/dataset_b_final/bfnet_source_side_takeoff/`
- `result/dataset_b_final/jssa_geometric_vs_source_takeoff/`
- `result/dataset_b_final/bfnet_geometric_vs_source_takeoff_paired_stats.json`

跨台站分布泛化测试：

- `result/cross_station_generalization_b500/`

真实工区无标签验证：

- `result/real_source_takeoff_jssa_holdout_50e3s/`

论文正文图片：

- `result/manuscript_figures/`

绘图脚本：

- `make_manuscript_figures.py`
- `make_graphical_abstract.py`

模型文件：

- `model/seed20260506_final/`
