# Source-side Takeoff Experiment Summary

Record date: 2026-05-16
Last updated: 2026-06-11

This document summarizes the experiments for source-side takeoff radiation correction in layered velocity models. The central question is whether replacing the conventional **geometric source-receiver direction** with the **source-side takeoff direction** obtained from multilayer ray tracing can improve jSSA focal mechanism inversion and further affect BFNet prediction performance.

All focal mechanism errors are reported as `mean / median` values in degrees. The errors are computed using a `componentwise equivalent-plane` metric. This means that the equivalence between the fault plane and the auxiliary plane is considered, and the strike, dip, and rake errors are computed over equivalent nodal-plane combinations between the predicted and true mechanisms. The minimum componentwise errors are then reported.

## 1. Experimental Data and Input Construction

### 1.1 Synthetic Datasets

Dataset A contains 5000 synthetic events generated with multilayer ray tracing. The station geometry is radial, with 40 surface stations.

- Waveforms: `synthetic/waveforms_ml_ray_5000`
- Geometric source-receiver input: `synthetic/bfnet_samples_ml_ray_5000_geom`
- Source-side takeoff input: `synthetic/bfnet_samples_ml_ray_5000_source`
- Station file: `conf/station_synth_dataset_a.xyz`

Dataset B uses a square-grid station geometry to test the applicability of the method under a different acquisition layout. It contains 5000 synthetic events and uses 64 surface stations arranged in an `8 x 8` grid.

- Station file: `conf/station_synth_dataset_b.xyz`
- Cross-station generalization subset: `synthetic/waveforms_ml_ray_500_b`
- Geometric source-receiver input: `synthetic/bfnet_samples_ml_ray_500_b_geom`
- Source-side takeoff input: `synthetic/bfnet_samples_ml_ray_500_b_source`

Both datasets use the same layered velocity model in `conf/vel_synth_jssa.txt`. The only difference between the two radiation-direction inputs is the direction used in theoretical radiation-intensity calculation. The geometric source-receiver input uses the straight direction from a candidate source to each station. The source-side takeoff input uses receiver-side incidence angles and azimuths obtained from multilayer ray tracing, and then converts them to source-side takeoff directions using ray-parameter conservation.

### 1.2 jSSA Brightness-field Construction

The BFNet inputs are constructed from jSSA brightness fields. For each event, a local spatial search grid is built around the event location. The spatial grid size is `8 x 8 x 8`, and the focal mechanism grid size is `24 x 7 x 24`. Given the travel-time table, theoretical radiation-intensity table, and observed waveforms, jSSA jointly scans the spatial grid, focal mechanism grid, and tau samples.

The main settings used for brightness-field construction are:

- `--tt-model raytrace`
- `--tau-search-radius-samples 40`
- `--tau-pick-mode center`
- `--tau-pick-spatial-radius 1`
- `--brightness-backend torch`
- `--dtype float16`

The tau-center strategy is used to select tau near the event center and reduce the influence of boundary-related false peaks on brightness-field slices. This strategy is an engineering setting for stable BFNet input construction, not the methodological contribution of this study.

### 1.3 BFNet Training Strategy

The final BFNet experiments use a three-stage training strategy:

```text
stage 1: SGD, 100 epochs, aux=0.02
stage 2: AdamW, 100 epochs, aux=0.05
finetune: AdamW, 50 epochs, aux=0.05, low learning rate
```

Here, `aux` denotes the weight of the equivalent-plane sine/cosine auxiliary loss. Stage 1 is trained first and the best validation checkpoint is saved. Stage 2 starts from the best Stage 1 checkpoint. The finetuning stage starts from the best Stage 2 checkpoint and continues training with a lower learning rate. Comparison tests showed that `stage 2 = 100 epochs + low-lr finetune = 50 epochs` is more stable than simply extending Stage 2 to 150 epochs.

## 2. Dataset A Results

| Input direction | Method | Strike | Dip | Rake | Total error |
|---|---:|---:|---:|---:|---:|
| Geometric source-receiver | jSSA | 7.83 / 4.46 | 3.72 / 3.03 | 17.75 / 13.19 | 29.30 / 20.68 |
| Geometric source-receiver | BFNet three-stage | 6.23 / 2.76 | 2.02 / 1.42 | 12.36 / 7.29 | 20.61 / 11.47 |
| Source-side takeoff | jSSA | 7.02 / 4.14 | 3.34 / 2.51 | 15.02 / 10.64 | 25.38 / 17.29 |
| Source-side takeoff | BFNet three-stage | 5.91 / 3.00 | 1.96 / 1.37 | 11.19 / 6.87 | 19.06 / 11.24 |

Dataset A supports three main observations:

1. The source-side takeoff direction reduces jSSA strike, dip, and rake errors compared with the geometric source-receiver direction.
2. BFNet further reduces focal mechanism errors compared with the corresponding jSSA results, indicating that the full brightness-field information is useful for focal mechanism prediction.
3. Source-side takeoff BFNet achieves lower total mean and median errors than geometric source-receiver BFNet, suggesting that the corrected brightness fields improve BFNet input quality.

## 3. Dataset B Results

| Input direction | Method | Strike | Dip | Rake | Total error |
|---|---:|---:|---:|---:|---:|
| Geometric source-receiver | jSSA | 9.26 / 5.33 | 3.73 / 2.97 | 16.56 / 11.58 | 29.55 / 19.88 |
| Geometric source-receiver | BFNet three-stage | 6.34 / 3.31 | 2.05 / 1.45 | 10.66 / 6.46 | 19.06 / 11.23 |
| Source-side takeoff | jSSA | 8.77 / 5.01 | 3.71 / 2.70 | 14.48 / 9.14 | 26.96 / 16.85 |
| Source-side takeoff | BFNet three-stage | 6.41 / 3.48 | 2.01 / 1.38 | 11.21 / 6.30 | 19.63 / 11.15 |

For Dataset B, the source-side takeoff correction still provides stable improvement for jSSA. The mean and median errors of the three mechanism components are generally reduced. BFNet also improves over the corresponding jSSA results for both input directions. The difference between the two BFNet inputs is smaller in Dataset B than in Dataset A. The total median error of source-side takeoff BFNet is slightly lower, and the median dip and rake errors are also lower, whereas the total mean error of geometric source-receiver BFNet is slightly lower, mainly due to strike mean and rake mean. A cautious interpretation is that the source-side takeoff correction has the most stable effect on jSSA, while its additional benefit for BFNet is more dataset-dependent.

## 4. Cross-station Generalization Test

To test BFNet generalization across station geometries, the three-stage BFNet model trained on Dataset A was directly evaluated on a 500-event subset of Dataset B without retraining on Dataset B. This test examines whether the source-side takeoff correction remains useful when the station layout changes from a radial distribution to an `8 x 8` square-grid distribution.

| Input direction | Method | Strike | Dip | Rake | Total error |
|---|---:|---:|---:|---:|---:|
| Geometric source-receiver | jSSA | 9.69 / 5.55 | 3.92 / 3.30 | 17.00 / 11.91 | 30.61 / 20.75 |
| Geometric source-receiver | BFNet three-stage | 9.25 / 4.43 | 2.90 / 1.97 | 14.75 / 9.07 | 26.90 / 15.47 |
| Source-side takeoff | jSSA | 9.14 / 5.40 | 3.78 / 2.79 | 16.23 / 10.52 | 29.16 / 18.72 |
| Source-side takeoff | BFNet three-stage | 8.69 / 4.05 | 2.74 / 2.22 | 13.19 / 7.64 | 24.61 / 13.91 |

The cross-station test shows that source-side takeoff jSSA still outperforms geometric source-receiver jSSA. Both BFNet models reduce errors relative to their corresponding jSSA inputs. Overall, source-side takeoff BFNet achieves lower total mean and median errors than geometric source-receiver BFNet, indicating that the correction retains some generalization value when station geometry changes.

## 5. Real-field Label-free Validation

To evaluate the applicability of the source-side takeoff correction to real data, jSSA comparison experiments were performed using continuous waveform data from a Ningxia coalbed methane hydraulic-fracturing monitoring project. Reliable true strike, dip, and rake labels are not available for the real-field events. Therefore, station-holdout validation and mechanism-response concentration were used as label-free evaluation indicators.

The real-field validation keeps the waveform data, station coordinates, velocity model, search grid, travel-time calculation, and jSSA processing workflow unchanged. Only the radiation direction used in theoretical radiation-intensity calculation is changed:

- Geometric source-receiver: theoretical radiation intensities are computed using the straight source-station direction.
- Source-side takeoff: theoretical radiation intensities are computed using source-side takeoff directions converted from receiver-side incidence angles and azimuths obtained by multilayer ray tracing.

In the station-holdout validation, 25% of the stations are randomly withheld and are not used in jSSA inversion. The remaining stations are used to estimate the optimal mechanism, and the estimated mechanism is then evaluated by its ability to explain the responses at the withheld stations. The current validation uses 50 events and 3 holdout splits, resulting in 150 paired comparisons.

| Metric | Meaning | Geometric source-receiver | Source-side takeoff | Source-side takeoff better cases |
|---|---|---:|---:|---:|
| Holdout residual | Normalized difference between predicted and observed responses at withheld stations; lower is better | 0.9132 / 0.9322 | 0.8990 / 0.9242 | 102 / 150 |
| Amplitude correlation | Correlation between predicted and observed responses at withheld stations; higher is better | -0.0212 / -0.0841 | -0.0091 / -0.0707 | 80 / 150 |
| Mechanism entropy | Dispersion of the brightness response over the mechanism grid; lower indicates a more concentrated mechanism response | 0.9817 / 0.9823 | 0.9811 / 0.9809 | 127 / 150 |

The values in the table are reported as `mean / median`. The results show that source-side takeoff improves all three proxy metrics. It produces lower holdout residuals in 102 of 150 comparisons and lower mechanism entropy in 127 of 150 comparisons. This indicates that the correction tends to improve withheld-station explanatory capability and produce more concentrated jSSA mechanism responses in the real field. The improvement in amplitude correlation is weaker, but source-side takeoff still performs better in 80 of 150 paired comparisons.

## 6. Overall Conclusions

Under two different station geometries, the source-side takeoff radiation direction consistently improves jSSA focal mechanism errors. This indicates that, in layered velocity models, using the source-side takeoff direction for theoretical radiation-intensity calculation is more consistent with ray propagation than using the geometric source-receiver direction.

BFNet improves over the corresponding jSSA results for both geometric source-receiver and source-side takeoff inputs, showing that the network can further learn the mapping between brightness-field structure and focal mechanism parameters. The benefit of source-side takeoff for BFNet is partly dataset-dependent. It is clear in Dataset A, closer in Dataset B, and still beneficial in the cross-station generalization test.

The real-field label-free validation further shows that source-side takeoff changes the jSSA mechanism response and improves proxy metrics such as holdout residual, amplitude correlation, and mechanism entropy. These results are consistent with the synthetic error reductions and suggest that the correction has practical value as an embeddable module in existing jSSA-BFNet workflows.

## 7. Result File Organization

Dataset A final results:

- `result/dataset_a_final/`
- `result/dataset_a_final/bfnet_geometric_source_receiver/`
- `result/dataset_a_final/bfnet_source_side_takeoff/`
- `result/dataset_a_final/jssa_geometric_vs_source_takeoff/`
- `result/dataset_a_final/bfnet_geometric_vs_source_takeoff_paired_stats.json`

Cross-station generalization test:

- `result/cross_station_generalization_b500/`

Real-field label-free validation:

- `result/real_source_takeoff_jssa_holdout_50e3s/`

Manuscript figures:

- `result/manuscript_figures/`

Figure-generation scripts:

- `make_manuscript_figures.py`
- `make_graphical_abstract.py`

Local model weights:

- `model/seed20260506_final/`

The local model-weight directory contains trained model checkpoints. Because these files can be large, they are usually not included in the GitHub repository. Reproduction of the manuscript results mainly relies on derived evaluation metrics, prediction outputs, and figure-generation scripts.
