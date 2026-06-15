# Dataset B Final Results

This directory contains the derived Dataset B results used in the source-side takeoff radiation-correction manuscript.

Dataset B contains 5000 synthetic events with an `8 x 8` square-grid station geometry. The files here are derived evaluation outputs and training logs, not raw waveform data or BFNet brightness-field sample arrays.

## Contents

- `bfnet_geometric_source_receiver/`
  BFNet metrics, predictions, and training logs for brightness fields constructed with the conventional geometric source-receiver radiation direction.

- `bfnet_source_side_takeoff/`
  BFNet metrics, predictions, and training logs for brightness fields constructed with the source-side takeoff radiation direction.

- `jssa_geometric_vs_source_takeoff/`
  jSSA comparison metrics extracted from the paired Dataset B prediction outputs.

- `bfnet_geometric_vs_source_takeoff_paired_stats.json`
  Paired BFNet comparison statistics between geometric source-receiver and source-side takeoff inputs.

The corresponding local model checkpoints are kept under `model/dataset_b_5000/` and are excluded from Git because model files can be large. The manuscript-level conclusions rely on the metrics, prediction outputs, and figure-generation scripts included in the repository.
