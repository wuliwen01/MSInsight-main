# MSInsight Source-side Takeoff Radiation Correction

This repository contains the code, configuration files, derived metrics, and figure-generation scripts used for the manuscript on source-side takeoff radiation correction for microseismic focal mechanism inversion in layered velocity models.

The main purpose of the repository is to document and reproduce the paper-level analyses comparing the conventional geometric source-receiver radiation direction with the proposed source-side takeoff radiation direction in jSSA and BFNet workflows.

## Repository Contents

- `conf/`
  Configuration files, station files, and velocity models used by SSA, jSSA, synthetic experiments, and real-field processing.

- `3draytracing/` and `fasterStackCUDA/`
  CUDA/C++ source code for ray tracing and stacking-related acceleration.

- `build_bfnet_brightness_dataset_raytrace.py`
  Builds BFNet brightness-field samples using ray-traced travel-time and radiation-direction information.

- `generate_synthetic_dataset_raytrace.py`
  Generates ray-tracing-based synthetic microseismic datasets.

- `train_bfnet_synthetic.py`, `evaluate_bfnet_synthetic.py`, and `visualize_bfnet_synthetic.py`
  BFNet training, evaluation, and visualization scripts.

- `validate_real_source_takeoff_jssa.py`
  Real-field station-holdout validation for comparing geometric source-receiver and source-side takeoff radiation directions.

- `make_manuscript_figures.py`
  Generates the manuscript figures from the derived metrics and processed comparison results.

- `make_graphical_abstract.py`
  Generates the graphical abstract.

- `result/source_takeoff_experiment_summary_en.md`
  English summary of the synthetic, BFNet, cross-station generalization, and real-field validation experiments.

- `result/source_takeoff_experiment_summary.md`
  Chinese version of the experiment summary.

- `result/dataset_a_final/`
  Dataset A derived jSSA and BFNet comparison results.

- `result/cross_station_generalization_b500/`
  Cross-station generalization results on the Dataset B 500-event subset.

- `result/real_source_takeoff_jssa_holdout_50e3s/`
  Real-field station-holdout validation metrics for the Ningxia coalbed methane field data.

- `result/manuscript_figures/`
  Manuscript figures and figure captions.

- `result/graphical_abstract/`
  Graphical abstract files.

- `docs/data_availability.md`
  Data availability statement and details about files not included in the repository.

## Data Availability

Raw synthetic waveform datasets, generated BFNet brightness-field sample arrays, and real-field waveform data are not included in this repository. The synthetic waveform datasets and brightness-field samples are excluded because of file-size limitations. The real-field waveform data from the Ningxia coalbed methane monitoring project are additionally subject to confidentiality agreements with the field operator.

The repository provides processing scripts, configuration files, derived evaluation metrics, processed comparison results, and figure-generation scripts that support the conclusions of the manuscript. See `docs/data_availability.md` for details.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The CUDA-based components require a CUDA-compatible GPU and a local CUDA/MSVC build environment on Windows when recompilation is needed. Precompiled DLLs may be used when compatible with the local system.

## Regenerating Manuscript Figures

To regenerate the manuscript figures:

```bash
python make_manuscript_figures.py
```

The figures are written to:

```text
result/manuscript_figures/
```

To regenerate the graphical abstract:

```bash
python make_graphical_abstract.py
```

The graphical abstract is written to:

```text
result/graphical_abstract/
```

## Notes for Reuse

The repository is organized around derived metrics and reproducible processing scripts rather than raw waveform data. To rerun the full waveform-level experiments, place the corresponding synthetic or real-field waveform data in the expected local directories and update the configuration files in `conf/` as needed.

## License

See `LICENSE`.
