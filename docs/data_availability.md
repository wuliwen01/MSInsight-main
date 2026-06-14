# Data Availability

The processing scripts, configuration files, derived evaluation metrics, processed comparison results, and figure-generation code used in this study are provided in this repository.

The raw synthetic waveform datasets and generated BFNet brightness-field samples are not included because of their large file size. The real-field waveform data from the Ningxia coalbed methane monitoring project are also not included because they are subject to confidentiality agreements with the field operator.

The repository provides the derived results needed to support the conclusions reported in the manuscript, including:

- Dataset A jSSA and BFNet comparison metrics in `result/dataset_a_final/`
- Cross-station generalization results in `result/cross_station_generalization_b500/`
- Real-field station-holdout validation metrics in `result/real_source_takeoff_jssa_holdout_50e3s/`
- Manuscript figures and captions in `result/manuscript_figures/`
- Figure-generation scripts `make_manuscript_figures.py` and `make_graphical_abstract.py`

The raw waveform datasets can be regenerated or reprocessed using the provided scripts and configuration files when the corresponding input data are available locally.
