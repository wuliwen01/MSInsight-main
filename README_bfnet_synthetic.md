# BFNet Synthetic Training Workflow

This add-on keeps the existing real-data pipeline untouched. It adds a small synthetic workflow for BFNet:

1. Generate synthetic waveform `.npy` files plus labels.
2. Convert each event to the BFNet input: one jSSA-style brightness-field slice with shape `8x8x8x24x7x24`.
3. Train BFNet with the thesis two-stage schedule.
4. Evaluate BFNet and compare it with the coarse jSSA-grid maximum stored in the sample metadata.

The default synthetic configuration follows the thesis Dataset A setup where it matters for BFNet training: 500 Hz sampling, events inside a 300 m sphere around `(0, 0, -0.8) km`, homogeneous `Vp=3.5 km/s`, 5 Hz Ricker wavelet, and a 40-station radial observation system with 8 rays and 5 stations per ray at 150 m spacing. BFNet/jSSA uses `8x8x8` local spatial samples and a 15 degree mechanism grid (`24x7x24`). The default synthetic waveform path is clean: no whitening, no filtering, no STA/LTA, and no added noise.

The files are separate from the real field configuration:

```text
conf/conf_synth_jssa.txt
conf/conf_synth_ssa.txt
conf/vel_synth.txt
conf/station_synth_dataset_a.xyz
conf/station_synth_dataset_b.xyz
```

## Quick Smoke Run

Use a tiny event count first:

```bash
python generate_synthetic_dataset.py --event-count 8 --out-dir synthetic/waveforms_smoke
python build_bfnet_brightness_dataset.py --events-csv synthetic/waveforms_smoke/events.csv --out-dir synthetic/bfnet_samples_smoke --max-events 8 --dtype float16
python train_bfnet_synthetic.py --samples-csv synthetic/bfnet_samples_smoke/samples.csv --stage1-epochs 1 --stage2-epochs 1 --stage1-batch-size 2 --stage2-batch-size 2 --output model/bfnet_synthetic_smoke.pt
python evaluate_bfnet_synthetic.py --samples-csv synthetic/bfnet_samples_smoke/samples.csv --model model/bfnet_synthetic_smoke.pt --batch-size 2
```

## Practical Server Run

For a small but useful experiment:

```bash
python generate_synthetic_dataset.py --event-count 256 --out-dir synthetic/waveforms_a
python build_bfnet_brightness_dataset.py --events-csv synthetic/waveforms_a/events.csv --out-dir synthetic/bfnet_samples_a --dtype float16
python train_bfnet_synthetic.py --samples-csv synthetic/bfnet_samples_a/samples.csv --output model/bfnet_synthetic_a.pt --checkpoint model/bfnet_synthetic_a.ckpt
python evaluate_bfnet_synthetic.py --samples-csv synthetic/bfnet_samples_a/samples.csv --model model/bfnet_synthetic_a.pt
```

For a thesis-style full schedule, leave `--stage1-epochs 200 --stage2-epochs 50` as defaults. For quick checks, reduce the two epoch arguments.

## Visualizing One Event

After building BFNet samples, visualize one brightness-field sample:

```bash
python visualize_bfnet_synthetic.py --sample-npy synthetic/bfnet_samples_a/br_000000.npy --samples-csv synthetic/bfnet_samples_a/samples.csv --out result/br_000000_visual.png
```

Or start from a synthetic waveform `.npy` and construct the brightness slice on the fly:

```bash
python visualize_bfnet_synthetic.py --event-npy synthetic/waveforms_a/ev_000000.npy --events-csv synthetic/waveforms_a/events.csv --out result/ev_000000_visual.png
```

The output figure contains raw synthetic Z traces, three spatial brightness cross-sections at the maximum mechanism, three mechanism-space brightness slices at the maximum location, the true beachball, and the brightness-maximum mechanism beachball.

## Using Existing `.npy` Synthetic Waveforms

If you already have synthetic waveform `.npy` files, skip `generate_synthetic_dataset.py` and create an `events.csv` next to them with these columns:

```text
event_id,waveform_path,x,y,z,strike,dip,rake,sample_rate,n_samples,origin_sample,snr_db
```

The waveform array should be either `(n_sta * 3, n_samples)` in `N/E/Z` order, or `(n_sta, n_samples)` if it is Z-only. Coordinates are in km for `x/y/z`; angles are in degrees. `origin_sample` is the source origin sample used to extract the peak-time brightness slice.

Then run:

```bash
python build_bfnet_brightness_dataset.py --events-csv path/to/events.csv --out-dir synthetic/bfnet_samples_custom --dtype float16
```

For a Dataset B style cross-observation-system test, use the matrix station file and the Dataset B source box manually:

```bash
python generate_synthetic_dataset.py --event-count 128 --station conf/station_synth_dataset_b.xyz --out-dir synthetic/waveforms_b --center-x 0 --center-y 0 --center-z -0.8 --radius-km 0.35
python build_bfnet_brightness_dataset.py --events-csv synthetic/waveforms_b/events.csv --station conf/station_synth_dataset_b.xyz --out-dir synthetic/bfnet_samples_b --dtype float16
```

## Notes

- The BFNet input is not the raw waveform. It is the peak-time brightness-field slice described in Chapter 4 of the thesis.
- The generated waveform files are `n_sta * 3` traces in `N/E/Z` order. The brightness builder uses the raw Z component, matching the existing jSSA stack code for synthetic data.
- Do not pass `--snr-db` for the clean synthetic set. That option is only kept for later robustness experiments.
- `model/bfnet_synthetic*.pt` is saved as a raw `state_dict`, so it can be loaded by the existing inference scripts in the same way as `model/bfnet_251104a.pt`.
- The brightness builder uses the synthetic source origin time from metadata and computes the corresponding jSSA-style slice directly. This avoids allocating the full `(grid, mechanism, time)` volume for every event.
