import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_bfnet_brightness_dataset import (
    brightness_slice,
    compute_intensity,
    make_local_conf,
    straight_ray_traveltime_samples,
)
from config import read_config_file, read_station_file
from data import gen_fm_grid
from synthetic_bfnet_utils import BFNET_SHAPE, ensure_dir, read_csv_records, resolve_record_path


def find_record_by_path(csv_path, npy_path, path_key):
    npy_path = Path(npy_path).resolve()
    for rec in read_csv_records(csv_path):
        candidate = resolve_record_path(csv_path, rec[path_key]).resolve()
        if candidate == npy_path:
            return rec
    raise ValueError(f"Could not find {npy_path} in {csv_path} column {path_key}")


def load_brightness_from_waveform(args):
    event_npy = Path(args.event_npy)
    events_csv = Path(args.events_csv) if args.events_csv else event_npy.parent / "events.csv"
    rec = find_record_by_path(events_csv, event_npy, "waveform_path")

    base_conf = read_config_file(args.conf)
    station = read_station_file(args.station, base_conf)
    fm_grid = gen_fm_grid(base_conf)
    conf = make_local_conf(base_conf, float(rec["x"]), float(rec["y"]), float(rec["z"]))
    data = np.load(event_npy)
    tt_samples = straight_ray_traveltime_samples(conf, station, float(rec["sample_rate"]), args.vp)
    intensity = compute_intensity(fm_grid, conf, station, args.attenuation, args.intensity_chunk_size)
    brightness_gf = brightness_slice(
        data,
        tt_samples,
        intensity,
        int(float(rec["origin_sample"])),
        len(station["x"]),
    )
    sample = brightness_gf.reshape(BFNET_SHAPE)
    return sample, data, rec, fm_grid, event_npy.stem


def load_brightness_from_sample(args):
    sample_npy = Path(args.sample_npy)
    sample = np.load(sample_npy).astype(np.float32)
    if sample.shape != BFNET_SHAPE:
        raise ValueError(f"Expected BFNet sample shape {BFNET_SHAPE}, got {sample.shape}")

    rec = None
    waveform = None
    if args.samples_csv:
        rec = find_record_by_path(args.samples_csv, sample_npy, "input_path")
        waveform_path = Path(rec.get("waveform_path", ""))
        if waveform_path.exists():
            waveform = np.load(waveform_path)
    base_conf = read_config_file(args.conf)
    fm_grid = gen_fm_grid(base_conf)
    return sample, waveform, rec, fm_grid, sample_npy.stem


def plot_waveforms(ax, waveform, sample_rate, max_traces):
    if waveform is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No waveform", ha="center", va="center")
        return
    z = waveform[2::3] if waveform.shape[0] % 3 == 0 else waveform
    z = z[:max_traces]
    time = np.arange(z.shape[1]) / sample_rate
    scale = np.max(np.abs(z)) + 1e-8
    for i, trace in enumerate(z):
        ax.plot(time, trace / scale + i, lw=0.8, color="black")
    ax.set_title("Synthetic Z Traces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Station")


def plot_spatial_cross_sections(fig, axes, volume, max_xyz):
    ix, iy, iz = max_xyz
    views = [
        (volume[ix, :, :].T, "YZ at max X"),
        (volume[:, iy, :].T, "XZ at max Y"),
        (volume[:, :, iz].T, "XY at max Z"),
    ]
    vmax = float(np.max(volume))
    vmin = float(np.min(volume))
    for ax, (view, title) in zip(axes, views):
        im = ax.imshow(view, origin="lower", cmap="gnuplot", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.75)


def plot_mechanism_slices(fig, axes, mech_volume, max_sdr_idx):
    is_, id_, ir_ = max_sdr_idx
    views = [
        (mech_volume[:, :, ir_].T, "Strike-Dip at max Rake"),
        (mech_volume[:, id_, :].T, "Strike-Rake at max Dip"),
        (mech_volume[is_, :, :].T, "Dip-Rake at max Strike"),
    ]
    vmax = float(np.max(mech_volume))
    vmin = float(np.min(mech_volume))
    for ax, (view, title) in zip(axes, views):
        im = ax.imshow(view, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.75)


def add_beachball(ax, sdr, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    try:
        from obspy.imaging.beachball import beach

        collection = beach(sdr, xy=(0.0, 0.0), width=1.6, linewidth=1, facecolor="black", bgcolor="white")
        ax.add_collection(collection)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.text(0.0, -1.15, f"S={sdr[0]:.1f} D={sdr[1]:.1f} R={sdr[2]:.1f}", ha="center", va="top", fontsize=8)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Beachball failed:\n{exc}", ha="center", va="center")


def parse_float_triplet(rec, prefix):
    return (float(rec[f"{prefix}strike"]), float(rec[f"{prefix}dip"]), float(rec[f"{prefix}rake"]))


def make_figure(sample, waveform, rec, fm_grid, title, args):
    max_idx = np.unravel_index(int(np.argmax(sample)), sample.shape)
    ix, iy, iz, is_, id_, ir_ = max_idx
    fm_idx = is_ * BFNET_SHAPE[4] * BFNET_SHAPE[5] + id_ * BFNET_SHAPE[5] + ir_
    jssa_sdr = tuple(float(v) for v in fm_grid[fm_idx])

    spatial_volume = sample[:, :, :, is_, id_, ir_]
    mech_volume = sample[ix, iy, iz, :, :, :]
    sample_rate = float(rec["sample_rate"]) if rec and "sample_rate" in rec else args.sample_rate

    fig = plt.figure(figsize=(15, 10), dpi=args.dpi, constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.9])
    fig.suptitle(
        f"{title} | max index xyz/sdr={tuple(int(v) for v in max_idx)} | max={float(sample[max_idx]):.4g}",
        fontsize=12,
    )

    ax_wave = fig.add_subplot(gs[0, :])
    plot_waveforms(ax_wave, waveform, sample_rate, args.max_traces)

    spatial_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    plot_spatial_cross_sections(fig, spatial_axes, spatial_volume, (ix, iy, iz))
    spatial_axes[0].set_ylabel("Spatial brightness at max mechanism")

    mech_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    plot_mechanism_slices(fig, mech_axes, mech_volume, (is_, id_, ir_))
    mech_axes[0].set_ylabel("Mechanism brightness at max location")

    ax_true = fig.add_subplot(gs[1, 3])
    if rec is not None and all(k in rec for k in ["strike", "dip", "rake"]):
        add_beachball(ax_true, parse_float_triplet(rec, ""), "True Mechanism")
    else:
        ax_true.axis("off")
        ax_true.text(0.5, 0.5, "No true mechanism", ha="center", va="center")

    ax_jssa = fig.add_subplot(gs[2, 3])
    add_beachball(ax_jssa, jssa_sdr, "Brightness Max Mechanism")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic BFNet brightness field and focal mechanisms.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-npy", help="BFNet brightness sample, e.g. synthetic/bfnet_samples_a/br_000000.npy")
    source.add_argument("--event-npy", help="Synthetic waveform file, e.g. synthetic/waveforms_a/ev_000000.npy")
    parser.add_argument("--samples-csv", default=None)
    parser.add_argument("--events-csv", default=None)
    parser.add_argument("--conf", default="conf/conf_synth_jssa.txt")
    parser.add_argument("--station", default="conf/station_synth_dataset_a.xyz")
    parser.add_argument("--vp", type=float, default=3.5)
    parser.add_argument("--attenuation", type=float, default=0.5)
    parser.add_argument("--intensity-chunk-size", type=int, default=128)
    parser.add_argument("--sample-rate", type=float, default=500.0)
    parser.add_argument("--max-traces", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.event_npy:
        sample, waveform, rec, fm_grid, title = load_brightness_from_waveform(args)
    else:
        sample, waveform, rec, fm_grid, title = load_brightness_from_sample(args)

    fig = make_figure(sample, waveform, rec, fm_grid, title, args)
    out = Path(args.out) if args.out else Path("result") / f"{title}_bfnet_visualization.png"
    ensure_dir(out.parent)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out}")


if __name__ == "__main__":
    main()
