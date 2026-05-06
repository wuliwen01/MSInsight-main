import argparse
from pathlib import Path

import numpy as np

from config import read_config_file, read_station_file
from data import gen_fm_grid
from synthetic_bfnet_utils import (
    BFNET_SHAPE,
    ensure_dir,
    read_csv_records,
    resolve_record_path,
    sdr_to_moment_tensor_np,
    write_csv_records,
    write_json,
)


def make_local_conf(base_conf, x, y, z):
    conf = dict(base_conf)
    conf["SearchOriginX"] = x - conf["SearchSizeX"] * conf["GridSpacingX"] / 2.0
    conf["SearchOriginY"] = y - conf["SearchSizeY"] * conf["GridSpacingX"] / 2.0
    conf["SearchOriginZ"] = z - conf["SearchSizeZ"] * conf["GridSpacingZ"] / 2.0
    return conf


def grid_points_km(conf):
    x = conf["SearchOriginX"] + np.arange(conf["SearchSizeX"], dtype=np.float32) * conf["GridSpacingX"]
    y = conf["SearchOriginY"] + np.arange(conf["SearchSizeY"], dtype=np.float32) * conf["GridSpacingX"]
    z = conf["SearchOriginZ"] + np.arange(conf["SearchSizeZ"], dtype=np.float32) * conf["GridSpacingZ"]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)


def straight_ray_traveltime_samples(conf, station, sample_rate, vp):
    grids = grid_points_km(conf)
    grid_m = grids * 1000.0
    sta_m = np.stack([station["x"], station["y"], station["z"]], axis=1).astype(np.float32)
    diff = sta_m[:, None, :] - grid_m[None, :, :]
    distance_km = np.linalg.norm(diff, axis=2) / 1000.0
    return np.rint(distance_km / vp * sample_rate).astype(np.int32)


def station_vectors_for_intensity(conf, station):
    grids = grid_points_km(conf)
    grid_m = grids * 1000.0
    sta_x = np.asarray(station["x"], dtype=np.float32)
    sta_y = np.asarray(station["y"], dtype=np.float32)
    sta_z = np.asarray(station["z"], dtype=np.float32)
    vectors = np.stack(
        [
            sta_y[None, :] - grid_m[:, 1:2],
            sta_x[None, :] - grid_m[:, 0:1],
            sta_z[None, :] - grid_m[:, 2:3],
        ],
        axis=2,
    ).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-8)
    distance_km = norms[..., 0] / 1000.0
    return unit, distance_km


def moment_tensor_matrix_from_sdr(strike, dip, rake):
    mt = sdr_to_moment_tensor_np(strike, dip, rake)
    return np.array(
        [
            [mt[0], mt[3], mt[4]],
            [mt[3], mt[1], mt[5]],
            [mt[4], mt[5], mt[2]],
        ],
        dtype=np.float32,
    )


def compute_intensity(fm_grid, conf, station, attenuation, chunk_size):
    unit, distance_km = station_vectors_for_intensity(conf, station)
    n_fm = fm_grid.shape[0]
    n_grid, n_sta, _ = unit.shape
    intensity = np.empty((n_fm, n_grid, n_sta), dtype=np.float32)
    atten = np.exp(-attenuation * distance_km).astype(np.float32)

    for start in range(0, n_fm, chunk_size):
        end = min(start + chunk_size, n_fm)
        mt = np.stack([moment_tensor_matrix_from_sdr(*row) for row in fm_grid[start:end]], axis=0)
        values = np.einsum("gsi,fij,gsj->fgs", unit, mt, unit, optimize=True)
        intensity[start:end] = (values * atten[None, :, :]).astype(np.float32)
    return intensity


def gather_samples(data, tt_samples, origin_sample, n_sta):
    z_data = data[2::3] if data.shape[0] == n_sta * 3 else data
    if z_data.shape[0] != n_sta:
        raise ValueError(f"Expected {n_sta} z traces, got {z_data.shape[0]}")
    idx = origin_sample + tt_samples
    valid = (idx >= 0) & (idx < z_data.shape[1])
    clipped = np.clip(idx, 0, z_data.shape[1] - 1)
    gathered = z_data[np.arange(n_sta)[:, None], clipped]
    gathered = np.where(valid, gathered, 0.0)
    return gathered.T.astype(np.float32)


def brightness_slice(data, tt_samples, intensity, origin_sample, n_sta):
    samples = gather_samples(data, tt_samples, origin_sample, n_sta)
    # intensity is (fm, grid, sta); samples is (grid, sta).
    return np.einsum("fgs,gs->gf", intensity, samples, optimize=True).astype(np.float32)


def calc_grid_position(conf, ix, iy, iz):
    return (
        conf["SearchOriginX"] + ix * conf["GridSpacingX"],
        conf["SearchOriginY"] + iy * conf["GridSpacingX"],
        conf["SearchOriginZ"] + iz * conf["GridSpacingZ"],
    )


def main():
    parser = argparse.ArgumentParser(description="Build BFNet brightness-field .npy samples from synthetic waveform metadata.")
    parser.add_argument("--events-csv", default="synthetic/waveforms/events.csv")
    parser.add_argument("--out-dir", default="synthetic/bfnet_samples")
    parser.add_argument("--conf", default="conf/conf_synth_jssa.txt")
    parser.add_argument("--station", default="conf/station_synth_dataset_a.xyz")
    parser.add_argument("--vp", type=float, default=3.5)
    parser.add_argument("--attenuation", type=float, default=0.5)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--intensity-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    base_conf = read_config_file(args.conf)
    station = read_station_file(args.station, base_conf)
    fm_grid = gen_fm_grid(base_conf)
    expected_shape = (
        base_conf["SearchSizeX"],
        base_conf["SearchSizeY"],
        base_conf["SearchSizeZ"],
        base_conf["SearchSizeStrike"],
        base_conf["SearchSizeDip"],
        base_conf["SearchSizeRake"],
    )
    if expected_shape != BFNET_SHAPE:
        raise ValueError(f"BFNet shape mismatch: config gives {expected_shape}, expected {BFNET_SHAPE}")

    records = read_csv_records(args.events_csv)
    if args.max_events is not None:
        records = records[: args.max_events]

    out_records = []
    n_sta = len(station["x"])
    for idx, rec in enumerate(records):
        x = float(rec["x"])
        y = float(rec["y"])
        z = float(rec["z"])
        sample_rate = float(rec["sample_rate"])
        origin_sample = int(float(rec["origin_sample"]))
        conf = make_local_conf(base_conf, x, y, z)

        tt_samples = straight_ray_traveltime_samples(conf, station, sample_rate, args.vp)
        intensity = compute_intensity(fm_grid, conf, station, args.attenuation, args.intensity_chunk_size)
        waveform_path = resolve_record_path(args.events_csv, rec["waveform_path"])
        data = np.load(waveform_path)
        brightness_gf = brightness_slice(data, tt_samples, intensity, origin_sample, n_sta)
        sample = brightness_gf.reshape(expected_shape)
        if args.dtype == "float16":
            sample_to_save = sample.astype(np.float16)
        else:
            sample_to_save = sample.astype(np.float32)

        out_name = f"br_{idx:06d}.npy"
        np.save(out_dir / out_name, sample_to_save)

        max_idx = np.unravel_index(int(np.argmax(sample)), sample.shape)
        fm_idx = max_idx[3] * base_conf["SearchSizeDip"] * base_conf["SearchSizeRake"] + max_idx[4] * base_conf["SearchSizeRake"] + max_idx[5]
        gx, gy, gz = calc_grid_position(conf, max_idx[0], max_idx[1], max_idx[2])
        jssa_fm = fm_grid[fm_idx]
        out_records.append(
            {
                "event_id": rec["event_id"],
                "input_path": out_name,
                "waveform_path": str(waveform_path),
                "x": x,
                "y": y,
                "z": z,
                "strike": rec["strike"],
                "dip": rec["dip"],
                "rake": rec["rake"],
                "origin_sample": origin_sample,
                "sample_rate": sample_rate,
                "jssa_x": gx,
                "jssa_y": gy,
                "jssa_z": gz,
                "jssa_strike": float(jssa_fm[0]),
                "jssa_dip": float(jssa_fm[1]),
                "jssa_rake": float(jssa_fm[2]),
                "brightness_max": float(np.max(sample)),
            }
        )
        print(
            f"[{idx + 1}/{len(records)}] wrote {out_name} "
            f"max_fm=({jssa_fm[0]:.1f},{jssa_fm[1]:.1f},{jssa_fm[2]:.1f})"
        )

    fieldnames = [
        "event_id",
        "input_path",
        "waveform_path",
        "x",
        "y",
        "z",
        "strike",
        "dip",
        "rake",
        "origin_sample",
        "sample_rate",
        "jssa_x",
        "jssa_y",
        "jssa_z",
        "jssa_strike",
        "jssa_dip",
        "jssa_rake",
        "brightness_max",
    ]
    write_csv_records(out_dir / "samples.csv", out_records, fieldnames)
    write_json(out_dir / "build_config.json", vars(args))
    print(f"BFNet sample metadata saved to {out_dir / 'samples.csv'}")


if __name__ == "__main__":
    main()
