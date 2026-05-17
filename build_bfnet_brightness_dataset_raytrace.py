import argparse
from pathlib import Path

import numpy as np
import torch

from config import read_config_file, read_station_file
from data import gen_fm_grid, generate_tt
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


def read_velocity_model(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise ValueError(f"No velocity layers found in {path}")
    return np.asarray(rows, dtype=np.float32)


def velocity_at_z_km(velocity_model, z_km):
    z = np.asarray(z_km, dtype=np.float32)
    out = np.full(z.shape, float(velocity_model[-1, 1]), dtype=np.float32)
    for idx, (upper, vp) in enumerate(velocity_model):
        lower = velocity_model[idx + 1, 0] if idx + 1 < len(velocity_model) else -np.inf
        out = np.where((z < upper) & (z >= lower), vp, out)
    if np.isscalar(z_km):
        return float(out)
    return out


def straight_ray_traveltime_samples(conf, station, sample_rate, vp):
    grids = grid_points_km(conf)
    grid_m = grids * 1000.0
    sta_m = np.stack([station["x"], station["y"], station["z"]], axis=1).astype(np.float32)
    diff = sta_m[:, None, :] - grid_m[None, :, :]
    distance_km = np.linalg.norm(diff, axis=2) / 1000.0
    return np.rint(distance_km / vp * sample_rate).astype(np.int32)


def raytrace_traveltime_samples(conf, station_path, vel_file, sample_rate):
    tt, incident, azimuth = generate_tt(conf, vel_file, station_path)
    return np.rint(tt * sample_rate).astype(np.int32), incident.astype(np.float32), azimuth.astype(np.float32)


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


def source_takeoff_vectors_for_intensity(conf, station, incident, azimuth, velocity_model):
    grids = grid_points_km(conf)
    n_sta = len(station["x"])
    if incident.shape != (n_sta, len(grids)):
        raise ValueError(f"Incident shape {incident.shape} does not match n_sta={n_sta}, n_grid={len(grids)}")
    _, distance_km = station_vectors_for_intensity(conf, station)
    station_z_km = np.asarray(station["z"], dtype=np.float32) / 1000.0
    receiver_vp = velocity_at_z_km(velocity_model, station_z_km)
    source_vp = velocity_at_z_km(velocity_model, grids[:, 2])
    incident_gs = incident.T.astype(np.float32)
    azimuth_gs = azimuth.T.astype(np.float32)
    ray_parameter = np.sin(incident_gs) / np.maximum(receiver_vp[None, :], 1e-8)
    theta_source = np.arcsin(np.clip(ray_parameter * source_vp[:, None], -1.0, 1.0))
    sin_theta = np.sin(theta_source)
    unit = np.stack(
        [
            sin_theta * np.cos(azimuth_gs),
            sin_theta * np.sin(azimuth_gs),
            np.cos(theta_source),
        ],
        axis=2,
    ).astype(np.float32)
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


def moment_tensor_matrices_from_fm_grid(fm_grid):
    return np.stack([moment_tensor_matrix_from_sdr(*row) for row in fm_grid], axis=0).astype(np.float32)


def deduplicate_fm_grid(fm_grid, decimals=5):
    seen = {}
    unique_indices = []
    inverse_indices = np.empty(len(fm_grid), dtype=np.int32)
    for idx, row in enumerate(fm_grid):
        mt = sdr_to_moment_tensor_np(*row).astype(np.float64)
        mt = mt / (np.linalg.norm(mt) + 1e-12)
        key = tuple(np.round(mt, decimals))
        unique_pos = seen.get(key)
        if unique_pos is None:
            unique_pos = len(unique_indices)
            seen[key] = unique_pos
            unique_indices.append(idx)
        inverse_indices[idx] = unique_pos
    unique_indices = np.asarray(unique_indices, dtype=np.int32)
    return fm_grid[unique_indices], unique_indices, inverse_indices


def compute_intensity_from_directions(fm_grid, unit, distance_km, attenuation, chunk_size, fm_matrices=None):
    n_fm = fm_grid.shape[0]
    n_grid, n_sta, _ = unit.shape
    intensity = np.empty((n_fm, n_grid, n_sta), dtype=np.float32)
    atten = np.exp(-attenuation * distance_km).astype(np.float32)
    if fm_matrices is None:
        fm_matrices = moment_tensor_matrices_from_fm_grid(fm_grid)

    for start in range(0, n_fm, chunk_size):
        end = min(start + chunk_size, n_fm)
        mt = fm_matrices[start:end]
        values = np.einsum("gsi,fij,gsj->fgs", unit, mt, unit, optimize=True)
        intensity[start:end] = (values * atten[None, :, :]).astype(np.float32)
    return intensity


def compute_intensity(fm_grid, conf, station, attenuation, chunk_size, fm_matrices=None):
    unit, distance_km = station_vectors_for_intensity(conf, station)
    return compute_intensity_from_directions(
        fm_grid,
        unit,
        distance_km,
        attenuation,
        chunk_size,
        fm_matrices=fm_matrices,
    )


def extract_z_data(data, n_sta):
    z_data = data[2::3] if data.shape[0] == n_sta * 3 else data
    if z_data.shape[0] != n_sta:
        raise ValueError(f"Expected {n_sta} z traces, got {z_data.shape[0]}")
    return z_data


def gather_samples(data, tt_samples, origin_sample, n_sta):
    z_data = extract_z_data(data, n_sta)
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


def tau_search_bounds(reference_sample, n_samples, tt_samples, radius_samples):
    max_tt = int(np.max(tt_samples))
    valid_end = max(0, n_samples - max_tt - 1)
    if radius_samples < 0:
        start = 0
        end = valid_end
    else:
        start = max(0, reference_sample - radius_samples)
        end = min(valid_end, reference_sample + radius_samples)
    if end < start:
        raise ValueError(
            f"Invalid tau search bounds: start={start}, end={end}, "
            f"reference={reference_sample}, n_samples={n_samples}, max_tt={max_tt}"
        )
    return start, end


def gather_samples_for_tau_values(z_data, tt_samples, tau_values):
    n_sta = z_data.shape[0]
    idx = tau_values[:, None, None] + tt_samples[None, :, :]
    valid = (idx >= 0) & (idx < z_data.shape[1])
    clipped = np.clip(idx, 0, z_data.shape[1] - 1)
    sta_idx = np.arange(n_sta, dtype=np.int32)[None, :, None]
    gathered = z_data[sta_idx, clipped]
    gathered = np.where(valid, gathered, 0.0)
    return gathered.transpose(0, 2, 1).astype(np.float32)


def spatial_pick_indices(shape, radius):
    x_dim, y_dim, z_dim = shape[:3]
    center = (x_dim // 2, y_dim // 2, z_dim // 2)
    radius = max(0, int(radius))
    ranges = [
        range(max(0, c - radius), min(dim, c + radius + 1))
        for c, dim in zip(center, (x_dim, y_dim, z_dim))
    ]
    indices = [
        ix * y_dim * z_dim + iy * z_dim + iz
        for ix in ranges[0]
        for iy in ranges[1]
        for iz in ranges[2]
    ]
    return np.asarray(indices, dtype=np.int32)


def peak_brightness_slice_numpy(data, tt_samples, intensity, tau_start, tau_end, n_sta, tau_chunk_size, pick_grid_indices):
    z_data = extract_z_data(data, n_sta)
    best_value = -np.inf
    best_tau = tau_start
    best_slice = None
    tau_chunk_size = max(1, int(tau_chunk_size))
    pick_grid_indices = None if pick_grid_indices is None else np.asarray(pick_grid_indices, dtype=np.int32)

    for chunk_start in range(tau_start, tau_end + 1, tau_chunk_size):
        chunk_end = min(tau_end, chunk_start + tau_chunk_size - 1)
        tau_values = np.arange(chunk_start, chunk_end + 1, dtype=np.int32)
        samples = gather_samples_for_tau_values(z_data, tt_samples, tau_values)
        # intensity is (fm, grid, sta); samples is (tau, grid, sta).
        brightness = np.einsum("fgs,tgs->tgf", intensity, samples, optimize=True).astype(np.float32)
        score = brightness if pick_grid_indices is None else brightness[:, pick_grid_indices, :]
        max_idx = np.unravel_index(int(np.argmax(score)), score.shape)
        max_value = float(score[max_idx])
        if max_value > best_value:
            best_value = max_value
            best_tau = int(tau_values[max_idx[0]])
            best_slice = brightness[max_idx[0]].copy()

    if best_slice is None:
        raise RuntimeError(f"No tau samples were evaluated for range [{tau_start}, {tau_end}]")
    return best_slice, best_tau, best_value


def peak_brightness_slice_torch(data, tt_samples, intensity, tau_start, tau_end, n_sta, tau_chunk_size, pick_grid_indices, device):
    z_data = extract_z_data(data, n_sta)
    device = torch.device(device)
    z_tensor = torch.as_tensor(z_data, dtype=torch.float32, device=device)
    tt_tensor = torch.as_tensor(tt_samples.T, dtype=torch.long, device=device)
    intensity_tensor = torch.as_tensor(np.ascontiguousarray(intensity), dtype=torch.float32, device=device)
    station_index = torch.arange(n_sta, dtype=torch.long, device=device)
    pick_tensor = None
    if pick_grid_indices is not None:
        pick_tensor = torch.as_tensor(pick_grid_indices, dtype=torch.long, device=device)

    best_value = -np.inf
    best_tau = tau_start
    best_slice = None
    tau_chunk_size = max(1, int(tau_chunk_size))

    with torch.no_grad():
        for chunk_start in range(tau_start, tau_end + 1, tau_chunk_size):
            chunk_end = min(tau_end, chunk_start + tau_chunk_size - 1)
            tau_values = torch.arange(chunk_start, chunk_end + 1, dtype=torch.long, device=device)
            idx = tau_values[:, None, None] + tt_tensor[None, :, :]
            valid = (idx >= 0) & (idx < z_tensor.shape[1])
            clipped = idx.clamp(0, z_tensor.shape[1] - 1)
            sta_idx = station_index[None, None, :].expand_as(clipped)
            samples = z_tensor[sta_idx, clipped]
            samples = torch.where(valid, samples, torch.zeros((), dtype=samples.dtype, device=device))
            brightness = torch.einsum("fgs,tgs->tgf", intensity_tensor, samples)
            score = brightness if pick_tensor is None else brightness[:, pick_tensor, :]
            max_value_tensor = torch.max(score)
            max_value = float(max_value_tensor.detach().cpu())
            if max_value > best_value:
                flat_idx = int(torch.argmax(score).detach().cpu())
                max_idx = np.unravel_index(flat_idx, tuple(int(v) for v in score.shape))
                best_value = max_value
                best_tau = int(tau_values[max_idx[0]].detach().cpu())
                best_slice = brightness[max_idx[0]].detach().cpu().numpy().astype(np.float32)

    if best_slice is None:
        raise RuntimeError(f"No tau samples were evaluated for range [{tau_start}, {tau_end}]")
    return best_slice, best_tau, best_value


def peak_brightness_slice(data, tt_samples, intensity, tau_start, tau_end, n_sta, tau_chunk_size, pick_grid_indices, backend, device):
    use_torch = backend == "torch" or (backend == "auto" and torch.cuda.is_available())
    if use_torch:
        return peak_brightness_slice_torch(data, tt_samples, intensity, tau_start, tau_end, n_sta, tau_chunk_size, pick_grid_indices, device)
    return peak_brightness_slice_numpy(data, tt_samples, intensity, tau_start, tau_end, n_sta, tau_chunk_size, pick_grid_indices)


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
    parser.add_argument("--tt-model", choices=["straight", "raytrace"], default="straight")
    parser.add_argument("--intensity-direction", choices=["geometry", "source-takeoff"], default="geometry")
    parser.add_argument("--vel-file", default="conf/vel_synth_jssa.txt")
    parser.add_argument("--attenuation", type=float, default=0.5)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--intensity-chunk-size", type=int, default=64)
    parser.add_argument(
        "--tau-search-radius-samples",
        type=int,
        default=40,
        help="Search tau* within origin_sample +/- this many samples. Use 0 for the old origin-only slice, or -1 for the full valid tau range.",
    )
    parser.add_argument("--tau-chunk-size", type=int, default=4)
    parser.add_argument(
        "--tau-pick-mode",
        choices=["center", "global"],
        default="center",
        help="Use center-grid brightness to choose tau*, or use the global space-mechanism maximum.",
    )
    parser.add_argument("--tau-pick-spatial-radius", type=int, default=1)
    parser.add_argument("--brightness-backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--no-fm-dedup", action="store_true", help="Disable moment-tensor de-duplication for equivalent focal mechanisms.")
    parser.add_argument("--fm-dedup-decimals", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    if args.intensity_direction == "source-takeoff" and args.tt_model != "raytrace":
        raise ValueError("--intensity-direction source-takeoff requires --tt-model raytrace")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    base_conf = read_config_file(args.conf)
    station = read_station_file(args.station, base_conf)
    velocity_model = read_velocity_model(args.vel_file) if args.intensity_direction == "source-takeoff" else None
    fm_grid = gen_fm_grid(base_conf)
    if args.no_fm_dedup:
        intensity_fm_grid = fm_grid
        inverse_fm_indices = None
        print(f"FM grid: using all {len(fm_grid)} mechanisms", flush=True)
    else:
        intensity_fm_grid, unique_fm_indices, inverse_fm_indices = deduplicate_fm_grid(
            fm_grid,
            decimals=args.fm_dedup_decimals,
        )
        reduction = 1.0 - len(intensity_fm_grid) / len(fm_grid)
        print(
            f"FM grid: {len(fm_grid)} full mechanisms, "
            f"{len(intensity_fm_grid)} unique moment tensors "
            f"({reduction:.1%} fewer brightness computations)",
            flush=True,
        )
    fm_matrices = moment_tensor_matrices_from_fm_grid(intensity_fm_grid)
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
    if args.tau_pick_mode == "center":
        pick_grid_indices = spatial_pick_indices(expected_shape, args.tau_pick_spatial_radius)
        print(
            f"Tau pick mode: center spatial radius={args.tau_pick_spatial_radius} "
            f"({len(pick_grid_indices)} grid point(s))",
            flush=True,
        )
    else:
        pick_grid_indices = None
        print("Tau pick mode: global space-mechanism maximum", flush=True)

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

        if args.tt_model == "raytrace":
            tt_samples, incident, azimuth = raytrace_traveltime_samples(conf, args.station, args.vel_file, sample_rate)
        else:
            tt_samples = straight_ray_traveltime_samples(conf, station, sample_rate, args.vp)
            incident = azimuth = None

        if args.intensity_direction == "source-takeoff":
            unit, distance_km = source_takeoff_vectors_for_intensity(conf, station, incident, azimuth, velocity_model)
            intensity = compute_intensity_from_directions(
                intensity_fm_grid,
                unit,
                distance_km,
                args.attenuation,
                args.intensity_chunk_size,
                fm_matrices=fm_matrices,
            )
        else:
            intensity = compute_intensity(
                intensity_fm_grid,
                conf,
                station,
                args.attenuation,
                args.intensity_chunk_size,
                fm_matrices=fm_matrices,
            )
        waveform_path = resolve_record_path(args.events_csv, rec["waveform_path"])
        data = np.load(waveform_path)
        tau_start, tau_end = tau_search_bounds(
            origin_sample,
            data.shape[1],
            tt_samples,
            args.tau_search_radius_samples,
        )
        brightness_gf_unique, tau_star, _ = peak_brightness_slice(
            data,
            tt_samples,
            intensity,
            tau_start,
            tau_end,
            n_sta,
            args.tau_chunk_size,
            pick_grid_indices,
            args.brightness_backend,
            args.torch_device,
        )
        if inverse_fm_indices is None:
            brightness_gf = brightness_gf_unique
        else:
            brightness_gf = brightness_gf_unique[:, inverse_fm_indices]
        sample = brightness_gf.reshape(expected_shape)
        brightness_max = float(np.max(sample))
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
                "tau_star_sample": tau_star,
                "tau_search_start_sample": tau_start,
                "tau_search_end_sample": tau_end,
                "tau_pick_mode": args.tau_pick_mode,
                "tau_pick_spatial_radius": args.tau_pick_spatial_radius,
                "sample_rate": sample_rate,
                "jssa_x": gx,
                "jssa_y": gy,
                "jssa_z": gz,
                "jssa_strike": float(jssa_fm[0]),
                "jssa_dip": float(jssa_fm[1]),
                "jssa_rake": float(jssa_fm[2]),
                "brightness_max": brightness_max,
            }
        )
        print(
            f"[{idx + 1}/{len(records)}] wrote {out_name} "
            f"tau*={tau_star} max_fm=({jssa_fm[0]:.1f},{jssa_fm[1]:.1f},{jssa_fm[2]:.1f})",
            flush=True,
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
        "tau_star_sample",
        "tau_search_start_sample",
        "tau_search_end_sample",
        "tau_pick_mode",
        "tau_pick_spatial_radius",
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
    print(f"BFNet sample metadata saved to {out_dir / 'samples.csv'}", flush=True)


if __name__ == "__main__":
    main()
