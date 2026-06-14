import argparse
import csv
import datetime
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from build_bfnet_brightness_dataset_raytrace import (
    compute_intensity_from_directions,
    moment_tensor_matrices_from_fm_grid,
    read_velocity_model,
    source_takeoff_vectors_for_intensity,
)
from config import read_config_file, read_station_file
from data import generate_tt, gen_fm_grid, readsegy
from serial_processing import find_peak_idx, preprocess_jssa, preprocess_ssa
from stackCU import calc_position as calc_position_ssa
from stackCU import stack_CUDA
from stackMechCU import calc_position as calc_position_jssa
from stackMechCU import gen_intensity_CUDA, stack_mech_CUDA
from synthetic_pipeline_io import pipeline_file_extension_from_env, pipeline_reader_from_env


def update_jssa_conf(base_conf, x, y, z):
    conf = dict(base_conf)
    conf["SearchOriginX"] = x - conf["SearchSizeX"] * conf["GridSpacingX"] / 2
    conf["SearchOriginY"] = y - conf["SearchSizeY"] * conf["GridSpacingX"] / 2
    conf["SearchOriginZ"] = z - conf["SearchSizeZ"] * conf["GridSpacingZ"] / 2
    return conf


def station_subset(station, indices):
    return {
        "sta_ids": [station["sta_ids"][i] for i in indices],
        "x": [station["x"][i] for i in indices],
        "y": [station["y"][i] for i in indices],
        "z": [station["z"][i] for i in indices],
    }


def select_station_components(data, station_indices):
    rows = []
    for idx in station_indices:
        rows.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    return data[rows, :]


def chunked_top2(values, chunk_size=5_000_000):
    flat = values.ravel()
    top1 = -np.inf
    top2 = -np.inf
    for start in range(0, flat.size, chunk_size):
        chunk = flat[start : start + chunk_size]
        if chunk.size == 0:
            continue
        if chunk.size == 1:
            c1 = float(chunk[0])
            c2 = -np.inf
        else:
            pair = np.partition(chunk, -2)[-2:]
            c2 = float(pair[0])
            c1 = float(pair[1])
            if c2 > c1:
                c1, c2 = c2, c1
        if c1 > top1:
            top2 = max(top1, c2)
            top1 = c1
        elif c1 > top2:
            top2 = c1
    return top1, top2


def normalized_entropy(scores):
    scores = np.asarray(scores, dtype=np.float64)
    weights = scores - np.nanmin(scores)
    total = np.nansum(weights)
    if not np.isfinite(total) or total <= 1e-12:
        return math.nan
    p = weights / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(len(scores)))


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def holdout_metrics(data_jssa_evt, tt_jssa, intensity_full, conf_jssa, max_index, holdout_indices, sample_rate):
    if len(holdout_indices) == 0:
        return {
            "holdout_polarity_match": math.nan,
            "holdout_amp_corr": math.nan,
            "holdout_norm_residual": math.nan,
        }

    # jSSA result is reshaped as (x, y, z, fm, time); intensity uses flattened
    # grid index with z changing fastest, matching numpy C-order raveling.
    # np.ravel_multi_index is therefore safe for (x, y, z).
    grid_shape = (
        int(conf_jssa["SearchSizeX"]),
        int(conf_jssa["SearchSizeY"]),
        int(conf_jssa["SearchSizeZ"]),
    )
    grid_flat = int(np.ravel_multi_index(max_index[:3], grid_shape))

    fm_idx = int(max_index[3])
    origin_sample = int(max_index[4])
    z_data = data_jssa_evt[2::3, :]
    obs = []
    pred = []
    for sta_idx in holdout_indices:
        sample_idx = origin_sample + int(round(float(tt_jssa[sta_idx, grid_flat]) * sample_rate))
        if 0 <= sample_idx < z_data.shape[1]:
            obs.append(float(z_data[sta_idx, sample_idx]))
            pred.append(float(intensity_full[fm_idx, grid_flat, sta_idx]))

    if len(obs) == 0:
        return {
            "holdout_polarity_match": math.nan,
            "holdout_amp_corr": math.nan,
            "holdout_norm_residual": math.nan,
        }

    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    nonzero = (np.abs(obs) > 1e-12) & (np.abs(pred) > 1e-12)
    polarity = math.nan
    if np.any(nonzero):
        polarity = float(np.mean(np.sign(obs[nonzero]) == np.sign(pred[nonzero])))

    corr = corrcoef_safe(obs, pred)
    denom = float(np.dot(pred, pred))
    if denom <= 1e-12 or np.linalg.norm(obs) <= 1e-12:
        residual = math.nan
    else:
        scale = float(np.dot(obs, pred) / denom)
        residual = float(np.linalg.norm(obs - scale * pred) / (np.linalg.norm(obs) + 1e-12))

    return {
        "holdout_polarity_match": polarity,
        "holdout_amp_corr": corr,
        "holdout_norm_residual": residual,
    }


def infer_one_jssa(
    data_jssa_evt,
    sample_rate,
    tt_jssa,
    intensity_full,
    conf_jssa,
    fm_grid,
    station_indices,
    holdout_indices=None,
):
    data_train = select_station_components(data_jssa_evt, station_indices)
    tt_train = tt_jssa[station_indices, :]
    intensity_train = intensity_full[:, :, station_indices]

    result = stack_mech_CUDA(data_train, sample_rate, tt_train, intensity_train)
    if isinstance(result, int) and result == -1:
        return None

    result = result.reshape(
        conf_jssa["SearchSizeX"],
        conf_jssa["SearchSizeY"],
        conf_jssa["SearchSizeZ"],
        fm_grid.shape[0],
        data_jssa_evt.shape[1],
    )
    max_index = np.array(np.unravel_index(np.argmax(result), result.shape), dtype=np.int64)
    top1, top2 = chunked_top2(result)
    x, y, z, fm_idx, t_rel = calc_position_jssa(conf_jssa, max_index, sample_rate)
    fm = fm_grid[fm_idx]
    fm_scores = result[max_index[0], max_index[1], max_index[2], :, max_index[4]]
    metrics = {
        "x": x,
        "y": y,
        "z": z,
        "strike": float(fm[0]),
        "dip": float(fm[1]),
        "rake": float(fm[2]),
        "fm_idx": int(fm_idx),
        "time_rel_s": float(t_rel),
        "max_value": float(top1),
        "top2_value": float(top2),
        "top1_top2_gap": float(top1 - top2) if np.isfinite(top2) else math.nan,
        "top1_top2_ratio": float(top1 / (top2 + 1e-12)) if np.isfinite(top2) and abs(top2) > 1e-12 else math.nan,
        "mechanism_entropy": normalized_entropy(fm_scores),
        "max_ix": int(max_index[0]),
        "max_iy": int(max_index[1]),
        "max_iz": int(max_index[2]),
        "max_it": int(max_index[4]),
    }
    if holdout_indices is not None:
        metrics.update(holdout_metrics(data_jssa_evt, tt_jssa, intensity_full, conf_jssa, max_index, holdout_indices, sample_rate))
    else:
        metrics.update(
            {
                "holdout_polarity_match": math.nan,
                "holdout_amp_corr": math.nan,
                "holdout_norm_residual": math.nan,
            }
        )
    del result
    gc.collect()
    return metrics


def make_splits(n_sta, holdout_frac, n_splits, rng):
    all_indices = np.arange(n_sta)
    splits = []
    holdout_n = max(1, int(round(n_sta * holdout_frac)))
    holdout_n = min(holdout_n, n_sta - 2)
    for _ in range(n_splits):
        perm = rng.permutation(all_indices)
        holdout = np.sort(perm[:holdout_n])
        train = np.sort(perm[holdout_n:])
        splits.append((train.tolist(), holdout.tolist()))
    return splits


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by(rows, group_fields):
    summary = {}
    numeric_fields = [
        "max_value",
        "top1_top2_gap",
        "top1_top2_ratio",
        "mechanism_entropy",
        "holdout_polarity_match",
        "holdout_amp_corr",
        "holdout_norm_residual",
    ]
    keys = sorted({tuple(r[field] for field in group_fields) for r in rows})
    for key in keys:
        sub = [r for r in rows if tuple(r[field] for field in group_fields) == key]
        label = " / ".join(str(v) for v in key)
        item = {"count": len(sub)}
        for field, value in zip(group_fields, key):
            item[field] = value
        for field in numeric_fields:
            vals = np.asarray([r[field] for r in sub], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            item[field + "_mean"] = float(np.mean(vals)) if len(vals) else math.nan
            item[field + "_median"] = float(np.median(vals)) if len(vals) else math.nan
        summary[label] = item
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare geometry and source-takeoff jSSA on real-field waveforms without true mechanism labels."
    )
    parser.add_argument("--folder-data", default=os.environ.get("MSI_PIPELINE_FOLDER_DATA", "./waveform"))
    parser.add_argument("--folder-conf", default=os.environ.get("MSI_PIPELINE_FOLDER_CONF", "./conf"))
    parser.add_argument("--station-file", default=None)
    parser.add_argument("--out-dir", default="result/real_source_takeoff_jssa_validation")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--peak-threshold", type=float, default=float(os.environ.get("MSI_PIPELINE_PEAK_THRESHOLD", "2.5")))
    parser.add_argument("--attenuation", type=float, default=0.2)
    parser.add_argument("--intensity-chunk-size", type=int, default=64)
    parser.add_argument("--directions", nargs="+", choices=["geometry", "source-takeoff"], default=["geometry", "source-takeoff"])
    parser.add_argument("--splits", type=int, default=0, help="Number of station holdout splits per event. 0 means all-station comparison only.")
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260528)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folder_conf = args.folder_conf
    station_file = args.station_file or str(Path(folder_conf) / "station_sorted.txt")
    conf_ssa = read_config_file(str(Path(folder_conf) / "conf_ssa.txt"))
    conf_jssa_base = read_config_file(str(Path(folder_conf) / "conf_jssa.txt"))
    station = read_station_file(station_file, conf_ssa)
    n_sta = len(station["sta_ids"])
    fm_grid = gen_fm_grid(conf_jssa_base)
    fm_matrices = moment_tensor_matrices_from_fm_grid(fm_grid)
    velocity_model = read_velocity_model(str(Path(folder_conf) / "vel_jssa.txt"))

    read_waveform = pipeline_reader_from_env(readsegy)
    file_extension = pipeline_file_extension_from_env()
    file_list = sorted(f for f in os.listdir(args.folder_data) if f.endswith(file_extension))
    if args.max_files:
        file_list = file_list[: args.max_files]

    print("=" * 70)
    print("Real-field geometry vs source-takeoff jSSA validation")
    print("=" * 70)
    print(f"waveform folder : {args.folder_data}")
    print(f"station file    : {station_file}")
    print(f"files           : {len(file_list)}")
    print(f"station count   : {n_sta}")
    print(f"fm grid         : {fm_grid.shape}")
    print(f"directions      : {args.directions}")
    print(f"splits/event    : {args.splits}")

    tt_ssa, _, _ = generate_tt(str(Path(folder_conf) / "conf_ssa.txt"), str(Path(folder_conf) / "vel.txt"), station_file)
    rng = np.random.default_rng(args.seed)
    rows = []
    errors = []
    total_events = 0
    start_time = time.time()

    for file_idx, file_name in enumerate(file_list, start=1):
        if args.max_events and total_events >= args.max_events:
            break
        file_path = str(Path(args.folder_data) / file_name)
        print(f"\n[{file_idx}/{len(file_list)}] {file_name}")
        try:
            data, sta_ids, sample_rate, datetime_start = read_waveform(file_path)
            if data.shape[0] != n_sta * 3:
                msg = f"data traces {data.shape[0]} do not match station count {n_sta} * 3"
                print(f"  [skip] {msg}")
                errors.append({"file": file_name, "error": msg})
                continue
            tt_max_sample = int(float(np.max(tt_ssa)) * sample_rate)
            data_ssa = preprocess_ssa(data, sample_rate)
            data_jssa = preprocess_jssa(data, sample_rate)
            result_ssa = stack_CUDA(data_ssa, sample_rate, tt_ssa)
            if isinstance(result_ssa, int) and result_ssa == -1:
                errors.append({"file": file_name, "error": "SSA failed"})
                continue
            result_ssa = result_ssa.reshape(
                conf_ssa["SearchSizeX"],
                conf_ssa["SearchSizeY"],
                conf_ssa["SearchSizeZ"],
                data.shape[1],
            )
            max_value_per_time = np.max(result_ssa, axis=(0, 1, 2))
            peaks_info = find_peak_idx(max_value_per_time, args.peak_threshold, 0.5, 1)
            print(f"  peaks: {len(peaks_info)}")

            for local_event_idx, peak in enumerate(peaks_info):
                if args.max_events and total_events >= args.max_events:
                    break
                peak_idx = int(peak["raise_idx"])
                ssa_start = max(0, peak_idx - 10)
                ssa_end = min(result_ssa.shape[3], peak_idx + 1)
                if ssa_end <= ssa_start:
                    continue
                ssa_window = result_ssa[:, :, :, ssa_start:ssa_end]
                max_index_ssa = np.array(np.unravel_index(np.argmax(ssa_window), ssa_window.shape), dtype=np.int64)
                x0, y0, z0, t0 = calc_position_ssa(conf_ssa, max_index_ssa, sample_rate)
                event_time = datetime_start + datetime.timedelta(seconds=t0 + ssa_start / sample_rate)
                conf_jssa = update_jssa_conf(conf_jssa_base, x0, y0, z0)
                tt_jssa, incident, azimuth = generate_tt(conf_jssa, str(Path(folder_conf) / "vel_jssa.txt"), station_file)

                jssa_start = max(0, peak_idx - 50)
                jssa_end = min(data_jssa.shape[1], peak_idx + tt_max_sample + 30)
                if jssa_end <= jssa_start:
                    continue
                data_jssa_evt = data_jssa[:, jssa_start:jssa_end]
                if data_jssa_evt.shape[1] <= 0:
                    continue

                split_defs = [("full", 0, list(range(n_sta)), [])]
                if args.splits > 0:
                    for split_id, (train, holdout) in enumerate(make_splits(n_sta, args.holdout_frac, args.splits, rng), start=1):
                        split_defs.append(("holdout", split_id, train, holdout))

                for direction in args.directions:
                    print(f"  event {total_events + 1}: {direction}")
                    if direction == "geometry":
                        intensity_full = gen_intensity_CUDA(fm_grid, conf_jssa, station, args.attenuation)
                    else:
                        unit, distance_km = source_takeoff_vectors_for_intensity(
                            conf_jssa,
                            station,
                            incident.astype(np.float32),
                            azimuth.astype(np.float32),
                            velocity_model,
                        )
                        intensity_full = compute_intensity_from_directions(
                            fm_grid,
                            unit,
                            distance_km,
                            args.attenuation,
                            args.intensity_chunk_size,
                            fm_matrices=fm_matrices,
                        )

                    for split_kind, split_id, train_indices, holdout_indices in split_defs:
                        metrics = infer_one_jssa(
                            data_jssa_evt,
                            sample_rate,
                            tt_jssa,
                            intensity_full,
                            conf_jssa,
                            fm_grid,
                            train_indices,
                            holdout_indices if split_kind == "holdout" else None,
                        )
                        if metrics is None:
                            errors.append({"file": file_name, "event": total_events + 1, "direction": direction, "error": "jSSA failed"})
                            continue
                        row = {
                            "file": file_name,
                            "event_id": total_events + 1,
                            "local_event_idx": local_event_idx,
                            "peak_idx": peak_idx,
                            "event_time": event_time.isoformat(),
                            "ssa_x": x0,
                            "ssa_y": y0,
                            "ssa_z": z0,
                            "direction": direction,
                            "split_kind": split_kind,
                            "split_id": split_id,
                            "n_train_sta": len(train_indices),
                            "n_holdout_sta": len(holdout_indices),
                        }
                        row.update(metrics)
                        rows.append(row)
                    del intensity_full
                    gc.collect()
                total_events += 1
        except Exception as exc:
            print(f"  [error] {exc}")
            errors.append({"file": file_name, "error": repr(exc)})
        finally:
            gc.collect()

    event_csv = out_dir / "real_jssa_geometry_vs_source_takeoff_events.csv"
    write_csv(event_csv, rows)
    summary = {
        "elapsed_s": time.time() - start_time,
        "n_rows": len(rows),
        "n_errors": len(errors),
        "settings": vars(args),
        "aggregate_by_direction": summarize_by(rows, ["direction"]),
        "aggregate_by_direction_split": summarize_by(rows, ["direction", "split_kind"]),
        "errors": errors,
    }
    summary_json = out_dir / "real_jssa_geometry_vs_source_takeoff_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Done")
    print(f"events csv  : {event_csv}")
    print(f"summary json: {summary_json}")
    print(json.dumps(summary["aggregate_by_direction_split"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
