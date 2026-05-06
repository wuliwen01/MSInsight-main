import argparse
from pathlib import Path

import numpy as np

from config import read_config_file, read_station_file
from synthetic_bfnet_utils import (
    ensure_dir,
    ricker_wavelet,
    sdr_to_moment_tensor_np,
    write_csv_records,
    write_json,
)


def sample_point_in_sphere(rng, center, radius):
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction) + 1e-12
    distance = radius * (rng.random() ** (1.0 / 3.0))
    return center + direction * distance


def radiation_value(moment_tensor, vector_m):
    norm = np.linalg.norm(vector_m)
    if norm < 1e-12:
        return 0.0
    unit = vector_m / norm
    return float(unit @ moment_tensor @ unit)


def insert_wavelet(trace, center_sample, wavelet, amplitude):
    half = len(wavelet) // 2
    start = center_sample - half
    end = start + len(wavelet)
    w_start = 0
    w_end = len(wavelet)
    if start < 0:
        w_start = -start
        start = 0
    if end > len(trace):
        w_end -= end - len(trace)
        end = len(trace)
    if end > start and w_end > w_start:
        trace[start:end] += amplitude * wavelet[w_start:w_end]


def build_event_waveform(event, station, args, wavelet, rng):
    n_sta = len(station["x"])
    data = np.zeros((n_sta * 3, args.n_samples), dtype=np.float32)
    mt = sdr_to_moment_tensor_np(event["strike"], event["dip"], event["rake"])
    mt_matrix = np.array(
        [
            [mt[0], mt[3], mt[4]],
            [mt[3], mt[1], mt[5]],
            [mt[4], mt[5], mt[2]],
        ],
        dtype=np.float32,
    )

    source_m = np.array([event["x"], event["y"], event["z"]], dtype=np.float32) * 1000.0
    for i in range(n_sta):
        station_m = np.array([station["x"][i], station["y"][i], station["z"][i]], dtype=np.float32)
        distance_km = np.linalg.norm(station_m - source_m) / 1000.0
        travel_samples = int(round(distance_km / args.vp * args.sample_rate))
        center_sample = args.origin_sample + travel_samples

        # Moment tensor coordinates: north, east, down.
        vector_m = np.array(
            [
                station["y"][i] - source_m[1],
                station["x"][i] - source_m[0],
                station["z"][i] - source_m[2],
            ],
            dtype=np.float32,
        )
        amp = radiation_value(mt_matrix, vector_m) * np.exp(-args.attenuation * distance_km)
        insert_wavelet(data[i * 3 + 2], center_sample, wavelet, amp)

    if args.snr_db is not None:
        z_data = data[2::3]
        signal_power = float(np.mean(z_data**2))
        if signal_power > 0:
            noise_power = signal_power / (10.0 ** (args.snr_db / 10.0))
            noise = rng.normal(0.0, np.sqrt(noise_power), size=data.shape).astype(np.float32)
            data += noise

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate small synthetic waveform .npy files for BFNet experiments.")
    parser.add_argument("--out-dir", default="synthetic/waveforms")
    parser.add_argument("--event-count", type=int, default=64)
    parser.add_argument("--conf", default="conf/conf_synth_jssa.txt")
    parser.add_argument("--station", default="conf/station_synth_dataset_a.xyz")
    parser.add_argument("--sample-rate", type=float, default=500.0)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--origin-sample", type=int, default=180)
    parser.add_argument("--vp", type=float, default=3.5, help="Homogeneous P velocity in km/s.")
    parser.add_argument("--wavelet-freq", type=float, default=5.0)
    parser.add_argument("--wavelet-duration", type=float, default=0.8)
    parser.add_argument("--attenuation", type=float, default=0.5)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--center-z", type=float, default=-0.8)
    parser.add_argument("--radius-km", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    conf = read_config_file(args.conf)
    station = read_station_file(args.station, conf)
    rng = np.random.default_rng(args.seed)
    wavelet = ricker_wavelet(args.wavelet_freq, args.sample_rate, args.wavelet_duration)
    center = np.array([args.center_x, args.center_y, args.center_z], dtype=np.float32)

    records = []
    for idx in range(args.event_count):
        x, y, z = sample_point_in_sphere(rng, center, args.radius_km)
        event = {
            "event_id": f"ev_{idx:06d}",
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "strike": float(rng.uniform(0.0, 360.0)),
            "dip": float(rng.uniform(0.0, 90.0)),
            "rake": float(rng.uniform(-180.0, 180.0)),
        }
        data = build_event_waveform(event, station, args, wavelet, rng)
        file_name = f"{event['event_id']}.npy"
        np.save(out_dir / file_name, data.astype(np.float32))

        records.append(
            {
                **event,
                "waveform_path": file_name,
                "sample_rate": args.sample_rate,
                "n_samples": args.n_samples,
                "origin_sample": args.origin_sample,
                "snr_db": "" if args.snr_db is None else args.snr_db,
            }
        )
        print(f"[{idx + 1}/{args.event_count}] wrote {file_name} shape={data.shape}")

    fieldnames = [
        "event_id",
        "waveform_path",
        "x",
        "y",
        "z",
        "strike",
        "dip",
        "rake",
        "sample_rate",
        "n_samples",
        "origin_sample",
        "snr_db",
    ]
    write_csv_records(out_dir / "events.csv", records, fieldnames)
    write_json(out_dir / "generation_config.json", vars(args))
    print(f"Metadata saved to {out_dir / 'events.csv'}")


if __name__ == "__main__":
    main()
