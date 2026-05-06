import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


BFNET_SHAPE = (8, 8, 8, 24, 7, 24)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_record_path(csv_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(csv_path).parent / path


def read_csv_records(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_records(path, records, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def normalize_angle_360(angle):
    return float(angle % 360.0)


def normalize_angle_180(angle):
    value = (angle + 180.0) % 360.0 - 180.0
    if value <= -180.0:
        value += 360.0
    return float(value)


def normalize_plane(strike, dip, rake):
    strike = normalize_angle_360(strike)
    rake = normalize_angle_180(rake)
    dip = float(dip)
    if dip < 0.0:
        dip = -dip
        strike = normalize_angle_360(strike + 180.0)
        rake = normalize_angle_180(-rake)
    if dip > 90.0:
        dip = 180.0 - dip
        strike = normalize_angle_360(strike + 180.0)
        rake = normalize_angle_180(-rake)
    return strike, dip, rake


def angle_diff_360(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def focal_planes(strike, dip, rake):
    strike, dip, rake = normalize_plane(strike, dip, rake)
    planes = [(strike, dip, rake)]
    try:
        from obspy.imaging.beachball import aux_plane

        s2, d2, r2 = aux_plane(strike, dip, rake)
        planes.append(normalize_plane(s2, d2, r2))
    except Exception:
        pass
    return planes


def mechanism_error(pred_sdr, true_sdr):
    pred_planes = focal_planes(*pred_sdr)
    true_planes = focal_planes(*true_sdr)
    best = None
    for pp in pred_planes:
        for tp in true_planes:
            err = (
                angle_diff_360(pp[0], tp[0]),
                abs(pp[1] - tp[1]),
                angle_diff_360(pp[2], tp[2]),
            )
            score = sum(err)
            if best is None or score < best[0]:
                best = (score, err)
    return best[1]


def encode_sdr_sincos_np(strike, dip, rake):
    strike = np.deg2rad(strike)
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    return np.array(
        [
            np.sin(strike),
            np.cos(strike),
            np.sin(dip),
            np.cos(dip),
            np.sin(rake),
            np.cos(rake),
        ],
        dtype=np.float32,
    )


def decode_sincos_np(output):
    output = np.asarray(output)
    strike = np.rad2deg(np.arctan2(output[..., 0], output[..., 1]))
    dip = np.rad2deg(np.remainder(np.arctan2(output[..., 2], output[..., 3]), np.pi))
    rake = np.rad2deg(np.arctan2(output[..., 4], output[..., 5]))
    if output.ndim == 1:
        return normalize_plane(float(strike), float(dip), float(rake))
    planes = [normalize_plane(float(s), float(d), float(r)) for s, d, r in zip(strike, dip, rake)]
    return np.asarray(planes, dtype=np.float32)


def sdr_to_moment_tensor_np(strike, dip, rake):
    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)

    m_xx = -np.sin(dip) * np.cos(rake) * np.sin(2 * strike) - np.sin(2 * dip) * np.sin(rake) * np.sin(strike) ** 2
    m_yy = np.sin(dip) * np.cos(rake) * np.sin(2 * strike) - np.sin(2 * dip) * np.sin(rake) * np.cos(strike) ** 2
    m_xy = np.sin(dip) * np.cos(rake) * np.cos(2 * strike) + 0.5 * np.sin(2 * dip) * np.sin(rake) * np.sin(2 * strike)
    m_xz = -np.cos(dip) * np.cos(rake) * np.cos(strike) - np.cos(2 * dip) * np.sin(rake) * np.sin(strike)
    m_yz = -np.cos(dip) * np.cos(rake) * np.sin(strike) + np.cos(2 * dip) * np.sin(rake) * np.cos(strike)
    m_zz = np.sin(2 * dip) * np.sin(rake)
    return np.array([m_xx, m_yy, m_zz, m_xy, m_xz, m_yz], dtype=np.float32)


def sdr_to_moment_tensor_torch(sdr_deg):
    strike = torch.deg2rad(sdr_deg[:, 0])
    dip = torch.deg2rad(sdr_deg[:, 1])
    rake = torch.deg2rad(sdr_deg[:, 2])

    sin_dip = torch.sin(dip)
    cos_dip = torch.cos(dip)
    sin_rake = torch.sin(rake)
    cos_rake = torch.cos(rake)
    sin_strike = torch.sin(strike)
    cos_strike = torch.cos(strike)
    sin_2strike = torch.sin(2 * strike)
    cos_2strike = torch.cos(2 * strike)
    sin_2dip = torch.sin(2 * dip)
    cos_2dip = torch.cos(2 * dip)

    m_xx = -sin_dip * cos_rake * sin_2strike - sin_2dip * sin_rake * sin_strike**2
    m_yy = sin_dip * cos_rake * sin_2strike - sin_2dip * sin_rake * cos_strike**2
    m_xy = sin_dip * cos_rake * cos_2strike + 0.5 * sin_2dip * sin_rake * sin_2strike
    m_xz = -cos_dip * cos_rake * cos_strike - cos_2dip * sin_rake * sin_strike
    m_yz = -cos_dip * cos_rake * sin_strike + cos_2dip * sin_rake * cos_strike
    m_zz = sin_2dip * sin_rake
    return torch.stack([m_xx, m_yy, m_zz, m_xy, m_xz, m_yz], dim=1)


def sincos_to_moment_tensor_torch(output):
    strike = torch.atan2(output[:, 0], output[:, 1])
    dip = torch.remainder(torch.atan2(output[:, 2], output[:, 3]), torch.pi)
    rake = torch.atan2(output[:, 4], output[:, 5])

    sin_dip = torch.sin(dip)
    cos_dip = torch.cos(dip)
    sin_rake = torch.sin(rake)
    cos_rake = torch.cos(rake)
    sin_strike = torch.sin(strike)
    cos_strike = torch.cos(strike)
    sin_2strike = torch.sin(2 * strike)
    cos_2strike = torch.cos(2 * strike)
    sin_2dip = torch.sin(2 * dip)
    cos_2dip = torch.cos(2 * dip)

    m_xx = -sin_dip * cos_rake * sin_2strike - sin_2dip * sin_rake * sin_strike**2
    m_yy = sin_dip * cos_rake * sin_2strike - sin_2dip * sin_rake * cos_strike**2
    m_xy = sin_dip * cos_rake * cos_2strike + 0.5 * sin_2dip * sin_rake * sin_2strike
    m_xz = -cos_dip * cos_rake * cos_strike - cos_2dip * sin_rake * sin_strike
    m_yz = -cos_dip * cos_rake * sin_strike + cos_2dip * sin_rake * cos_strike
    m_zz = sin_2dip * sin_rake
    return torch.stack([m_xx, m_yy, m_zz, m_xy, m_xz, m_yz], dim=1)


def minmax_normalize_tensor(x, eps=1e-8):
    dims = tuple(range(1, x.ndim))
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def load_state_dict_flexible(path, map_location):
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def ricker_wavelet(freq_hz, sample_rate, duration_s):
    n = max(3, int(round(duration_s * sample_rate)))
    if n % 2 == 0:
        n += 1
    t = (np.arange(n, dtype=np.float32) - n // 2) / float(sample_rate)
    arg = (math.pi * freq_hz * t) ** 2
    wavelet = (1.0 - 2.0 * arg) * np.exp(-arg)
    wavelet /= np.max(np.abs(wavelet)) + 1e-8
    return wavelet.astype(np.float32)

