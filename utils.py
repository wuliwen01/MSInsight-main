import segyio
from matplotlib import pyplot as plt
from datetime import datetime
from obspy.signal.trigger import classic_sta_lta
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball
import torch
import numpy as np

import torch.nn.functional as F


def angle_diff_deg(a, b):
    """
    Calculate the minimum difference between two angles (in degrees), considering periodicity.
    """
    a = a % 360
    b = b % 360
    b_eq = (b + 180) % 360
    diff1 = abs((a - b + 180) % 360 - 180)
    diff2 = abs((a - b_eq + 180) % 360 - 180)
    return min(diff1, diff2)


def tensor_cosine_loss(M_pred, M_true, eps=1e-8):
    """
    Calculate cosine similarity loss between moment tensors.

    Input:
        M_pred: [B, 6], model predicted tensor (6 independent components)
        M_true: [B, 6], ground truth tensor
    Returns:
        loss: scalar, 1 - cos_sim
    """
    # Normalization
    M_pred_norm = F.normalize(M_pred, p=2, dim=1, eps=eps)
    M_true_norm = F.normalize(M_true, p=2, dim=1, eps=eps)

    # Calculate batch cosine similarity
    cos_sim = torch.sum(M_pred_norm * M_true_norm, dim=1)  # shape [B]

    # Use 1 - similarity as loss (smaller for more similar tensors)
    loss = 1.0 - cos_sim
    return loss.mean()


def fault_parameters_to_moment_tensor(strike, dip, rake):
    """
    Convert fault parameters (strike, dip, rake) to moment tensor.
    Parameters are in degrees.
    x-axis points north, y-axis points east, z-axis points down.
    """
    # Convert angles to radians
    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)

    # Calculate moment tensor components
    m_xx = -np.sin(dip) * np.cos(rake) * np.sin(2 * strike) - \
        np.sin(2 * dip) * np.sin(rake) * np.sin(strike)**2
    m_yy = np.sin(dip) * np.cos(rake) * np.sin(2 * strike) - \
        np.sin(2 * dip) * np.sin(rake) * np.cos(strike)**2
    m_xy = np.sin(dip) * np.cos(rake) * np.cos(2 * strike) + 0.5 * \
        np.sin(2 * dip) * np.sin(rake) * np.sin(2 * strike)
    m_xz = -np.cos(dip) * np.cos(rake) * np.cos(strike) - \
        np.cos(2 * dip) * np.sin(rake) * np.sin(strike)
    m_yz = -np.cos(dip) * np.cos(rake) * np.sin(strike) + \
        np.cos(2 * dip) * np.sin(rake) * np.cos(strike)
    m_zz = np.sin(2 * dip) * np.sin(rake)

    # Build moment tensor matrix
    moment_tensor = np.array([[m_xx, m_xy, m_xz],
                              [m_xy, m_yy, m_yz],
                              [m_xz, m_yz, m_zz]])
    # print(moment_tensor)
    return moment_tensor


def compute_moment_tensor_torch(output):
    # Extract sin/cos pairs
    strike_sin = output[:, 0]
    strike_cos = output[:, 1]
    dip_sin = output[:, 2]
    dip_cos = output[:, 3]
    rake_sin = output[:, 4]
    rake_cos = output[:, 5]
    # Restore angles
    strike = torch.atan2(strike_sin, strike_cos)
    dip = torch.atan2(dip_sin, dip_cos)
    rake = torch.atan2(rake_sin, rake_cos)
    dip = torch.remainder(dip, torch.pi)  # Restrict dip to [0, pi]

    # Compute moment tensor components
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
    m_xy = sin_dip * cos_rake * cos_2strike + \
        0.5 * sin_2dip * sin_rake * sin_2strike
    m_xz = -cos_dip * cos_rake * cos_strike - cos_2dip * sin_rake * sin_strike
    m_yz = -cos_dip * cos_rake * sin_strike + cos_2dip * sin_rake * cos_strike
    m_zz = sin_2dip * sin_rake

    # Return moment tensor
    return torch.stack([m_xx, m_yy, m_zz, m_xy, m_xz, m_yz], dim=1)


def plot_beachball(source, size=200, linewidth=1):
    """
    Plot a beachball diagram using given focal mechanism parameters.

    :param source: Dictionary containing focal mechanism parameters.
    :param size: Size of the beachball diagram.
    :param linewidth: Line width of the beachball diagram.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(3, 3))

    # Plot the beachball
    fig=beachball(source, size=size,
              linewidth=linewidth, facecolor='black', bgcolor='white', fig=fig)

    # Show the plot
    plt.show()


def datetime_fromtraceheader(header):
    y = header[157]
    d = header[159]
    h = header[161]
    m = header[163]
    s = header[165]
    time_str = "%d,%03d,%02d,%02d,%02d" % (y, d, h, m, s)
    format_str = "%Y,%j,%H,%M,%S"
    return datetime.strptime(time_str, format_str)


def get_name_from_int(num):
    nameint = num & 0xFFFFFFFF
    namebytes = nameint.to_bytes(4, byteorder='little', signed=False)
    return "".join(["%02X" % (i) for i in namebytes])  # [2:]


def readsegy(fname):
    data = []
    sta_ids = []
    with segyio.open(fname, ignore_geometry=True) as f:
        sample_rate = int(1000000/f.header[0][117])
        tnum = f.tracecount
        datetime_start = datetime_fromtraceheader(f.header[0])
        for i in range(tnum):
            # data.append(abs(f.trace[i]))
            data.append(f.trace[i])
        for i in range(int(tnum/3)):
            sta_ids.append(get_name_from_int(f.header[i*3][61]))
    return np.array(data), sta_ids, sample_rate, datetime_start


def stalta(data, nsta, nlta):
    """
    Calculate STA/LTA ratio for seismic data.

    Parameters:
        data: Input seismic data (numpy array).
        nsta: Number of samples in short time window.
        nlta: Number of samples in long time window.

    Returns:
        stalta: Calculated STA/LTA ratio (numpy array).
    """
    # Ensure window length does not exceed data length
    if len(data) < nlta:
        raise ValueError("Data length must be greater than long time window length.")

    result = classic_sta_lta(data, nsta, nlta)
    # result = carl_sta_trig(result, nsta, nlta, ratio=5.0, quiet=1.0)

    return result


def whightening(data, f1, f2, f3, f4, sample_rate, epsilon=1e-10):
    """
    Apply spectral whitening to single-channel seismic waveform data within a band-pass frequency range,
    using trapezoidal band-pass (f1~f2~f3~f4).

    Parameters:
    data: 1D numpy array representing single-channel seismic waveform data
    f1: Lower cutoff frequency (Hz)
    f2: Lower passband frequency (Hz)
    f3: Upper passband frequency (Hz)
    f4: Upper cutoff frequency (Hz)
    sample_rate: Sampling rate of the data (Hz)
    epsilon: Small constant to avoid division by zero

    Returns:
    whitened_data: Whitened time-domain signal
    """
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)
    magnitude = np.abs(fft_data)
    whitened_fft_data = np.zeros_like(fft_data, dtype=complex)

    # Trapezoidal band-pass window function
    def trapezoid_window(f):
        af = np.abs(f)
        if af < f1 or af > f4:
            return 0.0
        elif f1 <= af < f2:
            return (af - f1) / (f2 - f1)
        elif f2 <= af <= f3:
            return 1.0
        elif f3 < af <= f4:
            return (f4 - af) / (f4 - f3)
        else:
            return 0.0

    for i, f in enumerate(freqs):
        w = trapezoid_window(f)
        if w > 0 and magnitude[i] > epsilon:
            whitened_fft_data[i] = fft_data[i] / magnitude[i] * w

    whitened_data = np.fft.ifft(whitened_fft_data)
    return np.real(whitened_data)


def preprocess(data_raw, sample_rate, w_l=25, w_h=60):
    data = np.copy(data_raw)
    for i in range(len(data)):
        # Remove mean
        data[i] = data[i] - np.mean(data[i])
        # Spectral whitening (do not apply to synthetic data)
        data[i] = whightening(data[i], w_l-20, w_l, w_h, w_h+20, sample_rate)
        # STALTA
        data[i] = stalta(data[i], 5, 40)
        # Data normalization
        # data[i] = data[i] / np.max(np.abs(data[i]))
    return np.asarray(data)
