# from datetime import timedelta
from config import read_config_file
from data import generate_tt, stalta
import numpy as np
import matplotlib.pyplot as plt
# import os
from draw import draw_maxbrightness
# from tqdm import tqdm
# from multiprocessing import Pool, shared_memory
import ctypes


def load_ttheader(file_path):
    """
    Load traveltime header from file.
    eg:
    166 102 50 5.5 8 -0.2 0.01 0.01 0.01 TIME
    000338 7.19004 8.4151 1.44578
    TRANSFORM  NONE
    format:
    xCnt, yCnt, zCnt, xStart, yStart, zStart, xDelta, yDelta, zDelta, dataType
    staID, stax, stay, staz
    """
    conf = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        line_1 = lines[0].split()
        line_2 = lines[1].split()
        # print(file_path, line_1, line_2)
        conf['xCnt'], conf['yCnt'], conf['zCnt'] = int(
            line_1[0]), int(line_1[1]), int(line_1[2])
        conf['xStart'], conf['yStart'], conf['zStart'] = float(
            line_1[3]), float(line_1[4]), float(line_1[5])
        conf['xDelta'], conf['yDelta'], conf['zDelta'] = float(
            line_1[6]), float(line_1[7]), float(line_1[8])
        conf['dataType'] = line_1[9]
        conf['staID'] = line_2[0]
        conf['stax'], conf['stay'], conf['staz'] = float(
            line_2[1]), float(line_2[2]), float(line_2[3])

    return conf


def stack_CUDA(data, sample_rate, traveltime, chn='nez'):
    """
    CUDA-accelerated stacking algorithm
    Args:
        data (ndarray): (n_sta*3, n_samples)
        sample_rate (float): sampling rate
        traveltime (ndarray): (n_sta, n_grid)
    Returns:
        ndarray: (n_grid, n_samples)
    """
    # Calculate parameters
    n_sta, n_grid = traveltime.shape

    n_samples = data.shape[1]
    # Convert traveltime to sample points
    tt_samples = (traveltime * sample_rate).round().astype(np.int32)
    # Extract z-component of data
    if chn == 'nez':
        data = data[2::3, :]
    
    # Convert data to C-style array
    data = np.ascontiguousarray(data, dtype=np.float32)
    tt_samples = np.ascontiguousarray(tt_samples, dtype=np.int32)
    # Initialize result array
    result = np.zeros((n_grid, n_samples), dtype=np.float32)

    clib = ctypes.CDLL('./SSA.dll')
    stackCUDA = getattr(clib, "?stackCUDA@@YAHPEBMPEBHHHHPEAM@Z")
    # Parameter configuration
    stackCUDA.argtypes = [ctypes.POINTER(ctypes.c_float),  # data
                          ctypes.POINTER(ctypes.c_int32),  # tt_samples
                          ctypes.c_int,                    # n_sta
                          ctypes.c_int,                    # n_samples
                          ctypes.c_int,                    # n_grid
                          ctypes.POINTER(ctypes.c_float)]  # result
    # Return value configuration
    stackCUDA.restype = ctypes.c_int

    # Data pointers
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    tt_samples_ptr = tt_samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Call C function
    print("Calling CUDA function... params:")
    print(f"n_sta: {n_sta}, n_samples: {n_samples}, n_grid: {n_grid}")
    ret = stackCUDA(data_ptr, tt_samples_ptr,
                    n_sta, n_samples, n_grid, result_ptr)
    if ret != 0:
        print("CUDA Error!")
        return -1
    # else:
    #     print("CUDA result shape:", result.shape)
    return result


def calc_position(conf, max_index, sample_rate):
    # Convert max_index to real coordinates
    x = max_index[0] * conf['GridSpacingX'] + conf['SearchOriginX']
    y = max_index[1] * conf['GridSpacingX'] + conf['SearchOriginY']
    z = max_index[2] * conf['GridSpacingZ'] + conf['SearchOriginZ']
    t = max_index[3] / sample_rate
    # round to 3 decimal places
    x = round(x, 3)
    y = round(y, 3)
    z = round(z, 3)
    t = round(t, 3)
    return x, y, z, t


def show_result(result, conf, sample_rate, show=True):
    max_index = np.array(np.unravel_index(np.argmax(result), result.shape))
    max_value = np.max(result)
    x, y, z, t = calc_position(conf, max_index, sample_rate)
    if show:
        print("max index:", max_index)
        print("max value:", max_value)
        print(f"max position: ({x:.3f}, {y:.3f}, {z:.3f}, {t:.3f})")
    return max_index


def preprocess(data_raw, sample_rate):
    data = np.copy(data_raw)
    for i in range(len(data)):
        # Remove mean
        data[i] = data[i] - np.mean(data[i])
        # Spectral whitening, not applied to synthetic data
        # data[i] = whightening(data[i], 15, 60, sample_rate)
        # STALTA
        data[i] = stalta(data[i], 5, 40)
        # Data normalization
        data[i] = data[i] / np.max(np.abs(data[i]))
    return np.asarray(data)


if __name__ == "__main__":
    # fpath = "F:/SimSeisData/out_chengyu/wav_120"
    # fpath = "F:/SimSeisData/out_chengyu/wav_4"
    # data_raw, sta_ids, sample_rate, datetime_start = readsac(
    #     fpath)
    simdata_path = "G:/seisdata/sim_waveform_250913/"
    fname = simdata_path + "ev_0002.npy"
    data_raw = np.load(fname)
    print("data shape:", data_raw.shape)
    sample_rate = 500

    folder_conf = './confsim'
    fname_conf = f'{folder_conf}/conf_ssa_draw.txt'
    fname_vel = f'{folder_conf}/vel_250913.txt'
    fname_sta = f'{folder_conf}/stloc_250912.xyz'
    conf = read_config_file(fname_conf)
    tt, inc, az = generate_tt(fname_conf,
                              fname_vel, fname_sta)
    print("tt shape:", tt.shape)

    # Data preprocessing
    data = preprocess(data_raw, sample_rate)
    # data = data[:, :3000]

    # Timer
    import time
    print("start stack")
    # result = stack(data, sample_rate, tt)
    start = time.time()
    result = stack_CUDA(data, sample_rate, tt, chn='z')
    end = time.time()
    print(f"stack time:{end - start:.3f}s")
    print("result shape:", result.shape)

    # Check result type
    if type(result) == int and result == -1:
        print("stack_CU error")
        exit(0)
    result = result.reshape(
        conf['SearchSizeX'], conf['SearchSizeY'], conf['SearchSizeZ'],
        data.shape[1])

    # Display results
    max_index = show_result(result, conf, sample_rate)
    # Maximum value-time plot
    max_eachtime = [np.max(result[:, :, :, i])
                    for i in range(data.shape[1])]
    plt.plot(max_eachtime)
    plt.show()
    # Maximum value 3D profile
    draw_maxbrightness(result[:, :, :, max_index[3],], "SSA", save=False)
    # Save results for plotting
    np.save("./data/result_ssa_event0002.npy", result)
    print(f'result shape {result.shape} saved to ./data/result_ssa_event0002.npy')
