import time
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool, shared_memory
import ctypes


def stack_mech_CUDA(data, sample_rate, traveltime, intensity):
    """
    Joint stacking algorithm accelerated with CUDA
    Args:
        data (ndarray): (n_sta*3, n_samples)
        sample_rate (float): sampling rate
        traveltime (ndarray): (n_sta, n_grid)
        intensity (ndarray): (n_fm, n_grid, n_sta)
    Returns:
        ndarray: (n_grid, n_fm, n_samples)
    """
    # Calculate parameters
    n_fm, n_grid, n_sta = intensity.shape
    n_samples = data.shape[1]
    
    # 有效性检查：样本数必须大于 0
    if n_samples <= 0:
        print(f"[Error] Invalid n_samples: {n_samples} (must be > 0)")
        return -1
    
    # Convert traveltime to sample points
    if traveltime is None or np.isnan(traveltime).any():
        print("[Error] Invalid traveltime data: None or NaN detected")
        return -1
    tt_samples = (traveltime * sample_rate).round().astype(np.int32)
    # print("tt_samples range:", tt_samples.min(), tt_samples.max())
    # Extract z component of data
    data = data[2::3, :]
    # Convert data to C-style array
    data = np.ascontiguousarray(data, dtype=np.float32)
    tt_samples = np.ascontiguousarray(tt_samples, dtype=np.int32)
    intensity = np.ascontiguousarray(intensity, dtype=np.float32)
    # Initialize result array
    result = np.zeros((n_grid, n_fm, n_samples), dtype=np.float32)
    # result = np.ascontiguousarray(result, dtype=np.float32)
    if (n_samples-np.max(tt_samples) > 1024):
        print("[Error] n_samples-max(tt_samples)  > 1024")
        return -1
    clib = ctypes.CDLL('./jSSA.dll')
    stackCUDA = getattr(clib, "?stackCUDA@@YAHPEBMPEBH0HHHHPEAM@Z")
    # parameters
    stackCUDA.argtypes = [ctypes.POINTER(ctypes.c_float),  # data
                          ctypes.POINTER(ctypes.c_int32),  # tt_samples
                          ctypes.POINTER(ctypes.c_float),  # intensity
                          ctypes.c_int,                    # n_sta
                          ctypes.c_int,                    # n_samples
                          ctypes.c_int,                    # n_grid
                          ctypes.c_int,                    # n_fm
                          ctypes.POINTER(ctypes.c_float)]  # result
    # return type
    stackCUDA.restype = ctypes.c_int

    # Data pointers
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    tt_samples_ptr = tt_samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    intensity_ptr = intensity.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Call C function
    ret = stackCUDA(data_ptr, tt_samples_ptr, intensity_ptr,
                    n_sta, n_samples, n_grid, n_fm, result_ptr)
    if ret != 0:
        print("CUDA Error! ret=", ret)
        return -1
    # else:
    #     print("CUDA result shape:", result.shape)
    return result


class gridConf(ctypes.Structure):
    _fields_ = [
        ("GridSpacingX", ctypes.c_float),
        ("GridSpacingZ", ctypes.c_float),
        ("SearchSizeX", ctypes.c_int),
        ("SearchSizeY", ctypes.c_int),
        ("SearchSizeZ", ctypes.c_int),
        ("SearchOriginX", ctypes.c_float),
        ("SearchOriginY", ctypes.c_float),
        ("SearchOriginZ", ctypes.c_float)
    ]


def gen_intensity_CUDA(fm_grid, conf, station, a=1.0):
    """
    Calculate radiation intensity
    Args:
        fm_grid (ndarray): focal mechanism grid
        conf (dict): configuration information
        station (dict): detector information
        a (float): distance attenuation factor

    Returns:
        ndarray: radiation intensity, shape=(n_fm, n_grid, n_sta)
    """
    # Calculate parameters
    n_sta = len(station['x'])
    n_fm = fm_grid.shape[0]
    n_grid = conf['SearchSizeX'] * conf['SearchSizeY'] * conf['SearchSizeZ']
    # Convert data to C-style format
    fm_grid_cformat = np.ascontiguousarray(fm_grid, dtype=np.float32)
    conf_cformat = gridConf()
    conf_cformat.GridSpacingX = conf['GridSpacingX']
    conf_cformat.GridSpacingZ = conf['GridSpacingZ']
    conf_cformat.SearchSizeX = conf['SearchSizeX']
    conf_cformat.SearchSizeY = conf['SearchSizeY']
    conf_cformat.SearchSizeZ = conf['SearchSizeZ']
    conf_cformat.SearchOriginX = conf['SearchOriginX']
    conf_cformat.SearchOriginY = conf['SearchOriginY']
    conf_cformat.SearchOriginZ = conf['SearchOriginZ']
    station_cformat = np.zeros((n_sta, 3), dtype=np.float32)
    for i in range(n_sta):
        station_cformat[i, 0] = station['x'][i]/1000
        station_cformat[i, 1] = station['y'][i]/1000
        station_cformat[i, 2] = station['z'][i]/1000
    station_cformat = np.ascontiguousarray(station_cformat, dtype=np.float32)
    # Initialize result array
    intensity = np.zeros((n_fm*n_grid*n_sta), dtype=np.float32)
    intensity = np.ascontiguousarray(intensity, dtype=np.float32)
    # Data pointers
    fm_grid_ptr = fm_grid_cformat.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    station_ptr = station_cformat.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    intensity_ptr = intensity.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))

    clib = ctypes.CDLL('./jSSA.dll')
    intensityCUDA = getattr(
        clib, "?intensityCUDA@@YAHPEBMPEAUGridConfig@@PEAMHHM2@Z")
    # Parameter configuration
    intensityCUDA.argtypes = [ctypes.POINTER(ctypes.c_float),    # fm_grid
                              ctypes.POINTER(gridConf),          # conf
                              ctypes.POINTER(ctypes.c_float),    # stations
                              ctypes.c_int,                      # n_sta
                              ctypes.c_int,                      # n_fm
                              ctypes.c_float,                    # a
                              ctypes.POINTER(ctypes.c_float)     # intensity
                              ]
    # Return value configuration
    intensityCUDA.restype = ctypes.c_int
    # Call C function
    ret = intensityCUDA(fm_grid_ptr, conf_cformat, station_ptr,
                        n_sta, n_fm, a, intensity_ptr)
    if ret != 0:
        print("CUDA Error!")
        return -1

    # Convert result to 3D array
    intensity = intensity.reshape(
        n_fm, n_grid, n_sta)
    # print("CUDA intensity shape:", intensity.shape)
    return intensity


def drawintensity(intensity, station, i_fm, i_grid, size=300):
    """
    Plot radiation intensity
    Args:
        intensity (ndarray): radiation intensity
        i_fm (int): focal mechanism index
        i_grid (int): grid index
        save (bool): whether to save the image
        size (int): point size
    """
    n_sta = intensity.shape[2]
    s = abs(intensity[i_fm, i_grid]) * size
    x = station['x']
    y = station['y']
    c = ['r' if intensity[i_fm, i_grid, i] > 0 else 'b' for i in range(n_sta)]
    print("intensity:", intensity[i_fm, i_grid])
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=s, c=c, alpha=0.7)
    for i in range(n_sta):
        plt.text(x[i], y[i], station['sta_ids'][i], fontsize=8)
    plt.title(f'fm: {i_fm}, grid: {i_grid}')
    plt.axis('equal')
    plt.show()


def preprocess_chunk(index, shared_name, shape, dtype, sample_rate, sta):
    """
    Process a single data chunk for parallel processing
    Args:
        index (int): index of the data chunk
        shared_name (str): name of the shared memory
        shape (tuple): shape of the data
        dtype (np.dtype): data type
        sample_rate (float): sampling rate
        sta (int): STALTA short window length
    """
    # Connect to shared memory
    shm = shared_memory.SharedMemory(name=shared_name)
    data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # Remove mean
    # data[index] -= np.mean(data[index])
    # Spectral whitening
    # data[index] = whightening(data[index], 15, 60, sample_rate)
    # STALTA
    # data[index] = stalta(data[index], sta, sta * 10)
    # Data normalization
    # data[index] /= np.max(np.abs(data[index]))
    shm.close()  # Close shared memory connection


def preprocess_parallel(data_raw, sample_rate, sta=5, num_workers=8):
    """
    Parallel preprocessing of data
    Args:
        data_raw (np.ndarray): raw data
        sample_rate (float): sampling rate
        sta (int): STALTA short window length
        num_workers (int): number of parallel processes
    Returns:
        np.ndarray: processed data
    """
    start = time.time()
    data = np.copy(data_raw)
    shape = data.shape
    dtype = data.dtype

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np.copyto(shared_data, data)

    # Create process pool
    with Pool(processes=num_workers) as pool:
        # Assign tasks to each process
        pool.starmap(preprocess_chunk, [
            (i, shm.name, shape, dtype, sample_rate, sta) for i in range(len(data))
        ])

    # Extract processed data from shared memory
    data = np.copy(shared_data)

    # Clean up shared memory
    shm.close()
    shm.unlink()
    end = time.time()
    print(f"preprocess time:{end - start:.3f}s")

    return data


def calc_position(conf, max_index, sample_rate):
    x = max_index[0] * conf['GridSpacingX'] + conf['SearchOriginX']
    y = max_index[1] * conf['GridSpacingX'] + conf['SearchOriginY']
    z = max_index[2] * conf['GridSpacingZ'] + conf['SearchOriginZ']
    fm = max_index[3]
    t = max_index[4] / sample_rate
    # round to 3 decimal places
    x = round(x, 3)
    y = round(y, 3)
    z = round(z, 3)
    t = round(t, 3)
    return x, y, z, fm, t


def show_result(result, conf, sample_rate, fm_grid, show=True):
    max_index = np.array(np.unravel_index(np.argmax(result), result.shape))
    max_value = np.max(result)
    # Convert maxindex to real coordinates
    x, y, z, fm, t = calc_position(conf, max_index, sample_rate)

    if show:
        print("max index:", max_index)
        print("max value:", max_value)
        print(f"max position: ({x:.3f}, {y:.3f}, {z:.3f}, {t:.3f})")
        print(f"max fm: {fm_grid[fm]}")
    return max_index


def checkData(data):
    """Visualize and check if data is normal
    """
    for i in range(len(data)):
        plt.plot(data[i])
        plt.show()
