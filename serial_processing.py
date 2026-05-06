"""
微震事件串行处理

基于 serial_processing.ipynb，添加性能统计
"""

import numpy as np
import torch
import os
import datetime
import time
from scipy.signal import find_peaks, butter, filtfilt
from data import readsegy, whightening, stalta, generate_tt, gen_fm_grid
from config import read_config_file, read_station_file
from stackCU import stack_CUDA, show_result as show_result_ssa, calc_position as calc_position_ssa
from stackMechCU import stack_mech_CUDA, gen_intensity_CUDA, show_result as show_result_jssa, calc_position as calc_position_jssa
from model_bfnet import StrikeDipRakeNet
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 配置路径
# ============================================================================
folder_data = './waveform'
folder_conf = './conf'
out_path = './result'
model_path = './model/bfnet_251104a.pt'
sta_fname = f'{folder_conf}/station_sorted.txt'


# ============================================================================
# 预处理函数
# ============================================================================

def preprocess_ssa(data_raw, sample_rate, w_l=25, w_h=60):
    """SSA 预处理：去均值、白化、STALTA"""
    data = np.copy(data_raw)
    for i in range(len(data)):
        data[i] = data[i] - np.mean(data[i])
        data[i] = whightening(data[i], w_l - 20, w_l, w_h, w_h + 20, sample_rate)
        data[i] = stalta(data[i], 5, 40)
    return np.asarray(data)


def seis_filter(data, order, lowcut, highcut, fs, filter_type='bandpass'):
    """带通/带阻滤波器"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if filter_type == 'bandpass':
        b, a = butter(order, [low, high], btype='bandpass')
    elif filter_type == 'bandstop':
        b, a = butter(order, [low, high], btype='bandstop')
    else:
        raise ValueError("filter_type must be 'bandpass' or 'bandstop'")
    return filtfilt(b, a, data)


def preprocess_jssa(data_raw, sample_rate, w_l=25, w_h=60):
    """jSSA 预处理：去均值、带通滤波"""
    data = np.copy(data_raw)
    for i in range(len(data)):
        data[i] = data[i] - np.mean(data[i])
        data[i] = seis_filter(data[i], 3, 22, 60, sample_rate, filter_type='bandpass')
        data[i] = seis_filter(data[i], 3, 30, 35, sample_rate, filter_type='bandstop')
    return np.asarray(data)


def write_conf_file(conf, filename):
    """写入配置文件"""
    with open(filename, 'w') as f:
        for key, value in conf.items():
            if isinstance(value, list):
                value_str = ' '.join(map(str, value))
            else:
                value_str = str(value)
            f.write(f"{key}={value_str}\n")


def find_peak_idx(signal, threshold, prominence=0, peak_pct=0.99):
    """查找峰值及其左侧阈值交叉点"""
    peaks, props = find_peaks(signal, height=threshold, prominence=prominence)
    peak_heights = props['peak_heights']
    left_bases = np.round(props['left_bases']).astype(int)

    peaks_info = []
    for idx, peak_idx in enumerate(peaks):
        peak_height = peak_heights[idx]
        ninety_pct_height = peak_height * peak_pct
        left_base = left_bases[idx]
        raise_idx = None
        for i in range(left_base, peak_idx + 1):
            if signal[i] >= ninety_pct_height:
                raise_idx = i
                break
        else:
            raise_idx = left_base

        peaks_info.append({
            "peak_idx": peak_idx,
            "peak_height": peak_height,
            "raise_idx": raise_idx,
        })

    return peaks_info


def main():
    print("=" * 60)
    print("MICROSEISMIC EVENT PROCESSING (Serial)")
    print("=" * 60)

    # ===== 系统信息 =====
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ===== 读取文件列表 =====
    file_list = [f for f in os.listdir(folder_data) if f.endswith(".sgy")]
    print(f"Found {len(file_list)} waveform files")

    # ===== 读取配置文件 =====
    conf_ssa = read_config_file(f'{folder_conf}/conf_ssa.txt')
    conf_jssa = read_config_file(f'{folder_conf}/conf_jssa.txt')

    # ===== 读取台站信息 =====
    station = read_station_file(sta_fname, conf_ssa)

    # ===== 生成 FM 网格 =====
    fm_grid = gen_fm_grid(conf_jssa)
    print(f"FM grid shape: {fm_grid.shape}")

    # ===== 生成 SSA 走时表 =====
    tt_ssa, _, _ = generate_tt(
        f'{folder_conf}/conf_ssa.txt',
        f'{folder_conf}/vel.txt',
        sta_fname
    )
    tt_max = np.max(tt_ssa)
    print(f"Max travel time: {tt_max:.2f}s")

    peak_threshold = 2.5

    print(f"\nSSA config: SearchSize=({conf_ssa['SearchSizeX']}, {conf_ssa['SearchSizeY']}, {conf_ssa['SearchSizeZ']})")
    print(f"jSSA config: SearchSize=({conf_jssa['SearchSizeX']}, {conf_jssa['SearchSizeY']}, {conf_jssa['SearchSizeZ']})")

    # ===== 加载模型 =====
    shape = (8, 8, 8, 24, 7, 24)
    model = StrikeDipRakeNet(shape)
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model = model.to('cuda')
    model.eval()
    print("Model loaded to GPU")

    # ===== 预热 GPU =====
    dummy = torch.zeros(1, 8, 8, 8, 24, 7, 24).cuda()
    _ = model(dummy)
    del dummy
    torch.cuda.synchronize()

    # ===== 初始化结果列表 =====
    results_ssa = []
    results_jssa = []
    results_bfnet = []

    # ===== CUDA 错误计数 =====
    cuda_errors = {'ssa': 0, 'jssa': 0}

    print("\n" + "=" * 60)
    print("Starting Serial Processing")
    print("=" * 60)

    # ===== 性能统计变量 =====
    cpu_preprocess_time = 0  # CPU 预处理总时间
    gpu_idle_time = 0        # GPU 空闲总时间
    gpu_busy_time = 0        # GPU 忙碌总时间

    start_time = time.time()

    # ===== 遍历每个文件 =====
    for file_idx, file_name in enumerate(file_list):
        file_path = os.path.join(folder_data, file_name)
        file_start = time.time()

        try:
            # ===== CPU 预处理阶段 =====
            preprocess_start = time.time()

            # 读取数据
            data, sta_ids, sample_rate, datetime_start = readsegy(file_path)
            n_time = data.shape[1]
            n_sta = data.shape[0]

            # SSA 预处理（CPU）
            data_ssa = preprocess_ssa(data, sample_rate)

            # jSSA 预处理（CPU）
            data_jssa = preprocess_jssa(data, sample_rate)

            cpu_preprocess_time += time.time() - preprocess_start

            # ===== GPU 处理阶段 =====
            gpu_start = time.time()

            # SSA 堆叠
            result_ssa = stack_CUDA(data_ssa, sample_rate, tt_ssa)

            if isinstance(result_ssa, int) and result_ssa == -1:
                print(f"[{file_idx+1}/{len(file_list)}] {file_name}: SSA stack FAILED")
                cuda_errors['ssa'] += 1
                continue

            result_ssa = result_ssa.reshape(
                conf_ssa['SearchSizeX'],
                conf_ssa['SearchSizeY'],
                conf_ssa['SearchSizeZ'],
                data.shape[1]
            )

            # 计算时间轴最大值
            max_value_per_time = np.max(result_ssa, axis=(0, 1, 2))

            # 查找峰值
            peaks_info = find_peak_idx(max_value_per_time, peak_threshold, 0.5, 1)

            if len(peaks_info) == 0:
                file_time = time.time() - file_start
                print(f"[{file_idx+1}/{len(file_list)}] {file_name}: No peaks found | Time: {file_time:.2f}s")
                continue

            tt_max_sample = int(tt_max * sample_rate)

            # ===== 处理每个峰值 =====
            valid_ssa_count = 0
            valid_jssa_count = 0
            valid_bfnet_count = 0

            for peak in peaks_info:
                peak_idx = peak['raise_idx']

                # ------------------------------------------------------------
                # SSA 定位窗口边界保护
                # 原窗口: peak_idx-10 : peak_idx+1
                # ------------------------------------------------------------
                ssa_start = max(0, peak_idx - 10)
                ssa_end = min(result_ssa.shape[3], peak_idx + 1)

                if ssa_end <= ssa_start:
                    print(f"  [Warning] Invalid SSA window: peak_idx={peak_idx}, start={ssa_start}, end={ssa_end}")
                    continue

                # === SSA 定位 ===
                max_index = show_result_ssa(
                    result_ssa[:, :, :, ssa_start:ssa_end],
                    conf_ssa,
                    sample_rate,
                    False
                )

                x, y, z, t = calc_position_ssa(conf_ssa, max_index, sample_rate)
                t = datetime_start + datetime.timedelta(seconds=t + ssa_start / sample_rate)

                results_ssa.append([x, y, z, t.timestamp()])
                valid_ssa_count += 1

                # === 更新 jSSA 配置 ===
                conf_jssa['SearchOriginX'] = x - conf_jssa['SearchSizeX'] * conf_jssa['GridSpacingX'] / 2
                conf_jssa['SearchOriginY'] = y - conf_jssa['SearchSizeY'] * conf_jssa['GridSpacingX'] / 2
                conf_jssa['SearchOriginZ'] = z - conf_jssa['SearchSizeZ'] * conf_jssa['GridSpacingZ'] / 2
                write_conf_file(conf_jssa, f'{folder_conf}/conf_jssa_tmp.txt')

                # === 生成 jSSA 走时表和强度 ===
                tt_jssa, _, _ = generate_tt(
                    f'{folder_conf}/conf_jssa_tmp.txt',
                    f'{folder_conf}/vel_jssa.txt',
                    sta_fname
                )
                intensity = gen_intensity_CUDA(fm_grid, conf_jssa, station, 0.2)

                # ------------------------------------------------------------
                # jSSA 事件窗边界保护
                # 原窗口: peak_idx-50 : peak_idx+tt_max_sample+30
                # ------------------------------------------------------------
                jssa_start = max(0, peak_idx - 50)
                jssa_end = min(data_jssa.shape[1], peak_idx + tt_max_sample + 30)

                if jssa_end <= jssa_start:
                    print(f"  [Warning] Invalid jSSA window: peak_idx={peak_idx}, start={jssa_start}, end={jssa_end}")
                    continue

                data_jssa_evt = data_jssa[:, jssa_start:jssa_end]

                if data_jssa_evt.shape[1] <= 0:
                    print(f"  [Warning] Empty jSSA window: peak_idx={peak_idx}, start={jssa_start}, end={jssa_end}")
                    continue

                # === jSSA 堆叠 ===
                result_jssa = stack_mech_CUDA(data_jssa_evt, sample_rate, tt_jssa, intensity)

                if isinstance(result_jssa, int) and result_jssa == -1:
                    cuda_errors['jssa'] += 1
                    continue

                result_jssa = result_jssa.reshape(
                    conf_jssa['SearchSizeX'],
                    conf_jssa['SearchSizeY'],
                    conf_jssa['SearchSizeZ'],
                    fm_grid.shape[0],
                    data_jssa_evt.shape[1]
                )

                max_index = show_result_jssa(
                    result_jssa,
                    conf_jssa,
                    sample_rate,
                    fm_grid,
                    False
                )

                # === jSSA 结果 ===
                x, y, z, fm_idx, t = calc_position_jssa(conf_jssa, max_index, sample_rate)
                t = datetime_start + datetime.timedelta(seconds=t + jssa_start / sample_rate)
                fm = fm_grid[fm_idx]
                results_jssa.append([x, y, z, *fm, t.timestamp()])
                valid_jssa_count += 1

                # === BFNet 处理 ===
                x_input = torch.from_numpy(result_jssa[:, :, :, :, max_index[4]]).float()
                min_val = x_input.min()
                max_val = x_input.max()
                x_input = (x_input - min_val) / (max_val - min_val + 1e-8)
                x_input = x_input.to('cuda').reshape(1, 8, 8, 8, 24, 7, 24)

                output = model(x_input)
                output = output.cpu().detach().numpy()

                # === 提取角度 ===
                strike = np.rad2deg(np.arctan2(output[0, 0], output[0, 1]))
                dip = np.rad2deg(np.remainder(np.arctan2(output[0, 2], output[0, 3]), np.pi))
                rake = np.rad2deg(np.arctan2(output[0, 4], output[0, 5]))

                from obspy.imaging.beachball import aux_plane
                s2, d2, r2 = aux_plane(strike, dip, rake)

                results_bfnet.append([strike, dip, rake, s2, d2, r2, t.timestamp()])
                valid_bfnet_count += 1

            gpu_busy_time += time.time() - gpu_start

            # ===== 文件处理完成统计 =====
            file_time = time.time() - file_start
            elapsed = time.time() - start_time
            rate = (file_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(file_list) - file_idx - 1) / rate if rate > 0 else 0

            print(f"[{file_idx+1}/{len(file_list)}] {file_name}: "
                  f"SSA={valid_ssa_count}, jSSA={valid_jssa_count}, BFNet={valid_bfnet_count} | "
                  f"Time: {file_time:.2f}s | ETA: {eta:.0f}s, Rate: {rate:.2f} files/s")

        except Exception as e:
            print(f"[{file_idx+1}/{len(file_list)}] {file_name}: ERROR - {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ===== 串行模式下 GPU 空闲时间计算 =====
    # 串行模式下，GPU 和 CPU 是交替执行的，所以 GPU 空闲时间 = CPU 预处理时间
    gpu_idle_time = cpu_preprocess_time
    total_time = time.time() - start_time

    # ===== 性能统计 =====
    print("\n" + "=" * 60)
    print("Performance Statistics (Serial)")
    print("=" * 60)

    if total_time > 0:
        gpu_util = gpu_busy_time / total_time * 100
        cpu_util = cpu_preprocess_time / total_time * 100

        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Time per file: {total_time/len(file_list):.2f}s")
        print(f"CPU preprocess time: {cpu_preprocess_time:.1f}s ({cpu_util:.1f}%)")
        print(f"GPU busy time: {gpu_busy_time:.1f}s ({gpu_util:.1f}% utilization)")
        print(f"GPU idle time: {gpu_idle_time:.1f}s (waiting for CPU, sequential processing)")
        print(f"CPU-GPU overlap efficiency: 0% (serial, no overlap)")

    # ===== 保存结果 =====
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    os.makedirs(out_path, exist_ok=True)

    # 保存 SSA 结果
    ssa_results_file = f'{out_path}/results_ssa.txt'
    with open(ssa_results_file, 'w') as f_ssa:
        f_ssa.write('X Y Z T\n')
        for res in results_ssa:
            f_ssa.write(' '.join([f'{v:.3f}' for v in res]) + '\n')
    print(f"SSA results saved: {len(results_ssa)} events -> {ssa_results_file}")

    # 保存 JSSA 结果
    jssa_results_file = f'{out_path}/results_jssa.txt'
    with open(jssa_results_file, 'w') as f_jssa:
        f_jssa.write('X Y Z Strike Dip Rake T\n')
        for res in results_jssa:
            f_jssa.write(' '.join([f'{v:.3f}' for v in res]) + '\n')
    print(f"JSSA results saved: {len(results_jssa)} events -> {jssa_results_file}")

    # 保存 BFNet 结果
    bfnet_results_file = f'{out_path}/results_bfnet.txt'
    with open(bfnet_results_file, 'w') as f_bfnet:
        f_bfnet.write('Strike Dip Rake Strike2 Dip2 Rake2 T\n')
        for res in results_bfnet:
            f_bfnet.write(' '.join([f'{v:.3f}' for v in res]) + '\n')
    print(f"BFNet results saved: {len(results_bfnet)} events -> {bfnet_results_file}")

    # ===== 最终统计 =====
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Total files processed: {len(file_list)}")
    print(f"Total SSA events: {len(results_ssa)}")
    print(f"Total JSSA events: {len(results_jssa)}")
    print(f"Total BFNet events: {len(results_bfnet)}")
    print(f"SSA CUDA errors: {cuda_errors['ssa']}")
    print(f"JSSA CUDA errors: {cuda_errors['jssa']}")


if __name__ == '__main__':
    main()