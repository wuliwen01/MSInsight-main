"""
微震事件并行处理 - 高性能版 (自适应缓冲流水线)

优化策略（针对单 GPU）：
1. 多进程 CPU 预处理（真正并行）
2. GPU 任务批量调度（减少锁开销）
3. 自适应预取深度（根据运行时状态动态调整缓冲目标深度）

架构：
- 主进程：GPU 任务消费、结果收集
- CPU 进程池：并行预处理
- 预取线程：异步将 CPU 结果送入缓冲队列
- 固定容量队列 + 自适应目标深度：逻辑上动态调整预取深度，不重建队列

自适应原理：
┌────────────────────────────────────────────────────────────────────┐
│                 Adaptive Prefetch Depth Pipeline                   │
│                                                                    │
│ CPU Pool ──▶ [ready_queue(maxsize=MAX)] ──▶ GPU Worker            │
│                 ▲                    │                              │
│                 │                    ▼                              │
│         target_buffer_level      Results                           │
│      (2 ~ max_buffer_size)                                          │
│                                                                    │
│ 调整规则：                                                          │
│ - GPU 经常等待且队列偏空 -> 增大 target_buffer_level                │
│ - GPU 几乎不等待且队列长期偏满 -> 减小 target_buffer_level          │
│ - 只调整预取深度，不热替换 Queue，避免并发风险                     │
└────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import torch
import os
import datetime
import time
from multiprocessing import Pool, Manager, cpu_count
from queue import Queue, Empty
from threading import Thread, Lock as ThreadLock
import queue
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks, butter, filtfilt
from data import readsegy, whightening, stalta, generate_tt, gen_fm_grid
from config import read_config_file, read_station_file
from stackCU import stack_CUDA, show_result as show_result_ssa, calc_position as calc_position_ssa
from stackMechCU import stack_mech_CUDA, gen_intensity_CUDA, show_result as show_result_jssa, calc_position as calc_position_jssa
from model_bfnet import StrikeDipRakeNet

# ============================================================================
# 配置路径
# ============================================================================
folder_data = './waveform'
folder_conf = './conf'
out_path = './result'
model_path = './model/bfnet_251104a.pt'
sta_fname = f'{folder_conf}/station_sorted.txt'


# ============================================================================
# 预处理函数（CPU 密集型，进程并行）
# ============================================================================

def preprocess_ssa(data_raw, sample_rate, w_l=25, w_h=60):
    """SSA 预处理"""
    data = np.copy(data_raw)
    for i in range(len(data)):
        data[i] = data[i] - np.mean(data[i])
        data[i] = whightening(data[i], w_l-20, w_l, w_h, w_h+20, sample_rate)
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
    return filtfilt(b, a, data)

def preprocess_jssa(data_raw, sample_rate):
    """jSSA 预处理"""
    data = np.copy(data_raw)
    for i in range(len(data)):
        data[i] = data[i] - np.mean(data[i])
        data[i] = seis_filter(data[i], 3, 22, 60, sample_rate, filter_type='bandpass')
        data[i] = seis_filter(data[i], 3, 30, 35, sample_rate, filter_type='bandstop')
    return np.asarray(data)

def find_peak_idx(signal, threshold, prominence=0, peak_pct=0.99):
    """查找峰值"""
    peaks, props = find_peaks(signal, height=threshold, prominence=prominence)
    peak_heights = props['peak_heights']
    left_bases = np.round(props['left_bases']).astype(int)
    
    peaks_info = []
    for idx, peak_idx in enumerate(peaks):
        peak_height = peak_heights[idx]
        ninety_pct_height = peak_height * peak_pct
        left_base = left_bases[idx]
        raise_idx = left_base
        for i in range(left_base, peak_idx + 1):
            if signal[i] >= ninety_pct_height:
                raise_idx = i
                break
        peaks_info.append({
            "peak_idx": peak_idx,
            "peak_height": peak_height,
            "raise_idx": raise_idx,
        })
    return peaks_info


# ============================================================================
# CPU 预处理任务（进程池执行）
# ============================================================================

def cpu_preprocess_task(args):
    """
    CPU 预处理任务：读取文件 + SSA 预处理 + jSSA 预处理
    在进程池中并行执行
    """
    file_name, peak_threshold = args
    
    try:
        file_path = os.path.join(folder_data, file_name)
        
        # 读取数据
        data, sta_ids, sample_rate, datetime_start = readsegy(file_path)
        n_time = data.shape[1]
        
        # SSA 预处理（CPU 并行）
        data_ssa = preprocess_ssa(data, sample_rate)
        
        # jSSA 预处理（CPU 并行）
        data_jssa = preprocess_jssa(data, sample_rate)
        
        # 返回预处理结果（内存共享避免大拷贝）
        return {
            'file_name': file_name,
            'data_ssa': data_ssa,
            'data_jssa': data_jssa,
            'n_time': n_time,
            'sample_rate': sample_rate,
            'datetime_start': datetime_start,
            'peak_threshold': peak_threshold,
        }
        
    except Exception as e:
        print(f"[CPU Worker] Error processing {file_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# GPU 工作器（串行执行，零锁竞争）
# ============================================================================

class GPUWorker:
    """
    GPU 工作器：单线程串行处理所有 GPU 任务
    完全避免锁竞争
    """
    def __init__(self, model_path, conf_ssa, conf_jssa, station, fm_grid, 
                 tt_ssa, tt_max, sta_fname):
        self.model_path = model_path
        self.conf_ssa = conf_ssa
        self.conf_jssa_base = conf_jssa.copy()
        self.station = station
        self.fm_grid = fm_grid
        self.tt_ssa = tt_ssa
        self.tt_max = tt_max
        self.sta_fname = sta_fname
        
        # 设置 CUDA 设备
        torch.cuda.set_device(0)
        
        # 加载 BFNet 模型（只加载一次）
        shape = (8, 8, 8, 24, 7, 24)
        self.model = StrikeDipRakeNet(shape)
        self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.model = self.model.to('cuda')
        self.model.eval()
        
        # 预热 GPU
        dummy = torch.zeros(1, 8, 8, 8, 24, 7, 24).cuda()
        _ = self.model(dummy)
        del dummy
        torch.cuda.synchronize()
        
    def write_conf_file(self, conf, filename):
        """写入配置文件"""
        with open(filename, 'w') as f:
            for key, value in conf.items():
                if isinstance(value, list):
                    value_str = ' '.join(map(str, value))
                else:
                    value_str = str(value)
                f.write(f"{key}={value_str}\n")
    
    def process_preprocessed_data(self, task):
        """
        处理预处理后的数据（GPU 操作）
        """
        file_name = task['file_name']
        data_ssa = task['data_ssa']
        data_jssa = task['data_jssa']
        n_time = task['n_time']
        sample_rate = task['sample_rate']
        datetime_start = task['datetime_start']
        peak_threshold = task['peak_threshold']

        results_ssa = []
        results_jssa = []
        results_bfnet = []
        error_msg = None

        try:
            # ===== SSA 堆叠（GPU）=====
            result_ssa = stack_CUDA(data_ssa, sample_rate, self.tt_ssa)

            if isinstance(result_ssa, int) and result_ssa == -1:
                return file_name, [], [], [], f"SSA stack failed for {file_name}"

            result_ssa = result_ssa.reshape(
                self.conf_ssa['SearchSizeX'],
                self.conf_ssa['SearchSizeY'],
                self.conf_ssa['SearchSizeZ'],
                data_ssa.shape[1]
            )

            # ===== 计算时间轴最大值 =====
            max_value_per_time = np.max(result_ssa, axis=(0, 1, 2))

            # ===== 查找峰值 =====
            peaks_info = find_peak_idx(max_value_per_time, peak_threshold, 0.5, 1)

            if len(peaks_info) == 0:
                return file_name, [], [], [], None

            tt_max_sample = int(self.tt_max * sample_rate)

            # ===== 处理每个峰值 =====
            for peak in peaks_info:
                peak_idx = peak['raise_idx']

                # ------------------------------------------------------------
                # SSA 定位窗口边界保护
                # 原窗口: peak_idx-10 : peak_idx+1
                # ------------------------------------------------------------
                ssa_start = max(0, peak_idx - 10)
                ssa_end = min(result_ssa.shape[3], peak_idx + 1)

                if ssa_end <= ssa_start:
                    print(f"  [Warning] Invalid SSA window in {file_name}: "
                        f"peak_idx={peak_idx}, start={ssa_start}, end={ssa_end}")
                    continue

                # === SSA 定位 ===
                max_index = show_result_ssa(
                    result_ssa[:, :, :, ssa_start:ssa_end],
                    self.conf_ssa,
                    sample_rate,
                    False
                )

                x, y, z, t = calc_position_ssa(self.conf_ssa, max_index, sample_rate)
                t = datetime_start + datetime.timedelta(seconds=t + ssa_start / sample_rate)

                results_ssa.append([x, y, z, t.timestamp()])

                # === jSSA 配置 ===
                conf_jssa_tmp = self.conf_jssa_base.copy()
                conf_jssa_tmp['SearchOriginX'] = x - conf_jssa_tmp['SearchSizeX'] * conf_jssa_tmp['GridSpacingX'] / 2
                conf_jssa_tmp['SearchOriginY'] = y - conf_jssa_tmp['SearchSizeY'] * conf_jssa_tmp['GridSpacingX'] / 2
                conf_jssa_tmp['SearchOriginZ'] = z - conf_jssa_tmp['SearchSizeZ'] * conf_jssa_tmp['GridSpacingZ'] / 2

                self.write_conf_file(conf_jssa_tmp, f'{folder_conf}/conf_jssa_tmp.txt')

                tt_jssa, _, _ = generate_tt(
                    f'{folder_conf}/conf_jssa_tmp.txt',
                    f'{folder_conf}/vel_jssa.txt',
                    self.sta_fname
                )
                intensity = gen_intensity_CUDA(self.fm_grid, conf_jssa_tmp, self.station, 0.2)

                # ------------------------------------------------------------
                # jSSA 事件窗边界保护
                # 原窗口: peak_idx-50 : peak_idx+tt_max_sample+30
                # ------------------------------------------------------------
                jssa_start = max(0, peak_idx - 50)
                jssa_end = min(data_jssa.shape[1], peak_idx + tt_max_sample + 30)

                if jssa_end <= jssa_start:
                    print(f"  [Warning] Invalid jSSA window in {file_name}: "
                        f"peak_idx={peak_idx}, start={jssa_start}, end={jssa_end}")
                    continue

                data_jssa_evt = data_jssa[:, jssa_start:jssa_end]

                if data_jssa_evt.shape[1] <= 0:
                    print(f"  [Warning] Empty jSSA window in {file_name}: "
                        f"peak_idx={peak_idx}, start={jssa_start}, end={jssa_end}")
                    continue

                # === jSSA 堆叠 ===
                result_jssa = stack_mech_CUDA(data_jssa_evt, sample_rate, tt_jssa, intensity)

                if isinstance(result_jssa, int) and result_jssa == -1:
                    continue

                result_jssa = result_jssa.reshape(
                    conf_jssa_tmp['SearchSizeX'],
                    conf_jssa_tmp['SearchSizeY'],
                    conf_jssa_tmp['SearchSizeZ'],
                    self.fm_grid.shape[0],
                    data_jssa_evt.shape[1]
                )

                max_index = show_result_jssa(
                    result_jssa,
                    conf_jssa_tmp,
                    sample_rate,
                    self.fm_grid,
                    False
                )

                # === jSSA 结果 ===
                x, y, z, fm_idx, t = calc_position_jssa(conf_jssa_tmp, max_index, sample_rate)
                t = datetime_start + datetime.timedelta(seconds=t + jssa_start / sample_rate)
                fm = self.fm_grid[fm_idx]
                results_jssa.append([x, y, z, *fm, t.timestamp()])

                # === BFNet 处理 ===
                x_input = torch.from_numpy(result_jssa[:, :, :, :, max_index[4]]).float()
                min_val = x_input.min()
                max_val = x_input.max()
                x_input = (x_input - min_val) / (max_val - min_val + 1e-8)
                x_input = x_input.to('cuda').reshape(1, 8, 8, 8, 24, 7, 24)

                output = self.model(x_input)
                output = output.cpu().detach().numpy()

                # === 提取角度 ===
                strike = np.rad2deg(np.arctan2(output[0, 0], output[0, 1]))
                dip = np.rad2deg(np.remainder(np.arctan2(output[0, 2], output[0, 3]), np.pi))
                rake = np.rad2deg(np.arctan2(output[0, 4], output[0, 5]))

                from obspy.imaging.beachball import aux_plane
                s2, d2, r2 = aux_plane(strike, dip, rake)

                results_bfnet.append([strike, dip, rake, s2, d2, r2, t.timestamp()])

        except Exception as e:
            error_msg = f"Error processing {file_name}: {str(e)}"
            import traceback
            traceback.print_exc()

        return file_name, results_ssa, results_jssa, results_bfnet, error_msg


# ============================================================================
# 双缓冲预取系统
# ============================================================================
import threading
from collections import deque

class AdaptiveBufferPipeline:
    """
    自适应缓冲流水线：固定物理队列容量，动态调整逻辑预取深度

    设计原则：
    - 不动态重建 ready_queue，避免并发风险
    - 只调整 target_buffer_level（逻辑目标缓冲深度）
    - 通过窗口统计 GPU 等待、预取受限、队列占用率来调整深度
    """

    def __init__(self, max_buffer_size=8, initial_target=2, min_target=2, window_size=16):
        self.max_buffer_size = max_buffer_size
        self.min_target = min_target
        self.target_buffer_level = max(min(initial_target, max_buffer_size), min_target)
        self.window_size = max(4, window_size)

        # 固定物理容量队列
        self.ready_queue = queue.Queue(maxsize=max_buffer_size)
        self.result_queue = queue.Queue()

        # 控制信号
        self.all_tasks_submitted = False
        self.active_gpu_tasks = 0

        # 线程安全锁
        self.lock = threading.Lock()

        # 总体统计
        self.submitted_count = 0
        self.completed_count = 0
        self.prefetch_block_time_total = 0.0
        self.prefetch_block_events_total = 0
        self.gpu_wait_time_total = 0.0
        self.gpu_wait_events_total = 0
        self.queue_samples_total = 0
        self.queue_fill_sum_total = 0.0

        # 窗口统计
        self.window_completed = 0
        self.window_prefetch_block_time = 0.0
        self.window_prefetch_block_events = 0
        self.window_gpu_wait_time = 0.0
        self.window_gpu_wait_events = 0
        self.window_queue_samples = 0
        self.window_queue_fill_sum = 0.0
        self.window_start_time = time.time()
        self.last_window_throughput = None

        # 控制项
        self.warmup_files = 8
        self.cooldown_windows = 0
        self.adjust_history = []

    def _sample_queue_fill_locked(self):
        denom = max(self.target_buffer_level, 1)
        fill = min(self.ready_queue.qsize() / denom, 1.0)
        self.queue_samples_total += 1
        self.queue_fill_sum_total += fill
        self.window_queue_samples += 1
        self.window_queue_fill_sum += fill

    def submit_preprocessed(self, task):
        """
        CPU 提交预处理完成的任务到缓冲队列。
        当达到 target_buffer_level 时，逻辑上限流，而不是重建 Queue。
        """
        blocked_time = 0.0
        blocked = False

        while True:
            with self.lock:
                qsize = self.ready_queue.qsize()
                target = self.target_buffer_level

            if qsize < target:
                break

            blocked = True
            t0 = time.time()
            time.sleep(0.01)
            blocked_time += time.time() - t0

        # 物理队列仍然保留硬上限保护
        self.ready_queue.put(task)

        with self.lock:
            self.submitted_count += 1
            if blocked:
                self.prefetch_block_time_total += blocked_time
                self.prefetch_block_events_total += 1
                self.window_prefetch_block_time += blocked_time
                self.window_prefetch_block_events += 1
            self._sample_queue_fill_locked()

    def mark_all_submitted(self):
        self.all_tasks_submitted = True

    def get_next_task(self, timeout=0.1):
        """GPU 获取下一个任务，并记录真实等待时间"""
        t0 = time.time()
        try:
            task = self.ready_queue.get(timeout=timeout)
            waited = time.time() - t0
            with self.lock:
                if waited > 1e-6:
                    self.gpu_wait_time_total += waited
                    self.window_gpu_wait_time += waited
                self.active_gpu_tasks += 1
                self._sample_queue_fill_locked()
            return task, False, waited
        except queue.Empty:
            waited = time.time() - t0
            with self.lock:
                self.gpu_wait_time_total += waited
                self.window_gpu_wait_time += waited
                self.gpu_wait_events_total += 1
                self.window_gpu_wait_events += 1
                self._sample_queue_fill_locked()
            if self.all_tasks_submitted:
                return None, True, waited
            return None, False, waited

    def _reset_window_locked(self):
        self.window_completed = 0
        self.window_prefetch_block_time = 0.0
        self.window_prefetch_block_events = 0
        self.window_gpu_wait_time = 0.0
        self.window_gpu_wait_events = 0
        self.window_queue_samples = 0
        self.window_queue_fill_sum = 0.0
        self.window_start_time = time.time()

    def maybe_adjust_target(self):
        """按窗口统计自适应调整 target_buffer_level"""
        with self.lock:
            if self.completed_count < self.warmup_files:
                return None
            if self.cooldown_windows > 0:
                return None
            if self.window_completed < self.window_size:
                return None

            elapsed = max(time.time() - self.window_start_time, 1e-6)
            throughput = self.window_completed / elapsed
            avg_fill = (self.window_queue_fill_sum / self.window_queue_samples) if self.window_queue_samples > 0 else 0.0
            gpu_wait_ratio = self.window_gpu_wait_time / elapsed
            prefetch_block_ratio = self.window_prefetch_block_time / elapsed

            old_target = self.target_buffer_level
            new_target = old_target
            reason = None

            # 扩容：GPU 等待较明显且队列偏空
            if gpu_wait_ratio > 0.10 and avg_fill < 0.40 and old_target < self.max_buffer_size:
                new_target = old_target + 1
                reason = (f"increase: gpu_wait_ratio={gpu_wait_ratio:.1%}, "
                          f"avg_fill={avg_fill:.1%}, throughput={throughput:.2f} files/s")

            # 缩容：GPU 基本不等待、队列长期偏满、预取经常受限、吞吐提升不明显
            elif (gpu_wait_ratio < 0.02 and avg_fill > 0.80 and prefetch_block_ratio > 0.10
                  and old_target > self.min_target):
                no_gain = (self.last_window_throughput is not None and
                           throughput <= self.last_window_throughput * 1.02)
                if no_gain:
                    new_target = old_target - 1
                    reason = (f"decrease: gpu_wait_ratio={gpu_wait_ratio:.1%}, "
                              f"avg_fill={avg_fill:.1%}, prefetch_block_ratio={prefetch_block_ratio:.1%}, "
                              f"throughput={throughput:.2f} files/s")

            self.last_window_throughput = throughput
            self._reset_window_locked()

            if new_target != old_target:
                self.target_buffer_level = new_target
                self.cooldown_windows = 1
                self.adjust_history.append((self.completed_count, old_target, new_target, reason))
                return old_target, new_target, reason
            return None

    def on_window_progress(self):
        with self.lock:
            if self.cooldown_windows > 0 and self.window_completed >= self.window_size:
                self.cooldown_windows -= 1

    def mark_task_completed(self, result):
        self.result_queue.put(result)
        with self.lock:
            self.active_gpu_tasks -= 1
            self.completed_count += 1
            self.window_completed += 1

    def get_completed_results(self):
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get_nowait())
        return results

    def is_finished(self):
        return self.all_tasks_submitted and self.ready_queue.empty() and self.active_gpu_tasks == 0


class PrefetchThread(threading.Thread):
    """
    预取线程：从 CPU Pool 异步获取预处理结果
    持续运行直到所有任务提交完毕
    """
    
    def __init__(self, pool, preprocess_iter, pipeline, cpu_workers_active):
        super().__init__(daemon=True)
        self.pool = pool
        self.preprocess_iter = preprocess_iter
        self.pipeline = pipeline
        self.cpu_workers_active = cpu_workers_active
        self.exceptions = []
    
    def run(self):
        try:
            for preprocessed in self.preprocess_iter:
                if preprocessed is None:
                    continue
                
                # 自适应逻辑预取：达到 target_buffer_level 时短暂等待
                self.pipeline.submit_preprocessed(preprocessed)
                
                # 检查 CPU workers 是否还在活跃
                with self.pipeline.lock:
                    pass  # 统计用
            
            # 所有任务已提交
            self.pipeline.mark_all_submitted()
            
        except Exception as e:
            self.exceptions.append(e)
            self.pipeline.mark_all_submitted()

def main():
    print("="*60)
    print("MICROSEISMIC EVENT PROCESSING (Adaptive Buffer Pipeline)")
    print("="*60)
    
    # ===== 系统信息 =====
    n_cpus = cpu_count() or 1
    print(f"CPU cores: {n_cpus}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ===== 读取配置 =====
    conf_ssa = read_config_file(f'{folder_conf}/conf_ssa.txt')
    conf_jssa = read_config_file(f'{folder_conf}/conf_jssa.txt')
    station = read_station_file(sta_fname, conf_ssa)
    fm_grid = gen_fm_grid(conf_jssa)
    tt_ssa, _, _ = generate_tt(f'{folder_conf}/conf_ssa.txt', f'{folder_conf}/vel.txt', sta_fname)
    tt_max = np.max(tt_ssa)
    peak_threshold = 2.5
    
    print(f"\nSSA config: SearchSize=({conf_ssa['SearchSizeX']}, {conf_ssa['SearchSizeY']}, {conf_ssa['SearchSizeZ']})")
    print(f"jSSA config: SearchSize=({conf_jssa['SearchSizeX']}, {conf_jssa['SearchSizeY']}, {conf_jssa['SearchSizeZ']})")
    print(f"FM grid shape: {fm_grid.shape}, Max travel time: {tt_max:.2f}s")
    
    # ===== 获取文件列表 =====
    file_list = [f for f in os.listdir(folder_data) if f.endswith(".sgy")]
    print(f"\nFound {len(file_list)} waveform files")
    
    # ===== 计算工作进程数 =====
    n_cpu_workers = min(n_cpus, len(file_list), 16)
    print(f"CPU workers: {n_cpu_workers}")
    
    # ===== 创建 GPU 工作器 =====
    print("\nInitializing GPU worker...")
    gpu_worker = GPUWorker(
        model_path=model_path,
        conf_ssa=conf_ssa,
        conf_jssa=conf_jssa,
        station=station,
        fm_grid=fm_grid,
        tt_ssa=tt_ssa,
        tt_max=tt_max,
        sta_fname=sta_fname
    )
    
    # ===== 缓冲流水线配置 =====
    MAX_BUFFER_SIZE = min(8, max(2, n_cpu_workers))  # 物理队列上限
    INITIAL_TARGET = 2  # 初始逻辑缓冲深度
    WINDOW_SIZE = 16
    pipeline = AdaptiveBufferPipeline(max_buffer_size=MAX_BUFFER_SIZE,
                                      initial_target=INITIAL_TARGET,
                                      min_target=2,
                                      window_size=WINDOW_SIZE)
    
    print(f"\n{'='*60}")
    print("Starting Adaptive Buffer Pipeline")
    print(f"{'='*60}")
    print(f"Queue capacity: {MAX_BUFFER_SIZE}")
    print(f"Initial target buffer level: {INITIAL_TARGET}")
    print(f"Adaptive window size: {WINDOW_SIZE}")
    print(f"CPU workers: {n_cpu_workers} (parallel preprocessing)")
    print(f"GPU worker: 1 (serial processing)")
    print("="*60)
    
    start_time = time.time()
    
    # ===== 启动 CPU 进程池和预取线程 =====
    with Pool(processes=n_cpu_workers) as pool:
        preprocess_args = [(f, peak_threshold) for f in file_list]
        preprocess_iter = pool.imap_unordered(cpu_preprocess_task, preprocess_args)
        
        # 启动预取线程：异步从 CPU Pool 取结果送入缓冲队列
        prefetch_thread = PrefetchThread(
            pool=pool,
            preprocess_iter=preprocess_iter,
            pipeline=pipeline,
            cpu_workers_active=True
        )
        prefetch_thread.start()
        
        # ===== GPU 主循环：双缓冲消费 =====
        completed = 0
        total_events_ssa = 0
        total_events_jssa = 0
        total_events_bfnet = 0
        cuda_errors = {'ssa': 0, 'jssa': 0}
        all_results_ssa = []
        all_results_jssa = []
        all_results_bfnet = []
        
        gpu_idle_time = 0
        gpu_busy_time = 0
        
        while True:
            task_start = time.time()
            
            # GPU 尝试获取任务（等待最多 0.1 秒）
            task, is_finished, wait_time = pipeline.get_next_task(timeout=0.1)
            
            if is_finished:
                # 所有任务完成，退出循环
                break
            
            if task is None:
                # 缓冲区暂时为空，GPU 空闲
                gpu_idle_time += wait_time
                continue
            
            # GPU 开始处理
            task_fetch_time = time.time() - task_start
            
            process_start = time.time()
            file_name, res_ssa, res_jssa, res_bfnet, error_msg = \
                gpu_worker.process_preprocessed_data(task)
            gpu_busy_time += time.time() - process_start
            
            # 标记任务完成
            pipeline.mark_task_completed({
                'file_name': file_name,
                'res_ssa': res_ssa,
                'res_jssa': res_jssa,
                'res_bfnet': res_bfnet,
                'error_msg': error_msg
            })
            
            completed += 1

            # 窗口推进与自适应调整
            pipeline.on_window_progress()
            adjust_info = pipeline.maybe_adjust_target()
            if adjust_info is not None:
                old_target, new_target, reason = adjust_info
                print(f"[Adaptive Buffer] target_buffer_level: {old_target} -> {new_target} | {reason}")

            # 收集结果
            if error_msg:
                print(f"[{completed}/{len(file_list)}] {file_name}: ERROR - {error_msg.split(':')[0]}")
                if 'SSA' in error_msg:
                    cuda_errors['ssa'] += 1
                else:
                    cuda_errors['jssa'] += 1
            else:
                all_results_ssa.extend(res_ssa)
                all_results_jssa.extend(res_jssa)
                all_results_bfnet.extend(res_bfnet)
                
                total_events_ssa += len(res_ssa)
                total_events_jssa += len(res_jssa)
                total_events_bfnet += len(res_bfnet)
                
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(file_list) - completed) / rate if rate > 0 else 0
                
                print(f"[{completed}/{len(file_list)}] {file_name}: "
                      f"SSA={len(res_ssa)}, jSSA={len(res_jssa)}, BFNet={len(res_bfnet)} | "
                      f"ETA: {eta:.0f}s, Rate: {rate:.2f} files/s")
        
        # 等待预取线程结束
        prefetch_thread.join(timeout=5.0)
    
    total_time = time.time() - start_time
    
    # ===== 性能统计 =====
    print("\n" + "="*60)
    print("Performance Statistics")
    print("="*60)
    if total_time > 0:
        gpu_util = gpu_busy_time / total_time * 100
        cpu_util = 100 - (gpu_idle_time / total_time * 100) if total_time > gpu_idle_time else 0
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Time per file: {total_time/len(file_list):.2f}s")
        print(f"GPU busy time: {gpu_busy_time:.1f}s ({gpu_util:.1f}% utilization)")
        print(f"GPU idle time: {gpu_idle_time:.1f}s (waiting for CPU)")
        print(f"CPU-GPU overlap efficiency: ~{min(gpu_util + cpu_util, 100):.0f}%")
        print(f"Final target buffer level: {pipeline.target_buffer_level}")
        if pipeline.adjust_history:
            print("Adaptive adjustment history:")
            for completed_cnt, old_t, new_t, reason in pipeline.adjust_history:
                print(f"  completed={completed_cnt}: {old_t} -> {new_t} | {reason}")
        else:
            print("Adaptive adjustment history: no changes")
    
    # ===== 保存结果 =====
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    os.makedirs(out_path, exist_ok=True)
    
    ssa_results_file = f'{out_path}/results_ssa_hp.txt'
    with open(ssa_results_file, 'w') as f:
        f.write('X Y Z T\n')
        for res in all_results_ssa:
            f.write(' '.join([f'{v:.3f}' for v in res]) + '\n')
    print(f"SSA results saved: {len(all_results_ssa)} events -> {ssa_results_file}")
    
    jssa_results_file = f'{out_path}/results_jssa_hp.txt'
    with open(jssa_results_file, 'w') as f:
        f.write('X Y Z Strike Dip Rake T\n')
        for res in all_results_jssa:
            f.write(' '.join([f'{v:.3f}' for v in res]) + '\n')
    print(f"JSSA results saved: {len(all_results_jssa)} events -> {jssa_results_file}")
    
    bfnet_results_file = f'{out_path}/results_bfnet_hp.txt'
    with open(bfnet_results_file, 'w') as f:
        f.write('Strike Dip Rake Strike2 Dip2 Rake2 T\n')
        for res in all_results_bfnet:
            f.write(' '.join([f'{v:.3f}' for v in res]) + '\n')
    print(f"BFNet results saved: {len(all_results_bfnet)} events -> {bfnet_results_file}")
    
    # ===== 最终统计 =====
    print("\n" + "="*60)
    print("Processing Complete")
    print("="*60)
    print(f"Total files processed: {len(file_list)}")
    print(f"Total SSA events: {total_events_ssa}")
    print(f"Total JSSA events: {total_events_jssa}")
    print(f"Total BFNet events: {total_events_bfnet}")
    print(f"SSA CUDA errors: {cuda_errors['ssa']}")
    print(f"JSSA CUDA errors: {cuda_errors['jssa']}")


if __name__ == '__main__':
    main()
