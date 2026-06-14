// traveltime table preload into shared memory

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

inline int getLinearIndex2d(int i, int j, int cols)
{
    return i * cols + j;
}

inline int getLinearIndex3d(int i, int j, int k, int cols, int rows)
{
    return i * cols * rows + j * cols + k;
}

/**
 * @brief CUDA kernel to perform stacking. Computes the stacked result across all stations
 *        for a given grid point, mechanism, and sample.
 *
 * @param data Input data array of size n_sta * n_samples, storing per-station data.
 * @param tt_samples Time-shift offsets array for each station (size n_sta * n_grid), storing
 *                   the offset for each station at each grid point.
 * @param result Output array (n_grid * n_samples) to store stacked results.
 * @param n_sta Number of stations.
 * @param n_samples Number of samples per station.
 * @param n_grid Number of grid points.
 * @param max_i_sample Maximum sample index each kernel should process.
 * @param i_sample_offset Sample offset for the current kernel.
 */
__global__ void stackKernel(const float *data, const int *tt_samples,
                            float *result, int n_sta, int n_samples, int n_grid,
                            int max_i_sample, int i_sample_offset)
{
    int i_sample = threadIdx.x;                    // sample index within this kernel
    int sample_index = i_sample + i_sample_offset; // global sample index across the data
    // if (sample_index >= max_i_sample)
    //     return;
    int64_t i_grid = blockIdx.x;

    // Load tt_samples into shared memory
    extern __shared__ int shared_tt_samples[];
    if (threadIdx.x == 0)
    {
        for (int i_sta = 0; i_sta < n_sta; ++i_sta)
        {
            shared_tt_samples[i_sta] = tt_samples[i_sta * n_grid + i_grid];
        }
    }
    __syncthreads(); // ensure all threads have finished loading

    float sum = 0.0f;
    float shifted_data = 0.0f;
    int offset = 0;
    for (int i_sta = 0; i_sta < n_sta; ++i_sta)
    {
        // iterate over stations and compute contributions to stacking
        offset = shared_tt_samples[i_sta];
        shifted_data = data[i_sta * n_samples + (sample_index + offset)];
        sum += shifted_data;
    }
    sum /= n_sta; // average across stations
    result[i_grid * n_samples + sample_index] = sum;
}

/**
 * @brief Function to invoke CUDA kernel for stacking across all grid points, mechanisms,
 *        and samples.
 *
 * @param data Input data array of size n_sta * n_samples, storing per-station data.
 * @param tt_samples Time-shift offsets array for each station (size n_sta * n_grid).
 * @param n_sta Number of stations.
 * @param n_samples Number of samples per station.
 * @param n_grid Number of grid points.
 * @param result Output array (n_grid * n_samples) to store stacked results.
 */
__declspec(dllexport) int stackCUDA(const float *data, const int *tt_samples,
                                    int n_sta, int n_samples, int n_grid,
                                    float *result)
{
    std::cout << "[SSA.cu] running stackCUDA " << std::endl;

    // Allocate device memory
    float *d_data, *d_result;
    int *d_tt_samples;
    std::cout << "[SSA.cu] n_sta: " << n_sta << " n_samples: " << n_samples << " n_grid: " << n_grid << std::endl;

    // print memory usage
    std::cout << "[SSA.cu] use GPU memory: " << n_sta * n_samples * sizeof(float) + n_sta * n_grid * sizeof(int) + n_grid * n_samples * sizeof(float) << std::endl;
    std::cout << "[SSA.cu] data size: " << n_sta * n_samples * sizeof(float) << std::endl;
    std::cout << "[SSA.cu] tt_samples size: " << n_sta * n_grid * sizeof(int) << std::endl;
    std::cout << "[SSA.cu] result size: " << n_grid * n_samples * sizeof(float) << std::endl;

    if (cudaMalloc(&d_data, n_sta * n_samples * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_data" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_tt_samples, n_sta * n_grid * sizeof(int)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_tt_samples" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_result, n_grid * n_samples * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_result" << std::endl;
        return -1;
    }

    // Copy data to device
    cudaMemcpy(d_data, data, n_sta * n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tt_samples, tt_samples, n_sta * n_grid * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, n_grid * n_samples * sizeof(float));

    size_t shared_mem_size = n_sta * sizeof(int);

    // Compute how many samples can be processed per batch.
    const int *p_max_tt = std::max_element(tt_samples, tt_samples + n_sta * n_grid); // find maximum travel time among all tt_samples
    int max_i_sample = n_samples - *p_max_tt; // maximum valid i_sample for computation; indices beyond this may be out of range
    int sample_offset = 0;                    // Offset into i_samples, starting at 0.
    std::cout << "[SSA.cu] max_tt: " << *p_max_tt << " max_i_sample : " << max_i_sample << std::endl;
    int sample_remaining = max_i_sample; // Remaining sample points.
    // Total time across all kernel launches.
    std::chrono::duration<double, std::milli> total_duration(0.0);

    // loop and process all data
    while (sample_remaining > 0)
    {
        // Define grid and block dimensions
        dim3 gridDim(n_grid);
        int n_threads = sample_remaining > 1024 ? 1024 : sample_remaining; // At most 1024 threads per block; ensure n_threads <= 1024
        dim3 blockDim(n_threads);

        std::cout << "[SSA.cu] sample_remaining: " << sample_remaining << " n_threads: " << n_threads << std::endl;

        auto start = std::chrono::high_resolution_clock::now(); // start time

        // Launch kernel
        stackKernel<<<gridDim, blockDim, shared_mem_size>>>(d_data, d_tt_samples, d_result,
                                                            n_sta, n_samples, n_grid, max_i_sample, sample_offset);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();             // end time
        std::chrono::duration<double, std::milli> duration = end - start; // compute elapsed time
        std::cout << "[SSA.cu] kernel execution time: " << duration.count() << " ms" << std::endl;
        total_duration += duration; // accumulate duration

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        sample_remaining -= 1024; // decrement remaining samples by the processed amount (1024 max per iteration)
        sample_offset += 1024;    // increase sample offset by processed amount (1024)
    }
    std::cout << "[SSA.cu] total kernel execution time: " << total_duration.count() << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(result, d_result, n_grid * n_samples * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_tt_samples);
    cudaFree(d_result);
    return 0;
}

void testStack()
{
    std::cout << "CUDA Version: " << CUDART_VERSION << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << ", sizeof(int): " << sizeof(int) << std::endl;
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Number of GPUs: " << count << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per block dimensions: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem << std::endl;
    // Example usage
    int n_sta = 40;        // Number of stations
    int n_samples = 7501; // Number of samples per station
    int n_grid = 48000;    // Number of grid points

    float *data = new float[n_sta * n_samples];
    int *tt_samples = new int[n_sta * n_grid];
    float *result = new float[n_grid * n_samples]{0};

    // Initialize example data
    for (int i = 0; i < n_sta * n_samples; ++i)
        data[i] = 1.0f;

    for (int i = 0; i < n_sta * n_grid; ++i)
    {
        tt_samples[i] = rand() % 100 + 1;
        // std::cout << tt_samples[i] << " ";
    }
    // std::cout << std::endl;
    std::cout << "tt_samples:";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << tt_samples[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Start stackCUDA" << std::endl;
    // Call the stackCUDA function
    stackCUDA(data, tt_samples, n_sta, n_samples, n_grid, result);
    std::cout << "End stackCUDA" << std::endl;
    // Print a portion of the result
    std::cout << "result:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 1; i < 11; ++i)
    {
        std::cout << result[n_grid * n_samples-i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] data;
    delete[] tt_samples;
    delete[] result;
}

int main()
{
    testStack();
    return 0;
}
