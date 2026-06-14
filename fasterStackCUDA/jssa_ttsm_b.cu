// traveltime table preload into shared memory
// batch load intensity, use asynchronous stream
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
 * @brief CUDA kernel function to perform stacking operation. It computes the stacking result of all station data at a specific grid point, mechanism, and sample point.
 *
 * @param data Input data array of size n_sta * n_samples, storing the data for each station.
 * @param tt_samples Time shift array for each station, of size n_sta * n_grid, storing the time shift for each station at each grid point.
 * @param intensity Intensity array of size n_fm * n_grid * n_sta, storing the intensity values for each mechanism, grid point, and station.
 * @param result Output result array of size n_grid * n_fm * n_samples, used to store the stacked result.
 * @param n_sta Number of stations.
 * @param n_samples Number of sample points for each station.
 * @param n_grid Number of grid points.
 * @param n_fm Number of mechanisms.
 */
__global__ void stackKernel(const float *data, const int *tt_samples, const float *intensity,
                            float *result, int n_sta, int n_samples, int n_grid, int n_fm)
{
    int i_grid = blockIdx.x;
    int i_fm = blockIdx.y;
    int i_sample = threadIdx.x;

    // Load tt_samples into shared memory
    extern __shared__ int shared_tt_samples[];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < n_sta; ++i)
        {
            shared_tt_samples[i] = tt_samples[i * n_grid + i_grid];
        }
    }
    __syncthreads(); // Ensure all threads in the grid have finished loading

    if (i_sample >= n_samples)
        return;

    float sum = 0.0f;
    // float shifted_data = 0.0f;
    int offset = 0;
    const int intensity_base = i_fm * n_grid * n_sta + i_grid * n_sta;
    // Use float4 to batch load intensity (process 4 stations per iteration, continuous address)
    int i_sta = 0;
    for (; i_sta < n_sta - 3; i_sta += 4)
    {
        offset = shared_tt_samples[i_sta]; // Get the offset of the first station in the current batch (assumes the same offset for all stations in the batch, adjust if different, see note)
        if (i_sample + offset < n_samples)
        {
            // Batch read 4 continuous intensity elements (float4)
            float4 inten4 = reinterpret_cast<const float4 *>(intensity + intensity_base)[i_sta / 4];

            // Batch read 4 data elements
            float data0 = data[i_sta * n_samples + (i_sample + offset)];
            float data1 = data[(i_sta + 1) * n_samples + (i_sample + offset)];
            float data2 = data[(i_sta + 2) * n_samples + (i_sample + offset)];
            float data3 = data[(i_sta + 3) * n_samples + (i_sample + offset)];

            // Accumulate the weighted results of 4 stations
            sum += data0 * inten4.x;
            sum += data1 * inten4.y;
            sum += data2 * inten4.z;
            sum += data3 * inten4.w;
        }
    }

    // Process remaining stations
    for (; i_sta < n_sta; ++i_sta)
    {
        offset = shared_tt_samples[i_sta];
        if (i_sample + offset < n_samples)
        {
            float inten = intensity[intensity_base + i_sta];
            float shifted_data = data[i_sta * n_samples + (i_sample + offset)];
            sum += shifted_data * inten;
        }
    }

    sum /= n_sta; // Average the stacked results for all stations
    result[i_grid * n_fm * n_samples + i_fm * n_samples + i_sample] = sum;
}

/**
 * @brief Function to perform joint stacking operations in CUDA. It calculates the stacking result for all station data at a specific grid point, mechanism, and sample point.
 *
 * @param data Input data array of size n_sta * n_samples, storing data for each station.
 * @param tt_samples Time shift array for each station, of size n_sta * n_grid, storing the time shift for each station at each grid point.
 * @param intensity Intensity array of size n_fm * n_grid * n_sta, storing the intensity values for each mechanism, grid point, and station.
 * @param n_sta Number of stations.
 * @param n_samples Number of sample points for each station.
 * @param n_grid Number of grid points.
 * @param n_fm Number of mechanisms.
 * @param result Output result array of size n_grid * n_fm * n_samples, used to store the stacked results.
 */
__declspec(dllexport) int stackCUDA(const float *data, const int *tt_samples, const float *intensity,
                                    int n_sta, int n_samples, int n_grid, int n_fm,
                                    float *result)
{
    std::cout << "[SSA.cu] running stackCUDA " << std::endl;
    // Allocate device memory
    float *d_data, *d_intensity, *d_result;
    int *d_tt_samples;
    std::cout << "[SSA.cu] n_sta: " << n_sta << " n_samples: " << n_samples << " n_grid: " << n_grid << " n_fm: " << n_fm << std::endl;

    size_t data_size = n_sta * n_samples * sizeof(float);
    size_t tt_samples_size = n_sta * n_grid * sizeof(int);
    size_t intensity_size = n_fm * n_grid * n_sta * sizeof(float);
    size_t result_size = n_grid * n_fm * n_samples * sizeof(float);

    std::cout << "[SSA.cu] allocate GPU memory: " << data_size + tt_samples_size + intensity_size + result_size << std::endl;
    std::cout << "[SSA.cu] data size(K): " << data_size / 1024 << std::endl;
    std::cout << "[SSA.cu] tt_samples size(K): " << tt_samples_size / 1024 << std::endl;
    std::cout << "[SSA.cu] intensity size(M): " << intensity_size / 1024 / 1024 << std::endl;
    std::cout << "[SSA.cu] result size(M): " << result_size / 1024 / 1024 << std::endl;

    cudaError_t err;
    err = cudaMalloc(&d_data, data_size);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_data: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaMalloc(&d_tt_samples, tt_samples_size);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_tt_samples: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return -1;
    }
    err = cudaMalloc(&d_intensity, intensity_size);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_intensity: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        return -1;
    }
    err = cudaMalloc(&d_result, result_size);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_result: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        return -1;
    }

    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA stream create failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }

    err = cudaMemcpyAsync(d_data, data, data_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        std::cerr << "H2D copy failed for d_data: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }
    err = cudaMemcpyAsync(d_tt_samples, tt_samples, tt_samples_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        std::cerr << "H2D copy failed for d_tt_samples: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }
    err = cudaMemcpyAsync(d_intensity, intensity, intensity_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        std::cerr << "H2D copy failed for d_intensity: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }

    err = cudaMemsetAsync(d_result, 0, result_size, stream);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemsetAsync failed for d_result: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }

    dim3 gridDim(n_grid, n_fm);
    dim3 blockDim(n_samples);
    size_t shared_mem_size = n_sta * sizeof(int);

    auto start = std::chrono::high_resolution_clock::now();

    stackKernel<<<gridDim, blockDim, shared_mem_size, stream>>>(d_data, d_tt_samples, d_intensity, d_result,
                                                                n_sta, n_samples, n_grid, n_fm);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }

    err = cudaMemcpyAsync(result, d_result, result_size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        std::cerr << "D2H copy failed for result: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        std::cerr << "Stream synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_data);
        cudaFree(d_tt_samples);
        cudaFree(d_intensity);
        cudaFree(d_result);
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "[SSA.cu] total execution time (copy + kernel + copy back): " << duration.count() << " ms" << std::endl;

    cudaStreamDestroy(stream);

    err = cudaFree(d_data);
    if (err != cudaSuccess)
    {
        printf("cudaFree d_data failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaFree(d_tt_samples);
    if (err != cudaSuccess)
    {
        printf("cudaFree d_tt_samples failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaFree(d_intensity);
    if (err != cudaSuccess)
    {
        printf("cudaFree d_intensity failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaFree(d_result);
    if (err != cudaSuccess)
    {
        printf("cudaFree d_result failed: %s\n", cudaGetErrorString(err));
    }

    return 0;
}

#include <cmath>
#include <array>
#include <vector>
#define M_PI 3.14159265358979323846

/**
 * @brief Converts fault parameters (strike, dip, rake) to a moment tensor.
 *
 * @param strike Strike angle in degrees.
 * @param dip Dip angle in degrees.
 * @param rake Rake angle in degrees.
 * @param moment_tensor Output moment tensor as a 3x3 matrix.
 */
__device__ void faultParametersToMomentTensor(float strike, float dip, float rake, float moment_tensor[3][3])
{
    // Convert angles from degrees to radians
    strike = strike * M_PI / 180.0f;
    dip = dip * M_PI / 180.0f;
    rake = rake * M_PI / 180.0f;

    // Calculate moment tensor components
    moment_tensor[0][0] = -std::sin(dip) * std::cos(rake) * std::sin(2 * strike) -
                          std::sin(2 * dip) * std::sin(rake) * std::pow(std::sin(strike), 2);
    moment_tensor[1][1] = std::sin(dip) * std::cos(rake) * std::sin(2 * strike) -
                          std::sin(2 * dip) * std::sin(rake) * std::pow(std::cos(strike), 2);
    moment_tensor[0][1] = std::sin(dip) * std::cos(rake) * std::cos(2 * strike) +
                          0.5f * std::sin(2 * dip) * std::sin(rake) * std::sin(2 * strike);
    moment_tensor[1][0] = moment_tensor[0][1];
    moment_tensor[0][2] = -std::cos(dip) * std::cos(rake) * std::cos(strike) -
                          std::cos(2 * dip) * std::sin(rake) * std::sin(strike);
    moment_tensor[2][0] = moment_tensor[0][2];
    moment_tensor[1][2] = -std::cos(dip) * std::cos(rake) * std::sin(strike) +
                          std::cos(2 * dip) * std::sin(rake) * std::cos(strike);
    moment_tensor[2][1] = moment_tensor[1][2];
    moment_tensor[2][2] = std::sin(2 * dip) * std::sin(rake);
}

/**
 * @brief Calculates the radiation intensity for a given vector and moment tensor.
 *        The vector direction should be from the source to the receiver (north, east, down components).
 * @param moment_tensor Moment tensor as a 3x3 matrix.
 * @param vector Direction vector from the source to the receiver (north, east, down components).
 *                The vector should be normalized before passing it to this function.
 * @return float Radiation intensity.
 */
__device__ void calculateRadiationIntensity(const float moment_tensor[3][3], const float vector[3], float &intensity)
{
    // Normalize the vector
    float norm = std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    float normalized_vector[3] = {vector[0] / norm, vector[1] / norm, vector[2] / norm};

    // Calculate the radiation intensity
    // float intensity = 0.0f;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            intensity += normalized_vector[i] * moment_tensor[i][j] * normalized_vector[j];
        }
    }
}

typedef struct
{
    float GridSpacingX;
    float GridSpacingZ;
    int SearchSizeX;
    int SearchSizeY;
    int SearchSizeZ;
    float SearchOriginX;
    float SearchOriginY;
    float SearchOriginZ;
} GridConfig;

__global__ void intensityKernel(const float *fm_grid, const GridConfig *conf, const float *stations,
                                float *result, int n_sta, int n_fm, int n_grid, float a)
{
    // Calculate the intensity values for all station data at each grid point and mechanism
    int i_grid = blockIdx.x; // Grid point index
    int i_fm = blockIdx.y;   // Mechanism index
    int i_sta = threadIdx.x; // Station index

    // if (i_sta >= n_sta)
    //     return;

    // Calculate grid point coordinates
    // Note: GridSpacingX and GridSpacingY are always the same, so we use GridSpacingX directly
    float x = conf->SearchOriginX + (i_grid / (conf->SearchSizeY * conf->SearchSizeZ)) * conf->GridSpacingX;
    float y = conf->SearchOriginY + ((i_grid / conf->SearchSizeZ) % conf->SearchSizeY) * conf->GridSpacingX;
    float z = conf->SearchOriginZ + (i_grid % conf->SearchSizeZ) * conf->GridSpacingZ;

    // Calculate distance
    float dx = x - stations[i_sta * 3];
    float dy = y - stations[i_sta * 3 + 1];
    float dz = z - stations[i_sta * 3 + 2];
    // vector direction: north, east, down
    float vector[3] = {dy, dx, dz};
    float v_len = sqrt(dx * dx + dy * dy + dz * dz + 1e-6);

    // calculate moment tensor
    float moment_tensor[3][3];
    faultParametersToMomentTensor(fm_grid[i_fm * 3], fm_grid[i_fm * 3 + 1], fm_grid[i_fm * 3 + 2], moment_tensor);
    // calculate intensity
    float *p = &result[i_fm * n_grid * n_sta + i_grid * n_sta + i_sta];
    calculateRadiationIntensity(moment_tensor, vector, *p);
    // *p = v_len;
    *p *= exp(-a * v_len);
}

/**
 * @brief Function to compute the intensity in CUDA. It calculates the intensity values for all station data at each grid point and mechanism.
 *
 * @param fm_grid Input data array of size n_fm * 3, storing the parameters for each mechanism.
 * @param conf Grid configuration structure containing grid parameters.
 * @param stations Array of stations of size n_sta * 3, storing the coordinates of each station.
 * @param n_sta Number of stations.
 * @param n_fm Number of mechanisms.
 * @param a Distance attenuation factor.
 * @param result Output result array of size n_fm * n_grid * n_sta, used to store the intensity values for each mechanism, grid point, and station.
 */

__declspec(dllexport) int intensityCUDA(const float *fm_grid, GridConfig *conf, float *stations,
                                        int n_sta, int n_fm, float a,
                                        float *result)
{
    std::cout << "[SSA.cu] running intensityCUDA " << std::endl;
    // Number of grid points
    int n_grid = conf->SearchSizeX * conf->SearchSizeY * conf->SearchSizeZ;
    std::cout << "[SSA.cu] n_sta: " << n_sta << " n_fm: " << n_fm << " n_grid: " << n_grid << std::endl;

    // Allocate device memory
    float *d_fm_grid, *d_stations, *d_result;
    GridConfig *d_conf;
    if (cudaMalloc(&d_fm_grid, n_fm * 3 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_fm_grid" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_conf, sizeof(GridConfig)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_conf" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_stations, n_sta * 3 * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_stations" << std::endl;
        return -1;
    }
    if (cudaMalloc(&d_result, n_fm * n_grid * n_sta * sizeof(float)) != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed for d_result" << std::endl;
        return -1;
    }
    std::cout << "[SSA.cu] use GPU memory: " << n_fm * 3 * sizeof(float) + sizeof(GridConfig) + n_sta * 3 * sizeof(float) + n_fm * n_grid * n_sta * sizeof(float) << std::endl;

    // Copy data to device
    cudaMemcpy(d_fm_grid, fm_grid, n_fm * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conf, conf, sizeof(GridConfig), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stations, stations, n_sta * 3 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemset(d_result, 0, n_fm * n_grid * n_sta * sizeof(float));

    // Define grid and block dimensions
    dim3 gridDim(n_grid, n_fm);
    dim3 blockDim(n_sta); // n_sta less than 1024
    // Launch kernel
    intensityKernel<<<gridDim, blockDim>>>(d_fm_grid, d_conf, d_stations, d_result,
                                           n_sta, n_fm, n_grid, a);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // Copy result back to host
    cudaMemcpy(result, d_result, n_fm * n_grid * n_sta * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_fm_grid);
    cudaFree(d_conf);
    cudaFree(d_stations);
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
    std::cout << "GPU num: " << count << std::endl;
    std::cout << "Maximum threads per block:: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum thread dimensions per block: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Maximum threads per grid: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
    std::cout << "Maximum shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem << std::endl;
    // Example usage
    int n_sta = 80;      // Number of stations
    int n_samples = 512; // Number of samples per station
    int n_grid = 512;    // Number of grid points
    int n_fm = 576;      // Number of focal mechanisms

    float *data = new float[n_sta * n_samples];
    int *tt_samples = new int[n_sta * n_grid];
    float *intensity = new float[n_fm * n_grid * n_sta];
    float *result = new float[n_grid * n_fm * n_samples]{0};

    // Initialize example data
    for (int i = 0; i < n_sta * n_samples; ++i)
        data[i] = 0.01f * i;

    // std::cout << "tt_samples:";
    for (int i = 0; i < n_sta * n_grid; ++i)
    {
        tt_samples[i] = 1;
        // std::cout << tt_samples[i] << " ";
    }
    // std::cout << std::endl;

    for (int i = 0; i < n_fm * n_grid * n_sta; ++i)
        intensity[i] = 0.0000001f * i;
    // std::cout << "intensity:" << std::endl;
    // for (int i = 0; i < n_fm * n_grid * n_sta; ++i)
    // {
    //     std::cout << intensity[i] << " ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < n_grid * n_fm * n_samples; ++i)
    //     result[i] = 0.0f;
    // std::cout << "result:";
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << result[i] << " ";
    // }
    // std::cout << std::endl;

    std::cout << "Start stackCUDA" << std::endl;
    // Call the stackCUDA function
    stackCUDA(data, tt_samples, intensity, n_sta, n_samples, n_grid, n_fm, result);
    std::cout << "End stackCUDA" << std::endl;
    // Print a portion of the result
    std::cout << "result:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] data;
    delete[] tt_samples;
    delete[] intensity;
    delete[] result;
}

void testIntensity()
{
    std::cout << "CUDA Version: " << CUDART_VERSION << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << ", sizeof(int): " << sizeof(int) << std::endl;

    // Example usage
    int n_sta = 4; // Number of stations
    int n_fm = 1;  // Number of frequency modes

    float *fm_grid = new float[n_fm * 3];
    fm_grid[0] = 339.0f; // Strike
    fm_grid[1] = 15.0f;  // Dip
    fm_grid[2] = 161.0f; // Rake
    GridConfig conf;
    conf.GridSpacingX = 1.0f;
    conf.GridSpacingZ = 1.0f;
    conf.SearchSizeX = 3;
    conf.SearchSizeY = 3;
    conf.SearchSizeZ = 3;
    conf.SearchOriginX = -1.0f;
    conf.SearchOriginY = -1.0f;
    conf.SearchOriginZ = -1.0f;
    int n_grid = conf.SearchSizeX * conf.SearchSizeY * conf.SearchSizeZ; // Number of grid points

    float *stations = new float[n_sta * 3];
    for (int i = 0; i < n_sta * 3; ++i)
        stations[i] = i * 0.001 + 1.0f;

    float a = 0.1f;

    float *result = new float[n_fm * n_grid * n_sta];

    std::cout << "Start intensityCUDA" << std::endl;
    intensityCUDA(fm_grid, &conf, stations, n_sta, n_fm, a, result);
    std::cout << "End intensityCUDA" << std::endl;

    // Print a portion of the result
    std::cout << "result:" << std::endl;
    for (int i = 0; i < 10; ++i)
        std::cout << result[i] << " ";
    std::cout << std::endl;

    // Free host memory
    delete[] fm_grid;
    delete[] stations;
    delete[] result;
}

int main()
{
    testStack();
    // testIntensity();
    return 0;
}