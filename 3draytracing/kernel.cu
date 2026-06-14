#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "kernel.cuh"

/**
 * Calculate all possible travel times from a layer to a specific elevation level
 */
__global__ void
cudaRayKernel(float *vel, float *thick, float *rho, float *travel_time, float *incident,
              float *epicentral_distance, int n_layers)
{
    /* get current thread's id */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("\ncurrent id:%d", index);
    // printf("dev_l_rho:%d\n", dev_l_rho);

    /* while this thread is dealing with a valid index */
    if (index < dev_l_rho)
    {
        float r = rho[index]; // horizontal slowness
        float tr_time = 0.0f;
        float ep_dist = 0.0f;
        for (int k = 0; k < n_layers; k++)
        {
            // Calculate travel time with rho
            tr_time += thick[k] / (vel[k] * sqrt(1 - (r * r * vel[k] * vel[k])));
            ep_dist += (r * vel[k] * thick[k]) / sqrt(1 - (r * r * vel[k] * vel[k]));
        }

        // Travel time result, stored in global array
        travel_time[index] = tr_time;
        // Epicentral distance, stored in global array
        epicentral_distance[index] = ep_dist;
        // Calculate and store incident angle in radians
        incident[index] = asin(r * vel[0]);
        // printf("\ntime:%f,dist:%f,angle:%f", tr_time, ep_dist, asin(r * vel[n_layers - 1]) * 180);
        return;
    }
}

void cudaCallRayKernel(const unsigned int blocks,
                       const unsigned int threadsPerBlock, // number of threads per block
                       float *vel,                         // input velocity of each layer
                       float *thick,                       // input thickness of each layer
                       float *rho,                         // input horizontal slowness component
                       float *travel_time,                 // output travel time
                       float *incident,                    // output azimuth angle
                       float *epicentral_distance,         // output epicentral distance
                       int n_layers                        // number of layers
)
{

    // Call the kernel above this function.
    cudaRayKernel<<<blocks, threadsPerBlock>>>(vel, thick, rho, travel_time, incident,
                                               epicentral_distance, n_layers);
}

/**
 * Find the closest ray from each grid point in a layer to each station
 */
__global__ void cudaFindKernel(unsigned int nx,
                               unsigned int ny,
                               unsigned int nsta,
                               float *travel_time,         // input
                               float *incident,            // input
                               float *epicentral_distance, // input
                               float *staxy_ed,            // input
                               float *staxy_tt,            // output
                               float *staxy_incident,      // output
                               float *staxy_err            // output
)
{
    /* get current thread's id */
    // index is directly the staxy array index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("\ncurrent id:%d", index);

    /* while this thread is dealing with a valid index */
    if (index < nx * ny * nsta)
    {
        float real_ed = staxy_ed[index];
        float best_err = FLT_MAX;
        float best_tt, best_incident;
        // Iterate to find minimum err, record corresponding tt, incident, err to array
        for (int k = 0; k < dev_l_rho; ++k)
        {
            float err = abs(epicentral_distance[k] - real_ed);
            // Update minimum error and related data
            if (abs(epicentral_distance[k] - real_ed) < best_err)
            {
                best_err = err;
                best_tt = travel_time[k];
                best_incident = incident[k];
            }
        }
        // Write results to output array
        staxy_tt[index] = best_tt;
        staxy_incident[index] = best_incident;
        staxy_err[index] = best_err;
        return;
    }
}

void cudaCallFindKernel(
    const unsigned int blocks,
    const unsigned int threadsPerBlock,
    unsigned int nx,
    unsigned int ny,
    unsigned int nsta,
    float *travel_time,
    float *incident,
    float *epicentral_distance,
    float *staxy_ed,
    float *staxy_tt,
    float *staxy_incident,
    float *staxy_err)
{
    cudaFindKernel<<<blocks, threadsPerBlock>>>(nx, ny, nsta, travel_time, incident,
                                                epicentral_distance, staxy_ed,
                                                staxy_tt, staxy_incident, staxy_err);
}

// Function using CUDA kernel to set device global variable
__global__ void cudaUpdateLenRho(int new_value)
{
    dev_l_rho = new_value;
    printf("dev_l_rho change to:%d\n", dev_l_rho);
}
void update_dev_l_rho(int l_rho)
{
    cudaUpdateLenRho<<<1, 1>>>(l_rho);
    cudaDeviceSynchronize();
}