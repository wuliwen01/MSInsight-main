/**
 * Code for directly returning travel time results
 * (instead of saving results to file)
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <string>
#include <sstream>
#include <istream>
#include <windows.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <cassert>

#include "kernel.cuh"
#include "raycuda.hpp"

using namespace std;

/**
 * @function cuda_raytrace
 * @brief Main function to run the CUDA raytracer
 * @param threads_per_block Number of threads per block
 * @param confpath Path to the configuration folder
 */
extern "C" __declspec(dllexport) int cuda_raytrace(const unsigned int threads_per_block,
                                                   const char *path_conf_c,    // path of configuration file
                                                   const char *path_vel_c,     // path of velocity model file
                                                   const char *path_station_c, // path of station file
                                                   float *output_tt,           // output traveltime
                                                   float *output_incident,     // output incident angle
                                                   float *output_azimuth       // output azimuth
)
{
    /**
     * must confirm that all the stations locates in the first layer
     */
    // redirect stdout to file
    // freopen("rt_output.txt", "w", stdout);

    // Print device information
    // PrintDeviceInfo();
    cout << endl
         << "[RTtraveltime.dll]" << endl;
    string path_conf = path_conf_c;
    string path_vel = path_vel_c;
    string path_station = path_station_c;

    initTiming();
    double time_initial, time_final, elapsed_ms;
    // Layer data
    vector<float> input_vel_p, input_vel_s, input_upperbound;

    // Read layer file
    ifstream modelfile(path_vel);
    string temp;
    float ub, vp, vs, density;
    while (getline(modelfile, temp))
    {
        stringstream ss(temp);
        ss >> ub >> vp >> vs >> density;
        input_upperbound.push_back(ub * 1000);
        input_vel_p.push_back(vp * 1000);
        input_vel_s.push_back(vs * 1000);
    }
    // Set the number of layers
    int n_layers = input_upperbound.size();

    // build grid
    Grid grid(path_conf);
    grid.printConf();

    // load observation system
    float xRef = grid.xRef;
    float yRef = grid.yRef;
    float zRef = grid.zRef;
    ObsSystem osys(path_station, xRef, yRef, zRef);
    // osys.printStations();

    // classify station altitudes
    vector<int> classifiedAltitudes = osys.classifiedAltitudes;

    // Find max velocity
    const float rho_unit = 0.00000025;
    float max_vel = 0;
    max_vel = *max_element(input_vel_p.begin(), input_vel_p.end());
    // cout << "max velocity:" << max_vel << endl;
    // Calculate max rho and create array of rho values using max_vel
    float max_rho = (1 / max_vel) - rho_unit;
    cout << "max rho:" << max_rho << endl;
    int l_rho = (int)(max_rho / rho_unit); // len of rho array
    cout << "l_rho: " << l_rho << endl;
    update_dev_l_rho(l_rho);
    unsigned int blocks = ceil(l_rho * 1.0f / threads_per_block);
    cout << "blocks: " << blocks << endl;

    auto rho = make_unique<float[]>(l_rho * sizeof(float));
    for (int i = 0; i < l_rho; i++)
    {
        rho[i] = i * rho_unit;
    }

    // Get number of stations
    unsigned int nsta = osys.nsta;
    cout << "nsta: " << nsta << endl;
    // Get number of grid points
    unsigned int ngrid = grid.nx * grid.ny * grid.nz;
    cout << "ngrid: " << ngrid << endl;
    // Number of results per layer grid.nx * grid.ny * nsta
    unsigned int layer_result_size = grid.nx * grid.ny * nsta;

    // second kernel results
    float *zstaxy_tt = (float *)malloc(ngrid * nsta * sizeof(float));       // travel time
    float *zstaxy_incident = (float *)malloc(ngrid * nsta * sizeof(float)); // incident angle
    float *zstaxy_err = (float *)malloc(ngrid * nsta * sizeof(float));      // distance error
    float *staxy_ed = (float *)malloc(layer_result_size * sizeof(float));   // epicentral distance

    // calc epicentral distances, checked
    osys.calculateEpicentralDistances(grid.xValues, grid.yValues, staxy_ed);

    // ray kernel input
    float *dev_vel;
    float *dev_thick;
    float *dev_rho;
    cudaMalloc((void **)&dev_rho, l_rho * sizeof(float));
    cudaMemcpy(dev_rho, rho.get(), l_rho * sizeof(float), cudaMemcpyHostToDevice);

    // ray kernel result, len = l_rho
    float *dev_ttime;
    float *dev_angle;
    float *dev_epdistance;
    cudaMalloc((void **)&dev_ttime, l_rho * sizeof(float));
    cudaMalloc((void **)&dev_angle, l_rho * sizeof(float));
    cudaMalloc((void **)&dev_epdistance, l_rho * sizeof(float));

    // findkernel input
    float *dev_staxy_ed; // calculated ed for findkernel input
    cudaMalloc((void **)&dev_staxy_ed, layer_result_size * sizeof(float));
    cudaMemcpy(dev_staxy_ed, staxy_ed, layer_result_size * sizeof(float), cudaMemcpyHostToDevice);
    // findkernel result
    float *dev_staxy_tt;
    float *dev_staxy_ea;
    float *dev_staxy_err;
    cudaMalloc((void **)&dev_staxy_tt, layer_result_size * sizeof(float));
    cudaMalloc((void **)&dev_staxy_ea, layer_result_size * sizeof(float));
    cudaMalloc((void **)&dev_staxy_err, layer_result_size * sizeof(float));

    // GPU raytracing
    cout << "GPU raytracing..." << endl;

    // Start timer
    time_initial = preciseClock();
    int sta_z;
    float grid_z;
    // Iterate through combinations of station z and grid z. sta_z and grid_z will cause different lengths of vel and thick, need to handle separately
    for (size_t i_sta_z = 0; i_sta_z < classifiedAltitudes.size(); ++i_sta_z)
    {
        sta_z = classifiedAltitudes[i_sta_z];
        for (size_t i_grid_z = 0; i_grid_z < grid.zValues.size(); ++i_grid_z)
        {
            // Calculate rays from grid points in one layer to each station
            grid_z = grid.zValues[i_grid_z];
            vector<float> cpu_vel, cpu_thick;
            calculateLayerProperties(input_upperbound.data(), input_vel_p.data(), n_layers, grid_z, sta_z,
                                     cpu_thick, cpu_vel);
            int cpu_nlayer = cpu_thick.size();

            // Copy input velocity and thickness data from host memory to the GPU
            cudaMalloc((void **)&dev_vel, cpu_nlayer * sizeof(float));
            cudaMemcpy(dev_vel, cpu_vel.data(), cpu_nlayer * sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc((void **)&dev_thick, cpu_nlayer * sizeof(float));
            cudaMemcpy(dev_thick, cpu_thick.data(), cpu_nlayer * sizeof(float), cudaMemcpyHostToDevice);

            // Calculate ray travel time from current grid layer to current staz
            cudaCallRayKernel(blocks, threads_per_block, dev_vel, dev_thick, dev_rho,
                              dev_ttime, dev_angle, dev_epdistance, cpu_nlayer);

            // Check for errors on kernel call
            cudaError err = cudaGetLastError();
            if (cudaSuccess != err)
                cerr << "Error " << cudaGetErrorString(err) << endl;

            unsigned int find_kernel_block = ceil(layer_result_size * 1.0f / threads_per_block);

            // Find minimum ray travel time from current grid layer to staz including stations
            cudaCallFindKernel(find_kernel_block, threads_per_block,
                               grid.nx, grid.ny, nsta, dev_ttime, dev_angle, dev_epdistance, dev_staxy_ed,
                               dev_staxy_tt, dev_staxy_ea, dev_staxy_err);
            unsigned offset = i_grid_z * layer_result_size;
            CHECK(cudaMemcpy(zstaxy_tt + offset, dev_staxy_tt, layer_result_size * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(zstaxy_incident + offset, dev_staxy_ea, layer_result_size * sizeof(float), cudaMemcpyDeviceToHost));
            // CHECK(cudaMemcpy(zstaxy_err + offset, dev_staxy_err, layer_result_size * sizeof(float), cudaMemcpyDeviceToHost));

            // print err, calc average err
            // double sum_err = accumulate(zstaxy_err, zstaxy_err + layer_result_size, 0.0f);
            // double avg_err = sum_err / (layer_result_size);
            // printf("staz: %d, gridz: %.6f, avg_err: %.6f\n", sta_z, grid_z, avg_err);

            // Free temporary memory on the GPU
            cudaFree(dev_thick);
            cudaFree(dev_vel);
            // return EXIT_SUCCESS;
        }
        // copy data of stations in this staz to output_tt and output_incident
        // staz result shape: z,sta,x,y
        // output shape: sta,x,y,z
        // cout << "grid_z: " << grid_z << endl;
        for (int ista = 0; ista < nsta; ++ista)
        {
            if (osys.roundedZValues[ista] != sta_z)
                // not in this altitude category
                continue;
            // cout << "sta " << ista << " in this altitude category" << endl;
            for (int ix = 0; ix < grid.nx; ++ix)
            {
                for (int iy = 0; iy < grid.ny; ++iy)
                {
                    for (int iz = 0; iz < grid.nz; ++iz)
                    {
                        int idx = getIndex_zstaxy(ista, ix, iy, iz, nsta, grid.nx, grid.ny);
                        int idx_out = getIndex_staxyz(ista, ix, iy, iz, grid.nx, grid.ny, grid.nz);
                        output_tt[idx_out] = zstaxy_tt[idx];
                        output_incident[idx_out] = zstaxy_incident[idx];
                    }
                }
            }
        }
    }

    // Stop timer
    time_final = preciseClock();
    elapsed_ms = time_final - time_initial;
    cout << "raytracing time: " << elapsed_ms << " milliseconds" << endl;

    // Free all allocated memory on the GPU
    cudaFree(dev_ttime);
    cudaFree(dev_angle);
    cudaFree(dev_epdistance);
    cudaFree(dev_rho);

    // calculate azimuth
    time_initial = preciseClock();
    for (int ista = 0; ista < nsta; ++ista)
    {
        float stationX = osys.xValues[ista];
        float stationY = osys.yValues[ista];
        float stationZ = osys.zValues[ista];

        for (int ix = 0; ix < grid.nx; ++ix)
        {
            float gridX = grid.xStart + ix * grid.xInterval;

            for (int iy = 0; iy < grid.ny; ++iy)
            {
                float gridY = grid.yStart + iy * grid.yInterval;

                for (int iz = 0; iz < grid.nz; ++iz)
                {
                    float gridZ = grid.zStart + iz * grid.zInterval;

                    // Index mapping
                    int idx_out = getIndex_staxyz(ista, ix, iy, iz, grid.nx, grid.ny, grid.nz);

                    // Calculate azimuth (in radians)
                    float azimuth = atan2(stationX - gridX, stationY - gridY);
                    if (azimuth < 0)
                        azimuth += 2 * M_PI; // Adjust to [0, 2*pi)
                    // Save azimuth to output array
                    output_azimuth[idx_out] = azimuth;
                }
            }
        }
    }
    time_final = preciseClock();
    elapsed_ms = time_final - time_initial;
    cout << "azimuth calc time: " << elapsed_ms << " milliseconds" << endl;
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    check_args(argc, argv);
    unsigned int threads_per_block = atoi(argv[1]);
    string conf_path = argv[2];
    string conf_fpath = conf_path + "/conf.txt";
    string sta_fpath = conf_path + "/station.txt";
    string vel_fpath = conf_path + "/vel.txt";
    cout << "threads_per_block: " << threads_per_block << endl;
    cout << "conf_path: " << conf_path << endl;
    // return 0;
    float *tt = new float[40 * 51 * 51 * 31];
    float *incident = new float[40 * 51 * 51 * 31];
    float *azimuth = new float[40 * 51 * 51 * 31];
    cuda_raytrace(threads_per_block, conf_fpath.data(), vel_fpath.data(), sta_fpath.data(), tt, incident, azimuth);
    return 0;
}
