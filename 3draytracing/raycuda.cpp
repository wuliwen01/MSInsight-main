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
__declspec(dllexport) int cuda_raytrace(const unsigned int threads_per_block,
                                        const string &confpath // path to .config folder
)
{
    PrintDeviceInfo();
    string path_vel = confpath + "/vel.txt";
    string path_station = confpath + "/station.txt";
    string path_conf = confpath + "/conf.txt";

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
    osys.printStations();

    // classify station altitudes
    vector<int> classifiedAltitudes = osys.classifiedAltitudes;
    cout << "Length of classifiedAltitudes: " << classifiedAltitudes.size() << endl;
    cout << "Length of grid.zValues: " << grid.zValues.size() << endl;

    // Find max velocity
    const float rho_unit = 0.0000005;
    float max_vel = 0;
    max_vel = *max_element(input_vel_p.begin(), input_vel_p.end());
    cout << "max velocity:" << max_vel << endl;
    // Calculate max rho and create array of rho values using max_vel
    float max_rho = (1 / max_vel) - rho_unit;
    printf("max_rho: %f\n", max_rho);
    int l_rho = (int)(max_rho / rho_unit); // len of rho array
    printf("l_rho: %d\n", l_rho);
    update_dev_l_rho(l_rho);
    unsigned int blocks = ceil(l_rho * 1.0f / threads_per_block);
    printf("blocks: %d\n", blocks);

    auto rho = make_unique<float[]>(l_rho * sizeof(float));
    for (int i = 0; i < l_rho; i++)
    {
        rho[i] = i * rho_unit;
    }

    // Get number of stations
    unsigned int nsta = osys.nsta;
    // Get number of grid points
    unsigned int ngrid = grid.nx * grid.ny * grid.nz;
    // Number of results per layer grid.nx * grid.ny * nsta
    unsigned int layer_result_size = grid.nx * grid.ny * nsta;
    // Create cpu array to hold kernel results
    // float *travel_time = (float *)malloc(l_rho * sizeof(float));
    // float *exit_angle = (float *)malloc(l_rho * sizeof(float));
    // float *epc_dist = (float *)malloc(l_rho * sizeof(float));
    // second kernel results
    float *zstaxy_tt = (float *)malloc(ngrid * nsta * sizeof(float));     // travel time
    float *zstaxy_ea = (float *)malloc(ngrid * nsta * sizeof(float));     // exit angel
    float *zstaxy_err = (float *)malloc(ngrid * nsta * sizeof(float));    // distance error
    float *staxy_ed = (float *)malloc(layer_result_size * sizeof(float)); // epicentral distance

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

    // Create float* type array to store results, shape is (nsta, nx, ny, nz)
    // float *output_tt, *output_angle;
    // auto output_tt = new (std::nothrow) float[nsta * ngrid];
    // auto output_angle = new (std::nothrow) float[nsta * ngrid];
    auto output_tt = make_unique<float[]>(nsta * ngrid);
    auto output_angle = make_unique<float[]>(nsta * ngrid);
    if (output_tt == nullptr || output_angle == nullptr)
    {
        perror("Failed to allocate memory for array");
        return EXIT_FAILURE;
    }

    // GPU raytracing
    cout << endl
         << "GPU raytracing..." << endl;

    // Start timer
    time_initial = preciseClock();
    int sta_z;
    float grid_z;
    // sta_z and grid_z will cause different lengths of vel and thick, need to handle separately
    for (size_t i_sta_z = 0; i_sta_z < classifiedAltitudes.size(); ++i_sta_z)
    {
        sta_z = classifiedAltitudes[i_sta_z];
        for (size_t i_grid_z = 0; i_grid_z < grid.zValues.size(); ++i_grid_z)
        {
            // Calculate rays from grid points in one layer to each station
            grid_z = grid.zValues[i_grid_z];
            // printf("grid_z=%f\n", grid_z);
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

            // // retrieve results
            // cudaMemcpy(travel_time, dev_ttime, l_rho * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(exit_angle, dev_angle, l_rho * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(epc_dist, dev_epdistance, l_rho * sizeof(float), cudaMemcpyDeviceToHost);
            // // print travel time
            // for (int i = 0; i < l_rho; ++i)
            // {
            //     printf("rho %.6f, tt %.2f ms, angle %.2f, epdist %.1f\n",
            //            rho[i], travel_time[i] * 1000, exit_angle[i], epc_dist[i]);
            // }

            unsigned int find_kernel_block = ceil(layer_result_size * 1.0f / threads_per_block);

            // Find minimum ray travel time from current grid layer to staz including stations
            cudaCallFindKernel(find_kernel_block, threads_per_block,
                               grid.nx, grid.ny, nsta, dev_ttime, dev_angle, dev_epdistance, dev_staxy_ed,
                               dev_staxy_tt, dev_staxy_ea, dev_staxy_err);
            unsigned offset = i_grid_z * layer_result_size;
            CHECK(cudaMemcpy(zstaxy_tt + offset, dev_staxy_tt, layer_result_size * sizeof(float), cudaMemcpyDeviceToHost));
            // CHECK(cudaMemcpy(zstaxy_ea + offset, dev_staxy_ea, layer_result_size * sizeof(float), cudaMemcpyDeviceToHost));
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
        // copy data of stations in this staz to output_tt and output_angle
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
                        // output_angle[idx_out] = zstaxy_ea[idx];
                        // cout << "zstaxy_tt[" << idx << "] " << zstaxy_tt[idx] << endl;
                        // cout << "zstaxy_ea[" << idx << "] " << zstaxy_ea[idx] << endl;
                        // cout << "output_tt[" << idx_out << "] " << output_tt[idx_out] << endl;
                        // cout << "output_angle[" << idx_out << "] " << output_angle[idx_out] << endl;
                    }
                }
            }
        }
    }

    // Stop timer
    time_final = preciseClock();
    elapsed_ms = time_final - time_initial;

    cout << "Total time: " << elapsed_ms << " milliseconds" << endl;

    // save results to .buf file
    saveData("time", osys.sta_ids, output_tt.get(), grid.ngrid);
    saveDataInfo("time", osys, grid);
    // saveData("angle", osys.sta_ids, output_angle.get(), grid.ngrid);

    // Free all allocated memory on the GPU
    cudaFree(dev_ttime);
    cudaFree(dev_angle);
    cudaFree(dev_epdistance);
    cudaFree(dev_rho);

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    check_args(argc, argv);
    unsigned int threads_per_block = atoi(argv[1]);
    string conf_path = argv[2];
    cout << "threads_per_block: " << threads_per_block << endl;
    cout << "conf_path: " << conf_path << endl;
    // return 0;
    cuda_raytrace(threads_per_block, conf_path);
    return 0;
}
