#ifndef RAY_CUH
#define RAY_CUH

#define M_PI 3.14159265358979323846

static __device__ int dev_l_rho;

void update_dev_l_rho(int l_rho);

void cudaCallRayKernel(const unsigned int blocks,
                       const unsigned int threadsPerBlock,
                       float *vel,
                       float *thick,
                       float *rho,
                       float *travel_time,
                       float *incident,
                       float *epicentral_distance,
                       int n_layers);

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
    float *staxy_err);

#define CHECK(call)                                     \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

#endif