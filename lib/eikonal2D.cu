#include <cmath>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <nppi.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <npp.h>
// #include <nppi.h>
// #include <nppi_resize.h>  // Specifically required for nppiResize_32f_C1R with scaling

// Error check macro
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

__global__ void fsm_kernel(float* T, float* S, int* sgnv, int* sgnt, int sgni, int sgnj,
                           int x_offset, int z_offset, int xd, int zd,
                           int nxx, int nzz, float dx, float dz, float dx2i, float dz2i)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;

    int i = z_offset + zd * element;
    int j = x_offset + xd * element;

    if ((i > 0) && (i < nzz - 1) && (j > 0) && (j < nxx - 1)) {
        int i1 = i - sgnv[sgni];
        int j1 = j - sgnv[sgnj];

        float tv = T[i - sgnt[sgni] + j * nzz];
        float te = T[i + (j - sgnt[sgnj]) * nzz];
        float tev = T[(i - sgnt[sgni]) + (j - sgnt[sgnj]) * nzz];

        float t1d1 = tv + dz * fminf(S[i1 + max(j - 1, 1) * nzz], S[i1 + min(j, nxx - 1) * nzz]);
        float t1d2 = te + dx * fminf(S[max(i - 1, 1) + j1 * nzz], S[min(i, nzz - 1) + j1 * nzz]);

        float t1D = fminf(t1d1, t1d2);
        float t1 = 1e6f, t2 = 1e6f, t3 = 1e6f;

        float Sref = S[i1 + j1 * nzz];

        if ((tv <= te + dx * Sref) && (te <= tv + dz * Sref) &&
            (te - tev >= 0.0f) && (tv - tev >= 0.0f)) {
            float ta = tev + te - tv;
            float tb = tev - te + tv;

            float disc = 4.0f * Sref * Sref * (dz2i + dx2i) - dz2i * dx2i * (ta - tb) * (ta - tb);
            if (disc > 0.0f) {
                t1 = ((tb * dz2i + ta * dx2i) + sqrtf(disc)) / (dz2i + dx2i);
            }
        }
        else if ((te - tev <= Sref * dz * dz / sqrtf(dx * dx + dz * dz)) && (te - tev > 0.0f)) {
            t2 = te + dx * sqrtf(Sref * Sref - ((te - tev) / dz) * ((te - tev) / dz));
        }
        else if ((tv - tev <= Sref * dx * dx / sqrtf(dx * dx + dz * dz)) && (tv - tev > 0.0f)) {
            t3 = tv + dz * sqrtf(Sref * Sref - ((tv - tev) / dx) * ((tv - tev) / dx));
        }

        float t2D = fminf(t1, fminf(t2, t3));

        T[i + j * nzz] = fminf(T[i + j * nzz], fminf(t1D, t2D));
    }
}

extern "C" {
float* fast_sweeping_method(float* Vp, float sx, float sz, float dx, float dz, int nx, int nz)
{
    int nb = 2;
    int nxx = nx + 2 * nb;
    int nzz = nz + 2 * nb;
    int matsize = nxx * nzz;

    float* T = new float[matsize]();
    float* S = new float[matsize]();
    float* eikonal = new float[nx * nz]();

    // Set slowness field S from velocity
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            S[(i + nb) + (j + nb) * nzz] = 1.0f / Vp[i + j * nz];
        }
    }

    // Expand boundaries (mirror)
    for (int i = 0; i < nb; i++) {
        for (int j = nb; j < nxx - nb; j++) {
            S[i + j * nzz] = S[nb + j * nzz];
            S[(nzz - i - 1) + j * nzz] = S[(nzz - nb - 1) + j * nzz];
        }
    }
    for (int i = 0; i < nzz; i++) {
        for (int j = 0; j < nb; j++) {
            S[i + j * nzz] = S[i + nb * nzz];
            S[i + (nxx - j - 1) * nzz] = S[i + (nxx - nb - 1) * nzz];
        }
    }

    int sIdx = (int)(sx / dx) + nb;
    int sIdz = (int)(sz / dz) + nb;

    for (int index = 0; index < matsize; index++) T[index] = 1e6f;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int xi = sIdx + (j - 1);
            int zi = sIdz + (i - 1);

            T[zi + xi * nzz] = S[zi + xi * nzz] * sqrtf(powf((xi - nb) * dx - sx, 2.0f) +
                                                        powf((zi - nb) * dz - sz, 2.0f));
        }
    }

    int nSweeps = 4;
    int meshDim = 2;
    int nThreads = 32;
    float dz2i = 1.0f / (dz * dz);
    float dx2i = 1.0f / (dx * dx);
    int min_level = std::min(nxx, nzz);
    int max_level = std::max(nxx, nzz);
    int total_levels = (nxx - 1) + (nzz - 1);

    std::vector<std::vector<int>> sgnv = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
    std::vector<std::vector<int>> sgnt = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

    int* h_sgnv = new int[nSweeps * meshDim]();
    int* h_sgnt = new int[nSweeps * meshDim]();

    for (int index = 0; index < nSweeps * meshDim; index++) {
        int j = index / nSweeps;
        int i = index % nSweeps;

        h_sgnv[i + j * nSweeps] = sgnv[i][j];
        h_sgnt[i + j * nSweeps] = sgnt[i][j];
    }

    std::vector<std::vector<int>>().swap(sgnv);
    std::vector<std::vector<int>>().swap(sgnt);

    // GPU memory allocation
    float* d_T = nullptr;
    float* d_S = nullptr;
    int* d_sgnv = nullptr;
    int* d_sgnt = nullptr;

    cudaMalloc((void**)&d_T, matsize * sizeof(float));
    cudaMalloc((void**)&d_S, matsize * sizeof(float));
    cudaMalloc((void**)&d_sgnv, nSweeps * meshDim * sizeof(int));
    cudaMalloc((void**)&d_sgnt, nSweeps * meshDim * sizeof(int));

    cudaMemcpy(d_T, T, matsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, S, matsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sgnv, h_sgnv, nSweeps * meshDim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sgnt, h_sgnt, nSweeps * meshDim * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_sgnv;
    delete[] h_sgnt;

    // Fast sweeping loop
    for (int sweep = 0; sweep < nSweeps; sweep++) {
        int zd = (sweep == 2 || sweep == 3) ? -1 : 1;
        int xd = (sweep == 0 || sweep == 2) ? -1 : 1;
        int sgni = sweep;
        int sgnj = sweep + nSweeps;

        for (int level = 0; level < total_levels; level++) {
            int z_offset = (sweep == 0) ? ((level < nxx) ? 0 : level - nxx + 1) :
                            (sweep == 1) ? ((level < nzz) ? nzz - level - 1 : 0) :
                            (sweep == 2) ? ((level < nzz) ? level : nzz - 1) :
                                           ((level < nxx) ? nzz - 1 : nzz - 1 - (level - nxx + 1));

            int x_offset = (sweep == 0) ? ((level < nxx) ? level : nxx - 1) :
                            (sweep == 1) ? ((level < nzz) ? 0 : level - nzz + 1) :
                            (sweep == 2) ? ((level < nzz) ? nxx - 1 : nxx - 1 - (level - nzz + 1)) :
                                           ((level < nxx) ? nxx - level - 1 : 0);

            int n_elements = (level < min_level) ? level + 1 :
                             (level >= max_level) ? total_levels - level :
                             total_levels - min_level - max_level + level;

            int nBlocks = (n_elements + nThreads - 1) / nThreads;

            fsm_kernel<<<nBlocks, nThreads>>>(d_T, d_S, d_sgnv, d_sgnt, sgni, sgnj,
                                              x_offset, z_offset, xd, zd,
                                              nxx, nzz, dx, dz, dx2i, dz2i);

            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(T, d_T, matsize * sizeof(float), cudaMemcpyDeviceToHost);

    #pragma omp parallel for
    for (int index = 0; index < nx * nz; index++) {
        int x = index / nz;
        int z = index % nz;
        eikonal[z + x * nz] = T[(z + nb) + (x + nb) * nzz];
    }

    // 🧹 Free device memory
    cudaFree(d_T);
    cudaFree(d_S);
    cudaFree(d_sgnv);
    cudaFree(d_sgnt);

    // 🧹 Free host memory
    delete[] T;
    delete[] S;

    return eikonal;
}

void free_eikonal(float* ptr) {
    delete[] ptr;
}
}

extern "C" {

    // Kernel that performs on-the-fly migration for one output pixel (as shown previously)
    __global__ void _migrate_constant_velocity(const float* __restrict__ data,
                                                const float* __restrict__ cdp,
                                                const float* __restrict__ offsets,
                                                float v,
                                                float dt, float dx, float dz,
                                                int nsmp, int ntraces,
                                                int nx, int nz,
                                                float* __restrict__ R)
    {
        // Each thread computes one output pixel (ix, iz)
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iz = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix >= nx || iz >= nz) return;
    
        float x = ix * dx;
        float z = iz * dz;
        float sum = 0.0f;
        
        // Loop over traces
        #pragma unroll 9
        for (int j = 0; j < ntraces; j++) {
            float cdp_val = cdp[j];    // effective CDP for trace j
            float h = offsets[j] * 0.5f; // half offset for trace j
            // float doffset = fabsf(offsets[j+1]-offsets[j]);
            float source = cdp_val - h;
            float receiver = cdp_val + h;
            
            // Compute distances:
            float dxs = x - source;
            float dxg = x - receiver;
            float rs = sqrtf(dxs*dxs + z*z);
            float rr = sqrtf(dxg*dxg + z*z);
            float eps = 1e-10f;
            if (rs < eps) rs = eps;
            if (rr < eps) rr = eps;
            
            // Compute two-way travel time:
            float t_val = (rs + rr) / v;
            int it = (int) floorf(t_val / dt);
            if (it < 0 || it >= nsmp) continue;
            
            // Get seismic amplitude: data is stored as row-major (nsmp, ntraces)
            float amp = data[j * nsmp + it];
            
            // Compute weight (geometric spreading correction)
            float sqrt_rs_rr = sqrtf(rs/rr);
            float sqrt_rr_rs = 1.0f/sqrt_rs_rr;
            float weight = ((z/rs)*sqrt_rs_rr + (z/rr)*sqrt_rr_rs) / v;
            weight *= 0.3989422804f;  // 1/sqrt(2*pi)
            
            sum += amp * weight;
        }
        // Write the accumulated sum to the output migrated image R (assume row-major, shape (nx, nz))
        R[ix + iz * nx] = sum;
    }

    
    // Host-callable wrapper function for migration
    // This function allocates memory, sets up kernel launch parameters, and calls the kernel.
    void migrate_constant_velocity(const float* data, const float* cdp, const float* offsets,
                                   float v, float dt, float dx, float dz,
                                   int nsmp, int ntraces, int nx, int nz,
                                   float* R)
    {
        // Define block and grid dimensions.
        dim3 block(32, 8);
        dim3 grid((nx + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
        // dim3 grid((ntraces + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
        
        // Launch the kernel.
        _migrate_constant_velocity<<<grid, block>>>(data, cdp, offsets, v, dt, dx, dz,
                                                     nsmp, ntraces, nx, nz, R);
        cudaDeviceSynchronize();
    }

}




//----------------------------------------------------------------------------
// Helper structure and functions for mapping a file into pinned memory.
// A simple structure to hold mapped file info.

extern "C" void init_cuda_with_mapped_host() {
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDeviceFlags failed: " << cudaGetErrorString(err) << std::endl;
    }
}

// Function: Load binary file into heap-allocated buffer
float* load_binary_file(const char* filename, size_t* out_count)
{
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Get size in bytes and compute number of float elements
    fseek(file, 0, SEEK_END);
    size_t filesize = ftell(file);
    rewind(file);

    if (filesize % sizeof(float) != 0) {
        fprintf(stderr, "File size is not a multiple of float size\n");
        fclose(file);
        return NULL;
    }

    *out_count = filesize / sizeof(float);

    float* data = (float*)malloc(filesize);
    if (!data) {
        perror("malloc failed");
        fclose(file);
        return NULL;
    }

    size_t read_count = fread(data, sizeof(float), *out_count, file);
    fclose(file);

    if (read_count != *out_count) {
        fprintf(stderr, "fread failed: expected %zu floats, got %zu\n", *out_count, read_count);
        free(data);
        return NULL;
    }

    return data;
}

// Function: Copy data from host to device
float* copy_to_device(const float* host_data, size_t count)
{
    float* device_data = NULL;
    CHECK_CUDA(cudaMalloc((void**)&device_data, count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_data, host_data, count * sizeof(float), cudaMemcpyHostToDevice));
    return device_data;
}


void resample_field_into_npp_cubic(float* d_base,
                                   float* d_dst,
                                   size_t offset_index,
                                   int nx_coarse, int nz_coarse,
                                   int nx_fine, int nz_fine)
{
    // Compute pointer to the selected field
    size_t field_size = nx_coarse * nz_coarse;
    float* d_src = d_base + offset_index * field_size;

    // Set up NPP image sizes
    NppiSize src_size = { nx_coarse, nz_coarse };
    NppiSize dst_size = { nx_fine, nz_fine };

    // Define source and destination ROIs
    NppiRect src_roi = { 0, 0, nx_coarse, nz_coarse };
    NppiRect dst_roi = { 0, 0, nx_fine, nz_fine };

    int src_step = nx_coarse * sizeof(float);
    int dst_step = nx_fine * sizeof(float);

    // Perform the resize operation
    NppStatus status = nppiResize_32f_C1R(
        d_src, src_step, src_size, src_roi,
        d_dst, dst_step, dst_size, dst_roi,
        NPPI_INTER_CUBIC
    );

    if (status != NPP_SUCCESS) {
        fprintf(stderr, "nppiResize_32f_C1R failed with code: %d\n", status);
    }
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Host-callable migration function.
// This version re-reads (via pointer arithmetic) a traveltime field from a memory-mapped file
// for each update, resamples it to fine dimensions using NPP, and calls the migration kernel.
// Parameters:
//  - cdp, offsets, v, dt, dx, dz, nsmp, ntraces: parameters for migration.
//  - nx_coarse, nz_coarse: dimensions of each coarse traveltime field in the file.
//  - nx_fine, nz_fine: desired fine output dimensions.
//  - R: output migrated image (device pointer).
//  - traveltime_filename: path to the binary file with appended traveltime fields.
//  - num_fields: number of traveltime fields in the file.

template <typename T>
T* allocateGpuMemory(size_t count) {
    T* d_ptr = nullptr;
    // Allocate memory on the device.
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(T));
    if (err != cudaSuccess) {
        // Print error message and return nullptr in case of allocation error.
        fprintf(stderr, "CUDA error in cudaMalloc for %zu elements of size %zu: %s\n",
                count, sizeof(T), cudaGetErrorString(err));
        return nullptr;
    }
    return d_ptr;
}

__device__ void copy_coarse_traveltimes_to_shared_memory(const float* __restrict__ globalData, float* sharedTile1, float* sharedTile2, int s0, int s1,
    int tileDimX, int tileDimY, int gx, int gz, int nx, int nz, int ratio_x, int ratio_z)
{

    // The number of unique coarse samples to load is tileDimX * tileDimY.
    int numElements = tileDimX * tileDimY;

    // Compute a linear thread id within the block.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;

    // Loop over the unique indices in the tile with stride = totalThreads.
    for (int idx = tid; idx < numElements; idx += totalThreads)
    {
        // Compute the 2D tile coordinates:
        int local_x = (idx % tileDimX)/ratio_x;
        int local_y = (idx / tileDimX)/ratio_z;

        // Clamp indices to global boundaries.
        if (gx >= nx)
            gx = nx - 1;
        if (gz >= nz)
            gz = nz - 1;

        // Compute the flat index into the global data array for each shot plane.
        int idx1 = s0 * (nx * nz) + gz * nx + gx;
        int idx2 = s1 * (nx * nz) + gz * nx + gx;

        // Compute the local (shared) index.
        int localIdx = local_x + local_y * tileDimX;
        sharedTile1[localIdx] = globalData[idx1];
        sharedTile2[localIdx] = globalData[idx2];
    }

    
    
}

// Device function that performs some operation on a block of data in shared memory.
__device__ float bilinearInterpolate(const float V000, const float V001, const float V010, const float V011,
                                     const float V100, const float V101, const float V110, const float V111,
    float s0, float s1, float x0, float x1, float z0, float z1, float s_query,
float x_query, float z_query) {
    // For example, perform a bilinear interpolation on the four corners.
    // Assume the sharedTile is a 2D array of dimensions tileDimY x tileDimX stored in row-major order.
    // Using linear basis functions:
    //   B0(u) = 1 - u,  B1(u) = u.

    // # Compute the fractional distances along each axis.
    const float u_s = (s_query - s0) / (s1 - s0);
    const float u_z = (z_query - z0) / (z1 - z0);
    const float u_x = (x_query - x0) / (x1 - x0);

    // # For linear interpolation, the local basis functions are:
    // #   B0(u) = 1 - u   and   B1(u) = u
    const float Bs0 = 1.0f - u_s;
    const float Bs1 = u_s;
    const float Bz0 = 1.0f - u_z;
    const float Bz1 = u_z;
    const float Bx0 = 1.0f - u_x;
    const float Bx1 = u_x;
    
    // # Compute the weighted sum.
    const float interpValue =  (Bs0 * Bz0 * Bx0 * V000 +
                                Bs0 * Bz0 * Bx1 * V001 +
                                Bs0 * Bz1 * Bx0 * V010 +
                                Bs0 * Bz1 * Bx1 * V011 +
                                Bs1 * Bz0 * Bx0 * V100 +
                                Bs1 * Bz0 * Bx1 * V101 +
                                Bs1 * Bz1 * Bx0 * V110 +
                                Bs1 * Bz1 * Bx1 * V111);

    return interpValue;
}

__device__ float hermiteInterpolate(const float V000, const float V001, const float V010, const float V011,
                                      const float V100, const float V101, const float V110, const float V111,
                                      float s0, float s1, 
                                      float x0, float x1, 
                                      float z0, float z1, 
                                      float s_query, float x_query, float z_query,
                                      float d0, float d1)
{
    // Compute the normalized parameter t for the shot dimension.
    // t=0 corresponds to s0 and t=1 corresponds to s1.
    float t = (s_query - s0) / (s1 - s0);

    // Compute the standard cubic Hermite basis functions:
    // h00(t) = 2t^3 - 3t^2 + 1
    // h10(t) = t^3 - 2t^2 + t
    // h01(t) = -2t^3 + 3t^2
    // h11(t) = t^3 - t^2
    float t2 = t * t;
    float t3 = t2 * t;
    float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 = t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 = t3 - t2;
    
    // For the spatial (x,z) bilinear interpolation, compute the local fractional coordinates.
    float u_x = (x_query - x0) / (x1 - x0);
    float u_z = (z_query - z0) / (z1 - z0);
    
    // Compute the bilinear interpolation on the first shot plane:
    float T0 = ( (1.0f - u_z) * (1.0f - u_x) * V000 +
                 (1.0f - u_z) * (        u_x) * V001 +
                 (        u_z) * (1.0f - u_x) * V010 +
                 (        u_z) * (        u_x) * V011 );
    
    // Compute the bilinear interpolation on the second shot plane:
    float T1 = ( (1.0f - u_z) * (1.0f - u_x) * V100 +
                 (1.0f - u_z) * (        u_x) * V101 +
                 (        u_z) * (1.0f - u_x) * V110 +
                 (        u_z) * (        u_x) * V111 );
    
    // Now perform cubic Hermite interpolation in the shot direction.
    // The derivatives d0 and d1 must represent dT/ds at s0 and s1, respectively.
    float ds = s1 - s0; // cell length in shot dimension
    float T_val = h00 * T0 + h10 * ds * d0 + h01 * T1 + h11 * ds * d1;
    
    return T_val;
}

#define INTERP_HALO 1
#define BLOCK_X 32
#define BLOCK_Y 32

extern "C" {

    // Kernel that performs on-the-fly migration for one output pixel (as shown previously)
    __global__ void _migrate_variable_velocity(const float* __restrict__ data,
                                                const float* __restrict__ cdp,
                                                const float* __restrict__ offsets,
                                                const float* __restrict__ Traveltime,
                                                const float* __restrict__ Gradient,
                                                int src_index,
                                                int rec_index,
                                                float s0, float s1,
                                                float r0, float r1,
                                                float v,
                                                float dt, float dx_fine, float dz_fine,
                                                float dx_coarse, float dz_coarse,
                                                int nsmp, int ntraces, int init_traces,
                                                int nx_coarse, int nz_coarse,
                                                int nx_fine, int nz_fine,
                                                float* __restrict__ R,
                                                int tileDimX, int tileDimZ, int num_eikonals)
    {
        // Each thread computes one output pixel (ix, iz)
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iz = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix >= nx_fine - 1 || iz >= nz_fine - 1) return;

        // Allocate shared memory for two shot planes for this coarse cell tile.
        extern __shared__ float s_traveltime_src1[];
        extern __shared__ float s_traveltime_src2[];

        extern __shared__ float s_traveltime_rec1[];
        extern __shared__ float s_traveltime_rec2[];



        float x = ix * dx_fine;
        float z = iz * dz_fine;

        float lx_fine = threadIdx.x * dx_fine;
        float lz_fine = threadIdx.y * dz_fine;

        int l_ix_coarse = static_cast<int>(lx_fine/dx_coarse);
        int l_iz_coarse = static_cast<int>(lz_fine/dz_coarse);

        int ix_coarse = static_cast<int>(x/dx_coarse);
        int iz_coarse = static_cast<int>(z/dz_coarse);

        if (ix_coarse >= nx_coarse)
            ix_coarse = nx_coarse - 2;
        
        if (iz_coarse >= nz_coarse)
            iz_coarse = nz_coarse - 2;

        float coarse_x0 = ix_coarse * dx_coarse;
        float coarse_x1 = coarse_x0 + dx_coarse;

        float coarse_z0 = iz_coarse * dz_coarse;
        float coarse_z1 = coarse_z0 + dz_coarse;


        // copy_coarse_traveltimes_to_shared_memory(Traveltime, s_traveltime_src1, 
        //     s_traveltime_src2, src_index, src_index+1, tileDimX, tileDimZ, ix_coarse, iz_coarse, 
        //     nx_coarse, nz_coarse, nx_fine/nx_coarse, nz_fine/nz_coarse);

        // copy_coarse_traveltimes_to_shared_memory(Traveltime, s_traveltime_rec1, 
        //     s_traveltime_rec2, rec_index, rec_index+1, tileDimX, tileDimZ, ix_coarse, iz_coarse, 
        //     nx_coarse, nz_coarse, nx_fine/nx_coarse, nz_fine/nz_coarse);

        // __syncthreads();

        int s_offset = src_index * nx_coarse * nz_coarse;
        int s_offset_1 = (src_index+1) * nx_coarse * nz_coarse;

        int r_offset = rec_index * nx_coarse * nz_coarse;
        int r_offset_1 = (rec_index + 1) * nx_coarse * nz_coarse;

        float sum = 0.0f;
        float t_val;
        // Loop over traces
        #pragma unroll 9
        for (int j = init_traces; j < init_traces+ntraces; j++) {
            
            float cdp_val = cdp[j];    // effective CDP for trace j
            float h = offsets[j] * 0.5f; // half offset for trace j

            float source = cdp_val - h;
            float receiver = cdp_val + h;
            
            // Compute distances:
            float dxs = x - source;
            float dxg = x - receiver;
            float rs = sqrtf(dxs*dxs + z*z);
            float rr = sqrtf(dxg*dxg + z*z);
            float eps = 1e-10f;
            if (rs < eps) rs = eps;
            if (rr < eps) rr = eps;

            float s_traveltime = bilinearInterpolate(Traveltime[s_offset + ix_coarse + nx_coarse * iz_coarse],      Traveltime[s_offset + ix_coarse + 1 + nx_coarse * iz_coarse],
                                                     Traveltime[s_offset + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[s_offset + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],

                                                     Traveltime[s_offset_1 + ix_coarse + nx_coarse * iz_coarse],    Traveltime[s_offset_1 + ix_coarse + 1 + nx_coarse * iz_coarse],
                                                     Traveltime[s_offset_1 + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[s_offset_1 + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],
                                                    s0, s1, coarse_x0, coarse_x1, coarse_z0, coarse_z1, source, x, z);

            float r_traveltime = bilinearInterpolate(Traveltime[r_offset + ix_coarse + nx_coarse * iz_coarse],      Traveltime[r_offset + ix_coarse + 1 + nx_coarse * iz_coarse],
                                                     Traveltime[r_offset + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[r_offset + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],
                                                     
                                                     Traveltime[r_offset_1 + ix_coarse + nx_coarse * iz_coarse],    Traveltime[r_offset_1 + ix_coarse + 1 + nx_coarse * iz_coarse],
                                                     Traveltime[r_offset_1 + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[r_offset_1 + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],
                                                    s0, s1, coarse_x0, coarse_x1, coarse_z0, coarse_z1, source, x, z);

            // float s_traveltime = hermiteInterpolate(Traveltime[s_offset + ix_coarse + nx_coarse * iz_coarse],      Traveltime[s_offset + ix_coarse + 1 + nx_coarse * iz_coarse],
            //                                          Traveltime[s_offset + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[s_offset + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],

            //                                          Traveltime[s_offset_1 + ix_coarse + nx_coarse * iz_coarse],    Traveltime[s_offset_1 + ix_coarse + 1 + nx_coarse * iz_coarse],
            //                                          Traveltime[s_offset_1 + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[s_offset_1 + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],
            //                                         s0, s1, coarse_x0, coarse_x1, coarse_z0, coarse_z1, source, x, z,
            //                                         Gradient[s_offset + ix_coarse + nx_coarse * iz_coarse], Gradient[s_offset_1 + ix_coarse + nx_coarse * iz_coarse]);

            // float r_traveltime = hermiteInterpolate(Traveltime[r_offset + ix_coarse + nx_coarse * iz_coarse],      Traveltime[r_offset + ix_coarse + 1 + nx_coarse * iz_coarse],
            //                                          Traveltime[r_offset + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[r_offset + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],
                                                     
            //                                          Traveltime[r_offset_1 + ix_coarse + nx_coarse * iz_coarse],    Traveltime[r_offset_1 + ix_coarse + 1 + nx_coarse * iz_coarse],
            //                                          Traveltime[r_offset_1 + ix_coarse + nx_coarse * (iz_coarse+1)],  Traveltime[r_offset_1 + ix_coarse + 1 + nx_coarse * (iz_coarse + 1)],
            //                                         s0, s1, coarse_x0, coarse_x1, coarse_z0, coarse_z1, source, x, z, 
            //                                         Gradient[r_offset + ix_coarse + nx_coarse * iz_coarse], Gradient[r_offset_1 + ix_coarse + nx_coarse * iz_coarse]);



            // Compute two-way travel time:
            // float t_val = s_traveltime + r_traveltime;
            t_val = s_traveltime + r_traveltime;
            // t_val = (rs + rr) / v;
            int it = (int) floorf(t_val / dt);
            if (it < 0 || it >= nsmp) continue;
            
            // Get seismic amplitude: data is stored as row-major (nsmp, ntraces)
            float amp = data[j * nsmp + it];
            
            // Compute weight (geometric spreading correction)
            float sqrt_rs_rr = sqrtf(rs/rr);
            float sqrt_rr_rs = 1.0f/sqrt_rs_rr;
            float weight = ((z/rs)*sqrt_rs_rr + (z/rr)*sqrt_rr_rs) / v;
            weight *= 0.3989422804f;  // 1/sqrt(2*pi)

            sum += amp * weight;
        }
        // Write the accumulated sum to the output migrated image R (assume row-major, shape (nx, nz))
        R[ix + iz * nx_fine] += sum;
        // R[ix + iz * nx_fine] = t_val;
    }
}


extern "C" void migrate_variable_velocity(const float* data, const float* cdp, const float* offsets, 
                                            const float* eikonal_positions, const int* segments,
                                            float v, float dt, float dx_fine, float dz_fine, 
                                            float dx_coarse, float dz_coarse,
                                            int nsmp, int ntraces,
                                            int nx_coarse, int nz_coarse,
                                            int nx_fine, int nz_fine,
                                            float* R,
                                            const char* traveltime_filename,
                                            const char* gradient_filename,
                                            int num_segments,
                                            int num_eikonals)
{


    // Assume each coarse traveltime field has the same size:

    size_t num_floats = num_eikonals * nx_coarse * nz_coarse;

    float* h_travelTimeCoarse = load_binary_file(traveltime_filename, &num_floats);
    if (!h_travelTimeCoarse) {
        fprintf(stderr, "Failed to load file.\n");
        // return 1;
    }

    float* h_GradientCoarse = load_binary_file(gradient_filename, &num_floats);
    if (!h_GradientCoarse) {
        fprintf(stderr, "Failed to load file.\n");
        // return 1;
    }

    int x_ratio = nx_fine/nx_coarse;
    int z_ratio = nz_fine/nz_coarse;

    float *d_Data = copy_to_device(data, ntraces*nsmp);
    float *d_cdp = copy_to_device(cdp, ntraces);
    float *d_offsets = copy_to_device(offsets, ntraces);
    float *d_R = allocateGpuMemory<float>(nx_fine * nz_fine);
    float* d_TravelTimeCoarse = copy_to_device(h_travelTimeCoarse, num_floats);
    float* d_GradientCoarse = copy_to_device(h_GradientCoarse, num_floats);

    float firstEikonal = eikonal_positions[0];

    float eikonal_spacing = eikonal_positions[1] - eikonal_positions[0];

    size_t init_traces = 0;

    for (int field = 0; field < num_segments; field++) {

        int n_traces = segments[field];
        
        float sx = cdp[init_traces] - 0.5f * offsets[init_traces];
        float rx = cdp[init_traces] + 0.5f * offsets[init_traces];

        int src_index = static_cast<int>((sx-firstEikonal)/eikonal_spacing);
        int rec_index = static_cast<int>((rx-firstEikonal)/eikonal_spacing);

        int s0_index = src_index;
        int s1_index = src_index+1;

        if (s1_index >= num_eikonals)
        {
            s0_index=num_eikonals-2;
            s1_index=num_eikonals-1;
        }

        int r0_index = rec_index;
        int r1_index = rec_index+1;

        if (r1_index >= num_eikonals)
        {
            r0_index=num_eikonals-2;
            r1_index=num_eikonals-1;
        }

        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((nx_fine + block.x - 1) / block.x, (nz_fine + block.y - 1) / block.y);

        // Suppose at runtime you calculate:
        int tileDimX = BLOCK_X/x_ratio + 1;
        int tileDimZ = BLOCK_Y/z_ratio + 1;
        int sharedMemSize = tileDimX * tileDimZ * sizeof(float);
        
        // Launch the kernel.
        _migrate_variable_velocity<<<grid, block, sharedMemSize>>>(d_Data, d_cdp, d_offsets, d_TravelTimeCoarse, d_GradientCoarse,
                                                    s0_index, r0_index, 
                                                    eikonal_positions[s0_index],eikonal_positions[s1_index],
                                                    eikonal_positions[r0_index],eikonal_positions[r1_index],
                                                    v, dt, dx_fine, dz_fine,
                                                    dx_coarse, dz_coarse,
                                                     nsmp, n_traces, init_traces, 
                                                     nx_coarse, nz_coarse,
                                                     nx_fine, nz_fine, d_R, 
                                                     tileDimX, tileDimZ, num_eikonals);
        cudaDeviceSynchronize();
        init_traces += n_traces;
    }

    CHECK_CUDA(cudaMemcpy(R, d_R, nx_fine * nz_fine * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_TravelTimeCoarse);
    cudaFree(d_offsets);
    cudaFree(d_cdp);
    cudaFree(d_R);
    cudaFree(d_Data);
    free(h_travelTimeCoarse);
    free(h_GradientCoarse);
}



