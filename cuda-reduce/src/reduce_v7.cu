#include "reduce_common.h"
#include <cuda_runtime.h>
#include <iostream>

namespace
{

constexpr int kBlockSize = 256;

template <int blockSize> __device__ void BlockSharedMemReduce(float* seme, int tid)
{
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            seme[tid] += seme[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            seme[tid] += seme[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            seme[tid] += seme[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            seme[tid] += seme[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        volatile float* vshm = seme;
        if (blockDim.x >= 64)
        {
            vshm[tid] += vshm[tid + 32];
        }
        vshm[tid] += vshm[tid + 16];
        vshm[tid] += vshm[tid + 8];
        vshm[tid] += vshm[tid + 4];
        vshm[tid] += vshm[tid + 2];
        vshm[tid] += vshm[tid + 1];
    }
}

template <int blockSize> __global__ void reduce_v7_kernel(const float* d_in, float* d_out, int n)
{
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = gtid; i < n; i += total_threads)
    {
        sum += d_in[i];
    }

    smem[tid] = sum;
    __syncthreads();

    BlockSharedMemReduce<blockSize>(smem, tid);
    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

} // namespace

void reduce_v7(const float* data, float* output, int n)
{
    if (n <= 0)
    {
        cudaMemset(output, 0, sizeof(float));
        return;
    }

    // Cache device-derived launch configuration once.
    static int s_max_grid = 0;
    if (s_max_grid == 0)
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        s_max_grid = prop.multiProcessorCount * 8;
        if (s_max_grid < 1)
        {
            s_max_grid = 1;
        }
    }

    // Reuse temporary buffer across calls to avoid per-iteration malloc/free.
    static float* s_partial = nullptr;
    static int s_partial_capacity = 0;

    int grid1 = (n + kBlockSize - 1) / kBlockSize;
    if (grid1 > s_max_grid)
    {
        grid1 = s_max_grid;
    }
    if (grid1 < 1)
    {
        grid1 = 1;
    }

    if (grid1 > s_partial_capacity)
    {
        if (s_partial != nullptr)
        {
            cudaFree(s_partial);
        }
        cudaMalloc(&s_partial, grid1 * sizeof(float));
        s_partial_capacity = grid1;
    }

    reduce_v7_kernel<kBlockSize><<<grid1, kBlockSize>>>(data, s_partial, n);

    if (grid1 > 1)
    {
        reduce_v7_kernel<kBlockSize><<<1, kBlockSize>>>(s_partial, output, grid1);
    }
    else
    {
        cudaMemcpy(output, s_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    }
}
