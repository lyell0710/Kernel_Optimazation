#include "reduce_common.h"
#include <cuda_runtime.h>
#include <iostream>

namespace
{

constexpr int kBlockSize = 256;
// v5: block 内归约循环展开（基于 v4 改写）
// 目标：减少 for 循环控制开销，保留 v4 的两元素加载与边界保护逻辑。
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

template <int blockSize> __global__ void reduce_v6_kernel(const float* d_in, float* d_out, int n)
{
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    int total_threads = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // v6 每个线程循环跨步读多个元素
    for (int i = gtid; i < n; i += total_threads)
    {
        sum += d_in[i];
    }

    smem[tid] = sum;

    __syncthreads();

    // v5 核心：展开版 block shared-memory reduction
    BlockSharedMemReduce<blockSize>(smem, tid);
    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

} // namespace

void reduce_v6(const float* data, float* output, int n)
{
    if (n <= 0)
    {
        cudaMemset(output, 0, sizeof(float));
        return;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    int grid1 = (n + kBlockSize - 1) / kBlockSize;
    int max_grid = prop.multiProcessorCount * 8;
    if (max_grid < 1)
    {
        max_grid = 1;
    }
    if (grid1 > max_grid)
    {
        grid1 = max_grid;
    }
    if (grid1 < 1)
    {
        grid1 = 1;
    }

    float* d_partial = nullptr;
    cudaMalloc(&d_partial, grid1 * sizeof(float));

    // pass1: N -> grid1 个 partial sums
    reduce_v6_kernel<kBlockSize><<<grid1, kBlockSize>>>(data, d_partial, n);

    // pass2: grid1 -> 1
    if (grid1 > 1)
    {
        reduce_v6_kernel<kBlockSize><<<1, kBlockSize>>>(d_partial, output, grid1);
    }
    else
    {
        cudaMemcpy(output, d_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_partial);
}