#include "gemv_common.h"

namespace
{
constexpr int kBlockSize = 128;
constexpr int kRowsPerBlock = 4;

template <int blockSize, int rowsPerBlock>
__global__ void gemv_v4_kernel(const float* __restrict__ mat,
                               const float* __restrict__ vec,
                               float* __restrict__ out,
                               int rows,
                               int cols)
{
    __shared__ float smem_x[blockSize];
    __shared__ float smem_partial[rowsPerBlock][blockSize];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * rowsPerBlock + ty;
    if (row >= rows)
    {
        return;
    }

    float local = 0.0f;
    int row_offset = row * cols;

    // 同一个block中的多个行共享x tile，降低对向量x的重复全局读取。
    for (int base = 0; base < cols; base += blockSize)
    {
        int c = base + tx;
        smem_x[tx] = (c < cols) ? vec[c] : 0.0f;
        __syncthreads();

        if (c < cols)
        {
            local += mat[row_offset + c] * smem_x[tx];
        }
        __syncthreads();
    }

    smem_partial[ty][tx] = local;
    __syncthreads();

    for (int s = blockSize >> 1; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            smem_partial[ty][tx] += smem_partial[ty][tx + s];
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        out[row] = smem_partial[ty][0];
    }
}
} // namespace

void gemv_v4(const float* mat, const float* vec, float* out, int rows, int cols)
{
    dim3 block(kBlockSize, kRowsPerBlock);
    dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_v4_kernel<kBlockSize, kRowsPerBlock><<<grid, block>>>(mat, vec, out, rows, cols);
}
