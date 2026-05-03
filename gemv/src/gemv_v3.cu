#include "gemv_common.h"

namespace
{
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__inline__ __device__ float warp_reduce_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// 一行交给一个warp，减少block级同步开销。
__global__ void gemv_v3_kernel(const float* __restrict__ mat,
                               const float* __restrict__ vec,
                               float* __restrict__ out,
                               int rows,
                               int cols)
{
    int lane = threadIdx.x;
    int warp_id_in_block = threadIdx.y;
    int row = blockIdx.x * blockDim.y + warp_id_in_block;
    if (row >= rows)
    {
        return;
    }

    const int row_offset = row * cols;
    float sum = 0.0f;
    for (int c = lane; c < cols; c += kWarpSize)
    {
        sum += mat[row_offset + c] * vec[c];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0)
    {
        out[row] = sum;
    }
}
} // namespace

void gemv_v3(const float* mat, const float* vec, float* out, int rows, int cols)
{
    dim3 block(kWarpSize, kWarpsPerBlock);
    dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    gemv_v3_kernel<<<grid, block>>>(mat, vec, out, rows, cols);
}
