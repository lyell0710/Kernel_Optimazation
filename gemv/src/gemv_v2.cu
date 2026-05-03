#include "gemv_common.h"

namespace
{
constexpr int kBlockSize = 256;

template <int blockSize>
__global__ void gemv_v2_kernel(const float* __restrict__ mat,
                               const float* __restrict__ vec,
                               float* __restrict__ out,
                               int rows,
                               int cols)
{
    __shared__ float smem[blockSize];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    // 向量化读 float4 提升内存访问效率，尾部不足4元素时走标量回退。
    float sum = 0.0f;
    const int row_offset = row * cols;
    for (int c = tid * 4; c < cols; c += blockSize * 4)
    {
        if (c + 3 < cols)
        {
            float4 a = reinterpret_cast<const float4*>(mat + row_offset + c)[0];
            float4 x = reinterpret_cast<const float4*>(vec + c)[0];
            sum += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
        }
        else
        {
            for (int k = 0; k < 4 && c + k < cols; ++k)
            {
                sum += mat[row_offset + c + k] * vec[c + k];
            }
        }
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockSize >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        out[row] = smem[0];
    }
}
} // namespace

void gemv_v2(const float* mat, const float* vec, float* out, int rows, int cols)
{
    gemv_v2_kernel<kBlockSize><<<rows, kBlockSize>>>(mat, vec, out, rows, cols);
}
