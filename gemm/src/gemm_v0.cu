#include "gemm_common.h"
// v0 · naive:一线程一输出。作用=正确性锚 + 讲"为什么要 tile"的账。
__global__ void gemm_v0_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    C[row * N + col] = __float2half(acc);
}
void gemm_v0(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
    gemm_v0_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
