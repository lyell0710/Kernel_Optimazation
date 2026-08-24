#include "gemm_common.h"
// v1 · shared memory 32x32 tile:K 维分块驻留片上,全局访存 /32。
// 仍是标量 FMA(CUDA core)——本级的意义是把"访存问题"和"算力问题"分开。
constexpr int T = 32;
__global__ void gemm_v1_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[T][T], Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k)
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();
    }
    C[row * N + col] = __float2half(acc);
}
void gemm_v1(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(T, T), grd(N / T, M / T);
    gemm_v1_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
