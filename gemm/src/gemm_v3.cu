#include <mma.h>
#include <cuda_pipeline.h>
#include "gemm_common.h"
// v3 · v2 + cp.async 双缓冲:计算 buf[p] 的同时 DMA 预取下一 K 块进
// buf[p^1]——第 5 课"中下"的正主。与 Triton num_stages=2 同一件事的手写形。
using namespace nvcuda;
constexpr int BM = 64, BN = 64, BK = 32;

__device__ __forceinline__ void load_tile_async(
    half (*As)[BK], half (*Bs)[BN],
    const half* A, const half* B, int M, int N, int K,
    int bm, int bn, int k0, int tid, int nthr) {
    for (int t = tid; t < BM * BK / 8; t += nthr) {
        int r = (t * 8) / BK, c = (t * 8) % BK;
        __pipeline_memcpy_async(&As[r][c], &A[(bm + r) * K + k0 + c], 16);
    }
    for (int t = tid; t < BK * BN / 8; t += nthr) {
        int r = (t * 8) / BN, c = (t * 8) % BN;
        __pipeline_memcpy_async(&Bs[r][c], &B[(k0 + r) * N + bn + c], 16);
    }
    __pipeline_commit();
}

__global__ void gemm_v3_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[2][BM][BK], Bs[2][BK][BN];
    const int warp_id = threadIdx.x / 32;
    const int wr = warp_id / 2, wc = warp_id % 2;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    load_tile_async(As[0], Bs[0], A, B, M, N, K, bm, bn, 0,
                    threadIdx.x, blockDim.x);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)                       // 预取下一块(与计算重叠的部分)
            load_tile_async(As[p ^ 1], Bs[p ^ 1], A, B, M, N, K,
                            bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);   // 只等当前块就绪
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[p][wr * 32 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[p][kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
        p ^= 1;
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> h;
            #pragma unroll
            for (int e = 0; e < h.num_elements; ++e)
                h.x[e] = __float2half(acc[i][j].x[e]);
            wmma::store_matrix_sync(
                &C[(bm + wr * 32 + i * 16) * N + bn + wc * 32 + j * 16],
                h, N, wmma::mem_row_major);
        }
}
void gemm_v3(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(128), grd(N / BN, M / BM);
    gemm_v3_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
