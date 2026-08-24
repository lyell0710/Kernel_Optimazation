#include <mma.h>
#include <cuda_pipeline.h>
#include "gemm_common.h"
// v4 · 大 tile(128x128, 8 warp, 每 warp 64x32=4x2 fragment)+ 双缓冲:
// 提高每字节 smem 的复用次数(算术强度),对标 Triton BM128/BN128 配置。
using namespace nvcuda;
constexpr int BM = 128, BN = 128, BK = 32;

__device__ __forceinline__ void load_tile_async4(
    half (*As)[BK], half (*Bs)[BN],
    const half* A, const half* B, int N, int K,
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

__global__ void gemm_v4_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[2][BM][BK], Bs[2][BK][BN];   // 2*(8+8)KB = 32KB
    const int warp_id = threadIdx.x / 32;           // 8 warp:2 行 x 4 列
    const int wr = warp_id / 4, wc = warp_id % 4;   // warp tile 64x32
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    load_tile_async4(As[0], Bs[0], A, B, N, K, bm, bn, 0,
                     threadIdx.x, blockDim.x);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)
            load_tile_async4(As[p ^ 1], Bs[p ^ 1], A, B, N, K,
                             bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(af[i], &As[p][wr * 64 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[p][kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
        p ^= 1;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> h;
            #pragma unroll
            for (int e = 0; e < h.num_elements; ++e)
                h.x[e] = __float2half(acc[i][j].x[e]);
            wmma::store_matrix_sync(
                &C[(bm + wr * 64 + i * 16) * N + bn + wc * 32 + j * 16],
                h, N, wmma::mem_row_major);
        }
}
void gemm_v4(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(256), grd(N / BN, M / BM);
    gemm_v4_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
