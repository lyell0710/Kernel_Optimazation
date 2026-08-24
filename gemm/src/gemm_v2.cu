#include <mma.h>
#include "gemm_common.h"
// v2 · wmma(Tensor Core 入门形):64x64 块,4 warp 各算 32x32(2x2 个
// 16x16x16 fragment)。同步 load → mma_sync。此级把算力路径从 CUDA core
// 换到 Tensor Core——对比 v1 看"指令世代"的贡献,访存策略先保持朴素。
using namespace nvcuda;
constexpr int BM = 64, BN = 64, BK = 32;

__global__ void gemm_v2_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[BM][BK], Bs[BK][BN];
    const int warp_id = threadIdx.x / 32;
    const int wr = warp_id / 2, wc = warp_id % 2;      // warp 的 2x2 排布
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 128 线程协作装载:A 64x32、B 32x64,各 2048 half → 每线程 16 half
        for (int t = threadIdx.x; t < BM * BK / 8; t += blockDim.x) {
            int r = (t * 8) / BK, c = (t * 8) % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(bm + r) * K + k0 + c]);
        }
        for (int t = threadIdx.x; t < BK * BN / 8; t += blockDim.x) {
            int r = (t * 8) / BN, c = (t * 8) % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + bn + c]);
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wr * 32 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float* cp = reinterpret_cast<float*>(As);   // 复用 smem 不必要,直写全局
            (void)cp;
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> h;
            #pragma unroll
            for (int e = 0; e < h.num_elements; ++e)
                h.x[e] = __float2half(acc[i][j].x[e]);
            wmma::store_matrix_sync(
                &C[(bm + wr * 32 + i * 16) * N + bn + wc * 32 + j * 16],
                h, N, wmma::mem_row_major);
        }
}
void gemm_v2(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(128), grd(N / BN, M / BM);
    gemm_v2_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
