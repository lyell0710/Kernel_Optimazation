#include "fa_common.h"
// v1 · K/V tile 进 smem:block=4 warp 管 4 行 q,64 键一批协同搬进 smem
// 再消费。对 v0 的唯一变量 = K/V 读取层级(L2 → smem),收益大小本身
// 就是「L2 已经把广播读扛住多少」的量度。
constexpr int BN = 64, WARPS = 4;

__global__ void fa2_v1_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    __shared__ half Ks[BN][FA_D], Vs[BN][FA_D];   // 16KB + 16KB
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + warp;
    const int h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (Hq / Hkv);
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;

    float qr[4] = {0, 0, 0, 0};
    if (row < S) {
        const half* q = Q + ((size_t)(b * Hq + h) * S + row) * FA_D;
        #pragma unroll
        for (int j = 0; j < 4; ++j) qr[j] = __half2float(q[lane * 4 + j]);
    }
    const float scale = rsqrtf((float)FA_D);
    float m = -1e30f, l = 0.f, acc[4] = {0, 0, 0, 0};
    const int rmax = blockIdx.x * WARPS + WARPS - 1;      // block 内最大行
    const int nlimit = causal ? min(rmax + 1, S) : S;

    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();
        for (int t = threadIdx.x; t < BN * FA_D / 8; t += blockDim.x) {
            int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
            if (n0 + r < S) {
                *(float4*)&Ks[r][c] = *(const float4*)&k[(size_t)(n0 + r) * FA_D + c];
                *(float4*)&Vs[r][c] = *(const float4*)&v[(size_t)(n0 + r) * FA_D + c];
            }
        }
        __syncthreads();
        if (row >= S) continue;
        const int jend = min(BN, (causal ? row + 1 : S) - n0);
        for (int j = 0; j < jend; ++j) {
            float s = 0;
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                s += qr[d] * __half2float(Ks[j][lane * 4 + d]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xffffffff, s, off);
            s *= scale;
            float mn = fmaxf(m, s), alpha = __expf(m - mn), p = __expf(s - mn);
            l = l * alpha + p;
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                acc[d] = acc[d] * alpha + p * __half2float(Vs[j][lane * 4 + d]);
            m = mn;
        }
    }
    if (row < S) {
        half* o = O + ((size_t)(b * Hq + h) * S + row) * FA_D;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[lane * 4 + j] = __float2half(acc[j] / l);
    }
}
void fa2_v1(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    fa2_v1_kernel<<<dim3((S + WARPS - 1) / WARPS, Hq, B), WARPS * 32>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
