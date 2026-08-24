#include "fa_common.h"
// v0 · warp-per-query-row:一个 warp 负责一行 q,顺序流过全部 K/V。
// 在线 softmax 三件套(m/l/α)最直白的形态;K/V 复用完全交给 L2。
__global__ void fa2_v0_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    const int row = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x;
    const int kvh = h / (Hq / Hkv);
    const half* q = Q + ((size_t)(b * Hq + h) * S + row) * FA_D;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;

    float qr[4];                                  // 每 lane 持 4 维
    #pragma unroll
    for (int j = 0; j < 4; ++j) qr[j] = __half2float(q[lane * 4 + j]);
    const float scale = rsqrtf((float)FA_D);
    float m = -1e30f, l = 0.f, acc[4] = {0, 0, 0, 0};

    const int nend = causal ? row + 1 : S;
    for (int n = 0; n < nend; ++n) {
        float s = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            s += qr[j] * __half2float(k[(size_t)n * FA_D + lane * 4 + j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)    // warp 内规约 → 全 lane 同值
            s += __shfl_xor_sync(0xffffffff, s, off);
        s *= scale;
        float mn = fmaxf(m, s), alpha = __expf(m - mn), p = __expf(s - mn);
        l = l * alpha + p;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[j] = acc[j] * alpha
                     + p * __half2float(v[(size_t)n * FA_D + lane * 4 + j]);
        m = mn;
    }
    half* o = O + ((size_t)(b * Hq + h) * S + row) * FA_D;
    #pragma unroll
    for (int j = 0; j < 4; ++j) o[lane * 4 + j] = __float2half(acc[j] / l);
}
void fa2_v0(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    fa2_v0_kernel<<<dim3(S, Hq, B), 32>>>(Q, K, V, O, Hq, Hkv, S, causal);
}
