#include "fa_common.h"
// fp32 精算参考:warp-per-row,两遍法(先全行 max,再 exp 求和),
// 与被测的在线单遍法算法路径独立,专用于正确性 gate。
__global__ void ref_kernel(const half* Q, const half* K, const half* V,
                           half* O, int Hq, int Hkv, int S, bool causal) {
    const int row = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x;
    const int kvh = h / (Hq / Hkv);
    const half* q = Q + ((size_t)(b * Hq + h) * S + row) * FA_D;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;
    float qr[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) qr[j] = __half2float(q[lane * 4 + j]);
    const float scale = rsqrtf((float)FA_D);
    const int nend = causal ? row + 1 : S;
    float m = -1e30f;
    for (int pass = 0; pass < 2; ++pass) {
        float l = 0.f, acc[4] = {0, 0, 0, 0};
        for (int n = 0; n < nend; ++n) {
            float s = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                s += qr[j] * __half2float(k[(size_t)n * FA_D + lane * 4 + j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xffffffff, s, off);
            s *= scale;
            if (pass == 0) { m = fmaxf(m, s); continue; }
            float p = __expf(s - m);
            l += p;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[j] += p * __half2float(v[(size_t)n * FA_D + lane * 4 + j]);
        }
        if (pass == 1) {
            half* o = O + ((size_t)(b * Hq + h) * S + row) * FA_D;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                o[lane * 4 + j] = __float2half(acc[j] / l);
        }
    }
}
void attn_ref_fp32(const half* Q, const half* K, const half* V, half* O,
                   int B, int Hq, int Hkv, int S, bool causal) {
    ref_kernel<<<dim3(S, Hq, B), 32>>>(Q, K, V, O, Hq, Hkv, S, causal);
}
