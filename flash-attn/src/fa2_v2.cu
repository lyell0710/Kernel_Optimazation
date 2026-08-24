#include <mma.h>
#include "fa_common.h"
// v2 · Tensor Core 版:QK^T 与 P·V 走 wmma,softmax/在线统计走 smem 标量段。
// 为什么 S/P 要 smem 往返:wmma accumulator fragment 的 lane→元素映射
// 未定义(编译器私有),行级 max/exp/α 无法在 fragment 上做——生产核
// (FA2/CUTLASS)用 mma PTX 拿到确定布局后在寄存器里做,这是 wmma→mma
// 的本质分界(与 gemm/ 记录 EXP-K02 §7 的 v5 backlog 同源)。
// 约束:D=128,S % 64 == 0(bench 形状均满足;通用尾块见 v0/v1)。
using namespace nvcuda;
constexpr int BM = 64, BN = 64, WARPS = 4;

// 动态 smem 分区(字节偏移,均 16B 对齐;合计 92928B,需 opt-in)
constexpr int OFF_O = 0;                       // float [64][128] 32768
constexpr int OFF_S = 32768;                   // float [64][68]  17408
constexpr int OFF_ML = 50176;                  // float m[64] l[64] a[64] 768
constexpr int OFF_K = 50944;                   // half  [64][128] 16384
constexpr int OFF_V = 67328;                   // half  [64][128] 16384
constexpr int OFF_P = 83712;                   // half  [64][72]   9216
constexpr int SMEM_BYTES = 92928;
constexpr int LDS = 68, LDP = 72;

__global__ void fa2_v2_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    extern __shared__ char smem[];
    float* Osm = (float*)(smem + OFF_O);
    float* Ssm = (float*)(smem + OFF_S);
    float* m_s = (float*)(smem + OFF_ML);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;
    half* Ks = (half*)(smem + OFF_K);
    half* Vs = (half*)(smem + OFF_V);
    half* Psm = (half*)(smem + OFF_P);

    const int tid = threadIdx.x, warp = tid / 32;
    const int h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (Hq / Hkv);
    const int q0 = blockIdx.x * BM;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* q = Q + ((size_t)(b * Hq + h) * S + q0) * FA_D;

    for (int i = tid; i < BM * FA_D; i += blockDim.x) Osm[i] = 0.f;
    if (tid < BM) { m_s[tid] = -1e30f; l_s[tid] = 0.f; }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[8];
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk)                     // Q 条带常驻寄存器
        wmma::load_matrix_sync(af[kk], q + (warp * 16) * FA_D + kk * 16, FA_D);

    const float scale = rsqrtf((float)FA_D);
    const int nlimit = causal ? min(q0 + BM, S) : S;

    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();
        for (int t = tid; t < BN * FA_D / 8; t += blockDim.x) {
            int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
            *(float4*)&Ks[r * FA_D + c] = *(const float4*)&k[(size_t)(n0 + r) * FA_D + c];
            *(float4*)&Vs[r * FA_D + c] = *(const float4*)&v[(size_t)(n0 + r) * FA_D + c];
        }
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 4; ++n) {                  // S 条带 16x64
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
            wmma::fill_fragment(sc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::col_major> bf;    // K^T 块 = K 行作列
                wmma::load_matrix_sync(bf, &Ks[(n * 16) * FA_D + kk * 16], FA_D);
                wmma::mma_sync(sc, af[kk], bf, sc);
            }
            wmma::store_matrix_sync(&Ssm[(warp * 16) * LDS + n * 16], sc,
                                    LDS, wmma::mem_row_major);
        }
        __syncthreads();
        if (tid < BM) {                                // 行级在线 softmax
            const int row = q0 + tid;
            const int jend = min(BN, (causal ? row + 1 : S) - n0);
            float rmax = -1e30f;
            for (int j = 0; j < jend; ++j)
                rmax = fmaxf(rmax, Ssm[tid * LDS + j] * scale);
            const float mn = fmaxf(m_s[tid], rmax);
            const float alpha = __expf(m_s[tid] - mn);
            float sum = 0.f;
            for (int j = 0; j < BN; ++j) {
                float p = j < jend ? __expf(Ssm[tid * LDS + j] * scale - mn) : 0.f;
                Psm[tid * LDP + j] = __float2half(p);
                sum += p;
            }
            l_s[tid] = l_s[tid] * alpha + sum;
            m_s[tid] = mn; a_s[tid] = alpha;
        }
        __syncthreads();
        for (int i = tid; i < BM * FA_D; i += blockDim.x)   // O *= α(行)
            Osm[i] *= a_s[i / FA_D];
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 8; ++c) {                  // O 条带 += P·V
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv, oacc;
            wmma::fill_fragment(pv, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                               wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(pf, &Psm[(warp * 16) * LDP + kk * 16], LDP);
                wmma::load_matrix_sync(vf, &Vs[(kk * 16) * FA_D + c * 16], FA_D);
                wmma::mma_sync(pv, pf, vf, pv);
            }
            float* optr = &Osm[(warp * 16) * FA_D + c * 16];
            wmma::load_matrix_sync(oacc, optr, FA_D, wmma::mem_row_major);
            #pragma unroll
            for (int e = 0; e < pv.num_elements; ++e) pv.x[e] += oacc.x[e];
            wmma::store_matrix_sync(optr, pv, FA_D, wmma::mem_row_major);
        }
    }
    __syncthreads();
    half* o = O + ((size_t)(b * Hq + h) * S + q0) * FA_D;
    for (int i = tid; i < BM * FA_D; i += blockDim.x)
        o[i] = __float2half(Osm[i] / l_s[i / FA_D]);
}

void fa2_v2(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(fa2_v2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v2_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
