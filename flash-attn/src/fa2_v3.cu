#include <mma.h>
#include "fa_common.h"
// v3 · 并行度翻倍:8 warp(行条带×列半区 4×2),softmax 每行 2 线程,
// 全部标量段 256 线程摊。对 v2 的变量只有并行组织,算法与 smem 布局不变。
using namespace nvcuda;
constexpr int BM = 64, BN = 64, WARPS = 8;

constexpr int OFF_O = 0;
constexpr int OFF_S = 32768;
constexpr int OFF_ML = 50176;
constexpr int OFF_K = 50944;
constexpr int OFF_V = 67328;
constexpr int OFF_P = 83712;
constexpr int SMEM_BYTES = 92928;
constexpr int LDS = 68, LDP = 72;

__global__ void fa2_v3_kernel(const half* Q, const half* K, const half* V,
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
    const int wr = warp / 2, wc = warp % 2;        // 行条带 / 列半区
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
    for (int kk = 0; kk < 8; ++kk)
        wmma::load_matrix_sync(af[kk], q + (wr * 16) * FA_D + kk * 16, FA_D);

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
        for (int n = 0; n < 2; ++n) {              // 本 warp 的 16x32 半区
            const int nc = wc * 2 + n;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
            wmma::fill_fragment(sc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &Ks[(nc * 16) * FA_D + kk * 16], FA_D);
                wmma::mma_sync(sc, af[kk], bf, sc);
            }
            wmma::store_matrix_sync(&Ssm[(wr * 16) * LDS + nc * 16], sc,
                                    LDS, wmma::mem_row_major);
        }
        __syncthreads();
        if (tid < 2 * BM) {                        // 每行 2 线程(同 warp 相邻 lane)
            const int row = tid / 2, hf = tid % 2;
            const int grow = q0 + row;
            const int jend = min(BN, (causal ? grow + 1 : S) - n0);
            const int j0 = hf * 32, j1 = min(j0 + 32, jend);
            float rmax = -1e30f;
            for (int j = j0; j < j1; ++j)
                rmax = fmaxf(rmax, Ssm[row * LDS + j] * scale);
            rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));
            const float mn = fmaxf(m_s[row], rmax);
            const float alpha = __expf(m_s[row] - mn);
            float sum = 0.f;
            for (int j = j0; j < j0 + 32; ++j) {
                float p = j < jend ? __expf(Ssm[row * LDS + j] * scale - mn) : 0.f;
                Psm[row * LDP + j] = __float2half(p);
                sum += p;
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            if (hf == 0) {
                l_s[row] = l_s[row] * alpha + sum;
                m_s[row] = mn; a_s[row] = alpha;
            }
        }
        __syncthreads();
        for (int i = tid; i < BM * FA_D; i += blockDim.x)
            Osm[i] *= a_s[i / FA_D];
        __syncthreads();
        #pragma unroll
        for (int c0 = 0; c0 < 4; ++c0) {           // 本 warp 的 4 个 16 列块
            const int c = wc * 4 + c0;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv, oacc;
            wmma::fill_fragment(pv, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                               wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(pf, &Psm[(wr * 16) * LDP + kk * 16], LDP);
                wmma::load_matrix_sync(vf, &Vs[(kk * 16) * FA_D + c * 16], FA_D);
                wmma::mma_sync(pv, pf, vf, pv);
            }
            float* optr = &Osm[(wr * 16) * FA_D + c * 16];
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

void fa2_v3(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(fa2_v3_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v3_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
