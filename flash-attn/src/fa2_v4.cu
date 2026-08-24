#include <mma.h>
#include <cuda_pipeline.h>
#include "fa_common.h"
// v4 · 访存重叠:①S/P 统一为一块 half 缓冲(S 写出即 fp16,exp 原位改写,
// 省 17KB smem 与一半 softmax 读流量;精度代价见 bench err 列);
// ②K 双缓冲 cp.async——QK^T(t) 进行时 K(t+1) 已在途,softmax 段到位;
// ③V(t) 与 QK^T(t) 重叠(V 只在 PV 段用)。组织结构继承 v3(8 warp)。
using namespace nvcuda;
constexpr int BM = 64, BN = 64, WARPS = 8;

constexpr int OFF_O = 0;                       // float [64][128] 32768
constexpr int OFF_ML = 32768;                  // m/l/a  768
constexpr int OFF_SP = 33536;                  // half [64][72]  9216(S 与 P 同址)
constexpr int OFF_K = 42752;                   // half [2][64][128] 32768
constexpr int OFF_V = 75520;                   // half [64][128] 16384
constexpr int SMEM_BYTES = 91904;
constexpr int LDSP = 72;

__device__ __forceinline__ void async_tile(half* dst, const half* src,
                                           int n0, int S, int tid, int nthr) {
    for (int t = tid; t < BN * FA_D / 8; t += nthr) {
        int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
        __pipeline_memcpy_async(&dst[r * FA_D + c],
                                &src[(size_t)(n0 + r) * FA_D + c], 16);
    }
    __pipeline_commit();
}

__global__ void fa2_v4_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    extern __shared__ char smem[];
    float* Osm = (float*)(smem + OFF_O);
    float* m_s = (float*)(smem + OFF_ML);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;
    half* SP = (half*)(smem + OFF_SP);
    half* Ks = (half*)(smem + OFF_K);              // [2] 双缓冲
    half* Vs = (half*)(smem + OFF_V);

    const int tid = threadIdx.x, warp = tid / 32;
    const int wr = warp / 2, wc = warp % 2;
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

    async_tile(Ks, k, 0, S, tid, blockDim.x);      // 预载 K(0)
    int p = 0;
    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();                           // PV(t-1) 已消费 Vs
        async_tile(Vs, v, n0, S, tid, blockDim.x); // V(t) 与 QK^T(t) 重叠
        __pipeline_wait_prior(1);                  // K(t) 到位(更老的组)
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int nc = wc * 2 + n;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
            wmma::fill_fragment(sc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &Ks[p * BN * FA_D
                                               + (nc * 16) * FA_D + kk * 16], FA_D);
                wmma::mma_sync(sc, af[kk], bf, sc);
            }
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> sh;
            #pragma unroll
            for (int e = 0; e < sh.num_elements; ++e)
                sh.x[e] = __float2half(sc.x[e]);
            wmma::store_matrix_sync(&SP[(wr * 16) * LDSP + nc * 16], sh,
                                    LDSP, wmma::mem_row_major);
        }
        if (n0 + BN < nlimit)                      // K(t+1) 入另一缓冲
            async_tile(Ks + (p ^ 1) * BN * FA_D, k, n0 + BN, S, tid, blockDim.x);
        __syncthreads();
        if (tid < 2 * BM) {
            const int row = tid / 2, hf = tid % 2;
            const int grow = q0 + row;
            const int jend = min(BN, (causal ? grow + 1 : S) - n0);
            const int j0 = hf * 32, j1 = min(j0 + 32, jend);
            float rmax = -1e30f;
            for (int j = j0; j < j1; ++j)
                rmax = fmaxf(rmax, __half2float(SP[row * LDSP + j]) * scale);
            rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));
            const float mn = fmaxf(m_s[row], rmax);
            const float alpha = __expf(m_s[row] - mn);
            float sum = 0.f;
            for (int j = j0; j < j0 + 32; ++j) {   // exp 原位改写 S→P
                float pv = j < jend
                    ? __expf(__half2float(SP[row * LDSP + j]) * scale - mn) : 0.f;
                SP[row * LDSP + j] = __float2half(pv);
                sum += pv;
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
        __pipeline_wait_prior(n0 + BN < nlimit ? 1 : 0);   // V(t) 到位
        __syncthreads();
        #pragma unroll
        for (int c0 = 0; c0 < 4; ++c0) {
            const int c = wc * 4 + c0;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv, oacc;
            wmma::fill_fragment(pv, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                               wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(pf, &SP[(wr * 16) * LDSP + kk * 16], LDSP);
                wmma::load_matrix_sync(vf, &Vs[(kk * 16) * FA_D + c * 16], FA_D);
                wmma::mma_sync(pv, pf, vf, pv);
            }
            float* optr = &Osm[(wr * 16) * FA_D + c * 16];
            wmma::load_matrix_sync(oacc, optr, FA_D, wmma::mem_row_major);
            #pragma unroll
            for (int e = 0; e < pv.num_elements; ++e) pv.x[e] += oacc.x[e];
            wmma::store_matrix_sync(optr, pv, FA_D, wmma::mem_row_major);
        }
        p ^= 1;
    }
    __syncthreads();
    half* o = O + ((size_t)(b * Hq + h) * S + q0) * FA_D;
    for (int i = tid; i < BM * FA_D; i += blockDim.x)
        o[i] = __float2half(Osm[i] / l_s[i / FA_D]);
}

void fa2_v4(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(fa2_v4_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v4_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
