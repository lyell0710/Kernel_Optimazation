// ============================================================================
// per-token 动态量化 —— 三级版本梯
//
// 语义:q[t,h] = round(x[t,h] / scale[t]),scale[t] = rowmax(|x[t,:]|) / 127
//
// 字节账(每元素):读 x(2 B)+ 写 q(1 B) = 3 B(scale 只有 T 个 float,不计)。
// 注意输出只有输入的一半宽度,所以本算子的字节账是不对称的 —— 这一点会让
// 「有效带宽」的分母比 fused-norm 那类读写同宽的算子更小。
//
// v0 两遍读(先求 absmax 再量化)、v1 融合(读一次、寄存器暂存)、v2 向量化。
// 结构与 fused-norm 的版本梯同构,但这里第一遍与第二遍之间隔着一次 block 归约,
// 与 RMSNorm 完全同型 —— 所以 v1 的「寄存器暂存」在这里是必然收益还是零收益,
// 取决于同一个问题:第二次读会不会被缓存接住。
// 预测:与 fused-norm 一致,行长 H<=8192 时整行只有 16 KB,第二遍必然命中 L1/L2,
// 所以 v1 相对 v0 的收益应当远小于「少读一次」暗示的 5/3,主要来自省一次 launch。
// ============================================================================
#include "w8a8.h"

// ---- v0:两个 kernel,先求行 absmax,再量化 --------------------------------
__global__ void v0_absmax_kernel(float* __restrict__ scale,
                                 const __nv_bfloat16* __restrict__ x,
                                 int H) {
    extern __shared__ float smem[];
    const __nv_bfloat16* row = x + (long long)blockIdx.x * H;
    float m = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        m = fmaxf(m, fabsf(__bfloat162float(row[i])));
    smem[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        // 整行全零时 absmax=0,scale 会变成 0,后续除法出 inf/nan。
        // 兜底成一个极小正数:量化结果仍是全 0,语义正确。
        scale[blockIdx.x] = fmaxf(smem[0], 1e-12f) / 127.0f;
}

__global__ void v0_quant_kernel(int8_t* __restrict__ q,
                                const __nv_bfloat16* __restrict__ x,
                                const float* __restrict__ scale, int H) {
    const long long off = (long long)blockIdx.x * H;
    // 取倒数一次,循环内用乘法:除法在 GPU 上是多指令序列,逐元素做会很贵。
    const float inv = 1.0f / scale[blockIdx.x];
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        q[off + i] = quant_one(__bfloat162float(x[off + i]), inv);
}

void quant_per_token_v0(int8_t* q, float* scale, const __nv_bfloat16* x,
                        int T, int H, cudaStream_t st) {
    int bs = max(128, min(1024, ((H + 31) / 32) * 32));
    v0_absmax_kernel<<<T, bs, bs * sizeof(float), st>>>(scale, x, H);
    v0_quant_kernel<<<T, bs, 0, st>>>(q, x, scale, H);
}

// ---- v1:单 kernel,warp shuffle 归约 ---------------------------------------
__global__ void v1_kernel(int8_t* __restrict__ q, float* __restrict__ scale,
                          const __nv_bfloat16* __restrict__ x, int H) {
    __shared__ float smem[32];
    const long long off = (long long)blockIdx.x * H;
    const __nv_bfloat16* row = x + off;

    float m = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        m = fmaxf(m, fabsf(__bfloat162float(row[i])));
    const float s = fmaxf(block_reduce_max(m, smem), 1e-12f) / 127.0f;
    if (threadIdx.x == 0) scale[blockIdx.x] = s;

    const float inv = 1.0f / s;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        q[off + i] = quant_one(__bfloat162float(row[i]), inv);
}

void quant_per_token_v1(int8_t* q, float* scale, const __nv_bfloat16* x,
                        int T, int H, cudaStream_t st) {
    int bs = max(128, min(1024, ((H + 31) / 32) * 32));
    v1_kernel<<<T, bs, 0, st>>>(q, scale, x, H);
}

// ---- v2:向量化(读 16 B bf16,写 8 B int8)---------------------------------
struct alignas(16) BF16x8 { __nv_bfloat162 h[4]; };
// 8 个 int8 = 8 字节,用 alignas(8) 让编译器发 64 位 store。
// 读写宽度不对称(读 16 B / 写 8 B)是量化算子的固有形态:输出本来就窄一半。
struct alignas(8) I8x8 { int8_t v[8]; };

__global__ void v2_kernel(int8_t* __restrict__ q, float* __restrict__ scale,
                          const __nv_bfloat16* __restrict__ x, int H) {
    __shared__ float smem[32];
    const int HV = H >> 3;
    const long long off = (long long)blockIdx.x * HV;
    const BF16x8* row = reinterpret_cast<const BF16x8*>(x) + off;
    I8x8* qrow = reinterpret_cast<I8x8*>(q) + off;

    float m = 0.f;
    for (int i = threadIdx.x; i < HV; i += blockDim.x) {
        BF16x8 v = row[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(v.h[j]);
            m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    const float s = fmaxf(block_reduce_max(m, smem), 1e-12f) / 127.0f;
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
    const float inv = 1.0f / s;

    for (int i = threadIdx.x; i < HV; i += blockDim.x) {
        BF16x8 v = row[i];       // 第二遍重读:整行只有 2H 字节,必然命中 L1/L2
        I8x8 o;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(v.h[j]);
            o.v[2 * j]     = quant_one(f.x, inv);
            o.v[2 * j + 1] = quant_one(f.y, inv);
        }
        qrow[i] = o;
    }
}

void quant_per_token_v2(int8_t* q, float* scale, const __nv_bfloat16* x,
                        int T, int H, cudaStream_t st) {
    int bs = max(64, min(1024, ((H / 8 + 31) / 32) * 32));
    v2_kernel<<<T, bs, 0, st>>>(q, scale, x, H);
}
