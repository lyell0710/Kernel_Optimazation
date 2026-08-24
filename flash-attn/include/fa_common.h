#pragma once
#include <cuda_fp16.h>
// FA2 forward 简化版版本梯。布局 (B,H,S,D) row-major,D=128 固定,
// fp16 输入 / fp32 在线累加,causal + GQA(Hq % Hkv == 0)。
void fa2_v0(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v1(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v2(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v4(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v3(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
// fp32 精算参考(非在线,物化 S²,正确性 gate 专用,慢)
void attn_ref_fp32(const half* Q, const half* K, const half* V, half* O,
                   int B, int Hq, int Hkv, int S, bool causal);
constexpr int FA_D = 128;
