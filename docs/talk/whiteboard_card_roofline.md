# 白板推导卡 · Roofline 三步定位


1. 算术强度 I = FLOPs / Bytes;平衡点 I* = 峰值算力/峰值带宽
   (4090:165.2 TF / 1.008 TB/s ≈ 164 FLOP/B)。
2. I ≪ I* → memory-bound(reduce:0.25 FLOP/B),优化访存
   (合并/向量化/复用);I ≫ I* → compute-bound(GEMM:I≈1365),
   优化指令(Tensor Core/流水)。
3. 验证闭环:达成带宽或算力占峰值的百分比 + NCU 计数器交叉
   (实测锚:reduce 带宽贴 roofline;GEMM v4 133.1 TF=峰值 81%,EXP-K02《CUDA Tensor Core GEMM 版本梯》)。
4. **面试点**:occupancy 33% 却最快(EXP-K02)——roofline 不看线程数,
   看每字节复用与 ILP;先判 bound 类型再选手段,反着做全是无用功
   (gemm v0→v1 仅 +25% 的教训)。
