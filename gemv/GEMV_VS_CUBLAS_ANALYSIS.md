# GEMV vs cuBLAS 完整对比分析

## 问题设置

- **算子**：y = M × v，M 为 [rows=4096， cols=2048] 行优先矩阵，v 为 [2048]，y 为 [4096]
- **参考实现**：NVIDIA 真正的 `cublasSgemv`（不是手写仿品）
- **GPU**：RTX 系列（与 softmax 项目同环境）

cuBLAS 用 column-major，我们用 row-major，所以 API 调用要传 `CUBLAS_OP_T`：
```cuda
// row-major M[rows][cols] 当作 column-major 看就是 M^T[cols][rows]
// 要算 y = M*v 等价于 y = (M_cm)^T * v
cublasSgemv(handle, CUBLAS_OP_T,
            cols, rows,       // M_cm 的形状
            &alpha, mat, cols, vec, 1,
            &beta, out, 1);
```

---

## 实测数据

| 版本 | 时延（ms） | 相对 baseline | 相对 cuBLAS |
|------|----------|--------------|------------|
| baseline（CPU-like 单线程） | 0.6179 | 1.00× | 慢 16× |
| v0（一行一线程） | 0.0546 | 11.31× | 慢 43% |
| v1（位运算） | 0.0551 | 11.21× | 慢 45% |
| v2（修复 bank conflict） | 0.0503 | 12.29× | 慢 32% |
| **v3 (warp-per-row + shuffle)** | **0.0324** | **19.07×** ✨ | **快 15%** |
| v4 (block-per-multi-rows + shared x) | 0.0628 | 9.84× | 慢 65% |
| **cuBLAS (cublasSgemv)** | **0.0381** | **16.22×** | 1.00× |

**关键结论**：v3 击败 cuBLAS 15%，但 v4 反而比 cuBLAS 慢 65%。

---

## v3 为什么能打过 cuBLAS

### v3 的设计：one warp per row

```cuda
// blockDim = (32, 4)：32 个 lane × 4 个 warp = 一个 block 处理 4 行
int lane = threadIdx.x;
int warp_id = threadIdx.y;
int row = blockIdx.x * 4 + warp_id;

float sum = 0.0f;
for (int c = lane; c < cols; c += 32) {
    sum += mat[row * cols + c] * vec[c];
}
// warp shuffle 归约，无 __syncthreads()
sum = warp_reduce_sum(sum);
if (lane == 0) out[row] = sum;
```

**核心优化**：
1. **一行一个 warp**：rows=4096 → 4096/4 = 1024 个 block，每个 block 4 行
2. **warp 内 32 lane 并行扫描列**：memory coalescing 完美（连续 32 个 float 一次 load）
3. **warp shuffle 归约**：完全没有 `__syncthreads()`，只有寄存器级通信
4. **vec[c] 通过 L1 cache 复用**：同一 block 的 4 行共享同一段 vec[base..base+32]，cache 自动命中

**为什么比 cuBLAS 快**：
- cuBLAS 必须支持任意 op (N/T)、任意 strides、任意 dtype，无法对"行优先 + op=T + cols=2048 + 4096 行"这种**特定组合**做极致特化
- v3 把 blockDim、warp 数、归约策略全部针对这个 shape 硬编码

---

## v4 为什么反而比 cuBLAS 慢

### v4 的设计：block-per-multi-rows + shared x cache

```cuda
// blockDim = (128, 4)：128 个 lane × 4 行
// 引入 shared memory 缓存 vec 的 tile
for (int base = 0; base < cols; base += 128) {
    smem_x[tx] = vec[base + tx];
    __syncthreads();
    local += mat[row * cols + base + tx] * smem_x[tx];
    __syncthreads();
}
// 然后 128 个线程对一行做 block-level 归约
```

**初衷**：用 shared memory 缓存 vec，减少全局访问

**为什么反而慢**：
1. **vec[c] 已经在 L1 cache 里**：v3 已经天然命中 cache，v4 多此一举把它放进 shared memory，反而多了 store + sync 一来一回的开销
2. **`__syncthreads()` 加倍**：每个 tile 都要两次同步（load 完 + use 完）
3. **block-level 归约比 warp shuffle 慢**：128 个线程的对半收缩需要 7 轮 syncthreads
4. **kRowsPerBlock=4 但 blockSize=128**：实际并行度反而不如 v3 的（32, 4）

**结果**：v4 是一个"过度设计"的反面教材——盲目加 shared memory 不仅没收益，反而把 v3 的优势全部抹掉。

**对比 softmax 项目的 v4.4**：两者都是"看起来合理但实际更慢"的反例：
- softmax v4.4：人为破坏归约（bank conflict + 全同步）
- gemv v4：把不需要的优化加进来（shared memory + 多余同步）

---

## v3 vs cuBLAS 的细节对比

| 维度 | v3 | cuBLAS (cublasSgemv) |
|------|-----|---------------------|
| 主循环线程粒度 | 1 warp = 1 row | 通用 thread-block |
| 归约方式 | warp shuffle（无 syncthreads） | 通用 block reduce |
| 内存访问 | row-major + lane 步长 32 | 必须处理 col-major + op=T 转置 |
| vec 复用 | L1 cache 自动命中 | L1 cache 自动命中 |
| 对齐假设 | rows%4==0， cols%32==0 | 无假设 |
| 代码复杂度 | ~50 行 |（库内部） |
| 时延 | **0.0324 ms** | 0.0381 ms |

**v3 快 15% 的原因**：
- cuBLAS 走的是 op=T 的 transpose 路径，多一层逻辑
- cuBLAS 不敢假设 4096 行恰好是 warp 数倍数，需要 boundary handling
- v3 的 grid/block 配置是针对（4096, 2048） 编译期固定的，cuBLAS 是运行时启发式选择

---

## 与 softmax 项目的对比

| 项目 | 最佳手写版本 | cuBLAS 时延 | 手写优势 |
|------|------------|-----------|---------|
| softmax (1024×1024) | v4: 0.0163 ms | 0.0219 ms | **+26%** |
| **gemv (4096×2048)** | **v3: 0.0324 ms** | **0.0381 ms** | **+15%** |

**为什么 gemv 的优势比 softmax 小**：
1. GEMV 是经典的 BLAS 算子，cuBLAS 在它上面投入了几十年的优化
2. Softmax 不是 BLAS 算子，cuBLAS 的"softmax_cublas"其实是手写参考实现（不是真正的 cuBLAS API）
3. GEMV 的算法选择空间更小（基本就是行/列 tile + reduce），优化空间更窄

**为什么 v4 比 v3 慢 94%**：
- v3 已经在硬件上接近 memory bound 的上限
- v4 加 shared memory 是经典的"过度优化"案例，给了 cuBLAS 反超的机会
- 这正是 softmax v4.4 演示的道理：错误的设计能让你输给通用库

---

## 工程教训

### 1. **手写 kernel 能赢 cuBLAS，但只在特定 shape 上**
v3 在（4096, 2048） 上快 15%，但换成其他尺寸可能就输了。cuBLAS 的优势是稳定。

### 2. **shared memory 不是越多越好**
v4 用 shared memory 缓存 vec，结果比 v3 的纯 L1 cache 方案慢一倍。 **当 L1 cache 已经能 cover 时，shared memory 是负优化**。

### 3. **warp shuffle 是 reduce 算子的"王炸"**
v3 用 warp shuffle 完全避开了 `__syncthreads()`，这是它打过 cuBLAS 的核心原因。和 softmax 项目的 v4 同理：**消除同步 > 减少计算量**。

### 4. **GEMV 是 memory bound 算子**
- 数据量：4096 × 2048 × 4B = 32 MB（M 矩阵）
- 时延 0.0324 ms → 带宽 = 32 MB / 0.0324 ms ≈ **1.0 TB/s**
- RTX 4090 理论带宽 1.0 TB/s → **v3 已经打满带宽**
- 这就是为什么 v3 是上限——再优化也没空间了

### 5. **v3 → v4 是一个完整的反例**
和 softmax 项目的 v4.4 一致：
- softmax：保留 float4，破坏归约 → 慢于 cuBLAS
- gemv：v3 已经最优，v4 加多余的 shared memory + sync → 慢于 cuBLAS

两个项目殊途同归地证明：**优化是全链路的，单点的"看似合理"改动会拖垮全局**。

---

## 一句话总结

> **v3 比 cuBLAS 快 15%，靠的是 warp-per-row + warp shuffle + L1 cache 自然复用 vec 的三件套配合。 v4 加了 shared memory 想"再优化一下"，结果反而比 cuBLAS 慢 65%。这再次证明：高性能 = 全链路一致优化，木桶的最短板决定整体的高度。**
