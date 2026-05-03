# GEMV 每版为什么这么改（v0 ~ v4）

本文档对应 `src/gemv_baseline.cu ~ src/gemv_v4.cu`，统一模板为：
**上一版瓶颈 -> 本版改动 -> 关键代码 -> 预期收益 -> NCU 观察点**。

## baseline -> v0：先把 dot product 并行化

### 上一版瓶颈
- 一行一个线程串行点积，SM 大量空闲。

### 本版改动
- 一个 block 处理一行，线程分摊列维度计算并做 block 归约。

### 关键代码
```cpp
for (int c = tid; c < cols; c += blockSize) {
    sum += mat[row * cols + c] * vec[c];
}
smem[tid] = sum;
// ... reduction
```

### 预期收益
- 并行度显著提升，延迟数量级下降。

### NCU 观察点
- 从 baseline 的 compute 串行形态转向高并行，瓶颈通常迁移到带宽侧。

---

## v0 -> v1：unroll-2

### 上一版瓶颈
- 每轮只处理 1 元素，循环控制和地址计算比例偏高。

### 本版改动
- 每线程每轮处理 2 元素。
- 使用 `__restrict__` 减少别名限制。

### 关键代码
```cpp
for (int c = tid * 2; c < cols; c += blockSize * 2) {
    sum += mat[row_offset + c] * vec[c];
    if (c + 1 < cols) sum += mat[row_offset + c + 1] * vec[c + 1];
}
```

### 预期收益
- 指令控制开销下降，吞吐小幅提升。

### NCU 观察点
- 指令数下降，但总体仍以 memory-bound 为主。

---

## v1 -> v2：float4 向量化

### 上一版瓶颈
- 标量访存事务较多，带宽利用仍不理想。

### 本版改动
- `A` 与 `x` 使用 `float4` 向量化加载，尾部标量回退。

### 关键代码
```cpp
float4 a = reinterpret_cast<const float4*>(mat + row_offset + c)[0];
float4 x = reinterpret_cast<const float4*>(vec + c)[0];
sum += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
```

### 预期收益
- 更好的内存合并访问，减少访存指令。

### NCU 观察点
- DRAM 吞吐接近上限，说明进入典型 GEMV 带宽受限区。

---

## v2 -> v3：warp 级归约

### 上一版瓶颈
- block 级归约需要多次 `__syncthreads()`。

### 本版改动
- 一行映射到一个 warp。
- 使用 `__shfl_down_sync` 做 warp 内归约。

### 关键代码
```cpp
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}
```

### 预期收益
- 同步开销下降，归约尾部更轻。

### NCU 观察点
- 指令数进一步下降，尽管整体仍受 DRAM 吞吐约束。

---

## v3 -> v4：跨行复用 x（结构性尝试）

### 上一版瓶颈
- 每行都重复读取同一段 `x`，跨行复用不足。

### 本版改动
- 一个 block 同时处理多行，把 `x` tile 缓存到 shared memory。

### 关键代码
```cpp
smem_x[tx] = (c < cols) ? vec[c] : 0.0f;
__syncthreads();
if (c < cols) local += mat[row_offset + c] * smem_x[tx];
__syncthreads();
```

### 预期收益
- 减少 `x` 的全局重复读取。

### NCU 观察点
- 该配置下同步/组织开销大于复用收益，性能出现回退。

---

## 当前实测（rows=4096, cols=2048）

- baseline: `0.618788 ms`（`1.00x`）
- v0: `0.054575 ms`（`11.34x`）
- v1: `0.054945 ms`（`11.26x`）
- v2: `0.043368 ms`（`14.27x`）
- v3: `0.032540 ms`（`19.02x`，当前最优）
- v4: `0.067143 ms`（`9.22x`）

## NCU 实测结论（同一轮 profile 均值）

- baseline: `SM 95.84% / DRAM 19.18% / OCC 46.58%`（compute-bound 串行形态）
- v0: `SM 39.26% / DRAM 95.11% / OCC 95.33%`（memory-bound）
- v1: `SM 38.27% / DRAM 95.27% / OCC 95.16%`（memory-bound）
- v2: `SM 27.97% / DRAM 95.22% / OCC 96.05%`（memory-bound）
- v3: `SM 16.01% / DRAM 95.19% / OCC 88.72%`（memory-bound，且同步更少）
- v4: `SM 40.09% / DRAM 95.03% / OCC 96.54%`（memory-bound，但额外开销导致回退）
