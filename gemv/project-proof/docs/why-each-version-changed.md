# GEMV 每版为什么这么改（含上一版瓶颈）

本文档对应 `src/gemv_baseline.cu ~ src/gemv_v4.cu`。  
核心目标：不仅说明“改了什么”，还要说明“上一版为什么慢”。

---

## baseline -> v0

### 上一版（baseline）限制了什么
- 一行只用 1 个线程串行点积，几乎没有并行度。
- 大量 SM 空闲，吞吐上不去。

### 本版（v0）怎么改
- 一个 block 负责一行。
- 每个线程处理该行的一部分列，最后做 block 内归约。

### 对应代码（关键片段）
```cpp
template <int blockSize>
__global__ void gemv_v0_kernel(const float* mat, const float* vec, float* out, int rows, int cols) {
    __shared__ float smem[blockSize];
    int row = blockIdx.x, tid = threadIdx.x;
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockSize) {
        sum += mat[row * cols + c] * vec[c];
    }
    smem[tid] = sum;
    // ... block reduction
}
```

### 预期收益
- 从“单线程串行”变成“block 级并行”。
- 延迟通常会出现数量级下降。

---

## v0 -> v1

### 上一版（v0）限制了什么
- 每线程每轮只处理 1 个元素，循环控制与地址计算比例偏高。

### 本版（v1）怎么改
- 每线程改成 2 元素展开（unroll-2）。
- 输入输出指针使用 `__restrict__`，减轻编译器别名保守假设。

### 对应代码（关键片段）
```cpp
for (int c = tid * 2; c < cols; c += blockSize * 2) {
    sum += mat[row_offset + c] * vec[c];
    if (c + 1 < cols) {
        sum += mat[row_offset + c + 1] * vec[c + 1];
    }
}
```

### 预期收益
- 降低指令/控制开销，提升 ALU 和访存流水效率。

---

## v1 -> v2

### 上一版（v1）限制了什么
- 仍然是标量访存，global memory transaction 利用率有限。

### 本版（v2）怎么改
- 使用 `float4` 向量化读取 `A` 和 `x`。
- 尾部不足 4 元素时回退标量路径，确保边界正确。

### 对应代码（关键片段）
```cpp
for (int c = tid * 4; c < cols; c += blockSize * 4) {
    if (c + 3 < cols) {
        float4 a = reinterpret_cast<const float4*>(mat + row_offset + c)[0];
        float4 x = reinterpret_cast<const float4*>(vec + c)[0];
        sum += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
    } else {
        for (int k = 0; k < 4 && c + k < cols; ++k) sum += mat[row_offset + c + k] * vec[c + k];
    }
}
```

### 预期收益
- memory coalescing 更好，访存指令更少。

---

## v2 -> v3

### 上一版（v2）限制了什么
- 仍依赖 block 级共享内存归约 + 多次 `__syncthreads()`。

### 本版（v3）怎么改
- 一行映射到一个 warp。
- 用 `__shfl_down_sync` 完成 warp 内归约，避免 block 级同步。

### 对应代码（关键片段）
```cpp
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

int row = blockIdx.x * blockDim.y + threadIdx.y;
for (int c = threadIdx.x; c < cols; c += 32) sum += mat[row_offset + c] * vec[c];
```

### 预期收益
- 减少同步等待，降低归约尾部开销。

---

## v3 -> v4

### 上一版（v3）限制了什么
- 每行都重复读取同一段 `x`，跨行复用不足。

### 本版（v4）怎么改
- 一个 block 同时处理多行，共享 `x` 的 tile 到 shared memory。
- 尝试用“跨行复用 x”换取更少 global 读取。

### 对应代码（关键片段）
```cpp
__shared__ float smem_x[blockSize];
int row = blockIdx.x * rowsPerBlock + threadIdx.y;
for (int base = 0; base < cols; base += blockSize) {
    int c = base + threadIdx.x;
    smem_x[threadIdx.x] = (c < cols) ? vec[c] : 0.0f;
    __syncthreads();
    if (c < cols) local += mat[row_offset + c] * smem_x[threadIdx.x];
    __syncthreads();
}
```

### 为什么这版在当前配置下反而慢
- 共享内存/同步开销增加，抵消了复用收益。
- `rowsPerBlock`、`blockSize` 和数据规模组合下，实际占用率与访存收益不匹配。

### 结论
- `v4` 是“结构性尝试版”，用于验证复用思路，不保证必然比 `v3` 快。

---

## 当前实测（rows=4096, cols=2048）

- baseline: `0.618788 ms` (`1.00x`)
- v0: `0.054575 ms` (`11.34x`)
- v1: `0.054945 ms` (`11.26x`)
- v2: `0.043368 ms` (`14.27x`)
- v3: `0.032540 ms` (`19.02x`, 当前最优)
- v4: `0.067143 ms` (`9.22x`)

---

## NCU 实测结论（同一轮 profile 的均值）

- baseline: `SM 95.84% / DRAM 19.18% / OCC 46.58%`，明显**compute-bound**（串行指令长链）。
- v0: `SM 39.26% / DRAM 95.11% / OCC 95.33%`，切到**memory-bound**，说明并行后主要瓶颈转为带宽。
- v1: `SM 38.27% / DRAM 95.27% / OCC 95.16%`，仍是 memory-bound，小幅降低了指令控制开销。
- v2: `SM 27.97% / DRAM 95.22% / OCC 96.05%`，向量化后依旧被带宽限制，符合 GEMV 的常见形态。
- v3: `SM 16.01% / DRAM 95.19% / OCC 88.72%`，**memory-bound 且同步/访存组织更轻**，对应当前最快版本。
- v4: `SM 40.09% / DRAM 95.03% / OCC 96.54%`，仍 memory-bound，但 shared-memory/同步额外成本上升，导致性能回退。

### 和版本演进的对应关系
- `baseline -> v0`：从 compute-bound 串行实现跃迁到 memory-bound 并行实现，这是正常且健康的瓶颈迁移。
- `v2 -> v3`：减少同步、缩短归约路径有效，尽管仍受带宽上限约束。
- `v3 -> v4`：跨行复用 x 的想法成立，但当前参数组合下复用收益未覆盖同步/组织成本。

---

## NCU 验证重点（建议）

- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__inst_executed.sum`

看法：
- v1/v2 主要看访存效率提升是否成立。
- v2/v3 主要看同步减少后 warp 活跃度和指令效率是否提升。
- v3/v4 重点确认 shared memory 额外开销是否导致整体退化。
