# 为什么每版这么改（softmax v0 ~ v4）

本文档对应 `src/softmax_v0.cu ~ src/softmax_v4.cu`，统一使用同一分析模板：
**上一版瓶颈 -> 本版改动 -> 关键代码 -> 预期收益 -> NCU 观察点**。

## baseline -> v0：并行化主流程

### 上一版瓶颈
- 一行只有一个线程顺序执行 max/sum/normalize，GPU 资源严重闲置。

### 本版改动
- 一个 block 负责一行。
- 每线程处理该行多个列元素。
- 行内执行两次 block reduction（max + sum）。

### 关键代码
```cpp
template <int blockSize>
__global__ void softmax_v0_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float smem[blockSize];
    int row = blockIdx.x, tid = threadIdx.x;
    int row_offset = row * cols;
    float thread_max = -1.0e20f;
    for (int c = tid; c < cols; c += blockSize) {
        thread_max = fmaxf(thread_max, input[row_offset + c]);
    }
    float row_max = block_reduce_max_v0<blockSize>(thread_max, smem, tid);
    // ... block_reduce_sum_v0 + normalize
}
```

### 预期收益
- 从串行一行升级为 block 级并行，通常获得数量级加速。

### NCU 观察点
- occupancy 应显著提升，SM/DRAM 利用率进入正常区间。

---

## v0 -> v1：低风险微优化

### 上一版瓶颈
- 交错归约中 `%` 判断开销偏高。
- 地址重复计算和别名保守假设影响编译器优化。

### 本版改动
- `%` 分支改位运算判断。
- 输入输出指针加 `__restrict__`。
- 缓存 `row_offset`。

### 关键代码
```cpp
if (((tid & ((stride << 1) - 1)) == 0) && (tid + stride) < blockSize) {
    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
}

__global__ void softmax_v1_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int rows, int cols) {
    int row_offset = blockIdx.x * cols;
}
```

### 预期收益
- 小幅降低 ALU/控制开销，收益稳定但不会特别大。

### NCU 观察点
- 指令数与时延相对 v0 小幅下降，瓶颈类型不发生大变化。

---

## v1 -> v2：对半归约重排

### 上一版瓶颈
- 交错归约访存模式不够规整，两次归约都受影响。

### 本版改动
- 统一采用 `for (stride = blockSize/2; stride > 0; stride >>= 1)` 的对半归约。

### 关键代码
```cpp
for (int stride = blockSize >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
        smem[tid] += smem[tid + stride];
    }
    __syncthreads();
}
```

### 预期收益
- shared memory 访问更规整，为后续 warp 归约打基础。

### NCU 观察点
- 指令数略降，SM 利用率更稳定。

---

## v2 -> v3：float4 向量化

### 上一版瓶颈
- 仍是标量读写，访存事务和指令条数偏高。

### 本版改动
- `float4` 向量化加载/写回。
- 每线程处理 4 元素，尾部不足 4 个走标量回退。

### 关键代码
```cpp
for (int c = tid * packSize; c < cols; c += blockSize * packSize) {
    if (c + packSize - 1 < cols) {
        const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
        thread_sum += expf(v.x - row_max) + expf(v.y - row_max)
                    + expf(v.z - row_max) + expf(v.w - row_max);
    } else {
        for (int k = 0; k < packSize && (c + k) < cols; ++k) {
            thread_sum += expf(input[row_offset + c + k] - row_max);
        }
    }
}
```

### 预期收益
- 访存吞吐提升，尤其在 `cols` 较大时效果明显。

### NCU 观察点
- DRAM 占比上升，瓶颈从 compute 向 compute/memory 混合转移。

---

## v3 -> v4：warp 尾归约

### 上一版瓶颈
- block reduction 全程依赖 `__syncthreads()`，尾部一个 warp 时同步浪费明显。

### 本版改动
- block 归约只做到 `s > 32`。
- 尾部切换为 `__shfl_down_sync` warp 归约（max/sum 复用一套逻辑）。

### 关键代码
```cpp
template <typename Op>
__device__ float warp_reduce(float value, Op op) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = op(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

for (int stride = blockSize >> 1; stride > 32; stride >>= 1) {
    if (tid < stride) smem[tid] = op(smem[tid], smem[tid + stride]);
    __syncthreads();
}
```

### 预期收益
- 减少同步开销，softmax 双归约场景下收益会被放大。

### NCU 观察点
- 指令数继续下降，最终版本更容易落到 memory-bound。

---

## 当前实测快照（rows=1024, cols=1024）

- baseline: `0.322205 ms`（`1.00x`）
- v0: `0.026267 ms`（`12.27x`）
- v1: `0.026240 ms`（`12.28x`）
- v2: `0.025930 ms`（`12.43x`）
- v3: `0.023244 ms`（`13.86x`）
- v4: `0.016554 ms`（`19.46x`，当前最优）
- 正确性：全部 `correctness_pass=true`，误差约 `1e-9 ~ 1e-8`

## NCU 实测结论（同一轮 profile 均值）

- baseline: `SM 51.77% / DRAM 5.00% / OCC 37.07%`（并行度不足 + latency/mixed）
- v0: `SM 75.55% / DRAM 55.26% / OCC 91.02%`（compute-dominant）
- v1: `SM 76.28% / DRAM 52.20% / OCC 91.00%`（compute-dominant）
- v2: `SM 77.35% / DRAM 52.93% / OCC 90.72%`（compute-dominant）
- v3: `SM 70.00% / DRAM 62.34% / OCC 91.11%`（compute/memory 混合）
- v4: `SM 45.27% / DRAM 78.56% / OCC 88.98%`（memory-bound）
