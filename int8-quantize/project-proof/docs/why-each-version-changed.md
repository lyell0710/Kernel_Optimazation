# INT8 Quantize 每版为什么这么改（v0 ~ v4）

本文档对应 `src/quantize_baseline.cu ~ src/quantize_v4.cu`，统一模板为：
**上一版瓶颈 -> 本版改动 -> 关键代码 -> 预期收益 -> NCU 观察点**。

## baseline -> v0：grid-stride 并行化

### 上一版瓶颈
- 单线程串行处理所有元素，GPU 并行能力几乎没被利用。

### 本版改动
- 改为 grid-stride 并行，一线程处理多个元素。

### 关键代码
```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int step = blockDim.x * gridDim.x;
for (int i = gid; i < total; i += step) {
    int c = i / hw;
    output[i] = quant_one(input[i], scales[c]);
}
```

### 预期收益
- 吞吐出现数量级提升，这是最核心的一步。

### NCU 观察点
- occupancy 和吞吐应明显上升，瓶颈会快速转到带宽侧。

---

## v0 -> v1：unroll-2

### 上一版瓶颈
- 每轮只量化 1 个元素，循环控制占比偏高。

### 本版改动
- 每线程每轮处理 2 元素。

### 关键代码
```cpp
for (int i = gid * 2; i < total; i += step * 2) {
    output[i] = quant_one(input[i], scales[i / hw]);
    if (i + 1 < total) output[i + 1] = quant_one(input[i + 1], scales[(i + 1) / hw]);
}
```

### 预期收益
- 降低控制流与索引开销，持续压低时延。

### NCU 观察点
- 指令效率提升，但整体仍是 memory-bound。

---

## v1 -> v2：unroll-4

### 上一版瓶颈
- 线程工作粒度仍偏小，发射效率还有空间。

### 本版改动
- 每线程每轮处理 4 元素（`#pragma unroll`）。

### 关键代码
```cpp
for (int base = gid * 4; base < total; base += step * 4) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int i = base + k;
        if (i < total) output[i] = quant_one(input[i], scales[i / hw]);
    }
}
```

### 预期收益
- 提高指令吞吐与流水线利用率。

### NCU 观察点
- `smsp__inst_executed.sum` 下降，吞吐维持高位。

---

## v2 -> v3：按 channel 绑定 block

### 上一版瓶颈
- 同一个 channel 的 `scale` 被反复读取。

### 本版改动
- 一个 block 负责一个 channel，`scale` 读一次后复用。

### 关键代码
```cpp
int c = blockIdx.x;
float s = scales[c];
int base = c * hw;
for (int i = threadIdx.x; i < hw; i += blockDim.x) {
    output[base + i] = quant_one(input[base + i], s);
}
```

### 预期收益
- 减少无效 scale 访存，进一步压低 memory traffic。

### NCU 观察点
- 带宽利用维持高位，且每元素开销继续下降。

---

## v3 -> v4：向量化读写（float4 + char4）

### 上一版瓶颈
- 输入/输出仍以标量访存，访存指令数量偏多。

### 本版改动
- 输入 `float4` 向量化读取，输出 `char4` 向量化写回。

### 关键代码
```cpp
float4 v = reinterpret_cast<const float4*>(input + base + i)[0];
char4 q;
q.x = static_cast<signed char>(quant_one(v.x, s));
q.y = static_cast<signed char>(quant_one(v.y, s));
q.z = static_cast<signed char>(quant_one(v.z, s));
q.w = static_cast<signed char>(quant_one(v.w, s));
reinterpret_cast<char4*>(output + base + i)[0] = q;
```

### 预期收益
- 连续访存效率提升，在 memory-bound 场景继续获取增量。

### NCU 观察点
- 在高 DRAM 占比下进一步降低每元素指令开销。

---

## 当前实测（channels=1024, hw=1024）

- baseline: `121.114876 ms`（`1.00x`）
- v0: `0.014809 ms`（`8178.49x`）
- v1: `0.011821 ms`（`10245.91x`）
- v2: `0.010482 ms`（`11554.29x`）
- v3: `0.007554 ms`（`16032.70x`）
- v4: `0.006633 ms`（`18260.45x`，当前最优）

## NCU 实测结论（同一轮 profile 均值）

- baseline: `SM 0.12% / DRAM 0.40% / OCC 8.33%`（latency/并行度不足）
- v0: `SM 60.52% / DRAM 68.78% / OCC 81.10%`（memory-bound）
- v1: `SM 55.14% / DRAM 83.30% / OCC 85.00%`（memory-bound）
- v2: `SM 43.05% / DRAM 81.48% / OCC 86.37%`（memory-bound）
- v3: `SM 24.88% / DRAM 84.03% / OCC 89.00%`（memory-bound）
- v4: `SM 24.88% / DRAM 84.36% / OCC 85.72%`（memory-bound，且当前最优）
