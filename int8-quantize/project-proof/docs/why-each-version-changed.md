# INT8 Quantize 每版为什么这么改（含上一版瓶颈）

本文档对应 `src/quantize_baseline.cu ~ src/quantize_v4.cu`。  
目标是把“上一版慢在哪、这一版怎么破”说清楚。

---

## baseline -> v0

### 上一版（baseline）限制了什么
- 单线程串行处理所有元素，GPU 并行能力完全浪费。

### 本版（v0）怎么改
- 改成 grid-stride 并行，一线程处理多个元素。

### 预期收益
- 吞吐数量级提升，是最关键的一步。

---

## v0 -> v1

### 上一版（v0）限制了什么
- 单次循环只量化 1 元素，循环和索引开销占比高。

### 本版（v1）怎么改
- 每线程一次处理 2 元素（unroll-2）。

### 预期收益
- 降低控制流和地址计算开销。

---

## v1 -> v2

### 上一版（v1）限制了什么
- 单线程工作粒度仍偏小，发射效率可继续提升。

### 本版（v2）怎么改
- 每线程处理 4 元素（unroll-4）。

### 预期收益
- 进一步提高指令吞吐与流水线利用率。

---

## v2 -> v3

### 上一版（v2）限制了什么
- 同一 channel 的 `scale` 被重复读取次数过多。

### 本版（v3）怎么改
- block 与 channel 绑定（一块一 channel）。
- 每个 block 对应 `scale` 只读取一次到寄存器复用。

### 预期收益
- 减少无效 scale 访存，降低 memory traffic。

---

## v3 -> v4

### 上一版（v3）限制了什么
- 输入仍标量加载，存储仍标量写回，访存指令较多。

### 本版（v4）怎么改
- `float4` 向量化读取输入，`char4` 向量化写回输出。

### 预期收益
- 连续访存效率提升，进一步降低延迟。

---

## 当前实测（channels=1024, hw=1024）

- baseline: `121.114876 ms` (`1.00x`)
- v0: `0.014809 ms` (`8178.49x`)
- v1: `0.011821 ms` (`10245.91x`)
- v2: `0.010482 ms` (`11554.29x`)
- v3: `0.007554 ms` (`16032.70x`)
- v4: `0.006633 ms` (`18260.45x`, 当前最优)

---

## NCU 实测结论（同一轮 profile 的均值）

- baseline: `SM 0.12% / DRAM 0.40% / OCC 8.33%`，典型**latency/并行度不足**，GPU 基本没被用起来。
- v0: `SM 60.52% / DRAM 68.78% / OCC 81.10%`，并行后立即转为**memory-bound**。
- v1: `SM 55.14% / DRAM 83.30% / OCC 85.00%`，仍 memory-bound，2 元素展开减少了控制开销。
- v2: `SM 43.05% / DRAM 81.48% / OCC 86.37%`，4 元素展开后带宽限制继续主导。
- v3: `SM 24.88% / DRAM 84.03% / OCC 89.00%`，按 channel 绑定后 scale 复用更好，但总体仍受 DRAM 吞吐限制。
- v4: `SM 24.88% / DRAM 84.36% / OCC 85.72%`，向量化读写进一步压低延迟，最终最优版本仍是 memory-bound。

### 和版本演进的对应关系
- `baseline -> v0`：最关键收益来自并行化本身，NCU 上体现在 occupancy 和吞吐跃升。
- `v1 -> v3`：展开与 scale 复用在 memory-bound 场景下继续挖出增量。
- `v4`：进一步优化指令/访存组织后达到当前最佳，但瓶颈仍是带宽上限。

---

## NCU 验证重点（建议）

- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__inst_executed.sum`
- `smsp__warps_active.avg.pct_of_peak_sustained_active`
- `l1tex__t_bytes.sum` / `lts__t_bytes.sum`

看法：
- v0/v1/v2 重点看指令与吞吐是否持续优化。
- v2/v3 重点看 scale 复用后 memory traffic 是否下降。
- v3/v4 重点看向量化是否带来更高有效带宽与更低指令数。
