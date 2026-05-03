# cuda-gemv

GEMV 手写 CUDA kernel 优化项目，采用和 `cuda-reduce` / `softmax` 一致的版本化实验方式。

## 版本
- `baseline`: 单线程行内累加
- `v0`: block 并行归约
- `v1`: 每线程双元素展开 + restrict
- `v2`: float4 向量化访存
- `v3`: 一行一 warp（shuffle 归约）
- `v4`: 多行共享 x-tile，降低重复读向量

## 构建运行
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gemv_bench
```

## 生成图表
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_correctness.py
```

## NCU 关键指标采集
```bash
bash project-proof/scripts/profile_ncu.sh
python project-proof/scripts/plot_ncu_summary.py
```

输出目录：`project-proof/profiling/ncu/`
图表目录：`project-proof/docs/figures/02-profiling/`

> 若提示 `ncu: command not found`，先安装 Nsight Compute（例如：`sudo apt install nsight-compute`）。
