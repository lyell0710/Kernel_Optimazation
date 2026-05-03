# cuda-int8-quantize

INT8 per-channel symmetric quantize CUDA kernel 优化项目，保持和 `reduce/softmax/gemv` 一致的工程组织方式。

## 版本
- `baseline`: 单线程串行量化
- `v0`: grid-stride 一线程一元素
- `v1`: 每线程2元素展开 + restrict
- `v2`: 每线程4元素展开
- `v3`: 一块一channel，scale寄存器复用
- `v4`: float4读 + char4写向量化

## 构建运行
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/int8_quantize_bench
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
