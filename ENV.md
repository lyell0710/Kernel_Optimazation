# 环境复现指南

## 测试环境
- OS: Ubuntu 22.04
- GPU: RTX 4070 Laptop (sm_89)
- CUDA Driver: 595.71.05（支持 CUDA 13.2）
- nvcc: 11.5
- CMake: 3.18+
- C++ 标准: C++17
- Python: 3.13

## 依赖

### 系统依赖
```bash
sudo apt install -y cmake g++ build-essential
# CUDA Toolkit >= 11.0（含 cublas）
# cuBLAS 随 CUDA Toolkit 一起安装，无需单独处理
# NCU（Nsight Compute）用于 profiling，随 CUDA Toolkit 附带
```

### Python 依赖（project-proof 绘图脚本）
```bash
pip install matplotlib numpy
```

## 编译（每个 kernel 独立 CMake）

```bash
# softmax
cd softmax && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# cuda-reduce
cd cuda-reduce && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# gemv
cd gemv && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# int8-quantize
cd int8-quantize && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

## Profiling
```bash
# 需要 sudo 或调整 perf_event_paranoid
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# 运行全量 NCU profile
bash scripts/run_ncu_all.sh

# 运行 benchmark + 画图
bash scripts/run_bench_and_plot_all.sh
```

## 说明
- 各 kernel 编译后可执行文件在各自 `build/` 目录下
- project-proof/ 下的 scripts/ 包含 NCU 采集脚本和 Python 绘图脚本
- sm_89（RTX 4070）兼容 CMakeLists 中的 sm_86 编译目标
