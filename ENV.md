# 环境复现指南

## 4090 容器(现行数字环境:EXP-K01/K02/K03,2026-08-23 起)
- GPU: RTX 4090(sm_89,~1008 GB/s GDDR6X)
- CUDA： 13.2（`cudaDriverGetVersion` 报 13.3 driver-API 版本——旧 raw 的 driver=13.3 即此误填，勘误见各 project-proof/data/manifest.txt）
- Driver: 610.57.04(NVIDIA UNIX Open Kernel Module,/proc/driver/nvidia/version)
- NCU： **受限**——容器 ERR_NVGPUCTRPERM 无性能计数器权限（records/EXP-K01 §7）；4090 端只有 bench 数字，NCU 机理参照沿用 Laptop 时代 ncu-rep（artifacts/ncu_for_mac/ 与各 profiling/ncu/）
- 本机实况（venv/启动命令）看 /root/work/infra/machine/ENV_REGISTRY.md；本文件只讲异地复现

## 采集主机(唯一有 NCU 计数器权限的环境;EXP-K07/K08/K09 的全部 .ncu-rep 出自此)

> **这套环境是唯一暴露 reduce harness 计时口径 bug 的环境**——主力机的 CUDA 13.2 把它
> 完全掩盖了（`cudaGetDeviceProperties` 在 driver 610 上很快，在 570 上单次达毫秒级，
> 使 v6/v7 测出 1.6–1.7 ms，慢 50 余倍）。将来若要复现该现象或重建采集机，靠的就是下面这几行。
> 详见 `records/EXP-K09` §5.17/§6.19。

| 项 | 值 |
|---|---|
| GPU | RTX 4090（sm_89，24564 MiB，**单卡**） |
| 驱动 | **570.153.02** |
| CUDA | **12.8.93**（driver 570 的上限；13.x 装不上） |
| Nsight Compute | **2025.1.1.0**（随 toolkit） |
| cuDNN | 8.9.7（**随 toolkit 自带**，在 `/usr/local/cuda-12.8/targets/x86_64-linux/`，通常无需另装） |
| 形态 | 虚机，GPU 直通（**不是容器**——容器改不了 modprobe 参数） |
| OS / Python | Ubuntu 22.04 / Python 3.10.12（系统解释器，**非 venv**） |
| Python 包路径 | `/home/ubuntu/.local/lib/python3.10/site-packages`（`pip3 install --user`） |
| torch / triton | 2.11.0+cu128 / 3.6.0 |
| ninja / numpy / matplotlib | 1.13.0 / 2.2.6 / 3.10.9 |

**计数器权限**（一次性，需重启后生效）：

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/zz-nvidia-profiler.conf
sudo update-initramfs -u && sudo reboot
grep -i RmProfilingAdminOnly /proc/driver/nvidia/params   # 必须为 0
bash scripts/probe_ncu_permission.sh                      # 退出码必须为 0
```

**重建要点（踩过的坑，按顺序）**：

1. cuDNN 包名是 `cudnn9-cuda-12-8`，**不是** `libcudnn9-cuda-12`；且 `apt-cache search` 搜不到它
   （该 repo 无 description 索引），要用
   `grep -h "^Package: " /var/lib/apt/lists/*nvidia*Packages | grep -i cudnn`。多数情况下这步可跳过。
2. torch 扩展算子额外需要 **ninja**（缺了直接 `RuntimeError`）与 **numpy**（缺了 `bench.py` 后续挂）；
   两者缺失都会让 `run_ncu_all.sh` 因 `set -e` **静默退出**，只留半份报告。
3. 改过 `.cu` 后必须 `rm -rf ~/.cache/torch_extensions/`，否则可能采到**旧代码**且日志看不出来；
   重编后用 `cuobjdump -sass <so> | grep -c LDG.E.128` 之类的 SASS 判据确认拿到的是新代码。
4. 姊妹仓（同 harness 对照用）：`git clone https://github.com/lyell0710/triton-kernels.git`，
   通过 `TRITON_KERNELS_SRC` 指向其 `src/`。

## 测试环境(RTX 4070 Laptop —— 仅对应旧代 Laptop 数据,现行 4090 数字见上节)
- OS: Ubuntu 22.04
- GPU: RTX 4070 Laptop (sm_89)
- CUDA Driver: 595.71.05（支持 CUDA 13.2）
- nvcc: 11.5
- CMake: 3.18+
- C++ 标准： C++17
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

> 若 CMake 报 "No CMAKE_CUDA_COMPILER could be found"，先把 nvcc 加入 PATH： `export PATH=/usr/local/cuda/bin:$PATH`（建议写入 `~/.bashrc`）

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
