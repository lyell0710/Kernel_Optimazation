# 采集主机任务书（临时文件）

> **这是临时文件。** 全部任务完成、报告取回并入库之后删除它，
> 把其中仍然有效的结论并入 `LEDGER.md` 与 `docs/ncu_reading_guide.md`。
> 留着它超过一轮采集，就会变成第二个状态源，违反铁律 1。
>
> **本文件已按第一轮（2026-08-29，EXP-K07）的实测踩坑修订，现服务于第二轮。**
> 第二轮的目的与第一轮不同，见文末「本轮与上一轮的差异」——**开工前先读那一节**。

给在**采集主机**上干活的 agent / 人。先读 `/root/standards/CORE.md`（如果那台机器上没有，
本文件第 3 节复述了本次必须遵守的部分）。

---

## 1 为什么会有这台机器

主力机是租用容器，`RmProfilingAdminOnly=1` 且无 `CAP_SYS_ADMIN`，
Nsight Compute 取不到性能计数器（`ERR_NVGPUCTRPERM`）。容器内无解。

采集主机是**虚机**，GPU 直通、nvidia 驱动在虚机内核里加载，因此
`/etc/modprobe.d/` 的参数由本机说了算。已设 `NVreg_RestrictProfilingToAdminUsers=0`。

这台机器**只做一件事**：把主力机上做不了的 NCU 计数器采集做掉，
产出 `.ncu-rep` 带回主力机。**不在这台机器上做开发、不改算法、不调优。**

## 2 两台机器的身份（严禁混排数字）

| | 主力机 | 采集主机 |
|---|---|---|
| GPU | RTX 4090 · sm_89 · 128 SM · 24 GB · L2 72 MB | 同左（同型号） |
| 驱动 | 610.57.04 | 570.153.02 |
| CUDA | 13.2 | **12.8**（驱动 570 的上限） |
| Nsight Compute | 2026.1.0.0（但采不了） | 2025.1.1.0 |
| 形态 | 容器，无计数器权限 | 虚机，计数器已开 |

**卡是同型号，所以机理与占用率结论可迁移。**
**工具链不同（ptxas 12.8 vs 13.2），所以：**

- 计数器类结论（stall 分解、sector 比值、bank conflict、achieved occupancy）→ 可用
- 绝对时延数字 → **不得与主力机的 benchmark 表混排成同一行**，引用时必须注明 CUDA 12.8
- SASS 指令数 → 可能与主力机 `cuobjdump` 的结果有出入，属预期

仓内另有 38 份历史 `.ncu-rep`，出处是 **RTX 4070 Laptop（36 SM / 8 GB）**，
不是 4090。见 `artifacts/ncu_for_mac/MANIFEST.md` 的批次表。三批数据三个身份，别搅在一起。

## 3 必须遵守的规矩

摘自 `/root/standards/CORE.md`，只列本次会碰到的：

- **铁律 3 raw 不可变**：禁止覆盖写入 `data/raw/`；坏数据不原地改，移 `data/archive/` 并注明。
- **profiler 隔离**：NCU/nsys 下跑 bench **一律不写 `project-proof/data/`**。
  已经出过事：十个 bench 里只有 `gemm` 与 `flash-attn` 读 `BENCH_OUT`，
  另外四个 C++ bench 把 CSV 路径写死成相对路径，在算子目录里跑 profiler
  会直接覆盖权威数据。现在 `profile_ncu.sh` 走 `<OP>_PROFILE_ONLY`（该分支在写 CSV 前 `return 0`），
  `run_nsys_all.sh` 走沙箱 cwd。**收尾必须核对**：
  ```bash
  git status --porcelain -- '*/project-proof/data/'   # 必须为空
  ```
- **铁律 6 对照物命名诚实**：`cublas` 只能指真实库调用。
  已知问题：`softmax` 的 `softmax_cublas_kernel` **是自写 kernel，不是 cuBLAS**
  （BLAS 规范里没有 softmax，正确的标准库对照是 cuDNN）。
  **这是已存档红线项**——采到的那份报告不得用于任何"对比 cuBLAS"的表述。
  对照：`gemv_cublas_profile` 里是 `gemv2T_kernel_val<...cublasGemvParams...>`，那才是真 cuBLAS。
- **铁律 1 单一事实源**：派生物不入库。`artifacts/ncu_for_mac/reports/`（副本）、
  `*.tar.gz`、`*.sqlite` 已在 `.gitignore`。
- **不要改源码**：`docs/lectures/` 里有 133 处按行号逐字引用源文件的代码块，
  改源码会静默打断它们。确需改动时，改完跑 `/root/standards/verify_lectures.py`。

## 4 任务清单

### T1 环境就绪（验收：三项全部满足）

```bash
sudo apt install -y cmake cudnn9-cuda-12-8
git clone https://github.com/lyell0710/Kernel_Optimazation.git && cd Kernel_Optimazation
```

| 检查 | 命令 | 验收 |
|---|---|---|
| 计数器已开 | `grep -i RmProfilingAdminOnly /proc/driver/nvidia/params` | `0` |
| 实采可行 | `bash scripts/probe_ncu_permission.sh` | 退出码 `0` |
| cuDNN 就位 | `find /usr -name cudnn.h 2>/dev/null \| head -1` | 有输出 |

**包名（EXP-K07 §7.3 实测）**：任务书原写的 `libcudnn9-cuda-12` / `libcudnn9-dev-cuda-12`
**在 NVIDIA CUDA repo 里不存在**，实际是 `cudnn9-cuda-12-8`。
`apt-cache search` 搜不到它（该 repo 的包无 description 索引），要用：

```bash
grep -h "^Package: " /var/lib/apt/lists/*nvidia*Packages | grep -i cudnn | sort -u
```

**多数情况下这一步可以跳过**：CUDA Toolkit 12.8 自带 cuDNN 8.9.7
（`/usr/local/cuda-12.8/targets/x86_64-linux/include/cudnn.h`），`softmax` 的 CMake 走多路径搜索能找到。
装 toolkit 后先跑一次验收命令，有输出就不用另装。

### T2 构建六个 C++ 算子（验收：六行全部 `sm_89`）

```bash
bash scripts/setup_profiling_host.sh
```

脚本会自检、构建、核对编译目标、采集、导出。**第 3 节「编译目标核对」必须六行全是 `sm_89`。**

为什么要盯这一行：`softmax` / `gemv` / `int8-quantize` / `cuda-reduce` 四个的
`CMakeLists.txt` 没写 `CMAKE_CUDA_ARCHITECTURES`，用 CMake 默认值会编成 **sm_75**。
在 4090 上跑的是驱动 JIT 出来的码，而 `cuobjdump` 看到的 sm_75 SASS **不是实际执行的码**，
寄存器分配与占用率也因此不可比。脚本用 `-DCMAKE_CUDA_ARCHITECTURES=89` 强制覆盖。

出现不符时：`rm -rf */build` 后重跑（CMake 缓存了旧配置）。

### T3 四个 torch 扩展算子（**本轮必做**，不再可选）

> 本轮（第二轮）的核心目的就是重采这四个——向量化修复后访存指令少了 4 倍，
> L1/L2/DRAM 的账必然变了。详见文末「本轮与上一轮的差异」。

```bash
sudo apt install -y python3-pip
pip3 install --user torch --index-url https://download.pytorch.org/whl/cu128
pip3 install --user triton ninja numpy
bash scripts/setup_profiling_host.sh --with-python
```

**`ninja` 和 `numpy` 不能漏**（EXP-K07 §7.3 实测踩坑）：

- 缺 `ninja` → torch JIT 扩展编译直接失败（`RuntimeError: Ninja is required to load C++ extensions`）
- 缺 `numpy` → torch 只报一句 UserWarning，但 `bench.py` 后续会失败

两种情况下 `run_ncu_all.sh` 都因 `set -e` **静默退出**，只留下半份报告、日志里看不出原因。
上一轮在这里损失了两轮时间。跑完务必核对报告份数：fused-norm / rope / activation / w8a8
各自的臂数应与其 `profile_ncu.sh` 里 `PROFILE_TARGETS` 的条目数一致。

torch + 依赖约 2.5 GB，下载可与 T2 的构建/采集并行（纯网络 IO，不占 GPU）。

### T4 校验与导出（验收：`FAIL 0`）

```bash
python3 scripts/export_ncu_for_mac.py
```

- **`FAIL` 必须为 0。** 出现 FAIL 说明一份报告里混入了多个 kernel
  （`-k regex:` 没锚住），那份报告的任何跨版本对比都无效，必须改 regex 重采。
- `WARN` 可以有，但要逐条看：
  - 「报告为空」= 采集时 regex 没匹配到 kernel。`gemm` 的 cuBLAS 臂
    regex 是猜的（`(cutlass|ampere_|turing_|sm\d+_).*gemm`），抓不到属预期，
    用下面的办法查真实符号名后改 `gemm/project-proof/scripts/profile_ncu.sh`：
    ```bash
    nsys profile -o /tmp/g --trace=cuda gemm/build/gemm_bench
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv /tmp/g.nsys-rep \
      | grep -iE "gemm|cutlass"
    ```
  - 「多个 grid」= 两种成因，**必须人工判断**：
    (a) 多级算法的各级（归约树 `65536→256→1` 都是同一个 kernel）—— 合法，
        但读图时要知道只有第一级反映真实 HBM 行为；
    (b) 多 regime 混样（bench 扫了 decode/l2/prefill/hbm）—— **不可跨实例比较**，
        L2 区间与 HBM 区间的结论会翻转（4090 的 L2 是 72 MB，等效带宽超硬件峰值
        就是落在 L2 的信号）。属 (b) 的用 `NCU_SKIP` / `NCU_COUNT` 钉窗口重采。

### T5 带回主力机

```bash
# 采集主机上确认包已生成
ls -lh artifacts/ncu_for_mac/ncu_for_mac_all.tar.gz
# 主力机 / Mac 上取回
scp <user>@<采集主机IP>:~/Kernel_Optimazation/artifacts/ncu_for_mac/ncu_for_mac_all.tar.gz .
```

`.ncu-rep` 原件由 git 跟踪（各算子 `project-proof/profiling/ncu/`），
所以也可以直接 `git add` + `push`，主力机 `git pull` 拿。二选一即可，别两条都做。

**注意**：本轮新采的报告与仓内既有的 4070 Laptop 报告会混在同一个包里。
`MANIFEST.md` 的「采集批次」表会自动按 GPU / SM 数 / NCU 版本 / 日期分开，不用手工标。

## 5 已知陷阱（都实际踩过）

| 陷阱 | 表现 | 处理 |
|---|---|---|
| `-k regex:` 未锚定 | 一份报告混入多个 kernel，跨版本对比失效 | 用 `^` 锚定。`w8a8` 的 `v1_kernel` 是 `dequant_v1_kernel` / `gemv_v1_kernel` 的子串 |
| 空报告静默 | ncu 退出码 0，报告零实例 | 已在 `ncu_profile_lib.inc.sh` 里拦截并删除，不中止整轮 |
| `sm_75` 默认目标 | SASS 不是实际执行的码 | 显式 `-DCMAKE_CUDA_ARCHITECTURES=89`，构建后 `cuobjdump -lelf` 核对 |
| cuDNN 硬编码路径 | `softmax` configure 阶段失败 | 已改为多路径搜索；装 `libcudnn9-dev-cuda-12` |
| CMake 缓存 | 改了 `-DCMAKE_CUDA_ARCHITECTURES` 却不生效 | `rm -rf */build` 重来 |
| `nsys stats` 前导行 | 导出的 CSV 表头被两行进度提示污染 | 从 `^Time (%)` 起截断 |
| `nsys` 陈旧 `.sqlite` | 重导时静默产出**空文件**而非报错 | 加 `--force-export=true` |
| NCU 锁时钟 | 默认 `--clock-control base`，共享机器上会拖慢别人 | 独占机器可不管；共享时加 `--clock-control none` |
| 设备身份靠推断 | 把 Ryzen 笔记本上的报告当成 "4090" | 设备型号一律从报告 Session 页的 `display_name` 读，不从主机型号推断 |
| 缺 ninja / numpy | torch 扩展算子只采出第一个臂就静默退出 | 见 T3；跑完核对臂数 |
| 指标名不存在 | `lts__t_sectors_op_read` 在 raw 页 1037 列中**没有** | 实际为 `lts__t_sectors_srcunit_tex_op_read.sum`；且扇区计数只在 `--page raw --csv`，不在 details 页 |
| 分管线指标命名不统一 | 按 `sm__pipe_*_cycles_active` 去 grep FMA/ALU 会误判为"没采到" | Tensor 是 `sm__pipe_tensor_cycles_active`，FMA/ALU 是 `sm__inst_executed_pipe_fma/alu` |
| raw CSV 第 1 行是单位行 | 把 `Mbyte`/`us` 当数据行；或忽略单位导致数值差 10^6 | 解析时跳过第 1 行，并**读取单位列**——值已是 Mbyte / us，不要再换算 |
| 对照臂未被采集 | CUB 臂在 bench 里但 `PROFILE_TARGETS` 没列，CSV 又被实例上限截断 | 采集前核对：bench 里跑了几个臂，`PROFILE_TARGETS` 就该有几条 |
| 异算子当对照物 | `cublasSasum` 算 Σ\|x\|，与 reduce 的 Σx 不是同一算子 | reduce 的合法对照是 CUB `DeviceReduce::Sum`；LEDGER 红线第一条即因此作废过一次 |

## 6 不要做的事

- 不要在这台机器上改算法、调参、改版本梯。这里只采数据。
- 不要把这台机器的时延数字写进 `README.md` / `PORTFOLIO.md` / 简历。
  CUDA 版本不同，口径不一致。
- 不要删除或覆盖 `project-proof/data/` 下的任何文件。
- 不要为了让 `FAIL` 变 0 而放宽 `export_ncu_for_mac.py` 的校验。
  那两条校验（kernel 混入、报告为空）拦的都是会让结论失效的静默错误。
- 不要把 `artifacts/ncu_for_mac/reports/` 或 `*.tar.gz` 提交进仓（已 gitignore，是派生物）。

## 7 完成后

1. 主力机上 `git pull`，跑 `python3 scripts/export_ncu_for_mac.py` 重建清单。
2. 做**修复前 vs 修复后**的对照（见文末），回写 `docs/lectures/03_memory_bound_fusion.md`
   与 `docs/ncu_reading_guide.md` 中涉及字节账的段落。
3. 新增 `records/EXP-K09`（八节模板；K07 = 计数器闭环，K08 = 向量化修复，本轮接 K09）。
4. **删除本文件**，有效结论并入 `LEDGER.md`。

---

## 8 本轮与上一轮的差异（第二轮，开工前必读）

**第一轮（EXP-K07，2026-08-29）**解决的是"有没有计数器"——把权限打通，
把三条推断转成实测。那一轮采到的四个 torch 算子数据，是**向量化修复之前**的。

**第二轮**解决的是"修复之后账本怎么变"。EXP-K08 已在主力机上把 `BF16x8` 的
16 B 向量化真正兑现（SASS 五处 `LDG.E.128` 由 0 转正），四个算子的访存指令因此少了约 4 倍。
所以：

**四个 torch 算子（fused-norm / rope / activation / w8a8）的 NCU 必须全部重采**，
现有报告全是修复前的，不能混用。

### 8.1 最有价值的一组对照

`fused-norm` 修复前的三层扇区账（EXP-K07 §6.3，grid `(32768,1,1)`，S = 8,388,608 sector）：

| 层级 | v3 | v4 |
|---|---|---|
| L1TEX 读请求 | 16.000×S | 12.000×S |
| L2 读 | 2.126×S | 2.019×S |
| DRAM 读 | 2.001×S | 2.000×S（算法下界） |
| L1 命中率 | 86.87% | **83.19%** |
| L2 命中率 | 5.71% | **0.94%** |

**预期**：向量化兑现后 L1TEX 请求应从 12–16×S 降到约 3–4×S，
而 DRAM 读**必须仍然是 2.000×S**（算法下界不因向量化改变）。
于是 L1 命中率会显著下降——不是变差，而是过滤压力本来就小了。

**如果 DRAM 读偏离 2.000×S，说明修复引入了新问题**，这是本轮最重要的一道守卫。

### 8.2 两个容易读错的地方

- **L2 一直没参与**（命中率 0.94%）。收敛发生在 L1，不是 L2。
  讲义/指南里"被 L2 接住"的措辞已在 4ecea50 修正为 L1，读旧稿时注意。
- **"向量化兑现" ≠ "向量化有收益"**（EXP-K08 §6）：SASS 判据四个算子全翻正，
  性能只有 fused-norm 与 rope 翻正，activation 与 w8a8 ≈ 0。
  采到数据后不要因为 `LDG.E.128 > 0` 就断言收益。

### 8.3 GEMM / FA2 要不要重采

**不必**。EXP-K08 只动了四个 torch 算子的 `BF16x8`，没碰 gemm / flash-attn / 六个 C++ 算子。
第一轮的报告继续有效。若时间富余，可选做两件第一轮留下的开放项：

1. **gemv 34.1% 未复现**：第一轮 profiler 只钉住 bench 多个 size 中的一个（该点仅快 1.4%）。
   要剖清 34.1%，须用 `NCU_SKIP` / `NCU_COUNT` 钉到 bench 报出该数字的同一 size。
2. **cuBLAS kernel 选型**：第一轮本机（CUDA 12.8）选的是 `ampere_fp16_s1688gemm_fp16_128x128`，
   grid `(32,32,3)`（split-K）。若新机器 CUDA 版本不同，记下它选了什么——
   这决定 GEMM 85.6% vs 77.9% 的归因能否彻底闭合。
