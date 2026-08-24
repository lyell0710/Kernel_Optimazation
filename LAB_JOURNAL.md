# LAB_JOURNAL — Kernel_Optimazation

## §1 4090 重基准 + roofline 迁移(2026-08-23,EXP-K01)

- **做了什么**:四 kernel(reduce/softmax/gemv/int8-quantize)在 4090 全量
  重跑既有 bench(未改源码),与 git HEAD 里的 4070-Laptop CSV 逐版本对照;
  重生成全部图。
- **为什么**:阶段一清单"简历数字换桌面卡 + 记录 roofline 位置迁移"。
- **关键数字**:**reduce v6/v7 排序反转**(Laptop 回退版 → 4090 最优且反超
  cuBLAS 25%);softmax v4 快 cuBLAS 26%(aligned)/34%(mis);gemv 84%
  (含 cuBLAS gemv 本卡表现平平的对照物因素,如实标注);int8 仓内 baseline
  系 CPU 口径,PyTorch-eager 4090 对照移 triton-kernels#EXP-T03。
- **受阻与整改**:NCU 无计数器权限(容器 ERR_NVGPUCTRPERM,细节重过降级
  为 Laptop ncu-rep 参照);本仓 bench 覆盖写 CSV 违 CORE bench 铁则,
  本次以提交对作版本锚,harness 改造列为后续整改。
- **产物**:records/EXP-K01、四项目新 CSV/stability/图(本提交),
  旧值锚 = 上一提交。
- **下一步**:triton-kernels 仓(FA2/双缓冲 GEMM/三件套移植)开发中,
  Triton vs CUDA 对比将回引本仓同尺寸数字。

## §2 红线级勘误:softmax 的"cuBLAS"对照系自写 kernel(2026-08-24)

- **发现(用户审计)**:softmax_cublas.cu 无任何 cuBLAS 调用,是自写
  warp 原语 kernel(注释自曝);cuBLAS 本无 softmax API。
- **影响与处置**:"softmax 快 cuBLAS 26%/34%"全链作废(EXP-K01 表格、
  README、PORTFOLIO 归因叙事加勘误横幅);简历禁用该句。gemv/reduce 的
  cublas 验真为真库调用,对照有效。
- **教训**:对照物命名必须开源码验真,不能信文件名——CORE 铁律 6 的
  反面教材实锤;后续任何 "vs X" 声明先查 X 的调用点。
