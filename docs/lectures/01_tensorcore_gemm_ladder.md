# 01 · Tensor Core GEMM 版本梯:v0→v4 全程走读

> 对象:`gemm/src/` 五个 kernel(v0 naive → v1 smem tile → v2 wmma → v3 cp.async 双缓冲 → v4 128×128 大 tile),对照真 cuBLAS(`cublasGemmEx`,调用点验真)。
> 数字权威:`gemm/project-proof/data/derived_gemm4096_stability.csv`(3 轮 mean±std),实验记录 = records/EXP-K02_cuda_gemm_tc_ladder.md(下文简称 EXP-K02)。
> 协议:M=N=K=4096,fp16 存储 / fp32 累加,行主序,RTX 4090(sm_89)。

## 1 这一篇回答什么问题

这一篇把 GEMM 从 5.2 TFLOPS(v0)走到 133.1 TFLOPS(v4,真 cuBLAS 的 85.6%,EXP-K02 §5)的每一级增量拆成可核验的账:为什么 smem tile 化只有 +25%,为什么换 wmma 一步 13.8 倍,cp.async 的 commit/wait_prior 计数如何推,128×128 大 tile 的复用账怎么算,以及理论 occupancy 33% 全梯最低的 v4 为什么反而最快。读完你应当能:手推 compute-bound 判定与 tile 复用的算术强度公式;逐行解释 v3/v4 的异步流水线为什么正确;面对「occupancy 低是不是问题」「你和 cuBLAS 差在哪」这类追问给出有实验锚的回答。

## 2 直觉与第一性原理

先想没有这套东西的世界。GEMM 的原始定义是三重循环:C[i][j] = Σ_k A[i][k]·B[k][j]。4096³ 要做 2·4096³ ≈ 1.374×10¹¹ 次浮点运算(每输出 1 乘 1 加)。若每次乘加都从显存现取两个操作数,访存量是运算量的同数量级——显存带宽立即成为天花板,再多算力也白给。所以 GEMM 优化的全部主题只有两个:**让每个字节被算尽可能多次(复用)**,以及**让做乘加这件事本身尽可能便宜(指令)**。

一个日常类比:工厂流水线。v0 是每拧一颗螺丝跑一趟仓库;v1 是把一箱零件搬到工位旁(smem tile);v2 是把手动螺丝刀换成气动批(Tensor Core:一条指令做 4096 次乘加);v3 是安排专人在你干活时提前去搬下一箱(cp.async 预取);v4 是把工位加大,一箱零件能装配更多产品(大 tile 提高复用)。类比失效点:工厂里搬运工和装配工是不同的人,而 GPU 里 cp.async 的「搬运」占用的是同一批线程发出的异步拷贝引擎,「重叠」不是免费的并行,而是把 DMA 排进指令流后靠计数器等待——这正是 §3.3 要精确推导的部分;另一个失效点是「工位加大」在 GPU 上有硬约束(寄存器与 smem 预算),放大到某一步会把驻留 block 数压成 0,类比给不出这个折断点,§3.5 的驻留计算才给得出。

## 3 完整推导与机制

### 3.1 第一步账:判定 compute-bound——v1 注定只有 +25%

优化前先判定瓶颈类型(roofline 三步,docs/talk/whiteboard_card_roofline.md 口径):

1. 算术强度 I = FLOPs / 必需字节。4096³ fp16 GEMM:
   - FLOPs = 2·M·N·K = 2·4096³ ≈ 1.374×10¹¹;
   - 必需字节(每个输入/输出至少动一次)= (M·K + K·N + M·N)·2B = 3·4096²·2B ≈ 100.7MB;
   - I ≈ 1.374×10¹¹ / 1.007×10⁸ ≈ 1365 FLOP/B。
2. 平衡点 I* = 峰值算力 / 峰值带宽 = 165.2 TFLOPS / 1.008 TB/s ≈ 164 FLOP/B(4090,同卡口径)。
3. I ≈ 1365 ≫ I* ≈ 164,深度 compute-bound:只要复用做得起码及格,带宽就不是限制,收益必须从指令侧找。

这笔账直接预言了 v0→v1 的结局。v0 复用为零,全局请求总量上界为 2·M·N·K·2B ≈ 275GB(每次乘加各取一个 A、B 元素);若全部打到 DRAM,1.008 TB/s 要 273ms,实测只有 26.369±0.472ms(EXP-K02 §5)——说明 L1/L2 已经隐性提供了约 10 倍复用,v0 并不是纯 DRAM 饥饿(账面推断)。v1 把复用显式化:32×32 tile 里每个元素装载一次供 32 个线程使用,全局读降到 (M/32)(N/32)·2·32·K·2B ≈ 8.6GB,摊到实测 21.114ms 上仅约 408 GB/s,远低于带宽峰值;且 A、B 各 32MB,合计装得进 4090 的 72MB L2,DRAM 强制流量只有 ~100MB。**v1 不缺带宽,慢在算力**:fp16 输入在 CUDA core 上算,每产出 2 FLOP 要 2 条 half→float 转换加 1 条 FFMA(指令路径账,推断级);EXP-K02 §6 的实测口径是「~6.5 TFLOPS 已近该路线实际上限」。所以 v0→v1 只有 +25%(5.2→6.5),而这 +25% 恰是本梯最重要的测量之一:它证明访存微调在这个算子上没有出路,台阶只能来自指令世代。

### 3.2 wmma fragment 模型:13.8 倍的台阶从哪来

`wmma::mma_sync` 是 warp 级协作指令:一个 warp 的 32 个线程共同完成一次 16×16×16 的小矩阵乘加 D = A·B + C,即 16³ = 4096 次乘加 = 8192 FLOP——一条指令顶 v1 内层循环的数千条标量指令,取指/发射/转换开销全部摊薄。这就是 v1→v2 实测 13.8 倍(6.5→89.5 TFLOPS,EXP-K02 §5)的来源;作为对照,v2 刻意保留了与 v1 同级的朴素同步装载(gemm_v2.cu:7),使这一级的唯一变量就是指令世代。

fragment 是 wmma 的数据模型,三条性质决定了整个 API 的能与不能:

1. **fragment 是 warp 级寄存器容器**:一个 16×16 fp32 accumulator 的 256 个元素分布在 32 个 lane 的寄存器里,每 lane 摊 8 个(256/32)——这也是后面算寄存器账的依据。
2. **lane→元素映射是编译器/架构私有的**:你只能通过 `load_matrix_sync`/`store_matrix_sync` 整块进出,不能问「我这个 lane 拿的是第几行第几列」。
3. **唯一可依赖的对称性**:同 shape 的 accumulator fragment 在同架构上映射一致,所以对两个同 shape fragment 做逐元素运算(如 fp32→fp16 转换、逐元素加)是合法的(gemm_v2.cu:17-21 面试点②)。

性质 2 对 GEMM 无害(GEMM 不需要知道元素在哪一行),对 FA2 是致命税负(行级 softmax 必须知道行号)——这是第二篇讲义的主线,伏笔埋在这里。

### 3.3 cp.async 组语义逐行推:commit / wait_prior 的在途计数

v2 的问题:每个 K 块「同步装载 → 计算」串行,Tensor Core 在装载期间空等。v3 引入 `__pipeline_memcpy_async`(编译为 cp.async:全局内存→smem 的异步拷贝,不经寄存器中转),配 smem 双缓冲 `As/Bs[2][...]`。其正确性建立在一套「组计数」上,值得逐条推:

- `__pipeline_memcpy_async(dst, src, 16)`:发出一条 16B 异步拷贝(16B 是 cp.async 最大粒度,兼作 float4 对齐要求),只是入队,不等完成。
- `__pipeline_commit()`:把此前发出、尚未封组的所有异步拷贝封成**一个组**,组按 commit 顺序排成 FIFO。
- `__pipeline_wait_prior(N)`:阻塞到「最新 N 个组之外」的所有组完成。N=0 即清空全部在途。

稳态归纳(gemm_v3.cu:15-22 文件头的展开):

- **序幕**:循环前先为 buf[0] 发出并 commit 第 0 组。归纳不变量:进入第 t 轮时,buf[p] 对应的第 t 组已在途(或已完成)。
- **循环体**:先为 buf[p^1] 发出并 commit 第 t+1 组(预取下一 K 块),然后 `wait_prior(1)`——此刻在途组至多两个:{第 t 组(老),第 t+1 组(新)};「最新 1 组」恰是刚发的第 t+1 组,被等掉的正是当前要消费的第 t 组。计算段随后消费 buf[p],与第 t+1 组的 DMA 重叠。
- **末轮**:不再发新组(否则越界读,且多出一组使计数指错对象),`wait_prior(0)` 清空。
- **危险面**:正确性完全押在「每轮恰好 commit 一组」的节奏上。多发或漏发一次,wait_prior 的计数就会指错组——等到的可能是还没搬完的 tile,读出未定义数据。这是 cp.async 手工调度最易错的地方(gemm_v3.cu:21-22)。

实测 v2→v3 只有 +6.7%(89.5→95.5,EXP-K02 §5)。不是白做:v2 已经 compute-bound,装载在总时间里本来占比有限,重叠能省的就这一点;这一级的教学价值在于对照 v4——**复用(+39%)比重叠(+6.7%)值钱**,顺序不能反。

### 3.4 128×128 大 tile 的复用账:+39% 值钱在哪

tile 复用的核心公式:一个 K 块内,block 从 smem 装载 (BM+BN)·BK 个元素,喂 2·BM·BN·BK 次浮点运算,所以**每装载一个元素支撑的 FLOP = 2·BM·BN/(BM+BN)**——只与 tile 的「面积/周长」有关,与 BK 无关。

- v2/v3(64×64):2·64·64/128 = 64 FLOP/元素;
- v4(128×128):2·128·128/256 = 128 FLOP/元素,翻倍。

全局流量同理:总全局读 = M·N·K·2B·(1/BM + 1/BN),v2/v3 ≈ 4.3GB,v4 ≈ 2.15GB,减半(大部分由 L2 供给,DRAM 强制流量不变,账面推断)。fragment 层的复用同步升级:v4 每个 kk 步 6 次 `load_matrix_sync`(4 个 af + 2 个 bf)喂 8 次 `mma_sync`,af 各复用 2 次、bf 各复用 4 次;v2 是 4 次 load 喂 4 次 mma。两层复用叠加,实测 v3→v4 +39%(95.5→133.1,EXP-K02 §5),是版本梯对 cuBLAS 差距的主要收口。

为什么 BK 停在 32 不再加大:上面公式已示算术强度与 BK 无关;BK 翻倍只会让 smem 从 32KB 冲到 64KB,把每 SM 驻留 block 从 2 压到 1,纯亏(gemm_v4.cu:23-24 面试点②)。为什么不再放大 BM/BN:每 warp 的 accumulator fragment 数量随 tile 面积增长,寄存器先爆(见下节账),这是「工位加大」类比折断的地方(推断级,本梯未做 256 级别的对照臂)。

### 3.5 occupancy 33% 全梯最低却最快:驻留计算与 ILP

资源画像(ptxas -v 实测,`gemm/project-proof/data/ptxas_resource_usage.txt`,EXP-K02 §5):

| 版本 | regs/thr | smem | 线程/block | 每 SM 驻留 block(限制因子) | 理论 occupancy |
|---|---|---|---|---|---|
| v2 | 54 | 8KB | 128 | 9(regs) | 75% |
| v3 | 61 | 16KB | 128 | 6(smem) | 50% |
| v4 | 92 | 32KB | 256 | 2(regs) | 33% |

v4 的驻留计算逐步做一遍:每 block 寄存器 = 92×256 = 23552;每 SM 寄存器文件 64K 个,⌊65536/23552⌋ = 2 block(smem 侧 32KB 允许 3 个,所以限制因子是寄存器)。2 block × 8 warp = 16 warp,除以 sm_89 每 SM 上限 48 warp,理论 occupancy = 33%,全梯最低。其中寄存器大头有账可查:8 个 fp32 accumulator fragment × 每 lane 8 个元素 = 64 个寄存器/线程,占 92 的七成——「大 tile」的代价直接写在寄存器文件里。

为什么最低的 occupancy 反而最快?occupancy 买的是**用别的 warp 顶上来遮蔽延迟**(TLP)。但延迟遮蔽的来源不止这一个:**同一 warp 内足够多的无依赖指令**(ILP)同样能把流水线填满。v4 的 8 个 accumulator 互不依赖,8 条 mma 可以背靠背发射,前一条没算完不妨碍下一条进管线;加上 af/bf 的多次复用摊薄了 smem 读,Tensor Core 的供数与发射两头都被 fragment 级 ILP 喂饱——它不需要靠换 warp 过日子。结论:**occupancy 是手段,不是目标**(EXP-K02 §6);判断标准是「延迟是否被遮蔽」,不是「线程是否足够多」。反例在第二篇讲义:FA2 被 90.75KB smem 钉死在 1 block/SM 时,加 warp(v3,+33%)就是唯一的放大招——同一个指标,两种读法,取决于瓶颈在哪。

## 4 代码逐段走读(按执行顺序)

### 4.1 v0:正确性锚与性能分母(gemm/src/gemm_v0.cu:13-25)

```cuda
__global__ void gemm_v0_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;   // 末块越界线程直接退出:各输出互不相交,无部分和要合并
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        // B[k*N+col]:warp 内 col 连续 → 按行合并;A[row*K+k]:warp 内同 row 广播。
        // 问题不在合并度,在复用为零:同一 A/B 元素被不同 block 反复从
        // DRAM/L2 拉取——这笔带宽账引出 v1 的 smem tile。
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    C[row * N + col] = __float2half(acc);   // fp32 累加全程保精度,仅写回舍入一次
}
```

角色:版本梯的分母,一线程一输出、肯定对。关键行:L17 的提前 return 之所以安全,是因为每个输出互不相交、没有部分和要跨线程合并(对比 FA2 v1 里越界线程必须活着陪跑 barrier);L23 的访存模式其实已经合并(col 连续),v0 的病不在合并度而在零复用。改错会怎样:去掉 guard,尾块线程越界读写;把 acc 换成 half 累加,4096 长点积的舍入误差会随 K 累积(大数吃小数),正确性 gate 未必再过。

### 4.2 v1:把复用显式化,并引入 barrier 纪律(gemm/src/gemm_v1.cu:24-35)

```cuda
    for (int k0 = 0; k0 < K; k0 += T) {
        // 装载:两条读 tx 连续 → 全局访存均按行合并
        As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
        __syncthreads();   // 防 tile 未装满即有线程开读:内积要读整行/列,
                           // 大部分由别的线程装载(跨线程 RAW)
        #pragma unroll
        for (int k = 0; k < T; ++k)
            // As[ty][k]:warp 内同址 → smem 广播;Bs[k][tx]:tx 连续,行内顺序访问
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();   // 防快线程进入下一 k0 覆盖 As/Bs 时,慢线程仍在读旧 tile(WAR)
    }
```

角色:smem tile 的教科书形态,每元素装载一次供 32 线程复用,全局访存 /32。两个 `__syncthreads` 各防一条竞态,方向相反:第一个防「读到没装完的」(跨线程 RAW),第二个防「快线程下一轮覆盖时慢线程还在读上一轮」(WAR)。改错会怎样:删第二个 barrier,错误只在个别调度时序下出现,是典型的「偶发错一位」难查 bug;这套 RAW/WAR 注释纪律贯穿后面所有版本。实测只 +25%(§3.1 的账),但它把「访存问题」与「算力问题」拆成两个独立变量,是控制变量设计,不是失败。

### 4.3 v2 装载段:float4 协同搬运(gemm/src/gemm_v2.cu:45-56)

```cuda
        for (int t = threadIdx.x; t < BM * BK / 8; t += blockDim.x) {
            int r = (t * 8) / BK, c = (t * 8) % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(bm + r) * K + k0 + c]);
        }
        for (int t = threadIdx.x; t < BK * BN / 8; t += blockDim.x) {
            int r = (t * 8) / BN, c = (t * 8) % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + bn + c]);
        }
        __syncthreads();   // 防 As/Bs 未写全即被读:装载按线性 tid 分片、
                           // 消费按 warp 分块,线程集不重合(跨线程 RAW)
```

角色:把「谁算」与「谁搬」解耦——装载按线性 tid 分片(全 block 128 线程摊 512 条 float4),消费按 warp 分块(每 warp 一个 32×32 子区),中间靠 barrier 交接。关键行:一条 float4 = 8 个 half = 16B;c 恒为 8 的倍数且 K%32==0 保证全局地址 16B 对齐——float4 的硬性要求(gemm_v2.cu:42-44 注释)。改错会怎样:把对齐前提破坏(如 K=4090),reinterpret_cast 的 128-bit 访问直接 misaligned address 错误;忘了 barrier,wmma 读到装了一半的 tile,结果错而不崩,是最恶劣的一类错。

### 4.4 v2 计算段:2×2 微内核,fragment 复用的起点(gemm/src/gemm_v2.cu:57-75)

```cuda
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // [计算段] 2x2x2 微内核:af[i]/bf[j] 各 load 1 次、各参与 2 次
            // mma_sync——fragment 级数据复用的起点(v4 扩到 4x2)。
            // af/bf 均 row_major(A、B 本就按行存),leading dim = 所在 tile 行宽。
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wr * 32 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
```

角色:版本梯最大台阶(13.8 倍)的现场。关键行:`load_matrix_sync` 的第三个参数是 leading dimension(源矩阵行宽),af 从 As 读用 BK、bf 从 Bs 读用 BN——写错不报错,只给你一片错位的数据;`mma_sync(acc, af, bf, acc)` 的第四参即累加输入,acc 全程驻寄存器不落 smem。改错会怎样:leading dim 传错是 wmma 新手第一大坑(结果全错但不崩);把 acc 声明挪进 kk 循环内,每次清零,等于只算最后一个 K 块。

### 4.5 v3:异步装载器与调度骨架(gemm/src/gemm_v3.cu:31-44、59-69)

```cuda
__device__ __forceinline__ void load_tile_async(
    half (*As)[BK], half (*Bs)[BN],
    const half* A, const half* B, int M, int N, int K,
    int bm, int bn, int k0, int tid, int nthr) {
    for (int t = tid; t < BM * BK / 8; t += nthr) {
        int r = (t * 8) / BK, c = (t * 8) % BK;
        __pipeline_memcpy_async(&As[r][c], &A[(bm + r) * K + k0 + c], 16);
    }
    for (int t = tid; t < BK * BN / 8; t += nthr) {
        int r = (t * 8) / BN, c = (t * 8) % BN;
        __pipeline_memcpy_async(&Bs[r][c], &B[(k0 + r) * N + bn + c], 16);
    }
    __pipeline_commit();   // 本 tile 封组:调用一次 = 恰好一组,组节奏见文件头
}
```

```cuda
    // 序幕:先发第 0 组,循环内的 wait_prior 才有对象;首轮无重叠(冷启动税只付一次)
    load_tile_async(As[0], Bs[0], A, B, M, N, K, bm, bn, 0,
                    threadIdx.x, blockDim.x);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)                       // 末轮不预取:越界读 + 多出一组破坏 wait 计数
            load_tile_async(As[p ^ 1], Bs[p ^ 1], A, B, M, N, K,
                            bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);   // 留 1 组(刚发的预取)在途,只等当前块;末轮清空
        __syncthreads();   // cp.async 的完成只对发起线程可见,而 tile 由全 block
                           // 分片搬运:barrier 后任何 warp 才能读到别的线程搬的段(跨线程 RAW)
```

角色:§3.3 推导的代码形态。关键行:`load_tile_async` 每次调用恰好 commit 一组——这个不变量是整套计数正确的前提;`wait_prior` 之后还要 `__syncthreads`,因为 cp.async 的完成只对发起线程可见,而 tile 是全 block 分片搬的,别的 warp 要靠 barrier 才能看到你搬的那段。改错会怎样:把 1 写成 0,连刚发的预取组一起等,重叠归零、性能退回 v2(正确但慢,最隐蔽);末轮照常预取,越界读加计数错乱(错误数据);在装载器里多调一次 commit,所有 wait_prior 集体指错组。

### 4.6 v4 主循环:全部要素合流(gemm/src/gemm_v4.cu:66-91)

```cuda
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)                    // 末轮不预取(越界 + 破坏组计数)
            load_tile_async4(As[p ^ 1], Bs[p ^ 1], A, B, N, K,
                             bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);   // 留预取组在途只等当前块;末轮清空(推导见 v3)
        __syncthreads();   // cp.async 完成仅发起线程可见 → barrier 后全 warp 才能读全 tile(跨线程 RAW)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // 4x2x2 微内核:6 次 load 喂 8 次 mma;af[i] 复用 2 次、bf[j] 复用 4 次
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(af[i], &As[p][wr * 64 + i * 16][kk], BK);   // wr*64:每 warp 管 64 行
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[p][kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();   // 防下一轮 cp.async 覆盖 buf[p] 时慢 warp 仍在读(WAR,同 v3)
        p ^= 1;
    }
```

角色:133.1 TFLOPS 的主体。8 warp 按 2×4 排布,每 warp 输出 64×32 = 4×2 个 fragment 常驻寄存器整个 kernel 生命周期(gemm_v4.cu:10-14 小图)。关键行:第二个 `__syncthreads` 防的竞态很微妙——双缓冲只隔离「计算 vs 在途预取」,不隔离「本轮读 buf[p] vs 下一轮 cp.async 写同一 buf[p]」(翻面之后 p 就成了预取目标),所以计算完还要一个 WAR barrier。改错会怎样:删它,大多数时序下结果还对,压力大时偶发错——比一直错更难修;把 `p ^= 1` 忘掉,双缓冲退化为单缓冲且计算读的永远是同一面,结果稳定地错。

### 4.7 对照物:真 cuBLAS 与行主序技巧(gemm/src/gemm_cublas.cu:14-22)

```cuda
static cublasHandle_t handle = nullptr;
void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K) {
    if (!handle) cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                 &beta, C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
```

角色:一切「vs cuBLAS」数字的对照物,155.4±0.62 TFLOPS(EXP-K02 §5)。关键行:cuBLAS 只认列主序,行主序的 C = A·B 在列主序视角下等价于 C^T = B^T·A^T,所以以 (N, M, K) 调用并交换 A/B 指针,零转置零拷贝;handle 懒建后进程内复用——cublasCreate 含上下文与 workspace 初始化(百 ms 级),每次重建会把初始化开销计进被测时延,对照失真。为什么强调「真」:本仓 softmax 项目曾有对照物经源码核查系自写 kernel(cuBLAS 并无 softmax API),相关对比句整体撤销(EXP-K01 §5)——此后立的规矩是凡「vs cublas」先验调用点,本文件就是被验对象。改错会怎样:不交换 A/B 直接按行主序传参,数值全错;compute type 用 16F,对照口径与版本梯(fp32 累加)不再可比。

## 5 实验数据怎么读

现行数字(`gemm/project-proof/data/derived_gemm4096_stability.csv`,3 轮 mean±std,EXP-K02 §5):

| 版本 | latency (ms) | TFLOPS | vs cuBLAS | 逐级归因 |
|---|---|---|---|---|
| v0 naive | 26.369±0.472 | 5.2±0.12 | 3.4% | — |
| v1 tile | 21.114±0.047 | 6.5±0.00 | 4.2% | smem tiling 仅 +25% |
| v2 wmma | 1.536±0.008 | 89.5±0.46 | 57.6% | Tensor Core ×13.8 |
| v3 dbuf | 1.439±0.007 | 95.5±0.49 | 61.4% | 双缓冲 +6.7% |
| v4 bigtile | 1.033±0.007 | 133.1±0.97 | 85.6% | 大 tile +39% |
| cublas(真库) | 0.884±0.004 | 155.4±0.62 | 100% | — |

**轴与口径**。TFLOPS = 2·M·N·K / 时延(gemm/src/main.cu:92,「每输出 1 mul + 1 add」口径);「3 轮」指三次独立进程运行(raw 各一份,UTC 前缀落盘),±号后是轮间 std;每轮的时延本身已是 3 次预热后 50 iters 的均值(v0/v1 慢版取 iters/10、下限 3,main.cu:105-107)。README 图 1(figures/01_gemm_tc_ladder.png)就是此表的水平条形图:横轴 TFLOPS,误差条 = 轮间 std,条上百分比 = vs cuBLAS;标题即结论句,脚注给源数据文件——图不携带表之外的信息,存疑时回 CSV。

**这个实验设计防了哪些坑**。预热 3 次驱走冷时钟与懒初始化(main.cu:109);计时用单 event 对包住整段循环再除 iters(main.cu:110-113),避免逐次 event 记录本身的开销偏置;固定 srand(42) 使所有版本所有轮吃同一输入(main.cu:53-55),否则「版本差异」里混着「输入差异」;正确性以 cuBLAS 输出为参考,相对误差分母用全局 absmax 而非逐元素(近零元素上逐元素相对误差无意义地爆炸),抽样步长取素数 997/97 避开 2 的幂结构的周期性采偏(main.cu:63-71、100-103);结果只写 BENCH_OUT 指定的新文件、首行 provenance(main.cu:77-90),历史数据永不覆盖。慢版少跑的合法性有数据支撑:v1 的 std≈0(21.114±0.047),5 iters 统计已够(EXP-K02 §7)。

**数字背后的机理账**。核验 133.1:1.374×10¹¹ FLOP / 1.033×10⁻³ s ≈ 1.33×10¹⁴ = 133 TFLOPS,自洽;对 4090 的 165.2 TFLOPS 峰值即 81%(docs/talk/whiteboard_card_roofline.md 口径),cuBLAS 155.4 约为峰值 94%(账面);带宽侧远未触顶(§3.1),所以逐级增量全部应从指令与复用侧解释,与归因列一致。正确性列还有一个值得读的细节:v0-v4 的 max_rel_err 全部等于 7.58e-04——误差被输入/输出的 fp16 舍入主导,与累加顺序无关(EXP-K02 §5),这是「fp32 累加口径下 fp16 存储误差有多大」的直接测量。

## 6 误区与边界

**误区 1:「occupancy 越高越好」。** v2 理论 occupancy 75% 却只有 89.5 TFLOPS,v4 以 33% 拿到 133.1(EXP-K02 §5)。occupancy 买的是 TLP 遮蔽,而 Tensor Core 吞吐靠 fragment 级 ILP 与复用喂满(§3.5)。但注意适用边界:这不是「occupancy 无用论」——FA2 v3 在 1 block/SM 的约束下加 warp 直接 +33%(EXP-K03),延迟遮蔽缺口真实存在时 occupancy 就是要害。先判断延迟是否已被遮蔽,再谈 occupancy。

**误区 2:「smem tiling 是 GEMM 优化的大头」。** 教科书顺序造成的错觉。本梯实测:tiling +25%,换指令世代 13.8 倍(EXP-K02 §5)。方向由 bound 类型决定:compute-bound 的 GEMM 里访存微调是坡,指令世代是台阶;memory-bound 的 gemv/reduce 里恰好反过来,指令层面微调收益趋近 0(PORTFOLIO「先判定 memory-bound 还是 compute-bound」节)。错配的优化在错误的方向上没有回报。

**误区 3:「对照物可以信文件名,单轮数字可以进结论」。** 本仓两个被证伪的实例,是最硬的教学材料:其一,softmax 曾有的对照库对比,对照物经源码核查系自写 warp 原语 kernel(cuBLAS 无 softmax API),整条对比链撤销(EXP-K01 §5)——凡「vs X」先验 X 的调用点,gemm_cublas.cu 的验真就是这条规矩的产物;其二,gemv 单轮曾测得领先 84%,让对照物同样跑 3 轮后回落到 37.8%——领先幅度的一半来自 cuBLAS 侧单轮波动(EXP-K01 §7,详见第二篇讲义附课 B)。自家 kernel 稳定不代表对照稳定。

**误区 4:「异步预取/双缓冲是万能提速键」。** v3 仅 +6.7%:compute-bound 下装载占比本来就小,重叠的上限就是这块占比。且 wait_prior 参数写保守(0)不会报错,只会把重叠静默归零——「没错但没用」的优化比报错更浪费时间。复用优先于重叠(+39% vs +6.7%)。

**边界声明**:本梯所有结论的实测范围是 4096³ 单一 shape、fp16 存储 / fp32 累加、行主序、RTX 4090;v1/v2/v4 有尺寸整除前置条件(gemm/include/gemm_common.h:8-13),无尾块处理——目标是归因,不是产品化。85.6% 不是「追平 cuBLAS」,措辞以 gemm/README.md 的约束表为准;剩余约 14% 差距的去向(smem swizzle、多级流水、tile 形状选择)为推断级,NCU 计数器在本容器不可用,未做计数器级确认(EXP-K02 §6)。

## 7 连环追问

**Q1:mma_sync 一条指令做多少 FLOP?**
16×16×16 tile 的 D = A·B + C:16³ = 4096 次乘加 = 8192 FLOP,由一个 warp 的 32 线程协作完成。对比:v1 内层每条 FFMA 只贡献 2 FLOP,还要配两条 half→float 转换。

**Q2:v0→v1 只有 +25%,是不是 v1 写坏了?**
不是。带宽账(§3.1)显示 v1 只用了约 408 GB/s,远未触顶;瓶颈在 CUDA core 算力(fp16 输入的转换 + FFMA 指令路径),EXP-K02 §6 口径「6.5 TFLOPS 已近该路线实际上限」。tiling 本身是对的,只是这个算子的病不在这。

**Q3:BK 为什么取 32,不是 16 或 64?**
复用公式 2·BM·BN/(BM+BN) 与 BK 无关(§3.4)。BK=16 使 __syncthreads 频率翻倍;BK=64 使 smem 翻倍、驻留 block 减半,复用却不增(gemm_v2.cu:24-25、gemm_v4.cu:23-24)。32 = 同步开销与 smem 预算的平衡点。

**Q4:__pipeline_wait_prior(1) 的「1」精确指什么?写 0 会怎样?**
「允许最新 1 个组仍在途」——那恰是刚为下一 K 块发出的预取组;被等掉的是当前块。写 0 = 连预取组一起等,重叠归零,性能退回 v2,而结果完全正确——最难发现的一类性能 bug(§3.3)。

**Q5:双缓冲已经隔离了读写,为什么每轮还要两个 __syncthreads?**
第一个:cp.async 完成只对发起线程可见,tile 由全 block 分片搬运,别的 warp 要靠 barrier 看到你搬的段(RAW)。第二个:双缓冲隔离的是「计算 vs 在途预取」,不隔离「本轮读 buf[p] vs 下轮 cp.async 写 buf[p]」——p 翻面后就成了预取目标(WAR,gemm_v3.cu:86-88)。

**Q6:accumulator 为什么必须 fp32?**
fp16 尾数 10 位,4096 长点积的部分和量级 ~O(20)(输入取 (-1,1) 零均值,main.cu:51),继续累加时小增量会被大部分和吃掉(舍入),误差随 K 累积。fp32 累加下实测 max_rel_err = 7.58e-04,且由 I/O 舍入主导(EXP-K02 §5)。

**Q7:写回时对 fragment 逐元素做 fp32→fp16 转换,凭什么合法?**
wmma 唯一公开承诺的对称性:同 shape 的 accumulator fragment 在同架构上 lane→元素映射一致,所以「fp32 acc 逐元素转入 fp16 acc」这种同位置操作合法(gemm_v2.cu:17-21)。但映射本身仍是黑箱——你不知道 x[e] 是第几行第几列,这正是 FA2 做不了行级 softmax 的根源。

**Q8:每 SM 只驻 2 个 block 是谁限的?算一遍。**
92 reg × 256 thr = 23552;⌊65536/23552⌋ = 2(寄存器限)。smem 侧 32KB 允许 3 个,不是瓶颈。2 block × 8 warp = 16 warp / 48 = 33%(§3.5)。其中 64 reg/thr 是 8 个 accumulator fragment 的账。

**Q9:为什么不把 tile 再放大(如 256×128)?**
复用公式还会涨,但 accumulator fragment 数量随面积线性涨,寄存器先爆(92 已用去 64 个在 acc 上);smem 也要翻倍。驻留 block 压到 0 就发射不出去。此为账面推断——本梯未做 256 级对照臂,cuBLAS 的多形状 tile 选择正是它领先的候选来源之一(EXP-K02 §6,推断级)。

**Q10:cuBLAS 对照是怎么「验真」的,为什么较真?**
源码核查调用点:gemm_cublas.cu 有 cublasCreate + cublasGemmEx,确系真库(EXP-K02 §2)。较真的原因是本仓 softmax 的教训:对照物系自写 kernel,整条对比叙事作废(EXP-K01 §5)。对照物命名诚实是所有「vs X」数字的地基。

**Q11(压力):85.6% 会不会只是 4096³ 一个点的幸运?**
诚实回答:是单 shape 实测,不外推。非方阵/LLM 实际形状(如小 M 大 N 的 decode 形状)未测,列为后续工作(EXP-K02 §7);wave quantization、tile 形状选择在别的 shape 上会改变双方相对位置。可以说的是:同协议同 harness 下 3 轮稳定(±0.97 TFLOPS),这个点本身立得住。

**Q12(压力):你说剩余差距在 swizzle/多级流水,证据呢?**
诚实回答:推断级,不可当实测说(gemm/README.md 约束表)。NCU 计数器在本容器不可用(EXP-K01 §7),没有 bank conflict 计数的直接证据。旁证有二:自家 Triton 版同尺寸 160.5 TFLOPS(triton-kernels 仓,跨 harness,其 harness 下 cuBLAS=159.8,本 harness 155.4,差约 3%),Triton 编译器发射 mma+ldmatrix+swizzle 而 wmma 不暴露 swizzle,差距方向与假设一致;结构分析上 wmma 固定布局的 smem 访问模式无法错位。检验方式明确:v5 用 mma PTX + ldmatrix + 手工 swizzle 重写,差距收窄即证实(EXP-K02 §7)。

## 8 工业对照与延伸

与生产实现的差距,逐层定位:

- **CUTLASS(cuBLAS 内核的开源近亲)**:三层 tile 化(threadblock / warp / instruction)与本梯同构,但多出:>2 级软件流水(multistage,cp.async 组深度 3-5,本梯只有双缓冲 2 级)、smem swizzle 布局(消 bank conflict,wmma API 不暴露)、多 tile 形状模板按 shape 启发式选择(本梯单一 128×128)、split-K 与 epilogue 融合。差距集中在 instruction/warp 层的布局控制,不在算法。入口:`include/cutlass/gemm/threadblock/mma_multistage.h`(流水线)、`media/docs/efficient_gemm.md`(官方分层讲解)。
- **Triton**:`tl.dot` 自动编译到 mma.sync + ldmatrix + swizzle,程序员只写 tile 逻辑。自家 Triton 版 160.5 TFLOPS(跨 harness,推断级)对本梯 v4 的 ~17% 领先,即「编译器代管布局」对「wmma 固定布局」的溢价(EXP-K02 §6)。
- **cuBLAS/cuBLASLt**:运行时按 shape/arch 启发式选 kernel;155.4 TFLOPS = 峰值 94% 是它在本协议点的位置。手写单 shape 追到 85.6% 的含义:通用库的领先主要是布局与流水线工程,不是不可知的魔法。
- **世代边界**:cp.async 是 Ampere/Ada 的机制;Hopper 换 TMA(张量内存加速器)+ warp specialization(生产者/消费者 warp 分工),DeepGEMM、CUTLASS 3.x 的 Hopper kernel 即此路线——本讲义的组计数推导在 Hopper 上对应 mbarrier 事务计数,概念同构、原语不同。

延伸阅读(带锚):

1. `gemm/src/gemm_v3.cu:15-24` 与 `gemm/src/gemm_v4.cu:20-25` 文件头——cp.async 组语义与 occupancy/ILP 的仓内一手推导。
2. records/EXP-K02_cuda_gemm_tc_ladder.md §6——H1/H2 假设的判定与 Triton 对照的完整口径。
3. CUDA C++ Programming Guide「Warp Matrix Functions」章——wmma fragment 的官方契约(映射不透明即在此声明)。
4. PTX ISA「Data Movement and Conversion Instructions: cp.async」章——commit-group/wait-group 的硬件语义,§3.3 的权威版本。
5. CUTLASS `media/docs/efficient_gemm.md`——工业级 GEMM 的分层设计,对照本梯看「还差哪几层」。
