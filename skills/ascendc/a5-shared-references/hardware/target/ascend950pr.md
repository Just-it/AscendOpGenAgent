---
platform: Ascend950PR
type: target
verified: true
cann_version: 9.0.T501
soc_version: Ascend950PR_9589
npu_arch: 3510
arch_codename: David V100 (DaVinci V351)
date_verified: 2026-03-28
---

# Ascend950PR (David V100) -- Target Platform

> The only Ascend chip supporting both SIMT and SIMD programming models.
> All data verified on A5 server (90.90.93.35) container can_torch_cann_device_1
>
> **完整版（含 PDF 手册页码）**: `docs/archive/A5_HARDWARE_DETAILS.md`
> **PDF 原始手册**: `/mnt/d/workspace/ai/a5/DavidV100用户手册*.pdf`
> unless otherwise noted.

## Compute

| Parameter | Value | Source |
|-----------|-------|--------|
| AICore total | 28 (PG variant, npu-smi confirmed) | npu-smi |
| AIV per AICore | 2 | CANN GetAicAivTaskRation() |
| **AIV total** | **56** | 28 x 2 |
| AIC total (Cube) | 28 | 28 x 1 |
| Warp size | 32 | simt_stub.h |
| Max threads / block (LAUNCH_BOUND) | 512 (typical), 1024, 2048 | DavidV100 manual |
| Max threads / AIV | 2048 | DavidV100 manual |
| Warp schedulers / AIV | 4 | DavidV100 manual |
| Register file / AIV | 128KB (64 reg/thread at 512 threads) | DavidV100 manual |
| Clock frequency | 1.65 GHz | DavidV100 manual |
| Block scheduling | **Software time-slicing** (measurable overhead vs GPU HW scheduler) | A5 benchmark |
| Max concurrent blocks | 56 (1 per AIV; beyond 56 = time-sliced) | A5 benchmark |

### SIMT vs SIMD compute width

| Mode | Compute width | Issue | Best for |
|------|--------------|-------|----------|
| SIMT | 4 x 128B = 512B | In-order, single-issue | Irregular access, scatter-add, cooperative |
| SIMD | 256B x 2 = 512B | Out-of-order, multi-issue | Contiguous data, vectorized compute |

### Warp-level API (SIMT mode, AscendC::Simt namespace)

- `WarpShflSync`, `WarpShflXorSync` -- warp shuffle
- `WarpReduceAddSync`, `WarpReduceMaxSync`, `WarpReduceMinSync` -- warp reduction
- `WarpBallotSync`, `WarpAllSync`, `WarpAnySync` -- warp vote
- `ThreadBarrier()` -- block-wide sync (__syncthreads equivalent)
- `ThreadFence()` -- memory fence
- `AtomicCas` -- compare-and-swap

## Memory

| Parameter | Value | Source |
|-----------|-------|--------|
| HBM type | BaiLu Memory (Huawei proprietary) | A5_HARDWARE_INFO.md |
| HBM capacity | 128GB (8 x 16GB) | npu-smi |
| HBM bandwidth (peak) | 1.6 TB/s | DavidV100 manual |
| HBM bandwidth (measured) | ~1.1 TB/s | A5 benchmark |
| L2 cache total | 128MB (2 die x 64MB, 16 bank/die x 4MB) | DavidV100 manual |
| L2 cacheline | 512B (4 x 128B sector) | DavidV100 manual |
| L2 read bandwidth | 5.28 TB/s | DavidV100 manual |
| UB (Unified Buffer) / AIV | 256KB | DavidV100 manual |
| SIMT DCache | 32-128KB (carved from UB) | DavidV100 manual |
| SIMT shared memory | min 128KB (carved from UB, 128B aligned) | DavidV100 manual |
| Per-AICore AXI interface | 2 x 128B read + 1 x 128B write | DavidV100 manual |
| `__ldg` cache hint | **NO measurable effect** (verified 2026-03-26) | tests/ldg_test/ |

**Key note**: L2 cache exists (128MB, larger than A100's 40MB) but does NOT accelerate
atomicAdd operations the way GPU L2 does. The L2 is primarily a read cache for DMA traffic.
SIMT atomicAdd goes through HBM, serialized.

## Atomic operations

| Operation | Latency / Notes | Source |
|-----------|----------------|--------|
| **atomicAdd FP32 (HBM)** | **15.9 cycles/op** (high fan-in backward) | msprof verified |
| **atomicAdd FP32 (HBM)** | **3.3 cycles/op** (low fan-in forward) | msprof verified |
| atomicAdd FP16, BF16 | Supported (hardware native) | DavidV100 manual |
| atomicCAS U32/S32/U64 | Supported via Simt::AtomicCas | DavidV100 manual |
| atomicAdd on UB | **Also slow** (not like GPU shared memory atomic) | Expert E7-4 |
| HA Reduce AtomicStore | FP32/FP16/BF16/INT types | DavidV100 manual |

**更正 (Batch 14, 手册确认)**: L2 cache **支持** Reduce Atomic coalescing (HA.FS009.02, HA.FS010.03)。同 cacheline 的 atomicAdd 可在 L2 并行执行。但实测 (msprof) 仍显示 atomicAdd 是瓶颈——可能是高 fan-in 下 cacheline 冲突率太高，L2 coalescing 无法完全消除竞争。sorted-edge (P-P21) 仍是最有效的优化。

## Load/Store

| Access width | Relative speed | Source |
|-------------|---------------|--------|
| 128-bit | **2.1x** faster than 32-bit | tests/load_width_test/ (verified) |
| 64-bit | 1.4x faster than 32-bit | tests/load_width_test/ (verified) |
| 32-bit | baseline | tests/load_width_test/ |

- Stride-1 access: coalesced automatically
- Alignment: 128B for optimal AXI utilization
- vec4 loads (128-bit): enabled when `dim % 4 == 0` (Pattern P-P3)

## SIMT Architecture Details (from DavidV100 手册 分卷2)

| Parameter | Value | Source |
|-----------|-------|--------|
| Warp schedulers | 4 | 手册 Table 25-1 Row 06.08 |
| Instruction issue | In-order, single-issue, 128B/instr | 手册 p4 |
| dcache (from UB) | 32KB~128KB configurable, 128B cacheline | 手册 p4 |
| SIMT usable UB | 256KB - dcache_size | 手册 p4 |
| Shared memory | min 128KB (from UB), 128B aligned | 手册 p405, SQE format |
| Register file | 128KB total, 4B/reg, shared across threads | 手册 p4 |
| LSU | 1 set, 256 miss handler entries | 手册 Table 25-1 |
| Memory path | Thread → dcache → L2 (64MB/die) → HBM | 手册 Fig 25-3 |

**关键: SIMT 线程读 GM 经过 dcache + L2 cache 两级缓存**。dcache 从 UB SRAM 切出，128B cacheline。L2 是 64MB/die 共享 cache (512B tag, 128B sector, 8-way)。同一 expert 行被多个 token 读时，L2 会缓存。

### SIMT/SIMD 混合模式

**硬件支持 SIMT 和 SIMD 在 VF 内/间切换** (手册 p4):
> VEC 可以支持 SIMD/SIMT 编程模型, 可以在 VF 中切换, 也可以在 VF 间切换. 切换间, 数据在 UB 交换.

这意味着可以: SIMD DataCopy (MTE2) 把数据搬到 UB → 切 SIMT 用线程做不规则计算 → 切回 SIMD (MTE3) 写回。兼得 MTE2 块传输带宽 + SIMT 线程灵活性。

**AscendC 已暴露混合 API** (Batch 14-7 确认): 华为内置算子 `diag_part_simt_simd.h` 使用了完整的混合模式。方法: SIMD `TQue::AllocTensor()` 分配 UB → `GetPhyAddr()` 获取 `__ubuf__` 地址 → `Simt::VF_CALL` 传入 UB 地址 → SIMT 线程读 GM 写 UB → 回到 SIMD 用 `EnQue/DeQue/DataCopyPad` 写 GM。

### L2 Cache 控制

DavidV100 支持软件控制 L2 cache 分配 (手册 p32):
- **Alloc hint**: normal / not-alloc / inter-domain-share / exclusive
- **Victim hint**: first_victim / last_victim / persistent (控制驻留优先级)
- **ReadOnce coalescing**: 同 cacheline 的读请求并行执行 (HA.FS009.01)
- **Reduce Atomic coalescing**: 同 cacheline 的 atomicAdd 并行执行 (HA.FS009.02)
- **DavidV100 新增**: RO prefetch, RO multicast, WU MERGE (上一代无)

**AscendC 已暴露 L2 cache hint API**（HKV 专家代码确认）:
```cpp
// 读: L2 不分配 + L1/dcache 缓存（读完清理 L2 tag，防污染）
T val = __ldg<LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(ptr);

// 写: L2 正常写回 + 不走 L1（一次性写）
__stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(ptr, val);
```
来源: `HierarchicalKV-ascend/include/utils.h:139`, `score_functor.h:77`

**OL-18 "\_\_ldg 无效" 更正**: 之前测的是不带模板参数的默认 `__ldg`，默认行为可能是 `L2_CACHE_HINT_NORMAL`（正常缓存），在大范围顺序扫描下与不用 `__ldg` 无差异。带 hint 的版本未测试。

**SG 算子的推荐 hint 策略**:
- expert 行读取: `L2_CACHE_HINT_NORMAL` + `CACHEABLE`（同一 expert 被 ~512 token 重复读，应保留在 L2）
- output 写: `NON_CACHEABLE`（一次性写，不需要缓存）
- index/weight 读: `NOTALLOC_CLEAN`（顺序扫描用完即丢，防污染 L2）

**实验结果 (Batch 14-5)**: SIMT persistent SG forward 测试无正面效果——dim=64 慢 24%，其余无变化。dcache 对 persistent kernel 的顺序 token 遍历已足够有效。L2 hint 在跨 core 共享热点数据场景（如 HKV bucket 查找）才有价值。

## SIMT vs SIMD compatibility

| Feature | SIMT | SIMD |
|---------|------|------|
| Programming model | CUDA-like threads, 4 warp schedulers | TPipe/TQue/DataCopy pipeline |
| Memory path | dcache (UB) → L2 → HBM | MTE2 DMA → UB, VEC compute, MTE3 → HBM |
| Pipeline parallelism | **单管线** (VEC only, in-order) | **4 管线并行** (MTE2+VEC+MTE3+scalar) |
| Scatter/indirect access | Native (threadIdx) | Requires manual loop or SetAtomicAdd |
| **Recommendation (P-P9)** | **仅 scatter-write (atomicAdd)** | **所有其他场景** (含 indirect-read) |
| Hybrid mode | 可在 VF 内切换到 SIMD 使用 MTE2 | 可在 VF 内切换到 SIMT 做不规则计算 |

## 通用算子优化决策指南

基于以上硬件架构，对任意 AscendC 算子适用的决策规则：

### 编程模型选择

| 算子特征 | 选择 | 原因 |
|---------|------|------|
| 纯连续读写（elementwise, 矩阵乘） | SIMD | MTE2+VEC+MTE3 三管线并行 |
| 间接读 + 连续写（gather, embedding lookup） | SIMD | expert/embedding 行是连续内存，DataCopy 块传输高效 |
| 连续读 + 间接写（scatter-add, pooling backward） | SIMT | atomicAdd 需要线程级控制 |
| 间接读 + 间接写（稀疏矩阵运算） | SIMT | 两端都不规则 |
| 上述 + 需要极致性能 | **混合模式** | SIMD 搬数据到 UB，SIMT 做不规则计算 |

### 内存层次优化

| 层次 | 大小 | 延迟 | 优化方向 |
|------|------|------|---------|
| Register file | 128KB/AIV | ~1 cycle | 减少线程数提高每线程寄存器数（排序+累加器） |
| UB (SIMD) / dcache (SIMT) | 256KB/AIV | ~几 cycles | SIMD: TQue 管理；SIMT: dcache 配大获得更多缓存 |
| L2 cache | 64MB/die (128MB total) | ~几十 cycles | 数据复用：同一数据被多个 core 读时 L2 自动缓存 |
| HBM | 128GB, 1.5 TB/s | ~百 cycles | 减少总读量（循环重排、数据复用、预排序） |

### Roofline 快速判断

```
OI = FLOPs / Bytes
Ridge point (fp32) = 28 TFLOPS / 1.5 TB/s ≈ 19 FLOP/byte

OI < 1    → 深度内存受限：优化数据搬运（减少 GM 读、管线重叠）
OI 1~19   → 内存/计算混合：两边都需要优化
OI > 19   → 计算受限：优化 VEC 利用率（向量化、unroll）
```

### Reg-based vs Mem-based SIMD 实现 (待验证)

```
专家建议 (2026-04):
  社区文档的 SIMD 优化基于 mem-based 实现（操作数在 UB 中）。
  A5 芯片作为新一代架构，可在此基础上改为 reg-based 实现以获得进一步性能提升。

当前理解:
  - 社区 SIMD (A2/A3): 全部 mem-based，VEC 指令操作 UB 中的数据
  - A5 SIMD: 同样支持 mem-based，但可能额外支持 reg-based 模式
  - Reg-based 优势: 寄存器访问延迟 ~1 cycle vs UB ~几 cycles

待验证:
  1. A5 的 SIMD 管线是否支持寄存器直接操作模式？
  2. 如果支持，AscendC API 层面如何启用？
  3. 哪些 VEC 指令支持 reg-based？
  4. 性能提升预期多少？

状态: UNVERIFIED — 需要查阅 ISA 规格或实验验证
```

### 线程数 vs 寄存器数 tradeoff

```
总寄存器 = 128KB = 32768 个 (4B/reg)
2048 线程: 16 reg/thread — 无法放累加器数组
1024 线程: 32 reg/thread — 可放 8 个 float 累加器
512 线程:  64 reg/thread — 可放 16 个 float 累加器 (MAX_ACCUM=16)
256 线程:  128 reg/thread — 充足，但并行度低

原则: scatter-add 用累加器需要多寄存器 → 减少线程到 512
      纯 elementwise 不需要累加器 → 增加线程到 1024-2048 隐藏延迟
```

### 原子操作优化路径

```
if 可预排序:
    sorted-edge + 寄存器累加 (P-P21) → atomicAdd 次数降 100x+
elif 可分区:
    per-core 本地累加 → 最后一次 atomicAdd 汇总
elif fan-in 低:
    直接 atomicAdd (3.3 cycles/op, L2 coalescing 帮助)
elif fan-in 高:
    WarpReduceAddSync → 每 warp 1 次 atomicAdd (减少 32x)
```

## Known bugs (CANN 9.0.T501 / bisheng 15.0.5)

| Bug | Severity | Workaround | Reference |
|-----|----------|------------|-----------|
| **TQue<VECIN,2> data corruption** | Critical | Use PipeBarrier<PIPE_ALL> instead of double-buffer | 99.5% elements corrupted; output/src/sparse_gather/sparse_gather_simd.h |
| **Typed kernel entry `_fp32` crash** | High | Use legacy entry points (extern "C" __global__) | Error code 507035 |
| **bf16 Cast not supported** | Medium | Use MicroAPI register-level Cast (Pattern F-P3) | bisheng 15.0.5 limitation |

## Compilation

```bash
# NPU mode
cmake .. -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9589 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.T501 \
  -DASC_DIR=/usr/local/Ascend/cann-9.0.T501/lib64/cmake
# Defines: __NPU_ARCH__=3510, uses arch35/ directory
```
