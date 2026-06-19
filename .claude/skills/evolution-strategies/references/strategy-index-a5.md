# A5 (351x/Regbase) Kernel Optimization Strategy Index

Source: A5 AscendC-A5-guide 知识库。本索引用于 A5 Regbase 架构的性能优化。

**注意**: 本索引与 A3 的 `strategy-index.md` 互补。A3 的 D/P/A/X 系列策略在 A5 上部分适用（见"A3 策略兼容性"章节），R 系列为 A5 Regbase 专用。

## Regbase 优化策略 (R系列)

| ID | Strategy | When to Apply | Optimization Type | Detail File |
|----|----------|---------------|-------------------|-------------|
| R1 | VF 函数融合 | 多个独立 VF 函数可合并 | vf_fusion | strategies/a5_R1_vf_fusion.md |
| R2 | 寄存器复用 | RegTensor 数量接近/超过 32 | register_opt | strategies/a5_R2_register_reuse.md |
| R3 | 指令双发射 | 指令依赖链长、双发射利用率低 | instruction_sched | strategies/a5_R3_dual_issue.md |
| R4 | Hardware Loop 规范化 | 循环未被识别为 Hardware Loop | instruction_sched | strategies/a5_R4_hw_loop.md |
| R5 | 非对齐访问优化 | 数据地址非 32B 对齐 | bandwidth | strategies/a5_R5_unalign_opt.md |
| R6 | 低延迟归约 | Reduce 类操作 | algorithm | strategies/a5_R6_low_latency_reduce.md |
| R7 | SIMD/SIMT 混合编程 | 同时存在规则和不规则访问 | algorithm | strategies/a5_R7_simd_simt_hybrid.md |
| R8 | Mutex 细粒度同步 | 粗粒度 SetFlag/WaitFlag 导致流水停顿 | instruction_sched | strategies/a5_R8_mutex_sync.md |

## A3 策略兼容性

### 直接复用（参数需适配 A5 UB 结构）
| A3 ID | 策略 | A5 注意事项 |
|-------|------|-----------|
| P1 | 双缓冲 | UB 248KB, bank 结构 8×2×16KB |
| P2 | 自适应分块 | tile 大小建议对齐到 VL/sizeof(T) |
| P4 | 负载均衡 | 核数变化: vector_core_cnt = 2 × cube_core_cnt |
| D1-D5 | 多数据类型 | 新增 fp8/hifloat8 类型，int4b_t 需先 Cast |
| A1-A3 | 精度优化 | 基本不变 |
| A5-A8 | 精度优化 | 基本不变 |

### 需改写
| A3 ID | 策略 | A5 替代方案 |
|-------|------|-----------|
| P5 | 流水线同步 | 用 LocalMemBar 替代 pipe_barrier（VF 内部） |
| P8 | UB 内存分区 | bank 结构变了 (8×2×16KB)，需重新规划 |
| A4 | SetFlag/WaitFlag | VF 内部用 LocalMemBar，VF 外部保持不变 |

### 不适用
| A3 ID | 策略 | 原因 |
|-------|------|------|
| P11 | 尾块处理 (scalar 循环) | A5 用 UpdateMask 自动处理尾部 |

## How to Use

1. 识别算子瓶颈类型 (bandwidth / tiling / algorithm / register_pressure / vf_fusion / instruction_sched)
2. 从 R 系列选择对应策略
3. 检查 A3 策略兼容性，适用则同时应用
4. **Read each referenced detail file** before modifying code

### Quick Reference by Bottleneck

| 瓶颈类型 | 推荐策略 |
|---------|---------|
| bandwidth (Load/Store 占比高) | R1, R5, P1, P10 |
| tiling (UB 利用率低) | P2, P3, P4 (适配 A5 参数) |
| algorithm (算法效率低) | R6, R7, P6, P13 |
| register_pressure (spill/fill 指令) | R2, R1 |
| vf_fusion (多 VF 未融合) | R1, R4 |
| instruction_sched (双发射利用率低) | R3, R4 |

## 探索发现策略

| ID | Strategy | When to Apply | Detail File |
|----|----------|---------------|-------------|
| (待发现) | A5 evo 过程中发现的新策略将记录在此 | - | - |
