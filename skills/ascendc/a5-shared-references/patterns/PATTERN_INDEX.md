# Pattern Library — Routing Index

> Always loaded by all skills. Use to identify which domain files to load.
> Domain files contain full pattern details; this index is for routing only.

## Domain Routing Table

| ID | Name | Domain | Trigger | Severity |
|----|------|--------|---------|----------|
| F-P1 | bf16 precision handling | precision | bf16 dtype in test | MEDIUM |
| F-P2 | Multi-dtype template | precision | multi-dtype kernel | LOW |
| F-P3 | SIMD bf16 MicroAPI Cast | precision | SIMD + bf16 | MEDIUM |
| F-P4 | SIMD PipeBarrier alignment | platform_compat | SIMD DataCopy | HIGH |
| F-P5 | Warp-aligned loop boundary | precision | cooperative group ops | HIGH |
| F-AP1 | dtype string match trap | precision | compare_data with dtype | CRITICAL |
| F-AP2 | __threadfence misuse | precision | __threadfence in code | MEDIUM |
| P-P1 | numBlocks dynamic | thread_utilization | any kernel launch | HIGH |
| P-P2 | WarpReduceAddSync | scatter_add | atomicAdd in reduction | HIGH |
| P-P3 | vec4 vectorization | memory_access | dim % 4 == 0 | MEDIUM |
| P-P4 | Dynamic block size | thread_utilization | variable work per block | MEDIUM |
| P-P5 | LAUNCH_BOUND + CHECK | kernel_launch | every kernel | MEDIUM |
| P-P6 | grid_y consistency | kernel_launch | multi-grid kernel | MEDIUM |
| P-P7 | pragma unroll scope | kernel_launch | inner loops | LOW |
| P-P8 | Host benchmark practice | kernel_launch | benchmark code | LOW |
| P-P9 | **SIMT vs SIMD 决策框架** | platform_compat | algorithm classification | **HIGH** |
| P-P10 | Block oversubscription | scatter_add | atomicAdd contention | HIGH |
| P-P11 | Adaptive tile size | memory_access | multi-dim tiling | HIGH |
| P-P12 | int32 inner loop | memory_access | int64 in hot loop | MEDIUM |
| P-P13 | Cooperative traversal | cooperative | GROUP_SIZE parallel | HIGH |
| P-P16 | Cooperative value copy | cooperative | large vector copy | MEDIUM |
| P-P17 | Prefix-sum + block atomic | scatter_add | scatter-add aggregation | HIGH |
| P-P18 | __ldg/__stg L2 cache hint | platform_compat | SIMT GM read/write with cache control | HIGH |
| P-P19 | Kernel UT requirement | kernel_launch | every kernel | HIGH |
| P-P20 | Thread Utilization (BRE=dim) | thread_utilization | multi-dim decomposition | HIGH |
| P-P21 | Sorted-edge accumulation | scatter_add | atomicAdd in scatter loop | HIGH |
| P-P22 | Persistent kernel | thread_utilization | work_items >> 56 | MEDIUM |
| P-P23 | Contiguous-chunk vs grid-stride | memory_access | arr[i-1] neighbor access in loop | HIGH |
| P-P24 | Sort-to-reuse (GM read amplification) | memory_access | indirect GM read shared by N items | **CRITICAL** |
| P-P25 | SetAtomicAdd + DataCopyPad | memory_access | SIMD scatter-add to GM | **CRITICAL** |
| P-P26 | SetFlag/WaitFlag fine-grained sync | memory_access | SIMD pipeline overlap | HIGH |
| P-P27 | bf16 scalar via Cast(bf16→float) | platform_compat | bfloat16_t GetValue or static_cast | **CRITICAL** |
| P-P28 | **TQue<4> 管线自动重叠** (替代 PipeBarrier Ping-Pong) | memory_access | SIMD kernel with DataCopy+VEC loop | **CRITICAL** |
| P-P29 | Batch preload cache (index/weight) | memory_access | GetValue in loop from GM (scalar bottleneck) | **CRITICAL** |
| P-P30 | fp16/bf16 scalar kernel arg (uint16_t bits) | platform_compat | half/bf16 scalar in extern "C" kernel params | **CRITICAL** |
| P-P31 | NPU native atomicAdd (no fastAtomicAdd) | platform_compat | CUDA fastAtomicAdd / packed half2 atomics | MEDIUM |
| P-P32 | Sorted-edge dedup (atomicCAS-free) | scatter_add | atomicCAS first-occurrence on sortable data | HIGH |
| P-P33 | **SIMT→SIMD for memory-bound elementwise** | memory_access | msprof MTE2=0% AND throughput < 50% theoretical BW | **HIGH** |
| P-P34 | **SIMT per-element indirect gather** | memory_access | torch.gather, index_select, per-element indirect addressing | **HIGH** |
| P-P35 | bf16 direct assign in SIMT (no Cast needed) | platform_compat | SIMT kernel with bf16 copy (no arithmetic) | MEDIUM |
| P-P36 | TQueBind for pure data-movement ops | memory_access | 纯搬运算子（无 VEC 计算）— 替代 Adds bridge | **HIGH** (unverified on A5) |
| P-P37 | DataCopyParams for strided/non-contiguous copy | memory_access | 非连续内存搬运（列、间隔数据）— 替代 for 循环 | **HIGH** |
| P-P38 | Vector Counter mode for auto tail handling | memory_access | VEC 操作的非对齐尾块处理 — 替代手动 mask 管理 | **HIGH** (unverified) |

## Domain Files

| Domain | File | When to Load | Pattern Count |
|--------|------|-------------|---------------|
| precision | `domains/precision.md` | Always (mandatory audits) | 7 |
| scatter_add | `domains/scatter_add.md` | atomicAdd detected in scatter pattern | 5 |
| thread_utilization | `domains/thread_utilization.md` | Multi-dim decomposition or launch config | 4 |
| memory_access | `domains/memory_access.md` | Memory optimization opportunity | 9 |
| kernel_launch | `domains/kernel_launch.md` | Every kernel (basic compliance) | 5 |
| cooperative | `domains/cooperative.md` | Cooperative group / shuffle ops | 2 |
| platform_compat | `domains/platform_compat.md` | SIMD or platform-specific features, bf16 | 5 |

## Operator-Specific Files

| File | When to Load |
|------|-------------|
| `ops_specific/hkv_patterns.md` | Hash-table (HKV) operations |
| `ops_specific/pooling_sg_patterns.md` | Pooling / Sparse-Gather (future) |

## Unverified

| File | Description |
|------|------------|
| `unverified/candidates.md` | Candidates awaiting validation on 2+ operators |

## Always-Load References

| File | Description |
|------|------------|
| `../ASCENDC_LANGUAGE_REFERENCE.md` | SIMD sync (TQue/TBuf/PipeBarrier), SIMT sync (ThreadBarrier/CrossCore/atomics), mixed mode, HardEvent table |
| `../SIMT_VS_SIMD_DECISION.md` | **P-P9 完整决策框架**: decision tree, 4 case studies, 精度约束 (OL-30) |
| `../ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md` | SIMD 整数/位操作 API, 950PR vs A3 差异, int32 类型限制 |

## Loading Protocol

```
1. Analyzer loads this INDEX (always, ~60 lines)
2. ALWAYS load ASCENDC_LANGUAGE_REFERENCE.md (covers both SIMD and SIMT)
3. Classifies CUDA kernel → identifies relevant domains
4. Loads ONLY matching domain files (typically 2-3 files, ~200 lines total)
5. Loads ops_specific if operator type matches
6. Generator/Optimizer receive exact pattern IDs from Analyzer output
```
