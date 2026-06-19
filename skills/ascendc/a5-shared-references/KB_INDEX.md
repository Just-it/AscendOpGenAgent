# AscendC Knowledge Base — Search Index

> **Workers: read this file FIRST.**
> Search the "Keywords/Aliases" column for your problem, then load the matched file.
> Same concept may have different names — aliases ensure you find it.

## How to Search

1. Identify your task: what op type? what problem? (e.g., "reduction", "alignment", "bf16")
2. Grep this file for keywords: `grep -i "reduction\|reduce" KB_INDEX.md`
3. Load the matched file(s) and read the specific entries

---

## Decision Frameworks

| File | What | Keywords/Aliases | When |
|------|------|-----------------|------|
| [SIMT_VS_SIMD_DECISION.md](SIMT_VS_SIMD_DECISION.md) | Architecture decision tree | SIMT, SIMD, 间接寻址, indirect indexing, atomicAdd, scatter, group-local, elementwise | **ALWAYS** |
| [ASCENDC_LANGUAGE_REFERENCE.md](ASCENDC_LANGUAGE_REFERENCE.md) | Core API: TQue, TBuf, DataCopy, sync | TQue, TBuf, TPosition, VECIN, VECOUT, EnQue, DeQue, DoubleBuffer, 双缓冲, pipeline overlap, 流水重叠, SetFlag, WaitFlag | **ALWAYS** |
| [ROOFLINE_MODEL.md](ROOFLINE_MODEL.md) | Performance upper bound | roofline, bandwidth, compute bound, 理论上界, HBM, throughput | Performance analysis |

## Error Corrections (EC-1..EC-22)

| File | Keywords/Aliases | When |
|------|-----------------|------|
| [ERROR_CORRECTIONS.md](ERROR_CORRECTIONS.md) | compile error, build fail, SyncFunc, TQue depth, PipeBarrier PIPE_S, DataCopy alignment, PadTiling name conflict, VECIN-only pipeline, multi-block race, EC-22 overlap-tail, 对齐溢出 | Build fails — match error to EC entry |

## Operational Knowledge (OL-1..OL-52)

| File | Keywords/Aliases | When |
|------|-----------------|------|
| [OPERATIONAL_KNOWLEDGE.md](OPERATIONAL_KNOWLEDGE.md) | process rules, platform bugs, measurement, algorithm selection, bf16 Cast, int64, nblk=1 diagnostic, CANN知识差距, TQueBind, DataCopyParams, Counter模式, bank冲突, 512B对齐, 16KB搬运, ReduceMax, 归约, DoubleBuffer使能, SIMT fallback EC-22 | Check for lessons learned on your op type |

## Platform Bugs (PB-1..PB-10)

| File | Keywords/Aliases | When |
|------|-----------------|------|
| [PLATFORM_BUGS.md](PLATFORM_BUGS.md) | 507035 typed entry, UB DataCopy corruption PB-9, TBuf VECCALC DataCopy loop corruption PB-11, bf16 scalar cast PB-4, NPU reboot, -O2 required, TQue depth bug | **ALWAYS** — avoid known pitfalls |

## Patterns (P-P1..P-P38)

| File | Keywords/Aliases | When |
|------|-----------------|------|
| [patterns/PATTERN_INDEX.md](patterns/PATTERN_INDEX.md) | pattern index, trigger conditions, optimization technique | Read index → load matching domain |
| [patterns/domains/memory_access.md](patterns/domains/memory_access.md) | TQue pipeline, DataCopy alignment, cache, DoubleBuffer, 双缓冲, pipeline overlap, MTE2/VEC重叠, SetFlag/WaitFlag, batch preload, DataCopyParams stride, TQueBind | Data-movement, bandwidth |
| [patterns/domains/thread_utilization.md](patterns/domains/thread_utilization.md) | block partition, SIMT thread config, BRE=emb_dim, runtime dispatch, 线程分配, persistent kernel, nblk超订 | SIMT kernels, multi-core |
| [patterns/domains/platform_compat.md](patterns/domains/platform_compat.md) | bf16 Cast, SIMT/SIMD compat, bfloat16, simt_to_float, uint16_t scalar param | bf16 ops, cross-platform |
| [patterns/domains/precision.md](patterns/domains/precision.md) | fp16/bf16 precision, accumulation, 寄存器累加, register accumulator, float accumulation, rounding | Reduction, normalization |
| [patterns/domains/kernel_launch.md](patterns/domains/kernel_launch.md) | entry point, LAUNCH_BOUND, extern C, untyped entry, 507035 | All kernels |
| [patterns/domains/scatter_add.md](patterns/domains/scatter_add.md) | atomicAdd, scatter, EmbeddingBackward, histogram, sorted-edge, 排序, dedup | Scatter ops |
| [patterns/unverified/candidates.md](patterns/unverified/candidates.md) | candidate, unverified, pending validation, TQueBind, Counter模式, **P-REG-1 reg-based fusion**, RegTensor, __simd_vf__ | Optimization ideas |

## SIMT Reference

| File | Keywords/Aliases | When |
|------|-----------------|------|
| [ASCENDC_SIMT_PATTERNS.md](ASCENDC_SIMT_PATTERNS.md) | __simt_vf__, VF_CALL, Simt::GetThreadIdx, bf16 bit-manipulation, grid-stride loop, LAUNCH_BOUND, threadIdx | SIMT kernel implementation |
| [ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md](ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md) | int32 bitwise, 950PR API restrictions, SIMD development | SIMD kernel implementation |

## Hardware Reference

| File | Keywords/Aliases | When |
|------|-----------------|------|
| [hardware/target/ascend950pr.md](hardware/target/ascend950pr.md) | A5 specs, UB 256KB, 56 AIV cores, HBM bandwidth, register file 128KB, dcache, L2 64MB, **reg-based SIMD confirmed**, RegTensor, __simd_vf__, asc_vf_call, LoadAlign/StoreAlign, mem-based, warp scheduler | Hardware constraints |
| [hardware/INDEX.md](hardware/INDEX.md) | GPU→NPU translation, thread waste, atomicAdd latency, shared memory | Porting from GPU |

## Benchmark & Profiling

| File | Keywords/Aliases | When |
|------|-----------------|------|
| [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) | performance measurement, A/B test, aclrtEvent, median, warmup | Performance testing |
| [MSPROF_AGENT_GUIDE.md](MSPROF_AGENT_GUIDE.md) | msprof, profiling, aiv_vec_ratio, MTE2 ratio, bottleneck diagnosis, cycle count | Performance optimization |
