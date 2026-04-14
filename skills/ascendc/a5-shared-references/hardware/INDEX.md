# Hardware Reference Index

> Lightweight index for the CUDA->AscendC skill system. Always loaded by skills.
> Detailed specs in per-platform files; this file provides quick lookup and translation deltas.

## Target platforms

| Platform | File | Cores | Max blocks | atomicAdd | Scheduling | Status |
|----------|------|-------|-----------|-----------|------------|--------|
| Ascend950PR | [target/ascend950pr.md](target/ascend950pr.md) | 56 AIV | 56 (software time-slicing beyond) | 15.9 cycles (HBM-serialized) | Software time-slicing | Verified |

## Source platforms

| Platform | File | Cores | Max blocks | atomicAdd | Scheduling | Status |
|----------|------|-------|-----------|-----------|------------|--------|
| A100-SXM4-80GB | [source/a100_sxm.md](source/a100_sxm.md) | 6912 CUDA (108 SMs) | 4096+ | ~3 cycles (L2 cache HW) | Hardware block scheduler | Verified |

## Key translation deltas (GPU -> NPU)

These are the critical architectural differences that require pattern-level intervention
when translating CUDA kernels to AscendC. Each delta links to the pattern that addresses it.

| Delta | GPU (A100) | NPU (950PR) | Ratio | Pattern | Impact |
|-------|-----------|-------------|-------|---------|--------|
| **Thread waste** | 6912 cores absorb idle threads | 56 cores cannot | 123x more costly | [P-P20](../patterns/general_patterns.md#p-p20-breemb_dim-动态线程分配) (BRE=emb_dim) | dim=9 wastes 72% on NPU vs negligible on GPU |
| **atomicAdd** | ~3 cycles, L2 cache HW | 15.9 cycles, HBM-serialized | 5x slower | [P-P21](../patterns/general_patterns.md#p-p21-scatter-add-排序寄存器累加) (sorted-edge accumulation) | Bwd -81% after sort+register accum |
| **Block scheduling** | Hardware scheduler, near-zero overhead | Software time-slicing, measurable overhead | N/A | [P-P22](../patterns/general_patterns.md#p-p22-常驻核心分发persistent-kernel) (persistent kernel) | SG fwd 1.35-3.4x speedup |
| **Shared memory** | 48-164KB per SM, low-latency | None in SIMT mode | N/A | [P-P2](../patterns/general_patterns.md#p-p2-warpreduceaddsync--warp-lane-0-原子操作) (WarpReduceAddSync) | Must restructure block-reduce algorithms |
| **Oversubscription** | 4096+ blocks, HW-scheduled | 56 blocks + software time-slicing | Different semantics | [P-P10](../patterns/general_patterns.md#p-p10-block-超订-oversubscription) | Helps unsorted scatter-add; hurts sorted |

## Loading rules (for skills)

```
Planner/Analyzer:  always load INDEX.md (this file)
Generator/Builder: load INDEX.md + target/{platform}.md + source/{platform}.md
QA/Optimizer:      load INDEX.md + target/{platform}.md (for msprof interpretation)
```
