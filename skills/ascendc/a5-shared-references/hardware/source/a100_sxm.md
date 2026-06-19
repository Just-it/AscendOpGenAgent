---
platform: A100-SXM4-80GB
type: source
verified: partial
architecture: Ampere (GA100)
date_verified: 2026-03-25
---

# NVIDIA A100 SXM4 80GB -- Source Platform

> Reference GPU specs for CUDA->AscendC translation cost analysis.
> Data from NVIDIA A100 whitepaper + datasheet; some values from A100 server measurement.

## Compute

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | Ampere (GA100) | NVIDIA whitepaper |
| SMs (Streaming Multiprocessors) | **108** (128 on die, 108 enabled) | NVIDIA whitepaper |
| CUDA Cores / SM | 64 (FP32) | NVIDIA whitepaper |
| **CUDA Cores total** | **6912** (108 x 64) | NVIDIA whitepaper |
| Tensor Cores / SM | 4 (3rd gen) | NVIDIA whitepaper |
| Max threads / SM | 2048 | NVIDIA whitepaper |
| Max threads / block | **1024** | CUDA spec |
| Warp size | 32 | CUDA spec |
| Max warps / SM | 64 | 2048 / 32 |
| Boost clock | 1410 MHz | SXM4 variant datasheet |
| Block scheduling | **Hardware**, near-zero overhead | CUDA architecture |
| Max concurrent blocks | **4096+** (hardware scheduler distributes) | Pooling benchmark |
| Total thread capacity | 221,184 (108 x 2048) | Computed |

## Memory

| Parameter | Value | Source |
|-----------|-------|--------|
| HBM type | HBM2e | NVIDIA datasheet |
| HBM capacity | **80GB** (5 stacks x 16GB) | NVIDIA datasheet |
| HBM bandwidth (peak) | **2039 GB/s (~2 TB/s)** | NVIDIA datasheet |
| HBM bandwidth (measured) | ~1555 GB/s | STREAM benchmark typical |
| **L2 cache** | **40MB** (shared across all SMs) | NVIDIA whitepaper |
| L2 bandwidth | ~5 TB/s | Estimated |
| Shared memory / SM | 48-164KB (configurable with L1) | NVIDIA whitepaper |
| L1 cache + shared memory | 192KB total per SM | NVIDIA whitepaper |
| Register file / SM | 256KB | NVIDIA whitepaper |
| `__ldg` cache hint | Effective (routes through L1 read-only cache) | CUDA docs |

## Atomic operations

| Operation | Latency / Notes | Source |
|-----------|----------------|--------|
| **atomicAdd FP32 (global)** | **~3 cycles** (L2 cache hardware acceleration) | Measurement + whitepaper |
| atomicAdd FP16 (global) | Hardware native (SM >= 70), half2 packed | CUDA spec |
| atomicAdd FP32 (shared) | Hardware native, very low latency | CUDA spec |
| atomicCAS | Hardware native | CUDA spec |
| L2 atomic throughput | Very high for scattered writes | NVIDIA whitepaper |

**Key**: The A100 L2 cache (40MB) provides hardware-accelerated atomic operations.
Scattered atomicAdd that hits L2 runs at ~3 cycles/op. This is the fundamental reason
GPU scatter-add is fast despite many atomic conflicts.

## Translation impact notes

These notes describe what changes when porting CUDA kernels from A100 to Ascend950PR.
Each maps to a pattern that addresses the gap.

| GPU advantage | Why it matters | NPU consequence | Pattern |
|--------------|---------------|-----------------|---------|
| **6912 cores** absorb thread waste | BRE=32 for dim=9 wastes 72% threads -- negligible on 6912 cores | 56 cores amplify waste 123x | P-P20 (BRE=emb_dim) |
| **L2 cache atomic** (40MB) | Scatter-add atomicAdd cached at L2, ~3 cycles | NPU atomicAdd HBM-serialized, 15.9 cycles high-contention | P-P21 (sorted-edge) |
| **Hardware block scheduler** | Zero-overhead dispatch of 4096+ blocks | NPU software time-slicing has measurable overhead | P-P22 (persistent kernel) |
| **Shared memory** (48-164KB/SM) | blockReduceSum via shared memory (1 final store, no atomicAdd) | SIMT has no shared memory equivalent | P-P2 (WarpReduceAddSync + warp-lane-0 atomic) |
| **4096+ concurrent blocks** | Oversubscription "free" with hardware scheduler | 56 blocks + time-slicing: oversubscription helps unsorted only | P-P10 (context-dependent) |
| **L1/L2 read caching** | `__ldg` useful for read-only random access | `__ldg` has no measurable effect on NPU (verified) | P-P18 (documented as no-op on A5) |
