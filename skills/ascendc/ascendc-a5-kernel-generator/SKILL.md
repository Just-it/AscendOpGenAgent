---
name: ascendc-a5-kernel-generator
description: >
  Generate AscendC SIMT/SIMD kernel code from source (CUDA, PyTorch spec, or AscendC port)
  using migration strategy and pattern library. Supports iterative compile-fix loop and
  optimization directives. Local-A5 variant.
argument-hint: >
  Required: source (file path — original source for reference), kernel_name,
  migration_strategy (path to strategy MD from analyzer).
  Optional: optimization_directive (from QA), variant_name, max_compile_attempts (default 5).
---

# AscendC Kernel Generator

You are an AscendC kernel developer. Generate correct, pattern-compliant AscendC code
from the source based on the migration strategy from the Analyzer.

## File Access Boundary

You may read ONLY:
- `source`
- `{output_dir}/**`
- `../a5-shared-references/**`
- `../a5-common-scripts/**`
- `utils/build_ascendc.py`

You MUST NOT read:
- `archive_tasks/**`
- `../ascendc-translator/**`
- `../tilelang-designer/**`
- `../performance-analyzer/**`
- `../trace-recorder/**`
- Any non-A5 skill directories

## Inputs

- `source`: source file path (CUDA, PyTorch, or AscendC from other chip)
- `kernel_name`: primary kernel function name
- `migration_strategy`: path to `migration_strategy.md` (from Analyzer)
- `pattern_match`: path to `pattern_match.md` (from Analyzer)
- `optimization_directive`: path to directive MD (from QA, for Stage 2 iterations)
- `int64_audit`: path to `int64_audit.md` (from Analyzer)
- `workspace_dir`: output directory

## Step 1: Load Context

```
ALWAYS load:
  ../a5-shared-references/hardware/target/ascend950pr.md   (NPU constants, known bugs)
  ../a5-shared-references/hardware/INDEX.md                (translation deltas)
  ../a5-shared-references/patterns/PATTERN_INDEX.md        (routing table — scan triggers)

Read from workspace:
  migration_strategy.md     → SIMT/SIMD decision, algorithm class
  pattern_match.md          → MANDATORY and RECOMMENDED patterns
  int64_audit.md            → int64 preservation requirements

PATTERN LOADING (critical):
  1. Read PATTERN_INDEX.md routing table
  2. For each pattern, check if its "Trigger" matches the current kernel:
     - bfloat16_t in code? → load platform_compat.md (P-P27 bf16 scalar conversion)
     - atomicAdd? → load scatter_add.md
     - SIMD DataCopy + VEC compute loop? → load memory_access.md (P-P28 TQue<4> — CRITICAL, 1.6-2.3x vs PipeBarrier)
     - PipeBarrier<PIPE_ALL> in SIMD loop? → **ANTI-PATTERN**, replace with TQue (P-P28)
     - GetValue in loop from GM? → load memory_access.md (P-P29 batch preload cache, backward only)
     - Multi-expert/top_k loop? → load memory_access.md (P-P28 TQue + P-P29 cache)
     - Multi-dim tiling? → load thread_utilization.md + memory_access.md
  3. ALWAYS load: precision.md (mandatory audits) + kernel_launch.md (compliance)
  4. Load matched domain files from ../a5-shared-references/patterns/domains/

If optimization_directive provided:
  Read directive → specific pattern to apply with msprof evidence

Load operational knowledge (process + algorithm_selection categories):
  ../a5-shared-references/OPERATIONAL_KNOWLEDGE.md → filter by loaded_by: [Builder]

AscendC LANGUAGE REFERENCE (ALWAYS load):
  ../a5-shared-references/ASCENDC_LANGUAGE_REFERENCE.md
  → SIMD: TQue/TBuf sync semantics, PipeBarrier hierarchy, accumulator patterns
  → SIMT: thread sync, memory model, atomics, mixed SIMT+SIMD mode
  → Anti-patterns (TBuf+TQue conflict, PIPE_ALL in hot loops, GM atomicAdd)

SIMT vs SIMD DECISION (load when deciding programming model):
  ../a5-shared-references/SIMT_VS_SIMD_DECISION.md
  → Decision tree, 4 verified case studies, precision constraints (OL-30)
  → Key rule: group-local (group<256) → SIMT; uniform vectorizable → SIMD

SIMD DEVELOPMENT REFERENCE (load when implementing SIMD kernels):
  ../a5-shared-references/ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md
  → int32 bitwise ops (Ands, ShiftRight), 950PR API type restrictions
  → A3 vs 950PR platform differences

EXTERNAL KNOWLEDGE (when stuck or unsure about AscendC API):
  - CANN source code: ~/workspace/cann (git fetch first, search for examples)
  - CANN reg_convert.h: type conversion API inventory
  - AscendC docs: use dev-browser plugin on hiascend.com (JS-rendered, see OL-22)

RESOURCE AVAILABILITY CHECK (before starting work):
  If any required resource is unavailable, STOP and ask user:
  1. CANN source (~workspace/cann/): needed for API examples → "CANN source not found. Continue without external examples?"
  2. Test data: needed for precision verification → "Test data missing. Skip verification?"
  3. NPU device: check npu-smi before benchmark → "NPU busy/alarm. Wait, try another NPU, or skip benchmark?"
  Never silently skip a verification step — always inform user what was skipped and why.
```

## Step 2: Generate AscendC Code

### 2.1 File Structure

Generate following the existing project convention:
```
gpu_{op}_forward.h           - Forward kernel VF functions
gpu_{op}_backward.h          - Backward kernel VF functions
{op}_kernels.cpp             - ALL dispatchers (extern "C" __global__ __aicore__)
{op}_launch_config.h         - Launch config helpers (nblk computation)
```

Additional files as needed:
```
fast_atomic_add.h            - fp16/bf16 atomicAdd wrapper (if scatter-add)
radix_sort_kernel.h          - NPU counting sort (if sorted variant needed)
```

### 2.2 Code Convention (AscendC SIMT)

#### File Header Template (MANDATORY for all .h files)

```cpp
#ifndef ASCENDC_{OP}_{DIRECTION}_H_
#define ASCENDC_{OP}_{DIRECTION}_H_

#include <kernel_operator.h>    // MANDATORY — primary AscendC header
#include <cstdint>
#if defined(ASCENDC_CPU_DEBUG)
#include "simt_compat.h"        // CPU mode only — conflicts with NPU intrinsics
#endif

using namespace AscendC;        // ONLY AscendC, NEVER AscendC::Simt (OL-14)

namespace ascendc_ops {         // ALL code inside this namespace

// bf16 bit-manipulation helpers (P-P27) — include in EVERY kernel header
#ifndef SIMT_BF16_CAST_DEFINED
#define SIMT_BF16_CAST_DEFINED
__aicore__ inline float simt_to_float(bfloat16_t val) {
    uint16_t bits;
    __builtin_memcpy(&bits, &val, sizeof(uint16_t));
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float result;
    __builtin_memcpy(&result, &f32_bits, sizeof(float));
    return result;
}
__aicore__ inline bfloat16_t simt_from_float(float val) {
    uint32_t f32_bits;
    __builtin_memcpy(&f32_bits, &val, sizeof(uint32_t));
    uint16_t bf16_bits = static_cast<uint16_t>(f32_bits >> 16);
    bfloat16_t result;
    __builtin_memcpy(&result, &bf16_bits, sizeof(uint16_t));
    return result;
}
// Generic template helpers for all types
template <typename T>
__aicore__ inline float simt_to_float_generic(T val) {
    return static_cast<float>(val);  // works for float, half
}
template <>
__aicore__ inline float simt_to_float_generic<bfloat16_t>(bfloat16_t val) {
    return simt_to_float(val);
}
template <typename T>
__aicore__ inline T simt_from_float_generic(float val) {
    return static_cast<T>(val);  // works for float, half
}
template <>
__aicore__ inline bfloat16_t simt_from_float_generic<float>(float val) {
    return simt_from_float(val);  // actually returns bfloat16_t — specialization for bf16
}
#endif

// ... kernel functions ...

} // namespace ascendc_ops

#endif // ASCENDC_{OP}_{DIRECTION}_H_
```

#### VF Function Naming Convention

Follow this naming pattern for kernel VF functions:
```
gpu_{op}_{direction}_kernel_vf<T, BRE, TI>     — baseline (template BRE/TI)
gpu_{op}_{direction}_sorted_kernel_vf<T, BRE, TI>  — sorted variant (template BRE/TI)
gpu_{op}_{direction}_sorted_rt_vf<T>            — sorted with runtime BRE/TI args
```

Template BRE/TI enables `#pragma unroll` optimization. Runtime `_rt_vf` variant is needed
for P-P20 BRE=emb_dim (when BRE varies per operator invocation). Generate BOTH:
- Template variant: for known-at-compile-time BRE values (e.g., BRE=32, BRE=512)
- Runtime `_rt_vf` variant: for BRE=emb_dim dispatch (P-P20)

#### VF Function Template

```cpp
template <typename DATA_TYPE, int BRE = BLOCK_READ_EMB, int TI = TILE_INDICES>
__simt_vf__ __aicore__
LAUNCH_BOUND({OP}_THREAD_NUM) inline void gpu_{op}_{dir}_kernel_vf(
    GM_ADDR param1_gm, GM_ADDR param2_gm, ...,
    int param_int, int64_t param_large,
    uint32_t block_index, uint32_t total_block_num) {

  __gm__ const DATA_TYPE* __restrict__ in = reinterpret_cast<__gm__ const DATA_TYPE*>(param1_gm);
  __gm__ DATA_TYPE* __restrict__ out = reinterpret_cast<__gm__ DATA_TYPE*>(param2_gm);

  // Block/thread decomposition
  if (block_index >= static_cast<uint32_t>(work_items)) return;

  // Compute in float for all types (half works with static_cast, bf16 uses helpers)
  for (int tid = threadIdx.x; tid < work_dim; tid += blockDim.x) {
    float val = simt_to_float_generic<DATA_TYPE>(in[offset + tid]);
    // ... compute ...
    out[offset + tid] = simt_from_float_generic<DATA_TYPE>(result);
  }
}
```

#### Dispatcher Template

```cpp
// In {op}_kernels.cpp:
#include <kernel_operator.h>
#if defined(ASCENDC_CPU_DEBUG)
#include "simt_compat.h"
#endif
#include "gpu_{op}_forward.h"
#include "gpu_{op}_backward.h"
using namespace AscendC;
using namespace ascendc_ops;

// One extern "C" per dtype. Use _fp32/_fp16/_bf16 suffix.
extern "C" __global__ __aicore__ void {op}_{dir}_kernel_fp32(
    GM_ADDR p1, GM_ADDR p2, ..., int param) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  uint32_t threads = {OP}_THREAD_NUM;
  Simt::VF_CALL<gpu_{op}_{dir}_kernel_vf<float>>(
      Simt::Dim3{threads},
      p1, p2, ..., param,
      GetBlockIdx(), GetBlockNum());
}
```

#### Passing fp16/bf16 Scalar Arguments (P-P30)

`extern "C"` kernel boundary cannot pass `half`/`bfloat16_t` scalars directly.
Use `uint16_t` bit-pattern:
```cpp
// Dispatcher:
extern "C" __global__ __aicore__ void {op}_init_kernel_fp16(
    GM_ADDR data_gm, uint16_t num_bits, int64_t size) {
  // Reconstruct half from bits
  half num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;
  // ... use num ...
}
```

### 2.3 Apply MANDATORY Patterns

From pattern_match.md, apply every MANDATORY pattern:

- **int64 preservation**: Every CUDA int64_t param stays int64_t. Every multiplication
  involving dim/offset MUST cast at least one operand: `static_cast<int64_t>(a) * b`.
  NEVER use `static_cast<int>(int64_var)`.

- **F-P2 multi-dtype**: Use template<typename T> with float compute, T storage.
  Generate dispatchers for fp32, fp16, bf16.

- **P-P1 numBlocks**: Use MAX_AIV_CORES (56) or work_items, whichever is smaller.
  Never hardcode CUDA's grid size.

- **P-P5 LAUNCH_BOUND**: Set to max threads the kernel uses. Add LAUNCH_CHECK macro.

- **P-P20 BRE=emb_dim** (if flagged): Runtime dispatch with `emb_dim` as block read size,
  NOT fixed CUDA constants like BRE=32.

### 2.4 Generate Variants (if flagged by Analyzer)

If scatter-add detected and RECOMMENDED:
- Generate baseline + sorted variant with register accumulation (P-P21)
- Generate sorted `_rt_vf` variant with runtime BRE/TI for P-P20 BRE=emb_dim
- Generate `generate_assign_edges_sorted_vf` (P-P32: atomicCAS-free dedup via pre-sorted edges)

If persistent kernel flagged:
- Generate baseline + persistent variant with token loop (P-P22):
  `for (uint32_t t = block_index; t < work_items; t += total_block_num)`

#### Multi-Dtype Dispatch Macro (for sorted _rt_vf variants)

When generating 3-dtype dispatchers for runtime-param variants, use a macro to avoid boilerplate:
```cpp
#define DEFINE_{OP}_SORTED_RT(suffix, dtype)                                    \
extern "C" __global__ __aicore__ void {op}_{dir}_sorted_rt_kernel##suffix(      \
    GM_ADDR p1, GM_ADDR p2, ...,                                                \
    int BRE, int TI, int block_read_indices, int iter_indices_block,             \
    int iter_indices_thread, int iter_emb) {                                     \
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                               \
  uint32_t threads = static_cast<uint32_t>(block_read_indices) * BRE;           \
  Simt::VF_CALL<gpu_{op}_{dir}_sorted_rt_vf<dtype>>(                           \
      Simt::Dim3{threads}, p1, p2, ...,                                         \
      BRE, TI, block_read_indices, iter_indices_block,                           \
      iter_indices_thread, iter_emb, GetBlockIdx(), GetBlockNum());              \
}

DEFINE_{OP}_SORTED_RT(, float)
DEFINE_{OP}_SORTED_RT(_fp16, half)
DEFINE_{OP}_SORTED_RT(_bf16, bfloat16_t)
```

#### Compile-Time Specialization Variants

For known BRE/TI combinations, generate explicit template instantiations:
```cpp
// Standard: BRE=32, TI=16 (CUDA defaults)
extern "C" ... {op}_{dir}_kernel_fp32(...) {
  VF_CALL<gpu_{op}_{dir}_kernel_vf<float, 32, 16>>(...);
}
// Large dim: BRE=512, TI=1 (when emb_dim > 256)
extern "C" ... {op}_{dir}_large_dim_kernel_fp32(...) {
  VF_CALL<gpu_{op}_{dir}_kernel_vf<float, 512, 1>>(...);
}
// Large edge: BRE=32, TI=1024 (when edge_count >> emb_dim)
extern "C" ... {op}_{dir}_large_edge_kernel_fp32(...) {
  VF_CALL<gpu_{op}_{dir}_kernel_vf<float, 32, 1024>>(...);
}
```

### 2.5 Apply Optimization Directive (Stage 2)

If `optimization_directive` is provided, it contains:
- Specific pattern ID to apply (e.g., "P-P21")
- msprof evidence (e.g., "vec_ratio=1.0, atomicAdd 15.9 cycles")
- Concrete implementation guidance

Read the directive and modify the kernel accordingly. Do NOT change unrelated code.

## Step 3: Compile

Build locally:
```bash
python3 utils/build_ascendc.py {workspace_dir} -v Ascend950PR_9589 --build-type Release
```

If compile error: parse error, apply fix, retry (max 5 attempts).

### 3.1 Pre-Compile Static Check (MANDATORY before build)

Run the static checker BEFORE building:
```bash
python3 ../a5-common-scripts/ascendc_static_check.py {workspace_dir}/
```
If any check fails → fix locally before wasting compile time.

### 3.2 Conductor Pattern (Error Classification + Repair)

After each compile attempt, classify the error and decide next action:

**Error Classification:**
- **Type A** (code error): Syntax, missing include, wrong API, namespace → auto-fix
- **Type B** (environment error): CANN missing, NPU unavailable, timeout, disk full → ABORT
- **Type C** (repeated failure): Same error type 3+ times → ESCALATE to user

**Decision Tree:**
```
1. Is error Type B (environment)? → ABORT. "Non-code error, cannot fix by regeneration."
2. Is error Type C (repeated ≥3 times)? → ESCALATE. "Repeated failure, manual intervention needed."
3. iteration >= max_attempts? → ABORT. "Max compile attempts reached."
4. Type A + iteration < max? → FIX using error corrections below, retry.
```

**History Tracking** (write after each attempt):
```json
// {workspace_dir}/compile_history.json
[{
  "attempt": 1,
  "error_type": "A",
  "error_message": "error: calling __host__ function from __aicore__",
  "matched_correction": "EC-1",
  "fix_applied": "Added __aicore__ to helper function",
  "decision": "retry"
}]
```

### 3.3 Error Corrections Reference

Load `../a5-shared-references/ERROR_CORRECTIONS.md` for structured error→repair mappings (EC-1 through EC-9). Match compile error output against the error patterns in that file.

Common fixes (quick reference):
- EC-1: Missing `__aicore__` on helper → add decorator
- EC-2: `GM_ADDR` needs `reinterpret_cast<__gm__ T*>`
- EC-3: `LAUNCH_BOUND > 512` → reduce to 512
- EC-4: `simt_compat.h` conflicts → guard with `#if ASCENDC_CPU_DEBUG`
- EC-5: `static_cast<float>(bf16)` → use `simt_to_float()` (P-P27)
- EC-6: `using namespace AscendC::Simt;` → use only `AscendC;` (OL-14)
- EC-7: `Simt::atomicAdd` → bare `atomicAdd` (global built-in)
- EC-8: Missing `#include <kernel_operator.h>` → add as first include
- EC-9: Missing `namespace ascendc_ops {}` → wrap all code

## Step 4: Output

Write to `{workspace_dir}/`:
```
{kernel_name}_kernel.h      - Kernel implementations
{kernel_name}_kernels.cpp   - Dispatchers
compile_report.md           - Success/failure, attempts, final status
```

`compile_report.md` first line: `COMPILE_SUCCESS` or `COMPILE_FAIL`
(machine-readable for orchestrator)

## Anti-Hack Rules

- NEVER use `static_cast<int>(int64_var)` unless int64_audit.md explicitly proves safety
- NEVER copy CUDA fixed constants (BRE=32, BLOCK_SIZE=256) without checking NPU docs
- NEVER skip multi-dtype generation (fp32 + fp16 + bf16 dispatchers mandatory)
- NEVER claim "compile success" without actually running cmake + make
- NEVER look at existing AscendC code in the repo for "inspiration" during generation
  (this is a fresh translation, not a copy job)
- **NEVER read or copy from ~/workspace/cann/ (CANN source code)** — this is prohibited.
  Patterns in the knowledge base (a5-shared-references/) are learned and allowed.
  Directly reading CANN source (ops-transformer, ops-nn, opbase, catlass) to copy
  implementation is reward hacking. If you need a reduction pattern, use what's in
  the pattern library. If nothing fits, implement from first principles.
- **NEVER use PipeBarrier<PIPE_ALL> in SIMD hot loops** — use TQue<VECIN,4> instead (P-P28, 1.6-2.3x faster)
- **NEVER call CANN built-in operator APIs** (aclnn*, aclop*, acl_op_*, aclrtLaunchKernel)
  — kernel must implement actual computation logic using AscendC primitives (DataCopy, VEC ops, TQue/TBuf).
  Wrapping CANN ops is reward hacking: it passes QA but defeats the purpose of kernel generation.
- **NEVER import or reference torch_npu / npu_bridge** in kernel .h/.cpp files
  — the kernel layer operates on raw GM_ADDR pointers, not PyTorch tensors
- Generated kernel .h MUST contain at minimum: TQue/TBuf declarations, DataCopy calls,
  and VEC/scalar computation. Static checker enforces ≥3 computation markers.
