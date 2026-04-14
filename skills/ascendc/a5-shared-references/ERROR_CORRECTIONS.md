# AscendC Error Corrections Reference

> Structured error→repair mappings for common AscendC SIMT compile errors.
> Load when: Generator encounters compile errors in Stage 1 compile-fix loop.
> Format: Error pattern → Root cause → Fix → Related pattern ID

---

## Compile Errors

### EC-1: Missing `__aicore__` on helper function

- **Error pattern**:
  ```
  error: calling a __host__ function("helper_func") from a __aicore__ function("kernel_vf") is not allowed
  ```
- **Root cause**: All functions called inside `__simt_vf__ __aicore__` kernel VF functions must themselves be decorated with `__aicore__`. Bisheng treats undecorated functions as `__host__`-only, and cross-domain calls are forbidden.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  inline float compute_weight(float x) { return x * 0.5f; }

  // AFTER (compiles):
  __aicore__ inline float compute_weight(float x) { return x * 0.5f; }
  ```
- **Note**: Template helper functions also need `__aicore__`:
  ```cpp
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }
  ```
- **Related**: None (basic AscendC requirement)

---

### EC-2: `GM_ADDR` needs typed pointer cast

- **Error pattern**:
  ```
  error: cannot initialize a variable of type '__gm__ float *' with an lvalue of type 'GM_ADDR' (aka 'uint8_t * __attribute__((address_space(1)))')
  ```
  or:
  ```
  error: subscript of pointer to type '__gm__ uint8_t' ... is not allowed
  ```
- **Root cause**: `GM_ADDR` is `__gm__ uint8_t*`. Kernel VF functions receive all GM pointers as untyped `GM_ADDR`. To access data as a specific type, you must cast with `reinterpret_cast<__gm__ T*>`. The `__gm__` qualifier must be preserved through the cast.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  __gm__ float* input = input_gm;              // type mismatch
  float val = input_gm[i];                      // subscript on uint8_t*

  // AFTER (compiles):
  __gm__ float* input = reinterpret_cast<__gm__ float*>(input_gm);
  float val = input[i];                          // correct typed access

  // For const pointers:
  __gm__ const int* edge_in = reinterpret_cast<__gm__ const int*>(edge_in_gm);
  ```
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK — kernel launch pattern)

---

### EC-3: `LAUNCH_BOUND` value exceeds 512

- **Error pattern**:
  ```
  error: 'LAUNCH_BOUND' attribute parameter 1024 exceeds maximum allowed value 512
  ```
  or at runtime: incorrect results / register spilling when LAUNCH_BOUND > 512 with complex kernel logic.
- **Root cause**: Ascend950PR supports LAUNCH_BOUND up to 2048 in theory, but **512 is the practical maximum** for kernels with non-trivial register usage. At 512 threads, each thread gets 64 registers (128KB register file / 512 threads / 4 bytes). Higher thread counts reduce per-thread registers, causing spills to slower memory and often incorrect codegen.
- **Fix**:
  ```cpp
  // BEFORE (risky or fails):
  LAUNCH_BOUND(1024) inline void kernel_vf(...) { ... }

  // AFTER (safe default):
  LAUNCH_BOUND(512) inline void kernel_vf(...) { ... }

  // Define as named constant:
  constexpr uint32_t OP_THREAD_NUM = 512;
  LAUNCH_BOUND(OP_THREAD_NUM) inline void kernel_vf(...) { ... }
  ```
- **Note**: CUDA `__launch_bounds__(1024)` must be reduced to 512 when migrating. The dispatcher `Simt::Dim3{OP_THREAD_NUM}` must match the LAUNCH_BOUND value.
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK)

---

### EC-4: `simt_compat.h` conflicts in NPU mode

- **Error pattern**:
  ```
  error: redefinition of 'blockDim' as different kind of symbol
  ```
  or:
  ```
  error: expected unqualified-id
  ```
  (when `#define blockDim` macro clashes with CANN's built-in `blockDim` in NPU mode)
- **Root cause**: `simt_compat.h` defines `blockDim` and `threadIdx` as macros that map to raw CPU-mode globals (`g_threadDimX`, `g_threadIdxX`). In NPU mode, CANN provides its own built-in `blockDim`/`threadIdx` — the macros collide with these built-ins. The header must only be included in CPU debug builds.
- **Fix**:
  ```cpp
  // BEFORE (fails on NPU):
  #include "simt_compat.h"    // unconditional include → macro conflicts

  // AFTER (conditional):
  #if defined(ASCENDC_CPU_DEBUG)
  #include "simt_compat.h"
  #endif
  ```
  The guard works because:
  - CPU debug mode: `ASCENDC_CPU_DEBUG` is defined by tikicpulib CMake target → macros active
  - NPU mode: `ASCENDC_CPU_DEBUG` is not defined → header skipped, CANN built-ins used
- **Related**: None (project-specific compatibility layer)

---

### EC-5: `static_cast<float>(bfloat16_t)` fails in bisheng

- **Error pattern**:
  ```
  error: not support bf16 type cast
  ```
  or:
  ```
  error: static_cast from 'bfloat16_t' to 'float' is not allowed
  ```
- **Root cause**: Bisheng compiler (CANN 9.0.0 and 9.0.T501) does not support scalar `static_cast` between `bfloat16_t` and `float` in either direction. The `half` (fp16) type works fine with `static_cast`. This is a known bisheng limitation (PB-4 in PLATFORM_BUGS.md).
- **Fix (SIMT kernel — use bit-manipulation)**:
  ```cpp
  // BEFORE (fails):
  bfloat16_t val = input[i];
  float fval = static_cast<float>(val);    // ❌ bisheng rejects this

  // AFTER (bit-manipulation workaround):
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }

  template <>
  __aicore__ inline float simt_to_float<bfloat16_t>(bfloat16_t v) {
    uint16_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    uint32_t f32bits = static_cast<uint32_t>(bits) << 16;
    float result;
    __builtin_memcpy(&result, &f32bits, sizeof(result));
    return result;
  }

  // Reverse: float → bfloat16_t
  template <typename T>
  __aicore__ inline T simt_from_float(float v) { return static_cast<T>(v); }

  template <>
  __aicore__ inline bfloat16_t simt_from_float<bfloat16_t>(float v) {
    uint32_t f32bits;
    __builtin_memcpy(&f32bits, &v, sizeof(f32bits));
    uint16_t bits = static_cast<uint16_t>(f32bits >> 16);  // truncate
    bfloat16_t result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
  }
  ```
- **Fix (SIMD kernel — use Cast intrinsic)**:
  ```cpp
  // Cast(bf16→float) is lossless and works:
  Cast(floatBuf, bf16Buf, RoundMode::CAST_NONE, count);
  float w = floatBuf.GetValue(i);
  ```
- **WARNING**: `Cast(bf16→half)` is LOSSY — bf16 exponent=8bit overflows half exponent=5bit, producing `inf` for large values. Always cast bf16→float (lossless).
- **Related**: P-P27 (bf16 scalar via Cast + GetValue)

---

### EC-6: `using namespace AscendC::Simt` causes `GetBlockIdx` ambiguity

- **Error pattern**:
  ```
  error: call to 'GetBlockIdx' is ambiguous
  note: candidate function: int32_t AscendC::Simt::GetBlockIdx()
  note: candidate function: int64_t GetBlockIdx()
  ```
  (typically 20+ errors across a file since every `GetBlockIdx`/`GetBlockNum` call is ambiguous)
- **Root cause**: CANN defines TWO `GetBlockIdx()` functions — `AscendC::Simt::GetBlockIdx()` returning `int32_t` and a basic API `GetBlockIdx()` returning `int64_t`. Adding `using namespace AscendC::Simt;` pulls the Simt version into the same scope as the basic API version, making every unqualified call ambiguous.
- **Fix**:
  ```cpp
  // BEFORE (ambiguous):
  using namespace AscendC;
  using namespace AscendC::Simt;   // ❌ pulls in Simt::GetBlockIdx

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // ambiguous: Simt::GetBlockIdx vs basic_api
  }

  // AFTER (unambiguous):
  using namespace AscendC;         // ✅ only basic API GetBlockIdx (int64_t)
  // No "using namespace AscendC::Simt;" — dispatchers use qualified Simt::VF_CALL

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // resolves to basic_api int64_t version
    Simt::VF_CALL<kernel_vf<T>>(   // Simt:: qualified prefix for VF_CALL
        Simt::Dim3{THREAD_NUM}, ...);
  }
  ```
- **Note**: Kernel VF functions themselves don't call `GetBlockIdx` — they receive `block_index` as a parameter from the dispatcher. Only dispatchers need `GetBlockIdx`/`GetBlockNum`.
- **Related**: OL-14 (OPERATIONAL_KNOWLEDGE.md)

---

### EC-7: `Simt::atomicAdd` — wrong namespace

- **Error pattern**:
  ```
  error: no member named 'atomicAdd' in namespace 'AscendC::Simt'
  ```
  or:
  ```
  error: call to 'atomicAdd' is ambiguous
  ```
  (when both `Simt::atomicAdd` and global `atomicAdd` are attempted)
- **Root cause**: `atomicAdd` on AscendC is a **global built-in function**, not a member of the `AscendC::Simt` namespace. This differs from other Simt APIs like `Simt::VF_CALL`, `Simt::Dim3`, `Simt::WarpReduceAddSync` which are namespaced. The CUDA migration tends to add `Simt::` prefix to everything — `atomicAdd` is the exception.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  Simt::atomicAdd(base + offset, value);       // ❌ not in Simt namespace
  AscendC::Simt::atomicAdd(base + offset, value);  // ❌ same error

  // AFTER (compiles):
  atomicAdd(base + offset, value);             // ✅ global built-in, no namespace
  ```
- **Supported types**: `float`, `half`, `bfloat16_t`, `int32_t` — all use the same unqualified `atomicAdd`.
- **Related**: P-P31 (NPU native atomicAdd — no fastAtomicAdd packed pair needed)

---

### EC-8: Missing `#include <kernel_operator.h>`

- **Error pattern**:
  ```
  error: unknown type name 'GM_ADDR'
  error: unknown type name '__gm__'
  error: use of undeclared identifier 'atomicAdd'
  error: unknown type name 'bfloat16_t'
  error: no member named 'VF_CALL' in namespace 'AscendC::Simt'
  ```
  (cascade of errors — types, macros, and functions all undefined)
- **Root cause**: `kernel_operator.h` is the master header for AscendC. It pulls in all CANN types (`GM_ADDR`, `__gm__`, `bfloat16_t`, `half`), SIMT APIs (`Simt::VF_CALL`, `Simt::Dim3`), SIMD APIs (`DataCopy`, `Cast`), atomics (`atomicAdd`), and platform macros (`LAUNCH_BOUND`, `__aicore__`). Without it, nothing AscendC-specific compiles.
- **Fix**:
  ```cpp
  // BEFORE (cascade of errors):
  #include <cstdint>
  // missing kernel_operator.h

  // AFTER (compiles):
  #include <kernel_operator.h>    // ✅ MUST be first AscendC include
  #include <cstdint>
  ```
- **Rule**: Every `.h` and `.cpp` file that uses any AscendC type or API must include `<kernel_operator.h>` as its first AscendC include. Standard library headers (`<cstdint>`, `<cstring>`) can come before or after.
- **Related**: None (basic AscendC requirement)

---

### EC-9: Missing `namespace ascendc_ops {}` wrapper

- **Error pattern**:
  ```
  error: redefinition of 'ITER'
  error: redefinition of 'simt_to_float'
  error: use of undeclared identifier 'POOLING_FWD_THREAD_NUM'
  ```
  (name collisions between kernel files, or missing constant/helper definitions when files are compiled together)
- **Root cause**: All kernel code in this project must be wrapped in `namespace ascendc_ops { ... }`. Without the namespace: (1) macros like `ITER(x,y)` and helper templates like `simt_to_float` collide when multiple kernel headers are included in the same translation unit; (2) dispatcher `.cpp` files use `using namespace ascendc_ops;` to access kernel VF functions and constants — if the VF functions are in the global namespace, `using namespace ascendc_ops;` finds nothing.
- **Fix**:
  ```cpp
  // BEFORE (collisions, missing symbols):
  #include <kernel_operator.h>
  using namespace AscendC;

  #define ITER(x, y) (((x) + (y) - 1) / (y))

  template <typename T>
  __simt_vf__ __aicore__
  LAUNCH_BOUND(512) inline void my_kernel_vf(GM_ADDR input_gm, ...) { ... }

  // AFTER (namespaced):
  #include <kernel_operator.h>

  namespace ascendc_ops {
  using namespace AscendC;

  #define ITER(x, y) (((x) + (y) - 1) / (y))

  template <typename T>
  __simt_vf__ __aicore__
  LAUNCH_BOUND(512) inline void my_kernel_vf(GM_ADDR input_gm, ...) { ... }

  }  // namespace ascendc_ops
  ```
  Corresponding dispatcher file:
  ```cpp
  #include "my_kernel.h"
  using namespace ascendc_ops;

  extern "C" __global__ __aicore__ void my_kernel_fp32(GM_ADDR input_gm, ...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Simt::VF_CALL<my_kernel_vf<float>>(
        Simt::Dim3{512}, input_gm, ..., GetBlockIdx(), GetBlockNum());
  }
  ```
- **Note**: `extern "C" __global__` dispatcher functions are in the global namespace (required by CANN runtime). Only the VF functions, helpers, and constants go inside `namespace ascendc_ops`.
- **Related**: None (project convention for multi-file compilation)

---

### EC-10: aclrtlaunch Undefined Reference (Linker)

- **Error pattern**: `undefined reference to 'aclrtlaunch_xxx(...)'`
- **Root cause**: Auto-generated `host_stub.cpp` exports kernel launch functions as **C symbols** (no mangling). Test code that declares them without `extern "C"` gets C++ mangled names → linker mismatch.
- **Fix**:
  ```cpp
  // ❌ Wrong — C++ mangling
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);

  // ✅ Correct — C linkage
  extern "C" {
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);
  }
  ```
- **Related**: PB-8 in PLATFORM_BUGS.md

---

### EC-11: CANN Build Fails at merge_mix_obj.sh (95%)

- **Error pattern**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh`
- **Root cause**: `CMAKE_BUILD_TYPE` not set → cmake passes empty `--build-type` to `merge_mix_obj.sh` → `shift 2` fails
- **Fix**: Always pass `-DCMAKE_BUILD_TYPE=Release` to cmake
- **Related**: PB-7 in PLATFORM_BUGS.md

---

### EC-12: `block_num` / `block_index` macro collision in parameter names

- **Error pattern**:
  ```
  error: cannot initialize a parameter of type 'int64_t (*)(void)' with an rvalue of type 'int64_t'
  note: expanded from macro 'block_num'
  #define block_num get_block_num()
  ```
- **Root cause**: CANN defines `block_num` as a macro expanding to `get_block_num()` (a function). When used as a function parameter name, `int64_t block_num` becomes `int64_t get_block_num()` -- a function declaration, not a parameter. Similarly, `block_index` may collide with other CANN macros.
- **Fix**: Rename parameters to avoid CANN macro names:
  ```cpp
  // BEFORE (fails):
  void Init(GM_ADDR x, int64_t block_index, int64_t block_num) { ... }

  // AFTER (compiles):
  void Init(GM_ADDR x, int64_t blk_idx, int64_t blk_cnt) { ... }
  ```
- **CANN macros to avoid as identifiers**: `block_num`, `block_idx`, and any other identifier in `__clang_cce_aicore_builtin_vars.h`.
- **Related**: OL-14 (namespace ambiguity)

---

### EC-13: `AscendC::SyncFunc<>` does not exist

- **Error pattern**:
  ```
  error: no member named 'SyncFunc' in namespace 'AscendC'
  ```
- **Root cause**: There is no `AscendC::SyncFunc` API. The generated code (from templates or LLM) may invent this API for pipe synchronization. The correct API uses `SetFlag`/`WaitFlag` with event IDs fetched from `GetTPipePtr()->FetchEventID()`.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();

  // AFTER (compiles):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ev);
  ```
- **Common sync events**: MTE2_S (GM→scalar), S_MTE3 (scalar→GM write), V_S (VEC→scalar), S_V (scalar→VEC)
- **Evidence**: Cumsum V1 build failure (2026-04-09)

---

### EC-14: `TQue<..., 0>` — depth must be >= 1

- **Error pattern**:
  ```
  error: static assertion failed: must use AllocTensor<LocalTensor&> api while tque's depth is zero
  ```
- **Root cause**: `TQue` template's second parameter is the depth (number of buffer slots). Depth 0 means "use pass-by-reference AllocTensor API" which has a completely different usage pattern. Standard AllocTensor/EnQue/DeQue/FreeTensor requires depth >= 1.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::TQue<AscendC::TPosition::VECIN, 0> xQueue_;

  // AFTER (works):
  AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_;
  ```
- **Evidence**: Cumsum V1 build failure (2026-04-09)

---

### EC-15: `PipeBarrier<PIPE_S>` not valid on Ascend950PR

- **Error pattern**:
  ```
  error: the range of 1st parameter must be [4, 6]
  ```
  (from `kernel_reg.h`, triggered by `PipeBarrier<PIPE_S>()`)
- **Root cause**: On Ascend950PR, `pipe_barrier()` only accepts pipe values 4 (PIPE_MTE2), 5 (PIPE_V), 6 (PIPE_MTE3). The scalar pipe (PIPE_S) is not supported for PipeBarrier. To synchronize the scalar pipe, use `SetFlag`/`WaitFlag` with appropriate event types.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::PipeBarrier<PIPE_S>();

  // AFTER (S→MTE3 sync for scalar writes visible to MTE3 output):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(ev);

  // For S→V sync (scalar writes visible to VEC):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev);
  ```
- **Valid PipeBarrier pipes**: PIPE_MTE2 (4), PIPE_V (5), PIPE_MTE3 (6) — that's it.
- **Evidence**: Sort V1 build failure (2026-04-09)

---

### EC-16: DataCopy alignment overwrite in strided/chunked copies

- **Error pattern**: Precision failures on specific test cases where non-aligned chunk sizes cause data corruption in adjacent output regions
- **Root cause**: DataCopy requires 32-byte aligned element counts. When `chunk_size % ALIGN != 0`, naively aligning up writes extra elements past the chunk boundary, corrupting adjacent tensor data.
- **Fix (overlapping tail write)**:
  ```
  1. Copy floor_aligned(chunk) elements normally
  2. Copy last ALIGN elements starting at (chunk - ALIGN), overlapping with already-written region
  ```
  The overlap is harmless (same values re-written), and tail elements are placed correctly without overflow.
- **Condition**: Strided/chunked DMA with non-aligned chunk boundaries (e.g., cat along non-last dim)
- **Evidence**: Cat V1 failed 3/51 cases, fixed in V2 (2026-04-09)

---

## Quick Lookup Table

| EC | Error keyword | One-line fix |
|----|--------------|--------------|
| EC-1 | `calling a __host__ function from __aicore__` | Add `__aicore__ inline` to helper |
| EC-2 | `cannot initialize '__gm__ T*' with 'GM_ADDR'` | `reinterpret_cast<__gm__ T*>(gm_addr)` |
| EC-3 | `LAUNCH_BOUND exceeds maximum` | Reduce to 512 |
| EC-4 | `redefinition of 'blockDim'` | Guard with `#if defined(ASCENDC_CPU_DEBUG)` |
| EC-5 | `not support bf16 type cast` | Use `simt_to_float()` bit-manipulation (P-P27) |
| EC-6 | `call to 'GetBlockIdx' is ambiguous` | Remove `using namespace AscendC::Simt;` (OL-14) |
| EC-7 | `no member 'atomicAdd' in 'Simt'` | Use unqualified `atomicAdd()` (global built-in) |
| EC-8 | `unknown type name 'GM_ADDR'` | Add `#include <kernel_operator.h>` as first include |
| EC-9 | `redefinition of 'ITER'` / missing symbols | Wrap all code in `namespace ascendc_ops {}` |
| EC-10 | `undefined reference to 'aclrtlaunch_'` | Add `extern "C" {}` around declaration |
| EC-11 | `merge_mix_obj.sh Error 1` at 95% | Add `-DCMAKE_BUILD_TYPE=Release` |
| EC-12 | `cannot initialize 'int64_t (*)(void)'` + `expanded from macro 'block_num'` | Rename param: `blk_idx`/`blk_cnt` |
| EC-13 | `no member named 'SyncFunc' in namespace 'AscendC'` | Use `SetFlag`/`WaitFlag` with `FetchEventID` |
| EC-14 | `static assertion failed: must use AllocTensor...depth is zero` | Change TQue depth from 0 to ≥1 |
| EC-15 | `the range of 1st parameter must be [4, 6]` | No `PipeBarrier<PIPE_S>`, use SetFlag/WaitFlag for S pipe |
| EC-16 | Non-aligned chunk DataCopy corrupts adjacent data | Overlapping tail write: copy last ALIGN elems separately |
| EC-17 | Sub-align chunk overwrite in compact output | nblk=1 + padded alloc + narrow view |

---

### EC-17: Sub-alignment chunk overwrite in compact (tightly-packed) output

- **Error pattern**: Precision failures when chunk_size < DataCopy alignment AND output elements are tightly packed (no gaps between chunks from different outer iterations)
- **Root cause**: DataCopy writes aligned count of elements. When chunk < align, excess elements overwrite the next chunk's data. In Cat's output (strided with gaps), overlapping tail write works. In Split's output (compact), there are no gaps — adjacent chunks are immediately adjacent.
- **Fix (host-side)**: Detect `chunk < align && outer > 1`. Use `nblk=1` (serial execution — overwrites self-correct within one block) + allocate padded output + narrow to exact size.
- **Applicability**: Any kernel writing to compact output with non-aligned chunk boundaries
- **Evidence**: Split V1 failed 4/57 cases, fixed in V2 (2026-04-09)

---

### EC-18: Forward-overwrite data race in multi-block non-aligned DMA

- **Error pattern**: Precision failures in multi-block kernels where non-aligned DataCopy uses forward-overwrite technique (write ALIGN elements, let next iteration overwrite tail). When multiple blocks process different rows in parallel, the overwrites from different blocks race.
- **Root cause**: Block K-1's tail overwrite extends into Block K's write region. Without ordering, Block K may read before K-1's overwrite completes, or K's write may be overwritten by K-1's stale data.
- **Fix**: Two approaches:
  1. **Per-row overlap** (chunk >= ALIGN): re-copy last ALIGN elements from `chunk - ALIGN` offset. No cross-row overwrite. Safe for multi-block.
  2. **nblk=1 + padded alloc** (chunk < ALIGN): serialize to one block. Over-allocate output with ALIGN padding, narrow() after kernel.
- **Evidence**: Split V3 — V2 forward-overwrite caused 12 new failures, fixed with per-row overlap (2026-04-09)

---

### EC-19: PadTiling name conflict with CANN built-in

- **Error pattern**: `error: reference to 'PadTiling' is ambiguous`
- **Root cause**: CANN `kernel_tiling.h` defines `PadTiling` in `AscendC::tiling` namespace and imports it via `using`. Custom struct with same name conflicts.
- **Fix**: Rename custom tiling struct to unique name (e.g., `PadOpTiling`, `MyPadTiling`)
- **Evidence**: Pad V2 first build (2026-04-09)

### EC-20: Tiling CPU→NPU copy must happen AFTER all fields finalized

- **Error pattern**: Wrong results — tiling field has stale value on NPU
- **Root cause**: If pybind writes `tiling.field = X` after `tiling_npu = tiling_cpu.to(device)`, the NPU copy has old value
- **Fix**: Finalize ALL tiling fields, then copy once
- **Evidence**: Pad V2 mode routing bug (2026-04-09)

### EC-21: VECIN-only pipeline cannot do GM→UB→GM pass-through

- **Error pattern**: Data corruption or sync hang when doing DataCopy(UB←GM) then DataCopy(GM←UB) through VECIN queue only
- **Root cause**: VECIN syncs MTE2→VEC, but MTE3 store needs VEC→MTE3 sync. Without a VEC op and VECOUT queue, the pipeline has a sync gap.
- **Fix**: Split-queue pattern: VECIN for load + VECOUT for store + VEC identity op (Adds 0.0f) between them
- **Evidence**: Pad V2, Cat, Split all use this pattern (P-CAT-1)

### EC-22: Multi-block aligned DataCopy overwrite race

- **Error pattern**: Precision failures that disappear with nblk=1 but appear with nblk>1. Same elements fail deterministically. Mismatch ratio ~0.01-18%.
- **Root cause**: DataCopy requires aligned element counts. When `count % ALIGN != 0`, writing `ceil(count/ALIGN)*ALIGN` elements overwrites adjacent output positions. Single-block: next tile overwrites stale values. Multi-block: overwrite lands in another block's range → write-write race.
- **Fix**: Overlap-tail technique for ALL DataCopy calls with non-aligned counts. Write `floor(count/ALIGN)*ALIGN` normally, then re-write last ALIGN elements starting at `count - ALIGN`.
- **Diagnostic**: nblk=1 vs nblk=N A/B test (OL-43) — if nblk=1 passes, it's this bug.
- **Evidence**: Pad V3-V5 (2026-04-10): nblk=1 → 51/51 PASS, nblk=56 → 28/51 PASS
- **Fix approach 1 (partial)**: Row-level partitioning — ensures block boundaries at row boundaries, reducing but not eliminating races (28→30 PASS)
- **Fix approach 2 (partial)**: Pre-fill output with fill_value (torch::full) — does NOT help because overflow writes source data, not fill_value
- **Fix approach 3 (verified)**: 3-phase segment processing (fill-left → source → fill-right). Source phase overflow lands in fill-right area, immediately overwritten. Verified: case 38 (previously always FAIL) now PASS.
- **Fix approach 4 (NOT recommended)**: SafeWrite with overlap-tail `local[t-AL]` — triggers UB alignment error (error code 80). SafeWrite with scalar GetValue also triggers VEC alignment errors due to pipeline interference.
- **Generalized fix**: For any multi-block SIMD kernel doing DataCopy-to-GM with non-aligned tile counts, ensure processing order guarantees that overflow regions are overwritten by subsequent writes. 3-phase decomposition (pre-fill → source → post-fill) is the most reliable pattern.
