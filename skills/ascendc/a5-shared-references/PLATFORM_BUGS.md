# Platform Known Bugs & Workarounds

Verified bugs in CANN/bisheng/hardware. Check this before debugging unexpected behavior.

## CANN Bugs

### PB-1: Typed Kernel Entry Crash (CANN 9.0.0)
- **Symptom**: Error 507035 on kernel launch with typed entry points (e.g., `_fp32` suffix)
- **Affected**: CANN 9.0.0 with bisheng 2026-03-21
- **Workaround**: Use legacy untyped entry points (single dispatcher .cpp, cast inside kernel)
- **Status**: OPEN (not fixed in CANN 9.0.T501)
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-16

### PB-2: TQue<VECIN,2> Data Corruption
- **Symptom**: 99.5% elements corrupted when using TQue with depth 2
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use TQue<VECIN,4> (depth 4 works correctly)
- **Status**: OPEN
- **Evidence**: hardware/target/ascend950pr.md, E13 test data

### PB-3: NPU Device 0 Post-Reboot Failure
- **Symptom**: Error 507033 on device 0 after server reboot (2026-04-01)
- **Affected**: A5 server 90.90.93.35, device 0 only
- **Workaround**: Use devices 1-4, 7
- **Status**: Hardware issue, may require RMA

## Bisheng Compiler Bugs

### PB-4: bf16 Scalar Cast Failure
- **Symptom**: `static_cast<float>(bf16_var)` produces wrong values in scalar context
- **Affected**: bisheng 2026-03-21 (CANN 9.0.0)
- **Workaround**: Use bit-manipulation helpers (`bf16_scalar_to_float`, `simt_to_float`, `simt_from_float`) OR SIMD `Cast()` intrinsic
- **Status**: OPEN
- **Evidence**: `tests/repro/bf16_cast_repro.cpp` (7 test cases), P-P27 pattern
- **Detail**: Scalar bf16→float cast emits wrong instruction sequence. SIMD Cast() with `RoundMode::CAST_NONE` works fine.

### PB-5: -O2 Required for NPU (No -O0)
- **Symptom**: Kernel may produce wrong results or crash with -O0 on NPU
- **Affected**: All NPU builds
- **Workaround**: Always use -O2 for NPU builds (-O0 only for CPU debug mode)
- **Status**: By design (bisheng optimizations required for correct codegen)

### PB-7: CANN merge_mix_obj.sh Crash with Empty --build-type
- **Symptom**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh` — `shift 2` fails
- **Root cause**: `cmake` invokes `merge_mix_obj.sh --build-type` without a value when `CMAKE_BUILD_TYPE` is unset. The bash `shift 2` fails because only 1 arg remains.
- **Affected**: CANN 9.0.0, AIV-only kernels (AIC dir empty, merge step still runs)
- **Workaround**: Always set `-DCMAKE_BUILD_TYPE=Release` in cmake invocation
- **Status**: OPEN (CANN build system bug)
- **Evidence**: MXFP4 project (2026-04-07), `merge_mix_obj.sh` line `shift 2` on `--build-type`

## Build Integration Issues

### PB-8: aclrtlaunch Stub Requires extern "C" Declaration
- **Symptom**: Linker error `undefined reference to aclrtlaunch_xxx(...)` when calling kernel from test code
- **Root cause**: Auto-generated `host_stub.cpp` exports functions as C symbols (no name mangling). Test code declaring them as C++ gets mangled names → linker mismatch.
- **Workaround**: Always use `extern "C" { uint32_t aclrtlaunch_xxx(...); }` in test code
- **Status**: By design (not a bug, but easy to forget)
- **Evidence**: MXFP4 test (2026-04-07)

## Operational Issues

### PB-6: Zombie Process Accumulation
- **Symptom**: Training/benchmark hangs, resource exhaustion after multiple runs
- **Affected**: Docker containers on A5 server
- **Workaround**: **Always restart container before every experiment**
- **Evidence**: 2280 zombies found after E13h

## Archived

### PB-7 (duplicate, line 68): Shared NPU Contention
- **Archived**: 2026-04-09. Reason: duplicate ID with PB-7 (line 43, merge_mix_obj). Content moved to PB-10.

---

### PB-10: Shared NPU Contention
- **Symptom**: Benchmark results vary wildly between runs
- **Affected**: A5 server (shared infrastructure)
- **Workaround**: Run `npu-smi info` before benchmarking, check for other processes
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-15

### PB-9: UB-to-UB DataCopy Silent Data Corruption
- **Symptom**: `DataCopy(localDst, localSrc, count)` between two LocalTensors (both in UB) silently produces garbage data. No compile error, no runtime error — just wrong values. Discovered when LayerNorm V2 passed for norm_size ≤ 4096 (single tile) but produced ~20% mismatch with mean_abs_diff ~1.14 for norm_size > 4096 (multi-tile). Removing the UB-to-UB DataCopy and operating directly on the dequeued tensor fixed it completely.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Never copy between LocalTensors using DataCopy. Instead:
  - Operate directly on the source tensor (e.g., run BinaryFoldReduceSum on the dequeued xd tensor)
  - Use VEC ops as a "copy": `Adds(dst, src, 0.0f, count)` if you must copy
  - Or use `Duplicate` to zero a buffer, then `Add(dst, dst, src, count)`
- **Status**: OPEN
- **Evidence**: LayerNorm V2 debugging session 2026-04-09; kernel/layernorm_kernel.h Pass 1 fix

### PB-11: DataCopy to TBuf<VECCALC> — silent corruption on multi-iteration loops
- **Symptom**: `DataCopy(TBuf<VECCALC>::Get<T>(), GM_tensor, count)` with manual `SetFlag<MTE2_V>/WaitFlag<MTE2_V>` sync produces correct data on the first loop iteration but stale/corrupt data on subsequent iterations. Related to PB-9 (both are DataCopy corruption) but distinct mechanism: PB-9 is UB→UB; PB-11 is GM→VECCALC with manual sync in a loop.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use `TQue<VECIN, 1>` instead of `TBuf<VECCALC>` for any buffer that receives DataCopy from GM in a loop. The TQue's `AllocTensor/EnQue/DeQue/FreeTensor` pattern provides reliable MTE2→VEC synchronization. Single-iteration usage of TBuf<VECCALC> with DataCopy appears safe.
- **Status**: OPEN
- **Evidence**: DynamicQuant (#29) smooth_scales — 3 cases with row_size > TILE_SIZE failed (2.7%-22.6% mismatch, max_abs_diff=252) due to stale smooth_scales data on 2nd+ tile. First mismatch always at exact TILE_SIZE boundary. Fixed by switching to TQue<VECIN,1>. 42/42 PASS after fix.

---

## How to Add New Bugs

Append to the appropriate section with:
```
### PB-N: Short Description
- **Symptom**: What you observe
- **Affected**: Platform/version
- **Workaround**: How to work around it
- **Status**: OPEN/FIXED(version)/BY_DESIGN
- **Evidence**: Link to test/doc
```
