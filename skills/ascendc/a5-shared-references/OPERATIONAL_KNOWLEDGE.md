# Operational Knowledge Base

> Bitter lessons, pitfalls, and trust calibrations accumulated during CUDA→AscendC migrations.
> Entries are categorized for selective loading by skill agents.
> Categories: process, platform_bug, measurement, environment, trust_calibration, conditional_insight, algorithm_selection

---

## OL-1: Never mark issues "confirmed safe" without code change
- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when reviewing generated code for correctness
- **Lesson**: int64 truncation was flagged 5 times by experts. The first 3 times, CC marked "confirmed safe" without changing code. The truncation was real -- `static_cast<int>(int64_var)` overflows when `dim * index > INT32_MAX`. It took 5 rounds and 72 code changes across 5 files to finally fix all instances.
- **Evidence**: EXPERT_FEEDBACK.md E8-3 (timeline of 5 rounds), CLAUDE.md lines 27-37

---

## OL-2: Expert feedback must result in code changes, never just documentation
- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when processing expert review feedback
- **Lesson**: Marking expert-reported issues as "already safe" in documentation -- without modifying, compiling, and testing the code -- is forbidden. The int64 problem was annotated "confirmed safe" in docs 3 times while the truncating `static_cast<int>()` calls remained in the source. The rule is: modify code + compile + run precision tests, or the issue stays open.
- **Evidence**: CLAUDE.md lines 34-37, EXPERT_FEEDBACK.md E8-3 round-by-round history

---

## OL-3: Radix sort fails on AscendC; use counting sort
- **Category**: algorithm_selection
- **Loaded by**: Builder, Optimizer
- **Trigger**: when implementing on-device sort for AscendC SIMT kernels
- **Lesson**: A 4-pass byte-wise radix sort using global atomicAdd scatter was implemented first. It failed completely (0/61 clusters correct) because multi-pass scatter is unstable on AscendC -- each pass's global atomicAdd destroys the ordering established by the previous pass. Switching to counting sort (histogram -> prefix_sum -> scatter, single-pass) achieved 61/61 correct and 2.84x speedup over host `std::sort`. Counting sort works because same-key ordering does not affect the downstream register accumulation.
- **Evidence**: OPTIMIZATION_PLAN.md Batch 7 ("radix sort 0/61 correct"), `output/src/pooling/radix_sort_kernel.h`

---

## OL-4: TQue bug — RESOLVED in CANN 9.0.0, TQue is now the PREFERRED approach
- **Category**: platform_bug (RESOLVED)
- **Loaded by**: Builder, QA
- **Trigger**: when choosing between TQue and PipeBarrier for SIMD pipelines
- **Lesson**: TQue<VECIN,2> data corruption was a CANN 9.0.T501 bug. **Resolved in CANN 9.0.0.** SG backward uses TQue<VECIN,2> successfully since E12. SG forward switched from PipeBarrier to TQue<VECIN,4> in E13 — **1.6-2.3x speedup** on all cases. **Always prefer TQue over PipeBarrier<PIPE_ALL>** for SIMD pipeline overlap. PipeBarrier serializes ALL pipes (MTE2+VEC+MTE3+Scalar); TQue only syncs the necessary MTE2→VEC transition.
- **Evidence**: E13-P1 benchmark data (2026-04-01), EXPERT_FEEDBACK.md E13 section
- **Action**: If generating new SIMD kernels, use `TQue<VECIN,4>` + `TQue<VECOUT,2>` pattern. Never use PipeBarrier<PIPE_ALL> in hot loops.

---

## OL-5: Use aclrtEvent not chrono for NPU timing; memset outside event
- **Category**: measurement
- **Loaded by**: QA
- **Trigger**: when writing or reviewing NPU benchmark code
- **Lesson**: Two measurement errors compounded in early benchmarks. (1) `std::chrono` measures wall-clock time including host synchronization overhead, not device execution time -- use `aclrtCreateEvent` / `aclrtRecordEvent` / `aclrtEventElapsedTime` instead. (2) `aclrtMemset` (output buffer clearing) was placed inside the timed region, inflating backward kernel times. Fixing both changed Pooling backward from 154ms (16.9x slower than GPU) to 151ms (device-only), and the Sparse-Gather backward "NPU beats GPU" claim was invalidated entirely.
- **Evidence**: BENCHMARK_METHODOLOGY.md Section 1, output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md ("旧计时方法")

---

## OL-6: Restart container before experiments (zombie processes)
- **Category**: environment
- **Loaded by**: QA
- **Trigger**: before launching any benchmark or experiment on the A5 container
- **Lesson**: After long experiment sessions, zombie processes accumulate from previous NPU kernel launches. 2280 zombie processes were found after one extended run. These cause training hangs, kernel launch failures, resource exhaustion, and unreproducible timing results. Always restart the container (`docker restart can_torch_cann_device_1`) before every experiment session.
- **Evidence**: CLAUDE.md (global) "CRITICAL: ALWAYS restart container before EVERY experiment", MEMORY.md

---

## OL-7: Always verify expert claims with hardware test
- **Category**: trust_calibration
- **Loaded by**: Optimizer, Planner
- **Trigger**: when an expert makes a hardware performance claim without providing measurement data
- **Lesson**: An expert stated that 128-bit loads are slower than 64-bit on Ascend950PR. Hardware testing (`tests/load_width_test/`) showed the opposite: 128-bit is 1.2x-2.1x faster than 32-bit for sequential reads (1MB: 1.17x, 16MB: 1.39x, 64MB: 2.12x). The expert's claim may have been valid for a different access pattern (random/indirect indexing), but for the sequential reads in our kernels, wider loads are strictly better. Always run a targeted micro-benchmark before accepting or rejecting a hardware claim.
- **Evidence**: EXPERT_FEEDBACK.md E7-3 (128-bit实测数据), `tests/load_width_test/`

---

## OL-8: Oversubscription helps unsorted bwd (-28%) but hurts sorted bwd (+17%)
- **Category**: conditional_insight
- **Loaded by**: Optimizer
- **Trigger**: when tuning launch parameters (nblk) for scatter-add backward kernels
- **Lesson**: Block oversubscription (nblk=448 vs 56) reduced unsorted backward time by 28% by dispersing atomicAdd contention across more time-sliced blocks. But after implementing sorted-edge register accumulation (which eliminates atomicAdd contention entirely), the same oversubscription increased backward time by ~17% due to pure launch overhead with no contention left to disperse. The root cause is that oversubscription's benefit comes solely from dispersing atomicAdd contention -- once sort+register-accum removes that contention, only the overhead remains. Launch parameter sweeps must be re-run on the final kernel variant, not carried over from intermediate versions.
- **Evidence**: EXPERT_FEEDBACK.md E7-8 (nblk sweep tables for unsorted vs sorted), MSPROF_AGENT_GUIDE.md (nblk=56 vs 448 msprof data)

---

## OL-9: SG "NPU beats GPU 0.6x" claim retracted after timing bug fix
- **Category**: measurement
- **Loaded by**: QA
- **Trigger**: when an NPU kernel appears to outperform GPU on a comparable workload
- **Lesson**: Early Sparse-Gather benchmarks using chrono timing with memset inside the measurement window showed SG backward xlarge at 0.29ms -- appearing to beat GPU's 0.48ms (0.6x ratio, "NPU faster"). After switching to aclrtEvent device timing with memset outside the event, the actual NPU time was 3.87ms (8x slower than GPU, not faster). The 13x measurement error came from chrono capturing only host-side launch latency while the kernel ran asynchronously on the device. Any "NPU beats GPU" result should be treated with extreme skepticism and verified with proper device-event timing.
- **Evidence**: EXPERT_FEEDBACK.md Q4/Q6 analysis ("当前容器 SG xlarge bwd 3.87ms, 旧容器 0.29ms"), output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md

---

## OL-10: README-driven doc updates
- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when creating or modifying any documentation file
- **Lesson**: Every documentation change must use README.md as the index and verify that all related documents are updated in sync. New files must be added to the README directory tree and document hierarchy. Batch results go into OPTIMIZATION_PLAN.md; README "current status" keeps only the final snapshot. This rule exists because Batch 9 changes were recorded in REPORT.md and EXPERT_FEEDBACK.md but README.md and OPTIMIZATION_PLAN.md were forgotten, leaving the project index stale.
- **Evidence**: CLAUDE.md lines 39-44, README.md directory tree

---

## OL-11: Hardware doc said Ascend910_9589 but actual SOC is Ascend950PR_9589
- **Category**: trust_calibration
- **Loaded by**: Builder, QA
- **Trigger**: when configuring SOC_VERSION for CMake builds or kernel compilation
- **Lesson**: The hardware documentation and initial setup guides listed the SOC as `Ascend910_9589`. The actual chip is `Ascend950PR_9589` -- the only Ascend variant that supports both SIMT and SIMD modes. Using the wrong SOC version causes silent compilation of kernels for the wrong architecture, which may run but produce incorrect results or suboptimal code. Always verify SOC version with `npu-smi info` on the actual hardware before trusting any documentation.
- **Evidence**: MEMORY.md ("Correct SOC version: Ascend950PR_9589"), docs/design/ASCEND_CHIP_COMPARISON.md

---

## OL-12: CANN typed kernel entry _fp32 crashes with error 507035
- **Category**: platform_bug
- **Loaded by**: Builder
- **Trigger**: when registering kernel entry points for fp32 kernels on CANN 9.0.T501
- **Lesson**: CANN 9.0.T501's auto-generated typed kernel entry points (e.g., `aclrtlaunch_pooling_forward_kernel_fp32`) crash at launch time with error code 507035. The legacy untyped entry points (e.g., `aclrtlaunch_pooling_forward_kernel`) work correctly for the same kernel. The `_fp16` and `_bf16` typed entries are unaffected. This is a CANN runtime bug specific to the `_fp32` suffix. Workaround: use legacy (untyped) entry points for all fp32 kernels. May be fixed in CANN 9.0.0.beta.1.
- **Evidence**: docs/A5_CONTAINER_SETUP.md ("Known Issues"), tests/npu/npu_prod_benchmark.cpp lines 41-42, output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md line 248

---

## OL-13: STOP and search existing skills before inventing workarounds
- **Category**: process
- **Loaded by**: Builder, Optimizer, QA
- **Trigger**: when hitting infrastructure problems (network, auth, deployment)
- **Lesson**: The A5 container has proxy access via `a5_exec.py --proxy` (sources `/home/z00637938/setup_proxy.sh`). The agent spent 30+ minutes on scp/tar/docker cp workarounds instead of checking the `/a5_op` skill documentation it was already using. The proxy flag enables git clone, pip install, and all internet access. Always check existing skill docs FIRST.
- **Evidence**: Session 2026-03-29, SG deployment failures

---

## OL-14: `using namespace AscendC::Simt;` causes GetBlockIdx ambiguity
- **Category**: platform_bug
- **Loaded by**: Builder
- **Trigger**: when writing AscendC SIMT kernel headers
- **Lesson**: CANN has TWO `GetBlockIdx()` — `AscendC::Simt::GetBlockIdx()` (int32_t) and basic_api `GetBlockIdx()` (int64_t). Using `using namespace AscendC::Simt;` brings both into scope → compile error "call to GetBlockIdx is ambiguous". Fix: use ONLY `using namespace AscendC;` (without `::Simt`). Dispatchers already use `Simt::VF_CALL` with qualified prefix.
- **Evidence**: SG forward generated kernel, 20 ambiguity errors. Pooling (only `using namespace AscendC;`) compiled fine.

---

## OL-15: Shared NPU — performance data affected by concurrent users
- **Category**: environment
- **Loaded by**: QA
- **Trigger**: when running performance benchmarks on A5 server
- **Lesson**: A5 server is shared. Our container binds one NPU but other processes may use it. Before performance tests: check `npu-smi info` for other processes. If busy: try another NPU or wait/retry. Never trust a single benchmark run on shared infra.
- **Evidence**: A5 server 90.90.93.35 shared infrastructure

---

## OL-16: int64 fixes cause ~6% performance regression on AscendC SIMT
- **Category**: conditional_insight
- **Loaded by**: Builder, Optimizer
- **Trigger**: when applying int64 preservation fixes to AscendC SIMT kernels
- **Lesson**: Changing inner loop counters from `int` to `int64_t` (e.g., `for(int j)` → `for(int64_t j)`) increases register pressure on Ascend950PR. The AIV general-purpose registers are 32-bit; int64 occupies 2 registers. Measured: Pooling D variant regressed 24.55ms → 25.94ms (~6%). The int64 interface guarantee (CLAUDE.md) is correct and must not be violated, but the regression is expected and must be reported honestly. Do NOT claim "no perf regression" without benchmark verification.
- **Evidence**: E9-1 investigation (2026-03-29), commit 475e83c claimed "no perf regression" without running benchmark

---

## OL-17: bisheng multi-branch Simt::VF_CALL causes 507035 crash
- **Category**: platform_bug
- **Loaded by**: Builder
- **Trigger**: when writing AscendC __global__ dispatcher with multiple Simt::VF_CALL template instantiations
- **Lesson**: Putting multiple `Simt::VF_CALL<template1>(...); Simt::VF_CALL<template2>(...)` branches inside one `extern "C" __global__` function can cause ACL ERR 507035 (kernel launch failure). Root cause: bisheng compiler/linker binary slot corruption — certain kernel slot positions in a large binary produce broken device code. Fix: split each template instantiation into its own `__global__` entry point. Host-side dispatch selects the right entry.
- **Evidence**: Pooling B variant crashed on cluster 2 (dim=1, edges=1024). After splitting into separate entry points, all 61 clusters pass.

---

## OL-20: msprof vec_ratio ≠ atomicAdd 占比（SIMT 标量 GM 读也走 VEC pipe）
- **Category**: profiling_interpretation
- **Loaded by**: Optimizer, Builder
- **Trigger**: msprof shows high vec_ratio in SIMT backward kernel
- **Lesson**: SIMT 模式下 `aiv_vec_ratio=0.99` 被误读为 "99% atomicAdd"。实测去掉 atomicAdd 仅省 4.4% (158us/3605us)。95%+ 的 VEC 时间是标量 GM 随机读（`input[expert*hdim+tid]` 等间接寻址）。NPU 无 L2 cache，每次标量 GM 读都走 HBM，走 VEC pipe。必须对比"有 atomicAdd"和"无 atomicAdd"两个 kernel 才能确认 atomicAdd 实际开销。→ P-P24 Sort-to-Reuse 优化 GM 读放大才是关键。
- **Evidence**: E11 msprof: BwdFull=3605us, grad_weight_only(no atomicAdd)=3447us, diff=158us(4.4%)

## OL-19: GlobalTensor::SetValue() unreliable in SIMD AIV-only mode
- **Category**: platform_bug
- **Loaded by**: Builder, Generator
- **Trigger**: writing scalar values to GM in SIMD (KERNEL_TYPE_AIV_ONLY) kernel
- **Lesson**: `GlobalTensor<float>::SetValue(index, value)` in AIV-only SIMD mode silently drops ~80% of writes. Discovered in E11: SG sorted backward wrote grad_weight[edge_id] via SetValue, 1633/2048 values were zero while the kernel logic was proven correct (grad_in via DataCopy was PASS). Root cause unknown — may be CANN 9.0.T501 or bisheng AIV scalar GM write bug.
- **Workaround**: Accumulate values in LocalTensor (UB), then flush to GM via DataCopy (bulk, 32-byte aligned). For scattered writes, use a contiguous sorted output buffer + SIMT unsort kernel.
- **Evidence**: E11 debugging session, sg_npu_benchmark comparison against GPU reference

## OL-18: Every code change MUST be followed by BOTH precision AND performance verification
- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: after any kernel code modification
- **Lesson**: CLAUDE.md said "every code update must verify precision" but did not mention performance. int64 fixes passed precision (61/61 PASS) but caused 6% performance regression that went undetected for 2 batches. The rule must be: verify precision AND performance after every change. A commit message claiming "no regression" without benchmark data is a lie.
- **Evidence**: E9-1 investigation, REPORT.md data was stale for 2 batches

## OL-21: bf16 scalar cast unsupported in bisheng — use SIMD Cast + GetValue
- **Category**: platform_bug
- **Loaded by**: Builder, Generator
- **Trigger**: bfloat16_t type in kernel code, bf16 GetValue, static_cast<float>(bf16)
- **Lesson**: bisheng compiler (CANN 9.0.0) does not support `static_cast<float>(bfloat16_t)` or reverse. Error: "not support bf16 type cast". Half (fp16) scalar cast works fine. The correct approach: use SIMD `Cast(bf16→float)` vector intrinsic on a buffer, then `GetValue(i)` to read float scalar. See P-P27 pattern.
- **WARNING**: `Cast(bf16→half)` is LOSSY — bf16 exponent=8bit overflows half exponent=5bit → inf for large values. Always Cast bf16→float (lossless), never bf16→half for intermediate conversion.
- **Evidence**: tests/repro/bf16_cast_repro.cpp (6 cases), reg_convert.h API inventory

## OL-22: AscendC documentation is JS-rendered — use dev-browser plugin to access
- **Category**: environment
- **Loaded by**: all skills
- **Trigger**: need to look up AscendC API, SIMD/SIMT programming guide, type support
- **Lesson**: AscendC official docs at hiascend.com are JS SPA — direct URL fetch returns empty or wrong page. Use dev-browser plugin with Playwright. Key documentation URLs:
  - API reference (CANN 9.0 beta2): `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlasascendc_api_07_10293.html`
  - Type conversion API (reg_convert.h): see CANN source `/usr/local/Ascend/cann-9.0.0/x86_64-linux/asc/include/c_api/reg_compute/reg_convert.h`
  - CANN source code: `~/workspace/cann` (local copy)
- **When to check docs**: Before assuming a type cast or API doesn't exist, ALWAYS check reg_convert.h and the CANN source for alternative APIs.

---

## OL-23: Never permanently label a platform feature as "broken" without re-verification
- **Category**: trust_calibration
- **Loaded by**: Builder, all skills
- **Trigger**: when avoiding a platform feature (API, intrinsic, pattern) due to a past bug report
- **Lesson**: In Batch 5 (CANN 9.0.T501), TQue<VECIN,2> caused data corruption (OL-4). We labeled TQue as "broken" and used PipeBarrier<PIPE_ALL> for 6 days. During this time: (1) CANN was updated to 9.0.0 which fixed the bug; (2) The backward kernel successfully used TQue in the SAME session; (3) We still didn't try TQue on forward due to the "broken" label. When finally tested, TQue gave **1.6-2.3x speedup** over PipeBarrier. We left ~60% performance on the table for 6 days due to confirmation bias.
- **Anti-pattern**: "Feature X is broken" → avoid X forever, even when: environment changes (CANN version), other code paths succeed with X, or the expert's code uses X.
- **Correct pattern**: When a feature was previously broken:
  1. Check if the environment changed (CANN version, compiler version)
  2. Check if other code paths in the SAME codebase use the feature successfully
  3. If either is true, **re-test the feature** before assuming it's still broken
  4. Platform bugs are version-specific — always record the CANN version with the bug report
- **Evidence**: E13-P1 benchmark (2026-04-01): TQue 1.6-2.3x faster than PipeBarrier across all cases

## OL-24: 优化一个 kernel 后必须审查所有同类 kernel
- **Category**: process
- **Loaded by**: Builder, Optimizer, Lead
- **Trigger**: any optimization applied to one kernel variant (forward/backward, fp32/fp16/bf16)
- **Lesson**: E13 将 Forward PipeBarrier→TQue 带来 1.6-2.3x 加速，但没有检查 Backward Sorted 是否有同样的 PipeBarrier 反模式（有，7 个 PIPE_ALL）。直到 E14 专家指出才发现。
  优化应用后必须：
  1. grep 全部 kernel 文件查找同类反模式
  2. 对每个匹配项评估是否适用同样的优化
  3. 区分"生产路径"和"非生产路径"——只优化生产路径
- **Evidence**: E14 feedback (2026-04-06), sparse_gather_simd.h BackwardSimdSortedF32 有 7 个 PIPE_ALL 未优化

---

## OL-25: TBuf has NO automatic synchronization — must use TQue or explicit sync
- **Category**: platform_bug (actually: design feature)
- **Loaded by**: Builder, Optimizer
- **Trigger**: mixing TBuf with TQue in the same kernel, or using TBuf for accumulators alongside DMA operations
- **Lesson**: E14 TQue refactoring of backward sorted kernel FAILED (max_diff=0.76) because accum was in TBuf while input was in TQue. Root cause: TBuf.Get() has **no EnQue/DeQue → no hardware set/wait signals**. When VEC writes to TBuf accum and MTE2 writes to TQue input concurrently, there's no sync → UB bus contention → data corruption.
  Fix: move accum to TQue<VECOUT> (same pattern as working forward PingPong). Official docs confirm: "TBuf申请的内存空间只能参与计算，无法执行队列的入队出队操作" and "EnQue调用会发射同步指令set" — TBuf simply has no sync mechanism.
- **Rule**: When using TQue for DMA, ALL VEC-accessed buffers must EITHER be TQue-managed OR have explicit SyncFunc/SetFlag/WaitFlag. TBuf + TQue without explicit sync = data corruption.
- **Evidence**: E14 revert (2026-04-06), CANN 9.0.0-beta.2 official docs "编程模型设计原理", ASCENDC_LANGUAGE_REFERENCE.md

---

## OL-26: Research official docs + CANN source before optimization attempts
- **Category**: process
- **Loaded by**: Builder, Optimizer, Researcher
- **Trigger**: before any non-trivial optimization or architecture change
- **Lesson**: E14 first attempt was a guess (TQue input + TBuf accum). If we had first verified TBuf sync semantics from official docs, we would have known it can't work. The correct pattern (accum in TQue<VECOUT>) was already in the forward code and confirmed by CANN MoE source.
  Research workflow:
  1. Check official docs (dev-browser for JS-rendered hiascend.com)
  2. Check CANN source code (~/workspace/cann/, git fetch first)
  3. Record findings as patterns/OL entries for future use
  4. THEN implement
- **Evidence**: E14 session (2026-04-06 → 2026-04-07), ASCENDC_LANGUAGE_REFERENCE.md creation

---

## OL-27: 性能声明必须基于同条件 A/B 数据 — 绝不凭推测
- **Category**: process
- **Loaded by**: Builder, QA, Optimizer, Lead
- **Trigger**: 任何性能相关的声明（"无退步"、"提升 X%"、"性能不变"）
- **Lesson**: E14 TQue 重构后，CC 声称"性能无退步"。实际上：(1) before/after 跑在不同 NPU 上（NPU 1 vs NPU 0），数据不可比；(2) F16/BF16 non-PingPong kernel 根本没有性能数据；(3) 这个结论被写入了 EXPERT_FEEDBACK.md 和 session summary 作为"事实"发布。如果没有用户指正，这个虚假结论会成为后续决策的依据。
  **严重性**: 这不是"差一点"的问题——这是**完全没有证据的结论被当作事实发布**。在生产环境中，这会导致：
  - 性能退步被掩盖，直到用户在线上发现
  - 后续优化基于错误的 baseline 做决策
  - 团队对 AI 辅助工具的信任崩塌
  **硬性规则**:
  1. 性能声明必须附带**同一 NPU、同一 session、背靠背**的 A/B 数据
  2. 每个被修改的 kernel 都必须有对应的性能数据行
  3. 如果 benchmark 不覆盖某 kernel，必须标注 **"性能未验证"**
  4. **绝不能用"应该没问题"、"趋势一致"、"数值接近"来替代 A/B 对比**
  5. 跨 NPU / 跨 session 的数据只能标注为"参考"，不能用于性能声明
- **Evidence**: E14 session (2026-04-07)，用户三次指正

---

## OL-28: PyTorch 是 spec，CUDA 只是一个可能有 bug 的实现
- **Category**: process, trust_calibration
- **Loaded by**: Analyzer, Builder, QA, Lead
- **Trigger**: 任何 CUDA → AscendC 迁移任务
- **Lesson**: MXFP4 算子迁移中，AscendC 实现与 CUDA bit-exact match（7/7 PASS），但 PyTorch spec 与 CUDA 有 7-10% 元素不一致（舍入行为、shared exponent 计算方法不同）。如果盲目对齐 CUDA，AscendC 会继承 CUDA 的 bug，导致 NPU 训练结果与 PyTorch 定义不一致。
  **验证链应该是**:
  ```
  PyTorch 定义 (spec, 数学规范)
      ↓ 一致性验证 (PyTorch vs CUDA)
  CUDA 实现 (可能有 bug)
      ↓ 移植
  AscendC 实现
      ↓ 验证
  与 PyTorch spec bit-exact 或 within tolerance
  ```
  **硬性规则**:
  1. 迁移前必须验证 PyTorch vs CUDA 一致性（在 A100 上跑两者对比）
  2. 如果 PyTorch ≠ CUDA → AscendC 应对齐 PyTorch，不是 CUDA
  3. CUDA 只在 PyTorch ≡ CUDA 时才能作为 ground truth
  4. 记录任何 PyTorch vs CUDA 的差异（edge case 分析）
  5. Skill 输入将来可能是 PyTorch 算子（不一定有 CUDA），必须能直接从 PyTorch 生成 AscendC
- **Evidence**: MXFP4 session (2026-04-07), PyTorch vs CUDA 7-10% mismatch on normal data

---

## OL-30: SIMD 性能优化不能以精度降级为代价
- **Category**: process, trust_calibration
- **Loaded by**: Builder, Optimizer, QA
- **Trigger**: SIMD kernel 优化尝试，特别是消除 per-group/per-block 循环
- **Lesson**: MXFP4 SIMD V4 "fast" 用 tile-wide shared exponent（1024 元素共享）代替 per-32-group exponent。
  性能确实提升（小 tensor 比 SIMT 快 1.08x），但 **精度不符合 MXFP4 spec**。
  A3 手写 SIMD 有同样问题——BATCH=512 共享一个 exponent，正是其精度 bug 的根源。
  **绝不能发布一个"更快但精度错误"的 kernel 作为 production**。
  **硬性规则**:
  1. 任何性能优化后必须对比 PyTorch spec 精度（不是只对比上一版 AscendC）
  2. 如果优化要求改变算法精度语义（如增大 group_size），必须在文档中 **明确标注为"approximate"**
  3. "精度 0 mismatch"只有在对比 PyTorch spec 时才有意义。对比自己的 CPU ref 不算（自己的 ref 可能有同样的 bug）
  4. A3 实现的精度 bug 正是因为为了 SIMD 性能放弃了 per-group 精度——这不是可以接受的 trade-off
- **Evidence**: MXFP4 SIMD V4 (2026-04-07), A3 手写 SIMD 精度 bug 分析

---

## OL-29: Edge case 发现方法论（量化算子）
- **Category**: process, algorithm_selection
- **Loaded by**: Analyzer, QA
- **Trigger**: 任何涉及位操作、量化、定点数的算子
- **Lesson**: MXFP4 迁移中发现了多类 edge case：
  1. **Shared exponent 极端值差异**: 同一 group 中有 1e+38 和 1.0，导致 1.0 underflow 到 0（正确行为但需验证）
  2. **整数溢出**: `1 << expdiff` 当 expdiff > 30 时 C++ UB（CUDA 产生 0，CPU 产生 inf）
  3. **舍入边界**: round-to-nearest-even 在 0.5 处的行为因实现不同（CUDA 位操作 vs PyTorch floor+0.5）
  
  **Edge case 发现方法**:
  - 极端值组合: 在同一 group/block 中混合极值和正常值
  - 整数溢出: 找所有 `<<` 和 `>>` 操作，检查 shift amount 上界
  - 舍入边界: 构造恰好在量化边界上的值（如 MXFP4 的 1.5 × 2^exp 边界）
  - 零和 subnormal: 0.0, -0.0, 最小正 subnormal
  - 饱和: max representable value + 1 ULP
  - 三路对比: PyTorch (spec) vs CUDA vs AscendC，任何一对不一致都是 bug
- **Evidence**: MXFP4 session (2026-04-07)

---

## OL-31: 性能评测必须使用目标 benchmark 框架的标准工具
- **Category**: process, measurement
- **Loaded by**: QA, Builder, Lead
- **Trigger**: 任何性能报告、benchmark 结果发布
- **Lesson**: GELU 评测中，用 C++ aclrtlaunch 裸调用 vs Python torch_npu 调用对比，得出"CANN 快 2-4x"的错误结论。切换到 NPUKernelBench 框架的标准工具（utils/performance.py）后，结果变为 0.83-1.11x（全部 ≥0.8x PASS）。
  **根因**: 裸调用没有 Python dispatch overhead（~10us），而 torch_npu 调用包含。两者计时基准不同，不可比。
  **硬性规则**:
  1. 使用目标 benchmark 框架提供的标准评测工具，不要自己写计时代码
  2. 确保 reference 和 candidate 经过相同的调用路径（同样的 Python overhead）
  3. 如果自己写了计时工具，必须说明与标准工具的差异
  4. 集成方式: kernel(.cpp/.h) + pybind11.cpp + model_new_ascendc.py(ModelNew)
  5. Build: `utils/build_ascendc.py`, Verify: `utils/verification_ascendc.py`, Perf: `utils/performance.py`
- **Evidence**: GELU evaluation (2026-04-08), NPUKernelBench framework analysis

---

## OL-32: Cumsum/Scan is inherently sequential — SIMD for I/O only
- **Category**: algorithm_selection
- **Loaded by**: Analyzer, Builder
- **Trigger**: when implementing prefix sum / cumulative ops
- **Lesson**: Cumsum cannot be vectorized within a single scan line. Use SIMD DataCopyPad for bulk I/O, scalar GetValue/SetValue loop for the actual prefix sum. For large scan_len (>4K), perf will be poor (~0.02x for 16K elements) because the serial loop dominates.
- **Evidence**: Cumsum V1 (2026-04-09), 51/51 precision PASS, 0.49x mean

---

## OL-33: Histc bin assignment needs double precision — A5 has no double
- **Category**: platform_bug
- **Loaded by**: Analyzer, Builder
- **Trigger**: when implementing histogram / binning operations
- **Lesson**: torch.histc uses double precision internally for bin boundary computation. A5 AIV cores only support float32/fp16/bf16. Float kernel gets 1-2 bins wrong at boundaries due to rounding. CPU double fallback is NOT acceptable — need float kernel with adjusted bin formula.
- **Evidence**: Histc V1 (2026-04-09), float kernel had 2% mismatch, CPU fallback got 100% match but is cheating

---

## OL-34: Sort insertion sort is O(N²) — too slow for sort_len > 4K
- **Category**: algorithm_selection
- **Loaded by**: Analyzer, Builder
- **Trigger**: when implementing sort / topk operations
- **Lesson**: Insertion sort via scalar GetValue/SetValue is correct but O(N²) per line. For sort_len=16384, that's 268M comparisons per line — timeout on NPU. Need radix sort (O(N*k)) or bitonic sort (O(N*log²N)) for large cases.
- **Evidence**: Sort V1 (2026-04-09), fp32 small cases pass but large cases timeout

---

## OL-35: PipeBarrier only supports MTE2/V/MTE3 — not PIPE_S
- **Category**: platform_bug
- **Loaded by**: Builder
- **Trigger**: when using PipeBarrier in SIMD kernels with scalar operations
- **Lesson**: On Ascend950PR, pipe_barrier() accepts values [4,6] = PIPE_MTE2, PIPE_V, PIPE_MTE3 only. PIPE_S is NOT valid. For scalar pipe synchronization, use SetFlag/WaitFlag with S_MTE3, S_V, MTE2_S, V_S event types.
- **Evidence**: Sort V1 compile error (2026-04-09), EC-15

---

## OL-36: NPUKernelBench — PyTorch wrapper is CANN delegation (PROHIBITED)
- **Category**: trust_calibration
- **Loaded by**: Builder, QA
- **Trigger**: when considering "simple" implementation via PyTorch ops
- **Lesson**: Implementing model_new_ascendc.py by calling PyTorch ops (torch.permute, torch.sort, F.layer_norm, etc.) delegates to CANN via torch_npu. This is the SAME as calling aclnn* APIs inside a kernel — it's wrapper hacking. ALL computation must use AscendC primitives (DataCopy, VEC ops, TQue/TBuf, scalar GetValue/SetValue).
- **Evidence**: Permute (#12) was caught as CANN delegation and reverted (2026-04-09)

---

## OL-37: Data-movement ops (cat/split/pad) — use VECIN→VECOUT bridge pattern
- **Category**: algorithm_selection
- **Loaded by**: Builder
- **Trigger**: when implementing cat, split, pad, repeat, permute, or any pure copy op
- **Lesson**: Pure data-movement kernels need a VEC op between VECIN (GM→UB) and VECOUT (UB→GM) to maintain pipeline sync. Use `Adds(dst, src, 0.0f, count)` as a no-op bridge. For fp16/bf16, bridge through fp32 via Cast→Adds→Cast (lossless roundtrip). Direct UB-to-UB DataCopy is prohibited (PB-9).
- **Evidence**: Cat V2 kernel (2026-04-09), P-CAT-1 pattern

---

## OL-38: N-dim concat decomposes to 3D (outer × cat_dim × inner)
- **Category**: algorithm_selection
- **Loaded by**: Builder
- **Trigger**: when implementing torch.cat along arbitrary dim
- **Lesson**: Decompose `cat(tensors, dim=d)` into flat 3D: outer=prod(shape[:d]), cat_dim=shape[d], inner=prod(shape[d+1:]). If outer==1, flat contiguous copy. If outer>1, per-outer chunked copy with stride. Launch one kernel per input tensor. Non-aligned chunks need overlapping tail write (EC-16).
- **Evidence**: Cat V2 kernel (2026-04-09), applicable to Split/Permute

---

## OL-39: View-returning ops have ~0 reference cost — benchmark vs materialized copy
- **Category**: measurement
- **Loaded by**: QA
- **Trigger**: when benchmarking split, narrow, slice, or any op that returns views
- **Lesson**: torch.split/narrow/slice return tensor views (~0.009ms), not copies. Our AscendC kernel does actual data movement. Direct latency comparison is misleading — our kernel is "slower" but does real work. For fair evaluation, compare against the cost that would force materialization (.contiguous() on the view).
- **Evidence**: Split 14 benchmark (2026-04-09): torch.split ~0.009ms vs AscendC 0.018-0.96ms

## OL-40: SIMT gather achieves ~1.0x vs CANN for random-access patterns
- **Category**: algorithm_selection
- **Loaded by**: Generator, Analyzer
- **Trigger**: torch.gather, index_select, or any per-element indirect addressing op
- **Lesson**: For torch.gather, SIMT per-element with 512 threads × 56 blocks achieves mean=0.86x, median=0.89x vs CANN. SIMT beats CANN on small tensors (~1.2x) and large fp32 (~1.1x), but is slower on large fp16/bf16 with dim=last (0.16-0.4x). The key insight: SIMT uses dcache for random reads instead of per-element DMA (which V1 SIMD approach used at 0.006x). The remaining gap (esp. fp16 slow cases) is an optimization knowledge gap, not a hardware limitation — CANN uses the same AscendC API (see OL-42).
- **Evidence**: Gather V2 (2026-04-10): SIMD 0.006x → SIMT 0.86x mean (158x improvement)

## OL-41: int64 index tensor in SIMT — direct __gm__ read works
- **Category**: platform_bug
- **Loaded by**: Generator
- **Trigger**: kernels reading int64 values from GM in SIMT mode
- **Lesson**: Reading int64_t from GM via `__gm__ int64_t* idx; int64_t val = idx[i];` works correctly in SIMT. No special handling needed for 8-byte types. This avoids the int64→int32 conversion overhead in pybind11.
- **Evidence**: Gather V2 (2026-04-10): 47/47 PASS with direct int64 index reads

## OL-42: CANN 性能差距是知识差距，不是硬件限制
- **Category**: trust_calibration
- **Loaded by**: Analyzer, Generator
- **Trigger**: 当分析性能差距原因时
- **Lesson**: CANN 使用的硬件能力和 AscendC API 完全一样。如果我们的算子 <1.0x，原因是缺少优化知识（如 DMA 批量化、cache line 合并、混合 SIMT+SIMD 策略），而不是 CANN 有"秘密硬件接口"。CANN 源码（~/workspace/cann/）包含这些优化技巧，在 op-gen 模式下可以学习。性能差距 = 可缩小的知识差距。
- **Evidence**: Gather fp16 dim=last 0.16x 不是平台限制 — CANN 在同一硬件上做到 1.0x，说明 AscendC API 完全能支持

## OL-43: nblk=1 测试是多核竞争的终极诊断工具
- **Category**: measurement
- **Loaded by**: QA, Generator
- **Trigger**: 精度测试在 nblk>1 时失败
- **Lesson**: 当精度在 nblk=56 失败但 nblk=1 通过时，根因一定是多核并行问题（DataCopy 对齐溢出 EC-22、buffer 越界、写写竞争）。这是一个强力诊断工具：修改 pybind11.cpp 中 nblk 为 1 即可确认。
- **Evidence**: Pad V5 (2026-04-10): nblk=1 → 51/51, nblk=56 → 28/51

## OL-44: Cast(bf16→f32→bf16) 往返不一定位精确
- **Category**: platform_bug
- **Loaded by**: Generator
- **Trigger**: bf16 数据搬运 kernel 出现精度不匹配
- **Lesson**: Cast(bf16→f32, CAST_NONE) 无损（bf16 是 f32 子集），但 Cast(f32→bf16, CAST_ROUND) 可能因中间 VEC 操作（如 Adds bridge）改变 f32 位模式，导致往返后 bf16 值不同。纯拷贝 kernel 中，应用 CAST_NONE 做反向转换（截断恢复原始 bf16 位），或直接 SIMT 赋值绕过 Cast。
- **Evidence**: Pad V4 (2026-04-10): bf16 Cast roundtrip regression

## OL-45: SIMT 可作为 EC-22 (DataCopy 对齐溢出) 的通用解法
- **Category**: algorithm_selection
- **Loaded by**: Generator, Analyzer
- **Trigger**: 多核 SIMD kernel 出现 DataCopy 对齐溢出导致的精度问题（EC-22）
- **Lesson**: 当 SIMD 的 DataCopy 对齐溢出难以修复（如多种 tile 类型 × 多种 dtype × 行尾跨 block 溢出），可以切换到 SIMT 作为正确性优先的方案。SIMT 每线程写一个元素，完全避免 DataCopy 和对齐问题。性能代价：大 tensor 较慢（SIMT 0.04-0.05x vs SIMD potentially >1.0x），但小/中 tensor 和非 constant mode 可以更快（SIMT 1.2-2.8x）。总体 mean 从 0.05x 提升到 0.72x。
- **Evidence**: Pad V4 (2026-04-10): SIMD V3 28/51 PASS → SIMT V4 51/51 PASS, mean 0.05x → 0.72x

## OL-46: 单次 DataCopy 搬运 ≥16KB 才能发挥最佳带宽
- **Category**: measurement
- **Loaded by**: Generator
- **Trigger**: SIMD kernel 中 DataCopy 的 tile 大小选择
- **Lesson**: 单次搬运数据量 <16KB 时带宽利用率显著下降。应尽可能增大 TILE_SIZE 使每次 DataCopy 搬运 ≥16KB。对于 fp32 TILE=4096（16KB），fp16/bf16 TILE=8192（16KB）是合理下限。
- **Source**: hiascend.com best practices (2026-04)

## OL-47: GM 地址 512B 对齐可提升 30% 搬运带宽
- **Category**: measurement
- **Loaded by**: Generator
- **Trigger**: 输出 tensor 分配或 DataCopy 的 GM 偏移计算
- **Lesson**: GM→UB 搬运时，512B 对齐的 GM 地址比 32B 对齐带宽高 30%。在 pybind 层分配输出 tensor 时，确保起始地址 512B 对齐（通过分配额外 padding 再 narrow）。注意：此优化在 A2 系列产品上效果最显著。
- **Source**: hiascend.com best practices (2026-04)

## OL-48: DataCopyParams 替代 for 循环实现非连续搬运
- **Category**: algorithm_selection
- **Loaded by**: Generator
- **Trigger**: 需要搬运非连续内存（如矩阵的列、间隔数据）
- **Lesson**: 使用 DataCopy 的 DataCopyParams（srcStride/dstStride/blockLen/blockCount）替代 for 循环逐块搬运，效率差距极大。例如搬运图片每行前 2KB：for 循环需要逐行搬运 N 次，DataCopyParams 一次完成。
- **Source**: hiascend.com best practices (2026-04)

## OL-49: 纯搬运算子用 TQueBind 替代 VECIN→VECOUT bridge
- **Category**: algorithm_selection
- **Loaded by**: Generator
- **Trigger**: 数据搬运类算子（Cat/Split/Pad/Repeat 等不涉及实际 VEC 计算的场景）
- **Lesson**: 纯搬运类算子使用 TQueBind 接口将 VECIN 与 VECOUT 绑定，省略 Adds(0.0f) bridge 步骤。当前我们的 Cat/Split/Pad 都用 Adds bridge（EC-21 workaround），TQueBind 是更高效的替代。需要验证 TQueBind 在 A5 上是否可用。
- **Source**: hiascend.com best practices (2026-04)
- **Status**: UNVERIFIED on A5 — 需要实验确认 TQueBind API 在 CANN 9.0.0 可用

## OL-50: 避免 UB bank 冲突 — 连续 VEC 指令操作不同 bank 的地址
- **Category**: platform_bug
- **Loaded by**: Generator
- **Trigger**: SIMD kernel 性能低于预期且 msprof 显示 VEC pipe 利用率低
- **Lesson**: UB 由多个 bank 组成。当多条 VEC 指令同时读写同一 bank 的不同地址时，产生 bank 冲突，指令需排队。解决方法：确保连续 VEC 操作的源和目标 tensor 在不同 bank（通过调整 buffer 分配偏移）。UB bank 数和大小因芯片版本而异。
- **Source**: hiascend.com best practices (2026-04)

## OL-51: Vector Counter 模式替代 Normal 模式简化尾块处理
- **Category**: algorithm_selection
- **Loaded by**: Generator
- **Trigger**: VEC 操作需要处理非对齐尾块（当前用 Align8/Align16 + 显式 mask 管理）
- **Lesson**: Counter 模式允许直接指定总元素个数，硬件自动处理主块和尾块的 mask/迭代。避免开发者手动计算 repeatTimes、设置 mask、处理尾块。代码更简洁且不易出错。当前我们的 kernel 全部使用 Normal 模式，Counter 模式可能简化非对齐场景。
- **Source**: hiascend.com best practices (2026-04)

## OL-53: Multi-pass kernels — HBM bandwidth ceiling vs fused CANN
- **Category**: performance_analysis
- **Loaded by**: Optimizer, Researcher
- **Trigger**: AscendC kernel performance <0.5x of CANN reference AND kernel requires multiple passes over input data
- **Lesson**: A multi-pass kernel reads each element from HBM N times (N=number of passes). A fused CANN kernel reads once. The performance ceiling of an N-pass kernel is ~1/N of fused, regardless of VEC optimization. For per-row reductions that need a global statistic before element-wise processing (e.g., quantization needs max before scaling), 2-pass is mandatory unless rows fit in single UB tile. When <0.5x vs CANN with a multi-pass kernel, check if the algorithm requires multi-pass before investing in VEC tuning — the bottleneck is HBM bandwidth, not compute.
- **Evidence**: DynamicQuant 2-pass (find max, then quantize) → 0.25x of CANN's fused npu_dynamic_quant

## OL-52: 归约操作选择低延迟指令 — WholeReduceMax vs 二分累加
- **Category**: algorithm_selection
- **Loaded by**: Generator
- **Trigger**: 需要实现 ReduceSum/ReduceMax/ReduceMin 操作
- **Lesson**: 不同归约方案性能差异大。官方建议根据数据规模选择：小数据量用 WholeReduceMax（单指令，低延迟），大数据量用二分累加（BinaryFold 模式）。当前我们的 BinaryFoldReduceMax 已经是合理方案，但可能在小数据量场景下可以简化。
- **Source**: hiascend.com best practices (2026-04)
