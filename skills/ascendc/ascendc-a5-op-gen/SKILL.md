---
name: ascendc-a5-op-gen
description: >
  AscendC operator generation — three modes:
  Benchmark: /ascendc-a5-op-gen 13_Cat (NPUKernelBench — auto-detects N_Name pattern)
  Op-gen: /ascendc-a5-op-gen path/to/source.py (general kernel generation from any source)
  Optimize: /ascendc-a5-op-gen --optimize 12_Permute (optimize existing kernel for performance)
argument-hint: >
  Benchmark mode: "13_Cat", "5_Cumsum" (matches N_Name pattern → NPUKernelBench)
  Op-gen mode: "path/to/source.py" or "path/to/source.cu" (file path → general op gen)
  Optimize mode: "--optimize 12_Permute" (optimize existing archived kernel)
  Explicit: "--benchmark 13_Cat" or "--opgen path/to/source.py" or "--optimize 12_Permute"
context: inline
---

# A5 AscendC Op Gen — Orchestrator

You are the orchestrator. Do NOT generate kernel code yourself.

## File Access Boundary

You may read ONLY:
- `benchmarks/NPUKernelBench/level1/{PROBLEM_ID}.py` or the provided `SOURCE_PATH`
- `{output_dir}/**`
- `agents/ascend-a5-kernel-developer.md`
- `skills/ascendc/ascendc-a5-*/**`
- `skills/ascendc/a5-shared-references/**`
- `skills/ascendc/a5-common-scripts/**`
- `utils/build_ascendc.py`
- `utils/verification_ascendc.py`
- `utils/performance.py`

You MUST NOT read:
- `archive_tasks/**`
- `skills/ascendc/ascendc-translator/**`
- `skills/ascendc/tilelang-designer/**`
- `skills/ascendc/performance-analyzer/**`
- `skills/ascendc/trace-recorder/**`
- Any non-A5 skill directories

## Step 0: Detect Mode and Flags

Parse flags first, then determine mode:
```
flags:
  --diag   → DIAGNOSTIC = true  (worker writes extra KB/decision/token info to PROGRESS.md)
  default  → DIAGNOSTIC = false (no extra diagnostic output)

Strip flags from arg, then detect mode from remaining:

if arg starts with "--optimize":
    MODE = "optimize"
    PROBLEM_ID = remaining arg (e.g., "12_Permute")
elif arg starts with "--benchmark":
    MODE = "benchmark"
    PROBLEM_ID = remaining arg (e.g., "13_Cat")
elif arg starts with "--opgen":
    MODE = "opgen"
    SOURCE_PATH = remaining arg
elif arg matches pattern /^\d+_\w+$/ (e.g., "13_Cat", "5_Cumsum"):
    MODE = "benchmark"
    PROBLEM_ID = arg
elif arg contains "/" or ends with ".py"/".cu"/".h":
    MODE = "opgen"
    SOURCE_PATH = arg
else:
    Print: "Cannot determine mode. Use:"
    Print: "  /ascendc-a5-op-gen 13_Cat              (benchmark — generate from scratch)"
    Print: "  /ascendc-a5-op-gen path/to/source.py   (op-gen — from source file)"
    Print: "  /ascendc-a5-op-gen --optimize 12_Permute (optimize existing kernel)"
    Print: "  Add --diag for diagnostic output in PROGRESS.md"
    STOP
```

Print: "Mode: {MODE}" + (if DIAGNOSTIC: " [DIAG]") so user knows which path is taken.

## Step 1: Environment Check

Default local-A5 configuration:
- `SOC_VERSION=Ascend950PR_9589`
- `CANN_PATH=/usr/local/Ascend/cann-9.0.T501/`

If local helpers (`utils/build_ascendc.py`, `utils/verification_ascendc.py`, `utils/performance.py`) are missing or broken, report the local dependency issue and stop.

**Optimize mode**: verify existing kernel exists:
- Kernel dir must exist: `output/npukernelbench/src/kernels/{PROBLEM_ID}/`
- Must contain: `kernel/{lower}_kernel.h`, `kernel/pybind11.cpp`, `model.py`, `model_new_ascendc.py`
- Read current RESULTS.md entry for baseline performance
- If missing: "No archived kernel found for {PROBLEM_ID}. Run benchmark mode first."

**Benchmark mode only**: verify source exists:
- `benchmarks/NPUKernelBench/level1/{PROBLEM_ID}.py`

If missing: tell user to provide a valid problem ID.

## Step 1.5: Compute Output Directory

Determine and create the output directory BEFORE spawning the worker:

```
if MODE == "benchmark":
    # e.g., "13_Cat" → output dir under current working directory
    OUTPUT_DIR = "output/npukernelbench/{PROBLEM_ID}"
elif MODE == "opgen":
    # e.g., "path/to/gelu.py" → extract op name
    OP_NAME = basename of SOURCE_PATH without extension, lowercased
    OUTPUT_DIR = "output/opgen/{OP_NAME}"
```

Create the directory structure:
```bash
mkdir -p {OUTPUT_DIR}/kernel
```

Verify the directory was created successfully. If `mkdir` fails, STOP and report the error.

Set `PROGRESS_FILE = {OUTPUT_DIR}/PROGRESS.md`.

Print: "Output directory: {OUTPUT_DIR}" so user knows where files will be written.

## Step 2: Spawn Worker Agent

Fill the appropriate prompt template based on mode, then spawn:

```
Agent(
    name: "{N_Name}-worker"
    subagent_type: "ascend-a5-kernel-developer"
    run_in_background: false
    prompt: <filled template — see below>
)
```

Worker must write progress to `{output_dir}/PROGRESS.md` after every major step.

## Step 3: Monitor Progress

Poll `{output_dir}/PROGRESS.md` every **60 seconds** (not 30s — saves ~6K tokens).
Only print status to user when stage changes (Phase 0→Stage 1→Stage 2→Finalize).
Do NOT print unchanged status — it wastes orchestrator context tokens.

## Step 4: Independent Verification

When worker completes:

### 4a. Anti-Hack Checks (ALL must pass, both modes)
1. Static check: `python3 skills/ascendc/a5-common-scripts/ascendc_static_check.py {output_dir}/kernel/`
2. CANN source check: `grep -riE "workspace/cann|gitee.com/ascend|github.com/Ascend" {output_dir}/kernel/`
3. PyTorch wrapper: `grep -c "torch\.\|F\.\|torch_npu" {output_dir}/kernel/{op}_kernel.h` (must be 0)
4. CANN API wrapper: `grep -ci "aclnn\|aclop\|acl_op" {output_dir}/kernel/` (must be 0)
5. CPU fallback: `grep -ci "\.cpu()\|to(at::kCPU)" {output_dir}/kernel/pybind11.cpp` (must be 0)
6. Genuine AscendC: `grep -c "DataCopy\|TQue\|TBuf" {output_dir}/kernel/{op}_kernel.h` (must be >0)

### 4b. Benchmark mode — Performance Gate (MANDATORY)
- Re-run `python3 utils/performance.py {output_dir} all` independently (NEVER trust worker-reported numbers)
- Compute ratios yourself from raw median data
- **HARD GATE**: if mean ratio < 0.6x AND worker did NOT attempt optimization:
  → REJECT result. Tell user: "Perf {ratio}x below 0.6x threshold. Worker skipped optimization."
  → Re-run worker with explicit optimization instruction, or flag for manual review
- **"N/A" is NEVER acceptable** — the benchmark framework always produces numbers. If worker says N/A, that's a bug.
- Check model_new doesn't call torch ops

### 4c. Op-gen mode only
- Print kernel files location
- Print build/test commands for user to integrate into their project

### 4d. Result Summary
Summarize to user:
- Output directory
- Key files created
- Independent verification results
- Performance ratio (benchmark mode)
- Whether it passed, needs optimization, or failed

---

## BENCHMARK MODE — Worker Prompt Template

Variables: `{PROBLEM_ID}`, `{OUTPUT_DIR}`, `{PROGRESS_FILE}` = `{OUTPUT_DIR}/PROGRESS.md`

```
You are an A5 AscendC kernel worker. Generate, build, test, and archive kernel for NPUKernelBench problem {PROBLEM_ID}.

## Local A5 Environment
- Source: benchmarks/NPUKernelBench/level1/{PROBLEM_ID}.py
- Output dir: {OUTPUT_DIR}
- Progress file: {PROGRESS_FILE}
- SOC_VERSION: Ascend950PR_9589
- CANN_PATH: /usr/local/Ascend/cann-9.0.T501/
- Build: python3 utils/build_ascendc.py {OUTPUT_DIR} -v Ascend950PR_9589 --build-type Release
- Verify: python3 utils/verification_ascendc.py {OUTPUT_DIR}
- Perf: python3 utils/performance.py {OUTPUT_DIR} all
- Static check: python3 skills/ascendc/a5-common-scripts/ascendc_static_check.py {OUTPUT_DIR}/kernel/

## Progress Reporting (MANDATORY)
Write {PROGRESS_FILE} after EVERY step.
After EVERY performance test, update the Optimization section:
```
Optimization:
  baseline: {X.XXx} (first perf measurement)
  current:  {X.XXx} (latest measurement)
  speedup:  {current/baseline}x
  history:
  - V1: {X.XXx} — initial implementation
  - Opt1: {X.XXx} — {what changed}
```

## Workflow (Benchmark)
1. Read source .py, analyze op. Then **read KB to make informed decisions**:
   - Read `skills/ascendc/a5-shared-references/KB_INDEX.md` — this is the knowledge base index
   - Read `skills/ascendc/a5-shared-references/SIMT_VS_SIMD_DECISION.md` — make SIMT/SIMD decision using the decision tree
   - Read `skills/ascendc/a5-shared-references/PLATFORM_BUGS.md` — avoid known pitfalls
   - Based on your SIMT/SIMD decision, read the relevant reference files listed in KB_INDEX
   - Document the decision AND reasoning in {PROGRESS_FILE}
2. Write 5 files to {OUTPUT_DIR}/:
   - kernel/{lower}_kernel.h
   - {lower}_kernels.cpp
   - pybind11.cpp
   - model.py (verbatim copy)
   - model_new_ascendc.py
3. Run static check + build (max 5 compile-fix iterations)
4. Precision test — ALL cases PASS (max 3 fix iterations)
5. Performance test — compute actual ratios (NEVER report "N/A")
6. **OPTIMIZATION GATE**: if mean ratio < 0.6x → enter Stage 2:
   - Identify bottleneck (bandwidth? compute? Python overhead? 3-pass vs 1-pass?)
   - Apply optimization (reduce passes, improve tiling, eliminate overhead)
   - Re-verify precision after each optimization
   - Max 3 optimization iterations
   - If still < 0.6x after optimization: report honestly with analysis of why
7. Write final status into {PROGRESS_FILE}.

## Hard Exit Rules (MANDATORY — saves tokens)
- Compile fix: max 5 attempts → if still failing, write FAIL to {PROGRESS_FILE} and STOP
- Precision fix: max 3 attempts → if still failing, write FAIL to {PROGRESS_FILE} and STOP
- Optimization: max 3 iterations → if still < 0.6x, write current best perf to {PROGRESS_FILE} and STOP
- NEVER continue retrying beyond these limits

## Anti-Hack & KB Rules
(Detailed rules in ascend-a5-kernel-developer agent definition — do not duplicate here)
- Anti-hack checks enforced by orchestrator Step 4a post-verification
- KB loading instructions in ascend-a5-kernel-developer Phase 0
```

---

## OP-GEN MODE — Worker Prompt Template

Variables: `{SOURCE_PATH}`, `{OUTPUT_DIR}`, `{PROGRESS_FILE}` = `{OUTPUT_DIR}/PROGRESS.md`, `{OP_NAME}`

```
You are an A5 AscendC kernel worker. Generate an AscendC kernel from the given source.

## Local A5 Environment
- Source: {SOURCE_PATH}
- Output dir: {OUTPUT_DIR}
- Progress file: {PROGRESS_FILE}
- SOC_VERSION: Ascend950PR_9589
- CANN_PATH: /usr/local/Ascend/cann-9.0.T501/
- Build: python3 utils/build_ascendc.py {OUTPUT_DIR} -v Ascend950PR_9589 --build-type Release
- Verify: python3 utils/verification_ascendc.py {OUTPUT_DIR}
- Perf: python3 utils/performance.py {OUTPUT_DIR} all
- Static check: python3 skills/ascendc/a5-common-scripts/ascendc_static_check.py {OUTPUT_DIR}/kernel/

## File Access Boundary
You may read ONLY:
- {SOURCE_PATH}
- {OUTPUT_DIR}/**
- skills/ascendc/ascendc-a5-*/**
- skills/ascendc/a5-shared-references/**
- skills/ascendc/a5-common-scripts/**
- utils/build_ascendc.py
- utils/verification_ascendc.py
- utils/performance.py

You MUST NOT read:
- archive_tasks/**
- skills/ascendc/ascendc-translator/**
- skills/ascendc/tilelang-designer/**
- skills/ascendc/performance-analyzer/**
- skills/ascendc/trace-recorder/**
- any non A5 workflow files

## Progress Reporting (MANDATORY)
Write {PROGRESS_FILE} after EVERY step.

## Workflow (Op-Gen)
1. Read source file — detect type (CUDA .cu/.cuh, PyTorch .py, AscendC .h)
2. Analyze: what the op computes, input/output types, algorithm
3. Generate AscendC kernel files:
   - {op_name}_kernel.h (AscendC kernel)
   - {op_name}_kernels.cpp (extern C entry)
   - pybind11.cpp (torch extension bridge, if PyTorch source)
   - CMakeLists.txt (if no existing build system)
4. Run static check + build (max 5 compile-fix iterations)
5. If test inputs available: verify precision
6. Output: kernel files in {OUTPUT_DIR}
7. Write final status into {PROGRESS_FILE}.

## Hard Exit Rules (MANDATORY — saves tokens)
- Compile fix: max 5 attempts → if still failing, write FAIL to {PROGRESS_FILE} and STOP
- Precision fix: max 3 attempts → if still failing, write FAIL to {PROGRESS_FILE} and STOP
- NEVER continue retrying beyond these limits

## Rules & KB
- Anti-hack enforced by orchestrator post-verification
- KB loading: follow ascend-a5-kernel-developer Phase 0
- NOTE: reading ~/workspace/cann/ IS allowed in op-gen mode (not benchmark)
- On build fail, load ERROR_CORRECTIONS.md (EC-13..EC-16, PB-9 most common)
```

---

## OPTIMIZE MODE — Worker Prompt Template

Skips Phase 0 analysis and Stage 1 code generation — starts directly at Stage 2 optimization.
Uses existing archived kernel as baseline.

Variables from arg: {PROBLEM_ID}, {NAME}, {LOWER_NAME}
Variables derived: {KERNEL_DIR} = output/npukernelbench/src/kernels/{PROBLEM_ID}
                   {BASELINE_PERF} = current mean ratio from RESULTS.md
                   {OUTPUT_DIR} = output/npukernelbench/{PROBLEM_ID}

```
You are an A5 AscendC kernel optimizer. Optimize existing kernel {PROBLEM_ID} for better performance.
The kernel already passes precision — your job is to improve performance while maintaining correctness.

## Baseline
- Kernel: {KERNEL_DIR}/ (5 files — read them first to understand current implementation)
- Source .py: benchmarks/NPUKernelBench/level1/{PROBLEM_ID}.py (reference implementation)
- Current performance: {BASELINE_PERF}
- Precision: PASS (must remain PASS after optimization)

## Local A5 Environment
- SOC_VERSION: Ascend950PR_9589
- CANN_PATH: /usr/local/Ascend/cann-9.0.T501/
- Build: python3 utils/build_ascendc.py {OUTPUT_DIR} -v Ascend950PR_9589 --build-type Release
- Verify: python3 utils/verification_ascendc.py {OUTPUT_DIR}
- Perf: python3 utils/performance.py {OUTPUT_DIR} all
- Static check: python3 skills/ascendc/a5-common-scripts/ascendc_static_check.py {OUTPUT_DIR}/kernel/

## Progress Reporting (MANDATORY)
Write {OUTPUT_DIR}/PROGRESS.md after EVERY step.
Use this format for the Optimization section:
```
Optimization:
  baseline: {BASELINE_PERF} (from RESULTS.md)
  current:  {X.XXx} (latest measurement)
  speedup:  {current/baseline}x
  history:
  - baseline: {BASELINE_PERF} — existing kernel
  - Opt1: {X.XXx} — {what changed}
  - Opt2: {X.XXx} — {what changed}
```

## Workflow (Optimize)
1. Read current kernel files in {KERNEL_DIR}/ — understand the implementation
2. Read source .py — understand the op semantics and test cases
3. Read KB for optimization ideas:
   - `skills/ascendc/a5-shared-references/KB_INDEX.md` → search for relevant optimization patterns
   - `skills/ascendc/a5-shared-references/ROOFLINE_MODEL.md` — compute theoretical upper bound
   - `skills/ascendc/a5-shared-references/MSPROF_AGENT_GUIDE.md` — if msprof data available
   - `skills/ascendc/a5-shared-references/patterns/domains/memory_access.md` — DMA/bandwidth patterns
4. Analyze bottleneck: why is current kernel slow? (algorithm? memory access? launch overhead?)
5. Design optimization strategy — document in PROGRESS.md before coding
6. Copy baseline to {OUTPUT_DIR}/, run perf test to confirm baseline number
7. Implement optimization (Edit existing files, do NOT rewrite from scratch)
8. Build + verify precision (must remain all-PASS)
9. Run perf test — compare vs baseline
10. If improved and precision PASS: record in PROGRESS.md
    If regressed or precision broken: revert changes and try different optimization
11. Repeat steps 7-10 (max 3 iterations)
12. Archive final version back to {KERNEL_DIR}/
13. Write final status into PROGRESS.md

## A5 Version Control
Use git in {OUTPUT_DIR}/ for rollback safety:
- Before first change: `cd {OUTPUT_DIR} && git init 2>/dev/null; git add -A && git commit -m "baseline: {BASELINE_PERF}" --allow-empty`
- After each successful optimization: `git add -A && git commit -m "OptN: description, perf Xx"`
- If precision breaks: `git checkout HEAD -- .` to rollback

## Anti-Hack Rules
- NEVER replace kernel with PyTorch/CANN delegation
- NEVER use CPU fallback
- Kernel must remain genuine AscendC
- NEVER claim improvement without actual perf data

## Hard Exit Rules (MANDATORY — saves tokens)
- Compile fix: max 5 attempts → if still failing, write FAIL to PROGRESS.md and STOP
- Precision fix: max 3 attempts → if still failing, write FAIL to PROGRESS.md and STOP
- Optimization: max 3 iterations → write current best perf to PROGRESS.md and STOP
- NEVER continue retrying beyond these limits
```

---

## DIAGNOSTIC MODE — Shared Prompt Appendix (all modes)

**If DIAGNOSTIC is true** (`--diag` flag), append this section to the worker prompt (benchmark, op-gen, or optimize):
```
## Diagnostic Output (--diag mode)
Write these extra sections to PROGRESS.md after each phase:

### DIAG: KB Loading
- Files read: [list each KB file read with line count]
- Files skipped: [list KB files NOT read and why]
- Keywords searched: [what terms you grepped in KB_INDEX]

### DIAG: Decision Trace
- SIMT/SIMD decision: [step-by-step traversal of decision tree]
- Patterns matched: [which P-PN entries you plan to apply]
- Pitfalls loaded: [which PB-N entries are relevant]

### DIAG: Token Estimate
- Source file: ~N lines read
- KB loaded: ~N lines total across all KB files
- Code generated: ~N lines

### DIAG: Anti-Hack Self-Check
- torch/CANN delegation: [yes/no, if yes what]
- CPU fallback: [yes/no]
- Genuine AscendC primitives used: [list: DataCopy, TQue, __gm__, etc.]
```
