---
name: ascendc-debugger
description: >
  Unified AscendC debugger. Handles compile errors, PyTorch degenerate fallback,
  and precision mismatches by deeply analyzing code context (TileLang design,
  kernel implementation, reference model) and making intelligent fixes.
subagent:
  enabled: true
  agent_type: general
  reason: >
    Debugging AscendC requires understanding the full context:
    TileLang design intent, generated kernel code, and PyTorch reference.
    The agent must read multiple files, analyze relationships, and make
    targeted fixes. This requires deep reasoning best done by a subagent.
  timeout: 3600
  max_iterations: 60
---

## What I do

Debug AscendC operators that fail at any stage of the build-verify pipeline:

1. **Compile errors** — Parse cmake/gcc output, read affected kernel code,
   compare with TileLang design intent, fix C++ code
2. **PyTorch degenerate** — Analyze model_new_ascendc.py wrapper,
   check kernel import/call chain, fix Python wrapper
3. **Precision mismatches** — Run numerical forensics, compare kernel with
   reference model.py logic, fix numerical issues in kernel

## When to use me

Called automatically from ascend-kernel-developer Phase 4 when any check fails:
- Degenerate check fails
- Compile fails
- Verification fails (precision or other)

## Prerequisites

- `{workdir}/{task}/design/tile_level/` — TileLang design
- `{workdir}/{task}/kernel/*.cpp` — AscendC kernel sources
- `{workdir}/{task}/model.py` — Reference PyTorch model
- `{workdir}/{task}/model_new_ascendc.py` — AscendC wrapper
- `utils/build_ascendc.py` and `utils/verification_ascendc.py`

## Argument

```
task_name=<task> workdir=<workdir> soc_version=<soc>
```

## Workflow

**All thinking, analysis, reasoning must use Chinese.**

**Core principle: Agent reads ALL context before making a fix. Never guess.**

---

### Step 0: Initialize

Set round counter `attempt = 0`.

Save baseline code snapshot:
```bash
mkdir -p "{task_dir}/debug/history/baseline/code_snapshot"
cp "{task_dir}/kernel/"*.cpp "{task_dir}/debug/history/baseline/code_snapshot/" 2>/dev/null || true
cp "{task_dir}/model_new_ascendc.py" "{task_dir}/debug/history/baseline/" 2>/dev/null || true
```

---

### Step 1: Run Diagnosis

Run the unified diagnostic tool:
```bash
cd {workdir}
python3 skills/ascendc/ascendc-debugger/scripts/diagnose.py \
    {task_name} --workdir "{workdir}" --soc-version {soc_version}
```

This runs three checks in order:
1. **Degenerate check** — validate_ascendc_impl.py
2. **Compile** — build_ascendc.py
3. **Verify** — verification_ascendc.py

Output: `{task_dir}/debug/diagnosis_report.json`

Validate with Gate-D:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/debug_gate.py \
    --step diagnose --task-name {task_name} --workdir "{workdir}" --attempt {attempt}
```

**Gate-D failure → STOP, check environment.**

---

### Step 2: Agent Deep Analysis

**Read the diagnosis report first:**
```bash
cat "{task_dir}/debug/diagnosis_report.json"
```

Determine `failure_mode`:
- `"degenerate"` — model_new_ascendc.py has PyTorch fallback
- `"compile"` — build_ascendc.py failed
- `"precision"` — verification failed with mismatch_ratio > 0
- `"verify"` — verification failed for other reasons (shape, NaN, etc.)

**Then read ALL relevant context:**

For ALL failure modes, read:
1. `{task_dir}/design/tile_level/` — TileLang design intent
2. `{task_dir}/design/block_level/` — Block-level design
3. `{task_dir}/model.py` — Reference PyTorch implementation
4. `{task_dir}/model_new_ascendc.py` — Current wrapper
5. `{task_dir}/kernel/*.cpp` and `*.h` — All kernel sources

**Failure-specific analysis:**

#### Compile errors (`failure_mode == "compile"`)

From diagnosis report, get:
- `checks.compile.errors` — structured error list
- `checks.compile.primary_error` — dominant error category
- `checks.compile.affected_files` — which files to focus on

For each error:
1. Read the error location in kernel code
2. Read the TileLang design to understand intended behavior
3. Check if the error is a **translation artifact** (TileLang→AscendC转译引入)
4. Check AscendC API usage against reference
5. Determine the correct fix

Common compile error categories and fixes:

| Category | Typical Cause | Fix Strategy |
|----------|--------------|--------------|
| `undefined_api` | Used non-existent API (e.g., `Vmax` instead of `Max`) | Replace with correct API name |
| `type_mismatch` | Wrong dtype in API call | Add explicit cast or use correct type |
| `syntax` | Missing semicolon/brace, bad macro | Fix syntax |
| `alignment` | Count not aligned to required boundary | Pad count or use aligned count |
| `header_missing` | Missing include | Add correct include |
| `link_error` | Undefined symbol in pybind11.cpp | Fix symbol name or add implementation |

#### Degenerate errors (`failure_mode == "degenerate"`)

From diagnosis report:
- `checks.degenerate.regression_type` — Type 1-4
- `checks.degenerate.suggestion` — specific suggestion

Read `model_new_ascendc.py`:
1. Check if AscendC extension is imported (`import _xxx_ext`)
2. Check if `forward()` calls the kernel extension
3. Check if any forbidden PyTorch ops remain
4. Check for element-wise Python for loops

Fix the wrapper code in `model_new_ascendc.py`.

#### Precision errors (`failure_mode == "precision"`)

From diagnosis report:
- `checks.verify.metrics.match_rate`
- `checks.verify.metrics.max_abs_diff`

**Run precision forensics for detailed analysis:**
```bash
python3 skills/ascendc/ascendc-precision-tuner/scripts/precision_forensics.py \
    {task_name} --workdir "{workdir}" --attempt {attempt}
```

Read `forensics_report_{attempt}.json` for L0-L8 analysis.

Then:
1. Read `model.py` forward() — understand reference computation
2. Read kernel `Compute()` or main kernel function
3. Decompose computation into steps
4. Map each kernel step to reference step
5. Identify where numerical deviation is introduced

Common precision issues:
- Tail block handling incorrect
- Accumulator not initialized to zero
- Missing upcast in float16 accumulation
- Wrong tiling parameter causing buffer overflow
- DataCopy count mismatch

#### Other verify errors (`failure_mode == "verify"`)

Read verification output from diagnosis report.
Common non-precision failures:
- Shape mismatch → check tiling parameters vs input shape
- NaN/Inf → check division by zero, sqrt of negative, log of non-positive
- Runtime error → check NPU memory access, buffer bounds

**Write analysis to `{task_dir}/debug/debug_audit_{attempt}.md`:**

```markdown
=== DEBUG AUDIT ===

[DIAGNOSIS]
  failure_mode: <compile|degenerate|precision|verify>
  diagnosis_summary: <one sentence>

[CONTEXT]
  TileLang design: <key design decisions from design/>
  Reference model: <model.py forward() logic summary>
  Kernel structure: <list of kernel files and their roles>

[ANALYSIS]
  <Detailed analysis of the failure>
  <For compile: each error with root cause>
  <For degenerate: which check failed and why>
  <For precision: where numerical deviation starts>
  <For verify: what caused the non-precision failure>

[ROOT_CAUSE]
  <Clear statement of the root cause>
  <Evidence from code inspection>

[FIX_PLAN]
  <Step-by-step fix instructions>
  <Files to modify and how>
  <Expected outcome after fix>

[TARGET_FILES]
  <List of files to modify>

=== END AUDIT ===
```

Validate with Gate-A:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/debug_gate.py \
    --step audit --task-name {task_name} --workdir "{workdir}" --attempt {attempt}
```

**Gate-A failure → complete missing sections.**

---

### Step 3: Code Fix (Agent execution)

Apply fixes according to [FIX_PLAN].

**Rules:**
1. **Follow FIX_PLAN strictly**, do not expand scope
2. **Write complete files**, do not truncate
3. **Use actual variable names** from the code
4. **For precision**: do NOT shrink shapes, skip cases, or enlarge tolerance

**For compile fixes:** Modify `kernel/*.cpp` and/or `kernel/*.h`
**For degenerate fixes:** Modify `model_new_ascendc.py`
**For precision fixes:** Modify `kernel/*.cpp` and/or `kernel/*.h`

**Diagnostic insertion (optional, for tough bugs):**
If you cannot determine the exact cause, insert diagnostic prints:
```cpp
// In kernel code, add temporary debug output
printf("DEBUG: tile_id=%d, count=%d, val=%f\n", tile_id, count, val);
```
Then rebuild and re-verify to see the debug output.
Remember to remove debug prints before finalizing.

Validate with Gate-X:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/debug_gate.py \
    --step fix --task-name {task_name} --workdir "{workdir}" --attempt {attempt}
```

**Gate-X failure → check files saved correctly.**

---

### Step 4: Validate (Re-diagnosis)

Re-run full diagnosis to verify the fix:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/diagnose.py \
    {task_name} --workdir "{workdir}" --soc-version {soc_version}
```

Validate with Gate-V:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/debug_gate.py \
    --step validate --task-name {task_name} --workdir "{workdir}" --attempt {attempt}
```

| loop_signal | Meaning | Action |
|-------------|---------|--------|
| **PASS** | All checks passed | → Step 5 (success) |
| **CONTINUE** | Still failing but making progress | → Archive, attempt+1, back to Step 1 |
| **STOP** | No progress or max/elastic attempts reached | → Step 6 (failure report) |

**Agent MUST obey loop_signal.**

**Elastic Attempt Limits (3–15 rounds, auto-selected):**

| failure_mode | Base max | Extend to 15 if... | Early stop if... |
|--------------|----------|-------------------|------------------|
| `degenerate` | 3 | — | stuck for 2 rounds |
| `compile` | 5 | 2 consecutive progress rounds | stuck for 2 rounds |
| `precision` | 10 | 2 consecutive progress rounds | stuck for 2 rounds |
| `verify` | 5 | 2 consecutive progress rounds | stuck for 2 rounds |

- Minimum attempts: **3** (always run at least 3 rounds)
- Maximum attempts: **15** (only when making consistent progress)
- Progress detection: error count decreasing OR match_rate improving for precision mode

---

### Archive (CONTINUE)

```bash
mkdir -p "{task_dir}/debug/history/attempt_{attempt}"
cp "{task_dir}/debug/diagnosis_report.json" \
   "{task_dir}/debug/history/attempt_{attempt}/diagnosis_report.json"
cp "{task_dir}/debug/debug_audit_{attempt}.md" \
   "{task_dir}/debug/history/attempt_{attempt}/debug_audit.md"
cp -r "{task_dir}/kernel" "{task_dir}/debug/history/attempt_{attempt}/" 2>/dev/null || true
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/debug/history/attempt_{attempt}/" 2>/dev/null || true
```

Rename current diagnosis report for history:
```bash
mv "{task_dir}/debug/diagnosis_report.json" \
   "{task_dir}/debug/history/attempt_{attempt}/diagnosis_report.json"
```

Then `attempt += 1`, back to Step 1.

---

### Step 5: Success

```bash
echo "DEBUG SUCCESS" > "{task_dir}/debug/status.txt"
```

Output:
```
[DEBUG_RESULT]
  status: SUCCESS
  attempts: <total rounds>
  failure_mode: <original failure mode>
  root_cause_summary: <one sentence>
  fix_summary: <one sentence>
```

---

### Step 6: Failure Report

```
[DEBUG_RESULT]
  status: FAILED
  attempts: <total rounds>
  original_failure_mode: <mode>
  loop_stop_reason: <Gate reason>
  history:
    attempt 0: mode=<mode>, errors=<count>, fix=<summary>
    ...
  remaining_issue: <description>
  suggestion: <advice for manual debugging>
```

Best code is saved in `debug/history/attempt_{best}/`.

---

## Note

- **Read all context before fixing** — TileLang design, kernel code, model.py, wrapper
- **Every Gate is mandatory**
- **loop_signal is determined by Gate, Agent must obey**
- **Compile errors and precision errors may share root causes** — a bad TileLang→AscendC translation can cause both
- **Debug prints are a legitimate tool** — use them when you cannot determine root cause from static analysis
