---
name: ascendc-debugger
description: >
  Unified AscendC debugger. Handles compile errors, PyTorch degenerate fallback,
  and precision mismatches by deeply analyzing code context (TileLang design,
  kernel implementation, reference model) and making intelligent fixes.
  Includes the precision-tuning forensics + Gate-controlled loop previously
  hosted in the separate ascendc-precision-tuner skill (now merged here).
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
3. **Precision mismatches** — Run numerical forensics (L0–L8 structured analysis),
   compare kernel with reference model.py logic, fix numerical issues in kernel
4. **Other verify failures** — Shape mismatch, NaN/Inf, runtime error

This skill **subsumes** the old `ascendc-precision-tuner` skill: when failure
mode is `precision`, the same forensics + Gate-A/X/V loop runs here, just under
`{task_dir}/debug/` instead of `{task_dir}/precision_tuning/`.

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

**Core principles:**
- Agent reads ALL context before making a fix. Never guess.
- Python scripts do deterministic work (diagnose, forensics, Gate). Agent does
  reasoning work (analysis, fix).

State files all live under `{task_dir}/debug/`:
- `diagnosis_report.json`            — produced by diagnose.py
- `forensics_report_{attempt}.json`  — produced by precision_forensics.py (precision mode)
- `debug_audit_{attempt}.md`         — written by Agent
- `validation_result_attempt_{attempt}.json` — saved by Agent after re-verify
- `baseline_state.json`              — captured by Gate-F at attempt 0 (precision mode)
- `history/attempt_{N}/...`          — archived per round
- `history/baseline/code_snapshot/`  — immutable starting code
- `history/current_best/code_snapshot/` — best code so far (precision mode)

---

### Step 0: Initialize

Set round counter `attempt = 0`.

**0.1 Save immutable baseline snapshot (first execution only):**
```bash
if [ ! -d "{task_dir}/debug/history/baseline/code_snapshot" ]; then
    mkdir -p "{task_dir}/debug/history/baseline/code_snapshot"
    cp "{task_dir}/kernel/"*.cpp "{task_dir}/kernel/"*.h \
       "{task_dir}/debug/history/baseline/code_snapshot/" 2>/dev/null || true
    cp "{task_dir}/model_new_ascendc.py" \
       "{task_dir}/debug/history/baseline/" 2>/dev/null || true
    echo "Baseline snapshot saved"
fi
```

**0.2 Save current round starting snapshot:**
```bash
mkdir -p "{task_dir}/debug/history/attempt_{attempt}/code_snapshot"
cp "{task_dir}/kernel/"*.cpp "{task_dir}/kernel/"*.h \
   "{task_dir}/debug/history/attempt_{attempt}/code_snapshot/" 2>/dev/null || true
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

Validate with **Gate-D**:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/debug_gate.py \
    --step diagnose --task-name {task_name} --workdir "{workdir}" --attempt {attempt}
```

**Gate-D failure → STOP, check environment.**

---

### Step 1.5 (precision mode only): Forensics

If `diagnosis_report.json.failure_mode == "precision"`, run forensics **before**
writing the audit:

```bash
python3 skills/ascendc/ascendc-debugger/scripts/precision_forensics.py \
    {task_name} --workdir "{workdir}" --attempt {attempt}
```

Output: `{task_dir}/debug/forensics_report_{attempt}.json` with L0–L8 structured
analysis (basic_stats, sign_analysis, worst_elements, tail_analysis,
dimension_analysis, L8_operator hints, etc.).

Validate with **Gate-F**:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/debug_gate.py \
    --step forensics --task-name {task_name} --workdir "{workdir}" --attempt {attempt}
```

**Gate-F failure → STOP, check forensics script output.** On Gate-F PASS at
attempt 0, the gate also writes `{task_dir}/debug/baseline_state.json` capturing
the initial match-rate / mismatch_ratio / max_abs_diff for later progress
comparison.

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

Plus failure-specific reads:
- precision: `{task_dir}/debug/forensics_report_{attempt}.json`
- precision (attempt > 0): `{task_dir}/debug/history/attempt_{attempt-1}/debug_audit.md`

#### Compile errors (`failure_mode == "compile"`)

From diagnosis report:
- `checks.compile.errors` — structured error list
- `checks.compile.primary_error` — dominant error category
- `checks.compile.affected_files` — which files to focus on

For each error:
1. Read the error location in kernel code
2. Read the TileLang design to understand intended behavior
3. Check if the error is a **translation artifact** (TileLang→AscendC转译引入)
4. Check AscendC API usage against reference
5. Determine the correct fix

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

Read `forensics_report_{attempt}.json` (Step 1.5) and decompose:

1. Read `model.py` forward() — understand reference computation
2. Read kernel `Compute()` or main kernel function
3. Decompose computation into ordered steps
4. Map each kernel step to reference step
5. Identify where numerical deviation is introduced (using forensics
   `dimension_analysis`, `worst_elements`, `tail_analysis`, `L8_operator` hints)

Common precision issues:
- Tail block handling incorrect (last partial tile)
- Accumulator not initialized to zero
- Missing upcast in float16 accumulation
- Wrong tiling parameter causing buffer overflow
- DataCopy count mismatch
- ReduceMax/ReduceSum dst overlapping src
- Non-32-byte-aligned single-element GM scalar write
- HardEvent flag id reused without ping-pong → cross-pipe race

**Optional knowledge base search:**
```bash
python3 skills/ascendc/ascendc-debugger/scripts/precision_knowledge.py search \
    --kb-path skills/ascendc/ascendc-debugger/references/precision_knowledge_base.json \
    --op-type <op_type> --pattern <primary_hint> --top-k 3
```

(`precision_knowledge.py` may be added later; if absent, skip this step.)

#### Other verify errors (`failure_mode == "verify"`)

Read verification output from diagnosis report:
- Shape mismatch → check tiling parameters vs input shape
- NaN/Inf → check division by zero, sqrt of negative, log of non-positive
- Runtime error → check NPU memory access, buffer bounds

---

**Write analysis to `{task_dir}/debug/debug_audit_{attempt}.md`:**

Common required sections (all failure modes):
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

[ROOT_CAUSE]
  <Clear statement of the root cause + evidence chain>

[FIX_PLAN]
  <Step-by-step fix instructions, files to modify, expected outcome>

[TARGET_FILES]
  <List of files to modify>

=== END AUDIT ===
```

**Additional sections required when `failure_mode == "precision"`:**

```markdown
[FORENSICS_SUMMARY]
  - primary_hint, primary_confidence, primary_evidence
  - mismatch_ratio, max_abs_diff, mean_abs_diff
  - error_distribution (sign_analysis.bias_direction)
  - worst_elements (top 3)
  - tail_analysis
  - dimension_analysis
  - L8_operator.op_type

[COMPUTATION_DECOMPOSITION]
  - Decompose model.py forward() into ordered computation steps
  - For each step: operation name, input source, output shape, precision risk

[KERNEL_STEP_TRACE]
  - Trace kernel/*.cpp Compute() function
  - Map each kernel step to computation decomposition step
  - Mark match status with ✅/⚠️/❌

[DIRECTION_ASSESSMENT]   (required when attempt > 0)
  - Previous round fix direction
  - Whether to continue same direction (是/否)
  - Reason
```

Validate with **Gate-A**:
```bash
python3 skills/ascendc/ascendc-debugger/scripts/debug_gate.py \
    --step audit --task-name {task_name} --workdir "{workdir}" --attempt {attempt}
```

**Gate-A failure → complete missing sections, do not count as an attempt.**

For precision mode, Gate-A enforces that all five extra sections above
(`FORENSICS_SUMMARY`, `COMPUTATION_DECOMPOSITION`, `KERNEL_STEP_TRACE`, plus
`DIRECTION_ASSESSMENT` when attempt > 0) are present in addition to the common
ones.

---

### Step 3: Code Fix (Agent execution)

Apply fixes according to [FIX_PLAN].

**Rules:**
1. **Follow FIX_PLAN strictly**, do not expand scope
2. **Write complete files**, do not truncate
3. **Use actual variable names** from the code
4. **For precision/verify**: do NOT shrink shapes, skip cases, or enlarge tolerance

**For compile fixes:** Modify `kernel/*.cpp` and/or `kernel/*.h`
**For degenerate fixes:** Modify `model_new_ascendc.py`
**For precision/verify fixes:** Modify `kernel/*.cpp` and/or `kernel/*.h`

**Diagnostic insertion (optional, for tough bugs):**
If you cannot determine the exact cause, insert diagnostic prints:
```cpp
// In kernel code, add temporary debug output
printf("DEBUG: tile_id=%d, count=%d, val=%f\n", tile_id, count, val);
```
Then rebuild and re-verify to see the debug output. Remove debug prints before
finalizing.

Validate with **Gate-X**:
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

(For precision mode, also re-run forensics so the next round has a fresh report:
this is handled at the start of attempt+1 via Step 1.5, no extra action here.)

**Save validation summary** for progress tracking. Parse verification output
into `{task_dir}/debug/validation_result_attempt_{attempt}.json`:

```json
{
  "attempt": <N>,
  "correctness_passed": true/false,
  "evaluate_stdout": "<verification output>",
  "match_rate": "<extracted percentage>",
  "max_diff": "<extracted max_diff>"
}
```

Extract rules:
- `PASS` in verification output → `correctness_passed: true`, `match_rate: "100.0"`, `max_diff: "0.0"`
- mismatch reported → parse mismatch_ratio, max_abs_diff

Validate with **Gate-V**:
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
# Precision-mode artifacts (only present when failure_mode == "precision")
cp "{task_dir}/debug/forensics_report_{attempt}.json" \
   "{task_dir}/debug/history/attempt_{attempt}/forensics_report.json" 2>/dev/null || true

cp -r "{task_dir}/kernel" "{task_dir}/debug/history/attempt_{attempt}/" 2>/dev/null || true
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/debug/history/attempt_{attempt}/" 2>/dev/null || true

# Track best code (precision mode):
current_mr=$(python3 -c "import json,os; p='{task_dir}/debug/validation_result_attempt_{attempt}.json'; r=json.load(open(p)) if os.path.exists(p) else {}; print(r.get('match_rate', '0'))")
best_mr=0
if [ -f "{task_dir}/debug/history/current_best/match_rate.txt" ]; then
    best_mr=$(cat "{task_dir}/debug/history/current_best/match_rate.txt")
fi
is_better=$(python3 -c "print('yes' if float('$current_mr') >= float('$best_mr') else 'no')")
if [ "$is_better" = "yes" ]; then
    mkdir -p "{task_dir}/debug/history/current_best/code_snapshot"
    cp "{task_dir}/kernel/"*.cpp "{task_dir}/kernel/"*.h \
       "{task_dir}/debug/history/current_best/code_snapshot/" 2>/dev/null || true
    echo "$current_mr" > "{task_dir}/debug/history/current_best/match_rate.txt"
fi

# Save next attempt starting snapshot
next=$((attempt + 1))
mkdir -p "{task_dir}/debug/history/attempt_${next}/code_snapshot"
cp "{task_dir}/kernel/"*.cpp "{task_dir}/kernel/"*.h \
   "{task_dir}/debug/history/attempt_${next}/code_snapshot/" 2>/dev/null || true
```

Then `attempt += 1`, back to Step 1.

---

### Step 5: Success

```bash
echo "DEBUG SUCCESS" > "{task_dir}/debug/status.txt"
mkdir -p "{task_dir}/debug/history/success/code_snapshot"
cp "{task_dir}/kernel/"*.cpp "{task_dir}/kernel/"*.h \
   "{task_dir}/debug/history/success/code_snapshot/" 2>/dev/null || true
echo "100.0" > "{task_dir}/debug/history/current_best/match_rate.txt"
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

Best code is saved in `debug/history/current_best/`. To restore:
```bash
cp "{task_dir}/debug/history/current_best/code_snapshot/"*.cpp \
   "{task_dir}/debug/history/current_best/code_snapshot/"*.h \
   "{task_dir}/kernel/"
```

---

## Note

- **Read all context before fixing** — TileLang design, kernel code, model.py, wrapper
- **Every Gate is mandatory** (D, F for precision, A, X, V)
- **loop_signal is determined by Gate, Agent must obey**
- **Compile errors and precision errors may share root causes** — a bad TileLang→AscendC translation can cause both
- **Debug prints are a legitimate tool** — use them when you cannot determine root cause from static analysis
- **Knowledge base entries** (under `references/precision_knowledge_base.json`) are optional reads when looking for known patterns; they should only be **written** on success of precision-tuning rounds
- **Compile failures inside a fix iteration do not count as a tuning round** — max 3 compile retries before restoring from `history/attempt_{attempt}/code_snapshot/` and restarting Step 2
