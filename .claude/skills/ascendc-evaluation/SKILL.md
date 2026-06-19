---
name: ascendc-evaluation
description: Deploy and evaluate AscendC operators
---

## What I do

Generate PyBind bindings for AscendC operator, install the operator run file, evaluate correctness, and export msprof profiling summaries.

## When to use me

Use this when you need to evaluate operator correctness against reference implementations and measure performance using msprof.

## Workflow

**Python环境**: 始终使用 `.venv/bin/python3` 代替 `python3` (Unless .venv is not available)。

**[注意] MANDATORY: You MUST follow ALL THREE steps below in order. Do NOT write your own evaluate.py or skip any step. The `generate_pybind` step is REQUIRED — without it, `torch.ops.cust.<op>` will not be registered and all evaluations will fail with `'_OpNamespace' 'cust' object has no attribute '<op>'`.**

1. **Install run file** - Install `custom_opp_ubuntu_aarch64.run`
2. **Generate pybind** - Generate PyBind bindings and install the whl
3. **Evaluate operator** - Test correctness and measure performance

### 1. Install run file

The run file is usually located in `output/{op_name}/{op_name}Custom/build_out/` directory.

**Important:** You must use an **absolute path** for `--install-path`. Relative paths will cause "ERROR: use absolute path" and install nothing.

```bash
# Use absolute path ($(pwd) expands to current directory)
bash output/{op_name}/{op_name}Custom/build_out/custom_opp_ubuntu_aarch64.run --install-path=$(pwd)/output/{op_name}
```

**[注意] CRITICAL: Never use relative paths like `--install-path=output/{op_name}` — always use absolute paths.**

This will create the following structure:
```
output/{op_name}/
├── vendors/
│   └── customize/
│       └── op_api/
│           └── lib/
│               └── libcust_opapi.so
```

### 2. Generate pybind

Run `generate_pybind.py` to generate PyBind whl and install it.

**Standard usage** (output dir defaults to `output/{op_name}`):
```shell
.venv/bin/python3 .claude/skills/ascendc-evaluation/scripts/generate_pybind.py <op_name>
```

**Evolution/custom work dir** (e.g. inside a parallel evo directory):
```shell
.venv/bin/python3 .claude/skills/ascendc-evaluation/scripts/generate_pybind.py <op_name> --work-dir <work_dir>
```

This will:
1. Generate PyBind bindings
2. Build the whl package
3. Install the whl with pip

### 3. Evaluate operator

Run `evaluate.py` to evaluate correctness and performance.

**All evaluations use advanced performance testing with warmup and L2 cache clearing for accurate results.**

**Standard usage** (output dir defaults to `output/{op_name}`):
```shell
# Basic evaluation (correctness + performance test)
.venv/bin/python3 .claude/skills/ascendc-evaluation/scripts/evaluate.py <op_name>

# IMPORTANT: specify device explicitly (recommended in multi-NPU environments)
.venv/bin/python3 .claude/skills/ascendc-evaluation/scripts/evaluate.py <op_name> --device-id 2



# Specify operator type for detailed performance analysis
.venv/bin/python3 .claude/skills/ascendc-evaluation/scripts/evaluate.py <op_name> --task-type vector --device-id 2

# Generate or append test_cases.csv from a JSON case spec (no evaluation)
.venv/bin/python3 .claude/skills/ascendc-evaluation/scripts/evaluate.py <op_name> \
    --case-spec /path/to/case_spec.json \
    --append-cases \
    --generate-cases-only
```

Options:
- `--device-id`: NPU device ID used by evaluation (recommended to pass explicitly)
  - Default: `ASCEND_DEVICE_ID` env var
  - Fallback: `0` when env var is not set
  - In shared servers, always pass `--device-id <id>` to avoid accidental use of device 0

- `--task-type`: Operator type for performance analysis (`vector`, `cube`, `cv-mix`, `unknown`)
- `--case-spec`: Path to JSON case spec for dynamic test case generation
- `--append-cases`: Append generated cases to existing `test_cases.csv`
- `--generate-cases-only`: Only generate `test_cases.csv` and exit

**What evaluate.py does:**

1. Sets up environment variables:
   - `ASCEND_CUSTOM_OPP_PATH` to `output/{op_name}/vendors/customize`
   - Adds `op_api/lib` to `LD_LIBRARY_PATH`
   - Uses NPU device from `--device-id` (or `ASCEND_DEVICE_ID`, default `0`)

2. Generates `test_cases.csv` in `output/{op_name}/` with test case info:
   - `case_id`: Test case identifier
   - `var{i}_shape`: Shape of input tensor i
   - `var{i}_dtype`: Data type of input tensor i
   - Special parameters: `batch_size`, `seq_len`, `dim`, `eps`, etc.
    - If `--case-spec` is provided, the JSON spec drives the case list and can be appended

3. Runs correctness test comparing reference implementation vs custom operator

4. If correctness passes, runs performance test:
   - Uses large matrix warmup (10240x10240) and L2 cache clearing for accurate results
   - Automatically filters warmup operations (MatMulV3 + ReduceMax) from profiling results
   - **Directly overwrites** `op_summary_*.csv` with filtered data (removes warmup ops)

5. Exports profiling results to `output/{op_name}/profiling/`:
   - `op_summary_*.csv`: Summary of operator execution
   - Performance metrics and task duration

## Directory Structure

After full evaluation:
```
output/{op_name}/
├── {op_name}_reference.py      # Reference implementation
├── {op_name}_custom.py          # Custom operator code
├── test_cases.csv               # Test case information (auto-generated)
├── profiling/
│   ├── Model_*/                 # Reference model profiling data
│   └── ModelNew_*/              # Custom operator profiling data
└── vendors/
    └── customize/
        └── op_api/
            └── lib/
                └── libcust_opapi.so
```

## API Usage

You can also use `AscendBackend` directly:

```python
from pathlib import Path
from evaluate import AscendBackend

# Read code files
eval_code = Path("output/{op_name}/{op_name}_custom.py").read_text()
ref_code = Path("output/{op_name}/{op_name}_reference.py").read_text()

# Create backend
backend = AscendBackend(eval_code, ref_code)

# Test correctness
success, message = backend.evaluate_correctness()

# Simple performance test
ref_summary, ref_total_us, custom_summary, custom_total_us = backend.compare_performance(
    profile_root=Path("output/{op_name}/profiling")
)

# Advanced performance test (with warmup and cache clearing)
ref_time, ref_data, custom_time, custom_data = backend.compare_performance_advanced(
    profile_root=Path("output/{op_name}/profiling"),
    num_trials=20,
    task_type="vector"
)
```

## Console Output

- **Correctness result**: Pass/fail message with shape and value comparison details
- **Performance result**:
  - Advanced mode: Detailed performance metrics including vector/cube utilization

## Remote Evaluation

For remote evaluation, use the wrapper in `server/evaluate_wrapper.py`, which sets `ASCEND_CUSTOM_OPP_PATH` from the install directory and calls `evaluate_operator(..., skip_env_setup=True)`.

## Parameterized test cases (case_spec)

When the user specifies a clear generalization range and test dimensions, generate or append test cases via a JSON spec and `--case-spec`. This is intended for tool-driven workflows (e.g., claude code / automation) to repeatedly add new cases without overwriting existing ones.

Example `case_spec.json`:
```json
{
    "case_id_start": 0,
    "cases": [
        {"batch_size": 1, "seq_len": 128, "var0_shape": [1, 128], "var0_dtype": "float16"}
    ],
    "grid": {
        "batch_size": [1, 2],
        "seq_len": [128, 256],
        "var0_shape": [[1, 128], [1, 256]],
        "var0_dtype": ["float16"]
    }
}
```

Append new cases without running evaluation:
```shell
python3 .claude/skills/ascendc-evaluation/scripts/evaluate.py <op_name> \
    --case-spec /path/to/case_spec.json \
    --append-cases \
    --generate-cases-only
```

Run evaluation using the same case spec:
```shell
python3 .claude/skills/ascendc-evaluation/scripts/evaluate.py <op_name> \
    --case-spec /path/to/case_spec.json
```

## [注意] CRITICAL: build.sh Must Use `source`, Not `bash`

In docker+tmux environments, **never run build.sh with `bash build.sh`**. This creates a new process group and causes cmake/make processes to receive SIGTTIN/SIGTTOU signals, stopping them (Tl state) with no output.

**Wrong:**
```bash
bash output/{op_name}/{op_name}Custom/build.sh
# → cmake/make stopped (Tl), returns empty output
```

**Correct:**
```bash
source output/{op_name}/{op_name}Custom/build.sh
# OR run in a dedicated tmux window/pane
```

This applies to **any compilation or build step** in this environment.

## Three-Way Precision Comparison

`evaluate.py` supports a three-way precision comparison to provide more accurate and meaningful correctness judgments. Instead of requiring a 100% match rate against a single reference, the system compares the custom kernel's error against PyTorch's own numerical error.

### Reference Models

- **Golden** (CPU fp64): The highest-precision reference. Computes the operator on CPU using float64 to produce a near-exact baseline.
- **Ref** (NPU, working dtype): The PyTorch reference implementation running on NPU in the operator's working dtype (e.g., float16, bfloat16, float32). This captures PyTorch's own numerical error.
- **Ans** (NPU, working dtype): The custom AscendC kernel running on NPU in the same working dtype.

### How It Works

The system computes errors for both `ref` and `ans` against the `golden` baseline, then forms ratios:

```
ratio = ans_error / max(ref_error, floor)
```

The `floor` prevents division-by-zero when PyTorch's error is extremely small. Ratio-based judgment means the custom kernel is evaluated relative to PyTorch's own precision, not against an absolute threshold.

## Pass/Fail Judgment

A test case passes when ALL of the following conditions are met:

```
passed = (
    NO new NaN/Inf introduced
    AND ratios.max_re  <= 10.0
    AND ratios.mean_re <= 2.0
    AND ratios.rmse    <= 2.0
    AND ratios.svec    <= 2.0
)
```

This means: as long as the custom kernel's error is within reasonable multiples of PyTorch's own error, it PASSES. A kernel does not need to be more precise than PyTorch itself.

- `max_re`: Maximum relative error ratio (allows up to 10x for isolated outliers)
- `mean_re`: Mean relative error ratio (must be within 2x on average)
- `rmse`: Root mean square error ratio (must be within 2x)
- `svec`: Small-value error count ratio (must be within 2x)

## Per-Dtype Tolerances

Default tolerance values used by the three-way comparison, per dtype:

| dtype | atol | rtol | ulp_tol | sv_th | sv_err |
|-------|------|------|---------|-------|--------|
| bfloat16 | 1e-2 | 1e-2 | 2 | 2^-8 | 2^-16 |
| float16 | 1e-3 | 1e-3 | 2 | 2^-11 | 2^-16 |
| float32 | 1e-5 | 1e-5 | 2 | 2^-14 | 2^-30 |

- `atol` / `rtol`: Absolute and relative tolerance for element-wise comparison
- `ulp_tol`: ULP (units in the last place) tolerance
- `sv_th`: Small-value threshold -- values below this magnitude are treated as "small"
- `sv_err`: Small-value error threshold

## --json-output Parameter

Export detailed precision results to a JSON file:

```shell
python3 .claude/skills/ascendc-evaluation/scripts/evaluate.py <op_name> \
    --json-output output/{op_name}/precision.json
```

This produces a structured precision report that can be used for automated analysis, dashboards, or CI pipelines.

## Precision JSON Format

The precision JSON file follows this structure:

```json
{
    "schema_version": 2,
    "cases": [
        {
            "case_id": "case_0",
            "forward": {
                "output_0": {
                    "passed": true,
                    "ratios": {
                        "max_re": 1.23,
                        "mean_re": 0.95,
                        "rmse": 0.88,
                        "svec": 1.01
                    },
                    "diagnosis": null
                }
            }
        }
    ]
}
```

Key fields:
- `schema_version`: Version of the precision JSON schema
- `cases`: Array of per-test-case results
- `forward`: Contains per-output precision data
- `ratios`: The computed error ratios (ans_error / max(ref_error, floor))
- `diagnosis`: Diagnosis pattern string if the test failed, or `null` if passed

## Diagnosis Patterns

When a test case fails, the system assigns a diagnosis pattern to help identify the root cause:

| Pattern | Description | Likely Cause |
|---------|-------------|--------------|
| `nan_inf_introduced` | NaN/Inf in ans but not in golden/ref | Division by zero, log(0), exp overflow |
| `zero_output` | Output near-zero everywhere | Placeholder code, incomplete computation |
| `localized_outlier` | max_re high, mean_re ok | Boundary or special-value path issue |
| `systematic_drift` | mean_re exceeded | Algorithm error, dtype cast loss |
| `small_value_error` | svec exceeded | Small-value underflow |
| `uneven_distribution` | rmse exceeded, others ok | Input-dependent precision patterns |

## Dashboard Visualization

Precision results are saved in `output/{op_name}/precision.json`.

## Backward Compatibility

- `evaluate_correctness()` still returns `(bool, str)` -- no downstream changes needed.
- If the golden model (CPU fp64) fails to compute (e.g., unsupported op on CPU), the system automatically falls back to legacy two-way comparison using direct match rate.
- All existing CLI arguments (`--task-type`, `--case-spec`, etc.) continue to work unchanged.
- `--advanced-perf` is accepted but ignored (all evaluations now use advanced mode).
