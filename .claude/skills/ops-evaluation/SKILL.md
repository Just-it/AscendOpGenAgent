---
name: ops-evaluation
description: Build, deploy, and evaluate AscendC operators from ops-nn/cv/math/transformer repositories, comparing baseline vs evolved performance.
---

## What I do

Build ops repository operators using `build.sh`, install to specified paths, generate PyBind bindings, run correctness verification and performance profiling, and compare baseline vs evolved versions.

## When to use me

Use this in the ops repository operator evolution optimization scenario, when you need to evaluate performance differences between the pre-optimization (baseline) and post-optimization (evolved) versions of an operator.

## Workflow

**Python environment**: Always use `.venv/bin/python3` instead of `python3` (unless .venv is not available).

**MANDATORY: Follow ALL steps below in order. Do NOT write your own evaluation scripts or skip any step.**

### Step 1: Build and install operator

Call `build_ops.py` to build and install the operator from the ops repository.

```bash
python3 .claude/skills/ops-evaluation/scripts/build_ops.py \
    --repo-root {REPO_ROOT} \
    --op-name {custom_op_name} \
    --soc {soc} \
    --install-path {absolute_install_path}
```

**Important:** You must use an **absolute path** for `--install-path`. Relative paths will cause installation failures.

The script internally executes:
```bash
cd {REPO_ROOT}
rm -rf build/ build_out/
bash build.sh --pkg --vendor_name=custom --soc={soc} --ops={custom_op_name} -j$(nproc)
./build_out/cann-ops-*-custom-linux.*.run --install-path={install_path}
```

You must build separately for baseline and evolved versions:
1. Build baseline first (original code in repository)
2. Apply modified code to the repository
3. Build evolved version to a different install path
4. Restore original code via `git checkout`

### Step 2: Generate PyBind

Run `generate_pybind.py` to generate PyBind bindings and install the whl.

```bash
python3 .claude/skills/ops-evaluation/scripts/generate_pybind.py {op_name} \
    --work-dir {install_path_dir}
```

PyBind only needs to be generated **once** (the aclnn interfaces are the same for baseline and evolved).

### Step 3: Evaluate correctness and performance

Call `evaluate_ops.py` to compare baseline vs evolved versions.

```bash
python3 .claude/skills/ops-evaluation/scripts/evaluate_ops.py {op_name} \
    --baseline-path {baseline_install_path} \
    --evolved-path {evolved_install_path} \
    --reference-py {path_to_reference.py} \
    --custom-py {path_to_custom.py} \
    --device-id {device_id} \
    --task-type {vector|cube}
```

The script uses **subprocess isolation** to evaluate each version separately (CANN runtime cannot switch OPP libraries once loaded):
- Subprocess 1: `ASCEND_CUSTOM_OPP_PATH={baseline}/vendors/{vendor_subdir}` -> correctness + profiling
- Subprocess 2: `ASCEND_CUSTOM_OPP_PATH={evolved}/vendors/{vendor_subdir}` -> correctness + profiling

Options:
- `--device-id`: NPU device ID (default: 0)
- `--task-type`: Operator type for profiling analysis (`vector`, `cube`, `cv-mix`, `unknown`)
- `--output`: Path for evaluation results JSON (default: `evaluation_results.json` in evolved path)
- `--num-trials`: Number of profiling trials (default: 20)

### Step 4: Compare output

The script automatically merges profiling data from both versions and generates a comparison report `evaluation_results.json`.

## Output Format

`evaluation_results.json`:
```json
{
  "op_name": "ada_layer_norm",
  "repo_type": "nn",
  "soc": "ascend910b",
  "baseline": {
    "install_path": "output/.../baseline",
    "time_us": 456.5,
    "precision_passed": true,
    "profiling_dir": "output/.../baseline_profiling/",
    "pipeline": {"mte2_pct": 38.0, "vec_pct": 47.0, "scalar_pct": 12.0, "mte3_pct": 18.0},
    "bottleneck": "memory_bound"
  },
  "evolved": {
    "install_path": "output/.../evolved",
    "time_us": 233.8,
    "precision_passed": true,
    "profiling_dir": "output/.../evolved_profiling/",
    "pipeline": {"mte2_pct": 28.0, "vec_pct": 49.0, "scalar_pct": 8.0, "mte3_pct": 15.0},
    "bottleneck": "balanced"
  },
  "comparison": {
    "speedup": 1.95,
    "time_delta_us": -222.7,
    "bottleneck_change": "memory_bound -> balanced",
    "compilation_success": true,
    "precision_passed": true
  }
}
```

## Error Guide

- `ASCEND custom OPP directory not found`: Check that `--install-path` uses an absolute path
- Build failure: Check that `ASCEND_HOME_PATH` is set; check that `--ops=` operator name is correct
- Repository type auto-detection: Detected via `.run` filename keywords (`ops-nn`/`ops-cv`/`ops-math`) or `REPOSITORY_NAME` in `build.sh`
- `Cannot switch OPP path at runtime`: The evaluate_ops.py script handles this via subprocess isolation; do not try to evaluate both versions in the same process

## build.sh Notes

**CRITICAL**: In docker+tmux environments, the ops repository's `build.sh` should be called directly (not via `source`), as it is designed to be run as a standalone script unlike the custom operator's `build.sh`.

## Directory Structure

After full evaluation:
```
output/{op_name}_ops-evo_{timestamp}/
├── baseline/                        # Baseline install directory
│   └── vendors/custom_nn/...
├── evolved/                         # Evolved install directory
│   └── vendors/custom_nn/...
├── baseline_profiling/              # Baseline profiling data
├── evolved_profiling/               # Evolved profiling data
└── evaluation_results.json          # Comparison report
```
