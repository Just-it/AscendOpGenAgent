#!/usr/bin/env python3
"""
Task directory scaffolder for Claude Code autoresearch.

Zero external dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (correctness baseline; AST-checked via utils.ref_ast.
    validate_ref before scaffold copies it. Runtime correctness is
    validated by --run-baseline whose verify routine tags error_source.)
  - kernel.py (editable seed; written from the user's --kernel file)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # NOTE: --devices values below are placeholders; pass the actual free
    # device id at invocation time.

    # Local eval (arch auto-derived via npu-smi):
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --devices <DEV>

    # Custom output directory:
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --devices <DEV> --output-dir /tmp/tasks

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import yaml


# ---------------------------------------------------------------------------
# Reference validation — delegated to the standalone library module so
# phase_machine.validators can call the same rule without importing this
# CLI script. The local re-export keeps callers that imported
# `scaffold.validate_ref` working.
# ---------------------------------------------------------------------------
from utils.ref_ast import validate_ref  # noqa: E402, F401  (re-export)


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def scaffold_task_dir(
    *,
    ref_code: str,
    kernel_code: str,
    op_name: str,
    desc: str = "",
    arch: str = "",
    devices: list | None = None,
    max_rounds: int = 20,
    eval_timeout: int = 600,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    code_checker_enabled: bool = True,
    ref_source_path: str | None = None,
) -> str:
    """Create task directory with all files. Returns absolute path."""
    # Determine base directory
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(os.getcwd(), "ar_tasks")

    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write reference.py and the seed kernel.py from the user's files.
    _write(task_dir, "reference.py", ref_code)
    _write(task_dir, editable_filename, kernel_code)

    # NPUKernelBench-style refs read shape lists from a sibling JSON via
    # `os.path.join(os.path.dirname(__file__), "<basename>.json")`. Copy
    # any *.json file in the source ref's directory into task_dir,
    # preserving names — the .py expects the JSON at task_dir at runtime
    # (dirname(__file__) becomes task_dir after the rename).
    if ref_source_path:
        try:
            import shutil as _shutil
            ref_dir_src = os.path.dirname(os.path.abspath(ref_source_path))
            for fname in os.listdir(ref_dir_src):
                if not fname.endswith(".json"):
                    continue
                src = os.path.join(ref_dir_src, fname)
                if not os.path.isfile(src):
                    continue
                _shutil.copy(src, os.path.join(task_dir, fname))
        except Exception as _e:
            print(f"[scaffold] WARNING: sidecar JSON copy failed: {_e}",
                  file=sys.stderr)

    # Generate task.yaml — only fields that vary per-task. dsl /
    # framework / backend are constants (triton_ascend / torch / ascend)
    # baked into TaskConfig; not written here.
    task_yaml = {
        "name": op_name,
        "description": desc or f"Optimize {op_name}",
        "arch": arch or None,
        "editable_files": [editable_filename],
        "eval": {
            "timeout": eval_timeout,
        },
        "metric": {
            "primary": "latency_us",
            "lower_is_better": True,
        },
        "agent": {
            "ref_file": "reference.py",
            "max_rounds": max_rounds,
        },
    }
    if devices:
        task_yaml["devices"] = list(devices)

    # Only emit the code_checker block when disabled — default-true tasks
    # stay clean. quick_check.py and phase_machine.validate_kernel honor
    # this field.
    if not code_checker_enabled:
        task_yaml["code_checker"] = {"enabled": False}

    yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
    _write(task_dir, "task.yaml", yaml_content)

    # Create .ar_state directory
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    # Git init + baseline commit
    _git_init(task_dir)

    return os.path.abspath(task_dir)


def _write(task_dir: str, rel_path: str, content: str):
    full_path = os.path.join(task_dir, rel_path)
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def _git_init(task_dir: str):
    """Initialize git repo and create baseline commit.

    The actual commit goes through git_utils.commit_in_task — same code
    path hooks use for round commits, so reliability is consistent.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.git_utils import commit_in_task

    subprocess.run(["git", "init"], cwd=task_dir, capture_output=True, check=True)
    ok, info = commit_in_task(task_dir, ["."], "scaffold: baseline")
    if not ok:
        raise RuntimeError(f"scaffold baseline commit failed: {info}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    """Construct scaffold's argparse, with no side effects.

    Extracted out of main() so parse_args.py can reuse the exact same flag
    spec without duplicating it. Single source of truth for which flags
    /autoresearch accepts and how they're typed/defaulted.
    """
    parser = argparse.ArgumentParser(
        description="Scaffold a task directory for Claude Code autoresearch",
    )
    parser.add_argument("--ref", required=True,
                        help="Path to reference.py (Model/get_inputs format)")
    parser.add_argument("--kernel", required=True,
                        help="Path to seed kernel file")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (required)")
    # The repo is locked to triton_ascend on Ascend NPU + PyTorch by
    # construction. arch is derived from the picked --devices via npu-smi.
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(e.g. '5' or '0,1,2,3'). Required.")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    parser.add_argument("--no-code-checker", action="store_true",
                        help=("Disable the static Triton regression check "
                              "(validate_triton_impl) for this task. "
                              "Useful when the regression rules are too "
                              "strict for the chosen kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    return parser


def main():
    parser = _make_arg_parser()
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.hw_detect import derive_arch

    # Hardware resolution: --devices is required (local-only). The repo
    # is locked to triton_ascend / torch / ascend by construction —
    # those constants live in TaskConfig defaults / generated templates,
    # not on `args`.
    devices_list: list = []
    args.arch = None

    if not args.devices:
        print(json.dumps({"status": "error",
                          "error": "--devices is required (local eval)."}))
        sys.exit(1)

    devices_list = [int(d.strip()) for d in args.devices.split(",")
                    if d.strip()]
    if not devices_list:
        print(json.dumps({"status": "error",
                          "error": "--devices parsed to an empty list"}))
        sys.exit(1)
    args.arch = derive_arch(devices_list[0])
    if not args.arch:
        print(json.dumps({"status": "error",
                          "error": (f"could not derive arch from "
                                    f"device {devices_list[0]} "
                                    f"(is npu-smi on PATH?)")}))
        sys.exit(1)

    if not args.op_name:
        print(json.dumps({"status": "error",
                          "error": "--op-name is required"}))
        sys.exit(1)

    if not os.path.isfile(args.ref):
        print(json.dumps({"status": "error",
                          "error": f"Reference file not found: {args.ref}"}))
        sys.exit(1)
    with open(args.ref, "r", encoding="utf-8") as f:
        ref_code = f.read()
    try:
        validate_ref(ref_code, args.ref)
    except ValueError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    if not os.path.isfile(args.kernel):
        print(json.dumps({"status": "error",
                          "error": f"Kernel file not found: {args.kernel}"}))
        sys.exit(1)
    with open(args.kernel, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    # devices_list was resolved above.
    print(f"[scaffold] Creating task directory for {args.op_name}...", file=sys.stderr)

    task_dir = scaffold_task_dir(
        ref_code=ref_code,
        kernel_code=kernel_code,
        op_name=args.op_name,
        devices=devices_list,
        arch=args.arch,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        code_checker_enabled=not args.no_code_checker,
        ref_source_path=args.ref,
    )

    print(f"[scaffold] Task directory created: {task_dir}", file=sys.stderr)
    print(f"[scaffold] Files:", file=sys.stderr)
    for f in sorted(os.listdir(task_dir)):
        print(f"  {f}", file=sys.stderr)

    # Write per-op pointer so batch/run.py picks the exact dir we just
    # made, not whichever <op>_* in ar_tasks/ happens to have the freshest
    # mtime (which races with concurrent runs and stale prior task_dirs).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phase_machine import write_task_dir_pointer
    write_task_dir_pointer(args.op_name, task_dir)

    # Reference validation is now a single path through baseline.py: the
    # generated verify routine splits ref-side and kernel-side try/excepts
    # and tags error_source on failure. Scaffold reads the resulting
    # baseline exit code and decides:
    #   - exit 4 (INFRA_FAIL) → reject task (operator must fix --ref or env)
    #   - any other non-zero → kernel-side failure, task activates and
    #     hook routes to PLAN
    # AST symbol presence was already checked earlier (validate_ref on
    # the source --ref file before copying), so import errors / missing
    # symbols never reach this point.
    if args.run_baseline:
        print(f"[scaffold] Running baseline eval...", file=sys.stderr)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_cmd = [sys.executable,
                        os.path.join(script_dir, "engine", "baseline.py"),
                        task_dir]
        rc = subprocess.run(baseline_cmd).returncode
        # baseline exit codes are binary now (workflow.baseline._EXIT_FOR):
        #   0 = task activatable (OK or KERNEL_FAIL — hook routes to PLAN)
        #   4 = task NOT activatable (INFRA_FAIL — operator must intervene)
        # Anything else here is an unexpected baseline crash.
        if rc == 4:
            try:
                with open(os.path.join(task_dir, ".ar_state",
                                       "progress.json")) as _f:
                    err_source = json.load(_f).get("baseline_error_source")
            except (OSError, ValueError):
                err_source = None
            if err_source == "ref":
                hint = ("The file passed via --ref is broken (import / "
                        "forward / device-only bug). Fix the SOURCE file "
                        "and re-run /autoresearch from scratch. The task "
                        "directory is left for inspection but MUST NOT be "
                        "activated — reference.py is not editable.")
            else:
                hint = ("INFRA_FAIL: no per-shape data — the seed kernel "
                        "wasn't meaningfully exercised. Fix env (device / "
                        "eval.timeout / worker / OOM) and re-run "
                        "`/autoresearch --resume <task_dir>`. Phase stays "
                        "at BASELINE.")
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": ("eval pipeline broken during baseline — see "
                          "[baseline]/[eval] stderr above"),
                "hint": hint,
            }))
            sys.exit(4)
        if rc != 0:
            # Unexpected baseline crash (not the 0/4 we know about).
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": (f"baseline crashed unexpectedly (exit {rc}); "
                          f"see [baseline]/[eval] stderr above"),
                "hint": ("This is not a classified outcome. Inspect the "
                         "baseline / eval stderr above and file a bug if "
                         "the exit code isn't in _EXIT_FOR."),
            }))
            sys.exit(rc)

    # Output
    # Surface baseline_outcome so callers can distinguish OK from
    # KERNEL_FAIL without re-reading progress.json. Both are activatable
    # (status=ok, rc=0); the difference is whether the seed kernel
    # produced valid timings or the first PLAN cycle has to rewrite it.
    outcome = None
    try:
        with open(os.path.join(task_dir, ".ar_state",
                               "progress.json")) as _f:
            outcome = json.load(_f).get("baseline_outcome")
    except (OSError, ValueError):
        pass
    print(json.dumps({"task_dir": task_dir, "status": "ok",
                      "baseline_outcome": outcome}))


if __name__ == "__main__":
    main()
