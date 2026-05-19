#!/usr/bin/env python3
"""
Task directory scaffolder for Claude Code autoresearch.

Zero external dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (correctness baseline; required to import + run end-to-end
    on CPU — scaffold gates on `phase_machine.validate_reference`)
  - kernel.py (editable; --kernel writes the user file directly, otherwise
    the canonical KERNEL_PLACEHOLDER from phase_machine — the placeholder
    routes the task to GENERATE_KERNEL on first activation)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # NOTE: --devices values below are placeholders; pass the actual free
    # device id at invocation time. Earlier versions of these examples all
    # used `--devices 0`, which biased the LLM driving /autoresearch into
    # silently rewriting the user's --devices to 0 on hook-blocked retries.
    # parse_args.py is now the single source of truth for flag values.

    # Local eval (arch auto-derived via npu-smi):
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --devices <DEV>

    # With initial kernel:
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --devices <DEV>

    # Custom output directory:
    python .autoresearch/scripts/scaffold.py --ref reference.py --op-name my_op --devices <DEV> --output-dir /tmp/tasks

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
from ref_ast import validate_ref  # noqa: E402, F401  (re-export)


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def scaffold_task_dir(
    *,
    ref_code: str,
    kernel_code: str | None = None,
    op_name: str,
    desc: str = "",
    arch: str = "",
    devices: list | None = None,
    max_rounds: int = 20,
    eval_timeout: int = 120,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    code_checker_enabled: bool = True,
    ref_source_path: str | None = None,
) -> str:
    """Create task directory with all files. Returns absolute path.

    Mirrors task_scaffolder.scaffold_task_dir
    but with zero external dependency.
    """
    # Determine base directory
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(os.getcwd(), "ar_tasks")

    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write reference.py
    _write(task_dir, "reference.py", ref_code)

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

    # Write editable file (kernel.py). With no initial kernel, write the
    # canonical TODO placeholder from phase_machine — phase_machine.is_
    # placeholder_file uses the matching predicate, so the routing logic
    # in hooks/scaffold/validators stays in lockstep with this template.
    if kernel_code is not None:
        _write(task_dir, editable_filename, kernel_code)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase_machine import KERNEL_PLACEHOLDER
        _write(task_dir, editable_filename, KERNEL_PLACEHOLDER)

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
    # this field; placeholder rejection still fires either way.
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
    path hook_post_edit uses for seed commits, so reliability differences
    between Mode-1 (scaffold-time) and Mode-2 (GENERATE_KERNEL-time)
    commits are eliminated.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from git_utils import commit_in_task

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
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref", default=None,
                           help="Path to reference.py (Model/get_inputs format)")
    ref_group.add_argument("--desc", default=None,
                           help="Natural language description → LLM generates reference")
    parser.add_argument("--kernel", default=None,
                        help="Path to initial kernel file (optional, skips generation)")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (auto-derived from --desc if omitted)")
    # The repo is locked to triton_ascend on Ascend NPU + PyTorch by
    # construction. arch is derived from the picked --devices via npu-smi.
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(e.g. '5' or '0,1,2,3'). Required.")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    parser.add_argument("--no-code-checker", action="store_true",
                        help=("Disable the static Triton regression check "
                              "(validate_triton_impl) for this task. "
                              "quick_check + validate_kernel still reject "
                              "the scaffold TODO placeholder; everything "
                              "else passes through. Useful when the "
                              "regression rules are too strict for the "
                              "chosen kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    return parser


def main():
    parser = _make_arg_parser()
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hw_detect import derive_arch

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

    # Derive op-name if not provided
    if not args.op_name:
        if args.desc:
            import re as _re
            words = _re.findall(r"[a-zA-Z]+", args.desc)[:4]
            args.op_name = "_".join(w.lower() for w in words) or "custom_op"
        else:
            args.op_name = "custom_op"

    if args.ref:
        if not os.path.isfile(args.ref):
            print(json.dumps({"status": "error", "error": f"Reference file not found: {args.ref}"}))
            sys.exit(1)
        with open(args.ref, "r", encoding="utf-8") as f:
            ref_code = f.read()
        try:
            validate_ref(ref_code, args.ref)
        except ValueError as e:
            print(json.dumps({"status": "error", "error": str(e)}))
            sys.exit(1)
    else:
        # --desc mode: scaffold without reference. Claude Code fills it later.
        # Source the placeholder from phase_machine so is_placeholder_file's
        # prefix predicate stays in lockstep with what we actually write.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase_machine import REFERENCE_PLACEHOLDER_PREFIX
        ref_code = f"{REFERENCE_PLACEHOLDER_PREFIX}\n# {args.desc}\n"

    # Read initial kernel (optional)
    kernel_code = None
    if args.kernel:
        if not os.path.isfile(args.kernel):
            print(json.dumps({"status": "error", "error": f"Kernel file not found: {args.kernel}"}))
            sys.exit(1)
        with open(args.kernel, "r", encoding="utf-8") as f:
            kernel_code = f.read()

    # devices_list was resolved above.
    print(f"[scaffold] Creating task directory for {args.op_name}...", file=sys.stderr)

    task_dir = scaffold_task_dir(
        ref_code=ref_code,
        kernel_code=kernel_code,
        op_name=args.op_name,
        desc=args.desc or "",
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

    # Runnability gate: any mode that supplied a real --ref must produce a
    # reference.py that imports AND survives one Model.forward() pass on CPU.
    # The reference is the correctness baseline for every subsequent verify;
    # if it doesn't run, nothing downstream is meaningful. AST symbol presence
    # is checked earlier (see validate_ref); this catches torch import errors,
    # bad get_inputs shapes, missing ops, etc. Skipped in --desc mode where
    # reference.py is still a TODO stub.
    if args.ref:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase_machine import validate_reference
        ok, err = validate_reference(task_dir)
        if not ok:
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": f"reference.py failed runnability check: {err}",
                "hint": ("Fix the source reference file (the one passed via "
                         "--ref) and re-run /autoresearch. scaffold left the "
                         "partial task_dir in place for inspection."),
            }))
            sys.exit(2)

    if args.run_baseline and args.ref and args.kernel:
        print(f"[scaffold] Running baseline eval...", file=sys.stderr)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_cmd = [sys.executable, os.path.join(script_dir, "baseline.py"), task_dir]
        rc = subprocess.run(baseline_cmd).returncode
        if rc != 0:
            # /autoresearch reads the JSON from scaffold stdout and proceeds
            # straight to `export AR_TASK_DIR=...`; if baseline failed but we
            # still printed status=ok, the slash command would resume as if
            # the task were in PLAN. Surface the failure so the caller stops
            # and surfaces it to the user instead.
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": (f"baseline eval failed (exit {rc}); "
                          f"see [baseline]/[eval] stderr above"),
                "hint": ("Inspect kernel.py / reference.py / eval stderr, "
                         "fix, then re-run: "
                         f"python .autoresearch/scripts/baseline.py "
                         f"\"{task_dir}\""),
            }))
            sys.exit(3)
    elif args.run_baseline:
        print(f"[scaffold] --run-baseline skipped: kernel.py not provided. "
              f"GENERATE_KERNEL phase will produce it; baseline runs after that.",
              file=sys.stderr)

    # Output
    print(json.dumps({"task_dir": task_dir, "status": "ok"}))


if __name__ == "__main__":
    main()
