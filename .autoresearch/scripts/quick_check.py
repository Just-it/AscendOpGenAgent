#!/usr/bin/env python3
"""
Quick static check for editable files before eval.

Delegates to `validate_triton_impl.validate` (in this same scripts/ dir —
copy of the kernel-verifier skill's regression checker). It catches the
three Triton-regression patterns before we pay the cost of a real eval:

  Type 1: no @triton.jit kernel at all (pure PyTorch impl)
  Type 2: kernel defined but ModelNew.forward() never launches it
  Type 3: forward() launches the kernel but still does compute via
          torch.* / F.* / tensor methods / @ operator / Python loops

Honors `config.code_checker_enabled` — when off (task.yaml
`code_checker.enabled: false` or scaffold's `--no-code-checker`), only
file existence + the optional smoke test run. The flag name predates
this rewrite; it toggles the static check itself, not any specific
implementation.

Usage:
    python .autoresearch/scripts/quick_check.py <task_dir>

Output:
    stdout: 'OK' on pass, JSON error blob on fail
    exit 0 = pass, 1 = fail
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config
from validate_triton_impl import validate as validate_triton_impl


def _format_regression_report(result: dict) -> str:
    """Render the validate_triton_impl result dict as a markdown report
    for guidance text / hook stderr."""
    rtype = result.get("regression_type")
    type_desc = {
        1: "no @triton.jit kernel (pure PyTorch implementation)",
        2: "kernel defined but ModelNew.forward() never launches it",
        3: "forward() still uses PyTorch for part of the compute",
    }.get(rtype, "(unknown regression type)")

    lines = [
        f"## Triton regression check failed — Type {rtype}: {type_desc}",
        "",
    ]
    for name, sub in result.get("checks", {}).items():
        status = "PASS" if sub.get("passed") else "FAIL"
        lines.append(f"### {name}: {status}")
        if sub.get("error"):
            lines.append(f"  {sub['error']}")
        if name == "no_forbidden_torch_ops":
            for v in sub.get("violations", []) or []:
                lines.append(
                    f"  - line {v.get('line', '?')}: {v.get('call', '?')} — "
                    f"{v.get('reason', '')}"
                )
        lines.append("")
    if result.get("suggestion"):
        lines.append("**Suggestion:**")
        lines.append(f"  {result['suggestion']}")
    return "\n".join(lines)


def _regression_to_errors(result: dict) -> list:
    """Flatten the validation dict into the {line, error_type, detail,
    suggestion, code_snippet} shape that history.jsonl / dashboard
    consume."""
    errors = []
    rtype = result.get("regression_type")
    for name, sub in result.get("checks", {}).items():
        if sub.get("passed"):
            continue
        if sub.get("error"):
            errors.append({
                "line": 0,
                "error_type": f"regression_type_{rtype}_{name}",
                "detail": sub["error"],
                "suggestion": result.get("suggestion", ""),
                "code_snippet": "",
            })
        for v in sub.get("violations", []) or []:
            errors.append({
                "line": v.get("line", 0),
                "error_type": f"regression_type_{rtype}_{name}",
                "detail": f"{v.get('call', '?')} — {v.get('reason', '')}",
                "suggestion": result.get("suggestion", ""),
                "code_snippet": "",
            })
    return errors


def check_editable_files(task_dir: str, config) -> list:
    """Run validate_triton_impl on every editable .py.

    Honors `config.code_checker_enabled` — when off, only the
    file-existence check fires; the AST regression check is skipped.
    This is the single gate consulted by both the runtime quick check
    and `phase_machine.validate_kernel`. Public lib API (no leading
    underscore): both the CLI `main()` below and
    `validators.validate_kernel` call this directly; do not duplicate.
    """
    issues = []
    use_checker = config.code_checker_enabled
    for fname in config.editable_files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(task_dir, fname)
        if not os.path.exists(fpath):
            issues.append({"file": fname, "report": "file not found", "errors": []})
            continue
        if not use_checker:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
        result = validate_triton_impl(code, filepath=fpath)
        if not result.get("valid"):
            issues.append({
                "file": fname,
                "report": _format_regression_report(result),
                "errors": _regression_to_errors(result),
            })
    return issues


def _run_smoke_test(task_dir: str, config) -> list:
    if not config.smoke_test_script:
        return []
    smoke_path = os.path.join(task_dir, config.smoke_test_script)
    if not os.path.exists(smoke_path):
        return []
    try:
        r = subprocess.run(
            [sys.executable, smoke_path],
            capture_output=True, text=True,
            timeout=config.smoke_test_timeout, cwd=task_dir,
        )
    except subprocess.TimeoutExpired:
        return [f"smoke test timed out after {config.smoke_test_timeout}s"]
    except Exception as e:
        return [f"smoke test launch error: {e}"]
    if r.returncode != 0:
        tail = (r.stderr or "")[-500:]
        return [f"smoke test failed (exit {r.returncode}): {tail}"]
    return []


def main():
    parser = argparse.ArgumentParser(description="Quick static check")
    parser.add_argument("task_dir", help="Path to task directory")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({"ok": False, "error": "task.yaml not found"}))
        sys.exit(1)

    if not config.code_checker_enabled:
        print("[quick_check] Triton regression check disabled in task.yaml — "
              "only file-existence and smoke test will run.", file=sys.stderr)

    file_issues = check_editable_files(task_dir, config)
    smoke_errors = _run_smoke_test(task_dir, config)

    if not file_issues and not smoke_errors:
        print("OK")
        sys.exit(0)

    blob = {"ok": False}
    if file_issues:
        blob["file_issues"] = file_issues
    if smoke_errors:
        blob["smoke_errors"] = smoke_errors
    print(json.dumps(blob, ensure_ascii=False))
    sys.exit(1)


if __name__ == "__main__":
    main()
