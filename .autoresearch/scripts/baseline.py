#!/usr/bin/env python3
"""Run baseline eval and initialize .ar_state.

Python replacement for baseline.sh — avoids bash-on-Windows path mangling.

Usage:
    python .autoresearch/scripts/baseline.py <task_dir> [--device-id N]
"""
import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from phase_machine import parse_last_json_line
from failure_extractor import format_for_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir")
    parser.add_argument("--device-id", default=None)
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    extra = []
    if args.device_id is not None:
        extra += ["--device-id", str(args.device_id)]

    print("[baseline] Running baseline eval...", flush=True)
    ev = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "eval_wrapper.py"), task_dir] + extra,
        capture_output=True, text=True,
    )
    if ev.stderr:
        print(ev.stderr, end="", file=sys.stderr, flush=True)

    # Echo eval_wrapper stdout EXCEPT the trailing JSON line — that line is
    # parsed below and (on failure) re-rendered via format_for_stdout, so
    # printing it raw here would duplicate the failure info as unreadable JSON.
    eval_data = parse_last_json_line(ev.stdout)
    if ev.stdout:
        lines = ev.stdout.splitlines(keepends=True)
        if eval_data is not None and lines:
            # parse_last_json_line consumes the LAST non-empty line; drop it.
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip():
                    lines.pop(i)
                    break
        if lines:
            print("".join(lines), end="", flush=True)

    if ev.returncode != 0:
        print(f"[baseline] eval_wrapper failed (rc={ev.returncode})", file=sys.stderr)
        sys.exit(ev.returncode)

    if eval_data is None:
        print("[baseline] ERROR: no JSON output from eval_wrapper", file=sys.stderr)
        sys.exit(1)

    # Pretty-print structured failure signals (UB overflow, aivec trap, OOM,
    # correctness mismatch, ...) — mirrors pipeline.py so the GENERATE_KERNEL
    # → BASELINE flow surfaces the same actionable summary the EDIT loop does.
    if not eval_data.get("correctness", False) or eval_data.get("error"):
        if eval_data.get("error"):
            print(f"[baseline] Error: {eval_data['error']}", flush=True)
        pretty = format_for_stdout(eval_data.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_data.get("raw_output_tail"):
            print("[baseline] Eval log tail (no structured signals matched):",
                  flush=True)
            print(eval_data["raw_output_tail"], flush=True)

    rc = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "_baseline_init.py"),
         task_dir, json.dumps(eval_data)],
    ).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
