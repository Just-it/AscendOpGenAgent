#!/usr/bin/env python3
"""
Eval wrapper for Claude Code autoresearch.

Local-only. Uses task_config.run_eval to materialize the eval dir
and dispatch verify/profile subprocesses.

Usage:
    python .autoresearch/scripts/eval_wrapper.py <task_dir> [--device-id N]

Output (last line of stdout):
    {"correctness": true, "metrics": {"latency_us": 145.3}, "error": null}
"""

import argparse
import json
import os
import sys

# Import from sibling module
sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config, run_eval, format_result_summary
from failure_extractor import extract_failure_signals


def main():
    parser = argparse.ArgumentParser(description="Run eval and output JSON")
    parser.add_argument("task_dir", help="Path to the task directory")
    parser.add_argument("--device-id", type=int, default=None, help="Device ID")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)

    config = load_task_config(task_dir)
    if config is None:
        print(json.dumps({
            "correctness": False,
            "metrics": {},
            "error": f"task.yaml not found in {task_dir}",
        }))
        sys.exit(0)

    # Probe the Ascend runtime up front so the log line tells the user
    # exactly why this run will / won't proceed.
    from eval_runner import detect_local_backend
    ok, why = detect_local_backend()
    status = "available" if ok else "UNAVAILABLE"
    print(f"[eval] Running eval for {config.name} "
          f"— ascend runtime {status}: {why}", file=sys.stderr)

    result = run_eval(task_dir, config, device_id=args.device_id)

    print(f"[eval] {format_result_summary(result)}", file=sys.stderr)

    output = {
        "correctness": result.correctness,
        "metrics": result.metrics,
        "error": result.error,
    }
    # Attach structured failure signals only when something went wrong — on
    # success the raw log is noisy and adds no value to downstream consumers.
    # `raw_output` is already capped upstream (4 KB tail in task_config), so
    # we forward it whole — the pipeline falls back to printing it verbatim
    # when no pattern matched.
    if not result.correctness or result.error:
        output["failure_signals"] = extract_failure_signals(result.raw_output).to_dict()
        output["raw_output_tail"] = result.raw_output or ""
    print(json.dumps(output))


if __name__ == "__main__":
    main()
