#!/usr/bin/env python3
"""CLI shell for `workflow.round.record_round`.

Usage:
    python keep_or_discard.py <task_dir> <eval_json>
    python keep_or_discard.py <task_dir> --eval-file <path>

Output (stdout, last line):
    {"decision": "KEEP", "best_metric": 145.3, "eval_rounds": 6,
     "consecutive_failures": 0}

The body lives in workflow/round.py so pipeline.py can call the same
logic in-process (no subprocess + last-line-JSON parse).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from workflow import record_round


def main():
    parser = argparse.ArgumentParser(description="Keep or discard eval result")
    parser.add_argument("task_dir", help="Path to task directory")
    parser.add_argument("eval_json", nargs="?",
                        help="Eval result as JSON string")
    parser.add_argument("--eval-file",
                        help="Path to file containing eval JSON")
    parser.add_argument("--description", default="optimization round",
                        help="Round description")
    parser.add_argument("--plan-item", default=None,
                        help="Plan item id (pN) this round settles")
    args = parser.parse_args()

    if args.eval_file:
        with open(args.eval_file, "r") as f:
            eval_data = json.load(f)
    elif args.eval_json:
        eval_data = json.loads(args.eval_json)
    else:
        print(json.dumps({"decision": "ERROR",
                          "error": "No eval result provided"}))
        sys.exit(1)

    out = record_round(
        os.path.abspath(args.task_dir),
        eval_data,
        description=args.description,
        plan_item=args.plan_item,
    )
    print(json.dumps(out))
    sys.exit(0 if out.get("decision") != "ERROR" else 1)


if __name__ == "__main__":
    main()
