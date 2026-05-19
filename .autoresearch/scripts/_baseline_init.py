#!/usr/bin/env python3
"""CLI shell — see workflow.baseline.run_baseline_init for the body.

Called by baseline.py:
    python _baseline_init.py <task_dir> <eval_json>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from workflow import run_baseline_init


if __name__ == "__main__":
    sys.exit(run_baseline_init(sys.argv[1], sys.argv[2]))
