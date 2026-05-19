"""Round-0 (SEED) eval recorder.

Lifted from `_baseline_init.py` so the body is reusable from other
entry points (notebook re-runs, future scaffold one-shots) without
re-invoking the CLI shell. The shell `_baseline_init.py` stays as a
thin wrapper so existing `python _baseline_init.py <td> <eval_json>`
call sites keep working.

Exit codes returned by `run_baseline_init`:
    0  baseline OK (correctness + seed_metric valid; phase -> PLAN)
    2  seed produced no valid primary metric
    3  baseline correctness=False
    1  task.yaml not found / unparseable
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, load_progress, save_progress,
)
from task_config import load_task_config  # noqa: E402

from .transition import PhaseController


def _valid(v) -> bool:
    return isinstance(v, (int, float)) and 0 < v < float("inf")


def _git_short_head(task_dir: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=task_dir, capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_baseline_init(task_dir: str, eval_json: str) -> int:
    """Library entry point. CLI shell `_baseline_init.py` calls this with
    sys.argv[2]. Returns process exit code. Side effects (progress,
    history, phase) are durable on disk before this returns."""
    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found", file=sys.stderr)
        return 1

    eval_data = json.loads(eval_json)
    correctness = eval_data.get("correctness", False)
    metrics = eval_data.get("metrics", {})
    seed_val = metrics.get(config.primary_metric) if _valid(
        metrics.get(config.primary_metric)) else None
    ref_val = metrics.get("ref_latency_us") if _valid(
        metrics.get("ref_latency_us")) else None

    # Sticky baseline (see _baseline_init history): pin first ref capture
    # so SEED retries don't drift speedup anchor.
    existing = load_progress(task_dir) or Progress()
    existing_baseline = existing.baseline_metric
    existing_source = existing.baseline_source
    existing_baseline_commit = existing.baseline_commit

    if _valid(existing_baseline) and existing_source == "ref":
        baseline_val = existing_baseline
        baseline_source = "ref"
        print(f"[baseline] sticky baseline = {existing_baseline} "
              f"(kept from first eval; this round's ref={ref_val} ignored)",
              file=sys.stderr)
    elif ref_val is not None:
        baseline_val = ref_val
        baseline_source = "ref"
        existing_baseline_commit = None
        print(f"[baseline] baseline = ref_latency_us = {ref_val} "
              f"(PyTorch reference)", file=sys.stderr)
    else:
        baseline_val = seed_val
        baseline_source = "seed_fallback"
        existing_baseline_commit = None
        print("[baseline] WARNING: ref_latency_us missing - baseline falls "
              "back to seed metric", file=sys.stderr)

    # initial_best stays None when seed didn't profile (see B1 in audit
    # history). keep_or_discard's `if best is None: KEEP` then accepts the
    # first real kernel timing without comparing against a fake anchor.
    initial_best = seed_val
    baseline_commit = _git_short_head(task_dir)
    baseline_commit_recorded = existing_baseline_commit or baseline_commit

    n_cases = metrics.get("num_cases")
    descs = metrics.get("per_shape_descs")
    save_progress(task_dir, Progress(
        task=config.name,
        eval_rounds=0,
        max_rounds=config.max_rounds,
        best_metric=initial_best,
        best_commit=(baseline_commit if seed_val is not None
                     else "seed_profile_failed"),
        baseline_commit=baseline_commit_recorded,
        baseline_metric=baseline_val,
        baseline_source=baseline_source,
        baseline_correctness=correctness,
        seed_metric=seed_val,
        consecutive_failures=0,
        plan_version=0,
        status="no_plan",
        num_cases=(int(n_cases) if isinstance(n_cases, int)
                   and n_cases >= 1 else None),
        per_shape_descs=(
            [str(d) for d in descs if d]
            if isinstance(descs, list) and descs else None
        ),
    ), stamp=True)

    append_history(task_dir, {
        "round": 0,
        "description": "seed kernel initial eval",
        "decision": "SEED",
        "metrics": metrics,
        "correctness": correctness,
        "commit": baseline_commit,
    })

    if not correctness:
        print(
            "[baseline] ERROR: baseline eval failed correctness check.\n"
            f"[baseline] seed {config.primary_metric}={seed_val} but output "
            "did not match reference. Phase stays at BASELINE; "
            "hook_post_bash will demote to GENERATE_KERNEL for a fix.",
            file=sys.stderr,
        )
        return 3

    if seed_val is None:
        print(
            f"[baseline] ERROR: seed kernel produced no valid "
            f"{config.primary_metric}.\n"
            f"[baseline] profile_{config.name}_generation.py ran but no "
            f"timing came back (result JSON missing, inf, or 0).\n"
            f"[baseline] Check eval logs, fix kernel.py, rerun baseline.",
            file=sys.stderr,
        )
        return 2

    PhaseController(task_dir).on_baseline_init_success()
    print(f"[baseline] Initialized: task={config.name}, "
          f"seed_{config.primary_metric}={seed_val}, "
          f"baseline({baseline_source})={baseline_val}, "
          f"commit={baseline_commit_recorded}", file=sys.stderr)
    print("[baseline] Phase -> PLAN", file=sys.stderr)
    return 0
