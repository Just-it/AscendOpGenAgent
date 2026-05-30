"""Round-N (post-EDIT) eval recorder. `record_round` is called in-process
by engine/pipeline.py and returns {decision, best_metric, eval_rounds,
max_rounds, consecutive_failures}."""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, auto_rollback, load_progress, save_progress,
)
from task_config import (  # noqa: E402
    EvalOutcome, EvalResult, check_constraints, is_improvement,
    load_task_config,
)
from utils.git_utils import commit_in_task, current_head_short  # noqa: E402

from .progress_reducer import eval_result_from_data, reduce_round_progress


def record_round(task_dir: str, eval_data: dict,
                 description: str = "optimization round",
                 plan_item: Optional[str] = None) -> dict:
    """Single library entry point for one round of EDIT settlement.

    Decision flow: correctness gate -> constraint gate -> primary-metric
    presence -> improvement check."""
    config = load_task_config(task_dir)
    if config is None:
        return {"decision": "ERROR", "error": "task.yaml not found"}

    progress = load_progress(task_dir) or Progress()
    eval_result = eval_result_from_data(eval_data)

    round_num = progress.eval_rounds + 1
    decision = "DISCARD"
    commit_hash: Optional[str] = None
    new_failures = progress.consecutive_failures
    new_best_metric = progress.best_metric
    new_best_commit = progress.best_commit

    if not eval_result.correctness:
        decision = "FAIL"
        new_failures = progress.consecutive_failures + 1
        print("[record_round] FAIL: correctness check failed",
              file=sys.stderr)
    else:
        violations = (check_constraints(eval_result, config.constraints)
                      if config.constraints else [])
        if violations:
            decision = "FAIL"
            new_failures = progress.consecutive_failures + 1
            print(f"[record_round] FAIL: constraint violations: "
                  f"{violations}", file=sys.stderr)
        else:
            cur = eval_result.metrics.get(config.primary_metric)
            best = progress.best_metric
            if (not isinstance(cur, (int, float))
                    or cur != cur):  # NaN guard
                decision = "FAIL"
                new_failures = progress.consecutive_failures + 1
                print(f"[record_round] FAIL: correctness=PASS but primary "
                      f"metric '{config.primary_metric}' missing from "
                      f"{sorted(eval_result.metrics)}", file=sys.stderr)
            elif best is None:
                decision = "KEEP"
            else:
                best_er = EvalResult(outcome=EvalOutcome.OK,
                                     metrics={config.primary_metric: best})
                if is_improvement(
                    eval_result, best_er,
                    metric=config.primary_metric,
                    lower_is_better=config.lower_is_better,
                    threshold=config.improvement_threshold,
                ):
                    decision = "KEEP"
                else:
                    decision = "DISCARD"

    if decision == "KEEP":
        metric_val = eval_result.metrics.get(config.primary_metric)
        metric_str = f"{config.primary_metric}={metric_val}"
        ok, info = commit_in_task(
            task_dir, config.editable_files,
            f"autoresearch: {description} | {metric_str}",
        )
        if not ok:
            # Couldn't preserve kernel state. Earlier we still wrote
            # best_metric=<this round's value> and best_commit=None,
            # which left progress.json pointing at a kernel that no
            # commit captured - rollback / resume / report all became
            # unreliable. Demote to FAIL: roll the working tree back,
            # bump consecutive_failures, leave best_* untouched.
            print(f"[record_round] git commit failed: {info}; demoting "
                  f"KEEP -> FAIL (kernel state not preserved)",
                  file=sys.stderr)
            decision = "FAIL"
            new_failures = progress.consecutive_failures + 1
            auto_rollback(task_dir)
        else:
            # "noop" means the edit produced no git-visible diff (e.g.
            # whitespace-only change, or a roll-back to an existing
            # commit's bytes). The kernel we just evaluated IS what HEAD
            # points at, so resolve commit_hash to HEAD instead of None
            # — otherwise a noisier rerun of an existing best would
            # advance best_metric while nulling best_commit, leaving
            # dashboard / report unable to retrieve the winning kernel.
            if info == "noop":
                commit_hash = current_head_short(task_dir) or progress.best_commit
            else:
                commit_hash = info
            new_best_metric = metric_val
            new_best_commit = commit_hash
            new_failures = 0
            print(f"[record_round] KEEP: {metric_str} "
                  f"(commit: {commit_hash})", file=sys.stderr)
    else:
        auto_rollback(task_dir)
        print(f"[record_round] {decision}: rolled back editable files",
              file=sys.stderr)

    # Keep baseline ownership centralized with baseline_init. This covers
    # missing anchors, seed_fallback upgrades, and fingerprint re-anchors.
    reduction = reduce_round_progress(
        progress, eval_result, round_num,
        consecutive_failures=new_failures,
        best_metric=new_best_metric,
        best_commit=new_best_commit,
    )
    if reduction.anchor.changed and reduction.anchor.message:
        print(f"[record_round] {reduction.anchor.message} from R{round_num}",
              file=sys.stderr)

    progress = reduction.progress
    save_progress(task_dir, progress)

    hist: dict[str, Any] = {
        "round": round_num,
        "plan_item": plan_item,
        "description": description,
        "decision": decision,
        "metrics": eval_result.metrics,
        "correctness": eval_result.correctness,
        "error": eval_result.error,
        "commit": commit_hash,
    }
    # Only FAIL rows carry the failure_signals + raw tail; KEEP/DISCARD
    # already passed correctness, attaching them is noise.
    if decision == "FAIL":
        sig = eval_data.get("failure_signals")
        if isinstance(sig, dict) and (sig.get("primary")
                                      or sig.get("python_error")
                                      or sig.get("signals")):
            hist["failure_signals"] = sig
        tail = (eval_data.get("raw_output_tail") or "").strip()
        if tail:
            hist["raw_output_tail"] = tail[-1500:]
    append_history(task_dir, hist)

    return {
        "decision": decision,
        "best_metric": progress.best_metric,
        "eval_rounds": round_num,
        "max_rounds": progress.max_rounds or config.max_rounds,
        "consecutive_failures": progress.consecutive_failures,
    }
