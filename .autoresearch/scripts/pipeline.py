#!/usr/bin/env python3
"""
Post-edit pipeline — runs ALL mechanical steps after Claude Code edits code.

Claude Code does the LLM work (plan, edit, diagnose). Then calls this:
    python .autoresearch/scripts/pipeline.py <task_dir>

This script does:
    1. quick_check → fail? rollback, report
    2. eval → get metrics
    3. keep_or_discard → KEEP/DISCARD/FAIL
    4. settle → update plan.md, advance (ACTIVE)
    5. compute next phase → write .phase
    6. print status + next guidance

Output: human-readable status to stdout. Claude Code sees it and acts accordingly.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task_config import load_task_config
from failure_extractor import format_for_stdout
from workflow import PhaseController, record_round
from phase_machine import (
    get_active_item,
    get_guidance, auto_rollback, load_progress, edit_marker_path,
    pending_settle_path, parse_last_json_line, FINISH,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_settle(task_dir: str, kd_json: dict) -> tuple:
    """Invoke settle.py with the given kd_json.

    Returns (rc, stdout_tail, stderr_tail, settle_json):
      - rc:           settle.py exit code
      - stdout_tail:  last 400 chars of stdout (for error reports)
      - stderr_tail:  last 400 chars of stderr
      - settle_json:  parsed last-JSON-line from stdout, or None on
                      parse failure / non-zero rc. Carries the
                      `settled_item` id, which the caller needs for the
                      status report — `get_active_item()` AFTER settle
                      points at the NEXT ACTIVE item, not the one we
                      just settled.
    """
    settle = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "settle.py"),
         task_dir, json.dumps(kd_json)],
        capture_output=True, text=True, timeout=10,
    )
    settle_json = parse_last_json_line(settle.stdout) if settle.returncode == 0 else None
    return (
        settle.returncode,
        (settle.stdout or "").strip()[-400:],
        (settle.stderr or "").strip()[-400:],
        settle_json,
    )


def _persist_pending_settle(task_dir: str, kd_json: dict) -> None:
    path = pending_settle_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kd_json, f)


def _clear_pending_settle(task_dir: str) -> None:
    path = pending_settle_path(task_dir)
    if os.path.exists(path):
        os.remove(path)


def _emit_settle_failure(task_dir: str, rc: int, tail_out: str,
                         tail_err: str) -> None:
    print(f"[PIPELINE] SETTLE FAILED (rc={rc}). plan.md was NOT updated. "
          f"progress.json + history.jsonl already moved during this round; "
          f"re-running this script will RETRY SETTLE ONLY (kd_json was "
          f"persisted to .ar_state/.pending_settle.json) — it will NOT "
          f"re-run quick_check/eval/keep_or_discard.\n"
          f"\n"
          f"Recovery options (do NOT hand-edit plan.md):\n"
          f"  1. Fix the underlying cause from the stderr tail below, "
          f"then re-run pipeline.py — the replay-only path will retry "
          f"settle on the same kd_json.\n"
          f"  2. If the failure is structural (plan.md malformed, no "
          f"(ACTIVE) item, etc.) and settle cannot recover, run "
          f"create_plan.py to write a fresh plan.md. While "
          f"pending_settle.json exists, hook_guard_bash allows "
          f"create_plan.py in EDIT phase as a recovery path; on "
          f"successful create_plan validation hook_post_bash clears "
          f"pending_settle.json. The orphan history.jsonl row stays "
          f"(audit trail) but no longer corresponds to any plan item.\n"
          f"\n"
          f"stdout tail: {tail_out}\n"
          f"stderr tail: {tail_err}", file=sys.stderr)


def _post_settle(task_dir: str, decision: str, settled_id: str) -> None:
    """Common path after a successful settle: advance phase, clear edit
    marker, print status. Runs whether settle succeeded the first time or
    on the replay-only retry."""
    next_phase = PhaseController(task_dir).on_round_settled()
    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        os.remove(marker)

    # FINISH is a one-way terminal transition — generate the deterministic
    # report.md (summary tables + inline SVG curve) here so it's on disk
    # before the FINISH guidance announces its path.
    if next_phase == FINISH:
        try:
            from report import write_report
            rp = write_report(task_dir)
            if rp:
                print(f"[PIPELINE] Report written: "
                      f"{os.path.relpath(rp, task_dir)}")
        except Exception as e:
            print(f"[PIPELINE] Report generation failed: {e}",
                  file=sys.stderr)

    progress = load_progress(task_dir) or {}
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)

    improv = ""
    if (
        best is not None and baseline is not None
        and isinstance(best, (int, float))
        and isinstance(baseline, (int, float))
        and baseline != 0 and best != 0
    ):
        pct = (baseline - best) / abs(baseline) * 100
        speedup = baseline / best
        improv = f" ({speedup:.2f}x vs ref, {pct:+.1f}%)"

    print(f"\n{'=' * 50}")
    print(f"[{decision}] {settled_id} | Round {rounds}/{max_rounds} | "
          f"Best: {best}{improv} | Failures: {failures}")
    print(f"Phase -> {next_phase}")
    print(f"{'=' * 50}")
    print(get_guidance(task_dir))


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <task_dir>")
        sys.exit(1)

    task_dir = os.path.abspath(sys.argv[1])

    # === Replay-only settle ===
    # If a previous pipeline.py invocation got past keep_or_discard but
    # settle.py failed, the kd_json was persisted to .pending_settle.json.
    # Re-running pipeline.py from scratch would re-eval and double-write
    # progress/history; instead, we ONLY retry settle here. Fix the
    # underlying cause (the agent saw the failure reason in stderr), then
    # invoke pipeline.py — same command, no flags — and this branch handles
    # the retry deterministically. Lives BEFORE task.yaml load so retry
    # works even if task config has drifted (settle only touches .ar_state).
    pending_path = pending_settle_path(task_dir)
    if os.path.exists(pending_path):
        try:
            with open(pending_path, "r", encoding="utf-8") as f:
                kd_json = json.load(f)
        except Exception as e:
            print(f"[PIPELINE] pending settle file unreadable ({e}). "
                  f"Removing it and bailing — please re-run pipeline.py "
                  f"to start a fresh round.", file=sys.stderr)
            _clear_pending_settle(task_dir)
            sys.exit(1)
        print(f"[PIPELINE] Retrying settle from {os.path.basename(pending_path)} "
              f"(skipping quick_check/eval/keep_or_discard).", flush=True)
        rc, tail_out, tail_err, settle_json = _run_settle(task_dir, kd_json)
        if rc != 0:
            _emit_settle_failure(task_dir, rc, tail_out, tail_err)
            sys.exit(1)
        _clear_pending_settle(task_dir)
        # Use settle.py's reported settled_item, not get_active_item — by
        # this point ACTIVE has already advanced to the NEXT pending item.
        settled_id = (settle_json or {}).get("settled_item") or "?"
        _post_settle(task_dir, kd_json.get("decision", "?"), settled_id)
        return

    config = load_task_config(task_dir)
    if config is None:
        print("[PIPELINE] ERROR: task.yaml not found")
        sys.exit(1)

    progress = load_progress(task_dir) or {}
    active = get_active_item(task_dir)
    # Persist the full description — dashboards/logs do their own display-time
    # truncation based on terminal width.
    desc = active["description"] if active else "optimization round"
    plan_item = active["id"] if active else None

    # === Step 1: Quick check ===
    print("[PIPELINE] Running quick_check...", flush=True)
    qc = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "quick_check.py"), task_dir],
        capture_output=True, text=True, timeout=60,
    )
    if qc.returncode != 0 or "OK" not in qc.stdout:
        auto_rollback(task_dir)
        # Clear edit marker — rollback means we're back to clean state
        marker = edit_marker_path(task_dir)
        if os.path.exists(marker):
            os.remove(marker)
        print(f"[PIPELINE] QUICK CHECK FAIL: {qc.stdout[:200]}")
        print(f"[PIPELINE] Auto-rolled back. Fix and re-edit.")
        print(get_guidance(task_dir))
        sys.exit(0)

    print("[PIPELINE] Quick check PASS", flush=True)

    # === Step 2: Eval ===
    # Outer cap mirrors eval_client's per-shape semantics so multi-shape
    # benches don't get killed at 600s before the inner verify even
    # finishes its first JIT pass. eval_client is authoritative; this is
    # just a safety net for hangs in eval_wrapper itself, hence the +300s
    # buffer over what eval_client will use internally.
    print("[PIPELINE] Running eval...", flush=True)
    config_for_timeout = load_task_config(task_dir)
    progress_for_count = load_progress(task_dir) or {}
    num_cases = progress_for_count.get("num_cases")
    if not isinstance(num_cases, int) or num_cases < 1:
        # Round 0 has no progress yet; probe the ref directly via the
        # same helper eval_client uses so the cap matches.
        try:
            from task_config.eval_client import _count_ref_cases  # type: ignore
            num_cases = _count_ref_cases(task_dir, config_for_timeout)
        except Exception:
            num_cases = 1
    per_shape_budget = int(getattr(config_for_timeout, "eval_timeout", 600) or 600)
    pipeline_eval_cap = per_shape_budget * max(num_cases, 1) + 300

    eval_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "eval_wrapper.py"), task_dir]
    try:
        ev = subprocess.run(eval_cmd, capture_output=True, text=True,
                            timeout=pipeline_eval_cap)
    except subprocess.TimeoutExpired:
        auto_rollback(task_dir)
        print(f"[PIPELINE] EVAL TIMEOUT after {pipeline_eval_cap}s "
              f"({per_shape_budget}s/shape x {num_cases} cases + 300s). "
              f"Rolled back.")
        sys.exit(0)

    eval_json = parse_last_json_line(ev.stdout)
    if eval_json is None:
        auto_rollback(task_dir)
        print(f"[PIPELINE] EVAL ERROR: no JSON output. stderr: {ev.stderr[:200]}")
        sys.exit(1)

    correctness = eval_json.get("correctness", False)
    metrics = eval_json.get("metrics", {})
    print(f"[PIPELINE] Eval: correctness={correctness}, metrics={metrics}", flush=True)

    # Surface structured failure signals (UB overflow, aivec trap, OOM, ...)
    # extracted from the eval subprocess's raw log. Without this, Claude
    # sees only a generic "verify failed" string and has nothing to act
    # on. Fall back
    # through increasingly coarse sources so *something* always reaches the
    # user on failure.
    if not correctness or eval_json.get("error"):
        if eval_json.get("error"):
            print(f"[PIPELINE] Error: {eval_json['error']}", flush=True)
        pretty = format_for_stdout(eval_json.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_json.get("raw_output_tail"):
            # No known pattern matched — dump the tail raw so Claude still
            # has something concrete to work with.
            print("[PIPELINE] Eval log tail (no structured signals matched):",
                  flush=True)
            print(eval_json["raw_output_tail"], flush=True)

    # === Step 3: Keep or discard ===
    # Direct library call (was a subprocess + last-line-JSON parse). Same
    # workflow.record_round body that `keep_or_discard.py` shells to;
    # in-process eliminates the JSON-protocol seam between pipeline and
    # the round writer.
    kd_json = record_round(task_dir, eval_json,
                           description=desc, plan_item=plan_item)
    if kd_json.get("decision") == "ERROR":
        print(f"[PIPELINE] KEEP/DISCARD ERROR: {kd_json.get('error')}")
        sys.exit(1)

    decision = kd_json.get("decision", "FAIL")

    # === Step 4: Settle (update plan.md) ===
    # progress.json + history.jsonl were already mutated by keep_or_discard;
    # plan.md is the only state piece settle.py owns. If settle fails, the
    # kd_json is persisted to .pending_settle.json so the NEXT invocation
    # of pipeline.py retries settle alone (no second eval, no duplicate
    # history row). The advance-phase block is gated on settle success.
    rc, tail_out, tail_err, _settle_json = _run_settle(task_dir, kd_json)
    if rc != 0:
        _persist_pending_settle(task_dir, kd_json)
        _emit_settle_failure(task_dir, rc, tail_out, tail_err)
        sys.exit(1)
    _clear_pending_settle(task_dir)

    # === Step 5+6: Advance phase + status report ===
    settled_id = active["id"] if active else "?"
    _post_settle(task_dir, decision, settled_id)


if __name__ == "__main__":
    main()
