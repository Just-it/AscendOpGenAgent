#!/usr/bin/env python3
"""
PostToolUse hook for Bash — phase auto-advancement after user-issued commands.

The only commands that advance phase from this hook are those Claude runs
directly via the Bash tool:
  - `export AR_TASK_DIR=...`  → activate task, compute starting phase
                                (fresh task: validate ref/kernel and pin the
                                appropriate GENERATE_* / BASELINE phase)
  - `baseline.py`             → PLAN on success;
                                GENERATE_KERNEL on seed-metric failure
                                (so kernel.py becomes editable again)
  - `pipeline.py`             → whatever phase pipeline.py itself wrote
  - `create_plan.py`          → EDIT on plan validation pass
                                (called from PLAN / DIAGNOSE / REPLAN)

The inner pipeline steps (quick_check / eval_wrapper / keep_or_discard /
settle) are subprocess children of pipeline.py and never re-enter this hook,
so they don't need their own phase constants or branches here.
"""
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status, emit_todowrite_context
from workflow import PhaseController
from phase_machine import (
    read_phase, get_guidance, compute_resume_phase,
    get_task_dir, set_task_dir, get_active_item, touch_heartbeat,
    load_progress, update_progress,
    validate_reference, validate_kernel, is_placeholder_file,
    parse_invoked_ar_script,
    progress_path, history_path, plan_path, edit_marker_path, state_path,
    PHASE_FILE,
    BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN, GENERATE_REF, GENERATE_KERNEL,
)


def _activation_target(command: str) -> str | None:
    r"""Extract the path from `export AR_TASK_DIR=<path>`. Uses shlex
    so quoted values with spaces (`AR_TASK_DIR="/path with space"`)
    survive — the earlier `[^"\';\s&]+` regex truncated at the first
    space."""
    if "AR_TASK_DIR=" not in command:
        return None
    try:
        tokens = shlex.split(command, posix=True, comments=False)
    except ValueError:
        return None
    for tok in tokens:
        if tok.startswith("AR_TASK_DIR="):
            return tok[len("AR_TASK_DIR="):] or None
    return None


# Script-invocation parsing lives in phase_machine.parse_invoked_ar_script,
# a thin view over `classify(command)` — returns the AR script basename
# only when the classifier sees a canonical AR shape (and None otherwise,
# including for non-canonical AR-mentions which PreToolUse already
# rejected). Under that contract the basename returned here is
# unambiguous, and shapes like `python --version ...X.py` or
# `python -c ... .../X.py` no longer falsely advance phase.


def _clean_stale_edit_marker(task_dir: str):
    """Remove .edit_started if git is clean (nothing to resume)."""
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    try:
        import subprocess as _sp
        diff = _sp.run(
            ["git", "status", "--porcelain"],
            cwd=task_dir, capture_output=True, text=True, timeout=5,
        )
        if not diff.stdout.strip():
            os.remove(marker)
            emit_status("[AR] Cleaned stale edit marker (git is clean).")
    except Exception:
        pass


def _handle_activation(new_task_dir: str):
    new_task_dir = os.path.abspath(new_task_dir)
    if not os.path.isdir(new_task_dir):
        emit_status(f"[AR] ERROR: task_dir not found: {new_task_dir}")
        return

    set_task_dir(new_task_dir)
    _clean_stale_edit_marker(new_task_dir)

    has_phase = os.path.exists(state_path(new_task_dir, PHASE_FILE))
    has_progress = os.path.exists(progress_path(new_task_dir))

    pc = PhaseController(new_task_dir)
    if has_phase:
        phase = read_phase(new_task_dir)
        emit_status(f"[AR] Resuming. Phase: {phase}.")
        _print_resume_context(new_task_dir)
        emit_status(get_guidance(new_task_dir))
    elif has_progress:
        phase = pc.on_activation_resume()
        emit_status(f"[AR] Resuming from progress. Phase -> {phase}.")
        _print_resume_context(new_task_dir)
        emit_status(get_guidance(new_task_dir))
    else:
        _fresh_start(new_task_dir)


def _fresh_start(task_dir: str):
    """Pick initial phase for a fresh task based on which files are present
    AND validate them. `is_placeholder_file` (canonical) lets us short-
    circuit the subprocess-import step on a known stub; otherwise the same
    validate_reference / validate_kernel that gates phase advances also
    pins the right phase from the moment of activation."""
    ref_path = os.path.join(task_dir, "reference.py")
    kernel_path = os.path.join(task_dir, "kernel.py")
    pc = PhaseController(task_dir)

    if is_placeholder_file(ref_path):
        pc.on_activation_no_ref()
        emit_status(f"[AR] Fresh start (no reference). Phase -> GENERATE_REF. {get_guidance(task_dir)}")
        return

    ok, err = validate_reference(task_dir)
    if not ok:
        pc.on_activation_invalid_ref()
        emit_status(
            f"[AR] reference.py present but invalid — Phase -> GENERATE_REF.\n"
            f"     {err}"
        )
        return

    if is_placeholder_file(kernel_path):
        pc.on_activation_no_kernel()
        emit_status(f"[AR] Fresh start (no kernel). Phase -> GENERATE_KERNEL. {get_guidance(task_dir)}")
        return

    ok, err = validate_kernel(task_dir)
    if not ok:
        pc.on_activation_invalid_kernel()
        emit_status(
            f"[AR] kernel.py present but invalid — Phase -> GENERATE_KERNEL.\n"
            f"     {err}"
        )
        return

    pc.on_activation_ready()
    emit_status(f"[AR] Fresh start. Phase -> BASELINE. {get_guidance(task_dir)}")


def _progress_update_for_plan(task_dir: str, phase: str):
    """Set status=active after a valid new plan. `plan_version` is owned and
    bumped by create_plan.py — this hook must NOT re-bump it (caused double
    increments that jumped plan_version by 2 each REPLAN)."""
    fields = {"status": "active"}
    if phase == DIAGNOSE:
        fields["consecutive_failures"] = 0
    update_progress(task_dir, **fields)


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    stdout = str(hook_input.get("tool_output", ""))

    # --- Activation (export AR_TASK_DIR=...) ---
    # Activation arrives as its own Bash call under the canonical-form
    # gate (any chain is rejected at PreToolUse), so we can return as
    # soon as `_handle_activation` has set up the task pointer + emitted
    # guidance — there is no AR-script invocation in the same command
    # to dispatch on.
    target = _activation_target(command)
    if target:
        _handle_activation(target)
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    if invoked == "baseline.py" and phase == BASELINE:
        progress = load_progress(task_dir)
        if not progress:
            emit_status("[AR] Baseline failed (no progress.json). Retry.")
        else:
            # PhaseController.on_baseline_settled re-reads progress and
            # picks PLAN vs GENERATE_KERNEL from seed_metric +
            # baseline_correctness; the demote message stays here
            # because it's user-facing.
            new_phase = PhaseController(task_dir).on_baseline_settled()
            if new_phase == GENERATE_KERNEL:
                reason = ("seed kernel produced no timing"
                          if progress.seed_metric is None
                          else "seed kernel failed correctness check")
                emit_status(
                    f"[AR] Baseline failed: {reason}. "
                    f"Phase -> GENERATE_KERNEL so kernel.py becomes editable. "
                    f"Fix the kernel, then re-run baseline.py."
                )
            else:
                emit_status(f"[AR] Baseline complete. Phase -> PLAN. "
                            f"{get_guidance(task_dir)}")

    elif invoked == "pipeline.py":
        # pipeline.py writes .phase itself; just project state + notify.
        new_phase = read_phase(task_dir)
        emit_status(f"[AR] Pipeline complete. Phase -> {new_phase}. {get_guidance(task_dir)}")
        emit_todowrite_context(task_dir, f"[AR] Round settled. Phase -> {new_phase}.")

    elif invoked == "create_plan.py" and phase in (PLAN, DIAGNOSE, REPLAN, EDIT):
        from phase_machine import validate_plan, pending_settle_path
        # PLAN/DIAGNOSE/REPLAN: normal plan-creation flow.
        # EDIT: only legal as a recovery path when settle.py kept failing
        # on a malformed plan.md (gated in hook_guard_bash by the
        # presence of .pending_settle.json). The new plan retires the
        # broken plan_version, so the orphan kd_json is no longer
        # actionable; clear the sidecar.
        #
        # NOTE: do NOT re-validate the diagnose artifact here. PreToolUse
        # (hook_guard_bash) already enforced the artifact gate against the
        # plan_version that existed BEFORE create_plan.py ran. By the time
        # this PostToolUse fires, create_plan.py has bumped plan_version
        # from N to N+1 — re-running diagnose_state would look for
        # diagnose_v(N+1).md (not yet created) and falsely reject.
        if phase == EDIT and not os.path.exists(pending_settle_path(task_dir)):
            # Defense-in-depth: hook_guard_bash should have blocked this,
            # but if it slipped through somehow, refuse to advance state.
            emit_status("[AR] create_plan.py in EDIT phase requires a "
                        "pending settle recovery state; nothing to do.")
            sys.exit(0)
        ok, err = validate_plan(task_dir)
        if ok:
            _progress_update_for_plan(task_dir, phase)
            PhaseController(task_dir).on_plan_validated()
            if phase == EDIT:
                # Recovery completed: discard the orphan kd_json. The new
                # plan_version starts fresh; the round whose decision was
                # waiting in pending_settle is recorded in history.jsonl
                # but no longer corresponds to any plan item.
                ps = pending_settle_path(task_dir)
                if os.path.exists(ps):
                    os.remove(ps)
                emit_status(f"[AR] Pending settle abandoned; new plan "
                            f"installed. Phase -> EDIT. {get_guidance(task_dir)}")
            else:
                emit_status(f"[AR] Plan validated. Phase -> EDIT. {get_guidance(task_dir)}")
            emit_todowrite_context(task_dir, "[AR] Plan validated. Phase -> EDIT.")
        else:
            emit_status(f"[AR] Plan not valid yet: {err}")

    sys.exit(0)


def _print_resume_context(task_dir: str):
    progress = load_progress(task_dir)
    if not progress:
        return
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)

    emit_status(
        f"[AR] Resume context: Round {rounds}/{max_rounds} | "
        f"Best: {best} | Baseline: {baseline} | "
        f"Failures: {failures} | Plan v{plan_ver}"
    )

    hpath = history_path(task_dir)
    if os.path.exists(hpath):
        with open(hpath, "r") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            emit_status(f"[AR] Last {min(3, len(lines))} rounds:")
            for rec in lines[-3:]:
                rnd = rec.get("round")
                rnd = "?" if rnd is None else str(rnd)
                dec = rec.get("decision", "?")
                desc = rec.get("description", "")[:40]
                emit_status(f"[AR]   R{rnd}: {dec} — {desc}")

    if os.path.exists(plan_path(task_dir)):
        active = get_active_item(task_dir)
        if active:
            emit_status(f"[AR] Active item: {active['id']}: {active['description'][:50]}")
        emit_status("[AR] Read .ar_state/plan.md and .ar_state/history.jsonl for full context.")


if __name__ == "__main__":
    main()
