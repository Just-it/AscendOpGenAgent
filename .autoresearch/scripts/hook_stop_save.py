#!/usr/bin/env python3
"""
Stop hook: When Claude Code stops (context limit, user interrupt, etc.),
stamp the reason into progress.json and print a final status summary.

Also enforces a DIAGNOSE invariant: the agent cannot Stop while phase is
DIAGNOSE — the phase exists to produce a new plan, and Stop short-circuits
that. The path out is always create_plan.py (subagent route normally,
manual-planning fallback after the subagent cap is reached). Stop only
becomes legal once create_plan.py has advanced phase out of DIAGNOSE.
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from hook_utils import read_hook_input, emit_status
from phase_machine import (
    DIAGNOSE, DIAGNOSE_ATTEMPTS_CAP, diagnose_state, get_task_dir,
    load_progress, read_phase, update_progress,
    DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
)


def _block_stop_with_reason(reason: str) -> None:
    """Tell Claude Code to refuse the stop and re-prompt the agent. Wire
    format follows Stop hook decision schema (`{decision: "block",
    reason: ...}`)."""
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def main():
    stop_reason = read_hook_input().get("stop_reason", "unknown")

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)

    progress = load_progress(task_dir)
    if progress is None:
        sys.exit(0)

    # DIAGNOSE Stop gate: the phase requires a new plan; Stop is blocked
    # until create_plan.py advances phase out of DIAGNOSE. Two sub-states:
    #   - subagent attempts < cap → instruct to re-issue Task (preferred path)
    #   - subagent attempts >= cap → instruct to manual planning fallback
    # Either way Stop is illegal here.
    if read_phase(task_dir) == DIAGNOSE:
        state = diagnose_state(task_dir, progress=progress)
        if state.action == DIAGNOSE_READY:
            _block_stop_with_reason(
                f"[AR] Cannot Stop: DIAGNOSE artifact is ready, but the "
                f"phase still needs a new plan. Write plan_items.xml from "
                f"diagnose_v{state.plan_version}.md, then run "
                f"create_plan.py."
            )
        elif state.action == DIAGNOSE_MANUAL_FALLBACK:
            _block_stop_with_reason(
                f"[AR] Cannot Stop: phase=DIAGNOSE requires a new plan. "
                f"Subagent attempts exhausted "
                f"({state.attempts}/{DIAGNOSE_ATTEMPTS_CAP}); switch to "
                f"manual planning — Write plan_items.xml directly using "
                f"history.jsonl + plan.md, then run create_plan.py."
            )
        else:
            _block_stop_with_reason(
                f"[AR] Cannot Stop: phase=DIAGNOSE requires a new plan. "
                f"Re-issue Task with subagent_type='ar-diagnosis'; only "
                f"create_plan.py advancing the phase out of DIAGNOSE "
                f"makes Stop legal. Attempts so far: "
                f"{state.attempts}/{DIAGNOSE_ATTEMPTS_CAP}."
            )

    update_progress(
        task_dir,
        last_stop_reason=stop_reason,
        last_stop_time=datetime.now(timezone.utc).isoformat(),
    )

    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 0)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")

    improv = ""
    if best is not None and baseline is not None and baseline != 0:
        pct = (baseline - best) / abs(baseline) * 100
        improv = f" ({pct:+.1f}%)"

    emit_status(f"\n[AR] Session stopped: {stop_reason}")
    emit_status(f"[AR] Progress: {rounds}/{max_rounds} rounds | Best: {best}{improv}")
    emit_status(f"[AR] Resume with: /autoresearch --resume {task_dir}")


if __name__ == "__main__":
    main()
