#!/usr/bin/env python3
"""Mechanical plan.md settlement — no LLM needed.

After keep_or_discard.py runs, this script:
1. Reads the decision (KEEP/DISCARD/FAIL) from keep_or_discard output
2. Updates plan.md via PlanStore.settle_active (mark active item [x],
   advance ACTIVE marker, append Settled History row)
3. Returns the next phase (computed by PhaseController.on_round_settled
   - keep using `compute_next_phase` here on its own to avoid double-
   writing .phase; pipeline.py is responsible for the actual phase
   write via PhaseController in the post-settle path)

Usage:
    python settle.py <task_dir> <decision_json>

Output (stdout, last line):
    {"next_phase": "EDIT", "settled_item": "p1", "decision": "KEEP",
     "metric": 1294.8}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from workflow import PlanStore
from phase_machine import compute_next_phase


def main():
    if len(sys.argv) != 3:
        print(json.dumps({
            "error": "invalid arguments",
            "usage": "python settle.py <task_dir> <decision_json>",
            "received_args": sys.argv[1:],
        }))
        sys.exit(1)

    task_dir = sys.argv[1]
    decision_json = sys.argv[2]

    try:
        decision_data = json.loads(decision_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": "invalid decision_json",
                          "details": str(exc)}))
        sys.exit(1)

    decision = decision_data.get("decision", "FAIL")
    best_metric = decision_data.get("best_metric")
    # KEEP carries this round's metric; DISCARD/FAIL leave it None.
    metric_val = best_metric if decision == "KEEP" else None

    ps = PlanStore(task_dir)
    if not ps.exists():
        print(json.dumps({"error": "plan.md not found"}))
        sys.exit(1)
    try:
        settled_id, _ = ps.settle_active(decision, metric_val)
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    print(json.dumps({
        "settled_item": settled_id,
        "decision": decision,
        "metric": metric_val,
        "next_phase": compute_next_phase(task_dir),
    }))


if __name__ == "__main__":
    main()
