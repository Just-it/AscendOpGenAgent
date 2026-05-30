#!/usr/bin/env python3
"""Mechanical plan.md settlement — no LLM needed.

After `workflow.record_round` (called in-process by pipeline.py) runs,
this script:
1. Reads the decision (KEEP/DISCARD/FAIL) from record_round's return dict
2. Updates plan.md via PlanStore.settle_active (mark active item [x],
   advance ACTIVE marker, append Settled History row)

Usage:
    python settle.py <task_dir> <decision_json>

Output (stdout, last line):
    {"settled_item": "p1", "decision": "KEEP", "metric": 1294.8}

Scope note: settle.py does NOT advance .ar_state/.phase, and does not
predict the next phase in its output either. The phase transition after
a settled round is owned by pipeline.py's `_post_settle` (via
PhaseController.on_round_settled). The earlier `next_phase` field was a
forecast emitted from a second copy of the rule (compute_next_phase
called here, then re-called by PhaseController), and no caller actually
read it — pipeline.py recomputes the transition itself. Keep the rule
on the pipeline side; settle owns plan.md only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow import PlanStore


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
    }))


if __name__ == "__main__":
    main()
