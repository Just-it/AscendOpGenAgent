"""workflow/ — orchestration layer between hooks and state_store.

Owns the rules that turn "what just happened" into "next phase" and the
record_round / run_baseline_init bodies. Both are called in-process by
engine/pipeline.py and engine/baseline.py respectively; the previous
shell wrappers (keep_or_discard.py, _baseline_init.py) have been
deleted now that no caller crosses a subprocess boundary.
"""
from .transition import PhaseController
from .planning import PlanStore
from .baseline import run_baseline_init
from .round import record_round

__all__ = ["PhaseController", "PlanStore", "run_baseline_init", "record_round"]
