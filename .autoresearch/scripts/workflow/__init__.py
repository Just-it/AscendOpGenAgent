"""workflow/ — orchestration layer between hooks and state_store.

Modules added incrementally; first one is `transition` (PhaseController),
which owns every .ar_state/.phase write. Subsequent steps will move
plan/round/baseline orchestration logic out of pipeline.py / hook_*.py
and into siblings here.
"""
from .transition import PhaseController
from .planning import PlanStore
from .baseline import run_baseline_init
from .round import record_round

__all__ = ["PhaseController", "PlanStore", "run_baseline_init", "record_round"]
