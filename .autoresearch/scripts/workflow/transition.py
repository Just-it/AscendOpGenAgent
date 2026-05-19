"""PhaseController — single owner of .ar_state/.phase writes.

Before this module, write_phase was called from 12 sites across
hook_post_bash, hook_post_edit, pipeline.py, and _baseline_init.
Each site embedded its own decision logic (read progress, inspect
files, branch on subagent state, ...). When phase rules drifted in one
place they didn't drift in the others, which is how
GENERATE_KERNEL-vs-BASELINE retry behaviour got tangled.

PhaseController takes EVENTS as input (what just happened) and is the
only thing that decides the target phase + writes it. Callers no longer
do `write_phase(...)` directly; they do `PhaseController(td).on_xxx()`.
The set of events here is exhaustive for the current call sites, by
design: a new event has to land here, not in the caller.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    BASELINE, EDIT, FINISH, GENERATE_KERNEL, GENERATE_REF, PLAN,
    compute_next_phase, compute_resume_phase, load_progress, read_phase,
    write_phase,
)


class PhaseController:
    def __init__(self, task_dir: str):
        self.task_dir = task_dir

    # ---- Activation -----------------------------------------------------
    def on_activation_resume(self) -> str:
        phase = compute_resume_phase(self.task_dir)
        return self._write(phase)

    def on_activation_no_ref(self) -> str:
        return self._write(GENERATE_REF)

    def on_activation_invalid_ref(self) -> str:
        return self._write(GENERATE_REF)

    def on_activation_no_kernel(self) -> str:
        return self._write(GENERATE_KERNEL)

    def on_activation_invalid_kernel(self) -> str:
        return self._write(GENERATE_KERNEL)

    def on_activation_ready(self) -> str:
        return self._write(BASELINE)

    # ---- Seed (post-Edit on reference.py / kernel.py) -------------------
    def on_seed_validated(self, next_phase: str) -> str:
        # next_phase chosen by caller (BASELINE after kernel seed,
        # GENERATE_KERNEL after ref seed if kernel still placeholder, ...).
        # Rule's encoded outside for now because the seed-validation flow in
        # hook_post_edit has its own sequencing; folding it here would
        # require moving the file-presence inspection too.
        return self._write(next_phase)

    # ---- Baseline -------------------------------------------------------
    def on_baseline_settled(self) -> str:
        """End of `baseline.py` invocation. Decides PLAN vs GENERATE_KERNEL
        from the freshly-written progress.json (so this captures a partial
        result from _baseline_init that exited 0 too)."""
        progress = load_progress(self.task_dir)
        if progress is None:
            # No progress.json → don't move; baseline.py crashed early.
            return read_phase(self.task_dir)
        if progress.seed_metric is None or not progress.baseline_correctness:
            return self._write(GENERATE_KERNEL)
        return self._write(PLAN)

    def on_baseline_init_success(self) -> str:
        # _baseline_init.py writes PLAN itself today; we keep that behaviour
        # so external state hasn't changed, but the caller now goes through
        # this method so all phase writes are visible here.
        return self._write(PLAN)

    # ---- Plan -----------------------------------------------------------
    def on_plan_validated(self) -> str:
        return self._write(EDIT)

    # ---- Round (post settle.py) ----------------------------------------
    def on_round_settled(self) -> str:
        """End of one EDIT round. Delegates to compute_next_phase, which
        reads eval_rounds / consecutive_failures / pending plan items
        from progress + plan.md."""
        next_phase = compute_next_phase(self.task_dir)
        return self._write(next_phase)

    # ---- internal -------------------------------------------------------
    def _write(self, phase: str) -> str:
        write_phase(self.task_dir, phase)
        return phase

    # ---- Convenience constants ----------------------------------------
    # Re-exported so callers don't need a second `from phase_machine
    # import FINISH` next to their PhaseController call.
    FINISH = FINISH
