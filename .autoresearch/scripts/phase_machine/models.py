"""Authoritative schema for .ar_state/progress.json.

Single dataclass owns the field set. Every writer constructs a complete
`Progress` (or applies field deltas via `.apply(**fields)`) so that:
  - Adding a field requires editing one place.
  - Forgetting to set a field in a writer no longer drops it from disk.
  - Readers can stay on `progress.get("X", default)` or move to attribute
    access; both work (`Progress.get` mirrors `dict.get`).

Co-located with state_store.py because they're inseparable: state_store
is the only entry point that turns these objects into JSON on disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, replace
from typing import Any, Optional


@dataclass
class Progress:
    # Identity
    task: str = ""

    # Round counters
    eval_rounds: int = 0
    # max_rounds defaults HIGH (not 0) so a legacy progress.json missing
    # this field doesn't trip `eval_rounds >= max_rounds` -> FINISH on the
    # first lookup. Real writers (_baseline_init via workflow.baseline)
    # always set the actual config value; the default only fires for
    # incomplete files. compute_next_phase and compute_resume_phase still
    # call `progress.get("max_rounds", 999)` for dict-era compatibility,
    # which now matches the dataclass field default and is consistent.
    max_rounds: int = 999
    consecutive_failures: int = 0

    # Best kernel measured so far
    best_metric: Optional[float] = None
    best_commit: Optional[str] = None

    # Sticky pytorch baseline (anchors speedup display; pinned by the first
    # baseline_init that captured ref_latency_us, see workflow/seed.py).
    baseline_metric: Optional[float] = None
    baseline_commit: Optional[str] = None
    baseline_source: Optional[str] = None      # "ref" | "seed_fallback"
    baseline_correctness: bool = False
    seed_metric: Optional[float] = None

    # Plan
    plan_version: int = 0
    next_pid: int = 0
    status: str = "no_plan"

    # Multi-shape detail (single-shape ops keep these absent)
    num_cases: Optional[int] = None
    per_shape_descs: Optional[list] = None

    # Diagnose subagent state
    diagnose_attempts: int = 0
    diagnose_attempts_for_version: Optional[int] = None
    last_diagnose_failure_reason: Optional[str] = None

    # Stop-hook trace
    last_stop_reason: Optional[str] = None
    last_stop_time: Optional[str] = None

    # Auto-stamped by state_store.save_progress when stamp=True
    last_updated: Optional[str] = None

    # ---- dict-compat read API --------------------------------------------
    # Existing readers do `progress.get("X", default)` everywhere; supplying
    # this method keeps them working without a rewrite. `keys()` /
    # `__iter__` / `__getitem__` cover the rest of the dict surface that
    # resume.py and report.py used to reach for (`set(progress.keys())`,
    # `for k in progress`, `progress["X"]`).
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:  # `"X" in progress`
        return key in {f.name for f in fields(self)}

    def keys(self):
        return [f.name for f in fields(self)]

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: str) -> Any:
        if key in {f.name for f in fields(self)}:
            return getattr(self, key)
        raise KeyError(key)

    # ---- mutation -------------------------------------------------------
    def apply(self, **changes: Any) -> "Progress":
        """Return a new Progress with `changes` overlaid. Validates field
        names so a typo becomes TypeError instead of a silently-dropped
        attribute (which is what `progress["typo"] = ...` did before)."""
        return replace(self, **changes)

    # ---- (de)serialisation ---------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Progress":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        kept = {k: v for k, v in data.items() if k in known}
        unknown = sorted(set(data) - known)
        if unknown:
            import sys
            print(f"[Progress.from_dict] dropping unknown fields: {unknown}",
                  file=sys.stderr)
        return cls(**kept)
