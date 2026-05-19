"""Metric comparison and constraint checking.

Pure data-shape and arithmetic logic — no I/O, no subprocess, no YAML.
The `EvalResult` dataclass is the contract eval_runner writes into;
downstream consumers (keep_or_discard, baseline_init, dashboard) read
from it.

What lives here:
  - `EvalResult`           — the result dataclass.
  - `is_improvement`       — current-vs-best comparison with relative-%
                             threshold and direction (`lower_is_better`).
  - `check_constraints`    — hard-constraint check
                             ({metric: (op_str, threshold)} →
                              list of violation strings).
  - `format_result_summary`— one-line human-readable summary used by
                             stderr logging in baseline / pipeline.

Why a separate module: the comparison logic is the only piece of
task_config that has zero external dependencies and zero side effects;
splitting it out lets every other module that needs only EvalResult
import from here without dragging in YAML / urllib / tarfile.
"""
import operator as _op
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalResult:
    """Evaluation result."""
    correctness: bool
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    raw_output: str = ""


# ---------------------------------------------------------------------------
# Constraint check
# ---------------------------------------------------------------------------

_CONSTRAINT_OPS = {"<=": _op.le, ">=": _op.ge, "<": _op.lt, ">": _op.gt, "==": _op.eq}


def check_constraints(result: EvalResult, constraints: dict) -> list:
    """Check hard constraints. Returns list of violation strings (empty = ok)."""
    violations = []
    for metric_name, (op_str, threshold) in constraints.items():
        func = _CONSTRAINT_OPS.get(op_str)
        if func is None:
            violations.append(f"{metric_name}: unknown operator '{op_str}'")
            continue
        value = result.metrics.get(metric_name)
        if value is None:
            violations.append(f"{metric_name}: metric missing (required {op_str} {threshold})")
            continue
        if not isinstance(value, (int, float)):
            violations.append(f"{metric_name}: non-numeric value {value!r}")
            continue
        if not func(value, threshold):
            violations.append(f"{metric_name}: {value} violates {op_str} {threshold}")
    return violations


# ---------------------------------------------------------------------------
# Improvement comparison
# ---------------------------------------------------------------------------

def is_improvement(
    current: EvalResult,
    best: EvalResult,
    metric: str = "latency_ms",
    lower_is_better: bool = True,
    threshold: float = 0.0,
) -> bool:
    """Check if current result improves on best.

    threshold is a relative percentage (e.g. 2.0 = needs >2% improvement).
    """
    if not current.correctness:
        return False
    cur_val = current.metrics.get(metric)
    best_val = best.metrics.get(metric)
    if cur_val is None:
        return False
    if best_val is None:
        return True
    if best_val == 0:
        return cur_val < 0 if lower_is_better else cur_val > 0
    if lower_is_better:
        relative_pct = (best_val - cur_val) / abs(best_val) * 100
    else:
        relative_pct = (cur_val - best_val) / abs(best_val) * 100
    return relative_pct > threshold


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def format_result_summary(result: EvalResult) -> str:
    """Human-readable one-line summary."""
    if not result.correctness:
        if result.error:
            return f"FAILED: {result.error}"
        return f"CORRECTNESS FAILED (metrics: {result.metrics})"
    parts = ["correctness: PASS"]
    for key, val in result.metrics.items():
        if isinstance(val, float):
            parts.append(f"{key}: {val:.4f}")
        else:
            parts.append(f"{key}: {val}")
    return "  |  ".join(parts)
