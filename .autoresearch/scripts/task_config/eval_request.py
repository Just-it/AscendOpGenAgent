"""Build canonical eval requests before invoking the subprocess.

This module owns request-time decisions: case-count probing, timeout
scaling, and sticky baseline override lookup. It returns data only;
the runner executes and the assembler interprets the response.

AOA only has a local-subprocess transport (no HTTP worker), so this
module's surface stays small — one factory function and a few helpers.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

from .loader import TaskConfig

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from utils.baseline_anchor import (  # noqa: E402
    current_fingerprint, sticky_override_from_progress,
)


@dataclass(frozen=True)
class EvalRequest:
    task_dir: str
    config: TaskConfig
    num_cases: int
    timeout: int
    override_base_us: Optional[float] = None
    override_base_per_shape_us: Optional[list[float]] = None

    @property
    def sticky(self) -> bool:
        return self.override_base_us is not None

    def sticky_note(self) -> str:
        if self.override_base_us is None:
            return ""
        if self.override_base_per_shape_us:
            extra = f" (+ per-shape {len(self.override_base_per_shape_us)})"
        else:
            extra = " (aggregate only)"
        return f"sticky baseline override={self.override_base_us:.2f} us{extra}"


def count_ref_cases(task_dir: str, config: TaskConfig) -> int:
    """Probe the ref module locally and count input cases.

    Mirrors what the generated verify script does: import ref + run
    input_groups.resolve, which duck-types between get_input_groups
    (multi-shape, NPUKernelBench) and get_inputs (single-shape collapsed
    to N=1). Used purely to scale eval_timeout — any failure falls back
    to 1 (single-shape semantics, no scaling).
    """
    ref_path = os.path.join(task_dir, config.ref_file)
    if not os.path.isfile(ref_path):
        return 1
    ref_dir = os.path.dirname(ref_path) or "."
    sys_path_added = ref_dir not in sys.path
    if sys_path_added:
        sys.path.insert(0, ref_dir)
    try:
        import importlib.util
        from utils.input_groups import resolve as _resolve  # type: ignore
        spec = importlib.util.spec_from_file_location(
            f"_count_ref_{config.name}", ref_path)
        if spec is None or spec.loader is None:
            return 1
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return max(len(_resolve(mod)), 1)
    except Exception as e:
        print(f"[eval_request] WARN: case-count probe failed "
              f"({type(e).__name__}: {e})", file=sys.stderr)
        return 1
    finally:
        if sys_path_added:
            try:
                sys.path.remove(ref_dir)
            except ValueError:
                pass


def effective_timeout(config: TaskConfig, num_cases: int) -> int:
    """config.eval_timeout is the budget per shape; scale by case count."""
    return int(config.eval_timeout) * max(int(num_cases), 1)


def override_base_from_progress(
        task_dir: str,
        num_cases: int,
        ) -> Optional[tuple[float, Optional[list[float]]]]:
    """Return sticky baseline override when the stored anchor is comparable."""
    try:
        from phase_machine import load_progress  # type: ignore
        progress = load_progress(task_dir) or {}
    except Exception:
        return None

    fingerprint = current_fingerprint(num_cases)
    decision = sticky_override_from_progress(progress, fingerprint)
    if decision.mismatch:
        print(f"[eval_request] sticky baseline invalidated: "
              f"fingerprint mismatch {decision.mismatch}; "
              f"will re-measure ref", file=sys.stderr)
    if decision.override is None:
        return None
    return decision.override.metric, decision.override.per_shape_us


def build_eval_request(task_dir: str, config: TaskConfig) -> EvalRequest:
    num_cases = count_ref_cases(task_dir, config)
    timeout = effective_timeout(config, num_cases)
    override = override_base_from_progress(task_dir, num_cases)
    if override is None:
        override_base_us, override_per_shape = None, None
    else:
        override_base_us, override_per_shape = override
    return EvalRequest(
        task_dir=task_dir,
        config=config,
        num_cases=num_cases,
        timeout=timeout,
        override_base_us=override_base_us,
        override_base_per_shape_us=override_per_shape,
    )
