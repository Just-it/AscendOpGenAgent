"""Convert raw eval transport responses into EvalResult.

This module owns metric semantics: sidecar interpretation, outcome
classification, per-shape aggregation, timing-method mismatch flags,
and verify failure detail. It has no transport, YAML, package, or
Progress I/O.

AOA's eval_runner.local_eval returns a (verify_resp, profile_resp)
tuple — keep that shape; `assemble_eval_result` accepts both dicts and
extracts what it needs.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

from .metric_policy import EvalOutcome, EvalResult

_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from utils.json_io import parse_last_json_line as _last_json_line  # noqa: E402


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and 0 < v < float("inf")


def _resolve_profile(resp: dict, key: str, artifact_name: str):
    """Return (top_level_time, parsed_artifact). Falls back to artifact
    `avg_time_us` when the transport didn't surface the field at top-level."""
    t = resp.get(key)
    artifacts = resp.get("artifacts") or {}
    art = None
    if artifact_name in artifacts:
        try:
            art = json.loads(artifacts[artifact_name])
        except (json.JSONDecodeError, TypeError):
            art = None
    if t is None and art is not None:
        t = art.get("avg_time_us")
    return t, art


def _per_shape_floats(art: Optional[dict]) -> Optional[list]:
    """List of `avg_time_us` values from a profile artifact, or None."""
    if not art:
        return None
    ps = art.get("per_shape")
    if not isinstance(ps, list) or not ps:
        return None
    return [(s.get("avg_time_us") if isinstance(s, dict) else None) for s in ps]


def _per_shape_methods(art: Optional[dict]) -> Optional[list]:
    """Per-shape `method` strings (e.g. "profiler", "fallback", "sticky").
    Used to detect kernel-vs-ref measurement-method mismatches so a
    cross-method comparison doesn't silently produce a bogus speedup."""
    if not art:
        return None
    ps = art.get("per_shape")
    if not isinstance(ps, list) or not ps:
        return None
    return [(s.get("method") if isinstance(s, dict) else None) for s in ps]


def assemble_eval_result(verify_resp: dict, profile_resp: dict) -> EvalResult:
    """Combine verify + profile responses into an EvalResult.

    Single invariant:
        correctness = (verify passed) AND (every per-shape profile timing
        is finite). Anything else - latency, speedup, per-shape arrays,
        failure detail - is just data populated into `metrics` for
        downstream readers (record_round, DIAGNOSE, report.py).

    `record_round`'s settlement gate keys off `correctness`, so a kernel
    that mis-matches ref on any shape OR crashes during any shape's
    profile run lands as FAIL with the same code path.
    """
    verify_log = verify_resp.get("log", "")
    verify_ok = bool(verify_resp.get("success", False))
    # error_source / verify_block come from eval_runner directly (it
    # parses .eval_result.json — eval_kernel doesn't print the verify
    # dict to stderr). Fall back to the log JSON tail for legacy producers.
    error_source = verify_resp.get("error_source") if not verify_ok else None
    verify_json = (verify_resp.get("verify_block")
                   or _last_json_line(verify_log)
                   or {})

    gen_time, gen_art = _resolve_profile(profile_resp, "gen_time",
                                         "generation_profile_result.json")
    base_time, base_art = _resolve_profile(profile_resp, "base_time",
                                           "base_profile_result.json")
    gen_ok = _finite(gen_time)
    base_ok = _finite(base_time)

    per_gen = _per_shape_floats(gen_art)
    per_base = _per_shape_floats(base_art)

    # `latency_us` aggregate is computed in eval_kernel as mean of finite
    # per-shape timings - so gen_ok being True does NOT imply every shape
    # finished. The strict crashed-shape list is what gates correctness.
    crashed_shapes = (
        [i for i, t in enumerate(per_gen) if not _finite(t)]
        if per_gen is not None else []
    )

    # Outcome — only two non-OK paths:
    #   error_source == "ref"  → broken --ref source file. INFRA_FAIL.
    #   anything else failing  → kernel responsibility. KERNEL_FAIL.
    # Pure-infra failures (no backend / no NPU) set INFRA_FAIL before
    # we ever reach this assembler.
    if error_source == "ref":
        outcome = EvalOutcome.INFRA_FAIL
    elif verify_ok and not crashed_shapes:
        outcome = EvalOutcome.OK
    else:
        outcome = EvalOutcome.KERNEL_FAIL

    metrics: dict = {}

    # --- timing + speedup ---------------------------------------------
    # ref_latency_us and latency_us are recorded INDEPENDENTLY: a SEED
    # round where the kernel crashed (gen_ok=False) but the PyTorch ref
    # measured cleanly still has a valid base_time we want to anchor
    # baseline_metric on.
    if gen_ok:
        metrics["latency_us"] = gen_time
    else:
        print(f"[eval] WARNING: no valid gen_time (got {gen_time!r}) - "
              f"kernel profile likely failed", file=sys.stderr)
    if base_ok:
        metrics["ref_latency_us"] = base_time
    else:
        print(f"[eval] WARNING: no valid base_time (got {base_time!r}) - "
              f"ref baseline unavailable this round", file=sys.stderr)
    if gen_ok and base_ok:
        metrics["speedup_vs_ref"] = base_time / gen_time
    elif profile_resp.get("speedup"):
        metrics["speedup_vs_ref"] = profile_resp["speedup"]

    # --- per-shape detail ---------------------------------------------
    if per_gen is not None:
        metrics["num_cases"] = len(per_gen)
        metrics["per_shape_gen_us"] = per_gen
        gen_methods = _per_shape_methods(gen_art)
        if gen_methods:
            metrics["per_shape_gen_method"] = gen_methods
            uniq_gen = sorted({m for m in gen_methods if m})
            if uniq_gen:
                metrics["timing_method_gen"] = (
                    uniq_gen[0] if len(uniq_gen) == 1 else "mixed")
        if crashed_shapes:
            metrics["profile_crashed_cases"] = crashed_shapes[:30]
            metrics["profile_crashed_count"] = len(crashed_shapes)
        if per_base is not None and len(per_base) == len(per_gen):
            metrics["per_shape_base_us"] = per_base
            base_methods = _per_shape_methods(base_art)
            if base_methods:
                metrics["per_shape_base_method"] = base_methods
                uniq_base = sorted({m for m in base_methods if m})
                if uniq_base:
                    metrics["timing_method_base"] = (
                        uniq_base[0] if len(uniq_base) == 1 else "mixed")

            # "sticky" base is a reused profiler measurement, not a
            # different timing method.
            mg = metrics.get("timing_method_gen")
            mb = metrics.get("timing_method_base")
            if (mg and mb and mg != mb
                    and mg != "sticky" and mb != "sticky"):
                metrics["timing_method_mismatch"] = {"gen": mg, "base": mb}

            per_speedup = [
                (b / g) if (_finite(b) and _finite(g)) else None
                for b, g in zip(per_base, per_gen)
            ]
            metrics["per_shape_speedup"] = per_speedup
            bad = [i for i, s in enumerate(per_speedup) if not _finite(s)]
            if bad:
                metrics["per_shape_speedup_bad_cases"] = bad

            # Aggregation contract: latency = arithmetic mean; speedup =
            # geometric mean of per-shape ratios. Single shape collapses
            # to scalar base/gen.
            valid = [s for s in per_speedup if _finite(s)]
            if valid:
                metrics["speedup_vs_ref"] = math.exp(
                    sum(math.log(s) for s in valid) / len(valid))
                metrics["speedup_aggregation"] = "geomean"

        descs = [s.get("case_desc") for s in (gen_art.get("per_shape") or [])
                 if isinstance(s, dict)]
        if any(descs):
            metrics["per_shape_descs"] = descs

    # --- pass-through scalars from profile_resp -----------------------
    _PROFILE_RESP_RESERVED = {"success", "log", "gen_time", "base_time",
                              "speedup", "artifacts", "task_id", "returncode"}
    for k, v in profile_resp.items():
        if k not in _PROFILE_RESP_RESERVED and isinstance(v, (int, float)):
            metrics[k] = v

    # --- verify failure detail ----------------------------------------
    # The verify-script template emits failed_indices / worst_case /
    # worst_max_abs_diff. Surfacing them lets DIAGNOSE / EDIT pinpoint
    # which shape the kernel is mis-handling without scraping stderr.
    if not verify_ok and verify_json:
        n_cases = verify_json.get("num_cases")
        if isinstance(n_cases, int) and n_cases >= 1:
            failed_idx = verify_json.get("failed_indices") or []
            if isinstance(failed_idx, list):
                metrics["correctness_failed_cases"] = failed_idx[:30]
                metrics["correctness_failed_count"] = len(failed_idx)
                metrics["correctness_total_cases"] = n_cases
            worst_idx = verify_json.get("worst_idx")
            if isinstance(worst_idx, int):
                metrics["correctness_worst_case"] = worst_idx
            worst_max = verify_json.get("worst_max_abs_diff")
            if isinstance(worst_max, (int, float)):
                metrics["correctness_worst_max_abs"] = worst_max

    if outcome == EvalOutcome.OK:
        error = None
    elif outcome == EvalOutcome.INFRA_FAIL:
        error = (f"reference.py failed: "
                 f"{verify_json.get('error') or '(no detail)'}")
    elif not verify_ok:
        error = "kernel output != reference"
    else:
        error = (f"kernel crashed during profile on {len(crashed_shapes)} "
                 f"of {len(per_gen)} shapes")

    profile_log = profile_resp.get("log", "")
    return EvalResult(
        outcome=outcome,
        metrics=metrics,
        error=error,
        raw_output=(verify_log + "\n" + profile_log)[-4096:],
        error_source=error_source,
    )
