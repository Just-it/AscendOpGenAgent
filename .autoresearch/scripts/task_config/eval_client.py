"""Eval dispatcher.

Public entry point: `run_eval(task_dir, config, device_id=None) ->
EvalResult`. Drives a single `eval_kernel.py` subprocess (via
`eval_runner.local_eval`) that runs verify + profile_gen — and
profile_base when no sticky baseline is recorded — in one shot.
Assembles the verify+profile responses into an `EvalResult`. If no
usable Ascend runtime is detected, run_eval returns an EvalResult with
a clear error instead of dispatching.

What lives here:
  - Result assembly (`_assemble_eval_result`).
  - Per-shape timeout scaling (`_count_ref_cases`, `_effective_timeout`).
  - Sticky baseline lookup (`_override_base_from_progress`).
  - Eval entry points (`run_local_eval`, `run_eval`).

What's NOT here:
  - The eval subprocess body itself — that's `eval_kernel.py` in
    `.autoresearch/scripts/`.
  - EvalResult / improvement / constraints — those live in metric_policy.
  - YAML parsing — those live in loader.
"""
import json
import math
import os
import sys
from typing import Optional

from .loader import TaskConfig
from .metric_policy import EvalResult


# ---------------------------------------------------------------------------
# Per-shape eval_timeout scaling
# ---------------------------------------------------------------------------

def _count_ref_cases(task_dir: str, config: TaskConfig) -> int:
    """Probe the ref module locally and count input cases.

    Mirrors what the generated verify script does: import ref + run
    input_groups.resolve, which duck-types between get_input_groups
    (multi-shape, NPUKernelBench) and get_inputs (single-shape collapsed
    to N=1). Used purely to scale eval_timeout — any failure falls back
    to 1 (single-shape semantics, no scaling).

    Cost: O(materialise all input tensors) once per eval call. Bounded
    by case set size; ~100ms-1s for typical multi-shape benches and
    swamped by the verify/profile call that follows.
    """
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    ref_path = os.path.join(task_dir, config.ref_file)
    if not os.path.isfile(ref_path):
        return 1
    ref_dir = os.path.dirname(ref_path) or "."
    sys_path_added = ref_dir not in sys.path
    if sys_path_added:
        sys.path.insert(0, ref_dir)
    try:
        import importlib.util
        from input_groups import resolve as _resolve  # type: ignore
        spec = importlib.util.spec_from_file_location(
            f"_count_ref_{config.name}", ref_path)
        if spec is None or spec.loader is None:
            return 1
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        n = len(_resolve(mod))
        return max(n, 1)
    except Exception as e:
        print(f"[eval_client] WARN: case-count probe failed ({type(e).__name__}: "
              f"{e}); eval_timeout will not scale per shape.", file=sys.stderr)
        return 1
    finally:
        if sys_path_added:
            try:
                sys.path.remove(ref_dir)
            except ValueError:
                pass


def _effective_timeout(config: TaskConfig, num_cases: int) -> int:
    """Per-shape semantics: total = config.eval_timeout * num_cases.

    config.eval_timeout is documented as the budget for ONE shape; scaling
    keeps single-shape behaviour identical (num_cases=1) while keeping a
    multi-shape verify from being killed at the first JIT compile.
    """
    return int(config.eval_timeout) * max(int(num_cases), 1)


def _override_base_from_progress(task_dir: str) -> Optional[float]:
    """Return the sticky pytorch baseline from progress.json so the
    profile pass can skip rerunning profile_<op>_base.py. Only honoured
    when the prior baseline_init recorded baseline_source='ref'.
    """
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from phase_machine import load_progress  # type: ignore
        progress = load_progress(task_dir) or {}
    except Exception:
        return None
    if progress.get("baseline_source") != "ref":
        return None
    v = progress.get("baseline_metric")
    if isinstance(v, (int, float)) and 0 < v < float("inf"):
        return float(v)
    return None


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _last_json_line(s: str) -> dict | None:
    """Pull the final `{...}` JSON object from a multi-line log."""
    for line in reversed((s or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return None


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


def _per_shape_floats(art: dict | None) -> list | None:
    """List of `avg_time_us` values from a profile artifact, or None."""
    if not art:
        return None
    ps = art.get("per_shape")
    if not isinstance(ps, list) or not ps:
        return None
    return [(s.get("avg_time_us") if isinstance(s, dict) else None) for s in ps]


def _assemble_eval_result(verify_resp: dict, profile_resp: dict) -> EvalResult:
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
    verify_json = _last_json_line(verify_log) or {}

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
    correctness = verify_ok and not crashed_shapes

    metrics: dict = {}

    # --- timing + speedup -------------------------------------------------
    if gen_ok:
        metrics["latency_us"] = gen_time
    else:
        print(f"[eval] WARNING: no valid gen_time (got {gen_time!r}) - "
              f"kernel profile likely failed", file=sys.stderr)
    if gen_ok and base_ok:
        # ref_latency_us is the speedup anchor; only meaningful next to a
        # valid latency_us. SEED round always satisfies both - baseline.py
        # picks ref_latency_us up from there to set baseline_metric.
        metrics["ref_latency_us"] = base_time
        metrics["speedup_vs_ref"] = base_time / gen_time
    elif not base_ok:
        print(f"[eval] WARNING: no valid base_time (got {base_time!r}) - "
              f"speedup vs reference unavailable", file=sys.stderr)
    elif profile_resp.get("speedup"):
        metrics["speedup_vs_ref"] = profile_resp["speedup"]

    # --- per-shape detail -------------------------------------------------
    # Single-shape ops collapse to N=1 under the same schema (`per_shape`
    # of length 1), so downstream readers see uniform keys regardless of
    # shape count and don't need single-vs-multi branches.
    if per_gen is not None:
        metrics["num_cases"] = len(per_gen)
        metrics["per_shape_gen_us"] = per_gen
        if crashed_shapes:
            metrics["profile_crashed_cases"] = crashed_shapes[:30]
            metrics["profile_crashed_count"] = len(crashed_shapes)
        if per_base is not None and len(per_base) == len(per_gen):
            metrics["per_shape_base_us"] = per_base
            per_speedup = [
                (b / g) if (_finite(b) and _finite(g)) else None
                for b, g in zip(per_base, per_gen)
            ]
            metrics["per_shape_speedup"] = per_speedup

            # Aggregate speedup: geomean of valid (>0, finite) per-shape
            # ratios. NaN / inf / non-positive shapes drop out of the
            # geomean but their indices are recorded. For N=1 the geomean
            # equals the plain ratio set above; the override is harmless
            # and keeps `speedup_aggregation` populated uniformly.
            valid_sp = [s for s in per_speedup if _finite(s)]
            if valid_sp:
                metrics["speedup_vs_ref"] = math.exp(
                    sum(math.log(s) for s in valid_sp) / len(valid_sp))
                metrics["speedup_aggregation"] = "geomean"
            bad_sp = [i for i, s in enumerate(per_speedup) if not _finite(s)]
            if bad_sp:
                metrics["per_shape_speedup_bad_cases"] = bad_sp
        descs = [s.get("case_desc") for s in (gen_art.get("per_shape") or [])
                 if isinstance(s, dict)]
        if any(descs):
            metrics["per_shape_descs"] = descs

    # --- pass-through scalars from profile_resp ---------------------------
    _PROFILE_RESP_RESERVED = {"success", "log", "gen_time", "base_time",
                              "speedup", "artifacts", "task_id", "returncode"}
    for k, v in profile_resp.items():
        if k not in _PROFILE_RESP_RESERVED and isinstance(v, (int, float)):
            metrics[k] = v

    # --- verify failure detail (only on verify-side failure) --------------
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

    if correctness:
        error = None
    elif crashed_shapes and verify_ok:
        error = (f"profile crashed on {len(crashed_shapes)} of "
                 f"{len(per_gen)} shapes")
    else:
        error = "verify failed (kernel broken); ref profile may still be present"

    profile_log = profile_resp.get("log", "")
    return EvalResult(
        correctness=correctness,
        metrics=metrics,
        error=error,
        raw_output=(verify_log + "\n" + profile_log)[-4096:],
    )


# ---------------------------------------------------------------------------
# Local eval (subprocess transport)
# ---------------------------------------------------------------------------

def run_local_eval(task_dir: str, config: TaskConfig,
                   device_id: Optional[int] = None) -> EvalResult:
    """Drive a single `eval_kernel.py` subprocess (verify + profile_gen
    + optional profile_base) and assemble an EvalResult."""
    # eval_runner is a top-level script (one level up from this package).
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from eval_runner import local_eval

    if device_id is not None:
        dev = int(device_id)
    elif config.devices:
        dev = int(config.devices[0])
    else:
        # No explicit device on the call AND task.yaml has no `devices`
        # field - fall back to NPU 0. We emit a loud warning instead of
        # raising because legitimate callers (notebooks, ad-hoc reruns)
        # do hit this path, but a SILENT fallback to 0 is what once let
        # `--devices 6` get rewritten to 0 and OOM on a busy NPU.
        dev = 0
        print(
            "[local_eval] WARNING: no device specified (no device_id arg, "
            "no `devices` field in task.yaml). Defaulting to NPU 0. If "
            "another card is intended, pass --device-id N or set "
            "`devices: [N]` in task.yaml.",
            file=sys.stderr,
        )

    num_cases = _count_ref_cases(task_dir, config)
    eff_timeout = _effective_timeout(config, num_cases)
    if num_cases > 1:
        print(f"[local_eval] eval_timeout scaled per shape: "
              f"{config.eval_timeout}s/shape x {num_cases} cases = {eff_timeout}s",
              file=sys.stderr)

    override_base = _override_base_from_progress(task_dir)
    if override_base is not None:
        print(f"[local_eval] Skipping ref profile; sticky baseline = "
              f"{override_base:.2f} us", file=sys.stderr)

    kernel_basename = config.editable_files[0].replace(".py", "")
    ref_basename = config.ref_file.replace(".py", "")
    print("[local_eval] Running eval_kernel.py "
          f"(verify + profile_gen{'' if override_base is not None else ' + profile_base'})...",
          file=sys.stderr)
    verify_resp, profile_resp = local_eval(
        task_dir=task_dir,
        op_name=config.name,
        kernel_file=kernel_basename,
        ref_file=ref_basename,
        timeout=eff_timeout,
        device_id=dev,
        override_base_time_us=override_base,
    )
    return _assemble_eval_result(verify_resp, profile_resp)


# ---------------------------------------------------------------------------
# Unified eval entry point
# ---------------------------------------------------------------------------

def run_eval(task_dir: str, config: TaskConfig,
             device_id: Optional[int] = None) -> EvalResult:
    """Probe the Ascend runtime; on success run `run_local_eval`,
    otherwise return an EvalResult carrying a clear "no execution
    backend" error so the user can fix the local install.
    """
    _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from eval_runner import detect_local_backend
    ok, why = detect_local_backend()
    if ok:
        print(f"[eval] ascend runtime ok: {why}", file=sys.stderr)
        return run_local_eval(task_dir, config, device_id=device_id)

    return EvalResult(
        correctness=False,
        error=(
            f"ascend runtime unavailable: {why}. Install torch + "
            f"torch_npu + CANN locally."
        ),
    )
