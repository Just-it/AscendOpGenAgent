#!/usr/bin/env python3
"""Static eval script — replaces the old materialize-and-subprocess
pipeline (package_builder + verify_<op>.py + profile_<op>_*.py).

Single subprocess runs any subset of {verify, profile_base, profile_gen}
sequentially, so the triton JIT cache populated during verify is reused
when profile_gen runs (warm start). Output: one JSON object on stdout
with each requested phase's result, mirroring the schemas the old
generated scripts emitted so eval_client._assemble_eval_result reads
them unchanged.

Standalone reproducer:
    python eval_kernel.py --task-dir <task_dir> --op-name <op> \\
        --kernel-file kernel --ref-file reference \\
        --device-id 0 \\
        --warmup 10 --repeats 100 --phases verify,profile_gen,profile_base

Precision: allclose-style |diff| <= atol + rtol*|ref| per `correctness.py`
(per-dtype rtol+atol).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _to_device(seq, dev):
    return [x.to(dev) if hasattr(x, "to") else x for x in seq]


def _to_cpu_list(out):
    import torch
    if isinstance(out, torch.Tensor):
        return [out.detach().cpu()]
    if isinstance(out, (list, tuple)):
        return [o.detach().cpu() if hasattr(o, "detach") else o for o in out]
    return [out]


def _build_target(target_cls, init_inputs, device):
    m = target_cls(*init_inputs)
    if hasattr(m, "to"):
        m = m.to(device)
    if hasattr(m, "eval"):
        m = m.eval()
    return m


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def run_verify(ref_mod, kernel_mod, cases_cpu, init_inputs, device) -> dict:
    """Match package_builder._gen_verify_script's emit shape."""
    import torch
    from correctness import compare_outputs_per_case
    from input_groups import describe_case as _describe_case

    Model = ref_mod.Model
    ModelNew = kernel_mod.ModelNew

    out_ref_per_case = []
    model_ref = _build_target(Model, init_inputs, device)
    with torch.no_grad():
        for case in cases_cpu:
            inp = _to_device(case, device)
            out_ref_per_case.append(_to_cpu_list(model_ref(*inp)))
            del inp
    del model_ref
    torch.npu.empty_cache()

    out_new_per_case = []
    model_new = _build_target(ModelNew, init_inputs, device)
    with torch.no_grad():
        for case in cases_cpu:
            inp = _to_device(case, device)
            out_new_per_case.append(_to_cpu_list(model_new(*inp)))
            del inp

    cmp_result = compare_outputs_per_case(out_ref_per_case, out_new_per_case)

    for d in cmp_result["diagnostics"]:
        print(d, file=sys.stderr)

    if not cmp_result["correctness"]:
        failed = [pc["idx"] for pc in cmp_result["per_case"]
                  if not pc["correctness"]]
        shape_strs = []
        for i in failed[:10]:
            if 0 <= i < len(cases_cpu):
                shape_strs.append(
                    f"case {i}={_describe_case(cases_cpu[i], model_new)}")
        if shape_strs:
            suffix = " ..." if len(failed) > 10 else ""
            print("[verify] FAILED_SHAPES: " + "; ".join(shape_strs) + suffix,
                  file=sys.stderr)

    return {
        "correctness": cmp_result["correctness"],
        "ref_source": "computed",
        "num_cases": len(cases_cpu),
        "per_case": cmp_result["per_case"],
        "diagnostics": cmp_result["diagnostics"],
        "failed_indices": cmp_result.get("failed_indices", []),
        "worst_idx": cmp_result.get("worst_idx"),
        "worst_max_abs_diff": cmp_result.get("worst_max_abs_diff"),
    }


def _bench_one_case(impl_model, inputs, warmup: int, repeats: int) -> float:
    """Wall-clock timer: warmup, then perf_counter between syncs.
    Same approach as the legacy profile_<op>_*.py templates and as
    skills/triton/kernel-verifier/scripts/benchmark.py's fallback path.
    """
    import torch
    with torch.no_grad():
        for _ in range(warmup):
            impl_model(*inputs)
        torch.npu.synchronize()
        times_us = []
        for _ in range(repeats):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            impl_model(*inputs)
            torch.npu.synchronize()
            times_us.append((time.perf_counter() - t0) * 1e6)
    return sum(times_us) / len(times_us)


def run_profile(target_cls, ref_mod, cases_cpu, init_inputs, device,
                warmup: int, repeats: int, mode: str) -> dict:
    """Match package_builder._gen_profile_script's emit shape (and the
    legacy *_profile_result.json file content)."""
    import torch
    from input_groups import describe_case as _describe_case

    per_shape = []
    for idx, case in enumerate(cases_cpu):
        model = target_cls(*init_inputs)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        inputs = _to_device(case, device)
        try:
            avg_us = _bench_one_case(model, inputs, warmup, repeats)
            if avg_us is None or avg_us <= 0 or avg_us == float("inf"):
                raise RuntimeError(f"benchmark returned invalid avg_us={avg_us!r}")
        except Exception as e:
            print(f"[profile {mode}] case {idx} benchmark failed: {e} "
                  f"(case marked inf so it doesn't poison the aggregate)",
                  file=sys.stderr)
            traceback.print_exc()
            avg_us = float("inf")
        per_shape.append({
            "idx": idx,
            "case_desc": _describe_case(case, model),
            "avg_time_us": avg_us,
        })
        del model, inputs
        torch.npu.empty_cache()

    # Aggregate over the cases that actually produced a finite timing.
    # The per-case `try/except` above marks crashed cases as `inf` so the
    # error doesn't bubble up and abort the whole profile pass; the comment
    # there used to claim "marked inf so it doesn't poison the aggregate"
    # but the old `sum(avgs) / len(avgs)` was inf-poisoned regardless. Filter
    # explicitly here. `eval_client` already filters per-shape speedups the
    # same way for the geomean.
    import math as _math
    avgs = [s["avg_time_us"] for s in per_shape]
    finite = [a for a in avgs if isinstance(a, (int, float)) and _math.isfinite(a)]
    agg = sum(finite) / len(finite) if finite else float("inf")
    return {
        "avg_time_us": agg,
        "execution_time_us": agg,
        "execution_time_ms": (agg / 1000.0) if finite else None,
        "warmup_times": warmup,
        "run_times": repeats,
        "num_cases": len(per_shape),
        "per_shape": per_shape,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _wrap_phase_error(phase: str, e: Exception) -> dict:
    return {
        "error": f"{type(e).__name__}: {e}",
        "traceback": traceback.format_exc(),
        "phase": phase,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--op-name", required=True)
    ap.add_argument("--kernel-file", required=True,
                    help="basename of kernel module without .py (e.g. 'kernel')")
    ap.add_argument("--ref-file", required=True,
                    help="basename of reference module without .py (e.g. 'reference')")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--phases", default="verify,profile_gen,profile_base",
                    help="comma-separated subset of "
                         "{verify, profile_gen, profile_base}")
    ap.add_argument("--output", default=None,
                    help="path to write the result JSON. If omitted, "
                         "writes to <task_dir>/.eval_result.json. The C "
                         "runtime (CANN) emits warnings on stdout that "
                         "concatenate without newlines, which makes "
                         "stdout JSON unreliable; using a file sidesteps "
                         "that.")
    args = ap.parse_args()

    requested = {p.strip() for p in args.phases.split(",") if p.strip()}
    valid = {"verify", "profile_gen", "profile_base"}
    bad = requested - valid
    if bad:
        print(f"unknown phase(s): {sorted(bad)}; valid: {sorted(valid)}",
              file=sys.stderr)
        sys.exit(2)

    task_dir = os.path.abspath(args.task_dir)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    for p in (scripts_dir, task_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    # Device id has to be set before torch_npu init.
    device_id = int(os.environ.get("DEVICE_ID", args.device_id))
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))
    import torch
    import torch_npu  # noqa: F401
    import triton  # noqa: F401
    import triton.language  # noqa: F401

    device = torch.device("npu:0")

    # Load ref + kernel modules from task_dir.
    ref_path = os.path.join(task_dir, args.ref_file + ".py")
    kernel_path = os.path.join(task_dir, args.kernel_file + ".py")

    result: dict = {"verify": None, "profile_base": None, "profile_gen": None,
                    "ok": True, "errors": []}
    out_path = args.output or os.path.join(task_dir, ".eval_result.json")

    def _write_and_exit(rc: int) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, default=str)
        print(f"[eval_kernel] result -> {out_path}", file=sys.stderr)
        sys.exit(rc)

    try:
        ref_mod = _load_module("_eval_ref", ref_path)
    except Exception as e:
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("import_ref", e))
        _write_and_exit(1)

    # Resolve cases once — all phases iterate the same case list.
    try:
        from input_groups import resolve as _resolve_groups
        cases_cpu = _resolve_groups(ref_mod)
        init_inputs = ref_mod.get_init_inputs()
        if not cases_cpu:
            raise RuntimeError("reference module returned 0 input cases")
    except Exception as e:
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("resolve_cases", e))
        _write_and_exit(1)

    # Kernel only required for verify / profile_gen.
    kernel_mod = None
    if "verify" in requested or "profile_gen" in requested:
        try:
            kernel_mod = _load_module("_eval_kernel", kernel_path)
        except Exception as e:
            # Verify reports correctness=False so eval_client surfaces a
            # clean failure; profile_gen is skipped.
            result["errors"].append(_wrap_phase_error("import_kernel", e))
            if "verify" in requested:
                result["verify"] = {
                    "correctness": False,
                    "error": f"import kernel failed: {type(e).__name__}: {e}",
                    "num_cases": len(cases_cpu),
                    "per_case": [], "diagnostics": [], "failed_indices": [],
                }
            requested.discard("verify")
            requested.discard("profile_gen")

    # ---- verify ----
    if "verify" in requested:
        try:
            result["verify"] = run_verify(
                ref_mod, kernel_mod, cases_cpu, init_inputs, device)
        except Exception as e:
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("verify", e))
            result["verify"] = {
                "correctness": False,
                "error": f"{type(e).__name__}: {e}",
                "num_cases": len(cases_cpu),
                "per_case": [], "diagnostics": [], "failed_indices": [],
            }

    # ---- profile_base ----
    if "profile_base" in requested:
        try:
            result["profile_base"] = run_profile(
                ref_mod.Model, ref_mod, cases_cpu, init_inputs, device,
                args.warmup, args.repeats, mode="base")
        except Exception as e:
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("profile_base", e))

    # ---- profile_gen ----
    if "profile_gen" in requested:
        try:
            result["profile_gen"] = run_profile(
                kernel_mod.ModelNew, ref_mod, cases_cpu, init_inputs, device,
                args.warmup, args.repeats, mode="generation")
        except Exception as e:
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("profile_gen", e))

    _write_and_exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
