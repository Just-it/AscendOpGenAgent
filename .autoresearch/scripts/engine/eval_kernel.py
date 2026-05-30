#!/usr/bin/env python3
"""Single-subprocess orchestrator: verify + profile_gen (+ profile_base).

The actual verification + measurement logic lives in the kernel-verifier
skill — this script does NOT reimplement either:

    skills/triton/kernel-verifier/scripts/verify.py     run_single_case
    skills/triton/kernel-verifier/scripts/benchmark.py  measure_single

Timing routes through skill `run_single_benchmark` (mid layer, not
low-level `measure_single`) so skill stays the single source of truth
for profiler config, fallback ladder, and per-case progress logs. Warm
rounds (profile_gen only) use `skip_framework=True`; cold rounds pair
fw + impl in one call. `profile_base` alone — used by autoresearch's
two-subprocess design (a ref pass isolated so kernel UB faults can't
take ref timing down with them) — drops to `measure_single` direct
since the skill mid layer has no impl-skipped mode.

`measure_single` pays ~3 sec/case in CANN parse — NPU events would be
faster but read 10-16× high on small kernels (host dispatch gaps), so
we accept the slow eval for honest absolute numbers.

Why a single-subprocess orchestrator instead of just spawning the two
skill CLIs separately?

  - Triton JIT cache populated during verify is reused by profile_gen
    in the SAME process (warm start). Splitting into two subprocesses
    means cold compilation each time — minutes on triton-ascend kernels
    with constexpr specializations.

  - autoresearch's eval_client._assemble_eval_result expects ONE JSON
    sidecar with {verify, profile_base, profile_gen} keys; this script
    glues the three skill calls into that schema. The skill CLIs each
    produce their own JSON shape and don't know about autoresearch's
    contract.

  - File naming: autoresearch uses `reference.py` + `kernel.py` per
    task_dir; the skill CLIs use `<op>_torch.py` + `<op>_<dsl>_impl.py`.
    Loading via importlib.util sidesteps that mismatch — we hand
    already-loaded module objects + cases to the skill functions.

Standalone reproducer:
    python .autoresearch/scripts/engine/eval_kernel.py \\
        --task-dir <task_dir> --op-name <op> \\
        --kernel-file kernel --ref-file reference \\
        --device-id 0 \\
        --warmup 10 --repeats 100 --phases verify,profile_gen,profile_base

Precision: per-dtype rtol+atol; fp16 / bf16 / fp32 thresholds in
skill verify.compare(). autoresearch's metric is profiler-based device
op time (skill benchmark.measure_single) — directly comparable to
numbers agents quote from running the kernel-verifier skill standalone.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback


# ---------------------------------------------------------------------------
# Module + skill loaders
# ---------------------------------------------------------------------------

def _load_module(name: str, path: str):
    """Load a Python file as a module under `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_skill_modules() -> tuple:
    """Load skill verify.py + benchmark.py under unique names so the
    generic basenames `verify` / `benchmark` can't shadow (or be shadowed
    by) other modules with the same names elsewhere on sys.path
    (e.g. .autoresearch/scripts/batch/verify.py)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # engine/ -> scripts/ -> .autoresearch/ -> repo root
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    skill_scripts = os.path.join(
        repo_root, "skills", "triton", "kernel-verifier", "scripts",
    )
    if not os.path.isdir(skill_scripts):
        raise RuntimeError(
            f"kernel-verifier skill not found at {skill_scripts!r} — "
            f"eval_kernel orchestrator depends on it. Layout assumption: "
            f"skills/ is a sibling of .autoresearch/."
        )
    verify_mod = _load_module(
        "_skill_verify", os.path.join(skill_scripts, "verify.py"))
    bench_mod = _load_module(
        "_skill_benchmark", os.path.join(skill_scripts, "benchmark.py"))
    return verify_mod, bench_mod


# ---------------------------------------------------------------------------
# Per-target build helpers
# ---------------------------------------------------------------------------

def _build_target(target_cls, init_inputs, device):
    m = target_cls(*init_inputs)
    if hasattr(m, "to"):
        m = m.to(device)
    if hasattr(m, "eval"):
        m = m.eval()
    return m


def _seed_npu():
    import torch
    torch.manual_seed(0)
    try:
        torch.npu.manual_seed(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def run_verify(ref_mod, kernel_mod, cases_cpu, init_inputs, device,
               skill_verify) -> dict:
    """Per-case correctness check with ref-side vs kernel-side blame.

    The verify loop is split into three independently-caught phases per
    case so failures can be attributed (error_source = "ref" | "kernel"):

      1. Build ref model + forward         → error_source="ref"
      2. Build kernel model + forward      → error_source="kernel"
      3. Compare ref vs kernel outputs     → error_source="kernel"
         (a comparison failure is a kernel numerical issue; the ref
         outputs from phase 1 are the ground truth.)

    Scaffold reads the resulting `error_source` on the verify dict and
    refuses to activate the task when it's "ref" — the user must fix
    the source --ref file. "kernel" failures take the normal seed-fail
    recovery path (PLAN -> EDIT rewrite).

    Schema mirrors the legacy shape `eval_client._assemble_eval_result`
    consumes: {correctness, error_source, num_cases, failed_indices,
    per_case, diagnostics}.
    """
    from utils.input_groups import describe_case as _describe_case
    from utils.correctness import compare_outputs

    Model = ref_mod.Model
    ModelNew = kernel_mod.ModelNew

    num_cases = len(cases_cpu)
    per_case: list = []
    failed_indices: list[int] = []
    diagnostics: list[str] = []
    overall_correct = True
    # Track which side broke. "ref" takes priority over "kernel" because
    # a ref failure invalidates the whole eval regardless of what the
    # kernel does. None when verify passes.
    overall_error_source: str | None = None

    def _tag_ref_fail() -> None:
        nonlocal overall_error_source
        # ref-side errors override kernel-side ones — see docstring.
        overall_error_source = "ref"

    def _tag_kernel_fail() -> None:
        nonlocal overall_error_source
        if overall_error_source != "ref":
            overall_error_source = "kernel"

    def _to_cpu_list(out):
        import torch
        if isinstance(out, torch.Tensor):
            return [out.detach().cpu()]
        if isinstance(out, (list, tuple)):
            return [o.detach().cpu() if hasattr(o, "detach") else o
                    for o in out]
        return [out]

    import torch  # noqa: F401  — used by _to_cpu_list above

    for idx, inputs in enumerate(cases_cpu):
        framework_model = None
        impl_model = None
        case_desc = _describe_case(inputs, None)
        case_entry = {
            "idx": idx,
            "case_desc": case_desc,
            "correctness": False,
            "error": None,
            "error_source": None,
        }

        # --- Phase 1: ref build + forward ---
        out_ref_cpu = None
        try:
            _seed_npu()
            framework_model = _build_target(Model, init_inputs, device)
            ref_inputs_dev = [x.to(device) if hasattr(x, "to") else x
                              for x in inputs]
            with torch.no_grad():
                out_ref = framework_model(*ref_inputs_dev)
            out_ref_cpu = _to_cpu_list(out_ref)
            del ref_inputs_dev, out_ref
        except Exception as e:
            overall_correct = False
            failed_indices.append(idx)
            _tag_ref_fail()
            err = f"{type(e).__name__}: {e}"
            case_entry["error"] = f"ref-side: {err}"
            case_entry["error_source"] = "ref"
            diagnostics.append(skill_verify.truncate_error(
                f"[case {idx}] ref-side: {err}\n{traceback.format_exc()}"
            ))
            del framework_model
            skill_verify.cleanup_npu_memory()
            per_case.append(case_entry)
            continue

        # Free ref model before building kernel; HBM doesn't fit both at
        # once for BatchNorm-scale tensors.
        del framework_model
        framework_model = None
        skill_verify.cleanup_npu_memory()

        # --- Phase 2: kernel build + forward ---
        out_new_cpu = None
        try:
            _seed_npu()
            impl_model = _build_target(ModelNew, init_inputs, device)
            new_inputs_dev = [x.to(device) if hasattr(x, "to") else x
                              for x in inputs]
            with torch.no_grad():
                out_new = impl_model(*new_inputs_dev)
            out_new_cpu = _to_cpu_list(out_new)
            del new_inputs_dev, out_new
        except Exception as e:
            overall_correct = False
            failed_indices.append(idx)
            _tag_kernel_fail()
            err = f"{type(e).__name__}: {e}"
            case_entry["error"] = f"kernel-side: {err}"
            case_entry["error_source"] = "kernel"
            diagnostics.append(skill_verify.truncate_error(
                f"[case {idx}] kernel-side: {err}\n{traceback.format_exc()}"
            ))
            del impl_model
            skill_verify.cleanup_npu_memory()
            per_case.append(case_entry)
            continue

        # --- Phase 3: compare ---
        # compare_outputs (utils/correctness.py) delegates to the skill's
        # _check_accuracy_allclose for the actual element-wise check so
        # per-dtype tolerances stay in lockstep with batch verify.
        try:
            cmp = compare_outputs(list(out_ref_cpu), list(out_new_cpu))
            if cmp["correctness"]:
                case_entry["correctness"] = True
            else:
                overall_correct = False
                failed_indices.append(idx)
                _tag_kernel_fail()
                case_entry["error_source"] = "kernel"
                case_entry["error"] = "kernel output != reference"
                for d in cmp["diagnostics"]:
                    diagnostics.append(f"[case {idx}] {d}")
        except Exception as e:
            overall_correct = False
            failed_indices.append(idx)
            _tag_kernel_fail()
            err = f"{type(e).__name__}: {e}"
            case_entry["error"] = f"compare: {err}"
            case_entry["error_source"] = "kernel"
            diagnostics.append(skill_verify.truncate_error(
                f"[case {idx}] compare: {err}\n{traceback.format_exc()}"
            ))
        finally:
            del impl_model
            skill_verify.cleanup_npu_memory()

        per_case.append(case_entry)

    return {
        "correctness": overall_correct,
        "error_source": overall_error_source,  # "ref" | "kernel" | None
        "ref_source": "computed",
        "num_cases": num_cases,
        "failed_indices": failed_indices,
        "per_case": per_case,
        "diagnostics": diagnostics,
        # Worst-case detail comes from compare_outputs in utils.correctness;
        # skill's allclose path does not surface them. Keep keys present so
        # eval_client doesn't trip KeyError on the failure-detail path.
        "worst_idx": None,
        "worst_max_abs_diff": None,
    }


def _aggregate_per_shape(per_shape: list, warmup: int, repeats: int) -> dict:
    """Shared aggregator: mean over finite per-case timings. Matches the
    geomean filter `eval_client` applies for per-shape speedup."""
    import math
    finites = [s["avg_time_us"] for s in per_shape
               if isinstance(s["avg_time_us"], (int, float))
               and math.isfinite(s["avg_time_us"])]
    agg_us = sum(finites) / len(finites) if finites else float("inf")
    return {
        "avg_time_us": agg_us,
        "execution_time_us": agg_us,
        "execution_time_ms": (agg_us / 1000.0) if finites else None,
        "warmup_times": warmup,
        "run_times": repeats,
        "num_cases": len(per_shape),
        "per_shape": per_shape,
    }


def run_profile_ref_only(ref_mod, cases_cpu, init_inputs, device,
                         warmup: int, repeats: int,
                         skill_benchmark) -> dict:
    """Ref-side only — used by autoresearch's isolated ref subprocess.

    Skill's mid layer requires an impl_model and always measures it, so
    we drop to `measure_single` directly here. Same per-case skeleton
    as the paired path; just one model and one timing per case.
    """
    from utils.input_groups import describe_case as _describe_case
    per_shape: list = []
    for idx, case in enumerate(cases_cpu):
        model = None
        try:
            _seed_npu()
            model = _build_target(ref_mod.Model, init_inputs, device)
            inputs_dev = [x.to(device) if hasattr(x, "to") else x for x in case]
            _, ms, _ = skill_benchmark.measure_single(
                model, inputs_dev, warmup, repeats,
                f"ar_eval_base_case{idx}", device=device,
            )
            us = (float(ms) * 1000.0
                  if isinstance(ms, (int, float)) and ms > 0 and ms == ms
                  else float("inf"))
            method = "profiler" if us != float("inf") else None
        except Exception as e:
            print(f"[profile_base] case {idx}: {e}", file=sys.stderr)
            traceback.print_exc()
            us, method = float("inf"), None
        per_shape.append({"idx": idx, "case_desc": _describe_case(case, model),
                          "avg_time_us": us, "method": method})
        del model
        skill_benchmark.cleanup_npu_memory()
    return _aggregate_per_shape(per_shape, warmup, repeats)


def run_profile_paired(ref_mod, kernel_mod, cases_cpu, init_inputs, device,
                       warmup: int, repeats: int,
                       do_base: bool, skill_benchmark) -> dict:
    """Paired pass via skill `run_single_benchmark` per case.

    Skill's mid layer always measures impl (no impl-skipped mode), so
    profile_gen is implicit. `do_base` toggles `skip_framework`: cold
    round measures both sides, warm round only impl. Returns
    `{"base": <profile_dict>|None, "gen": <profile_dict>}`.

    Ref-only subprocess (without kernel_mod) goes through `run_profile_ref_only`
    instead — see module docstring.
    """
    from utils.input_groups import describe_case as _describe_case

    config = skill_benchmark.BenchmarkConfig(
        op_name="ar_eval", verify_dir="",
        warmup=warmup, repeats=repeats,
        skip_framework=not do_base, framework_latency_ms=0.0,
    )

    def _row(idx: int, desc: str, ms) -> dict:
        us = (float(ms) * 1000.0
              if isinstance(ms, (int, float)) and ms > 0 and ms == ms
              else float("inf"))
        return {"idx": idx, "case_desc": desc, "avg_time_us": us,
                "method": "profiler" if us != float("inf") else None}

    base_rows: list = []
    gen_rows: list = []
    for idx, case in enumerate(cases_cpu):
        fw_model = None
        impl_model = None
        try:
            _seed_npu()
            impl_model = _build_target(kernel_mod.ModelNew, init_inputs, device)
            if do_base:
                fw_model = _build_target(ref_mod.Model, init_inputs, device)
            inputs_dev = [x.to(device) if hasattr(x, "to") else x for x in case]
            # skill insists on a non-None framework arg even when skipping;
            # impl_model is a safe placeholder (its forward isn't invoked
            # from the fw path under skip_framework=True).
            fw_perf, impl_perf, _ = skill_benchmark.run_single_benchmark(
                fw_model or impl_model, impl_model, inputs_dev,
                config, device, idx + 1, len(cases_cpu),
            )
            desc = _describe_case(case, impl_model)
            gen_rows.append(_row(idx, desc, impl_perf.avg_latency_ms))
            if do_base:
                base_rows.append(_row(idx, desc, fw_perf.avg_latency_ms))
        except Exception as e:
            print(f"[profile] case {idx}: {e}", file=sys.stderr)
            traceback.print_exc()
            desc = _describe_case(case, fw_model or impl_model)
            gen_rows.append(_row(idx, desc, None))
            if do_base:
                base_rows.append(_row(idx, desc, None))
        finally:
            del fw_model, impl_model
            skill_benchmark.cleanup_npu_memory()

    return {
        "base": _aggregate_per_shape(base_rows, warmup, repeats) if do_base else None,
        "gen": _aggregate_per_shape(gen_rows, warmup, repeats),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _wrap_phase_error(phase: str, e: Exception) -> dict:
    return {
        "phase": phase,
        "type": type(e).__name__,
        "msg": str(e),
        "trace": traceback.format_exc(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="autoresearch eval orchestrator (verify + profile)")
    ap.add_argument("--task-dir", required=True,
                    help="task directory containing reference.py + kernel.py")
    ap.add_argument("--op-name", required=True,
                    help="operator name (recorded in result; not used for "
                         "file naming since we load by path)")
    ap.add_argument("--kernel-file", required=True,
                    help="kernel module name without .py (default convention: kernel)")
    ap.add_argument("--ref-file", required=True,
                    help="reference module name without .py (default convention: reference)")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--phases", default="verify,profile_gen,profile_base",
                    help="comma-separated subset of {verify, profile_gen, "
                         "profile_base}; run order matters because verify "
                         "warms the JIT cache profile_gen reuses")
    ap.add_argument("--output", default=None,
                    help="JSON sidecar path (default: <task_dir>/.eval_result.json)")
    args = ap.parse_args()

    requested = {p.strip() for p in args.phases.split(",") if p.strip()}
    valid = {"verify", "profile_gen", "profile_base"}
    bad = requested - valid
    if bad:
        print(f"unknown phase(s): {sorted(bad)}; valid: {sorted(valid)}",
              file=sys.stderr)
        sys.exit(2)

    task_dir = os.path.abspath(args.task_dir)
    # __file__ is scripts/engine/eval_kernel.py — climb one to scripts/ so
    # `from utils.input_groups import describe_case` resolves at runtime.
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (scripts_dir, task_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    # Load skill core functions (verify.run_single_case + benchmark.cleanup_npu_memory)
    # before touching torch — surfaces a missing-skill setup error early.
    try:
        skill_verify, skill_benchmark = _load_skill_modules()
    except Exception as e:
        print(f"[eval_kernel] failed to load kernel-verifier skill: {e}",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Device id has to be set before torch_npu init.
    device_id = int(os.environ.get("DEVICE_ID", args.device_id))
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))
    import torch
    import torch_npu  # noqa: F401
    import triton  # noqa: F401
    import triton.language  # noqa: F401

    device = torch.device("npu:0")

    ref_path = os.path.join(task_dir, args.ref_file + ".py")
    kernel_path = os.path.join(task_dir, args.kernel_file + ".py")

    result: dict = {
        "verify": None,
        "profile_base": None,
        "profile_gen": None,
        "ok": True,
        "errors": [],
    }
    out_path = args.output or os.path.join(task_dir, ".eval_result.json")

    def _write_and_exit(rc: int) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, default=str)
        print(f"[eval_kernel] result -> {out_path}", file=sys.stderr)
        sys.exit(rc)

    try:
        ref_mod = _load_module("_eval_ref", ref_path)
    except Exception as e:
        # ref-side: source file broken. Scaffold rejects + asks user to
        # fix the source --ref. Surface the verify block too so eval_client
        # can read error_source through the normal path.
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("import_ref", e))
        result["verify"] = {
            "correctness": False,
            "error_source": "ref",
            "error": f"import reference failed: {type(e).__name__}: {e}",
            "num_cases": 0,
            "per_case": [],
            "diagnostics": [],
            "failed_indices": [],
        }
        _write_and_exit(1)

    try:
        from utils.input_groups import resolve as _resolve_groups
        cases_cpu = _resolve_groups(ref_mod)
        init_inputs = ref_mod.get_init_inputs()
        if not cases_cpu:
            raise RuntimeError("reference module returned 0 input cases")
    except Exception as e:
        # ref-side: get_input_groups / get_inputs / get_init_inputs crashed
        # or returned nothing. Same recovery as import_ref above.
        result["ok"] = False
        result["errors"].append(_wrap_phase_error("resolve_cases", e))
        result["verify"] = {
            "correctness": False,
            "error_source": "ref",
            "error": f"resolve cases failed: {type(e).__name__}: {e}",
            "num_cases": 0,
            "per_case": [],
            "diagnostics": [],
            "failed_indices": [],
        }
        _write_and_exit(1)

    kernel_mod = None
    if "verify" in requested or "profile_gen" in requested:
        try:
            kernel_mod = _load_module("_eval_kernel", kernel_path)
        except Exception as e:
            # kernel-side: ModelNew import / module-level code crashed.
            # eval_client treats this as a correctness=False round; PLAN
            # takes over to rewrite the kernel. profile_gen is impossible
            # without a kernel module.
            result["errors"].append(_wrap_phase_error("import_kernel", e))
            if "verify" in requested:
                result["verify"] = {
                    "correctness": False,
                    "error_source": "kernel",
                    "error": f"import kernel failed: {type(e).__name__}: {e}",
                    "num_cases": len(cases_cpu),
                    "per_case": [],
                    "diagnostics": [],
                    "failed_indices": [],
                }
            requested.discard("verify")
            requested.discard("profile_gen")

    # ---- verify (warms JIT cache for profile_gen) ----
    if "verify" in requested:
        try:
            result["verify"] = run_verify(
                ref_mod, kernel_mod, cases_cpu, init_inputs, device,
                skill_verify=skill_verify)
        except Exception as e:
            # run_verify itself crashed (not a per-case failure caught
            # internally). Without more info we conservatively tag this
            # as kernel-side; ref-only crashes would be caught by Phase 1
            # of the per-case loop and surface with error_source="ref".
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("verify", e))
            result["verify"] = {
                "correctness": False,
                "error_source": "kernel",
                "error": f"{type(e).__name__}: {e}",
                "num_cases": len(cases_cpu),
                "per_case": [],
                "diagnostics": [],
                "failed_indices": [],
            }

    # ---- profile ----
    # autoresearch isolates ref into its own subprocess so kernel UB
    # faults can't take ref timing down — handle that "base only" path
    # via `run_profile_ref_only`. With kernel_mod present, paired through
    # skill.run_single_benchmark; warm JIT cache from verify carries over.
    do_gen = "profile_gen" in requested
    do_base = "profile_base" in requested
    if do_gen:
        try:
            prof = run_profile_paired(
                ref_mod, kernel_mod, cases_cpu, init_inputs, device,
                warmup=args.warmup, repeats=args.repeats,
                do_base=do_base, skill_benchmark=skill_benchmark,
            )
            result["profile_gen"] = prof["gen"]
            if do_base:
                result["profile_base"] = prof["base"]
        except Exception as e:
            result["ok"] = False
            phase = "profile_gen+base" if do_base else "profile_gen"
            result["errors"].append(_wrap_phase_error(phase, e))
    elif do_base:
        try:
            result["profile_base"] = run_profile_ref_only(
                ref_mod, cases_cpu, init_inputs, device,
                warmup=args.warmup, repeats=args.repeats,
                skill_benchmark=skill_benchmark,
            )
        except Exception as e:
            result["ok"] = False
            result["errors"].append(_wrap_phase_error("profile_base", e))

    _write_and_exit(0)


if __name__ == "__main__":
    main()
