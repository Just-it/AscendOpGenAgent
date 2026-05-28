#!/usr/bin/env python3
"""Lingxi operator benchmark with device-side event timing.

Replaces host-side time.perf_counter() with torch.npu.Event for accurate
NPU kernel timing, excluding Python dispatch overhead.

Key improvements over performance.py:
- Device-side event timing (torch.npu.Event) instead of wall-clock
- IQR-based outlier removal for robust statistics
- Geometric mean + latency-weighted mean for per-case speedup aggregation
- Welch's t-test for statistical significance of improvements
- Baseline save/load with drift detection

Usage:
    # Baseline: test both reference and ascendc, save baseline
    python .claude/skills/performance-analyzer/scripts/lingxi_perf_bench.py matmul_leakyrelu --impl all --repeat 30 \\
        --save-baseline baseline_per_case.json --output baseline_results.json

    # Evolution round: test ascendc only, compare against saved baseline
    python .claude/skills/performance-analyzer/scripts/lingxi_perf_bench.py _evo_variant_xxx --impl ascendc --repeat 30 \\
        --baseline baseline_per_case.json --output eval_results.json

    # Drift check: re-measure reference to detect baseline drift
    python .claude/skills/performance-analyzer/scripts/lingxi_perf_bench.py matmul_leakyrelu --impl reference --repeat 15 \\
        --drift-check baseline_per_case.json --drift-threshold 0.10
"""

import argparse
import copy
import importlib.util
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMPLEMENTATIONS = {
    "reference": {"filename": "model.py", "preferred_class": "Model", "label": "Reference"},
    "tilelang": {"filename": "model_new_tilelang.py", "preferred_class": "ModelNew", "label": "TileLang"},
    "ascendc": {"filename": "model_new_ascendc.py", "preferred_class": "ModelNew", "label": "AscendC"},
}

DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 30

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _has_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


# ---------------------------------------------------------------------------
# Module loading (mirrors performance.py)
# ---------------------------------------------------------------------------


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class(module, preferred_name: str):
    candidate = getattr(module, preferred_name, None)
    if inspect.isclass(candidate) and issubclass(candidate, nn.Module):
        return candidate
    for _, value in vars(module).items():
        if inspect.isclass(value) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise AttributeError(f"No nn.Module subclass found in {module.__file__}")


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------


def _robust_stats(times_ms: List[float]) -> Dict[str, Any]:
    """IQR-based outlier removal + summary statistics."""
    n_total = len(times_ms)
    if n_total < 4:
        clean = list(times_ms)
        n_outliers = 0
    else:
        sorted_t = sorted(times_ms)
        q1 = sorted_t[n_total // 4]
        q3 = sorted_t[3 * n_total // 4]
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clean = [t for t in times_ms if lower <= t <= upper]
        n_outliers = n_total - len(clean)
        if len(clean) < 3:
            clean = list(times_ms)
            n_outliers = 0

    mean_val = statistics.mean(clean)
    std_val = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return {
        "mean_ms": round(mean_val, 4),
        "median_ms": round(statistics.median(clean), 4),
        "min_ms": round(min(clean), 4),
        "max_ms": round(max(clean), 4),
        "std_ms": round(std_val, 4),
        "cv_pct": round(std_val / mean_val * 100, 2) if mean_val > 0 else 0.0,
        "n_samples": len(clean),
        "n_outliers": n_outliers,
        "raw_times_ms": [round(t, 4) for t in times_ms],
        "clean_times_ms": [round(t, 4) for t in clean],
    }


# ---------------------------------------------------------------------------
# Timing engines
# ---------------------------------------------------------------------------


def _measure_case_event(model, inputs, device, warmup: int, repeat: int) -> Dict[str, Any]:
    """Device-side event timing for a single case (NPU only)."""
    import torch_npu  # noqa: F401

    with torch.no_grad():
        # Warmup with per-iteration sync for stable frequency
        for _ in range(warmup):
            model(*inputs)
            torch.npu.synchronize()

        times_ms = []
        for _ in range(repeat):
            start_evt = torch.npu.Event(enable_timing=True)
            end_evt = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()  # drain queue before recording
            start_evt.record()
            model(*inputs)
            end_evt.record()
            end_evt.synchronize()
            times_ms.append(start_evt.elapsed_time(end_evt))

    return _robust_stats(times_ms)


def _measure_case_perf_counter(model, inputs, device, warmup: int, repeat: int) -> Dict[str, Any]:
    """Fallback: host-side perf_counter with sync (CPU / non-NPU)."""
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
            _synchronize(device)

        times_ms = []
        for _ in range(repeat):
            _synchronize(device)
            t0 = time.perf_counter()
            model(*inputs)
            _synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return _robust_stats(times_ms)


def _measure_case(model, inputs, device, warmup: int, repeat: int, use_event: bool) -> Dict[str, Any]:
    """Dispatch to event or perf_counter timing."""
    if use_event:
        return _measure_case_event(model, inputs, device, warmup, repeat)
    return _measure_case_perf_counter(model, inputs, device, warmup, repeat)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


def _welch_ttest_greater(sample_a: List[float], sample_b: List[float]) -> Tuple[float, float]:
    """Welch's t-test: H1: mean(a) > mean(b).

    Returns (t_statistic, p_value). Uses scipy if available, otherwise
    a pure-Python approximation.
    """
    try:
        from scipy.stats import ttest_ind
        t_stat, p_two = ttest_ind(sample_a, sample_b, equal_var=False)
        # one-sided: P(mean_a > mean_b)
        p_value = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        return float(t_stat), float(p_value)
    except ImportError:
        pass

    # Pure-Python fallback (Welch's t approximation)
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0
    mean_a, mean_b = statistics.mean(sample_a), statistics.mean(sample_b)
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)
    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-12
    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    # Approximate one-sided p-value using normal CDF
    # For df > 30, normal approximation is accurate.
    # For df <= 30, this underestimates p (too aggressive). Apply a conservative
    # correction: inflate p by factor (1 + 3/df) to approximate t-distribution's
    # heavier tails. This keeps us from false positives at small sample sizes.
    p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))
    if df <= 30:
        p_value = min(1.0, p_value * (1 + 3.0 / max(df, 1)))

    return float(t_stat), float(p_value)


def check_significance(
    parent_times: List[float],
    variant_times: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test if variant is significantly faster than parent.

    parent_times should be SLOWER (higher values) if variant is better.
    Tests H1: mean(parent) > mean(variant).
    """
    t_stat, p_value = _welch_ttest_greater(parent_times, variant_times)
    parent_mean = statistics.mean(parent_times)
    variant_mean = statistics.mean(variant_times)
    effect_pct = (parent_mean - variant_mean) / parent_mean * 100 if parent_mean > 0 else 0
    return {
        "significant": p_value < alpha,
        "p_value": round(p_value, 6),
        "t_statistic": round(t_stat, 4),
        "effect_size_pct": round(effect_pct, 2),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# NKB metrics (per-case speedup aggregation)
# ---------------------------------------------------------------------------


def compute_nkb_metrics(
    ref_cases: List[Dict], variant_cases: List[Dict],
) -> Dict[str, Any]:
    """Compute NPUKernelBench-style per-case metrics.

    Uses geometric mean (robust to outlier ratios) and latency-weighted mean
    (big shapes contribute proportionally to their runtime).
    """
    per_case = []
    speedups = []
    for r, v in zip(ref_cases, variant_cases):
        ref_ms = r["mean_ms"]
        var_ms = v["mean_ms"]
        spd = ref_ms / var_ms if var_ms > 0 else 0.0
        per_case.append({
            "index": r.get("index", len(per_case)),
            "ref_mean_ms": round(ref_ms, 4),
            "var_mean_ms": round(var_ms, 4),
            "speedup_vs_ref": round(spd, 4),
        })
        if spd > 0:
            speedups.append(spd)

    if not speedups:
        return {"error": "no valid speedup data", "per_case": per_case}

    # Geometric mean: robust to outlier ratios
    log_mean = statistics.mean([math.log(s) for s in speedups])
    geo_mean = math.exp(log_mean)

    # Latency-weighted mean: big shapes contribute proportionally
    total_ref = sum(c["ref_mean_ms"] for c in per_case if c["speedup_vs_ref"] > 0)
    if total_ref > 0:
        weighted = sum(
            c["ref_mean_ms"] / total_ref * c["speedup_vs_ref"]
            for c in per_case if c["speedup_vs_ref"] > 0
        )
    else:
        weighted = 0.0

    arith_mean = statistics.mean(speedups)

    ge_06x = sum(1 for s in speedups if s >= 0.6)
    ge_10x = sum(1 for s in speedups if s >= 1.0)
    total = len(speedups)

    return {
        "per_case": per_case,
        "geo_mean_speedup_vs_ref": round(geo_mean, 4),
        "weighted_mean_speedup_vs_ref": round(weighted, 4),
        "arith_mean_speedup_vs_ref": round(arith_mean, 4),
        "best_speedup_vs_ref": round(max(speedups), 4),
        "worst_speedup_vs_ref": round(min(speedups), 4),
        "ge_06x_count": ge_06x,
        "ge_10x_count": ge_10x,
        "total_cases": total,
        "ge_06x_ratio": round(ge_06x / total, 4) if total > 0 else 0,
        "ge_10x_ratio": round(ge_10x / total, 4) if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Baseline drift detection
# ---------------------------------------------------------------------------


def check_drift(
    current_ref_cases: List[Dict],
    saved_ref_cases: List[Dict],
    threshold: float = 0.10,
) -> Dict[str, Any]:
    """Compare current reference measurement against saved baseline.

    Returns drift status per case and overall.
    """
    drifts = []
    for cur, saved in zip(current_ref_cases, saved_ref_cases):
        saved_mean = saved["mean_ms"]
        current_mean = cur["mean_ms"]
        drift_ratio = abs(current_mean - saved_mean) / saved_mean if saved_mean > 0 else 0
        drifts.append({
            "index": cur.get("index", len(drifts)),
            "saved_ms": round(saved_mean, 4),
            "current_ms": round(current_mean, 4),
            "drift_pct": round(drift_ratio * 100, 2),
            "drifted": drift_ratio > threshold,
        })

    drifted_count = sum(1 for d in drifts if d["drifted"])
    overall_saved = statistics.mean([s["mean_ms"] for s in saved_ref_cases]) if saved_ref_cases else 0
    overall_current = statistics.mean([c["mean_ms"] for c in current_ref_cases]) if current_ref_cases else 0
    overall_drift = abs(overall_current - overall_saved) / overall_saved if overall_saved > 0 else 0

    return {
        "overall_drifted": overall_drift > threshold,
        "overall_drift_pct": round(overall_drift * 100, 2),
        "overall_saved_mean_ms": round(overall_saved, 4),
        "overall_current_mean_ms": round(overall_current, 4),
        "drifted_cases": drifted_count,
        "total_cases": len(drifts),
        "threshold_pct": round(threshold * 100, 1),
        "per_case": drifts,
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def _resolve_task_dir(op: str, workdir: Path) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()
    direct = workdir / op
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"Cannot find task directory for op '{op}' under {workdir}")


def _find_workdir(workdir_arg: Optional[str] = None) -> Path:
    if workdir_arg:
        p = Path(workdir_arg)
        if p.is_dir():
            return p
    env_val = __import__("os").environ.get("CV_AGENT_WORKDIR")
    if env_val:
        p = Path(env_val) / "agent_workdir"
        if p.is_dir():
            return p
        p = Path(env_val)
        if (p / "utils").is_dir():
            return p
    candidates = []
    fallback_env = os.environ.get("CV_AGENT_WORKDIR_FALLBACK")
    if fallback_env:
        candidates.append(Path(fallback_env))
    for c in candidates:
        if c.is_dir():
            return c
    return Path.cwd()


def run_benchmark(
    task_name: str,
    workdir: Path,
    impls: List[str],
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    device_id: int = 0,
    baseline_path: Optional[str] = None,
    save_baseline_path: Optional[str] = None,
    drift_check_path: Optional[str] = None,
    drift_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Run full benchmark and return structured results.

    Args:
        task_name: Task directory name under workdir.
        workdir: Agent workdir path.
        impls: List of implementations to benchmark.
        warmup: Warmup iterations per case.
        repeat: Timed iterations per case.
        seed: Random seed for input generation.
        device_id: NPU device ID.
        baseline_path: Path to saved baseline JSON (for speedup computation).
        save_baseline_path: If set, save baseline per-case data to this path.
        drift_check_path: If set, compare reference against this saved baseline.
        drift_threshold: Drift detection threshold (fraction, e.g. 0.10 = 10%).
    """
    import os
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(device_id))

    device = _get_device()
    use_event = _has_npu() and device.type == "npu"

    task_dir = _resolve_task_dir(task_name, workdir)

    # Load reference module for input groups
    ref_module_path = task_dir / "model.py"
    if not ref_module_path.is_file():
        raise FileNotFoundError(f"Missing reference model: {ref_module_path}")
    ref_module = _load_module(ref_module_path, f"{task_name}_ref_bench")
    input_groups = ref_module.get_input_groups()
    init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()

    # Load saved baseline if provided
    saved_baseline = None
    if baseline_path and Path(baseline_path).is_file():
        saved_baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))

    report = {
        "task": task_name,
        "device": str(device),
        "timing_method": "npu_event" if use_event else "perf_counter",
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "total_cases": len(input_groups),
        "results": {},
    }

    # Benchmark each implementation
    for impl in impls:
        config = IMPLEMENTATIONS.get(impl)
        if not config:
            report["results"][impl] = {"ok": False, "error": f"Unknown impl: {impl}"}
            continue

        module_path = task_dir / config["filename"]
        if not module_path.is_file():
            report["results"][impl] = {"ok": False, "error": f"File not found: {module_path}"}
            continue

        try:
            torch.manual_seed(seed)
            module = _load_module(module_path, f"{task_name}_{impl}_bench")
            model_cls = _find_model_class(module, config["preferred_class"])
            model = model_cls(*_clone_value(init_inputs)).to(device).eval()

            case_results = []
            for idx, inputs in enumerate(input_groups):
                model_inputs = _move_to_device(_clone_value(inputs), device)
                stats = _measure_case(model, model_inputs, device, warmup, repeat, use_event)
                stats["index"] = idx
                case_results.append(stats)

            # Overall: mean of case means (equal weight per case)
            case_means = [c["mean_ms"] for c in case_results]
            overall_mean = statistics.mean(case_means) if case_means else 0
            overall_median = statistics.median(case_means) if case_means else 0

            report["results"][impl] = {
                "ok": True,
                "label": config["label"],
                "case_results": case_results,
                "overall_mean_ms": round(overall_mean, 4),
                "overall_median_ms": round(overall_median, 4),
            }

        except Exception as exc:
            report["results"][impl] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Save baseline if requested
    if save_baseline_path:
        baseline_data = {}
        for impl in impls:
            r = report["results"].get(impl, {})
            if r.get("ok"):
                baseline_data[impl] = r["case_results"]
        baseline_data["total_cases"] = len(input_groups)
        baseline_data["timing_method"] = report["timing_method"]
        baseline_data["warmup"] = warmup
        baseline_data["repeat"] = repeat
        out_p = Path(save_baseline_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(baseline_data, indent=2, ensure_ascii=False), encoding="utf-8")
        report["baseline_saved"] = str(out_p)

    # Compute NKB metrics if we have reference data
    asc_result = report["results"].get("ascendc", {})
    if asc_result.get("ok"):
        ref_cases = None
        # Priority: saved baseline > current run reference
        if saved_baseline and "reference" in saved_baseline:
            ref_cases = saved_baseline["reference"]
        elif "reference" in report["results"] and report["results"]["reference"].get("ok"):
            ref_cases = report["results"]["reference"]["case_results"]

        if ref_cases:
            nkb = compute_nkb_metrics(ref_cases, asc_result["case_results"])
            report["nkb_metrics"] = nkb

        # Speedup vs baseline ascendc if available
        if saved_baseline and "ascendc" in saved_baseline:
            baseline_asc_cases = saved_baseline["ascendc"]
            baseline_asc_mean = statistics.mean([c["mean_ms"] for c in baseline_asc_cases])
            evolved_mean = asc_result["overall_mean_ms"]
            if evolved_mean > 0:
                report["speedup_vs_baseline"] = round(baseline_asc_mean / evolved_mean, 4)

    # Drift check if requested
    if drift_check_path and Path(drift_check_path).is_file():
        saved = json.loads(Path(drift_check_path).read_text(encoding="utf-8"))
        ref_result = report["results"].get("reference", {})
        if ref_result.get("ok") and "reference" in saved:
            drift = check_drift(ref_result["case_results"], saved["reference"], drift_threshold)
            report["drift_check"] = drift

    # Statistical significance if we have per-case clean times and baseline
    if asc_result.get("ok") and saved_baseline and "ascendc" in saved_baseline:
        sig_results = []
        for vc, bc in zip(asc_result["case_results"], saved_baseline["ascendc"]):
            # Use clean_times_ms (IQR-filtered) for t-test consistency with mean_ms
            vc_times = vc.get("clean_times_ms") or vc.get("raw_times_ms")
            bc_times = bc.get("clean_times_ms") or bc.get("raw_times_ms")
            if vc_times and bc_times:
                sig = check_significance(bc_times, vc_times)
                sig["index"] = vc.get("index", len(sig_results))
                sig_results.append(sig)
        if sig_results:
            n_sig = sum(1 for s in sig_results if s["significant"])
            report["statistical"] = {
                "per_case": sig_results,
                "significant_cases": n_sig,
                "total_cases": len(sig_results),
                "significant_ratio": round(n_sig / len(sig_results), 4) if sig_results else 0,
            }

    return report


# ---------------------------------------------------------------------------
# Text output (compatible with performance.py format for agent parsing)
# ---------------------------------------------------------------------------


def _print_report(report: Dict[str, Any]):
    """Print human-readable report, compatible with performance.py output format."""
    print("=" * 88)
    print("Performance Report (Event Timing)")
    print("=" * 88)
    print(f"Operator  : {report['task']}")
    print(f"Device    : {report['device']}")
    print(f"Timing    : {report['timing_method']}")
    print(f"Warmup    : {report['warmup']}")
    print(f"Repeat    : {report['repeat']}")
    print(f"Cases     : {report['total_cases']}")

    # Summary table (compatible with performance.py format for _parse_performance_output)
    print("-" * 88)
    print(f"{'Impl':<12} {'Status':<8} {'Mean(ms)':>12} {'Median':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    print("-" * 88)
    for impl, r in report["results"].items():
        if r.get("ok"):
            cases = r["case_results"]
            means = [c["mean_ms"] for c in cases]
            overall_mean = r["overall_mean_ms"]
            overall_median = r["overall_median_ms"]
            overall_min = min(c["min_ms"] for c in cases)
            overall_max = max(c["max_ms"] for c in cases)
            overall_std = statistics.stdev(means) if len(means) > 1 else 0
            print(
                f"{impl:<12} {'OK':<8} "
                f"{overall_mean:>12.3f} {overall_median:>12.3f} "
                f"{overall_min:>12.3f} {overall_max:>12.3f} "
                f"{overall_std:>12.3f}"
            )
        else:
            print(f"{impl:<12} {'ERROR':<8} {'-':>12} {'-':>12} {'-':>12} {'-':>12} {'-':>12}")

    # Per-case details (compatible format)
    for impl, r in report["results"].items():
        print("-" * 88)
        print(f"{impl} ->")
        if r.get("ok"):
            for c in r["case_results"]:
                samples = ", ".join(f"{t:.3f}" for t in c.get("raw_times_ms", []))
                print(
                    f"case[{c['index']}] mean={c['mean_ms']:.3f} ms, "
                    f"median={c['median_ms']:.3f} ms, "
                    f"cv={c['cv_pct']:.1f}%, outliers={c['n_outliers']}, "
                    f"samples(ms): [{samples}]"
                )
        else:
            print(f"  error: {r.get('error', 'unknown')}")

    # NKB metrics
    nkb = report.get("nkb_metrics")
    if nkb and "error" not in nkb:
        print("=" * 88)
        print("NPUKernelBench Metrics (vs Reference)")
        print("-" * 88)
        print(f"  Geometric Mean Speedup:  {nkb['geo_mean_speedup_vs_ref']:.4f}x")
        print(f"  Weighted Mean Speedup:   {nkb['weighted_mean_speedup_vs_ref']:.4f}x")
        print(f"  Arithmetic Mean Speedup: {nkb['arith_mean_speedup_vs_ref']:.4f}x")
        print(f"  Best / Worst:            {nkb['best_speedup_vs_ref']:.4f}x / {nkb['worst_speedup_vs_ref']:.4f}x")
        print(f"  >=0.6x: {nkb['ge_06x_count']}/{nkb['total_cases']} ({nkb['ge_06x_ratio']:.0%})")
        print(f"  >=1.0x: {nkb['ge_10x_count']}/{nkb['total_cases']} ({nkb['ge_10x_ratio']:.0%})")

    # Speedup vs baseline
    if "speedup_vs_baseline" in report:
        print(f"  Speedup vs Baseline AscendC: {report['speedup_vs_baseline']:.4f}x")

    # Drift check
    drift = report.get("drift_check")
    if drift:
        status = "DRIFTED" if drift["overall_drifted"] else "STABLE"
        print(f"\nBaseline Drift: {status} ({drift['overall_drift_pct']:.1f}%, "
              f"threshold={drift['threshold_pct']}%, "
              f"drifted_cases={drift['drifted_cases']}/{drift['total_cases']})")

    # Statistical significance
    stat = report.get("statistical")
    if stat:
        print(f"\nStatistical Significance (vs baseline, alpha=0.05): "
              f"{stat['significant_cases']}/{stat['total_cases']} cases significant")

    print("=" * 88)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Lingxi operator benchmark with device-side event timing",
    )
    parser.add_argument("task", help="Task directory name (under workdir)")
    parser.add_argument("--workdir", type=str, default=None, help="Path to agent_workdir/")
    parser.add_argument("--impl", default="all",
                        help="Implementation(s): reference|tilelang|ascendc|all (default: all)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"Warmup iterations (default: {DEFAULT_WARMUP})")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help=f"Timed iterations (default: {DEFAULT_REPEAT})")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--save-baseline", type=str, default=None,
                        help="Save per-case baseline data to this JSON path")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Load saved baseline for speedup computation")
    parser.add_argument("--drift-check", type=str, default=None,
                        help="Compare reference against saved baseline for drift detection")
    parser.add_argument("--drift-threshold", type=float, default=0.10,
                        help="Drift threshold as fraction (default: 0.10 = 10%%)")
    parser.add_argument("--quiet", action="store_true", help="Suppress text output, only write JSON")

    args = parser.parse_args()

    workdir = _find_workdir(args.workdir)
    if not workdir.is_dir():
        print(json.dumps({"error": f"workdir not found: {workdir}"}), file=sys.stderr)
        sys.exit(1)

    if args.impl == "all":
        impls = ["reference", "ascendc"]
    elif "," in args.impl:
        impls = [s.strip() for s in args.impl.split(",")]
    else:
        impls = [args.impl]

    report = run_benchmark(
        task_name=args.task,
        workdir=workdir,
        impls=impls,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        device_id=args.device,
        baseline_path=args.baseline,
        save_baseline_path=args.save_baseline,
        drift_check_path=args.drift_check,
        drift_threshold=args.drift_threshold,
    )

    if not args.quiet:
        _print_report(report)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.quiet:
            print(f"\nJSON results written to {out_p}")

    # Exit code: 0 if any impl succeeded
    ok = any(r.get("ok") for r in report["results"].values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
