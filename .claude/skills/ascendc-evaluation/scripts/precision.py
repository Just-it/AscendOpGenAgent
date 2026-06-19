"""Precision comparison utilities — v2 schema.

Provides three-way symmetric comparison (ans, ref, golden) with
12-field comparison groups, ULP metrics, and element statistics.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from constants import DEFAULT_TOLERANCES
from ulp import ulp_metrics


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ElementStats:
    nan: int
    inf: int
    subnormal: int
    small: int
    total: int

    def to_dict(self) -> dict:
        return {
            "nan": self.nan,
            "inf": self.inf,
            "subnormal": self.subnormal,
            "small": self.small,
            "total": self.total,
        }


@dataclass
class ComparisonGroup:
    """12-field symmetric comparison metrics."""
    ae_max: float
    ae_mean: float
    re_max: float
    re_mean: float
    rmse: float
    mean: float
    std: float
    svec: int
    ulp_mean: float
    ulp_max: int
    ulp_miss_rate: float
    mismatch_rate: float

    def to_dict(self) -> dict:
        return {
            "ae_max": self.ae_max,
            "ae_mean": self.ae_mean,
            "re_max": self.re_max,
            "re_mean": self.re_mean,
            "rmse": self.rmse,
            "mean": self.mean,
            "std": self.std,
            "svec": self.svec,
            "ulp_mean": self.ulp_mean,
            "ulp_max": self.ulp_max,
            "ulp_miss_rate": self.ulp_miss_rate,
            "mismatch_rate": self.mismatch_rate,
        }


@dataclass
class Ratios:
    max_re: float
    mean_re: float
    rmse: float
    svec: float

    def to_dict(self) -> dict:
        return {
            "max_re": self.max_re,
            "mean_re": self.mean_re,
            "rmse": self.rmse,
            "svec": self.svec,
        }


@dataclass
class ComponentResult:
    passed: bool
    ans_counts: ElementStats
    ref_counts: ElementStats
    golden_counts: ElementStats
    ans_vs_golden: ComparisonGroup
    ref_vs_golden: ComparisonGroup
    ans_vs_ref: ComparisonGroup
    ratios: Ratios
    diagnosis: dict

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "ans": self.ans_counts.to_dict(),
            "ref": self.ref_counts.to_dict(),
            "golden": self.golden_counts.to_dict(),
            "ans_vs_golden": self.ans_vs_golden.to_dict(),
            "ref_vs_golden": self.ref_vs_golden.to_dict(),
            "ans_vs_ref": self.ans_vs_ref.to_dict(),
            "ratios": self.ratios.to_dict(),
            "diagnosis": self.diagnosis,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_element_stats(t: torch.Tensor, sv_th: float = 0.0) -> ElementStats:
    """Count element statistics: NaN, Inf, subnormal, small-value, total."""
    abs_t = torch.abs(t)
    is_finite = torch.isfinite(t)

    # Subnormal: 0 < |x| < finfo.tiny (only for float types)
    if t.is_floating_point():
        tiny = torch.finfo(t.dtype).tiny
        subnormal = int(((abs_t > 0) & (abs_t < tiny) & is_finite).sum().item())
    else:
        subnormal = 0

    # Small-value region: |x| < sv_th
    if sv_th > 0:
        small = int((abs_t < sv_th).sum().item())
    else:
        small = 0

    return ElementStats(
        nan=int(torch.isnan(t).sum().item()),
        inf=int(torch.isinf(t).sum().item()),
        subnormal=subnormal,
        small=small,
        total=t.numel(),
    )


def _compute_comparison(a: torch.Tensor, b: torch.Tensor,
                        golden: torch.Tensor,
                        sv_th: float, sv_err: float,
                        atol: float, rtol: float, ulp_tol: int,
                        dtype: torch.dtype,
                        ulp_method: str = "bitwise",
                        include_subnormal: bool = True) -> ComparisonGroup:
    """Compute 12-field comparison group between tensors a and b.

    golden is used only to determine the large/small value regions.
    """
    a64 = a.double()
    b64 = b.double()
    g64 = golden.double()

    # Signed and absolute error — all elements
    diff = a64 - b64
    ae = torch.abs(diff)
    ae_max = torch.max(ae).item() if ae.numel() > 0 else 0.0
    ae_mean = torch.mean(ae).item() if ae.numel() > 0 else 0.0
    mean_val = torch.mean(diff).item() if diff.numel() > 0 else 0.0
    std_val = torch.std(diff).item() if diff.numel() > 1 else 0.0

    # Large-value region: |golden| >= sv_th
    abs_golden = torch.abs(g64)
    large_mask = abs_golden >= sv_th

    n_large = large_mask.sum().item()
    if n_large > 0:
        re = ae[large_mask] / abs_golden[large_mask]
        re_max = torch.max(re).item()
        re_mean = torch.mean(re).item()
        rmse = torch.sqrt(torch.mean(ae[large_mask] ** 2)).item()
    else:
        re_max = 0.0
        re_mean = 0.0
        rmse = 0.0

    # Small-value region: |golden| < sv_th, error > sv_err
    small_mask = ~large_mask
    if small_mask.sum().item() > 0:
        svec = int((ae[small_mask] > sv_err).sum().item())
    else:
        svec = 0

    # ULP metrics
    ulp = ulp_metrics(a, b, dtype, ulp_tol=ulp_tol, method=ulp_method,
                      include_subnormal=include_subnormal)

    # Mismatch rate
    close = torch.isclose(a.float(), b.float(), atol=atol, rtol=rtol)
    mismatch_rate = 1.0 - close.sum().item() / max(close.numel(), 1)

    return ComparisonGroup(
        ae_max=ae_max,
        ae_mean=ae_mean,
        re_max=re_max,
        re_mean=re_mean,
        rmse=rmse,
        mean=mean_val,
        std=std_val,
        svec=svec,
        ulp_mean=ulp["ulp_mean"],
        ulp_max=ulp["ulp_max"],
        ulp_miss_rate=ulp["ulp_miss_rate"],
        mismatch_rate=mismatch_rate,
    )


# ---------------------------------------------------------------------------
# Diagnosis engine
# ---------------------------------------------------------------------------

DIAGNOSIS_PATTERNS = [
    {
        "id": "nan_inf_introduced",
        "detect": lambda r: (
            (r.ans_counts.nan > 0 or r.ans_counts.inf > 0)
            and r.golden_counts.nan == 0 and r.golden_counts.inf == 0
            and r.ref_counts.nan == 0 and r.ref_counts.inf == 0
        ),
        "likelihood": "high",
        "category": "numerical_stability",
        "description": "Custom kernel introduces NaN/Inf that don't exist in golden or ref",
        "suggestions": [
            "Check for division by zero in kernel code",
            "Check log/sqrt inputs — ensure they receive positive values",
            "Check exp inputs — large values cause overflow to Inf",
            "Add numerical guards: clamp inputs before unstable operations",
        ],
    },
    {
        "id": "zero_output",
        "detect": lambda r: (
            r.ans_vs_golden.mismatch_rate > 0.9
            and r.ans_vs_golden.ae_mean < 1e-6
        ),
        "likelihood": "high",
        "category": "incomplete_implementation",
        "description": "Output is near-zero everywhere — likely placeholder or uninitialized output",
        "suggestions": [
            "Verify Process() implements the full algorithm, not just Duplicate(output, 0.0f, size)",
            "Check that all input tensors are loaded and used in computation",
            "Verify DataCopy moves data from GM to local buffer before computing",
        ],
    },
    {
        "id": "localized_outlier",
        "detect": lambda r: (
            not r.passed
            and r.ratios.max_re > r.ratios.mean_re * 5
            and r.ratios.mean_re <= 2.0
        ),
        "likelihood": "high",
        "category": "boundary_handling",
        "description": "max_re ratio is high but mean_re is acceptable — few extreme outliers",
        "suggestions": [
            "Check kernel handling of boundary/edge-case inputs (near-zero, very large, subnormal)",
            "Review tiling boundary: last tile may have fewer elements than expected",
            "Check DataCopyPad usage for non-32B-aligned data sizes",
        ],
    },
    {
        "id": "systematic_drift",
        "detect": lambda r: (
            not r.passed and r.ratios.mean_re > 2.0
        ),
        "likelihood": "high",
        "category": "algorithm_error",
        "description": "Systematic precision drift — mean relative error ratio exceeded",
        "suggestions": [
            "Verify the algorithm matches the DSL/reference exactly",
            "Check for dtype cast precision loss (e.g., fp32 intermediate cast to fp16 too early)",
            "Check cross-tile accumulation — results may only reflect the last tile",
            "Verify all input tensors are used — missing an input causes wrong computation",
        ],
    },
    {
        "id": "small_value_error",
        "detect": lambda r: (
            not r.passed and r.ratios.svec > 2.0
        ),
        "likelihood": "medium",
        "category": "small_value_handling",
        "description": "Small-value region errors exceed tolerance",
        "suggestions": [
            "Check computation near zero — small values may underflow or lose precision",
            "Verify subnormal handling in the kernel",
            "Consider using higher-precision intermediate computation for small inputs",
        ],
    },
    {
        "id": "uneven_distribution",
        "detect": lambda r: (
            not r.passed
            and r.ratios.rmse > 2.0
            and r.ratios.max_re <= 10.0
            and r.ratios.mean_re <= 2.0
        ),
        "likelihood": "medium",
        "category": "error_distribution",
        "description": "RMSE ratio exceeded while max/mean RE are acceptable — uneven error distribution",
        "suggestions": [
            "Check for input-dependent precision patterns (certain value ranges trigger larger errors)",
            "Review vectorized computation — some lanes may compute differently",
            "Profile with different input distributions to identify problematic ranges",
        ],
    },
]


def _diagnose(result):
    """Match diagnostic patterns against a ComponentResult."""
    failed_checks = []
    if result.ratios.max_re > 10.0:
        failed_checks.append("max_re")
    if result.ratios.mean_re > 2.0:
        failed_checks.append("mean_re")
    if result.ratios.rmse > 2.0:
        failed_checks.append("rmse")
    if result.ratios.svec > 2.0:
        failed_checks.append("svec")
    golden_ref_clean = (
        result.golden_counts.nan == 0 and result.golden_counts.inf == 0
        and result.ref_counts.nan == 0 and result.ref_counts.inf == 0
    )
    if golden_ref_clean and (result.ans_counts.nan > 0 or result.ans_counts.inf > 0):
        failed_checks.append("nan_inf")

    if result.passed:
        return {"verdict": "pass", "failed_checks": [], "pattern": None, "root_causes": []}

    matched = []
    for pat in DIAGNOSIS_PATTERNS:
        try:
            if pat["detect"](result):
                matched.append({
                    "likelihood": pat["likelihood"],
                    "category": pat["category"],
                    "description": pat["description"],
                    "suggestions": pat["suggestions"],
                })
        except Exception:
            continue

    pattern_id = matched[0]["category"] if matched else "unknown"
    return {
        "verdict": "fail",
        "failed_checks": failed_checks,
        "pattern": pattern_id,
        "root_causes": matched if matched else [{
            "likelihood": "low",
            "category": "unknown",
            "description": "No known pattern matched — manual investigation needed",
            "suggestions": [
                "Compare ans_vs_golden and ref_vs_golden metrics side by side",
                "Check the precision JSON for detailed per-element statistics",
                "Generate a dashboard with ascend_test_visualize for visual analysis",
            ],
        }],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dual_inspect(y_ans: torch.Tensor, y_ref: torch.Tensor,
                 y_golden: torch.Tensor, name: str, *,
                 tolerances: Optional[dict] = None,
                 dtype: Optional[torch.dtype] = None) -> ComponentResult:
    """Three-way precision comparison producing v2 ComponentResult.

    Args:
        y_ans: Answer tensor (custom NPU kernel output).
        y_ref: Reference tensor (NPU eager baseline).
        y_golden: Golden tensor (CPU fp64 baseline).
        name: Component name (e.g. "Output", "Grad-X").
        tolerances: Override tolerance dict. If None, uses DEFAULT_TOLERANCES
                    for the given dtype.
        dtype: Data type for ULP calculation and default tolerances.
               If None, inferred from y_ans.dtype.
    """
    if dtype is None:
        dtype = y_ans.dtype

    # Resolve tolerances
    if tolerances is None:
        tol = DEFAULT_TOLERANCES.get(dtype, DEFAULT_TOLERANCES[torch.bfloat16])
    else:
        tol = tolerances

    atol = tol["atol"]
    rtol = tol["rtol"]
    ulp_tol = tol.get("ulp_tol", 2)
    sv_th = tol["sv_th"]
    sv_err = tol["sv_err"]
    max_re_ratio_limit = tol.get("max_re_ratio_limit", 10.0)
    mean_re_ratio_limit = tol.get("mean_re_ratio_limit", 2.0)
    rmse_ratio_limit = tol.get("rmse_ratio_limit", 2.0)
    ulp_method = tol.get("ulp_method", "bitwise")
    include_subnormal = tol.get("include_subnormal", True)

    # 1. Element statistics
    ans_counts = _count_element_stats(y_ans, sv_th)
    ref_counts = _count_element_stats(y_ref, sv_th)
    golden_counts = _count_element_stats(y_golden, sv_th)

    # 2. Three comparison groups
    ans_vs_golden = _compute_comparison(
        y_ans, y_golden, y_golden, sv_th, sv_err, atol, rtol, ulp_tol, dtype,
        ulp_method=ulp_method, include_subnormal=include_subnormal)
    ref_vs_golden = _compute_comparison(
        y_ref, y_golden, y_golden, sv_th, sv_err, atol, rtol, ulp_tol, dtype,
        ulp_method=ulp_method, include_subnormal=include_subnormal)
    ans_vs_ref = _compute_comparison(
        y_ans, y_ref, y_golden, sv_th, sv_err, atol, rtol, ulp_tol, dtype,
        ulp_method=ulp_method, include_subnormal=include_subnormal)

    # 3. Ratios: ans / max(ref, floor)
    ref_floor = 1e-7
    ratios = Ratios(
        max_re=ans_vs_golden.re_max / max(ref_vs_golden.re_max, ref_floor),
        mean_re=ans_vs_golden.re_mean / max(ref_vs_golden.re_mean, ref_floor),
        rmse=ans_vs_golden.rmse / max(ref_vs_golden.rmse, ref_floor),
        svec=ans_vs_golden.svec / max(ref_vs_golden.svec, 1),
    )

    # 4. Pass judgment per v2 design doc
    # NaN/Inf: fail only when golden+ref are clean but ans is not
    golden_ref_clean = (
        golden_counts.nan == 0 and golden_counts.inf == 0
        and ref_counts.nan == 0 and ref_counts.inf == 0
    )
    ans_has_anomaly = ans_counts.nan > 0 or ans_counts.inf > 0
    nan_inf_fail = golden_ref_clean and ans_has_anomaly

    passed = (
        not nan_inf_fail
        and ratios.max_re <= max_re_ratio_limit
        and ratios.mean_re <= mean_re_ratio_limit
        and ratios.rmse <= rmse_ratio_limit
        and ratios.svec <= 2.0
    )

    # 5. Diagnosis
    # Build a temporary result for diagnosis (without diagnosis field)
    temp_result = ComponentResult(
        passed=passed, ans_counts=ans_counts, ref_counts=ref_counts,
        golden_counts=golden_counts, ans_vs_golden=ans_vs_golden,
        ref_vs_golden=ref_vs_golden, ans_vs_ref=ans_vs_ref,
        ratios=ratios, diagnosis={},
    )
    diagnosis = _diagnose(temp_result)

    return ComponentResult(
        passed=passed, ans_counts=ans_counts, ref_counts=ref_counts,
        golden_counts=golden_counts, ans_vs_golden=ans_vs_golden,
        ref_vs_golden=ref_vs_golden, ans_vs_ref=ans_vs_ref,
        ratios=ratios, diagnosis=diagnosis,
    )
