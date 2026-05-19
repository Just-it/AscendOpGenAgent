"""Output comparison: torch.allclose-style standard.

Canonical implementation lives at
`skills/triton/kernel-verifier/scripts/verify.py` on `main`; this
module ports its `_check_accuracy_allclose` verbatim. Used by:
  - the static eval_kernel.py verify phase (one subprocess per round)
  - the batch-time pre-flight verify.py Tier 2 check

Algorithm:
  1. NaN positions in ref vs kernel must match exactly.
  2. Inf positions and signs must match exactly.
  3. bool tensors must compare exactly.
  4. For finite floating values:
       tol             = get_allclose_tolerance(ref.dtype)   # {rtol, atol}
       allowed_error   = tol.atol + tol.rtol * |ref|
       PASS  iff  all(|ref - new| <= allowed_error)
  5. Integer dtypes must compare exactly.

Per-dtype tolerances (rtol = 2^n, atol = absolute floor):
  fp16          rtol 2^-10 ~ 9.77e-4   atol 1e-3
  bfloat16      rtol 2^-7  ~ 7.81e-3   atol 1e-2
  fp32          rtol 2^-13 ~ 1.22e-4   atol 1e-5
  (unknown)     fallback to fp32 tolerance
"""
from __future__ import annotations


_DEFAULT_TOL = {"rtol": 2 ** -13, "atol": 1e-5}


def get_allclose_tolerance(data_type) -> dict:
    """Per-dtype (rtol, atol) for torch.allclose-style comparison.

    Mirrors `skills/triton/kernel-verifier/scripts/verify.py` on `main`.
    Accepts either a torch.dtype or a lowercase dtype string. Unknown
    dtypes fall back to fp32 tolerance.
    """
    import torch

    if isinstance(data_type, str):
        key = data_type.lower().replace("torch.", "")
        return {
            "float32":  {"rtol": 2 ** -13, "atol": 1e-5},
            "float":    {"rtol": 2 ** -13, "atol": 1e-5},
            "float16":  {"rtol": 2 ** -10, "atol": 1e-3},
            "half":     {"rtol": 2 ** -10, "atol": 1e-3},
            "bfloat16": {"rtol": 2 ** -7,  "atol": 1e-2},
        }.get(key, _DEFAULT_TOL)

    return {
        torch.float32:  {"rtol": 2 ** -13, "atol": 1e-5},
        torch.float16:  {"rtol": 2 ** -10, "atol": 1e-3},
        torch.bfloat16: {"rtol": 2 ** -7,  "atol": 1e-2},
    }.get(data_type, _DEFAULT_TOL)


def _check_one_tensor(ref, new, idx: int, diagnostics: list) -> tuple[bool, float | None]:
    """torch.allclose-style check on a single (ref, new) tensor pair.

    Returns (passed, max_abs_diff_or_None). Appends one diagnostic line.
    Mirrors `compare()` + `_check_accuracy_allclose()` from `main`'s
    verify.py - same NaN/Inf handling, same bool short-circuit, same
    `|diff| <= atol + rtol * |ref|` element-wise gate.
    """
    import torch

    if ref.dtype != new.dtype:
        diagnostics.append(
            f"out{idx}: dtype mismatch ref={ref.dtype} new={new.dtype}"
        )
        return False, None

    rf = ref.detach().cpu().flatten()
    nf = new.detach().cpu().flatten()

    if rf.shape != nf.shape:
        diagnostics.append(
            f"out{idx}: shape mismatch ref={tuple(rf.shape)} new={tuple(nf.shape)}"
        )
        return False, None

    rf_nan = torch.isnan(rf)
    nf_nan = torch.isnan(nf)
    if not torch.equal(rf_nan, nf_nan):
        diagnostics.append(
            f"out{idx}: NaN position mismatch ref={int(rf_nan.sum())}"
            f" new={int(nf_nan.sum())} of {rf.numel()}"
        )
        return False, None

    rf_inf = torch.isinf(rf)
    nf_inf = torch.isinf(nf)
    if not torch.equal(rf_inf, nf_inf):
        diagnostics.append(
            f"out{idx}: Inf position mismatch ref={int(rf_inf.sum())}"
            f" new={int(nf_inf.sum())} of {rf.numel()}"
        )
        return False, None
    if rf_inf.any() and not torch.equal(
        torch.sign(rf[rf_inf]), torch.sign(nf[nf_inf])
    ):
        diagnostics.append(f"out{idx}: Inf sign mismatch")
        return False, None

    finite = torch.isfinite(rf) & torch.isfinite(nf)
    if int(finite.sum()) == 0:
        diagnostics.append(f"out{idx}: OK (all non-finite, skipped)")
        return True, None

    rf_fin = rf[finite]
    nf_fin = nf[finite]

    if rf_fin.dtype == torch.bool:
        if not torch.equal(rf_fin, nf_fin):
            n_bad = int((rf_fin != nf_fin).sum())
            diagnostics.append(
                f"out{idx}: bool mismatch {n_bad}/{int(rf_fin.numel())}"
            )
            return False, None
        diagnostics.append(f"out{idx}: OK (bool exact)")
        return True, 0.0

    if not rf_fin.is_floating_point():
        if not torch.equal(rf_fin, nf_fin):
            n_bad = int((rf_fin != nf_fin).sum())
            diagnostics.append(
                f"out{idx}: int mismatch {n_bad}/{int(rf_fin.numel())} dtype={rf_fin.dtype}"
            )
            return False, None
        diagnostics.append(f"out{idx}: OK (int exact dtype={rf_fin.dtype})")
        return True, 0.0

    tol = get_allclose_tolerance(rf_fin.dtype)
    rtol = tol["rtol"]
    atol = tol["atol"]
    rf_f = rf_fin.float()
    nf_f = nf_fin.float()
    diff = (nf_f - rf_f).abs()
    max_abs = float(diff.max().item())
    allowed = atol + rtol * rf_f.abs()
    max_allowed = float(allowed.max().item())
    close = diff <= allowed
    passed = bool(close.all().item())

    if passed:
        diagnostics.append(
            f"out{idx}: OK (max_abs_err={max_abs:.3e} "
            f"max_allowed={max_allowed:.3e} rtol={rtol:.3e} "
            f"atol={atol:.3e} dtype={rf_fin.dtype})"
        )
        return True, max_abs

    n_bad = int((~close).sum())
    diagnostics.append(
        f"out{idx}: max_abs_err={max_abs:.3e} "
        f"max_allowed={max_allowed:.3e} rtol={rtol:.3e} "
        f"atol={atol:.3e} dtype={rf_fin.dtype} "
        f"bad_elems={n_bad}/{int(rf_fin.numel())}"
    )
    return False, max_abs


def compare_outputs(out_ref: list, out_new: list) -> dict:
    """allclose-style comparison for a single shape case.

    Tolerances are derived from the reference dtype via
    `get_allclose_tolerance()`.

    Behavior:
      - Output count mismatch: hard fail.
      - Empty output lists on both sides: hard fail (wrapper bug).
      - Non-tensor outputs require same type and `==`.
      - For each tensor pair, NaN/Inf position equality + bool/int exact +
        floating `|diff| <= atol + rtol*|ref|` element-wise gate per
        `_check_one_tensor`.

    Returns:
      {"correctness": bool,
       "diagnostics": list[str],     # one per output entry
       "max_abs_diff": float | None} # max over all floating tensor pairs
    """
    import torch

    diagnostics: list[str] = []

    if len(out_ref) != len(out_new):
        return {
            "correctness": False,
            "diagnostics": [
                f"output count: ref={len(out_ref)} new={len(out_new)}"
            ],
            "max_abs_diff": None,
        }

    if len(out_ref) == 0:
        return {
            "correctness": False,
            "diagnostics": ["both ref and kernel returned 0 outputs (wrapper failure?)"],
            "max_abs_diff": None,
        }

    all_pass = True
    max_abs_overall: float | None = None

    for i, (r, n) in enumerate(zip(out_ref, out_new)):
        if not (isinstance(r, torch.Tensor) and isinstance(n, torch.Tensor)):
            if type(r) is not type(n):
                all_pass = False
                diagnostics.append(
                    f"out{i}: type mismatch ref={type(r).__name__} new={type(n).__name__}"
                )
                continue
            try:
                eq = bool(r == n)
            except Exception:
                eq = (r is n)
            if not eq:
                all_pass = False
                diagnostics.append(
                    f"out{i}: non-tensor mismatch ref={r!r} new={n!r}"
                )
            else:
                diagnostics.append(f"out{i}: OK (non-tensor exact)")
            continue

        ok, m = _check_one_tensor(r, n, i, diagnostics)
        if not ok:
            all_pass = False
        if m is not None and (max_abs_overall is None or m > max_abs_overall):
            max_abs_overall = m

    return {
        "correctness": all_pass,
        "diagnostics": diagnostics,
        "max_abs_diff": max_abs_overall,
    }


def compare_outputs_per_case(out_ref_per_case: list,
                             out_new_per_case: list) -> dict:
    """Multi-shape allclose-style check: hard-gate on every case.

    Inputs are List[List[Tensor]] - one outer entry per shape case, each
    inner list is the model's outputs for that case (already moved to CPU
    by the caller). Returns:

      {"correctness": bool,                # AND of every case
       "per_case": [
           {"idx": int, "correctness": bool, "diagnostics": [...],
            "max_abs_diff": float | None}, ...
       ],
       "max_abs_diff": float | None,       # max over all cases
       "diagnostics": [str, ...],          # flat aggregate, prefixed [case i]
       "failed_indices": list[int],
       "worst_idx": int | None,
       "worst_max_abs_diff": float | None}

    Both single- and multi-shape callers go through this wrapper - the
    static `eval_kernel.py` (verify phase) and the batch Tier-2 verify
    (`batch/verify.py`) call it unconditionally, with single-shape
    inputs collapsed to `List[List[Tensor]]` of length 1.
    """
    if len(out_ref_per_case) != len(out_new_per_case):
        return {
            "correctness": False,
            "per_case": [],
            "diagnostics": [
                f"case count: ref={len(out_ref_per_case)} "
                f"new={len(out_new_per_case)}"
            ],
            "max_abs_diff": None,
            "failed_indices": [],
            "worst_idx": None,
            "worst_max_abs_diff": None,
        }

    per_case = []
    flat_diag: list[str] = []
    all_pass = True
    max_abs_overall: float | None = None

    for i, (out_ref, out_new) in enumerate(zip(out_ref_per_case,
                                               out_new_per_case)):
        sub = compare_outputs(list(out_ref), list(out_new))
        per_case.append({
            "idx": i,
            "correctness": sub["correctness"],
            "diagnostics": sub["diagnostics"],
            "max_abs_diff": sub["max_abs_diff"],
        })
        if not sub["correctness"]:
            all_pass = False
        for d in sub["diagnostics"]:
            flat_diag.append(f"[case {i}] {d}")
        m = sub["max_abs_diff"]
        if m is not None and (max_abs_overall is None or m > max_abs_overall):
            max_abs_overall = m

    failed_indices = [pc["idx"] for pc in per_case if not pc["correctness"]]
    worst_idx: int | None = None
    worst_max: float | None = None
    if not all_pass:
        candidates = [pc for pc in per_case
                      if not pc["correctness"]
                      and isinstance(pc.get("max_abs_diff"), (int, float))]
        if candidates:
            best = max(candidates, key=lambda x: x["max_abs_diff"])
            worst_idx = best["idx"]
            worst_max = best["max_abs_diff"]
        flat_diag.append(
            f"[verify] CORRECTNESS_SUMMARY: failed={len(failed_indices)}/"
            f"{len(per_case)} failed_idx={failed_indices} "
            f"worst_case={worst_idx} max_abs="
            f"{(f'{worst_max:.3e}' if worst_max is not None else 'None')}"
        )

    return {
        "correctness": all_pass,
        "per_case": per_case,
        "diagnostics": flat_diag,
        "max_abs_diff": max_abs_overall,
        "failed_indices": failed_indices,
        "worst_idx": worst_idx,
        "worst_max_abs_diff": worst_max,
    }
