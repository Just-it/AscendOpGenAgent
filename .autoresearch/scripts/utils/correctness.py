"""Output comparison: torch.allclose-style standard, dict-returning API.

Single source for the per-pair allclose math:
    skills/triton/kernel-verifier/scripts/verify.py
        — get_allclose_tolerance(), compare(), _check_accuracy_allclose()

This module wraps skill verify.compare() (which raises AssertionError on
failure) with a dict-returning multi-case API that batch/verify.py
consumes (it needs per-case status + max_abs_diff to render
verify_results.json — info skill's raise-on-failure surface doesn't
provide).

Algorithm (lives entirely in skill verify.py; reproduced here only as a
narrative for readers — any implementation drift is a bug):
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

import importlib.util
import os
import sys


# ---------------------------------------------------------------------------
# Skill loader: import skill verify.py once at module load. Loading via
# importlib.util keeps the generic basename `verify` from colliding with
# `.autoresearch/scripts/batch/verify.py` if both end up on sys.path.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
# .../utils -> .../scripts -> .../.autoresearch -> repo root
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SKILL_VERIFY_PATH = os.path.join(
    _REPO_ROOT, "skills", "triton", "kernel-verifier", "scripts", "verify.py",
)


def _load_skill_verify():
    if not os.path.isfile(_SKILL_VERIFY_PATH):
        raise RuntimeError(
            f"kernel-verifier skill not found at {_SKILL_VERIFY_PATH!r} — "
            f"utils.correctness wraps its compare() helper. Layout assumption: "
            f"skills/ is a sibling of .autoresearch/."
        )
    spec = importlib.util.spec_from_file_location(
        "_skill_verify_for_correctness", _SKILL_VERIFY_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_skill = _load_skill_verify()

# Re-export the per-dtype tolerance lookup so callers that previously did
# `from utils.correctness import get_allclose_tolerance` still resolve.
get_allclose_tolerance = _skill.get_allclose_tolerance


# ---------------------------------------------------------------------------
# Per-tensor comparison: thin wrapper over skill verify.compare()
# ---------------------------------------------------------------------------

def _check_one_tensor(ref, new, idx: int, diagnostics: list) -> tuple:
    """torch.allclose-style check on a single (ref, new) tensor pair.

    Returns (passed, max_abs_diff_or_None). Appends one diagnostic line.

    Implementation: delegates the pass/fail decision to skill verify.compare()
    so the allclose math + NaN / Inf / bool / int rules live in exactly one
    place. We only add the bits skill compare() doesn't surface:
      - explicit dtype-mismatch shortcut (skill compare auto-coerces; utils
        callers expect a hard fail with a clean diagnostic)
      - max_abs_diff for both pass and fail (batch/verify.py reports it)
      - dict-friendly diagnostic line (skill raises a multi-line metrics
        block on failure; we trim to the first line for the per-pair feed).
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

    # Pre-compute max_abs on the finite floating portion. We do this BEFORE
    # delegating because skill compare() doesn't return max_abs on success
    # (and on failure raises a string; parsing that back is fragile). For
    # bool / int / all-non-finite cases we follow the legacy convention
    # below: 0.0 for exact-comparison dtypes, None when there's nothing
    # finite to measure.
    finite = torch.isfinite(rf) & torch.isfinite(nf)
    rf_fin = rf[finite]
    nf_fin = nf[finite]
    if int(finite.sum()) == 0:
        max_abs = None
    elif rf_fin.dtype == torch.bool or not rf_fin.is_floating_point():
        max_abs = 0.0
    else:
        max_abs = float((nf_fin.float() - rf_fin.float()).abs().max().item())

    try:
        _skill.compare(ref, new, ref.dtype)
    except AssertionError as e:
        first_line = str(e).split("\n", 1)[0]
        diagnostics.append(f"out{idx}: {first_line}")
        return False, max_abs

    if max_abs is None:
        diagnostics.append(f"out{idx}: OK (all non-finite, skipped)")
    elif rf_fin.dtype == torch.bool:
        diagnostics.append(f"out{idx}: OK (bool exact)")
    elif not rf_fin.is_floating_point():
        diagnostics.append(f"out{idx}: OK (int exact dtype={rf_fin.dtype})")
    else:
        diagnostics.append(
            f"out{idx}: OK (max_abs_err={max_abs:.3e} dtype={rf_fin.dtype})"
        )
    return True, max_abs


# ---------------------------------------------------------------------------
# Single-case wrapper
# ---------------------------------------------------------------------------

def compare_outputs(out_ref: list, out_new: list) -> dict:
    """allclose-style comparison for a single shape case.

    Returns:
      {"correctness": bool,
       "diagnostics": list[str],     # one per output entry
       "max_abs_diff": float | None} # max over all floating tensor pairs
    """
    import torch

    diagnostics: list = []

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
    max_abs_overall = None

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


# ---------------------------------------------------------------------------
# Multi-case wrapper (the API batch/verify.py imports)
# ---------------------------------------------------------------------------

def compare_outputs_per_case(out_ref_per_case: list,
                             out_new_per_case: list) -> dict:
    """Multi-shape allclose-style check: hard-gate on every case.

    Inputs are List[List[Tensor]] - one outer entry per shape case, each
    inner list is the model's outputs for that case (already moved to CPU
    by the caller).

    Returns:
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

    Single-shape callers collapse to `List[List[Tensor]]` of length 1.
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
    flat_diag: list = []
    all_pass = True
    max_abs_overall = None

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
    worst_idx = None
    worst_max = None
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
