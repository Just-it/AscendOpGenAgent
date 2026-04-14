#!/usr/bin/env python3
"""
AscendC Static Checker — validates generated AscendC code for common anti-patterns.

Runs 10 checks on .h and .cpp files and outputs a JSON report to stdout.

Usage:
    python3 src/scripts/ascendc_static_check.py <directory>

Example:
    python3 src/scripts/ascendc_static_check.py workspace/pooling_skills_test/generated/
    python3 src/scripts/ascendc_static_check.py output/src/pooling/

Exit codes:
    0 — all checks passed
    1 — one or more violations found
    2 — usage error (bad args, directory not found)

Requires Python 3.8+, no external dependencies.
"""

import json
import os
import re
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Violation = Dict[str, object]  # {"file": str, "line": int, "detail": str}
CheckResult = Dict[str, object]  # {"passed": bool, "violations": [...]}


# ---------------------------------------------------------------------------
# File reading helper
# ---------------------------------------------------------------------------
def read_lines(filepath: str) -> List[str]:
    """Read file lines, tolerating encoding errors."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


# ---------------------------------------------------------------------------
# Check 1: missing_namespace
#   Any .h file with __simt_vf__ but no `namespace ascendc_ops`
# ---------------------------------------------------------------------------
def check_missing_namespace(filepath: str, lines: List[str]) -> List[Violation]:
    if not filepath.endswith(".h"):
        return []

    has_simt_vf = False
    has_namespace = False
    simt_vf_line = 0

    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        is_comment = (stripped.startswith("//") or stripped.startswith("/*")
                      or stripped.startswith("*"))
        if not is_comment and "__simt_vf__" in line:
            has_simt_vf = True
            if simt_vf_line == 0:
                simt_vf_line = i
        if not is_comment and re.search(r"\bnamespace\s+ascendc_ops\b", line):
            has_namespace = True

    if has_simt_vf and not has_namespace:
        return [{
            "file": filepath,
            "line": simt_vf_line,
            "detail": "Header has __simt_vf__ kernels but no 'namespace ascendc_ops' declaration"
        }]
    return []


# ---------------------------------------------------------------------------
# Check 2: missing_kernel_operator
#   Any .h/.cpp with __aicore__ but no #include <kernel_operator.h>
#   or #include "kernel_operator.h"
# ---------------------------------------------------------------------------
_RE_KERNEL_OP_INCLUDE = re.compile(
    r'#\s*include\s*[<"]kernel_operator\.h[>"]'
)


def check_missing_kernel_operator(filepath: str, lines: List[str]) -> List[Violation]:
    has_aicore = False
    has_include = False
    aicore_line = 0

    for i, line in enumerate(lines, 1):
        if "__aicore__" in line:
            has_aicore = True
            if aicore_line == 0:
                aicore_line = i
        if _RE_KERNEL_OP_INCLUDE.search(line):
            has_include = True

    # Also check if file includes another header that would transitively include it.
    # We only flag if the file itself has __aicore__ and no direct include —
    # .cpp files that include a .h with the include are still flagged since the
    # checker should be conservative.
    if has_aicore and not has_include:
        return [{
            "file": filepath,
            "line": aicore_line,
            "detail": "File uses __aicore__ but does not #include <kernel_operator.h>"
        }]
    return []


# ---------------------------------------------------------------------------
# Check 3: unconditional_simt_compat
#   #include "simt_compat.h" NOT inside #if.*ASCENDC_CPU_DEBUG
# ---------------------------------------------------------------------------
_RE_SIMT_COMPAT_INCLUDE = re.compile(r'#\s*include\s*"simt_compat\.h"')
_RE_CPU_DEBUG_IF = re.compile(r'#\s*if.*ASCENDC_CPU_DEBUG')


def check_unconditional_simt_compat(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    for i, line in enumerate(lines, 1):
        if _RE_SIMT_COMPAT_INCLUDE.search(line):
            # Check if the preceding non-blank line is #if.*ASCENDC_CPU_DEBUG
            guarded = False
            for j in range(i - 2, max(i - 4, -1), -1):  # look up to 2 lines back
                if j < 0:
                    break
                prev = lines[j].strip()
                if not prev:
                    continue  # skip blank lines
                if _RE_CPU_DEBUG_IF.search(prev):
                    guarded = True
                break  # only check the first non-blank preceding line
            if not guarded:
                violations.append({
                    "file": filepath,
                    "line": i,
                    "detail": '#include "simt_compat.h" is not guarded by '
                              "#if defined(ASCENDC_CPU_DEBUG)"
                })
    return violations


# ---------------------------------------------------------------------------
# Check 4: bf16_static_cast
#   static_cast<float>(...) involving bfloat16 / bf16 on same or adjacent line
# ---------------------------------------------------------------------------
_RE_STATIC_CAST_FLOAT = re.compile(r"static_cast\s*<\s*float\s*>\s*\(")
_RE_BF16_TOKEN = re.compile(r"\b(?:bfloat16_t|bfloat16|bf16_t|bf16)\b", re.IGNORECASE)


def _is_comment_line(line: str) -> bool:
    """Check if a line is a single-line comment (// or /* ... */)."""
    stripped = line.lstrip()
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")


def check_bf16_static_cast(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    n = len(lines)
    for i in range(n):
        # Skip lines that are comments — they often discuss the pattern
        if _is_comment_line(lines[i]):
            continue
        if _RE_STATIC_CAST_FLOAT.search(lines[i]):
            # Check current line and adjacent lines (i-1, i, i+1),
            # but only non-comment lines contribute bf16 tokens
            has_bf16_nearby = False
            for j in range(max(0, i - 1), min(n, i + 2)):
                if _is_comment_line(lines[j]):
                    continue
                if _RE_BF16_TOKEN.search(lines[j]):
                    has_bf16_nearby = True
                    break
            if has_bf16_nearby:
                violations.append({
                    "file": filepath,
                    "line": i + 1,
                    "detail": "static_cast<float>() used near bfloat16/bf16 type — "
                              "bisheng does not support this; use bit-manipulation helpers"
                })
    return violations


# ---------------------------------------------------------------------------
# Check 5: simt_namespace
#   using namespace AscendC::Simt  (should be just AscendC)
# ---------------------------------------------------------------------------
_RE_SIMT_NAMESPACE = re.compile(r"using\s+namespace\s+AscendC\s*::\s*Simt")


def check_simt_namespace(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    for i, line in enumerate(lines, 1):
        if _RE_SIMT_NAMESPACE.search(line):
            violations.append({
                "file": filepath,
                "line": i,
                "detail": "'using namespace AscendC::Simt' — should be 'using namespace AscendC' "
                          "(Simt is accessed via AscendC::Simt::VF_CALL, not imported)"
            })
    return violations


# ---------------------------------------------------------------------------
# Check 6: float_fp16_param
#   In extern "C" functions whose name contains fp16/bf16, detect `float`
#   parameters with names containing "num" or "val" (heuristic for mistyped
#   scalar passing — fp16/bf16 scalars should be passed as uint16_t bits).
# ---------------------------------------------------------------------------
_RE_EXTERN_C_FUNC = re.compile(
    r'extern\s+"C".*?void\s+(\w+)\s*\(([^)]*)\)',
    re.DOTALL
)
_RE_FLOAT_SUSPECT_PARAM = re.compile(
    r"\bfloat\s+(\w*(?:num|val)\w*)\b", re.IGNORECASE
)


def check_float_fp16_param(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    full_text = "".join(lines)

    for m in _RE_EXTERN_C_FUNC.finditer(full_text):
        func_name = m.group(1)
        params_text = m.group(2)

        # Only check fp16/bf16 kernels
        if not re.search(r"(?:fp16|bf16)", func_name, re.IGNORECASE):
            continue

        for pm in _RE_FLOAT_SUSPECT_PARAM.finditer(params_text):
            param_name = pm.group(1)
            # Compute line number of the match
            match_pos = m.start(2) + pm.start()
            line_num = full_text[:match_pos].count("\n") + 1
            violations.append({
                "file": filepath,
                "line": line_num,
                "detail": "float parameter '{}' in fp16/bf16 kernel '{}' — "
                          "fp16/bf16 scalars should be passed as uint16_t bits "
                          "to avoid implicit promotion".format(param_name, func_name)
            })
    return violations


# ---------------------------------------------------------------------------
# Check 7: sort_bounds_missing
#   histogram[ without a preceding if.*<.*max_key guard within 5 lines
# ---------------------------------------------------------------------------
_RE_HISTOGRAM_ACCESS = re.compile(r"\bhistogram\s*\[")
_RE_MAX_KEY_GUARD = re.compile(r"\bif\b.*<.*\bmax_key\b")


def check_sort_bounds_missing(filepath: str, lines: List[str]) -> List[Violation]:
    violations = []
    n = len(lines)
    for i in range(n):
        line_stripped = lines[i].lstrip()
        # Skip comment-only lines (// ... or /* ... */)
        if line_stripped.startswith("//") or line_stripped.startswith("/*"):
            continue
        if _RE_HISTOGRAM_ACCESS.search(lines[i]):
            # Check if any of the preceding 5 lines has a max_key guard
            guarded = False
            for j in range(max(0, i - 5), i):
                if _RE_MAX_KEY_GUARD.search(lines[j]):
                    guarded = True
                    break
            if not guarded:
                violations.append({
                    "file": filepath,
                    "line": i + 1,
                    "detail": "histogram[] access without preceding "
                              "'if ... < max_key' bounds guard within 5 lines"
                })
    return violations


_RE_SIMT_VF_NONVOID = re.compile(
    r'__simt_vf__\s+__aicore__\s+(?:inline\s+)?(?!void\b)(\w+)')


def check_simt_vf_nonvoid(filepath: str, lines: List[str]) -> List[Violation]:
    """__simt_vf__ functions MUST return void. Helper functions should not have __simt_vf__."""
    violations = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        m = _RE_SIMT_VF_NONVOID.search(line)
        if m:
            violations.append({
                "file": filepath,
                "line": i + 1,
                "detail": f"__simt_vf__ function returns '{m.group(1)}' instead of void "
                          f"— helper functions should use __aicore__ inline without __simt_vf__"
            })
    return violations


# ---------------------------------------------------------------------------
# Check 9: cann_wrapper_call
#   Detect calls to CANN built-in operator APIs (aclnn*, aclop*, acl_op_*,
#   torch_npu.*, npu_*) that indicate wrapping instead of genuine kernel code.
#   This is a reward-hacking guardrail: generated kernels must implement actual
#   computation logic, not forward to CANN built-in implementations.
# ---------------------------------------------------------------------------
_RE_CANN_WRAPPER_PATTERNS = [
    (re.compile(r'\baclnn[A-Z]\w*\s*\('), "aclnn* API call"),
    (re.compile(r'\baclop[A-Z]\w*\s*\('), "aclop* API call"),
    (re.compile(r'\bacl_op_\w+\s*\('), "acl_op_* API call"),
    (re.compile(r'\baclrtLaunchKernel\s*\('), "aclrtLaunchKernel call (launching pre-built kernel)"),
    (re.compile(r'#\s*include\s*[<"]acl/acl_op_compiler\.h[>"]'), "ACL op compiler header"),
    (re.compile(r'#\s*include\s*[<"]aclnn/\w+\.h[>"]'), "aclnn API header"),
    (re.compile(r'\btorch_npu\b'), "torch_npu reference in kernel code"),
    (re.compile(r'\bnpu_bridge\b'), "npu_bridge reference in kernel code"),
]


def check_cann_wrapper_call(filepath: str, lines: List[str]) -> List[Violation]:
    """Detect CANN built-in operator API calls — kernels must implement logic, not wrap.
    Excludes pybind11 bridge files (they legitimately reference torch_npu for binding)."""
    # Skip pybind11 bridge files — they're the Python binding layer, not kernel code
    basename = os.path.basename(filepath)
    if basename.startswith("pybind11") or basename == "torch_binding.cpp":
        return []

    violations = []
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        for pattern, desc in _RE_CANN_WRAPPER_PATTERNS:
            if pattern.search(line):
                violations.append({
                    "file": filepath,
                    "line": i,
                    "detail": f"CANN wrapper detected: {desc} — "
                              f"kernel must implement actual computation, not call built-in ops"
                })
    return violations


# ---------------------------------------------------------------------------
# Check 10: kernel_has_computation
#   Verify that kernel .h files contain actual computation logic (memory ops,
#   arithmetic, control flow) rather than being trivial stubs or wrappers.
#   A genuine AscendC kernel should have: TQue/TBuf declarations, DataCopy
#   or SetAtomicAdd/etc calls, and VEC or scalar computation.
# ---------------------------------------------------------------------------
# SIMD kernels use TQue/DataCopy/VEC; SIMT kernels use raw GM pointers + scalar loops.
# Both are legitimate — the check must detect WHICH style and validate accordingly.
_RE_SIMD_MARKERS = {
    "tque_or_tbuf": re.compile(r'\bT(?:Que|Buf)\s*<'),
    "data_copy": re.compile(r'\bDataCopy\b'),
    "vec_op": re.compile(r'\b(?:Add|Sub|Mul|Div|Abs|Exp|Reciprocal|Muls|Adds|Cast|'
                         r'ReduceSum|ReduceMax|ReduceMin|WholeReduceSum|Duplicate|'
                         r'Compare|Select|Gather|Scatter)\s*[<(]'),
    "pipe_or_enque": re.compile(r'\b(?:EnQue|DeQue|SetFlag|WaitFlag)\b'),
    "global_tensor": re.compile(r'\bGlobalTensor\s*<'),
    "local_tensor": re.compile(r'\bLocalTensor\s*<'),
}

_RE_SIMT_MARKERS = {
    "gm_addr": re.compile(r'\bGM_ADDR\b'),
    "simt_vf": re.compile(r'__simt_vf__'),
    "simt_thread": re.compile(r'\bSimt::(?:GetThreadIdx|GetThreadNum)\b'),
    "gm_pointer": re.compile(r'__gm__\s+(?:const\s+)?(?:float|half|int|uint)'),
    "scalar_loop": re.compile(r'\bfor\s*\(.*<.*\)'),
    "arithmetic": re.compile(r'[+\-*/]=|[+\-*/]\s'),
}

# Minimum markers for each style
_MIN_SIMD_MARKERS = 3
_MIN_SIMT_MARKERS = 3


def check_kernel_has_computation(filepath: str, lines: List[str]) -> List[Violation]:
    """Verify kernel files contain actual AscendC computation, not trivial stubs.
    Distinguishes SIMD (TQue/DataCopy/VEC) from SIMT (raw GM pointers/scalar loops)."""
    if not filepath.endswith(".h"):
        return []

    full_text = "".join(lines)
    if "__aicore__" not in full_text:
        return []

    # Detect kernel style
    is_simt = bool(re.search(r'__simt_vf__', full_text))

    if is_simt:
        markers = _RE_SIMT_MARKERS
        min_required = _MIN_SIMT_MARKERS
        style = "SIMT"
    else:
        markers = _RE_SIMD_MARKERS
        min_required = _MIN_SIMD_MARKERS
        style = "SIMD"

    found = set()
    for name, pat in markers.items():
        if pat.search(full_text):
            found.add(name)

    if len(found) < min_required:
        return [{
            "file": filepath,
            "line": 1,
            "detail": f"{style} kernel has only {len(found)}/{min_required} "
                      f"computation markers (found: {sorted(found)}). "
                      f"This may be a trivial stub or CANN wrapper."
        }]
    return []


# ---------------------------------------------------------------------------
# Registry of all checks
# ---------------------------------------------------------------------------
CHECKS = [
    ("missing_namespace", check_missing_namespace),
    ("missing_kernel_operator", check_missing_kernel_operator),
    ("unconditional_simt_compat", check_unconditional_simt_compat),
    ("bf16_static_cast", check_bf16_static_cast),
    ("simt_namespace", check_simt_namespace),
    ("float_fp16_param", check_float_fp16_param),
    ("sort_bounds_missing", check_sort_bounds_missing),
    ("simt_vf_nonvoid", check_simt_vf_nonvoid),
    ("cann_wrapper_call", check_cann_wrapper_call),
    ("kernel_has_computation", check_kernel_has_computation),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_files(directory: str) -> List[str]:
    """Recursively collect .h and .cpp files."""
    result = []
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith((".h", ".cpp")):
                result.append(os.path.join(root, fname))
    return result


def run_checks(directory: str) -> dict:
    files = collect_files(directory)
    if not files:
        print("Warning: no .h or .cpp files found in '{}'".format(directory),
              file=sys.stderr)

    report = {
        "passed": True,
        "checks": {},
        "summary": "",
    }

    total_violations = 0
    checks_passed = 0

    for check_name, check_fn in CHECKS:
        all_violations = []  # type: List[Violation]
        for fpath in files:
            lines = read_lines(fpath)
            # Make paths relative to the scanned directory for cleaner output
            rel_path = os.path.relpath(fpath, directory)
            violations = check_fn(fpath, lines)
            # Rewrite file paths to relative
            for v in violations:
                v["file"] = rel_path
            all_violations.extend(violations)

        check_passed = len(all_violations) == 0
        if check_passed:
            checks_passed += 1
        else:
            report["passed"] = False

        total_violations += len(all_violations)
        report["checks"][check_name] = {
            "passed": check_passed,
            "violations": all_violations,
        }

    report["summary"] = "{}/{} checks passed, {} violations found".format(
        checks_passed, len(CHECKS), total_violations
    )

    return report


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: {} <directory>".format(sys.argv[0]), file=sys.stderr)
        return 2

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Error: '{}' is not a directory".format(directory), file=sys.stderr)
        return 2

    report = run_checks(directory)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
