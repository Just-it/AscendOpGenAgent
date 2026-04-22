#!/usr/bin/env python3
"""
diagnose.py — AscendC unified diagnostic tool

Runs the complete diagnostic chain:
  1. Degenerate check (validate_ascendc_impl.py)
  2. Compilation (build_ascendc.py)
  3. Verification (verification_ascendc.py)

Outputs structured diagnosis_report.json for the debugger agent.

Usage:
    python3 diagnose.py <task_name> [--workdir <path>] [--soc-version <ver>]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class AscendCDiagnoser:
    def __init__(self, task_name: str, workdir: str = ".", soc_version: str = "Ascend910B2"):
        self.task_name = task_name
        self.workdir = Path(workdir).resolve()
        self.task_dir = self.workdir / task_name
        self.soc_version = soc_version
        self.debug_dir = self.task_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        report = {
            "version": "1.0",
            "task_name": self.task_name,
            "timestamp": time.time(),
            "checks": {},
        }

        # Run checks in order: degenerate -> compile -> verify
        report["checks"]["degenerate"] = self._check_degenerate()

        if report["checks"]["degenerate"]["passed"]:
            report["checks"]["compile"] = self._check_compile()
        else:
            report["checks"]["compile"] = {
                "passed": None,
                "reason": "Skipped due to degenerate check failure",
            }

        if report["checks"]["degenerate"]["passed"] and report["checks"]["compile"]["passed"]:
            report["checks"]["verify"] = self._check_verify()
        else:
            report["checks"]["verify"] = {
                "passed": None,
                "reason": "Skipped due to earlier failure",
            }

        report["failure_mode"] = self._determine_failure_mode(report)

        # Save report
        report_path = self.debug_dir / "diagnosis_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Print summary
        self._print_summary(report)
        return report

    # ================================================================
    # Check 1: Degenerate check
    # ================================================================

    def _check_degenerate(self) -> dict:
        validate_script = self.workdir / "skills" / "ascendc" / "ascendc-translator" / "scripts" / "validate_ascendc_impl.py"
        model_new_path = self.task_dir / "model_new_ascendc.py"

        if not validate_script.exists():
            return {
                "passed": True,  # Skip if script missing
                "reason": "validate_ascendc_impl.py not found, skipping degenerate check",
            }

        if not model_new_path.exists():
            return {
                "passed": False,
                "reason": f"model_new_ascendc.py not found: {model_new_path}",
            }

        try:
            result = subprocess.run(
                [sys.executable, str(validate_script), str(model_new_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Try to parse JSON output
            try:
                parsed = json.loads(result.stdout)
                passed = parsed.get("valid", result.returncode == 0)
                return {
                    "passed": passed,
                    "regression_type": parsed.get("regression_type"),
                    "checks": parsed.get("checks", {}),
                    "suggestion": parsed.get("suggestion", ""),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except json.JSONDecodeError:
                # Fallback: check return code
                return {
                    "passed": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "Degenerate check timed out"}
        except Exception as e:
            return {"passed": False, "reason": str(e)}

    # ================================================================
    # Check 2: Compilation
    # ================================================================

    def _check_compile(self) -> dict:
        build_script = self.workdir / "utils" / "build_ascendc.py"

        if not build_script.exists():
            return {
                "passed": False,
                "reason": f"build_ascendc.py not found: {build_script}",
            }

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(build_script),
                    self.task_name,
                    "-v", self.soc_version,
                    "--clean",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.workdir),
            )

            errors, warnings = self._parse_compile_output(result.stdout + "\n" + result.stderr)

            return {
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "errors": errors,
                "warnings": warnings,
                "num_errors": len(errors),
                "num_warnings": len(warnings),
                "primary_error": errors[0]["category"] if errors else None,
                "affected_files": list(set(e["file"] for e in errors if e.get("file"))),
                "stdout_tail": self._tail(result.stdout, 50),
                "stderr_tail": self._tail(result.stderr, 50),
            }

        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "Compilation timed out"}
        except Exception as e:
            return {"passed": False, "reason": str(e)}

    def _parse_compile_output(self, output: str) -> tuple:
        """Parse cmake/gcc compile output into structured errors and warnings."""
        errors = []
        warnings = []

        lines = output.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # GCC/Clang error pattern
            # Examples:
            #   kernel/file.cpp:45:12: error: 'Vmax' was not declared
            #   kernel/file.cpp:45:12: warning: unused variable
            match = re.match(
                r"(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning):\s*(?P<message>.+)",
                line,
            )
            if match:
                entry = {
                    "file": match.group("file"),
                    "line": int(match.group("line")),
                    "column": int(match.group("col")),
                    "severity": match.group("severity"),
                    "message": match.group("message"),
                    "category": self._classify_compile_error(match.group("message")),
                    "raw": line,
                }
                if match.group("severity") == "error":
                    errors.append(entry)
                else:
                    warnings.append(entry)
                continue

            # CMake error pattern
            # Example: CMake Error at CMakeLists.txt:45
            cmake_match = re.match(
                r"CMake Error(?: at (?P<file>[^:]+):(?P<line>\d+))?:\s*(?P<message>.+)",
                line,
            )
            if cmake_match:
                entry = {
                    "file": cmake_match.group("file") or "CMakeLists.txt",
                    "line": int(cmake_match.group("line")) if cmake_match.group("line") else 0,
                    "column": 0,
                    "severity": "error",
                    "message": cmake_match.group("message"),
                    "category": "cmake_error",
                    "raw": line,
                }
                errors.append(entry)
                continue

            # Linker error pattern
            # Example: undefined reference to `FuncName'
            link_match = re.match(
                r".*(?:undefined reference|cannot find|ld:).*",
                line,
            )
            if link_match:
                entry = {
                    "file": "linker",
                    "line": 0,
                    "column": 0,
                    "severity": "error",
                    "message": line,
                    "category": "link_error",
                    "raw": line,
                }
                errors.append(entry)

        return errors, warnings

    def _classify_compile_error(self, message: str) -> str:
        """Classify compile error message into category."""
        msg = message.lower()

        patterns = [
            ("undefined_api", r"was not declared|undeclared identifier|no member named"),
            ("type_mismatch", r"cannot convert|no matching function|invalid conversion|incompatible types"),
            ("syntax", r"expected|missing|syntax error|unexpected token|invalid token"),
            ("alignment", r"count must be multiple|alignment|not aligned|must be aligned"),
            ("header_missing", r"no such file|cannot find.*include|fatal error"),
            ("template", r"template|specialization|instantiation"),
            ("constexpr", r"constexpr|constant expression"),
            ("attribute", r"__attribute__|deprecated|nodiscard"),
        ]

        for category, pattern in patterns:
            if re.search(pattern, msg):
                return category

        return "other_compile"

    # ================================================================
    # Check 3: Verification
    # ================================================================

    def _check_verify(self) -> dict:
        verify_script = self.workdir / "utils" / "verification_ascendc.py"

        if not verify_script.exists():
            return {
                "passed": None,
                "reason": f"verification_ascendc.py not found: {verify_script}",
            }

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.workdir)

            result = subprocess.run(
                [sys.executable, str(verify_script), self.task_name],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.workdir),
                env=env,
            )

            return self._parse_verify_output(result)

        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": "Verification timed out"}
        except Exception as e:
            return {"passed": False, "reason": str(e)}

    def _parse_verify_output(self, result: subprocess.CompletedProcess) -> dict:
        output = result.stdout + "\n" + result.stderr

        # Check for PASS/FAIL in output
        has_pass = "PASS" in output and "FAIL" not in output.split("PASS")[0].split("\n")[-1]
        has_fail = "FAIL" in output

        # Extract metrics
        metrics = {}

        # Match rate: "match_rate=87.50%" or "mismatch_ratio=12.50%"
        match_rate_match = re.search(r"match_rate[=:]?\s*(\d+\.?\d*)%", output)
        if match_rate_match:
            metrics["match_rate"] = float(match_rate_match.group(1))

        mismatch_match = re.search(r"mismatch_ratio[=:]?\s*(\d+\.?\d*)%", output)
        if mismatch_match:
            metrics["mismatch_ratio"] = float(mismatch_match.group(1))

        max_diff_match = re.search(r"max_abs_diff[=:]?\s*([0-9.eE+-]+)", output)
        if max_diff_match:
            metrics["max_abs_diff"] = float(max_diff_match.group(1))

        # Count mismatched elements
        mismatch_count_match = re.search(r"unequal_elements[=:]?\s*(\d+)", output)
        if mismatch_count_match:
            metrics["num_mismatched"] = int(mismatch_count_match.group(1))

        total_match = re.search(r"total[=:]?\s*(\d+)", output)
        if total_match:
            metrics["total_elements"] = int(total_match.group(1))

        # Determine if it's a precision issue vs other failure
        is_precision_issue = (
            has_fail
            and metrics.get("mismatch_ratio", 0) > 0
            and "shape mismatch" not in output.lower()
            and "nan" not in output.lower()
            and "inf" not in output.lower()
        )

        return {
            "passed": result.returncode == 0 and not has_fail,
            "returncode": result.returncode,
            "has_pass": has_pass,
            "has_fail": has_fail,
            "is_precision_issue": is_precision_issue,
            "metrics": metrics,
            "stdout": output,
        }

    # ================================================================
    # Helpers
    # ================================================================

    def _determine_failure_mode(self, report: dict) -> Optional[str]:
        checks = report["checks"]

        if not checks["degenerate"]["passed"]:
            return "degenerate"

        if not checks["compile"]["passed"]:
            return "compile"

        if checks["verify"]["passed"] is False:
            if checks["verify"].get("is_precision_issue", False):
                return "precision"
            return "verify"

        if all(c["passed"] for c in checks.values()):
            return None  # All passed

        return "unknown"

    def _tail(self, text: str, n_lines: int) -> str:
        lines = text.split("\n")
        return "\n".join(lines[-n_lines:])

    def _print_summary(self, report: dict) -> None:
        mode = report["failure_mode"]
        print(f"[DIAGNOSE] Task: {report['task_name']}")

        if mode is None:
            print("  Status: ALL CHECKS PASSED")
            return

        print(f"  Failure mode: {mode}")

        if mode == "degenerate":
            dg = report["checks"]["degenerate"]
            print(f"  Degenerate check: FAILED")
            print(f"    Type: {dg.get('regression_type', 'unknown')}")
            print(f"    Suggestion: {dg.get('suggestion', 'N/A')[:100]}")

        elif mode == "compile":
            comp = report["checks"]["compile"]
            print(f"  Compile: FAILED")
            print(f"    Errors: {comp.get('num_errors', 'N/A')}")
            print(f"    Warnings: {comp.get('num_warnings', 'N/A')}")
            if comp.get("primary_error"):
                print(f"    Primary error type: {comp['primary_error']}")
            if comp.get("affected_files"):
                print(f"    Affected files: {', '.join(comp['affected_files'])}")

        elif mode == "precision":
            ver = report["checks"]["verify"]
            print(f"  Verification: PRECISION MISMATCH")
            metrics = ver.get("metrics", {})
            if "match_rate" in metrics:
                print(f"    Match rate: {metrics['match_rate']:.2f}%")
            if "max_abs_diff" in metrics:
                print(f"    Max abs diff: {metrics['max_abs_diff']:.6e}")

        elif mode == "verify":
            ver = report["checks"]["verify"]
            print(f"  Verification: FAILED (non-precision)")
            print(f"    Output: {ver.get('stdout', '')[:200]}")


def main():
    parser = argparse.ArgumentParser(description="AscendC unified diagnostic tool")
    parser.add_argument("task_name", help="Task directory name")
    parser.add_argument("--workdir", default=".", help="AscendOpGenAgent root directory")
    parser.add_argument("--soc-version", default="Ascend910B2", help="SoC version")
    args = parser.parse_args()

    diagnoser = AscendCDiagnoser(args.task_name, args.workdir, args.soc_version)
    report = diagnoser.run()

    if report["failure_mode"] is None:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
