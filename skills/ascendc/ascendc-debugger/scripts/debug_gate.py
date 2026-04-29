#!/usr/bin/env python3
"""
debug_gate.py — Gate control for ascendc-debugger loop (with merged precision tuner)

Usage:
    python3 debug_gate.py --step <step> --task-name <name> [--workdir <path>] [--attempt <N>]

Steps: diagnose, forensics, audit, fix, validate
  - diagnose : check diagnosis_report.json exists/parseable (non-precision modes)
  - forensics: check forensics_report_{attempt}.json exists/parseable (precision mode only)
  - audit    : check debug_audit_{attempt}.md has required sections
               (precision mode requires extra sections: FORENSICS_SUMMARY,
                COMPUTATION_DECOMPOSITION, KERNEL_STEP_TRACE, DIRECTION_ASSESSMENT)
  - fix      : check kernel sources still exist after edit
  - validate : re-run diagnose, decide PASS / CONTINUE / STOP via elastic limits
"""

import argparse
import json
import os
import sys
from pathlib import Path


MIN_ATTEMPTS = 3
MAX_ATTEMPTS = 15

# Elastic attempt limits per failure mode
BASE_ATTEMPTS = {
    "degenerate": 3,
    "compile": 5,
    "precision": 10,
    "verify": 5,
}


class DebugGateChecker:
    def __init__(self, task_name: str, workdir: str = ".", attempt: int = 0):
        self.task_name = task_name
        self.workdir = Path(workdir).resolve()
        self.task_dir = self.workdir / task_name
        self.attempt = attempt
        self.debug_dir = self.task_dir / "debug"

    # ================================================================
    # Gate-D: Diagnosis report
    # ================================================================

    def check_diagnose(self) -> dict:
        path = self.debug_dir / "diagnosis_report.json"
        checks = {
            "report_exists": path.exists(),
            "report_parseable": False,
            "has_failure_mode": False,
            "has_checks": False,
        }
        if checks["report_exists"]:
            try:
                with open(path) as f:
                    r = json.load(f)
                checks["report_parseable"] = True
                checks["has_failure_mode"] = r.get("failure_mode") is not None
                checks["has_checks"] = bool(r.get("checks"))
            except (json.JSONDecodeError, KeyError):
                pass
        return self._result("GATE-D", checks)

    # ================================================================
    # Gate-F: Precision forensics report (precision mode only)
    # ================================================================

    def check_forensics(self) -> dict:
        path = self.debug_dir / f"forensics_report_{self.attempt}.json"
        checks = {
            "report_exists": path.exists(),
            "report_parseable": False,
            "status_completed": False,
            "has_primary_hint": False,
            "has_outputs": False,
            "has_basic_stats": False,
            "attempt_matches": False,
        }
        r = None
        if checks["report_exists"]:
            try:
                with open(path) as f:
                    r = json.load(f)
                checks["report_parseable"] = True
                checks["status_completed"] = r.get("status") == "completed"
                checks["has_primary_hint"] = bool(r.get("primary_hint"))
                checks["has_outputs"] = len(r.get("outputs", [])) > 0
                if checks["has_outputs"]:
                    checks["has_basic_stats"] = "basic_stats" in r["outputs"][0]
                checks["attempt_matches"] = r.get("attempt", -1) == self.attempt
            except (json.JSONDecodeError, KeyError):
                r = None

        gate_result = self._result("GATE-F", checks)
        if gate_result["passed"] and self.attempt == 0 and r is not None:
            self._write_baseline_from_forensics(r)
        return gate_result

    def _write_baseline_from_forensics(self, forensics: dict) -> None:
        baseline_path = self.debug_dir / "baseline_state.json"
        if baseline_path.exists():
            return
        try:
            outputs = forensics.get("outputs", [])
            if not outputs:
                return
            stats = outputs[0].get("basic_stats", {})
            raw_match_rate = stats.get("match_rate")
            if raw_match_rate is None:
                return
            baseline_match_rate = round(float(raw_match_rate) * 100, 4)
            baseline_state = {
                "match_rate": baseline_match_rate,
                "mismatch_ratio": stats.get("mismatch_ratio"),
                "max_abs_diff": stats.get("max_abs_diff"),
                "mean_abs_diff": stats.get("mean_abs_diff"),
                "primary_hint": forensics.get("primary_hint"),
                "source": "forensics_report.json/outputs[0]/basic_stats",
                "note": "Initial precision captured at Gate-F before any code modification",
            }
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_state, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # ================================================================
    # Gate-A: Audit / fix plan
    # ================================================================

    def check_audit(self) -> dict:
        prereq = self._check_prerequisite_diagnose()
        if not prereq["satisfied"]:
            checks = {"prerequisite_diagnose": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-A", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        # Precision mode also requires forensics_report
        failure_mode = self._get_failure_mode()
        if failure_mode == "precision":
            forensics_prereq = self._check_prerequisite_forensics()
            if not forensics_prereq["satisfied"]:
                checks = {"prerequisite_diagnose": True, "prerequisite_forensics": False}
                checks.update(forensics_prereq["detail"])
                result = self._result("GATE-A", checks)
                result["prerequisite_error"] = forensics_prereq["reason"]
                return result

        path = self.debug_dir / f"debug_audit_{self.attempt}.md"
        checks = {
            "prerequisite_diagnose": True,
            "report_exists": path.exists(),
            "report_nonempty": False,
            "has_diagnosis": False,
            "has_root_cause": False,
            "has_fix_plan": False,
        }
        # Extra sections required for precision mode
        if failure_mode == "precision":
            checks["prerequisite_forensics"] = True
            checks["has_forensics_summary"] = False
            checks["has_computation_decomposition"] = False
            checks["has_kernel_step_trace"] = False
            # direction_assessment only required when attempt > 0 (history available)
            if self.attempt > 0:
                checks["has_direction_assessment"] = False

        content = None
        if checks["report_exists"]:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            checks["report_nonempty"] = len(content) > 200
            tag_map = [("DIAGNOSIS", "has_diagnosis"),
                       ("ROOT_CAUSE", "has_root_cause"),
                       ("FIX_PLAN", "has_fix_plan")]
            if failure_mode == "precision":
                tag_map += [("FORENSICS_SUMMARY", "has_forensics_summary"),
                            ("COMPUTATION_DECOMPOSITION", "has_computation_decomposition"),
                            ("KERNEL_STEP_TRACE", "has_kernel_step_trace")]
                if self.attempt > 0:
                    tag_map.append(("DIRECTION_ASSESSMENT", "has_direction_assessment"))
            for tag, key in tag_map:
                checks[key] = f"[{tag}]" in content

        return self._result("GATE-A", checks)

    # ================================================================
    # Gate-X: Code integrity
    # ================================================================

    def check_fix(self) -> dict:
        prereq = self._check_prerequisite_audit()
        if not prereq["satisfied"]:
            checks = {"prerequisite_audit": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-X", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        kernel_dir = self.task_dir / "kernel"
        checks = {
            "prerequisite_audit": True,
            "kernel_dir_exists": kernel_dir.is_dir(),
            "has_kernel_sources": False,
        }
        if checks["kernel_dir_exists"]:
            sources = [s for s in kernel_dir.glob("*.cpp") if s.name != "pybind11.cpp"]
            checks["has_kernel_sources"] = len(sources) > 0
        return self._result("GATE-X", checks)

    # ================================================================
    # Gate-V: Re-diagnosis after fix
    # ================================================================

    def check_validate(self) -> dict:
        prereq = self._check_prerequisite_code()
        if not prereq["satisfied"]:
            checks = {"prerequisite_code": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-V", checks)
            result["prerequisite_error"] = prereq["reason"]
            result["loop_signal"] = "STOP"
            result["loop_reason"] = f"Prerequisite failed: {prereq['reason']}"
            result["stop_reason_code"] = "prerequisite_failure"
            return result

        # Re-run diagnose
        diagnose_script = self.workdir / "skills" / "ascendc" / "ascendc-debugger" / "scripts" / "diagnose.py"
        checks = {
            "prerequisite_code": True,
            "diagnose_ran": False,
            "all_passed": False,
        }

        if diagnose_script.exists():
            try:
                import subprocess
                env = os.environ.copy()
                env["PYTHONPATH"] = str(self.workdir)
                result = subprocess.run(
                    [sys.executable, str(diagnose_script), self.task_name,
                     "--workdir", str(self.workdir)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(self.workdir),
                    env=env,
                )
                checks["diagnose_ran"] = True
                checks["all_passed"] = result.returncode == 0
            except Exception:
                pass

        loop_signal, loop_reason, stop_reason_code = self._compute_loop_signal(checks["all_passed"])

        gate_result = self._result("GATE-V", checks)
        gate_result["loop_signal"] = loop_signal
        gate_result["loop_reason"] = loop_reason
        gate_result["stop_reason_code"] = stop_reason_code
        gate_result["attempt"] = self.attempt
        gate_result["max_attempts"] = MAX_ATTEMPTS
        return gate_result

    # ================================================================
    # Prerequisites
    # ================================================================

    def _check_prerequisite_diagnose(self) -> dict:
        path = self.debug_dir / "diagnosis_report.json"
        if not path.exists():
            return {"satisfied": False, "reason": "diagnosis_report.json missing",
                    "detail": {"diagnose_exists": False}}
        try:
            with open(path) as f:
                r = json.load(f)
            if not r.get("checks"):
                return {"satisfied": False, "reason": "diagnosis has no checks",
                        "detail": {"diagnose_exists": True}}
            return {"satisfied": True, "reason": "", "detail": {}}
        except (json.JSONDecodeError, KeyError) as e:
            return {"satisfied": False, "reason": f"diagnosis parse error: {e}",
                    "detail": {"diagnose_exists": True}}

    def _check_prerequisite_audit(self) -> dict:
        path = self.debug_dir / f"debug_audit_{self.attempt}.md"
        if not path.exists():
            return {"satisfied": False, "reason": f"debug_audit_{self.attempt}.md missing",
                    "detail": {"audit_exists": False}}
        if path.stat().st_size < 100:
            return {"satisfied": False, "reason": "audit too small",
                    "detail": {"audit_exists": True}}
        return {"satisfied": True, "reason": "", "detail": {}}

    def _check_prerequisite_forensics(self) -> dict:
        """Precision mode requires a completed forensics_report for the current attempt."""
        path = self.debug_dir / f"forensics_report_{self.attempt}.json"
        if not path.exists():
            return {"satisfied": False, "reason": f"forensics_report_{self.attempt}.json missing",
                    "detail": {"forensics_exists": False, "forensics_attempt_match": False}}
        try:
            with open(path) as f:
                r = json.load(f)
            if r.get("status") != "completed":
                return {"satisfied": False, "reason": f"forensics status: {r.get('status')}",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            if r.get("attempt", -1) != self.attempt:
                return {"satisfied": False,
                        "reason": f"forensics attempt={r.get('attempt')} != current={self.attempt}",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            return {"satisfied": True, "reason": "",
                    "detail": {"forensics_exists": True, "forensics_attempt_match": True}}
        except (json.JSONDecodeError, KeyError) as e:
            return {"satisfied": False, "reason": f"forensics parse error: {e}",
                    "detail": {"forensics_exists": True, "forensics_attempt_match": False}}

    def _check_prerequisite_code(self) -> dict:
        kernel_dir = self.task_dir / "kernel"
        if not kernel_dir.is_dir():
            return {"satisfied": False, "reason": "kernel/ directory missing",
                    "detail": {"kernel_exists": False}}
        sources = [s for s in kernel_dir.glob("*.cpp") if s.name != "pybind11.cpp"]
        if not sources:
            return {"satisfied": False, "reason": "no kernel .cpp sources",
                    "detail": {"kernel_exists": False}}
        return {"satisfied": True, "reason": "", "detail": {}}

    # ================================================================
    # Loop control
    # ================================================================

    def _get_failure_mode(self) -> str:
        """Read current diagnosis report to determine failure mode."""
        path = self.debug_dir / "diagnosis_report.json"
        if not path.exists():
            return "unknown"
        try:
            with open(path) as f:
                r = json.load(f)
            return r.get("failure_mode", "unknown") or "unknown"
        except (json.JSONDecodeError, KeyError):
            return "unknown"

    def _get_elastic_max_attempts(self, failure_mode: str) -> int:
        """Determine max attempts based on failure mode and progress history."""
        base = BASE_ATTEMPTS.get(failure_mode, 5)

        # Check progress history to decide if we should extend
        progress_history = []
        for i in range(1, self.attempt + 1):
            prev_path = self.debug_dir / f"diagnosis_report_attempt_{i - 1}.json"
            curr_path = self.debug_dir / f"diagnosis_report_attempt_{i}.json"
            if prev_path.exists() and curr_path.exists():
                try:
                    with open(prev_path) as f:
                        prev = json.load(f)
                    with open(curr_path) as f:
                        curr = json.load(f)
                    prev_err = self._count_errors(prev)
                    curr_err = self._count_errors(curr)
                    if curr_err < prev_err:
                        progress_history.append("progress")
                    elif curr_err == prev_err and curr_err == 0:
                        # Precision still improving (match_rate changes)
                        prev_metrics = prev.get("checks", {}).get("verify", {}).get("metrics", {})
                        curr_metrics = curr.get("checks", {}).get("verify", {}).get("metrics", {})
                        prev_match = prev_metrics.get("match_rate", 0)
                        curr_match = curr_metrics.get("match_rate", 0)
                        if curr_match > prev_match:
                            progress_history.append("progress")
                        else:
                            progress_history.append("stuck")
                    else:
                        progress_history.append("stuck")
                except (json.JSONDecodeError, KeyError):
                    pass

        # If making consistent progress, extend up to MAX_ATTEMPTS
        if len(progress_history) >= 2 and progress_history[-1] == "progress" and progress_history[-2] == "progress":
            return MAX_ATTEMPTS

        # If stuck for 2 consecutive rounds, cap at base
        if len(progress_history) >= 2 and progress_history[-1] == "stuck" and progress_history[-2] == "stuck":
            return min(base, self.attempt + 1)

        return base

    def _compute_loop_signal(self, passed: bool) -> tuple:
        if passed:
            return "PASS", "All checks passed", "all_passed"

        failure_mode = self._get_failure_mode()
        elastic_max = self._get_elastic_max_attempts(failure_mode)

        # Always do at least MIN_ATTEMPTS
        if self.attempt + 1 < MIN_ATTEMPTS:
            return "CONTINUE", f"Attempt {self.attempt + 1}/{MIN_ATTEMPTS} (minimum), mode={failure_mode}", None

        if self.attempt + 1 >= elastic_max:
            return "STOP", f"Max elastic attempts ({elastic_max}) reached for mode={failure_mode}", "max_attempts_reached"

        # Check if we're making progress by comparing with previous diagnosis
        prev_diagnose = self.debug_dir / f"diagnosis_report_attempt_{self.attempt - 1}.json"
        curr_diagnose = self.debug_dir / "diagnosis_report.json"

        if self.attempt > 0 and prev_diagnose.exists() and curr_diagnose.exists():
            try:
                with open(prev_diagnose) as f:
                    prev = json.load(f)
                with open(curr_diagnose) as f:
                    curr = json.load(f)

                prev_errors = self._count_errors(prev)
                curr_errors = self._count_errors(curr)

                if curr_errors < prev_errors:
                    return "CONTINUE", f"Progress: {prev_errors} errors -> {curr_errors} errors (max={elastic_max})", None
                elif curr_errors == prev_errors and curr_errors == 0:
                    # Same error count but still failing (e.g. precision still bad)
                    prev_match = prev.get("checks", {}).get("verify", {}).get("metrics", {}).get("match_rate", 0)
                    curr_match = curr.get("checks", {}).get("verify", {}).get("metrics", {}).get("match_rate", 0)
                    if curr_match > prev_match:
                        return "CONTINUE", f"Precision improving: {prev_match:.2f}% -> {curr_match:.2f}% (max={elastic_max})", None
                    return "CONTINUE", f"Still fixing, continue next attempt (max={elastic_max})", None
                else:
                    return "STOP", f"No progress: {prev_errors} errors -> {curr_errors} errors", "no_progress"
            except (json.JSONDecodeError, KeyError):
                pass

        return "CONTINUE", f"Not passed, entering attempt {self.attempt + 2} (max={elastic_max}, mode={failure_mode})", None

    def _count_errors(self, report: dict) -> int:
        checks = report.get("checks", {})
        total = 0
        if not checks.get("degenerate", {}).get("passed", True):
            total += 1
        compile_checks = checks.get("compile", {})
        if not compile_checks.get("passed", True):
            total += compile_checks.get("num_errors", 1)
        verify_checks = checks.get("verify", {})
        if not verify_checks.get("passed", True):
            total += verify_checks.get("metrics", {}).get("num_mismatched", 1)
        return total

    def _result(self, gate_name: str, checks: dict) -> dict:
        return {"gate": gate_name, "passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="AscendC debug gate")
    parser.add_argument("--step", required=True,
                        choices=["diagnose", "forensics", "audit", "fix", "validate"])
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()

    ck = DebugGateChecker(args.task_name, args.workdir, args.attempt)
    dispatch = {
        "diagnose": ck.check_diagnose,
        "forensics": ck.check_forensics,
        "audit": ck.check_audit,
        "fix": ck.check_fix,
        "validate": ck.check_validate,
    }

    result = dispatch[args.step]()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("prerequisite_error"):
        print(f"\n[{result['gate']}] PREREQUISITE FAILED — {result['prerequisite_error']}")
        sys.exit(2)

    if result["passed"]:
        print(f"\n[{result['gate']}] PASSED")
        if args.step == "validate":
            print(f"  loop_signal: {result.get('loop_signal')}")
            print(f"  reason: {result.get('loop_reason')}")
        sys.exit(0)
    else:
        failed = [k for k, v in result["checks"].items() if not v]
        print(f"\n[{result['gate']}] FAILED — missing: {failed}")
        if args.step == "validate":
            print(f"  loop_signal: {result.get('loop_signal')}")
            print(f"  reason: {result.get('loop_reason')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
