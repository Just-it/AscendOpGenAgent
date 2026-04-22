#!/usr/bin/env python3
"""
precision_gate.py — AscendOpGenAgent adapted precision gate

Chain-of-gates validation + loop control for precision tuning.

Usage:
    python3 precision_gate.py --step <step> --task-name <name> [--workdir <path>] --attempt <N>

Steps: forensics, audit, fix, validate
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


MAX_ATTEMPTS = 2
MAX_STAGNANT_ROUNDS = 2


class GateChecker:
    def __init__(self, task_name: str, workdir: str = ".", attempt: int = 0):
        self.task_name = task_name
        self.workdir = Path(workdir).resolve()
        self.task_dir = self.workdir / task_name
        self.attempt = attempt
        self.tuning_dir = self.task_dir / "precision_tuning"

    # ================================================================
    # Gate-F: Forensics report
    # ================================================================

    def check_forensics(self) -> dict:
        path = self.tuning_dir / f"forensics_report_{self.attempt}.json"
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
        baseline_path = self.tuning_dir / "baseline_state.json"
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
                "note": "Initial precision captured at Gate-F before any code modification"
            }
            self.tuning_dir.mkdir(parents=True, exist_ok=True)
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_state, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # ================================================================
    # Gate-A: Audit report
    # ================================================================

    def check_audit(self) -> dict:
        prereq = self._check_prerequisite_forensics()
        if not prereq["satisfied"]:
            checks = {"prerequisite_forensics": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-A", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        path = self.tuning_dir / f"precision_audit_{self.attempt}.md"
        checks = {
            "prerequisite_forensics": True,
            "report_exists": path.exists(),
            "report_nonempty": False,
            "has_forensics_summary": False,
            "has_computation_decomposition": False,
            "has_kernel_step_trace": False,
            "has_root_cause": False,
            "has_fix_plan": False,
            "has_target_files": False,
            "has_direction_assessment": True,
        }
        content = None
        if checks["report_exists"]:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            checks["report_nonempty"] = len(content) > 200
            for tag, key in [("FORENSICS_SUMMARY", "has_forensics_summary"),
                             ("COMPUTATION_DECOMPOSITION", "has_computation_decomposition"),
                             ("KERNEL_STEP_TRACE", "has_kernel_step_trace"),
                             ("ROOT_CAUSE", "has_root_cause"),
                             ("FIX_PLAN", "has_fix_plan"),
                             ("TARGET_FILES", "has_target_files")]:
                checks[key] = f"[{tag}]" in content
            checks["has_direction_assessment"] = (
                self.attempt == 0 or "[DIRECTION_ASSESSMENT]" in content
            )
            if self.attempt > 0 and "[DIRECTION_ASSESSMENT]" in content:
                checks["direction_assessment_binary"] = self._validate_direction_binary(content)
            elif self.attempt > 0:
                checks["direction_assessment_binary"] = False

        gate_result = self._result("GATE-A", checks)
        if gate_result["passed"] and content:
            self._write_audit_index(content)
        return gate_result

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
            sources = list(kernel_dir.glob("*.cpp"))
            checks["has_kernel_sources"] = len(sources) > 0 and any(s.name != "pybind11.cpp" for s in sources)
        return self._result("GATE-X", checks)

    # ================================================================
    # Gate-V: Validation + loop control
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

        result_path = self.tuning_dir / f"validation_result_attempt_{self.attempt}.json"
        checks = {
            "prerequisite_code": True,
            "result_exists": result_path.exists(),
            "result_parseable": False,
            "precision_passed": False,
        }

        correctness_passed = False
        if checks["result_exists"]:
            try:
                with open(result_path) as f:
                    r = json.load(f)
                checks["result_parseable"] = True
                correctness_passed = r.get("correctness_passed", False)
                checks["precision_passed"] = correctness_passed
            except (json.JSONDecodeError, KeyError):
                pass

        loop_signal, loop_reason, stop_reason_code = self._compute_loop_signal(correctness_passed)

        gate_result = self._result("GATE-V", checks)
        gate_result["loop_signal"] = loop_signal
        gate_result["loop_reason"] = loop_reason
        gate_result["stop_reason_code"] = stop_reason_code
        gate_result["attempt"] = self.attempt
        gate_result["max_attempts"] = MAX_ATTEMPTS

        self._write_round_summary(stop_reason_code)
        self._write_tuning_directions(stop_reason_code)
        return gate_result

    # ================================================================
    # Prerequisites
    # ================================================================

    def _check_prerequisite_forensics(self) -> dict:
        path = self.tuning_dir / f"forensics_report_{self.attempt}.json"
        if not path.exists():
            return {"satisfied": False,
                    "reason": f"forensics_report_{self.attempt}.json missing",
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
            return {"satisfied": True, "reason": "", "detail": {}}
        except (json.JSONDecodeError, KeyError) as e:
            return {"satisfied": False, "reason": f"forensics parse error: {e}",
                    "detail": {"forensics_exists": True, "forensics_attempt_match": False}}

    def _check_prerequisite_audit(self) -> dict:
        path = self.tuning_dir / f"precision_audit_{self.attempt}.md"
        if not path.exists():
            return {"satisfied": False, "reason": f"precision_audit_{self.attempt}.md missing",
                    "detail": {"audit_exists": False}}
        if path.stat().st_size < 100:
            return {"satisfied": False, "reason": f"audit too small",
                    "detail": {"audit_exists": True}}
        return {"satisfied": True, "reason": "", "detail": {}}

    def _check_prerequisite_code(self) -> dict:
        kernel_dir = self.task_dir / "kernel"
        if not kernel_dir.is_dir():
            return {"satisfied": False, "reason": "kernel/ directory missing",
                    "detail": {"kernel_exists": False}}
        sources = [s for s in kernel_dir.glob("*.cpp") if s.name != "pybind11.cpp"]
        if not sources:
            return {"satisfied": False, "reason": "no kernel .cpp sources found",
                    "detail": {"kernel_exists": False}}
        return {"satisfied": True, "reason": "", "detail": {}}

    # ================================================================
    # Loop control
    # ================================================================

    def _compute_loop_signal(self, passed: bool) -> tuple:
        if passed:
            return "PASS", "Precision verification passed", "precision_passed"

        if self.attempt + 1 >= MAX_ATTEMPTS:
            return "STOP", f"Max attempts ({MAX_ATTEMPTS}) reached", "max_attempts_reached"

        forensics_path = self.tuning_dir / f"forensics_report_{self.attempt}.json"
        if forensics_path.exists():
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)
                trend = fr.get("history_trend")
                if trend:
                    trend_list = trend.get("trend", [])
                    if self._detect_harmful_regression(trend_list):
                        return "STOP", "Detected A-B-A oscillation regression", "harmful_regression"
                    if not trend.get("mismatch_improving", True):
                        stagnant = self._count_stagnant(trend_list)
                        if stagnant >= MAX_STAGNANT_ROUNDS:
                            direction_ok = self._check_direction_assessment()
                            if direction_ok == "continue":
                                return "CONTINUE", f"Stagnant {stagnant} rounds but direction changed", "stagnant_new_direction"
                            else:
                                return "STOP", f"Stagnant {stagnant} rounds, same direction", "stagnant_same_direction"
            except (json.JSONDecodeError, KeyError):
                pass

        return "CONTINUE", f"Precision not passed, entering attempt {self.attempt + 2}", None

    def _count_stagnant(self, trend: list) -> int:
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 2:
            return 0
        count = 0
        for i in range(len(ratios) - 1, 0, -1):
            if ratios[i] >= ratios[i - 1]:
                count += 1
            else:
                break
        return count

    def _detect_harmful_regression(self, trend: list) -> bool:
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 3:
            return False
        r_prev, r_mid, r_curr = ratios[-3], ratios[-2], ratios[-1]
        mid_improved = (r_prev - r_mid) > 0.01
        curr_regressed = r_curr >= (r_prev - 0.005)
        return mid_improved and curr_regressed

    def _check_direction_assessment(self) -> str:
        path = self.tuning_dir / f"precision_audit_{self.attempt}.md"
        if not path.exists():
            return "unknown"
        try:
            with open(path) as f:
                content = f.read()
            marker = "[DIRECTION_ASSESSMENT]"
            start = content.find(marker)
            if start == -1:
                return "unknown"
            start += len(marker)
            next_bracket = content.find("\n[", start)
            section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()
            if not section:
                return "unknown"
            for line in section.split("\n"):
                if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                    continue
                colon_pos = line.find(":")
                if colon_pos == -1:
                    continue
                answer = line[colon_pos + 1:].strip()
                first = answer.split()[0] if answer.split() else answer
                first = first.rstrip("，。！？、；：…,.")
                if first == "否":
                    return "continue"
                elif first == "是":
                    return "stop"
            return "unknown"
        except (OSError, UnicodeDecodeError):
            return "unknown"

    def _validate_direction_binary(self, content: str) -> bool:
        marker = "[DIRECTION_ASSESSMENT]"
        start = content.find(marker)
        if start == -1:
            return False
        start += len(marker)
        next_bracket = content.find("\n[", start)
        section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            answer = line[colon_pos + 1:].strip()
            first = answer.split()[0] if answer.split() else answer
            first = first.rstrip("，。！？、；：…,.")
            return first in ("是", "否")
        return False

    # ================================================================
    # Index / summary writers
    # ================================================================

    def _extract_section(self, content: str, section_name: str):
        marker = f"[{section_name}]"
        start = content.find(marker)
        if start == -1:
            return None
        start += len(marker)
        end_marker = content.find("\n[", start)
        end_audit = content.find("=== END AUDIT ===", start)
        candidates = [pos for pos in [end_marker, end_audit] if pos != -1]
        end = min(candidates) if candidates else len(content)
        text = content[start:end].strip()
        return text if text else None

    def _extract_fix_type(self, content: str):
        section = self._extract_section(content, "FIX_PLAN")
        if not section:
            return None
        m = re.search(r"FIX_PRECISION_\w+", section)
        return m.group(0) if m else None

    def _write_audit_index(self, content: str) -> None:
        attempt_dir = self.tuning_dir / "history" / f"attempt_{self.attempt}"
        sections_dir = attempt_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)

        SECTION_MAP = [
            ("forensics_summary", "FORENSICS_SUMMARY"),
            ("computation_decomposition", "COMPUTATION_DECOMPOSITION"),
            ("kernel_step_trace", "KERNEL_STEP_TRACE"),
            ("knowledge_match", "KNOWLEDGE_MATCH"),
            ("root_cause", "ROOT_CAUSE"),
            ("fix_plan", "FIX_PLAN"),
            ("target_files", "TARGET_FILES"),
            ("direction_assessment", "DIRECTION_ASSESSMENT"),
        ]

        sections_index = {}
        base = f"precision_tuning/history/attempt_{self.attempt}/sections"
        for key, tag in SECTION_MAP:
            sec_text = self._extract_section(content, tag)
            rel_path = f"{base}/{key}.md"
            if sec_text is not None:
                abs_path = sections_dir / f"{key}.md"
                try:
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(f"[{tag}]\n\n{sec_text}\n")
                    sections_index[key] = rel_path
                except OSError:
                    sections_index[key] = None
            else:
                sections_index[key] = None

        diagnostics = {
            "forensics_hint": None,
            "op_type": None,
            "fix_type": self._extract_fix_type(content),
            "changed_locations": [],
            "direction_verdict": None,
        }

        n = self.attempt
        index = {
            "forensics": f"precision_tuning/history/attempt_{n}/forensics_report.json",
            "audit_full": f"precision_tuning/precision_audit_{n}.md",
            "sections": sections_index,
            "code_snapshot": f"precision_tuning/history/attempt_{n}/code_snapshot/",
            "validation": f"precision_tuning/validation_result_attempt_{n}.json",
            "compilation_log": None,
            "tuning_directions": "precision_tuning/tuning_directions.json",
            "forensics_used": f"precision_tuning/forensics_report_{n}.json",
        }

        initial_summary = {
            "attempt": self.attempt,
            "metrics": {
                "match_rate": None,
                "mismatch_ratio": None,
                "improvement_ratio": None,
                "absolute_improvement": None,
                "stop_reason_code": None,
            },
            "diagnostics": diagnostics,
            "index": index,
        }
        summary_path = self.tuning_dir / f"round_summary_{self.attempt}.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(initial_summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _write_round_summary(self, stop_reason_code) -> None:
        summary_path = self.tuning_dir / f"round_summary_{self.attempt}.json"
        if stop_reason_code is None:
            stop_reason_code = "validation_failed"

        existing = {}
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        match_rate = None
        mismatch_ratio = None
        improvement_ratio = None
        absolute_improvement = None
        forensics_hint = None
        op_type = None

        result_path = self.tuning_dir / f"validation_result_attempt_{self.attempt}.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    r = json.load(f)
                mr_str = r.get("match_rate")
                if mr_str is not None:
                    match_rate = round(float(mr_str), 4)
                    mismatch_ratio = round(1 - match_rate / 100, 8)
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                pass

        if match_rate is not None:
            if self.attempt == 0:
                baseline_match_rate = self._get_baseline_match_rate()
                if baseline_match_rate is not None:
                    baseline_mismatch = 1 - baseline_match_rate / 100
                    curr_mismatch = 1 - match_rate / 100
                    remaining = 100 - baseline_match_rate
                    if remaining > 0:
                        improvement_ratio = round((match_rate - baseline_match_rate) / remaining, 4)
                    absolute_improvement = round(match_rate - baseline_match_rate, 4)
            elif self.attempt > 0:
                prev_result_path = self.tuning_dir / f"validation_result_attempt_{self.attempt - 1}.json"
                if prev_result_path.exists():
                    try:
                        with open(prev_result_path) as f:
                            prev_r = json.load(f)
                        prev_mr_str = prev_r.get("match_rate")
                        if prev_mr_str is not None:
                            prev_match_rate = float(prev_mr_str)
                            prev_mismatch = 1 - prev_match_rate / 100
                            curr_mismatch = 1 - match_rate / 100
                            remaining = 100 - prev_match_rate
                            if remaining > 0:
                                improvement_ratio = round((match_rate - prev_match_rate) / remaining, 4)
                            absolute_improvement = round(match_rate - prev_match_rate, 4)
                    except (json.JSONDecodeError, KeyError, OSError, ValueError):
                        pass

        forensics_path = self.tuning_dir / f"forensics_report_{self.attempt}.json"
        if forensics_path.exists():
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)
                forensics_hint = fr.get("primary_hint")
                op_type = fr.get("op_type") or fr.get("L8_operator", {}).get("op_type")
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        summary = dict(existing)
        summary["attempt"] = self.attempt
        metrics = summary.get("metrics", {})
        metrics.update({
            "match_rate": match_rate,
            "mismatch_ratio": mismatch_ratio,
            "improvement_ratio": improvement_ratio,
            "absolute_improvement": absolute_improvement,
            "stop_reason_code": stop_reason_code,
        })
        summary["metrics"] = metrics
        diagnostics = summary.get("diagnostics", {})
        diagnostics["forensics_hint"] = forensics_hint
        diagnostics["op_type"] = op_type
        summary["diagnostics"] = diagnostics
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _write_tuning_directions(self, stop_reason_code) -> None:
        directions_path = self.tuning_dir / "tuning_directions.json"
        data = {"task_name": self.task_name, "final_status": "in_progress", "entries": []}
        if directions_path.exists():
            try:
                with open(directions_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        fix_type = None
        direction_verdict = None
        forensics_hint = None
        improvement_ratio = None
        absolute_improvement = None
        match_rate = None
        mismatch_ratio = None

        summary_path = self.tuning_dir / f"round_summary_{self.attempt}.json"
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
                diag = summary.get("diagnostics", {})
                fix_type = diag.get("fix_type")
                direction_verdict = diag.get("direction_verdict")
                forensics_hint = diag.get("forensics_hint")
                metrics = summary.get("metrics", {})
                improvement_ratio = metrics.get("improvement_ratio")
                absolute_improvement = metrics.get("absolute_improvement")
                match_rate = metrics.get("match_rate")
                mismatch_ratio = metrics.get("mismatch_ratio")
            except (json.JSONDecodeError, OSError):
                pass

        if stop_reason_code == "precision_passed":
            outcome = "passed"
        elif improvement_ratio is None:
            outcome = "stagnant"
        elif improvement_ratio < -0.05:
            outcome = "regressed"
        elif improvement_ratio >= 0.1:
            outcome = "improved"
        else:
            outcome = "stagnant"

        new_entry = {
            "attempt": self.attempt,
            "fix_type": fix_type,
            "forensics_hint": forensics_hint,
            "direction_verdict": direction_verdict,
            "improvement_ratio": improvement_ratio,
            "absolute_improvement": absolute_improvement,
            "outcome": outcome,
            "evidence": {
                "forensics_ref": f"precision_tuning/forensics_report_{self.attempt}.json",
                "audit_ref": f"precision_tuning/precision_audit_{self.attempt}.md",
                "match_rate": match_rate,
                "mismatch_ratio": mismatch_ratio,
            }
        }

        data["entries"] = [e for e in data["entries"] if e.get("attempt") != self.attempt]
        data["entries"].append(new_entry)
        data["entries"].sort(key=lambda e: e.get("attempt", 0))

        terminal_codes = {
            "max_attempts_reached", "stagnant_same_direction",
            "stagnant_new_direction", "harmful_regression", "prerequisite_failure",
        }
        if stop_reason_code == "precision_passed":
            data["final_status"] = "success"
            for entry in data["entries"]:
                ir = entry.get("improvement_ratio")
                same_fix = entry.get("fix_type") == fix_type
                nonneg = ir is None or ir >= 0
                entry["contributed"] = same_fix and nonneg
            for entry in data["entries"]:
                if entry.get("attempt") == self.attempt:
                    entry["contributed"] = True
        elif stop_reason_code in terminal_codes:
            data["final_status"] = "failed"

        try:
            with open(directions_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _get_baseline_match_rate(self) -> float | None:
        baseline_path = self.tuning_dir / "baseline_state.json"
        if baseline_path.exists():
            try:
                with open(baseline_path) as f:
                    bs = json.load(f)
                mr = bs.get("match_rate")
                if mr is not None:
                    return float(mr)
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        forensics_path = self.tuning_dir / f"forensics_report_{self.attempt}.json"
        if forensics_path.exists():
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)
                history_trend = fr.get("history_trend")
                if history_trend:
                    trend_list = history_trend.get("trend", [])
                    if len(trend_list) >= 2:
                        baseline_mismatch = trend_list[0].get("mismatch_ratio")
                        if baseline_mismatch is not None:
                            return round((1 - float(baseline_mismatch)) * 100, 4)
                if self.attempt == 0:
                    outputs = fr.get("outputs", [])
                    if outputs:
                        raw_mr = outputs[0].get("basic_stats", {}).get("match_rate")
                        if raw_mr is not None:
                            return round(float(raw_mr) * 100, 4)
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass
        return None

    def _result(self, gate_name: str, checks: dict) -> dict:
        return {"gate": gate_name, "passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description="AscendOpGenAgent precision gate")
    parser.add_argument("--step", required=True, choices=["forensics", "audit", "fix", "validate"])
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()

    ck = GateChecker(args.task_name, args.workdir, args.attempt)
    dispatch = {
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
