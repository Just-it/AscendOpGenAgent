#!/usr/bin/env python3
"""
precision_gate.py — 精度调优 Gate 结构化验证 + 链式前置检查 + 循环控制

核心设计:
  每个 Gate 不仅检查当前步骤的产物, 还验证前序步骤是否已完成。
  即使 Agent 试图跳步, Gate 脚本会拒绝 (返回码 ≠ 0)。

链式依赖:
  Gate-F (forensics)  → 无前置依赖
  Gate-A (audit)      → 前置: forensics_report 存在且 attempt 匹配
  Gate-X (fix)        → 前置: precision_audit_{attempt}.md 存在
  Gate-V (validate)   → 前置: 代码已修复 (kernel 存在)

循环控制:
  Gate-V 额外输出 loop_signal: PASS / CONTINUE / STOP
  Agent 无权覆盖此信号。

用法:
    python3 precision_gate.py --step <step> --op-name <n> --output-path <path> --attempt <N>

返回码: 0=通过, 1=未通过(可重试), 2=致命错误(前置缺失)
"""

import argparse
import json
import os
import re
import sys


MAX_ATTEMPTS = 2
MAX_STAGNANT_ROUNDS = 2


class GateChecker:

    def __init__(self, op_name: str, output_path: str, attempt: int = 0):
        self.op_name = op_name
        self.output_path = output_path
        self.attempt = attempt
        self.tuning_dir = os.path.join(output_path, "precision_tuning")

    # ================================================================
    # Gate-F: 取证报告 (无前置依赖)
    # ================================================================

    def check_forensics(self) -> dict:
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        checks = {
            "report_exists": os.path.exists(path),
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
                # 验证 attempt 号匹配 (防止用旧报告)
                checks["attempt_matches"] = r.get("attempt", -1) == self.attempt
            except (json.JSONDecodeError, KeyError):
                r = None

        gate_result = self._result("GATE-F", checks)

        # Gate-F 通过且为 attempt 0 时：从 forensics outputs 写 baseline_state.json
        # 此时代码尚未被修改，forensics 中的精度数据就是真正的 baseline
        # 仅当 baseline_state.json 不存在时写入（幂等）
        if gate_result["passed"] and self.attempt == 0 and r is not None:
            self._write_baseline_from_forensics(r)

        return gate_result

    def _write_baseline_from_forensics(self, forensics: dict) -> None:
        """
        从 forensics_report 的 outputs[0].basic_stats 提取精度数据，
        写入 baseline_state.json。

        只在 baseline_state.json 不存在时写入（幂等），确保 baseline
        永远记录第一次 Gate-F 时代码未修改的原始状态。
        """
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if os.path.exists(baseline_path):
            return  # 已存在，不覆盖

        try:
            outputs = forensics.get("outputs", [])
            if not outputs:
                return
            stats = outputs[0].get("basic_stats", {})
            raw_match_rate = stats.get("match_rate")
            raw_mismatch_ratio = stats.get("mismatch_ratio")
            if raw_match_rate is None:
                return

            # forensics 里的 match_rate 单位是 0~1 的比例（非百分比）
            # 需要乘以 100 转换为百分比
            baseline_match_rate = round(float(raw_match_rate) * 100, 4)
            baseline_mismatch_ratio = float(raw_mismatch_ratio) if raw_mismatch_ratio is not None else None

            baseline_state = {
                "match_rate":      baseline_match_rate,
                "mismatch_ratio":  baseline_mismatch_ratio,
                "max_abs_diff":    stats.get("max_abs_diff"),
                "mean_abs_diff":   stats.get("mean_abs_diff"),
                "primary_hint":    forensics.get("primary_hint"),
                "source":          "forensics_report_0.json/outputs[0]/basic_stats",
                "note":            "Initial precision captured at Gate-F before any code modification"
            }
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_state, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # ================================================================
    # Gate-A: 审计报告 — 前置: forensics 已完成
    # ================================================================

    def check_audit(self) -> dict:
        # 链式前置检查: forensics 必须存在且 attempt 匹配
        prereq = self._check_prerequisite_forensics()
        if not prereq["satisfied"]:
            checks = {"prerequisite_forensics": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-A", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        checks = {
            "prerequisite_forensics": True,
            "report_exists": os.path.exists(path),
            "report_nonempty": False,
            "has_forensics_summary": False,
            "has_computation_decomposition": False,
            "has_reference_impl_spec": False,
            "has_kernel_step_trace": False,
            "has_root_cause": False,
            "has_fix_plan": False,
            "has_target_files": False,
            "has_direction_assessment": True,  # 第一轮可选，后续必填
        }
        content = None
        if checks["report_exists"]:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            checks["report_nonempty"] = len(content) > 200
            for tag, key in [("FORENSICS_SUMMARY", "has_forensics_summary"),
                             ("COMPUTATION_DECOMPOSITION", "has_computation_decomposition"),
                             ("REFERENCE_IMPL_SPEC", "has_reference_impl_spec"),
                             ("KERNEL_STEP_TRACE", "has_kernel_step_trace"),
                             ("ROOT_CAUSE", "has_root_cause"),
                             ("FIX_PLAN", "has_fix_plan"),
                             ("TARGET_FILES", "has_target_files")]:
                checks[key] = f"[{tag}]" in content
            # DIRECTION_ASSESSMENT 仅在 attempt > 0 时要求
            checks["has_direction_assessment"] = (
                self.attempt == 0 or "[DIRECTION_ASSESSMENT]" in content
            )
            # 二值格式校验：attempt > 0 时额外验证答案为 "是" 或 "否"（防止 Agent 填写模糊内容）
            if self.attempt > 0:
                if "[DIRECTION_ASSESSMENT]" in content:
                    checks["direction_assessment_binary"] = self._validate_direction_binary(content)
                else:
                    checks["direction_assessment_binary"] = False

        gate_result = self._result("GATE-A", checks)

        # Gate-A 通过后：自动提取 sections + 写 round_summary 初始字段（diagnostics + index）
        if gate_result["passed"] and content:
            self._write_audit_index(content)

        return gate_result

    # ================================================================
    # Gate-X: 代码完整性 — 前置: audit 已完成
    # ================================================================

    def check_fix(self) -> dict:
        # 链式前置检查: audit 必须存在
        prereq = self._check_prerequisite_audit()
        if not prereq["satisfied"]:
            checks = {"prerequisite_audit": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-X", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        project_dir = self._find_project_dir()
        checks = {
            "prerequisite_audit": True,
            "project_dir_exists": project_dir is not None,
            "kernel_exists": False,
            "kernel_nonempty": False,
            "host_exists": False,
        }
        if project_dir:
            kp = os.path.join(project_dir, "op_kernel", f"{self.op_name.lower()}_custom.cpp")
            hp = os.path.join(project_dir, "op_host", f"{self.op_name.lower()}_custom.cpp")
            checks["kernel_exists"] = os.path.exists(kp)
            if checks["kernel_exists"]:
                checks["kernel_nonempty"] = os.path.getsize(kp) > 100
            checks["host_exists"] = os.path.exists(hp)
        return self._result("GATE-X", checks)

    # ================================================================
    # Gate-V: 验证结果 + 循环控制 — 前置: 代码已修复
    # ================================================================

    def check_validate(self) -> dict:
        # 链式前置检查: 代码文件必须存在
        prereq = self._check_prerequisite_code()
        if not prereq["satisfied"]:
            checks = {"prerequisite_code": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-V", checks)
            result["prerequisite_error"] = prereq["reason"]
            result["loop_signal"] = "STOP"
            result["loop_reason"] = f"前置条件不满足: {prereq['reason']}"
            result["stop_reason_code"] = "prerequisite_failure"
            return result

        result_path = os.path.join(self.tuning_dir,
                                   f"validation_result_attempt_{self.attempt}.json")
        checks = {
            "prerequisite_code": True,
            "result_exists": os.path.exists(result_path),
            "result_parseable": False,
            "precision_passed": False,
        }

        correctness_passed = False
        match_rate_str = None
        if checks["result_exists"]:
            try:
                with open(result_path) as f:
                    r = json.load(f)
                checks["result_parseable"] = True
                correctness_passed = r.get("correctness_passed", False)
                checks["precision_passed"] = correctness_passed
                match_rate_str = r.get("match_rate")
            except (json.JSONDecodeError, KeyError):
                pass

        loop_signal, loop_reason, stop_reason_code = self._compute_loop_signal(
            correctness_passed, match_rate_str
        )

        gate_result = self._result("GATE-V", checks)
        gate_result["loop_signal"] = loop_signal
        gate_result["loop_reason"] = loop_reason
        gate_result["stop_reason_code"] = stop_reason_code
        gate_result["attempt"] = self.attempt
        gate_result["max_attempts"] = MAX_ATTEMPTS

        # 写入 round_summary_{N}.json（合并 Agent 语义字段 + Gate 数值字段）
        self._write_round_summary(stop_reason_code)

        # 追加本轮方向记录到 tuning_directions.json（读 round_summary 获取 diagnostics）
        self._write_tuning_directions(stop_reason_code)

        return gate_result

    # ================================================================
    # 前置依赖检查
    # ================================================================

    def _check_prerequisite_forensics(self) -> dict:
        """检查 forensics_report_{attempt}.json 存在且 attempt 匹配"""
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if not os.path.exists(path):
            return {"satisfied": False,
                    "reason": f"forensics_report_{self.attempt}.json 不存在, 必须先运行 precision_forensics.py",
                    "detail": {"forensics_exists": False, "forensics_attempt_match": False}}
        try:
            with open(path) as f:
                r = json.load(f)
            if r.get("status") != "completed":
                return {"satisfied": False,
                        "reason": f"forensics 状态异常: {r.get('status')}",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            if r.get("attempt", -1) != self.attempt:
                return {"satisfied": False,
                        "reason": f"forensics attempt={r.get('attempt')} 不匹配当前 attempt={self.attempt}, "
                                  f"必须重新运行 precision_forensics.py",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            return {"satisfied": True, "reason": "", "detail": {}}
        except (json.JSONDecodeError, KeyError) as e:
            return {"satisfied": False, "reason": f"forensics 解析失败: {e}",
                    "detail": {"forensics_exists": True, "forensics_attempt_match": False}}

    def _check_prerequisite_audit(self) -> dict:
        """检查 precision_audit_{attempt}.md 存在"""
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return {"satisfied": False,
                    "reason": f"precision_audit_{self.attempt}.md 不存在, 必须先完成审计 (Step 2)",
                    "detail": {"audit_exists": False}}
        if os.path.getsize(path) < 100:
            return {"satisfied": False,
                    "reason": f"precision_audit_{self.attempt}.md 内容过少, 审计可能未完成",
                    "detail": {"audit_exists": True}}
        return {"satisfied": True, "reason": "", "detail": {}}

    def _check_prerequisite_code(self) -> dict:
        """检查代码文件存在"""
        project_dir = self._find_project_dir()
        if not project_dir:
            return {"satisfied": False,
                    "reason": "找不到项目目录",
                    "detail": {"project_exists": False}}
        kp = os.path.join(project_dir, "op_kernel", f"{self.op_name.lower()}_custom.cpp")
        if not os.path.exists(kp) or os.path.getsize(kp) < 100:
            return {"satisfied": False,
                    "reason": f"{kp} 不存在或为空, 必须先完成代码修复 (Step 3)",
                    "detail": {"kernel_exists": False}}
        return {"satisfied": True, "reason": "", "detail": {}}

    # ================================================================
    # 循环控制
    # ================================================================

    def _compute_loop_signal(self, passed: bool, match_rate_str=None) -> tuple:
        if passed:
            return "PASS", "精度验证通过", "precision_passed"

        # Nearly success: evaluate.py 返回 FAIL 但 match_rate >= 99%
        # 通常为量化噪声或精度阈值过严（非实现错误），标注供人工确认
        if match_rate_str is not None:
            try:
                mr = float(match_rate_str)
                if mr >= 99.0:
                    return (
                        "NEARLY_SUCCESS",
                        f"精度接近通过 (match_rate={mr:.2f}%), 可能为量化误差或精度阈值过严，建议人工确认后决定是否接受",
                        "nearly_success",
                    )
            except (ValueError, TypeError):
                pass

        if self.attempt + 1 >= MAX_ATTEMPTS:
            return "STOP", f"已达最大轮次 ({MAX_ATTEMPTS})", "max_attempts_reached"

        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if os.path.exists(forensics_path):
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)
                trend = fr.get("history_trend")
                if trend:
                    trend_list = trend.get("trend", [])

                    # 先检测 A→B→A 振荡型有害回退
                    if self._detect_harmful_regression(trend_list):
                        return "STOP", "检测到 A→B→A 振荡型有害回退，需人工分析", "harmful_regression"

                    if not trend.get("mismatch_improving", True):
                        stagnant = self._count_stagnant(trend_list)
                        if stagnant >= MAX_STAGNANT_ROUNDS:
                            direction_ok = self._check_direction_assessment()
                            if direction_ok == "continue":
                                return "CONTINUE", (
                                    f"mismatch 连续 {stagnant} 轮未改善, "
                                    f"但 Agent 已明确换方向，继续探索"
                                ), "stagnant_new_direction"
                            else:
                                return "STOP", (
                                    f"mismatch 连续 {stagnant} 轮未改善, "
                                    f"Agent 仍沿用同一方向，可能方向错误，需人工分析"
                                ), "stagnant_same_direction"
            except (json.JSONDecodeError, KeyError):
                pass

        return "CONTINUE", f"精度未通过, 进入第 {self.attempt + 2} 轮", None

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
        """
        检测 A→B→A 振荡型有害回退。
        条件: 最近3轮中，中间轮 mismatch 改善 > 1%（绝对），当前轮回退至 ≥ 初始轮 - 0.5%。
        """
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 3:
            return False
        r_prev, r_mid, r_curr = ratios[-3], ratios[-2], ratios[-1]
        mid_improved = (r_prev - r_mid) > 0.01      # 中间轮改善 > 1%
        curr_regressed = r_curr >= (r_prev - 0.005)  # 当前轮回退至接近初始水平
        return mid_improved and curr_regressed

    def _compute_improvement_ratio(self, prev_mismatch: float, curr_mismatch: float):
        """
        计算本轮改善比率（剩余可修复空间中的改善幅度）。
        improvement_ratio = (curr_match_rate - prev_match_rate) / (100 - prev_match_rate)
        返回 None 如果上一轮 match_rate 已为 100%（无剩余空间）。
        """
        prev_match = (1 - prev_mismatch) * 100
        curr_match = (1 - curr_mismatch) * 100
        remaining = 100 - prev_match
        if remaining <= 0:
            return None
        return round((curr_match - prev_match) / remaining, 4)

    def _write_round_summary(self, stop_reason_code) -> None:
        """
        Gate-V 调用：将 metrics 数值字段合并进 round_summary（Gate-A 已写 diagnostics + index）。
        同时补充 diagnostics.forensics_hint/op_type 和 index.compilation_log。

        注意：stop_reason_code 可能为 None（当 Gate-V 返回 CONTINUE 且无特殊原因时），
        此时应设为默认值 "validation_failed"。
        """
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")

        # 确保 stop_reason_code 不为 None
        if stop_reason_code is None:
            stop_reason_code = "validation_failed"

        # 读取 Gate-A 已写入的初始结构（diagnostics + index）
        existing = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # 自愈：若 Gate-A 数据缺失（crash/重入），从 audit 文件重新提取
        if not existing.get("gate_a_complete"):
            audit_path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
            if os.path.exists(audit_path):
                try:
                    with open(audit_path, encoding="utf-8") as f:
                        _content = f.read()
                    self._write_audit_index(_content)
                    # 重新加载刚写入的 initial_summary
                    if os.path.exists(summary_path):
                        with open(summary_path) as f:
                            existing = json.load(f)
                except (OSError, json.JSONDecodeError):
                    pass

        match_rate = None
        mismatch_ratio = None
        improvement_ratio = None
        absolute_improvement = None
        forensics_hint = None
        op_type = None

        # 从 validation_result_attempt_N.json 读取精确的验证后 match_rate（修复前的来源有误）
        result_path = os.path.join(self.tuning_dir, f"validation_result_attempt_{self.attempt}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    r = json.load(f)
                mr_str = r.get("match_rate")
                if mr_str is not None:
                    match_rate = round(float(mr_str), 4)
                    mismatch_ratio = round(1 - match_rate / 100, 8)
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                pass

        # 计算相对上一轮的 improvement_ratio（对比 N-1 的 validation_result）
        # 同时计算绝对改善幅度（match_rate 的百分点变化）
        if match_rate is not None:
            if self.attempt == 0:
                # attempt 0: 对比 baseline（从 forensics_report 的 history_trend 读取）
                baseline_match_rate = self._get_baseline_match_rate()
                if baseline_match_rate is not None:
                    baseline_mismatch = 1 - baseline_match_rate / 100
                    curr_mismatch = 1 - match_rate / 100
                    improvement_ratio = self._compute_improvement_ratio(
                        baseline_mismatch, curr_mismatch
                    )
                    absolute_improvement = round(match_rate - baseline_match_rate, 4)
            elif self.attempt > 0:
                # attempt N (N>0): 对比 attempt N-1
                prev_result_path = os.path.join(
                    self.tuning_dir, f"validation_result_attempt_{self.attempt - 1}.json"
                )
                if os.path.exists(prev_result_path):
                    try:
                        with open(prev_result_path) as f:
                            prev_r = json.load(f)
                        prev_mr_str = prev_r.get("match_rate")
                        if prev_mr_str is not None:
                            prev_match_rate = float(prev_mr_str)
                            prev_mismatch = 1 - prev_match_rate / 100
                            curr_mismatch = 1 - match_rate / 100
                            improvement_ratio = self._compute_improvement_ratio(
                                prev_mismatch, curr_mismatch
                            )
                            # 计算绝对改善（百分点变化）
                            absolute_improvement = round(match_rate - prev_match_rate, 4)
                    except (json.JSONDecodeError, KeyError, OSError, ValueError):
                        pass

        # 从 forensics_report 读取元数据（hint/op_type），不用于 metrics 计算
        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if os.path.exists(forensics_path):
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)
                forensics_hint = fr.get("primary_hint")
                op_type = fr.get("op_type") or fr.get("L8_operator", {}).get("op_type")
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        # 检查 compilation_log 文件
        compile_log_abs = os.path.join(
            self.tuning_dir, f"compilation_log_{self.attempt}.json"
        )
        compilation_log_ref = (
            f"precision_tuning/compilation_log_{self.attempt}.json"
            if os.path.exists(compile_log_abs) else None
        )

        # 在 existing 基础上合并
        summary = dict(existing)
        summary["attempt"] = self.attempt

        metrics = summary.get("metrics", {})
        metrics.update({
            "match_rate":            match_rate,
            "mismatch_ratio":        mismatch_ratio,
            "improvement_ratio":     improvement_ratio,
            "absolute_improvement":  absolute_improvement,
            "stop_reason_code":      stop_reason_code,
        })
        summary["metrics"] = metrics

        diagnostics = summary.get("diagnostics", {})
        diagnostics["forensics_hint"] = forensics_hint
        diagnostics["op_type"] = op_type
        summary["diagnostics"] = diagnostics

        index = summary.get("index", {})
        index["compilation_log"] = compilation_log_ref
        summary["index"] = index

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _write_tuning_directions(self, stop_reason_code) -> None:
        """
        Gate-V 调用：将本轮方向学习记录追加到 tuning_directions.json。
        精度通过时回溯标注所有 entry 的 contributed 字段。

        Schema:
          {
            "op_name": "...",
            "final_status": "in_progress" | "success" | "failed",
            "entries": [
              {
                "attempt": N,
                "fix_type": "FIX_PRECISION_XXX" | null,
                "forensics_hint": "tail_spike" | null,
                "direction_verdict": "是" | "否" | null,
                "improvement_ratio": float | null,
                "outcome": "passed" | "improved" | "stagnant" | "regressed",
                "contributed": bool  # 仅 final_status=success 时存在
              }
            ]
          }
        """
        directions_path = os.path.join(self.tuning_dir, "tuning_directions.json")

        # 读取已有文件或初始化
        data = {"op_name": self.op_name, "final_status": "in_progress", "entries": []}
        if os.path.exists(directions_path):
            try:
                with open(directions_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # 从 round_summary 读取 Gate-A 已写的 diagnostics + Gate-V 刚写的 metrics
        fix_type = None
        direction_verdict = None
        forensics_hint = None
        improvement_ratio = None
        absolute_improvement = None
        match_rate = None
        mismatch_ratio = None

        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        if os.path.exists(summary_path):
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

        # 判断 outcome
        if stop_reason_code in ("precision_passed", "nearly_success"):
            outcome = "passed"
        elif improvement_ratio is None:
            outcome = "stagnant"   # attempt=0 无前轮可比，或计算失败
        elif improvement_ratio < -0.05:
            outcome = "regressed"
        elif improvement_ratio >= 0.1:
            outcome = "improved"
        else:
            outcome = "stagnant"

        # 提取 direction_reason（换方向理由）
        direction_reason = self._extract_direction_reason()

        # 构造新 entry（contributed 暂不写入，精度通过后回溯）
        new_entry = {
            "attempt":               self.attempt,
            "fix_type":              fix_type,
            "forensics_hint":        forensics_hint,
            "direction_verdict":     direction_verdict,
            "direction_reason":      direction_reason,
            "improvement_ratio":     improvement_ratio,
            "absolute_improvement":  absolute_improvement,
            "outcome":               outcome,
            "evidence": {
                "forensics_ref":     f"precision_tuning/forensics_report_{self.attempt}.json",
                "audit_ref":         f"precision_tuning/precision_audit_{self.attempt}.md",
                "match_rate":        match_rate,
                "mismatch_ratio":    mismatch_ratio,
            }
        }

        # 幂等：移除同 attempt 的旧 entry 再追加
        data["entries"] = [e for e in data["entries"] if e.get("attempt") != self.attempt]
        data["entries"].append(new_entry)
        data["entries"].sort(key=lambda e: e.get("attempt", 0))

        # 更新 final_status，精度通过时回溯标注 contributed
        terminal_codes = {
            "max_attempts_reached", "stagnant_same_direction",
            "stagnant_new_direction", "harmful_regression", "prerequisite_failure",
            "nearly_success",
        }
        if stop_reason_code == "precision_passed":
            data["final_status"] = "success"
            # 与成功轮 fix_type 相同且未回退的历史轮标注为 contributed
            for entry in data["entries"]:
                ir = entry.get("improvement_ratio")
                same_fix = entry.get("fix_type") == fix_type
                nonneg = ir is None or ir >= 0  # attempt=0 improvement_ratio=None 视为中性
                entry["contributed"] = same_fix and nonneg
            # 成功轮本身必为 contributed
            for entry in data["entries"]:
                if entry.get("attempt") == self.attempt:
                    entry["contributed"] = True
        elif stop_reason_code == "nearly_success":
            data["final_status"] = "nearly_success"
            # nearly_success 不回溯 contributed，由人工确认后决定
        elif stop_reason_code in terminal_codes:
            data["final_status"] = "failed"

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(directions_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _extract_direction_reason(self) -> str | None:
        """
        从 precision_audit_{attempt}.md 的 [DIRECTION_ASSESSMENT] section 提取换方向理由。

        返回:
          换方向理由字符串，如果未找到或不适用则返回 None
        """
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return None

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            marker = "[DIRECTION_ASSESSMENT]"
            start = content.find(marker)
            if start == -1:
                return None
            start += len(marker)
            next_bracket = content.find("\n[", start)
            section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()

            if not section:
                return None

            # 提取"换方向理由"字段
            for line in section.split("\n"):
                if "换方向理由" in line:
                    colon_pos = line.find(":")
                    if colon_pos == -1:
                        continue
                    reason = line[colon_pos + 1:].strip()
                    return reason if reason else None

            return None
        except (OSError, UnicodeDecodeError):
            return None

    def _check_direction_assessment(self) -> str:
        """
        读取当前轮 precision_audit_{attempt}.md 中的 [DIRECTION_ASSESSMENT] section，
        判断 Agent 是否在主动换方向。

        返回:
          "continue" — Agent 明确换了方向，可以继续探索
          "stop"     — Agent 仍沿用同一方向，停滞无改善，大概率方向错了
          "unknown"  — section 不存在或解析失败，保守返回 stop
        """
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
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

            # 提取"本轮是否延续上一轮方向"字段，只检查冒号后的答案值
            for line in section.split("\n"):
                key = "本轮是否延续上一轮方向"
                if key not in line and "本轮是否延续" not in line:
                    continue
                # 提取冒号后的值并去除空白
                colon_pos = line.find(":")
                if colon_pos == -1:
                    continue
                answer = line[colon_pos + 1:].strip()
                # 检查答案值：必须精确匹配冒号后的词
                # 否/换方向/换了 → 换方向
                # 是 → 沿用方向
                first_word = self._extract_direction_first_word(answer)
                if first_word == "否":
                    return "continue"
                elif first_word == "是":
                    return "stop"
            return "unknown"
        except (OSError, UnicodeDecodeError):
            return "unknown"

    def _extract_direction_first_word(self, text: str) -> str:
        """从答案文本中提取首个判断词，去除尾部标点干扰。
        例: "否，已换方向" → "否"；"是（继续分析）" → "是（继续分析）".rstrip → "是"
        """
        if not text:
            return ""
        first = text.split()[0] if text.split() else text
        return first.rstrip("，。！？、；：…,.")

    def _validate_direction_binary(self, content: str) -> bool:
        """验证 [DIRECTION_ASSESSMENT] section 中的答案是严格二值（是/否）。"""
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
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            return first_word in ("是", "否")
        return False

    # ================================================================
    # Gate-A: Section 提取与 round_summary 初始写入
    # ================================================================

    def _extract_section(self, content: str, section_name: str):
        """提取 precision_audit_{attempt}.md 中指定 section 的内容。
        边界: [SECTION_NAME] 到下一个 \\n[ 或 === END AUDIT === 结束。
        返回 None 如果 section 不存在（不阻断 Gate-A 通过）。
        """
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
        """从 [FIX_PLAN] section 中提取 FIX_PRECISION_XXX 类型标识。"""
        section = self._extract_section(content, "FIX_PLAN")
        if not section:
            return None
        m = re.search(r"FIX_PRECISION_\w+", section)
        return m.group(0) if m else None

    def _extract_changed_locations(self, content: str) -> list:
        """从 [TARGET_FILES] section 中解析被修改的文件列表（保留文件名 token）。"""
        section = self._extract_section(content, "TARGET_FILES")
        if not section:
            return []
        locations = []
        seen = set()
        for line in section.split("\n"):
            line = line.strip().lstrip("-*•·").strip()
            for part in line.split():
                part = part.rstrip(",:;")
                if re.search(r"\.\w{1,5}$", part) and part not in seen:
                    locations.append(part)
                    seen.add(part)
        return locations

    def _extract_direction_verdict_value(self, content: str):
        """从 [DIRECTION_ASSESSMENT] 提取实际的 '是'/'否' 字符串。attempt==0 时返回 None。"""
        if self.attempt == 0:
            return None
        section = self._extract_section(content, "DIRECTION_ASSESSMENT")
        if not section:
            return None
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            if first_word in ("是", "否"):
                return first_word
        return None

    def _write_audit_index(self, content: str) -> None:
        """Gate-A 通过后：提取各 section 为小文件，写 round_summary 的 diagnostics + index 初始字段。"""
        attempt_dir = os.path.join(self.tuning_dir, "history", f"attempt_{self.attempt}")
        sections_dir = os.path.join(attempt_dir, "sections")
        try:
            os.makedirs(sections_dir, exist_ok=True)
        except OSError:
            return

        SECTION_MAP = [
            ("forensics_summary",        "FORENSICS_SUMMARY"),
            ("computation_decomposition","COMPUTATION_DECOMPOSITION"),
            ("reference_impl_spec",      "REFERENCE_IMPL_SPEC"),
            ("kernel_step_trace",        "KERNEL_STEP_TRACE"),
            ("knowledge_match",          "KNOWLEDGE_MATCH"),
            ("root_cause",               "ROOT_CAUSE"),
            ("fix_plan",                 "FIX_PLAN"),
            ("target_files",             "TARGET_FILES"),
            ("direction_assessment",     "DIRECTION_ASSESSMENT"),
        ]

        sections_index = {}
        base = f"precision_tuning/history/attempt_{self.attempt}/sections"
        for key, tag in SECTION_MAP:
            sec_text = self._extract_section(content, tag)
            rel_path = f"{base}/{key}.md"
            if sec_text is not None:
                abs_path = os.path.join(sections_dir, f"{key}.md")
                try:
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(f"[{tag}]\n\n{sec_text}\n")
                    sections_index[key] = rel_path
                except OSError:
                    sections_index[key] = None
            else:
                sections_index[key] = None

        diagnostics = {
            "forensics_hint":    None,  # Gate-V 补充
            "op_type":           None,  # Gate-V 补充
            "fix_type":          self._extract_fix_type(content),
            "changed_locations": self._extract_changed_locations(content),
            "direction_verdict": self._extract_direction_verdict_value(content),
        }

        n = self.attempt
        index = {
            "forensics":          f"precision_tuning/history/attempt_{n}/forensics_report.json",
            "audit_full":         f"precision_tuning/precision_audit_{n}.md",
            "sections":           sections_index,
            "code_snapshot":      f"precision_tuning/history/attempt_{n}/code_snapshot/",
            "validation":         f"precision_tuning/validation_result_attempt_{n}.json",
            "compilation_log":    None,  # Gate-V 回填（路径: precision_tuning/compilation_log_N.json）
            "tuning_directions":  "precision_tuning/tuning_directions.json",  # 跨轮方向学习表
            "forensics_used":     f"precision_tuning/forensics_report_{n}.json",  # 本轮使用的取证报告（当前轮）
        }

        initial_summary = {
            "attempt": self.attempt,
            "gate_a_complete": True,
            "metrics": {
                "match_rate":            None,
                "mismatch_ratio":        None,
                "improvement_ratio":     None,
                "absolute_improvement":  None,
                "stop_reason_code":      None,
            },
            "diagnostics": diagnostics,
            "index": index,
        }
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(initial_summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    # ================================================================
    # 工具
    # ================================================================

    def _get_baseline_match_rate(self) -> float | None:
        """
        获取 baseline match_rate（修改前的原始精度）。

        三层回退策略（按优先级）：
        1. 读 baseline_state.json（由 Gate-F 在 attempt 0 写入，最可靠）
        2. 读 forensics_report_{attempt}.json/history_trend/trend[0]（多轮时有历史数据）
        3. 直接读 forensics_report_{attempt}.json/outputs[0]/basic_stats（当前轮的原始数据）

        注意：层 3 仅当 attempt == 0 时语义正确（forensics 是修改前的状态）。
        attempt > 0 时 forensics 已是修改后重新取证的数据，不能用作 baseline。
        """
        # --- 层 1：baseline_state.json（最可靠，Gate-F 时固化的原始状态）---
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path) as f:
                    bs = json.load(f)
                mr = bs.get("match_rate")
                if mr is not None:
                    return float(mr)
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        # --- 层 2：forensics_report history_trend（attempt >= 1 时有效）---
        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if os.path.exists(forensics_path):
            try:
                with open(forensics_path) as f:
                    fr = json.load(f)

                history_trend = fr.get("history_trend")
                if history_trend:
                    trend_list = history_trend.get("trend", [])
                    if len(trend_list) >= 2:
                        # 第一个 entry 是历史最初状态
                        baseline_mismatch = trend_list[0].get("mismatch_ratio")
                        if baseline_mismatch is not None:
                            return round((1 - float(baseline_mismatch)) * 100, 4)

                # --- 层 3：当前 forensics outputs（仅 attempt 0 语义正确）---
                if self.attempt == 0:
                    outputs = fr.get("outputs", [])
                    if outputs:
                        raw_mr = outputs[0].get("basic_stats", {}).get("match_rate")
                        if raw_mr is not None:
                            # forensics 里 match_rate 是 0~1 的比例
                            return round(float(raw_mr) * 100, 4)

            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass

        return None

    # ================================================================
    # 工具
    # ================================================================

    def _find_project_dir(self) -> str | None:
        pascal = "".join(w.capitalize() for w in self.op_name.split("_")) + "Custom"
        c = os.path.join(self.output_path, pascal)
        if os.path.isdir(c):
            return c
        try:
            for item in os.listdir(self.output_path):
                full = os.path.join(self.output_path, item)
                if item.endswith("Custom") and os.path.isdir(full):
                    return full
        except FileNotFoundError:
            pass
        return None

    def _result(self, gate_name: str, checks: dict) -> dict:
        return {"gate": gate_name, "passed": all(checks.values()), "checks": checks}


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="精度调优 Gate 验证 (链式)")
    parser.add_argument("--step", required=True,
                        choices=["forensics", "audit", "fix", "validate"])
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()

    ck = GateChecker(args.op_name, args.output_path, args.attempt)
    dispatch = {
        "forensics": ck.check_forensics,
        "audit": ck.check_audit,
        "fix": ck.check_fix,
        "validate": ck.check_validate,
    }

    result = dispatch[args.step]()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 前置依赖失败用返回码 2 (致命), 区别于产物不完整的返回码 1
    if result.get("prerequisite_error"):
        print(f"\n[{result['gate']}] ⛔ PREREQUISITE FAILED — {result['prerequisite_error']}")
        sys.exit(2)

    if result["passed"]:
        print(f"\n[{result['gate']}] ✅ PASSED")
        if args.step == "validate":
            print(f"  loop_signal: {result.get('loop_signal')}")
            print(f"  reason: {result.get('loop_reason')}")
        sys.exit(0)
    else:
        failed = [k for k, v in result["checks"].items() if not v]
        print(f"\n[{result['gate']}] ❌ FAILED — missing: {failed}")
        if args.step == "validate":
            print(f"  loop_signal: {result.get('loop_signal')}")
            print(f"  reason: {result.get('loop_reason')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
