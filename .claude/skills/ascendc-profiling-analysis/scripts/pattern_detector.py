#!/usr/bin/env python3
"""
T7: pattern_detector.py — 周期性模式检测

检测空泡是否呈周期性模式（每个 tile 重复出现），
区分系统性空泡 vs 偶发空泡 vs 冷启动/尾部效应。

用法:
    python3 pattern_detector.py --simulator-dir <path> --core-id <core>
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, get_pipeline_events,
    find_concurrent_events, ps_to_us, ps_to_ns,
    is_cubecore, PID_VECTOR, PID_CUBE,
)
from bubble_classifier import (
    classify_gap, build_concurrent_state, BubbleType, BubbleSubType,
)

# D 类间隙阈值
D_CLASS_GAP_THRESHOLD = 500.0


def _compute_autocorrelation(gaps: list[float], max_lag: int) -> list[float]:
    """
    计算间隙序列的自相关系数。

    Returns:
        autocorr[lag] for lag in 0..max_lag
    """
    n = len(gaps)
    if n < 3:
        return [1.0]

    mean = sum(gaps) / n
    var = sum((g - mean) ** 2 for g in gaps) / n
    if var == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for lag in range(min(max_lag + 1, n)):
        cov = sum((gaps[i] - mean) * (gaps[i + lag] - mean)
                  for i in range(n - lag)) / n
        result.append(cov / var)
    return result


def _classify_pattern_type(autocorr: list[float], phase_stats: dict) -> str:
    """根据自相关和阶段统计判定模式类型"""
    # 检查周期性: 找 lag>0 的最大自相关峰
    if len(autocorr) > 2:
        peak_lag = 0
        peak_val = 0.0
        for i in range(1, len(autocorr)):
            if autocorr[i] > peak_val:
                peak_val = autocorr[i]
                peak_lag = i
        if peak_val > 0.6:
            return "periodic"

    # 检查冷启动主导
    cold = phase_stats.get("cold_start", {})
    steady = phase_stats.get("steady_state", {})
    if cold.get("mean_gap_ps", 0) > steady.get("mean_gap_ps", 1) * 2:
        if cold.get("pct_of_total", 0) > 30:
            return "cold_start_dominant"

    # 检查尾部主导
    tail = phase_stats.get("tail", {})
    if tail.get("mean_gap_ps", 0) > steady.get("mean_gap_ps", 1) * 2:
        if tail.get("pct_of_total", 0) > 30:
            return "tail_dominant"

    return "sporadic"


def detect_patterns(simulator_dir: str, core_id: str) -> dict:
    """
    检测空泡的周期性模式。

    Args:
        simulator_dir: simulator 输出目录
        core_id: 核标识

    Returns:
        模式检测结果 dict
    """
    core_paths = get_all_core_paths(simulator_dir)
    if core_id not in core_paths:
        return {"tool": "T7", "error": f"Core {core_id} not found"}

    pipelines = load_core_trace(core_paths[core_id])

    # 自动检测核类型
    is_cube = is_cubecore(core_id)
    core_type = "cubecore" if is_cube else "veccore"
    main_pipeline_name = "CUBE" if is_cube else "VECTOR"
    main_pid = PID_CUBE if is_cube else PID_VECTOR

    main_events = get_pipeline_events(pipelines, main_pipeline_name)
    if not main_events:
        return {"tool": "T7", "core_id": core_id, "core_type": core_type,
                "error": f"No {main_pipeline_name} events"}

    # 过滤 WAIT_FLAG
    compute_events = [e for e in main_events if not e.is_wait_flag()]
    if len(compute_events) < 3:
        return {"tool": "T7", "core_id": core_id, "error": "Too few compute events"}

    # 提取所有间隙及其分类
    gaps = []
    gap_subtypes = []
    for i in range(1, len(compute_events)):
        gap = compute_events[i].ts - compute_events[i - 1].end_ts
        if gap > D_CLASS_GAP_THRESHOLD:
            concurrent = find_concurrent_events(
                pipelines, compute_events[i - 1].end_ts, compute_events[i].ts,
                exclude_pid=main_pid,
            )
            cs = build_concurrent_state(
                concurrent, pipelines,
                compute_events[i - 1].end_ts, compute_events[i].ts,
                core_type=core_type,
            )
            classification = classify_gap(
                gap_ps=gap,
                op_before=compute_events[i - 1],
                op_after=compute_events[i],
                concurrent_state=cs,
                core_type=core_type,
            )
            gaps.append(gap)
            gap_subtypes.append(classification.sub_type)

    if not gaps:
        return {
            "tool": "T7",
            "core_id": core_id,
            "pattern_type": "no_significant_gaps",
            "periodicity": {"detected": False},
            "phase_analysis": {},
            "dominant_subtype": None,
            "recommendations": ["No significant D-class gaps detected"],
        }

    n = len(gaps)

    # 自相关分析
    max_lag = min(n // 2, 50)
    autocorr = _compute_autocorrelation(gaps, max_lag)

    # 找周期峰
    peak_lag = 0
    peak_val = 0.0
    if len(autocorr) > 1:
        for i in range(1, len(autocorr)):
            if autocorr[i] > peak_val:
                peak_val = autocorr[i]
                peak_lag = i

    periodicity_detected = peak_val > 0.6
    confidence = "high" if peak_val > 0.8 else ("medium" if peak_val > 0.6 else "low")

    # 阶段分析: 首 10% / 中间 80% / 末 10%
    cold_end = max(1, n // 10)
    tail_start = n - max(1, n // 10)
    if tail_start <= cold_end:
        tail_start = cold_end + 1

    cold_gaps = gaps[:cold_end]
    steady_gaps = gaps[cold_end:tail_start] if tail_start > cold_end else []
    tail_gaps = gaps[tail_start:] if tail_start < n else []

    total_gap_ps = sum(gaps)

    def _phase_stats(phase_gaps, label):
        if not phase_gaps:
            return {"mean_gap_ps": 0, "count": 0, "pct_of_total": 0.0}
        s = sum(phase_gaps)
        return {
            "mean_gap_ps": round(s / len(phase_gaps), 1),
            "count": len(phase_gaps),
            "pct_of_total": round(s / total_gap_ps * 100, 1) if total_gap_ps > 0 else 0.0,
        }

    phase_analysis = {
        "cold_start": _phase_stats(cold_gaps, "cold"),
        "steady_state": _phase_stats(steady_gaps, "steady"),
        "tail": _phase_stats(tail_gaps, "tail"),
    }

    pattern_type = _classify_pattern_type(autocorr, phase_analysis)

    # 主导子类型
    subtype_counts = {}
    for st in gap_subtypes:
        subtype_counts[st.value] = subtype_counts.get(st.value, 0) + 1
    dominant_subtype = max(subtype_counts, key=subtype_counts.get) if subtype_counts else None

    # 生成建议
    recommendations = []
    if pattern_type == "periodic" and dominant_subtype:
        if "mte2" in dominant_subtype:
            recommendations.append("周期性 MTE2 等待 → 双缓冲 (P1) 可系统性消除")
        elif "mte3" in dominant_subtype:
            recommendations.append("周期性 MTE3 等待 → 双缓冲 (P1) 可系统性消除")
        else:
            recommendations.append(f"周期性空泡 ({dominant_subtype}) → 需针对性优化")
    if pattern_type == "cold_start_dominant":
        recommendations.append("冷启动空泡占比高 → 可通过预取首 tile 数据缓解")
    if pattern_type == "tail_dominant":
        recommendations.append("尾部空泡占比高 → 检查尾块处理逻辑是否可优化")
    if pattern_type == "sporadic":
        recommendations.append("空泡为偶发性，非系统性瓶颈，优先关注其他方向")

    return {
        "tool": "T7",
        "core_id": core_id,
        "core_type": core_type,
        "pattern_type": pattern_type,
        "periodicity": {
            "detected": periodicity_detected,
            "period_ticks": peak_lag if periodicity_detected else None,
            "autocorr_peak": round(peak_val, 3),
            "confidence": confidence,
        },
        "phase_analysis": phase_analysis,
        "dominant_subtype": dominant_subtype,
        "sub_type_breakdown": subtype_counts,
        "total_gaps_analyzed": n,
        "recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser(description="T7: Periodic pattern detection")
    parser.add_argument("--simulator-dir", required=True)
    parser.add_argument("--core-id", required=True)
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    result = detect_patterns(args.simulator_dir, args.core_id)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T7 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
