#!/usr/bin/env python3
"""
T6: diff_profiling.py — 修改前后 profiling 对比

对 before/after 两个 simulator 目录分别执行 T1+T2+T3，计算所有指标的 delta。

用法:
    python3 diff_profiling.py --before-dir <path1> --after-dir <path2>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_cross_core import analyze_cross_core_balance
from analyze_pipeline_bubbles import analyze_pipeline_bubbles
from analyze_vec_internal import analyze_vec_internal
from pattern_detector import detect_patterns
from overlap_analyzer import analyze_overlap


def _safe_get(d: dict, *keys, default=0.0):
    """安全嵌套取值"""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current if current is not None else default


def diff_profiling(before_dir: str, after_dir: str) -> dict:
    """
    对比两次 profiling 结果。

    Args:
        before_dir: 修改前的 simulator 目录
        after_dir: 修改后的 simulator 目录

    Returns:
        对比结果 dict
    """
    # T1: 跨核均衡
    t1_before = analyze_cross_core_balance(before_dir)
    t1_after = analyze_cross_core_balance(after_dir)

    # 找到最慢核用于 T2/T3
    slowest_before = t1_before.get("slowest_core", "core0.veccore0")
    slowest_after = t1_after.get("slowest_core", "core0.veccore0")

    # T2: D 类空泡
    t2_before = analyze_pipeline_bubbles(before_dir, slowest_before)
    t2_after = analyze_pipeline_bubbles(after_dir, slowest_after)

    # T3: VEC 内部
    t3_before = analyze_vec_internal(before_dir, slowest_before)
    t3_after = analyze_vec_internal(after_dir, slowest_after)

    # T7: 周期性模式
    t7_before = detect_patterns(before_dir, slowest_before)
    t7_after = detect_patterns(after_dir, slowest_after)

    # T8: 重叠度
    t8_before = analyze_overlap(before_dir, slowest_before)
    t8_after = analyze_overlap(after_dir, slowest_after)

    # 提取关键指标
    before_max_us = _safe_get(t1_before, "stats", "max_us")
    after_max_us = _safe_get(t1_after, "stats", "max_us")

    before_imbalance = _safe_get(t1_before, "imbalance_ratio", default=1.0)
    after_imbalance = _safe_get(t1_after, "imbalance_ratio", default=1.0)

    before_d_pct = _safe_get(t2_before, "d_class_pct")
    after_d_pct = _safe_get(t2_after, "d_class_pct")

    before_c_pct = _safe_get(t3_before, "bubble_summary", "c_class", "pct")
    after_c_pct = _safe_get(t3_after, "bubble_summary", "c_class", "pct")

    before_compute_pct = _safe_get(t3_before, "bubble_summary", "pure_compute_pct")
    after_compute_pct = _safe_get(t3_after, "bubble_summary", "pure_compute_pct")

    # 计算 delta
    time_improvement_pct = 0.0
    if before_max_us > 0:
        time_improvement_pct = (before_max_us - after_max_us) / before_max_us * 100

    d_class_reduction = before_d_pct - after_d_pct
    c_class_reduction = before_c_pct - after_c_pct
    compute_pct_change = after_compute_pct - before_compute_pct

    # T7 模式变化
    before_pattern = t7_before.get("pattern_type", "unknown")
    after_pattern = t7_after.get("pattern_type", "unknown")
    pattern_improved = (
        before_pattern == "periodic" and after_pattern != "periodic"
    )

    # T8 重叠度变化
    before_overlap = _safe_get(t8_before, "overlap_status", default="unknown")
    after_overlap = _safe_get(t8_after, "overlap_status", default="unknown")
    _overlap_rank = {"no_overlap": 0, "partial_overlap": 1, "good_overlap": 2}
    overlap_improved = (
        _overlap_rank.get(after_overlap, 0) > _overlap_rank.get(before_overlap, 0)
    )

    before_vec_mte2_overlap = _safe_get(
        t8_before, "pairwise_overlap", "vec_mte2", "overlap_pct")
    after_vec_mte2_overlap = _safe_get(
        t8_after, "pairwise_overlap", "vec_mte2", "overlap_pct")

    # 判定 — 增加重叠度改善的考量
    if time_improvement_pct > 10:
        verdict = "significant_improvement"
    elif time_improvement_pct > 2:
        verdict = "moderate_improvement"
    elif time_improvement_pct > -2:
        if overlap_improved or pattern_improved:
            verdict = "structural_improvement"
        else:
            verdict = "no_significant_change"
    elif time_improvement_pct > -10:
        verdict = "moderate_regression"
    else:
        verdict = "significant_regression"

    return {
        "tool": "T6",
        "before": {
            "simulator_dir": before_dir,
            "slowest_core": slowest_before,
            "total_us": before_max_us,
            "imbalance_ratio": before_imbalance,
            "d_class_pct": before_d_pct,
            "c_class_pct": before_c_pct,
            "pure_compute_pct": before_compute_pct,
            "primary_bottleneck": t2_before.get("primary_bottleneck"),
            "pattern_type": before_pattern,
            "overlap_status": before_overlap,
        },
        "after": {
            "simulator_dir": after_dir,
            "slowest_core": slowest_after,
            "total_us": after_max_us,
            "imbalance_ratio": after_imbalance,
            "d_class_pct": after_d_pct,
            "c_class_pct": after_c_pct,
            "pure_compute_pct": after_compute_pct,
            "primary_bottleneck": t2_after.get("primary_bottleneck"),
            "pattern_type": after_pattern,
            "overlap_status": after_overlap,
        },
        "deltas": {
            "time_improvement_pct": round(time_improvement_pct, 1),
            "imbalance_change": round(after_imbalance - before_imbalance, 3),
            "d_class_reduction_pct": round(d_class_reduction, 1),
            "c_class_reduction_pct": round(c_class_reduction, 1),
            "compute_pct_change": round(compute_pct_change, 1),
            "pattern_change": {
                "before": before_pattern,
                "after": after_pattern,
                "improved": pattern_improved,
            },
            "overlap_change": {
                "before": before_overlap,
                "after": after_overlap,
                "improved": overlap_improved,
                "vec_mte2_overlap_before": before_vec_mte2_overlap,
                "vec_mte2_overlap_after": after_vec_mte2_overlap,
            },
        },
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="T6: Before/after profiling comparison")
    parser.add_argument("--before-dir", required=True, help="Before simulator directory")
    parser.add_argument("--after-dir", required=True, help="After simulator directory")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    result = diff_profiling(args.before_dir, args.after_dir)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T6 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
