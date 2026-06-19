#!/usr/bin/env python3
"""
T1: analyze_cross_core.py — 跨核负载均衡分析

加载 simulator 目录下所有核的 trace.json，计算每核总执行时间，
输出 imbalance_ratio、CV、最慢/最快核等指标。

用法:
    python3 analyze_cross_core.py --simulator-dir <path>
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, compute_core_duration,
    compute_pipeline_utilization, ps_to_us, is_cubecore, is_veccore,
    PID_VECTOR, PID_MTE2, PID_MTE3,
    PID_CUBE, PID_MTE1, PID_FIXPIPE,
    PID_SCALAR, PID_SCALARLDST,
)


def analyze_cross_core_balance(simulator_dir: str) -> dict:
    """
    分析跨核负载均衡。支持 veccore 和 cubecore 两种核类型。

    Returns:
        诊断结果 dict (可直接 JSON 序列化)
    """
    core_paths = get_all_core_paths(simulator_dir, core_type="all")
    if not core_paths:
        return {
            "tool": "T1",
            "error": f"No core trace files found in {simulator_dir}",
        }

    durations: dict[str, float] = {}
    per_core_utilization: dict[str, dict] = {}
    for core_id, trace_path in core_paths.items():
        pipelines = load_core_trace(trace_path)
        dur = compute_core_duration(pipelines)
        durations[core_id] = dur
        if dur > 0:
            if is_cubecore(core_id):
                per_core_utilization[core_id] = {
                    "core_type": "cubecore",
                    "cube_pct": round(compute_pipeline_utilization(pipelines, PID_CUBE) * 100, 1),
                    "mte1_pct": round(compute_pipeline_utilization(pipelines, PID_MTE1) * 100, 1),
                    "mte2_pct": round(compute_pipeline_utilization(pipelines, PID_MTE2) * 100, 1),
                    "fixpipe_pct": round(compute_pipeline_utilization(pipelines, PID_FIXPIPE) * 100, 1),
                    "scalar_pct": round(compute_pipeline_utilization(pipelines, PID_SCALAR) * 100, 1),
                    "scalarldst_pct": round(compute_pipeline_utilization(pipelines, PID_SCALARLDST) * 100, 1),
                }
            else:
                per_core_utilization[core_id] = {
                    "core_type": "veccore",
                    "vec_pct": round(compute_pipeline_utilization(pipelines, PID_VECTOR) * 100, 1),
                    "mte2_pct": round(compute_pipeline_utilization(pipelines, PID_MTE2) * 100, 1),
                    "mte3_pct": round(compute_pipeline_utilization(pipelines, PID_MTE3) * 100, 1),
                    "scalar_pct": round(compute_pipeline_utilization(pipelines, PID_SCALAR) * 100, 1),
                    "scalarldst_pct": round(compute_pipeline_utilization(pipelines, PID_SCALARLDST) * 100, 1),
                }

    # 过滤掉零时长核 (未分配工作)
    active_cores = {k: v for k, v in durations.items() if v > 0}
    if not active_cores:
        return {
            "tool": "T1",
            "num_cores": len(durations),
            "active_cores": 0,
            "diagnosis": "no_active_cores",
        }

    # 分离 veccore 和 cubecore
    vec_cores = {k: v for k, v in active_cores.items() if is_veccore(k)}
    cube_cores = {k: v for k, v in active_cores.items() if is_cubecore(k)}

    def _compute_balance_stats(cores: dict[str, float]) -> dict:
        if not cores:
            return {"count": 0}
        vals = list(cores.values())
        n = len(vals)
        min_dur = min(vals)
        max_dur = max(vals)
        mean_dur = sum(vals) / n
        if mean_dur > 0:
            variance = sum((v - mean_dur) ** 2 for v in vals) / n
            std_dur = math.sqrt(variance)
            cv = std_dur / mean_dur
        else:
            std_dur = 0.0
            cv = 0.0
        imbalance_ratio = max_dur / min_dur if min_dur > 0 else float("inf")
        slowest = max(cores, key=cores.get)
        fastest = min(cores, key=cores.get)
        sorted_cores = sorted(cores.items(), key=lambda x: -x[1])
        return {
            "count": n,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "is_imbalanced": imbalance_ratio > 1.3 or cv > 0.15,
            "slowest_core": slowest,
            "fastest_core": fastest,
            "min_us": round(ps_to_us(min_dur), 3),
            "max_us": round(ps_to_us(max_dur), 3),
            "mean_us": round(ps_to_us(mean_dur), 3),
            "std_us": round(ps_to_us(std_dur), 3),
            "cv": round(cv, 3),
            "top5_slowest": [
                {"core": c, "duration_us": round(ps_to_us(d), 3)}
                for c, d in sorted_cores[:5]
            ],
        }

    # 总体统计 (向后兼容)
    vals = list(active_cores.values())
    n = len(vals)
    min_dur = min(vals)
    max_dur = max(vals)
    mean_dur = sum(vals) / n

    # 变异系数 (CV)
    if mean_dur > 0:
        variance = sum((v - mean_dur) ** 2 for v in vals) / n
        std_dur = math.sqrt(variance)
        cv = std_dur / mean_dur
    else:
        std_dur = 0.0
        cv = 0.0

    imbalance_ratio = max_dur / min_dur if min_dur > 0 else float("inf")

    slowest = max(active_cores, key=active_cores.get)
    fastest = min(active_cores, key=active_cores.get)

    # 判定: ratio > 1.3 或 CV > 0.15 视为不均衡
    is_imbalanced = imbalance_ratio > 1.3 or cv > 0.15

    # 按时长降序排列所有核
    sorted_cores = sorted(active_cores.items(), key=lambda x: -x[1])

    return {
        "tool": "T1",
        "num_cores": len(durations),
        "active_cores": n,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "is_imbalanced": is_imbalanced,
        "slowest_core": slowest,
        "fastest_core": fastest,
        "stats": {
            "min_us": round(ps_to_us(min_dur), 3),
            "max_us": round(ps_to_us(max_dur), 3),
            "mean_us": round(ps_to_us(mean_dur), 3),
            "std_us": round(ps_to_us(std_dur), 3),
            "cv": round(cv, 3),
        },
        "diagnosis": "tiling_imbalance" if is_imbalanced else "balanced",
        "top5_slowest": [
            {"core": c, "duration_us": round(ps_to_us(d), 3)}
            for c, d in sorted_cores[:5]
        ],
        "top5_fastest": [
            {"core": c, "duration_us": round(ps_to_us(d), 3)}
            for c, d in sorted_cores[-5:]
        ],
        "veccore_balance": _compute_balance_stats(vec_cores) if vec_cores else None,
        "cubecore_balance": _compute_balance_stats(cube_cores) if cube_cores else None,
        "per_core_pipeline_utilization": per_core_utilization,
    }


def main():
    parser = argparse.ArgumentParser(description="T1: Cross-core load balance analysis")
    parser.add_argument("--simulator-dir", required=True, help="Path to simulator output directory")
    parser.add_argument("--output", help="Output JSON file path (default: stdout)")
    args = parser.parse_args()

    result = analyze_cross_core_balance(args.simulator_dir)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T1 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
