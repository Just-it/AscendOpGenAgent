#!/usr/bin/env python3
"""
T8: overlap_analyzer.py — 流水线重叠度分析

量化 CopyIn(MTE2) / Compute(VEC) / CopyOut(MTE3) 三阶段的时间重叠程度，
评估双缓冲效果。

用法:
    python3 overlap_analyzer.py --simulator-dir <path> --core-id <core> [--window-ps 1000]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, compute_core_duration,
    get_pipeline_events, compute_pipeline_coverage, compute_pipeline_utilization,
    ps_to_us, ps_to_ns, is_cubecore,
    PID_VECTOR, PID_MTE2, PID_MTE3,
    PID_CUBE, PID_MTE1, PID_FIXPIPE,
)


def _compute_pairwise_overlap(
    pipelines: dict,
    pid_a: int,
    pid_b: int,
    total_dur: float,
    window_ps: float = 1000,
) -> dict:
    """
    计算两个 pipeline 的时间重叠率。

    使用滑动窗口方法: 将时间线切片，在每个窗口内检测两者是否同时活跃。
    """
    if total_dur <= 0:
        return {"overlap_pct": 0.0, "verdict": "none"}

    events_a = pipelines.get(pid_a, [])
    events_b = pipelines.get(pid_b, [])

    if not events_a or not events_b:
        return {"overlap_pct": 0.0, "verdict": "none"}

    # 找全局时间范围
    all_events = list(events_a) + list(events_b)
    global_start = min(e.ts for e in all_events)
    global_end = max(e.end_ts for e in all_events)

    if global_end <= global_start:
        return {"overlap_pct": 0.0, "verdict": "none"}

    # 计算精确重叠: 遍历 A 的每个事件，找与 B 重叠的部分
    overlap_total = 0.0
    b_idx = 0
    for ea in events_a:
        # 推进 b_idx 到可能重叠的位置
        while b_idx < len(events_b) and events_b[b_idx].end_ts <= ea.ts:
            b_idx += 1
        # 检查所有可能重叠的 B 事件
        j = b_idx
        while j < len(events_b) and events_b[j].ts < ea.end_ts:
            overlap_start = max(ea.ts, events_b[j].ts)
            overlap_end = min(ea.end_ts, events_b[j].end_ts)
            if overlap_end > overlap_start:
                overlap_total += overlap_end - overlap_start
            j += 1

    # 重叠率 = 重叠时间 / min(A总时间, B总时间)
    dur_a = sum(e.dur for e in events_a)
    dur_b = sum(e.dur for e in events_b)
    min_dur = min(dur_a, dur_b)

    if min_dur <= 0:
        overlap_pct = 0.0
    else:
        overlap_pct = overlap_total / min_dur * 100

    if overlap_pct < 5:
        verdict = "none"
    elif overlap_pct < 30:
        verdict = "poor"
    elif overlap_pct < 60:
        verdict = "moderate"
    else:
        verdict = "good"

    return {
        "overlap_pct": round(overlap_pct, 1),
        "overlap_ps": round(overlap_total, 1),
        "verdict": verdict,
    }


def _detect_double_buffer(vec_events: list, mte2_events: list) -> dict:
    """
    检测双缓冲是否生效。

    双缓冲的特征: VEC 和 MTE2 有显著的时间重叠，
    即在 VEC 计算当前 tile 时，MTE2 同时在搬运下一个 tile。
    """
    if not vec_events or not mte2_events:
        return {
            "detected": False,
            "reason": "Missing VEC or MTE2 events",
        }

    # 统计 VEC 计算期间 MTE2 活跃的比例
    vec_compute = [e for e in vec_events if not e.is_wait_flag()]
    if not vec_compute:
        return {"detected": False, "reason": "No VEC compute events"}

    overlap_count = 0
    for ve in vec_compute:
        for me in mte2_events:
            if me.ts >= ve.end_ts:
                break
            if me.end_ts > ve.ts:
                overlap_count += 1
                break

    overlap_ratio = overlap_count / len(vec_compute) if vec_compute else 0

    detected = overlap_ratio > 0.3
    if detected:
        reason = f"VEC 计算期间 {overlap_ratio:.0%} 的指令与 MTE2 搬运重叠，双缓冲生效"
    else:
        reason = f"VEC 和 MTE2 几乎无时间重叠 ({overlap_ratio:.0%})，未启用双缓冲或双缓冲无效"

    return {
        "detected": detected,
        "overlap_ratio": round(overlap_ratio, 3),
        "reason": reason,
    }


def analyze_overlap(
    simulator_dir: str,
    core_id: str,
    window_ps: float = 1000,
) -> dict:
    """
    分析流水线重叠度。

    Args:
        simulator_dir: simulator 输出目录
        core_id: 核标识
        window_ps: 分析窗口大小 (ps)

    Returns:
        重叠度分析结果 dict
    """
    core_paths = get_all_core_paths(simulator_dir)
    if core_id not in core_paths:
        return {"tool": "T8", "error": f"Core {core_id} not found"}

    pipelines = load_core_trace(core_paths[core_id])
    total_dur = compute_core_duration(pipelines)

    if total_dur <= 0:
        return {"tool": "T8", "core_id": core_id, "error": "No events found"}

    # 自动检测核类型
    is_cube = is_cubecore(core_id)
    core_type = "cubecore" if is_cube else "veccore"

    if is_cube:
        # Cubecore: CUBE/MTE1/MTE2/FIXPIPE 重叠分析
        cube_util = compute_pipeline_utilization(pipelines, PID_CUBE)
        mte1_util = compute_pipeline_utilization(pipelines, PID_MTE1)
        mte2_util = compute_pipeline_utilization(pipelines, PID_MTE2)
        fixpipe_util = compute_pipeline_utilization(pipelines, PID_FIXPIPE)

        cube_mte1 = _compute_pairwise_overlap(pipelines, PID_CUBE, PID_MTE1, total_dur, window_ps)
        cube_mte2 = _compute_pairwise_overlap(pipelines, PID_CUBE, PID_MTE2, total_dur, window_ps)
        cube_fixpipe = _compute_pairwise_overlap(pipelines, PID_CUBE, PID_FIXPIPE, total_dur, window_ps)
        mte1_mte2 = _compute_pairwise_overlap(pipelines, PID_MTE1, PID_MTE2, total_dur, window_ps)

        # 综合判定
        max_overlap = max(cube_mte1["overlap_pct"], cube_mte2["overlap_pct"])
        if max_overlap < 5:
            overlap_status = "no_overlap"
        elif max_overlap < 30:
            overlap_status = "partial_overlap"
        else:
            overlap_status = "good_overlap"

        recommendations = []
        if overlap_status == "no_overlap":
            recommendations.append(
                f"CUBE-MTE1 重叠率 {cube_mte1['overlap_pct']}% → 启用双缓冲 (P1) 可改善 L1→L0 搬运重叠"
            )
        elif overlap_status == "partial_overlap":
            recommendations.append(
                f"CUBE-MTE1 重叠率 {cube_mte1['overlap_pct']}% → 调整 tile 大小可进一步改善"
            )
        if fixpipe_util > 0.3:
            recommendations.append(
                f"FIXPIPE 利用率 {fixpipe_util*100:.1f}% → L0C→UB 转换开销较高，结构性限制"
            )
        if mte1_mte2["overlap_pct"] > 30:
            recommendations.append(
                f"MTE1-MTE2 重叠率 {mte1_mte2['overlap_pct']}% → L1→L0 与 GM→L1 并行搬运"
            )

        return {
            "tool": "T8",
            "core_id": core_id,
            "core_type": core_type,
            "overlap_status": overlap_status,
            "global_utilization": {
                "cube_pct": round(cube_util * 100, 1),
                "mte1_pct": round(mte1_util * 100, 1),
                "mte2_pct": round(mte2_util * 100, 1),
                "fixpipe_pct": round(fixpipe_util * 100, 1),
            },
            "pairwise_overlap": {
                "cube_mte1": cube_mte1,
                "cube_mte2": cube_mte2,
                "cube_fixpipe": cube_fixpipe,
                "mte1_mte2": mte1_mte2,
            },
            "recommendations": recommendations,
        }

    # Veccore: 原有 VEC/MTE2/MTE3 分析逻辑
    vec_util = compute_pipeline_utilization(pipelines, PID_VECTOR)
    mte2_util = compute_pipeline_utilization(pipelines, PID_MTE2)
    mte3_util = compute_pipeline_utilization(pipelines, PID_MTE3)

    # 两两重叠
    vec_mte2 = _compute_pairwise_overlap(pipelines, PID_VECTOR, PID_MTE2, total_dur, window_ps)
    vec_mte3 = _compute_pairwise_overlap(pipelines, PID_VECTOR, PID_MTE3, total_dur, window_ps)
    mte2_mte3 = _compute_pairwise_overlap(pipelines, PID_MTE2, PID_MTE3, total_dur, window_ps)

    # 双缓冲检测
    vec_events = get_pipeline_events(pipelines, "VECTOR")
    mte2_events = get_pipeline_events(pipelines, "MTE2")
    double_buf = _detect_double_buffer(vec_events, mte2_events)

    # 综合判定
    max_overlap = max(vec_mte2["overlap_pct"], vec_mte3["overlap_pct"])
    if max_overlap < 5:
        overlap_status = "no_overlap"
    elif max_overlap < 30:
        overlap_status = "partial_overlap"
    else:
        overlap_status = "good_overlap"

    # 建议
    recommendations = []
    if overlap_status == "no_overlap":
        recommendations.append(
            f"VEC-MTE2 重叠率仅 {vec_mte2['overlap_pct']}% → 启用双缓冲 (P1) 可显著提升"
        )
    elif overlap_status == "partial_overlap":
        recommendations.append(
            f"VEC-MTE2 重叠率 {vec_mte2['overlap_pct']}% → 调整 tile 大小或 UB 分区可改善"
        )

    if mte2_util > 0 and mte2_util < 0.3:
        recommendations.append(
            f"MTE2 利用率 {mte2_util*100:.1f}% → 增大 tile 可提高搬运效率"
        )

    if mte2_mte3["overlap_pct"] > 30:
        recommendations.append(
            f"MTE2-MTE3 重叠率 {mte2_mte3['overlap_pct']}% → 存在总线竞争，考虑错开搬运时序"
        )

    return {
        "tool": "T8",
        "core_id": core_id,
        "overlap_status": overlap_status,
        "global_utilization": {
            "vec_pct": round(vec_util * 100, 1),
            "mte2_pct": round(mte2_util * 100, 1),
            "mte3_pct": round(mte3_util * 100, 1),
        },
        "pairwise_overlap": {
            "vec_mte2": vec_mte2,
            "vec_mte3": vec_mte3,
            "mte2_mte3": mte2_mte3,
        },
        "double_buffer_effectiveness": double_buf,
        "recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser(description="T8: Pipeline overlap analysis")
    parser.add_argument("--simulator-dir", required=True)
    parser.add_argument("--core-id", required=True)
    parser.add_argument("--window-ps", type=float, default=1000,
                        help="Analysis window size in ps (default: 1000)")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    result = analyze_overlap(args.simulator_dir, args.core_id, args.window_ps)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T8 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
