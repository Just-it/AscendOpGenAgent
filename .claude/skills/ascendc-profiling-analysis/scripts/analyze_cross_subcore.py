#!/usr/bin/env python3
"""
T10: analyze_cross_subcore.py — 同一物理核内 cubecore/veccore 跨子核协同分析

从子核 trace.json 中提取同一物理核的 cubecore + veccore 时间线，
分析 SET_CROSS_CORE 握手延迟、pipeline 时间重叠、瓶颈因果链。

子核 trace.json 的时间戳是全局对齐的，可直接跨文件比较。

用法:
    python3 analyze_cross_subcore.py --simulator-dir <path> --physical-core 0
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, compute_core_duration,
    ps_to_us, ps_to_ns,
    PID_SCALAR, PID_SCALARLDST, PID_VECTOR, PID_MTE2, PID_MTE3,
    PID_CUBE, PID_MTE1, PID_FIXPIPE, PID_FLOWCTRL,
)


def _time_range(pipelines):
    """计算所有 pipeline 的时间范围 (ps)"""
    min_ts = float("inf")
    max_end = 0.0
    for events in pipelines.values():
        for e in events:
            if e.ts < min_ts:
                min_ts = e.ts
            if e.end_ts > max_end:
                max_end = e.end_ts
    return (min_ts, max_end) if min_ts != float("inf") else (0.0, 0.0)


def _pipeline_busy(pipelines, pid):
    """计算指定 pipeline 的总 busy 时间 (ps)"""
    return sum(e.dur for e in pipelines.get(pid, []))


def _compute_overlap_ps(events_a, events_b):
    """
    精确计算两组事件的时间重叠 (ps)。
    events_a 和 events_b 必须按 ts 排序。
    """
    if not events_a or not events_b:
        return 0.0

    overlap_total = 0.0
    b_idx = 0
    for ea in events_a:
        while b_idx < len(events_b) and events_b[b_idx].end_ts <= ea.ts:
            b_idx += 1
        j = b_idx
        while j < len(events_b) and events_b[j].ts < ea.end_ts:
            os_ = max(ea.ts, events_b[j].ts)
            oe_ = min(ea.end_ts, events_b[j].end_ts)
            if oe_ > os_:
                overlap_total += oe_ - os_
            j += 1
    return overlap_total


def _find_sync_events(pipelines, core_type):
    """
    从 trace 中提取 SET_CROSS_CORE 同步事件。
    cubecore: FIXPIPE (PID 80) 中的 SET_CROSS_CORE
    veccore: MTE3 (PID 70) 中的 SET_CROSS_CORE
    """
    if core_type == "cubecore":
        candidates = pipelines.get(PID_FIXPIPE, []) + pipelines.get(PID_MTE3, [])
    else:
        candidates = pipelines.get(PID_MTE3, []) + pipelines.get(PID_MTE2, [])

    # 也搜索所有 pipeline
    all_sync = []
    for pid, events in pipelines.items():
        for e in events:
            if e.name == "SET_CROSS_CORE":
                all_sync.append(e)

    all_sync.sort(key=lambda e: e.ts)
    return all_sync


def _analyze_handoff_pattern(cube_syncs, vec0_syncs, vec1_syncs,
                              cube_fixp_events, vec0_events, vec1_events):
    """
    分析 cubecore → veccore 的数据交接模式。

    检测:
    1. cubecore SET_CROSS_CORE 后，veccore 多久开始消费
    2. veccore SET_CROSS_CORE (完成信号) 后，cubecore 多久开始下一轮
    3. 是否存在 cubecore 等 veccore 的情况
    """
    handoff_cube_to_vec = []
    handoff_vec_to_cube = []

    # cubecore sync → veccore 最近后续 VEC 事件 (unused but kept for reference)
    # vec_all would combine vec0 + vec1 VEC events

    # 简化: 用 sync 时间序列对齐
    # cubecore 的 SET_CROSS_CORE 标记一个 tile 的 CUBE 输出完成
    # veccore 的 SET_CROSS_CORE 标记一个 tile 的 VEC 处理完成

    # 交替模式检测
    all_syncs = []
    for e in cube_syncs:
        all_syncs.append(("cube", e.ts, e.end_ts))
    for e in vec0_syncs:
        all_syncs.append(("vec0", e.ts, e.end_ts))
    for e in vec1_syncs:
        all_syncs.append(("vec1", e.ts, e.end_ts))
    all_syncs.sort(key=lambda x: x[1])

    # 检测 cube→vec 和 vec→cube 的间隔
    for i in range(1, len(all_syncs)):
        prev_type, prev_ts, _ = all_syncs[i - 1]
        curr_type, curr_ts, _ = all_syncs[i]
        gap_ps = curr_ts - prev_ts
        if prev_type == "cube" and curr_type.startswith("vec"):
            handoff_cube_to_vec.append(gap_ps)
        elif prev_type.startswith("vec") and curr_type == "cube":
            handoff_vec_to_cube.append(gap_ps)

    return {
        "cube_to_vec_gaps": {
            "count": len(handoff_cube_to_vec),
            "mean_ps": round(sum(handoff_cube_to_vec) / len(handoff_cube_to_vec), 3) if handoff_cube_to_vec else None,
            "min_ps": round(min(handoff_cube_to_vec), 3) if handoff_cube_to_vec else None,
            "max_ps": round(max(handoff_cube_to_vec), 3) if handoff_cube_to_vec else None,
            "mean_ns": round(sum(handoff_cube_to_vec) / len(handoff_cube_to_vec) / 1000, 3) if handoff_cube_to_vec else None,
        },
        "vec_to_cube_gaps": {
            "count": len(handoff_vec_to_cube),
            "mean_ps": round(sum(handoff_vec_to_cube) / len(handoff_vec_to_cube), 3) if handoff_vec_to_cube else None,
            "min_ps": round(min(handoff_vec_to_cube), 3) if handoff_vec_to_cube else None,
            "max_ps": round(max(handoff_vec_to_cube), 3) if handoff_vec_to_cube else None,
            "mean_ns": round(sum(handoff_vec_to_cube) / len(handoff_vec_to_cube) / 1000, 3) if handoff_vec_to_cube else None,
        },
        "sync_sequence_length": len(all_syncs),
        "sync_sequence_sample": [
            {"type": t, "ts_ns": round(ts / 1000, 3)}
            for t, ts, _ in all_syncs[:20]
        ],
    }


def analyze_cross_subcore(
    simulator_dir: str,
    physical_core: int = 0,
) -> dict:
    """
    分析指定物理核上 cubecore/veccore 的跨子核协同。
    """
    core_paths = get_all_core_paths(simulator_dir)

    cube_id = f"core{physical_core}.cubecore0"
    vec0_id = f"core{physical_core}.veccore0"
    vec1_id = f"core{physical_core}.veccore1"

    missing = []
    for cid in [cube_id, vec0_id, vec1_id]:
        if cid not in core_paths:
            missing.append(cid)
    if missing:
        return {"tool": "T10", "error": f"Missing cores: {missing}",
                "available": sorted(core_paths.keys())[:10]}

    # 加载三个子核的 trace
    print(f"Loading {cube_id}...", file=sys.stderr)
    cube_pipelines = load_core_trace(core_paths[cube_id])
    print(f"Loading {vec0_id}...", file=sys.stderr)
    vec0_pipelines = load_core_trace(core_paths[vec0_id])
    print(f"Loading {vec1_id}...", file=sys.stderr)
    vec1_pipelines = load_core_trace(core_paths[vec1_id])

    # 时间范围
    cube_start, cube_end = _time_range(cube_pipelines)
    vec0_start, vec0_end = _time_range(vec0_pipelines)
    vec1_start, vec1_end = _time_range(vec1_pipelines)

    global_start = min(cube_start, vec0_start, vec1_start)
    global_end = max(cube_end, vec0_end, vec1_end)
    global_dur = global_end - global_start

    # Pipeline 利用率 (基于全局时间范围)
    def util(pipelines, pid):
        busy = _pipeline_busy(pipelines, pid)
        return round(busy / global_dur * 100, 1) if global_dur > 0 else 0.0

    # 跨子核时间重叠
    # cubecore CUBE events vs veccore VECTOR events
    cube_events = sorted(cube_pipelines.get(PID_CUBE, []), key=lambda e: e.ts)
    fixp_events = sorted(cube_pipelines.get(PID_FIXPIPE, []), key=lambda e: e.ts)
    mte1_events = sorted(cube_pipelines.get(PID_MTE1, []), key=lambda e: e.ts)
    vec0_vec_events = sorted(vec0_pipelines.get(PID_VECTOR, []), key=lambda e: e.ts)
    vec1_vec_events = sorted(vec1_pipelines.get(PID_VECTOR, []), key=lambda e: e.ts)
    vec0_mte3_events = sorted(vec0_pipelines.get(PID_MTE3, []), key=lambda e: e.ts)
    vec1_mte3_events = sorted(vec1_pipelines.get(PID_MTE3, []), key=lambda e: e.ts)

    # 跨子核重叠分析
    cube_vs_vec0 = _compute_overlap_ps(cube_events, vec0_vec_events)
    cube_vs_vec1 = _compute_overlap_ps(cube_events, vec1_vec_events)
    fixp_vs_vec0 = _compute_overlap_ps(fixp_events, vec0_vec_events)
    fixp_vs_vec1 = _compute_overlap_ps(fixp_events, vec1_vec_events)
    vec0_vs_vec1 = _compute_overlap_ps(vec0_vec_events, vec1_vec_events)

    cube_busy = _pipeline_busy(cube_pipelines, PID_CUBE)
    fixp_busy = _pipeline_busy(cube_pipelines, PID_FIXPIPE)
    vec0_busy = _pipeline_busy(vec0_pipelines, PID_VECTOR)
    vec1_busy = _pipeline_busy(vec1_pipelines, PID_VECTOR)

    def safe_pct(overlap, ref):
        return round(overlap / ref * 100, 1) if ref > 0 else 0.0

    # SET_CROSS_CORE 同步分析
    cube_syncs = _find_sync_events(cube_pipelines, "cubecore")
    vec0_syncs = _find_sync_events(vec0_pipelines, "veccore")
    vec1_syncs = _find_sync_events(vec1_pipelines, "veccore")

    # Handoff 分析
    handoff = _analyze_handoff_pattern(
        cube_syncs, vec0_syncs, vec1_syncs,
        fixp_events, list(vec0_pipelines.get(PID_VECTOR, [])),
        list(vec1_pipelines.get(PID_VECTOR, [])),
    )

    # 检测 cubecore 是否等待 veccore
    # 如果 cubecore 结束时间远早于 veccore → cubecore 不是关键路径
    # 如果 cubecore 和 veccore 结束时间接近 → 两者都可能是关键路径
    cube_dur = cube_end - cube_start
    vec0_dur = vec0_end - vec0_start
    vec1_dur = vec1_end - vec1_start
    max_vec_dur = max(vec0_dur, vec1_dur)

    if cube_dur > 0 and max_vec_dur > 0:
        cube_vec_ratio = cube_dur / max_vec_dur
    else:
        cube_vec_ratio = 0.0

    if cube_vec_ratio > 1.1:
        critical_path = "cubecore"
        critical_reason = f"cubecore 时长 ({ps_to_ns(cube_dur):.1f}ns) > veccore ({ps_to_ns(max_vec_dur):.1f}ns)，cubecore 是关键路径"
    elif cube_vec_ratio < 0.9:
        critical_path = "veccore"
        critical_reason = f"veccore 时长 ({ps_to_ns(max_vec_dur):.1f}ns) > cubecore ({ps_to_ns(cube_dur):.1f}ns)，veccore 是关键路径"
    else:
        critical_path = "balanced"
        critical_reason = f"cubecore ({ps_to_ns(cube_dur):.1f}ns) ≈ veccore ({ps_to_ns(max_vec_dur):.1f}ns)，两者均衡"

    # FIXP 指令级分析
    fixp_all = cube_pipelines.get(PID_FIXPIPE, [])
    fixp_fix_events = [e for e in fixp_all if e.name == "FIX_L0C_TO_DST"]
    fixp_mov_events = [e for e in fixp_all if e.name == "MOV_SPR_XN"]
    fixp_bar_events = [e for e in fixp_all if e.name == "BAR"]
    fixp_cross = [e for e in fixp_all if e.name == "SET_CROSS_CORE"]

    # CUBE 指令分析
    cube_all = cube_pipelines.get(PID_CUBE, [])
    mmad_events = [e for e in cube_all if e.name == "MMAD"]
    cube_bar_events = [e for e in cube_all if e.name == "BAR"]

    return {
        "tool": "T10",
        "physical_core": physical_core,
        "global_time": {
            "start_ns": round(ps_to_ns(global_start), 3),
            "end_ns": round(ps_to_ns(global_end), 3),
            "duration_ns": round(ps_to_ns(global_dur), 3),
        },
        "subcore_timing": {
            "cubecore": {
                "start_ns": round(ps_to_ns(cube_start), 3),
                "end_ns": round(ps_to_ns(cube_end), 3),
                "duration_ns": round(ps_to_ns(cube_dur), 3),
            },
            "veccore0": {
                "start_ns": round(ps_to_ns(vec0_start), 3),
                "end_ns": round(ps_to_ns(vec0_end), 3),
                "duration_ns": round(ps_to_ns(vec0_dur), 3),
            },
            "veccore1": {
                "start_ns": round(ps_to_ns(vec1_start), 3),
                "end_ns": round(ps_to_ns(vec1_end), 3),
                "duration_ns": round(ps_to_ns(vec1_dur), 3),
            },
        },
        "critical_path": {
            "path": critical_path,
            "reason": critical_reason,
            "cube_vec_ratio": round(cube_vec_ratio, 3),
        },
        "pipeline_utilization_global": {
            "cube_pct": util(cube_pipelines, PID_CUBE),
            "fixpipe_pct": util(cube_pipelines, PID_FIXPIPE),
            "mte1_pct": util(cube_pipelines, PID_MTE1),
            "cube_mte2_pct": util(cube_pipelines, PID_MTE2),
            "vec0_pct": util(vec0_pipelines, PID_VECTOR),
            "vec0_mte2_pct": util(vec0_pipelines, PID_MTE2),
            "vec0_mte3_pct": util(vec0_pipelines, PID_MTE3),
            "vec1_pct": util(vec1_pipelines, PID_VECTOR),
            "vec1_mte2_pct": util(vec1_pipelines, PID_MTE2),
            "vec1_mte3_pct": util(vec1_pipelines, PID_MTE3),
        },
        "cross_subcore_overlap": {
            "cube_vs_vec0_pct": safe_pct(cube_vs_vec0, min(cube_busy, vec0_busy)),
            "cube_vs_vec1_pct": safe_pct(cube_vs_vec1, min(cube_busy, vec1_busy)),
            "fixp_vs_vec0_pct": safe_pct(fixp_vs_vec0, min(fixp_busy, vec0_busy)),
            "fixp_vs_vec1_pct": safe_pct(fixp_vs_vec1, min(fixp_busy, vec1_busy)),
            "vec0_vs_vec1_pct": safe_pct(vec0_vs_vec1, min(vec0_busy, vec1_busy)),
            "detail": {
                "cube_busy_ns": round(ps_to_ns(cube_busy), 3),
                "fixp_busy_ns": round(ps_to_ns(fixp_busy), 3),
                "vec0_busy_ns": round(ps_to_ns(vec0_busy), 3),
                "vec1_busy_ns": round(ps_to_ns(vec1_busy), 3),
            },
        },
        "sync_events": {
            "cubecore_SET_CROSS_CORE": len(cube_syncs),
            "veccore0_SET_CROSS_CORE": len(vec0_syncs),
            "veccore1_SET_CROSS_CORE": len(vec1_syncs),
        },
        "handoff_analysis": handoff,
        "instruction_breakdown": {
            "cube": {
                "MMAD": len(mmad_events),
                "BAR": len(cube_bar_events),
                "total": len(cube_all),
                "mmad_total_ns": round(ps_to_ns(sum(e.dur for e in mmad_events)), 3),
                "mmad_mean_ps": round(sum(e.dur for e in mmad_events) / len(mmad_events), 3) if mmad_events else 0,
            },
            "fixpipe": {
                "FIX_L0C_TO_DST": len(fixp_fix_events),
                "MOV_SPR_XN": len(fixp_mov_events),
                "BAR": len(fixp_bar_events),
                "SET_CROSS_CORE": len(fixp_cross),
                "fix_total_ns": round(ps_to_ns(sum(e.dur for e in fixp_fix_events)), 3),
                "fix_mean_ps": round(sum(e.dur for e in fixp_fix_events) / len(fixp_fix_events), 3) if fixp_fix_events else 0,
                "mov_total_ns": round(ps_to_ns(sum(e.dur for e in fixp_mov_events)), 3),
                "mov_mean_ps": round(sum(e.dur for e in fixp_mov_events) / len(fixp_mov_events), 3) if fixp_mov_events else 0,
            },
            "veccore0": {
                "VEC_total": len(vec0_pipelines.get(PID_VECTOR, [])),
                "MTE2_total": len(vec0_pipelines.get(PID_MTE2, [])),
                "MTE3_total": len(vec0_pipelines.get(PID_MTE3, [])),
                "BAR_in_VEC": sum(1 for e in vec0_pipelines.get(PID_VECTOR, []) if e.name == "BAR"),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="T10: Cross-subcore coordination analysis")
    parser.add_argument("--simulator-dir", required=True)
    parser.add_argument("--physical-core", type=int, default=0, help="Physical core ID (default: 0)")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    result = analyze_cross_subcore(args.simulator_dir, args.physical_core)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T10 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
