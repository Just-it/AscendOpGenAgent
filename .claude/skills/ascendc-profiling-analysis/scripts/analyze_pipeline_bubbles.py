#!/usr/bin/env python3
"""
T2: analyze_pipeline_bubbles.py — 跨 pipeline D 类空泡检测

提取 VEC pipeline 的 WAIT_FLAG 事件，计算等待时长，
识别被等待的 pipeline，按时长排序输出 Top-N。

用法:
    python3 analyze_pipeline_bubbles.py --simulator-dir <path> --core-id <core> [--top-n 5]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, compute_core_duration,
    get_pipeline_events, find_concurrent_events, ps_to_us, ps_to_ns,
    PIPELINE_NAME_MAP, is_cubecore,
    PID_VECTOR, PID_CUBE,
)
from bubble_classifier import (
    classify_gap, build_concurrent_state, BubbleType, BubbleSubType,
)


def analyze_pipeline_bubbles(
    simulator_dir: str,
    core_id: str,
    top_n: int = 5,
) -> dict:
    """
    分析指定核的 D 类空泡 (WAIT_FLAG 事件)。

    Args:
        simulator_dir: simulator 输出目录
        core_id: 核标识 (如 "core0.veccore0")
        top_n: 返回 Top-N 最大空泡

    Returns:
        诊断结果 dict
    """
    core_paths = get_all_core_paths(simulator_dir)
    if core_id not in core_paths:
        return {
            "tool": "T2",
            "error": f"Core {core_id} not found. Available: {sorted(core_paths.keys())[:5]}",
        }

    pipelines = load_core_trace(core_paths[core_id])
    total_dur = compute_core_duration(pipelines)

    # 自动检测核类型
    is_cube = is_cubecore(core_id)
    core_type = "cubecore" if is_cube else "veccore"
    # 主 pipeline: veccore → VECTOR (PID 30), cubecore → CUBE (PID 50)
    main_pipeline_name = "CUBE" if is_cube else "VECTOR"
    main_pid = PID_CUBE if is_cube else PID_VECTOR

    # 获取主 pipeline 事件
    main_events = get_pipeline_events(pipelines, main_pipeline_name)
    if not main_events:
        return {
            "tool": "T2",
            "core_id": core_id,
            "core_type": core_type,
            "error": f"No {main_pipeline_name} events found",
        }

    # 提取 WAIT_FLAG 事件
    wait_events = [e for e in main_events if e.is_wait_flag()]

    # 同时检测主 pipeline 事件间的大间隙 (>500ps，即使没有显式 WAIT_FLAG)
    large_gaps = []
    for i in range(1, len(main_events)):
        gap = main_events[i].ts - main_events[i - 1].end_ts
        if gap > 500:  # >500ps threshold for D-class
            large_gaps.append({
                "gap_ps": gap,
                "before_event": main_events[i - 1],
                "after_event": main_events[i],
                "is_wait_flag": main_events[i].is_wait_flag(),
            })

    # 构建 D 类空泡列表
    d_bubbles = []

    # 从 WAIT_FLAG 事件
    for evt in wait_events:
        waited = evt.waited_pipeline()
        # 构建并发状态用于子类型分类
        concurrent = find_concurrent_events(
            pipelines, evt.ts, evt.end_ts, exclude_pid=main_pid
        )
        cs = build_concurrent_state(
            concurrent, pipelines, evt.ts, evt.end_ts,
            core_type=core_type,
        )
        classification = classify_gap(
            gap_ps=evt.dur,
            op_before=None,
            op_after=None,
            concurrent_state=cs,
            has_wait_flag=True,
            wait_flag_event=evt,
            core_type=core_type,
        )
        d_bubbles.append({
            "dur_ps": evt.dur,
            "dur_us": round(ps_to_us(evt.dur), 6),
            "dur_ns": round(ps_to_ns(evt.dur), 3),
            "waited_pipeline": waited or "unknown",
            "pc_addr": evt.pc_addr or "",
            "ts_us": round(ps_to_us(evt.ts), 6),
            "source": "WAIT_FLAG",
            "detail": evt.detail or "",
            "category": classification.category.value,
            "sub_type": classification.sub_type.value,
        })

    # 从大间隙 (非 WAIT_FLAG)
    for gap_info in large_gaps:
        if not gap_info["is_wait_flag"]:
            before_evt = gap_info["before_event"]
            after_evt = gap_info["after_event"]
            gap_ps = gap_info["gap_ps"]
            concurrent = find_concurrent_events(
                pipelines, before_evt.end_ts, after_evt.ts, exclude_pid=main_pid
            )
            cs = build_concurrent_state(
                concurrent, pipelines, before_evt.end_ts, after_evt.ts,
                core_type=core_type,
            )
            classification = classify_gap(
                gap_ps=gap_ps,
                op_before=before_evt,
                op_after=after_evt,
                concurrent_state=cs,
                core_type=core_type,
            )
            d_bubbles.append({
                "dur_ps": gap_ps,
                "dur_us": round(ps_to_us(gap_ps), 6),
                "dur_ns": round(ps_to_ns(gap_ps), 3),
                "waited_pipeline": classification.waited_pipeline or "implicit",
                "pc_addr": after_evt.pc_addr or "",
                "ts_us": round(ps_to_us(before_evt.end_ts), 6),
                "source": "large_gap",
                "detail": f"{before_evt.name} -> {after_evt.name}",
                "category": classification.category.value,
                "sub_type": classification.sub_type.value,
            })

    # 按时长降序排序
    d_bubbles.sort(key=lambda x: -x["dur_ps"])

    # 统计
    total_d_ps = sum(b["dur_ps"] for b in d_bubbles)
    d_class_pct = (total_d_ps / total_dur * 100) if total_dur > 0 else 0.0

    # 按被等待 pipeline 分组统计
    pipeline_breakdown = {}
    for b in d_bubbles:
        wp = b["waited_pipeline"]
        if wp not in pipeline_breakdown:
            pipeline_breakdown[wp] = {"count": 0, "total_ps": 0}
        pipeline_breakdown[wp]["count"] += 1
        pipeline_breakdown[wp]["total_ps"] += b["dur_ps"]

    # 按子类型分组统计
    sub_type_breakdown = {}
    for b in d_bubbles:
        st = b.get("sub_type", "unknown")
        if st not in sub_type_breakdown:
            sub_type_breakdown[st] = {"count": 0, "total_ps": 0}
        sub_type_breakdown[st]["count"] += 1
        sub_type_breakdown[st]["total_ps"] += b["dur_ps"]

    # 主瓶颈
    primary_bottleneck = None
    if pipeline_breakdown:
        primary_bottleneck = max(pipeline_breakdown, key=lambda k: pipeline_breakdown[k]["total_ps"])

    # 添加排名
    ranked = []
    for i, b in enumerate(d_bubbles[:top_n]):
        ranked.append({"rank": i + 1, **b})

    return {
        "tool": "T2",
        "core_id": core_id,
        "core_type": core_type,
        "main_pipeline": main_pipeline_name,
        "total_duration_us": round(ps_to_us(total_dur), 3),
        "d_class_bubbles": ranked,
        "total_d_class_count": len(d_bubbles),
        "total_d_class_us": round(ps_to_us(total_d_ps), 6),
        "d_class_pct": round(d_class_pct, 1),
        "primary_bottleneck": primary_bottleneck,
        "pipeline_breakdown": {
            k: {
                "count": v["count"],
                "total_us": round(ps_to_us(v["total_ps"]), 6),
            }
            for k, v in pipeline_breakdown.items()
        },
        "sub_type_breakdown": {
            k: {
                "count": v["count"],
                "total_us": round(ps_to_us(v["total_ps"]), 6),
            }
            for k, v in sub_type_breakdown.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="T2: Pipeline bubble (D-class) analysis")
    parser.add_argument("--simulator-dir", required=True, help="Path to simulator output directory")
    parser.add_argument("--core-id", required=True, help="Core ID (e.g., core0.veccore0)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top bubbles to show")
    parser.add_argument("--output", help="Output JSON file path (default: stdout)")
    args = parser.parse_args()

    result = analyze_pipeline_bubbles(args.simulator_dir, args.core_id, args.top_n)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T2 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
