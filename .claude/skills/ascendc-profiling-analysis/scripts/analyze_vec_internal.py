#!/usr/bin/env python3
"""
T3: analyze_vec_internal.py — VEC 内部 A/B/C 类空泡分析

将 VEC 事件按 D 类间隙分割成「计算块」，对块间间隙做 C 类因果归因，
块内计算 issue rate 并检测异常指令间隔。

用法:
    python3 analyze_vec_internal.py --simulator-dir <path> --core-id <core>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, compute_core_duration,
    get_pipeline_events, find_concurrent_events, ps_to_us, ps_to_ns,
    is_cubecore, PID_VECTOR, PID_CUBE,
)
from bubble_classifier import (
    classify_gap, build_concurrent_state, BubbleType, BubbleSubType,
    THRESHOLD_C_MAX_PS,
)

# D 类间隙阈值 (ps): 用于分割计算块
D_CLASS_GAP_THRESHOLD = 500.0


def _split_into_blocks(vec_events: list) -> list[list]:
    """将 VEC 事件按 D 类间隙 (>500ps) 分割成计算块"""
    if not vec_events:
        return []

    blocks = []
    current_block = [vec_events[0]]

    for i in range(1, len(vec_events)):
        gap = vec_events[i].ts - vec_events[i - 1].end_ts
        if gap > D_CLASS_GAP_THRESHOLD:
            blocks.append(current_block)
            current_block = [vec_events[i]]
        else:
            current_block.append(vec_events[i])

    if current_block:
        blocks.append(current_block)

    return blocks


def _analyze_block_internal(
    block: list,
    pipelines: dict,
    iteration_index: int = None,
    total_iterations: int = None,
    core_type: str = "veccore",
) -> dict:
    """分析单个计算块内部的间隙"""
    main_pid = PID_CUBE if core_type == "cubecore" else PID_VECTOR
    a_count = 0
    a_total_ps = 0.0
    b_count = 0
    b_total_ps = 0.0
    c_count = 0
    c_total_ps = 0.0
    c_details = []
    sub_type_counts = {}

    total_compute_ps = sum(e.dur for e in block)

    for i in range(1, len(block)):
        gap = block[i].ts - block[i - 1].end_ts
        if gap <= 0:
            continue

        # 查找间隙期间的并发事件
        concurrent = find_concurrent_events(
            pipelines, block[i - 1].end_ts, block[i].ts, exclude_pid=main_pid
        )
        concurrent_state = build_concurrent_state(
            concurrent, pipelines,
            block[i - 1].end_ts, block[i].ts,
            core_type=core_type,
        )

        classification = classify_gap(
            gap_ps=gap,
            op_before=block[i - 1],
            op_after=block[i],
            concurrent_state=concurrent_state,
            iteration_index=iteration_index,
            total_iterations=total_iterations,
            core_type=core_type,
        )

        # 统计子类型
        st = classification.sub_type.value
        sub_type_counts[st] = sub_type_counts.get(st, 0) + 1

        if classification.bubble_type == BubbleType.A:
            a_count += 1
            a_total_ps += gap
        elif classification.bubble_type == BubbleType.B:
            b_count += 1
            b_total_ps += gap
        elif classification.bubble_type == BubbleType.C:
            c_count += 1
            c_total_ps += gap
            c_details.append({
                "dur_ps": round(gap, 3),
                "reason": classification.reason,
                "concurrent_cause": classification.concurrent_cause,
                "optimization_hint": classification.optimization_hint,
                "before_op": block[i - 1].name,
                "after_op": block[i].name,
                "ts_us": round(ps_to_us(block[i - 1].end_ts), 6),
                "sub_type": st,
            })

    # Issue rate: 指令数 / 块总时间跨度
    if len(block) >= 2:
        block_span = block[-1].end_ts - block[0].ts
        issue_rate = len(block) / block_span if block_span > 0 else 0
    else:
        block_span = block[0].dur if block else 0
        issue_rate = 1.0 / block_span if block_span > 0 else 0

    return {
        "num_instructions": len(block),
        "compute_ps": round(total_compute_ps, 3),
        "block_span_ps": round(block_span, 3),
        "issue_rate": round(issue_rate, 4),
        "a_class": {"count": a_count, "total_ps": round(a_total_ps, 3)},
        "b_class": {"count": b_count, "total_ps": round(b_total_ps, 3)},
        "c_class": {"count": c_count, "total_ps": round(c_total_ps, 3)},
        "c_details": c_details[:5],  # Top 5 C-class gaps
        "sub_type_counts": sub_type_counts,
    }


def analyze_vec_internal(simulator_dir: str, core_id: str) -> dict:
    """
    分析指定核的 VEC 内部空泡。

    Returns:
        诊断结果 dict
    """
    core_paths = get_all_core_paths(simulator_dir)
    if core_id not in core_paths:
        return {
            "tool": "T3",
            "error": f"Core {core_id} not found",
        }

    pipelines = load_core_trace(core_paths[core_id])
    total_dur = compute_core_duration(pipelines)

    # 自动检测核类型
    is_cube = is_cubecore(core_id)
    core_type = "cubecore" if is_cube else "veccore"
    main_pipeline_name = "CUBE" if is_cube else "VECTOR"
    main_pid = PID_CUBE if is_cube else PID_VECTOR

    main_events = get_pipeline_events(pipelines, main_pipeline_name)
    if not main_events:
        return {"tool": "T3", "core_id": core_id, "core_type": core_type,
                "error": f"No {main_pipeline_name} events"}

    # 过滤掉 WAIT_FLAG (它们是 D 类，不属于内部分析)
    compute_events = [e for e in main_events if not e.is_wait_flag()]

    blocks = _split_into_blocks(compute_events)

    # 分析每个块
    block_results = []
    total_a = {"count": 0, "total_ps": 0.0}
    total_b = {"count": 0, "total_ps": 0.0}
    total_c = {"count": 0, "total_ps": 0.0}
    total_compute_ps = 0.0
    all_c_details = []
    all_sub_type_counts = {}

    num_blocks = len(blocks)
    for i, block in enumerate(blocks):
        br = _analyze_block_internal(
            block, pipelines,
            iteration_index=i,
            total_iterations=num_blocks,
            core_type=core_type,
        )
        block_results.append({
            "block_index": i,
            "num_instructions": br["num_instructions"],
            "span_ns": round(ps_to_ns(br["block_span_ps"]), 3),
            "issue_rate": br["issue_rate"],
        })
        total_a["count"] += br["a_class"]["count"]
        total_a["total_ps"] += br["a_class"]["total_ps"]
        total_b["count"] += br["b_class"]["count"]
        total_b["total_ps"] += br["b_class"]["total_ps"]
        total_c["count"] += br["c_class"]["count"]
        total_c["total_ps"] += br["c_class"]["total_ps"]
        total_compute_ps += br["compute_ps"]
        all_c_details.extend(br["c_details"])
        for st, cnt in br.get("sub_type_counts", {}).items():
            all_sub_type_counts[st] = all_sub_type_counts.get(st, 0) + cnt

    # 百分比
    total_bubble_ps = total_a["total_ps"] + total_b["total_ps"] + total_c["total_ps"]
    non_d_total = total_bubble_ps + total_compute_ps
    if non_d_total <= 0:
        non_d_total = 1.0  # avoid div by zero

    # 也分析块间间隙 (D 类间隙之间的 gap)
    inter_block_gaps = []
    for i in range(1, len(blocks)):
        prev_end = blocks[i - 1][-1].end_ts
        next_start = blocks[i][0].ts
        gap = next_start - prev_end
        if gap > 0:
            inter_block_gaps.append({
                "between_blocks": f"{i-1}->{i}",
                "gap_ps": round(gap, 3),
                "gap_ns": round(ps_to_ns(gap), 3),
            })

    # C 类主因分析
    c_class_primary_cause = None
    if all_c_details:
        # 按 concurrent_cause 分组
        cause_totals = {}
        for d in all_c_details:
            cause = d.get("concurrent_cause", "unknown")
            cause_totals[cause] = cause_totals.get(cause, 0) + d["dur_ps"]
        c_class_primary_cause = max(cause_totals, key=cause_totals.get)

    # 排序 C 类 details 按时长
    all_c_details.sort(key=lambda x: -x["dur_ps"])

    return {
        "tool": "T3",
        "core_id": core_id,
        "core_type": core_type,
        "main_pipeline": main_pipeline_name,
        "num_blocks": len(blocks),
        "bubble_summary": {
            "a_class": {
                "count": total_a["count"],
                "total_us": round(ps_to_us(total_a["total_ps"]), 6),
                "pct": round(total_a["total_ps"] / non_d_total * 100, 1),
            },
            "b_class": {
                "count": total_b["count"],
                "total_us": round(ps_to_us(total_b["total_ps"]), 6),
                "pct": round(total_b["total_ps"] / non_d_total * 100, 1),
            },
            "c_class": {
                "count": total_c["count"],
                "total_us": round(ps_to_us(total_c["total_ps"]), 6),
                "pct": round(total_c["total_ps"] / non_d_total * 100, 1),
            },
            "pure_compute_us": round(ps_to_us(total_compute_ps), 6),
            "pure_compute_pct": round(total_compute_ps / non_d_total * 100, 1),
        },
        "c_class_detail": {
            "primary_cause": c_class_primary_cause,
            "top_gaps": all_c_details[:5],
        },
        "sub_type_breakdown": all_sub_type_counts,
        "blocks": block_results,
        "inter_block_gaps": inter_block_gaps[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="T3: VEC internal bubble analysis")
    parser.add_argument("--simulator-dir", required=True)
    parser.add_argument("--core-id", required=True)
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    result = analyze_vec_internal(args.simulator_dir, args.core_id)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T3 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
