#!/usr/bin/env python3
"""
T4: concurrent_pipeline_view.py — 时间窗口并发状态快照

在给定时间窗口内，对每个 pipeline 查找覆盖事件，输出并发状态和因果链。

用法:
    python3 concurrent_pipeline_view.py --simulator-dir <path> --core-id <id> \
        --start-us 10.46 --end-us 12.28
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, find_concurrent_events,
    compute_pipeline_coverage,
    PIPELINE_PID_MAP, ps_to_us, ps_to_ns,
    PID_CACHEMISS, PID_FLOWCTRL, is_cubecore,
)


def analyze_concurrent_view(
    simulator_dir: str,
    core_id: str,
    start_us: float,
    end_us: float,
) -> dict:
    """
    分析指定时间窗口内各 pipeline 的并发状态。

    Args:
        simulator_dir: simulator 输出目录
        core_id: 核标识
        start_us: 窗口起始时间 (us)
        end_us: 窗口结束时间 (us)

    Returns:
        并发状态诊断 dict
    """
    core_paths = get_all_core_paths(simulator_dir)
    if core_id not in core_paths:
        return {"tool": "T4", "error": f"Core {core_id} not found"}

    pipelines = load_core_trace(core_paths[core_id])

    # 自动检测核类型
    is_cube = is_cubecore(core_id)
    core_type = "cubecore" if is_cube else "veccore"

    # us → ps
    start_ps = start_us * 1e6
    end_ps = end_us * 1e6

    # 查找所有 pipeline 在窗口内的事件
    all_concurrent = find_concurrent_events(pipelines, start_ps, end_ps)

    pipeline_states = {}
    busy_pipelines = []
    idle_pipelines = []

    for pid, name in sorted(PIPELINE_PID_MAP.items()):
        if pid in (100,):  # 跳过 ALL (保留 CACHEMISS 和 FLOWCTRL)
            continue

        events = all_concurrent.get(pid, [])
        # 计算覆盖率
        coverage = compute_pipeline_coverage(pipelines, pid, start_ps, end_ps)

        if events:
            busy_pipelines.append(name)
            event_summaries = []
            for e in events[:5]:  # 最多显示 5 个事件
                overlap_start = max(e.ts, start_ps)
                overlap_end = min(e.end_ts, end_ps)
                overlap_dur = overlap_end - overlap_start
                event_summaries.append({
                    "name": e.name,
                    "dur_ns": round(ps_to_ns(e.dur), 3),
                    "overlap_ns": round(ps_to_ns(overlap_dur), 3),
                    "pc_addr": e.pc_addr or "",
                    "detail": e.detail or "",
                })
            pipeline_states[name] = {
                "state": "busy",
                "coverage_pct": round(coverage * 100, 1),
                "event_count": len(events),
                "events": event_summaries,
            }
        else:
            idle_pipelines.append(name)
            pipeline_states[name] = {
                "state": "idle",
                "coverage_pct": round(coverage * 100, 1),
            }

    # 因果链推理
    causal_chain = _infer_causal_chain(pipeline_states, busy_pipelines, idle_pipelines, core_type)

    return {
        "tool": "T4",
        "core_id": core_id,
        "core_type": core_type,
        "window": {
            "start_us": start_us,
            "end_us": end_us,
            "duration_ns": round((end_us - start_us) * 1000, 3),
        },
        "pipelines": pipeline_states,
        "busy_pipelines": busy_pipelines,
        "idle_pipelines": idle_pipelines,
        "causal_chain": causal_chain,
    }


def _infer_causal_chain(
    pipeline_states: dict,
    busy_pipelines: list[str],
    idle_pipelines: list[str],
    core_type: str = "veccore",
) -> str:
    """推理因果链描述"""
    parts = []

    # CACHEMISS 检测
    cachemiss_detected = "CACHEMISS" in busy_pipelines

    # 根据核类型选择主 pipeline
    main_pipe = "CUBE" if core_type == "cubecore" else "VECTOR"
    main_label = "CUBE" if core_type == "cubecore" else "VEC"

    if main_pipe in idle_pipelines:
        parts.append(f"{main_label} idle")

        if cachemiss_detected:
            parts.append("icache miss detected (CACHEMISS active)")

        if "FLOWCTRL" in busy_pipelines:
            parts.append("FLOWCTRL active (flow control overhead)")

        # cubecore-specific pipelines
        if core_type == "cubecore":
            if "MTE1" in busy_pipelines:
                mte1_events = pipeline_states["MTE1"].get("events", [])
                if mte1_events:
                    top_op = mte1_events[0]["name"]
                    dur = mte1_events[0]["dur_ns"]
                    parts.append(f"MTE1 busy ({top_op}, {dur}ns) [L1→L0 load]")

            if "FIXPIPE" in busy_pipelines:
                fixpipe_events = pipeline_states["FIXPIPE"].get("events", [])
                if fixpipe_events:
                    top_op = fixpipe_events[0]["name"]
                    dur = fixpipe_events[0]["dur_ns"]
                    parts.append(f"FIXPIPE busy ({top_op}, {dur}ns) [L0C→UB conversion]")

        if "MTE2" in busy_pipelines:
            mte2_events = pipeline_states["MTE2"].get("events", [])
            if mte2_events:
                top_op = mte2_events[0]["name"]
                dur = mte2_events[0]["dur_ns"]
                label = "GM→L1 load" if core_type == "cubecore" else "data load"
                parts.append(f"MTE2 busy ({top_op}, {dur}ns) [{label}]")

        if core_type != "cubecore" and "MTE3" in busy_pipelines:
            mte3_events = pipeline_states["MTE3"].get("events", [])
            if mte3_events:
                top_op = mte3_events[0]["name"]
                dur = mte3_events[0]["dur_ns"]
                parts.append(f"MTE3 busy ({top_op}, {dur}ns)")

        if "SCALARLDST" in busy_pipelines:
            sldst_events = pipeline_states["SCALARLDST"].get("events", [])
            if sldst_events:
                top_op = sldst_events[0]["name"]
                parts.append(f"SCALARLDST busy ({top_op})")

        if "SCALAR" in busy_pipelines:
            scalar_events = pipeline_states["SCALAR"].get("events", [])
            if scalar_events:
                top_op = scalar_events[0]["name"]
                parts.append(f"SCALAR busy ({top_op})")

    elif main_pipe in busy_pipelines:
        main_events = pipeline_states[main_pipe].get("events", [])
        if main_events:
            # Check if main pipe is doing WAIT_FLAG
            wait_flags = [e for e in main_events if e["name"] == "WAIT_FLAG"]
            if wait_flags:
                detail = wait_flags[0].get("detail", "")
                parts.append(f"{main_label} waiting ({detail})")
            else:
                parts.append(f"{main_label} computing ({main_events[0]['name']})")

        if cachemiss_detected:
            parts.append("icache miss detected (CACHEMISS active)")

    if not parts:
        if not busy_pipelines:
            parts.append("All pipelines idle (possible icache miss or extended drain)")
        else:
            parts.append(f"Active: {', '.join(busy_pipelines)}")

    return " → ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="T4: Concurrent pipeline view")
    parser.add_argument("--simulator-dir", required=True)
    parser.add_argument("--core-id", required=True)
    parser.add_argument("--start-us", type=float, required=True, help="Window start (us)")
    parser.add_argument("--end-us", type=float, required=True, help="Window end (us)")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    result = analyze_concurrent_view(
        args.simulator_dir, args.core_id, args.start_us, args.end_us
    )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T4 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
