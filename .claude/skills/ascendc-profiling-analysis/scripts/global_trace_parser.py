#!/usr/bin/env python3
"""
global_trace_parser.py — 全局 trace.json 流式解析器

解析 simulator 顶层 trace.json (11GB+)，该文件的 pid 为 core 名字符串，
tid 为 pipeline 名字符串，与子核 trace.json 的数字格式不同。

使用 ijson 流式解析避免内存爆炸，按需提取指定核/pipeline 的事件。

用法:
    from global_trace_parser import stream_global_trace, parse_cross_core_events
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Iterator, Callable


@dataclass
class GlobalTraceEvent:
    """全局 trace 事件 (pid/tid 为字符串)"""
    name: str
    ph: str
    core_id: str       # e.g. "core0.cubecore0"
    pipeline: str      # e.g. "SCALAR", "CUBE", "VECTOR", "MTE1", ...
    ts: float          # 起始时间 (ns, 全局trace的displayTimeUnit=ns)
    dur: float         # 持续时间 (ns)
    args: dict = field(default_factory=dict)
    cname: Optional[str] = None

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur

    @property
    def physical_core(self) -> int:
        """提取物理核编号: core12.cubecore0 → 12"""
        m = re.match(r"core(\d+)\.", self.core_id)
        return int(m.group(1)) if m else -1

    @property
    def subcore_type(self) -> str:
        """提取子核类型: cubecore0, veccore0, veccore1"""
        m = re.match(r"core\d+\.(.+)", self.core_id)
        return m.group(1) if m else ""

    @property
    def is_cubecore(self) -> bool:
        return "cubecore" in self.core_id

    @property
    def is_veccore(self) -> bool:
        return "veccore" in self.core_id

    @property
    def ts_ps(self) -> float:
        """转换为皮秒 (与子核 trace 时间单位一致)"""
        return self.ts * 1000  # ns → ps

    @property
    def dur_ps(self) -> float:
        return self.dur * 1000

    @property
    def end_ts_ps(self) -> float:
        return self.end_ts * 1000


def stream_global_trace(
    trace_path: str,
    core_filter: Optional[set] = None,
    pipeline_filter: Optional[set] = None,
    event_filter: Optional[Callable] = None,
    max_events: int = 0,
    chunk_size: int = 64 * 1024,
) -> Iterator[GlobalTraceEvent]:
    """
    流式解析全局 trace.json，逐个 yield 事件。

    Args:
        trace_path: 全局 trace.json 路径
        core_filter: 仅返回指定核的事件 (如 {"core0.cubecore0", "core0.veccore0"})
        pipeline_filter: 仅返回指定 pipeline 的事件 (如 {"CUBE", "FIXPIPE", "VECTOR"})
        event_filter: 自定义过滤函数 (event_dict → bool)
        max_events: 最大返回事件数 (0=无限)
        chunk_size: 读取块大小

    Yields:
        GlobalTraceEvent
    """
    count = 0
    depth = 0
    event_chars = []
    in_events = False

    with open(trace_path, "r", buffering=1024*1024) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for ch in chunk:
                if not in_events:
                    # 跳过到 traceEvents 数组开始
                    event_chars.append(ch)
                    buf = "".join(event_chars[-20:])
                    if '"traceEvents":[' in buf or "'traceEvents':[" in buf:
                        in_events = True
                        event_chars = []
                        depth = 0
                    continue

                if ch == '{':
                    depth += 1
                if depth > 0:
                    event_chars.append(ch)
                if ch == '}':
                    depth -= 1
                    if depth == 0 and event_chars:
                        event_str = "".join(event_chars)
                        event_chars = []
                        try:
                            raw = json.loads(event_str)
                        except json.JSONDecodeError:
                            continue

                        # 跳过非 duration 事件
                        if raw.get("ph") != "X":
                            continue

                        pid_str = str(raw.get("pid", ""))
                        tid_str = str(raw.get("tid", ""))

                        # 过滤
                        if core_filter and pid_str not in core_filter:
                            continue
                        if pipeline_filter and tid_str not in pipeline_filter:
                            continue
                        if event_filter and not event_filter(raw):
                            continue

                        evt = GlobalTraceEvent(
                            name=raw.get("name", ""),
                            ph="X",
                            core_id=pid_str,
                            pipeline=tid_str,
                            ts=float(raw.get("ts", 0)),
                            dur=float(raw.get("dur", 0)),
                            args=raw.get("args", {}),
                            cname=raw.get("cname"),
                        )
                        yield evt
                        count += 1
                        if max_events > 0 and count >= max_events:
                            return


def collect_cross_core_sync_events(
    trace_path: str,
    physical_core: int = 0,
) -> dict:
    """
    收集指定物理核上 cubecore/veccore 之间的同步事件。

    分析 SET_CROSS_CORE 和 BAR 事件的时序关系，
    量化 cubecore → veccore 的数据交接延迟。

    Returns:
        {
            "cubecore_sync": [events...],
            "veccore0_sync": [events...],
            "veccore1_sync": [events...],
            "handoff_gaps": [gap_info...],
        }
    """
    cores = {
        f"core{physical_core}.cubecore0",
        f"core{physical_core}.veccore0",
        f"core{physical_core}.veccore1",
    }

    events_by_core = {c: [] for c in cores}

    for evt in stream_global_trace(
        trace_path,
        core_filter=cores,
        event_filter=lambda e: e.get("name") in (
            "SET_CROSS_CORE", "BAR", "WAIT_FLAG",
            "FIX_L0C_TO_DST", "MMAD",
        ),
    ):
        events_by_core.setdefault(evt.core_id, []).append(evt)

    return events_by_core


def analyze_cubecore_veccore_timeline(
    trace_path: str,
    physical_core: int = 0,
) -> dict:
    """
    分析指定物理核上 cubecore 和 veccore 的时间线协同。

    重点:
    1. cubecore FIXP 输出 → veccore VEC 消费 的延迟
    2. cubecore 和 veccore 的时间重叠度
    3. SET_CROSS_CORE 握手模式

    Returns:
        诊断结果 dict
    """
    cube_core = f"core{physical_core}.cubecore0"
    vec_core0 = f"core{physical_core}.veccore0"
    vec_core1 = f"core{physical_core}.veccore1"
    cores = {cube_core, vec_core0, vec_core1}

    # 收集关键 pipeline 事件
    cube_events = []   # cubecore: CUBE + FIXPIPE
    vec0_events = []   # veccore0: VECTOR
    vec1_events = []   # veccore1: VECTOR
    sync_events = {"cube": [], "vec0": [], "vec1": []}

    for evt in stream_global_trace(trace_path, core_filter=cores):
        if evt.core_id == cube_core:
            if evt.pipeline in ("CUBE", "FIXPIPE", "MTE1", "MTE2"):
                cube_events.append(evt)
            if evt.name == "SET_CROSS_CORE":
                sync_events["cube"].append(evt)
        elif evt.core_id == vec_core0:
            if evt.pipeline in ("VECTOR", "MTE2", "MTE3"):
                vec0_events.append(evt)
            if evt.name == "SET_CROSS_CORE":
                sync_events["vec0"].append(evt)
        elif evt.core_id == vec_core1:
            if evt.pipeline in ("VECTOR", "MTE2", "MTE3"):
                vec1_events.append(evt)
            if evt.name == "SET_CROSS_CORE":
                sync_events["vec1"].append(evt)

    # 计算各子核的时间范围
    def time_range(events):
        if not events:
            return 0, 0
        return min(e.ts for e in events), max(e.end_ts for e in events)

    cube_start, cube_end = time_range(cube_events)
    vec0_start, vec0_end = time_range(vec0_events)
    vec1_start, vec1_end = time_range(vec1_events)

    # 计算子核之间的时间重叠
    def overlap_pct(s1, e1, s2, e2):
        if e1 <= s1 or e2 <= s2:
            return 0.0
        overlap_s = max(s1, s2)
        overlap_e = min(e1, e2)
        if overlap_e <= overlap_s:
            return 0.0
        overlap_dur = overlap_e - overlap_s
        min_dur = min(e1 - s1, e2 - s2)
        return overlap_dur / min_dur * 100 if min_dur > 0 else 0.0

    # Pipeline 利用率 (在各自时间范围内)
    def pipeline_util(events, pipeline_name, total_start, total_end):
        total_dur = total_end - total_start
        if total_dur <= 0:
            return 0.0
        busy = sum(e.dur for e in events if e.pipeline == pipeline_name)
        return min(busy / total_dur * 100, 100.0)

    # SET_CROSS_CORE 分析: cubecore 发出 → veccore 收到
    cube_syncs = sorted(sync_events["cube"], key=lambda e: e.ts)
    vec0_syncs = sorted(sync_events["vec0"], key=lambda e: e.ts)
    vec1_syncs = sorted(sync_events["vec1"], key=lambda e: e.ts)

    # 估算 handoff latency: cubecore SET_CROSS_CORE → veccore 最近的后续事件
    handoff_latencies = []
    for cs in cube_syncs:
        # 找 veccore0 中 cs.ts 之后最近的事件
        nearest = None
        for ve in vec0_events:
            if ve.ts >= cs.ts:
                nearest = ve
                break
        if nearest:
            latency_ns = nearest.ts - cs.ts
            handoff_latencies.append(latency_ns)

    # FIXP 完成 → VEC 开始消费 的延迟
    fixp_events = sorted(
        [e for e in cube_events if e.pipeline == "FIXPIPE" and e.name == "FIX_L0C_TO_DST"],
        key=lambda e: e.ts
    )

    result = {
        "physical_core": physical_core,
        "cubecore": {
            "core_id": cube_core,
            "time_range_ns": [round(cube_start, 3), round(cube_end, 3)],
            "duration_ns": round(cube_end - cube_start, 3),
            "event_count": len(cube_events),
            "cube_util_pct": round(pipeline_util(cube_events, "CUBE", cube_start, cube_end), 1),
            "fixpipe_util_pct": round(pipeline_util(cube_events, "FIXPIPE", cube_start, cube_end), 1),
            "mte1_util_pct": round(pipeline_util(cube_events, "MTE1", cube_start, cube_end), 1),
            "mte2_util_pct": round(pipeline_util(cube_events, "MTE2", cube_start, cube_end), 1),
        },
        "veccore0": {
            "core_id": vec_core0,
            "time_range_ns": [round(vec0_start, 3), round(vec0_end, 3)],
            "duration_ns": round(vec0_end - vec0_start, 3),
            "event_count": len(vec0_events),
            "vec_util_pct": round(pipeline_util(vec0_events, "VECTOR", vec0_start, vec0_end), 1),
            "mte2_util_pct": round(pipeline_util(vec0_events, "MTE2", vec0_start, vec0_end), 1),
            "mte3_util_pct": round(pipeline_util(vec0_events, "MTE3", vec0_start, vec0_end), 1),
        },
        "veccore1": {
            "core_id": vec_core1,
            "time_range_ns": [round(vec1_start, 3), round(vec1_end, 3)],
            "duration_ns": round(vec1_end - vec1_start, 3),
            "event_count": len(vec1_events),
            "vec_util_pct": round(pipeline_util(vec1_events, "VECTOR", vec1_start, vec1_end), 1),
            "mte2_util_pct": round(pipeline_util(vec1_events, "MTE2", vec1_start, vec1_end), 1),
            "mte3_util_pct": round(pipeline_util(vec1_events, "MTE3", vec1_start, vec1_end), 1),
        },
        "cross_core_overlap": {
            "cube_vec0_overlap_pct": round(overlap_pct(cube_start, cube_end, vec0_start, vec0_end), 1),
            "cube_vec1_overlap_pct": round(overlap_pct(cube_start, cube_end, vec1_start, vec1_end), 1),
            "vec0_vec1_overlap_pct": round(overlap_pct(vec0_start, vec0_end, vec1_start, vec1_end), 1),
        },
        "sync_analysis": {
            "cube_set_cross_core_count": len(cube_syncs),
            "vec0_set_cross_core_count": len(vec0_syncs),
            "vec1_set_cross_core_count": len(vec1_syncs),
            "handoff_latency_ns": {
                "count": len(handoff_latencies),
                "mean": round(sum(handoff_latencies) / len(handoff_latencies), 4) if handoff_latencies else None,
                "min": round(min(handoff_latencies), 4) if handoff_latencies else None,
                "max": round(max(handoff_latencies), 4) if handoff_latencies else None,
            },
        },
        "fixp_analysis": {
            "fix_l0c_to_dst_count": len(fixp_events),
            "total_fixp_ns": round(sum(e.dur for e in fixp_events), 4),
            "mean_fixp_ns": round(sum(e.dur for e in fixp_events) / len(fixp_events), 4) if fixp_events else 0,
        },
    }

    return result


def analyze_all_physical_cores(
    trace_path: str,
    max_cores: int = 20,
) -> dict:
    """
    从全局 trace 扫描所有物理核的高层指标。

    一次流式遍历收集所有核的时间范围和 pipeline 利用率。
    """
    core_data = {}  # core_id → {pipeline → [events]}

    for evt in stream_global_trace(trace_path):
        if evt.core_id not in core_data:
            core_data[evt.core_id] = {}
        pipeline_events = core_data[evt.core_id]
        if evt.pipeline not in pipeline_events:
            pipeline_events[evt.pipeline] = {"count": 0, "busy_ns": 0.0, "min_ts": evt.ts, "max_end": evt.end_ts}
        stats = pipeline_events[evt.pipeline]
        stats["count"] += 1
        stats["busy_ns"] += evt.dur
        if evt.ts < stats["min_ts"]:
            stats["min_ts"] = evt.ts
        if evt.end_ts > stats["max_end"]:
            stats["max_end"] = evt.end_ts

    # 汇总
    result = {}
    for core_id in sorted(core_data.keys()):
        pipelines = core_data[core_id]
        # 计算总时间范围
        min_ts = min(p["min_ts"] for p in pipelines.values())
        max_end = max(p["max_end"] for p in pipelines.values())
        total_dur = max_end - min_ts

        core_info = {
            "duration_ns": round(total_dur, 3),
            "pipelines": {},
        }
        for pname, stats in sorted(pipelines.items()):
            util = stats["busy_ns"] / total_dur * 100 if total_dur > 0 else 0
            core_info["pipelines"][pname] = {
                "count": stats["count"],
                "busy_ns": round(stats["busy_ns"], 3),
                "util_pct": round(min(util, 100.0), 1),
            }
        result[core_id] = core_info

    return result


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Global trace.json analyzer")
    parser.add_argument("trace_path", help="Path to global trace.json")
    parser.add_argument("--mode", choices=["summary", "cross-core", "all-cores"],
                        default="summary", help="Analysis mode")
    parser.add_argument("--core", type=int, default=0, help="Physical core ID for cross-core analysis")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    if args.mode == "summary":
        # Quick summary: count events per core
        counts = {}
        for evt in stream_global_trace(args.trace_path, max_events=100000):
            counts[evt.core_id] = counts.get(evt.core_id, 0) + 1
        result = {"event_counts_sample": counts, "total_sampled": sum(counts.values())}
    elif args.mode == "cross-core":
        result = analyze_cubecore_veccore_timeline(args.trace_path, args.core)
    elif args.mode == "all-cores":
        result = analyze_all_physical_cores(args.trace_path)
    else:
        result = {"error": f"Unknown mode: {args.mode}"}

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Result written to {args.output}")
    else:
        print(output_json)
