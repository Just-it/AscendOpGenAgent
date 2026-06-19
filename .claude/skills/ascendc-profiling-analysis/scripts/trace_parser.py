#!/usr/bin/env python3
"""
trace_parser.py — Chrome Trace JSON 解析器 + 公共数据结构

解析 msprof simulator 产出的 trace.json (Chrome Trace Format)，
提供按 pipeline (PID) 分组的事件访问、多核路径发现等基础能力。

用法:
    from trace_parser import load_core_trace, get_all_core_paths, PIPELINE_PID_MAP

    # 加载单核 trace
    pipelines = load_core_trace("path/to/core0.veccore0/trace.json")
    vec_events = pipelines.get(30, [])  # PID 30 = VECTOR

    # 发现所有核的 trace 路径
    core_paths = get_all_core_paths("path/to/simulator/")
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# PID → pipeline 名称映射 (msprof simulator 约定)
# veccore 和 cubecore 共享部分 PID (SCALAR=10, SCALARLDST=20, MTE2=60, MTE3=70, FLOWCTRL=90)
# cubecore 独有: CUBE=40, MTE1=50, FIXPIPE=80
# veccore 独有: VECTOR=30, MTE3=70
PIPELINE_PID_MAP = {
    10: "SCALAR",
    20: "SCALARLDST",
    30: "VECTOR",
    40: "CUBE",
    50: "MTE1",
    60: "MTE2",
    70: "MTE3",
    80: "FIXPIPE",
    90: "FLOWCTRL",
    100: "ALL",
    110: "CACHEMISS",
}

# 反向映射
PIPELINE_NAME_MAP = {v: k for k, v in PIPELINE_PID_MAP.items()}

# cannsim PID -> msprof PID 映射
# cannsim 使用不同的 PID 编号，需要转换为 msprof 标准 PID
# Ascend950 细分了 VEC流水线为 RVECSU/RVECEX/RVECLD/RVECST，
# 统一映射到 VECTOR (PID 30)
_CANNSIM_PID_REMAP = {
    1: 10,   # SCALAR
    2: 20,   # SCALARLDST
    4: 60,   # MTE2
    5: 30,   # VECTOR (cannsim: "05_VEC")
    6: 40,   # CUBE
    7: 70,   # MTE3
    3: 50,   # MTE1
    11: 30,   # RVECSU -> VECTOR (register vec setup)
    12: 30,   # RVECEX -> VECTOR (register vec execution)
    13: 30,   # RVECLD -> VECTOR (register vec load)
    14: 30,   # RVECST -> VECTOR (register vec store)
    15: 90,   # FLOWCTRL
    17: 80,   # FIXPIPE (cannsim: "17_FIXP")
    18: 30,   # RVECLP -> VECTOR (register vec loop)
}

# cannsim process_name 前缀 -> msprof PID 映射 （兜底映射）
_CANNSIM_NAME_REMAP = {
    "SCALAR": 10,
    "SCALARLDST": 20,
    "VEC": 30,
    "VECTOR": 30,
    "CUBE": 40,
    "MTE1": 50,
    "MTE2": 60,
    "MTE3": 70,
    "FIXP": 80,
    "FIXPIPE": 80,
    "FLOWCTRL": 90,
    "FLOWCONTROL": 90,
    "RVECSU": 30,
    "RVECEX": 30,
    "RVECLD": 30,
    "RVECST": 30,
    "RVECLP": 30,
    "PUSHQ": 90,  # queue control -> FLOWCTRL
}

# ── PID 常量 (避免硬编码) ──
# 共享 pipeline
PID_SCALAR = 10
PID_SCALARLDST = 20
PID_MTE2 = 60
PID_FLOWCTRL = 90
PID_ALL = 100
PID_CACHEMISS = 110
# veccore 独有
PID_VECTOR = 30
PID_MTE3 = 70
# cubecore 独有
PID_CUBE = 40
PID_MTE1 = 50
PID_FIXPIPE = 80

# 已知的 drain / barrier 指令名
DRAIN_OPS = frozenset({
    "VNCHWCONV", "BAR", "PipeBarrier", "PIPE_BARRIER",
    "VNCHWCONV_B16", "VNCHWCONV_B32",
})


@dataclass
class TraceEvent:
    """单条 trace 事件"""
    name: str
    ph: str           # 事件类型: "X"=duration, "M"=metadata, "i"=instant
    pid: int          # pipeline ID (10/20/30/60/70/90/100/110)
    tid: int          # thread ID
    ts: float         # 起始时间 (ps)
    dur: float        # 持续时间 (ps)
    args: dict = field(default_factory=dict)
    cname: Optional[str] = None  # 颜色标记

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur

    @property
    def pipeline_name(self) -> str:
        return PIPELINE_PID_MAP.get(self.pid, f"UNKNOWN_{self.pid}")

    @property
    def pc_addr(self) -> Optional[str]:
        return self.args.get("pc_addr")

    @property
    def detail(self) -> Optional[str]:
        return self.args.get("detail")

    def is_wait_flag(self) -> bool:
        return self.name == "WAIT_FLAG"

    def waited_pipeline(self) -> Optional[str]:
        """从 WAIT_FLAG 的 detail 字段提取被等待的 pipeline"""
        if not self.is_wait_flag():
            return None
        detail = self.detail or ""
        m = re.search(r"PIPE:(\w+)", detail)
        return m.group(1) if m else None

    def is_drain_op(self) -> bool:
        return self.name in DRAIN_OPS


def _parse_event(raw: dict) -> Optional[TraceEvent]:
    """将原始 JSON dict 转为 TraceEvent，跳过非 duration 事件"""
    ph = raw.get("ph", "")
    if ph != "X":
        return None
    return TraceEvent(
        name=raw.get("name", ""),
        ph=ph,
        pid=raw.get("pid", 0),
        tid=raw.get("tid", 0),
        ts=float(raw.get("ts", 0)),
        dur=float(raw.get("dur", 0)),
        args=raw.get("args", {}),
        cname=raw.get("cname"),
    )

def _detect_cannsim_pid_map(raw_events: list[dict]) -> dict[int, int]:
    """从 cannsim trace 的 metadata 事件中构建 PID 重映射表。

    cannsim 使用不同的 PID 编号（1=SCALAR, 5=VEC 等），
    需要映射到 msprof 标准 PID (10=SCALAR, 30=VECTOR等)。

    Returns：
        {cannsim_pid: msprof_pid} 映射表；若不是cannsim 格式则返回空dict
    """
    pid_map = {}
    is_cannsim = False
    for raw in raw_events:
        if raw.get("ph") == "M" and raw.get("name") == "process_name":
            cannsim_pid = raw.get("pid", -1)
            pname = raw.get("args", {}).get("name", "")
            # cannsim 格式："01_SCALAR", "05_VEC", "04_MTE2" 等
            # 去掉前缀数字和下划线
            suffix = re.sub(r'^\d+_', '', pname).upper()
            if suffix in _CANNSIM_NAME_REMAP:
                msprof_pid = _CANNSIM_NAME_REMAP[suffix]
                pid_map[cannsim_pid] = msprof_pid
                is_cannsim = True
            elif cannsim_pid in _CANNSIM_PID_REMAP:
                pid_map[cannsim_pid] = _CANNSIM_PID_REMAP[cannsim_pid]
                is_cannsim = True
    return pid_map if is_cannsim else {}

def load_core_trace(trace_path: str) -> dict[int, list[TraceEvent]]:
    """
    加载单核 trace.json，返回按 PID 分组、按 ts 排序的事件字典。

    自动检测 cannsim 格式并将 PID 重映射为 msprof 标准  PID。

    Returns:
        {pid: [TraceEvent, ...]} — 仅包含 ph="X" 的 duration 事件
    """
    with open(trace_path, "r") as f:
        data = json.load(f)
    
    # 兼容两种 trace 格式
    #  msprof: {"traceEvents": [...]}
    #  cannsim: [...] (直接数组)
    if isinstance(data, list):
        raw_events = data
    else:
        raw_events = data.get("traceEvents", [])

    # 检测 cannsim 格式并构建 PID 重映射表
    cannsim_pid_map = _detect_cannsim_pid_map(raw_events)

    pipelines: dict[int, list[TraceEvent]] = {}

    for raw in raw_events:
        evt = _parse_event(raw)
        if evt is None:
            continue
        # 重映射 cannsim PID 到 msprof 标准 PID
        if cannsim_pid_map and evt.pid in cannsim_pid_map:
            evt.pid = cannsim_pid_map[evt.pid]
        pipelines.setdefault(evt.pid, []).append(evt)

    # 按时间排序
    for pid in pipelines:
        pipelines[pid].sort(key=lambda e: e.ts)

    return pipelines


def is_cubecore(core_id: str) -> bool:
    """判断 core_id 是否为 cubecore"""
    return ".cubecore" in core_id


def is_veccore(core_id: str) -> bool:
    """判断 core_id 是否为 veccore"""
    return ".veccore" in core_id


def get_all_core_paths(
    simulator_dir: str,
    core_type: str = "all",
) -> dict[str, str]:
    """
    发现 simulator_dir 下所有核的 trace.json。

    支持三种核类型:
      - coreN.veccoreM  (Vector 核)
      - coreN.cubecore0  (Cube 核, AIC:AIV 异构算子)
      - 两者兼有 (MIX 类型算子)

    支持两种目录结构:
      1. simulator_dir/coreN.{veccore|cubecore}M/trace.json  (直接)
      2. simulator_dir/.../simulator/coreN.{veccore|cubecore}M/trace.json  (嵌套)

    Args:
        simulator_dir: simulator 输出目录
        core_type: "all" (默认) | "veccore" | "cubecore"

    Returns:
        {core_id: trace_json_path}
    """
    # 匹配 veccore 和 cubecore 两种目录名
    core_pattern = re.compile(r"^core\d+\.(veccore\d+|cubecore\d+)$")
    result: dict[str, str] = {}

    def _should_include(entry: str) -> bool:
        if not core_pattern.match(entry):
            return False
        if core_type == "veccore":
            return is_veccore(entry)
        if core_type == "cubecore":
            return is_cubecore(entry)
        return True  # "all"

    # 策略1: 直接子目录
    if os.path.isdir(simulator_dir):
        for entry in os.listdir(simulator_dir):
            if _should_include(entry):
                trace_path = os.path.join(simulator_dir, entry, "trace.json")
                if os.path.isfile(trace_path):
                    result[entry] = trace_path

    if result:
        return result

    # 策略2: 递归查找 simulator/ 子目录
    for root, dirs, files in os.walk(simulator_dir):
        if os.path.basename(root) == "simulator":
            for d in dirs:
                if _should_include(d):
                    trace_path = os.path.join(root, d, "trace.json")
                    if os.path.isfile(trace_path):
                        result[d] = trace_path
            if result:
                break

    return result


def compute_core_duration(pipelines: dict[int, list[TraceEvent]]) -> float:
    """计算单核的总执行时间 (ps)：所有 pipeline 中最晚结束时间 - 最早开始时间"""
    min_ts = float("inf")
    max_end = 0.0
    for events in pipelines.values():
        for e in events:
            if e.ts < min_ts:
                min_ts = e.ts
            end = e.ts + e.dur
            if end > max_end:
                max_end = end
    if min_ts == float("inf"):
        return 0.0
    return max_end - min_ts


def get_pipeline_events(
    pipelines: dict[int, list[TraceEvent]],
    pipeline_name: str,
) -> list[TraceEvent]:
    """按 pipeline 名称获取事件列表"""
    pid = PIPELINE_NAME_MAP.get(pipeline_name)
    if pid is None:
        return []
    return pipelines.get(pid, [])


def find_concurrent_events(
    pipelines: dict[int, list[TraceEvent]],
    ts: float,
    end_ts: float,
    exclude_pid: Optional[int] = None,
) -> dict[int, list[TraceEvent]]:
    """
    查找在 [ts, end_ts] 时间窗口内与之重叠的所有 pipeline 事件。

    Returns:
        {pid: [overlapping TraceEvent, ...]}
    """
    result: dict[int, list[TraceEvent]] = {}
    for pid, events in pipelines.items():
        if pid == exclude_pid:
            continue
        overlapping = []
        for e in events:
            # 事件与窗口有重叠: e.ts < end_ts and e.end_ts > ts
            if e.ts < end_ts and e.end_ts > ts:
                overlapping.append(e)
            elif e.ts >= end_ts:
                break  # 已排序，后续不会再有重叠
        if overlapping:
            result[pid] = overlapping
    return result


def ps_to_us(ps: float) -> float:
    """皮秒 → 微秒"""
    return ps / 1e6


def ps_to_ns(ps: float) -> float:
    """皮秒 → 纳秒"""
    return ps / 1e3


def compute_pipeline_coverage(
    pipelines: dict[int, list[TraceEvent]],
    pid: int,
    window_start: float,
    window_end: float,
) -> float:
    """
    计算指定 pipeline 在时间窗口内的覆盖率 (0.0~1.0)。

    遍历该 pid 的事件，累加与窗口重叠的时长，除以窗口总时长。
    """
    window_dur = window_end - window_start
    if window_dur <= 0:
        return 0.0

    events = pipelines.get(pid, [])
    covered = 0.0
    for e in events:
        if e.ts >= window_end:
            break
        if e.end_ts <= window_start:
            continue
        overlap_start = max(e.ts, window_start)
        overlap_end = min(e.end_ts, window_end)
        covered += overlap_end - overlap_start

    return min(covered / window_dur, 1.0)


def has_cachemiss_in_window(
    pipelines: dict[int, list[TraceEvent]],
    window_start: float,
    window_end: float,
) -> bool:
    """检查窗口内是否有 CACHEMISS (PID=110) 事件"""
    for e in pipelines.get(PID_CACHEMISS, []):
        if e.ts >= window_end:
            break
        if e.end_ts > window_start:
            return True
    return False


def has_flowctrl_in_window(
    pipelines: dict[int, list[TraceEvent]],
    window_start: float,
    window_end: float,
) -> bool:
    """检查窗口内是否有 FLOWCTRL (PID=90) 事件"""
    for e in pipelines.get(PID_FLOWCTRL, []):
        if e.ts >= window_end:
            break
        if e.end_ts > window_start:
            return True
    return False


def compute_pipeline_utilization(
    pipelines: dict[int, list[TraceEvent]],
    pid: int,
) -> float:
    """计算指定 pipeline 的整体利用率 (busy_time / total_duration)"""
    total_dur = compute_core_duration(pipelines)
    if total_dur <= 0:
        return 0.0

    busy_time = sum(e.dur for e in pipelines.get(pid, []))
    return min(busy_time / total_dur, 1.0)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 trace_parser.py <trace.json | simulator_dir>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isfile(path):
        pipelines = load_core_trace(path)
        for pid, events in sorted(pipelines.items()):
            name = PIPELINE_PID_MAP.get(pid, f"PID_{pid}")
            dur_total = sum(e.dur for e in events)
            print(f"  {name} (PID {pid}): {len(events)} events, total_dur={ps_to_ns(dur_total):.3f} ns")
    elif os.path.isdir(path):
        cores = get_all_core_paths(path)
        print(f"Found {len(cores)} cores:")
        for core_id, trace_path in sorted(cores.items()):
            pipelines = load_core_trace(trace_path)
            dur = compute_core_duration(pipelines)
            print(f"  {core_id}: duration={ps_to_us(dur):.3f} us")
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)
