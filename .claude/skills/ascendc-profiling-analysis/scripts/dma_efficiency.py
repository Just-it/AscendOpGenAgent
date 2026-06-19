#!/usr/bin/env python3
"""
T9: dma_efficiency.py — DMA 搬运效率分析

分析 MTE2/MTE3 的搬运效率 — 每次搬运的数据量是否充分利用了带宽，
是否存在大量小粒度搬运。

用法:
    python3 dma_efficiency.py --simulator-dir <path> --core-id <core> [--hw-params <path>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parser import (
    get_all_core_paths, load_core_trace, compute_core_duration,
    get_pipeline_events, ps_to_us, ps_to_ns, is_cubecore,
    PID_MTE2, PID_MTE3, PID_MTE1,
)

# 短搬运阈值 (ps): 持续时间低于此值视为小粒度搬运
DEFAULT_SHORT_THRESHOLD_PS = 200


def _analyze_dma_pipeline(events: list, short_threshold_ps: float) -> dict:
    """分析单个 DMA pipeline 的搬运效率"""
    if not events:
        return {
            "count": 0,
            "mean_dur_ps": 0,
            "median_dur_ps": 0,
            "short_transfer_pct": 0.0,
            "short_threshold_ps": short_threshold_ps,
            "verdict": "no_events",
        }

    durations = sorted(e.dur for e in events)
    n = len(durations)
    total = sum(durations)
    mean_dur = total / n
    median_dur = durations[n // 2] if n % 2 == 1 else (durations[n // 2 - 1] + durations[n // 2]) / 2

    short_count = sum(1 for d in durations if d < short_threshold_ps)
    short_pct = short_count / n * 100

    if short_pct > 40:
        verdict = "undersize_transfers"
    elif short_pct > 20:
        verdict = "moderate"
    else:
        verdict = "acceptable"

    # 分布统计
    p10 = durations[max(0, n // 10)]
    p90 = durations[min(n - 1, n * 9 // 10)]

    return {
        "count": n,
        "mean_dur_ps": round(mean_dur, 1),
        "median_dur_ps": round(median_dur, 1),
        "min_dur_ps": round(durations[0], 1),
        "max_dur_ps": round(durations[-1], 1),
        "p10_dur_ps": round(p10, 1),
        "p90_dur_ps": round(p90, 1),
        "total_dur_ps": round(total, 1),
        "short_transfer_count": short_count,
        "short_transfer_pct": round(short_pct, 1),
        "short_threshold_ps": short_threshold_ps,
        "verdict": verdict,
    }


def _estimate_bandwidth(
    mte2_stats: dict,
    mte3_stats: dict,
    total_dur_ps: float,
    hw_params: dict = None,
) -> dict:
    """估算带宽利用率"""
    if hw_params is None:
        return {"estimated_pct": None, "comment": "No hw_params provided"}

    peak_bw_gbps = hw_params.get("peak_bw_gbps")
    if not peak_bw_gbps or total_dur_ps <= 0:
        return {"estimated_pct": None, "comment": "Missing peak_bw_gbps or zero duration"}

    # 总 DMA 活跃时间
    mte2_total = mte2_stats.get("total_dur_ps", 0)
    mte3_total = mte3_stats.get("total_dur_ps", 0)
    dma_total = mte2_total + mte3_total

    # DMA 占比作为带宽利用率的粗略估计
    dma_utilization = dma_total / total_dur_ps * 100 if total_dur_ps > 0 else 0

    comment = ""
    if dma_utilization < 30:
        comment = f"DMA 活跃时间仅占 {dma_utilization:.1f}%，搬运效率低"
    elif dma_utilization < 60:
        comment = f"DMA 活跃时间占 {dma_utilization:.1f}%，有提升空间"
    else:
        comment = f"DMA 活跃时间占 {dma_utilization:.1f}%，带宽利用较充分"

    # 如果短搬运多，补充说明
    mte2_short = mte2_stats.get("short_transfer_pct", 0)
    if mte2_short > 30:
        comment += f"；MTE2 短搬运占 {mte2_short:.0f}%，tile 过小是主因"

    return {
        "estimated_pct": round(dma_utilization, 1),
        "hw_peak_gbps": peak_bw_gbps,
        "comment": comment,
    }


def analyze_dma_efficiency(
    simulator_dir: str,
    core_id: str,
    hw_params: dict = None,
    short_threshold_ps: float = DEFAULT_SHORT_THRESHOLD_PS,
) -> dict:
    """
    分析 DMA 搬运效率。

    Args:
        simulator_dir: simulator 输出目录
        core_id: 核标识
        hw_params: 硬件参数 (可选)
        short_threshold_ps: 短搬运阈值 (ps)

    Returns:
        DMA 效率分析结果 dict
    """
    core_paths = get_all_core_paths(simulator_dir)
    if core_id not in core_paths:
        return {"tool": "T9", "error": f"Core {core_id} not found"}

    pipelines = load_core_trace(core_paths[core_id])
    total_dur = compute_core_duration(pipelines)

    # 自动检测核类型
    is_cube = is_cubecore(core_id)
    core_type = "cubecore" if is_cube else "veccore"

    mte2_events = get_pipeline_events(pipelines, "MTE2")
    mte2_stats = _analyze_dma_pipeline(mte2_events, short_threshold_ps)

    if is_cube:
        # Cubecore: MTE1 (L1→L0A/L0B) + MTE2 (GM→L1)
        mte1_events = get_pipeline_events(pipelines, "MTE1")
        mte1_stats = _analyze_dma_pipeline(mte1_events, short_threshold_ps)

        bw = _estimate_bandwidth(mte2_stats, mte1_stats, total_dur, hw_params)

        recommendations = []
        if mte1_stats["verdict"] == "undersize_transfers":
            recommendations.append(
                f"MTE1 短搬运占 {mte1_stats['short_transfer_pct']}% → 增大 tile 提升 L1→L0 搬运效率"
            )
        if mte2_stats["verdict"] == "undersize_transfers":
            recommendations.append(
                f"MTE2 短搬运占 {mte2_stats['short_transfer_pct']}% → 增大 tile 提升 GM→L1 搬运效率"
            )
        if mte1_stats["count"] > 0 and mte2_stats["count"] > 0:
            ratio = mte1_stats["count"] / mte2_stats["count"]
            if ratio > 3:
                recommendations.append(
                    f"MTE1 搬运次数是 MTE2 的 {ratio:.1f} 倍 → 可能存在冗余 L1→L0 搬运"
                )
        if not recommendations:
            recommendations.append("DMA 搬运效率正常，无明显优化空间")

        return {
            "tool": "T9",
            "core_id": core_id,
            "core_type": core_type,
            "mte1_stats": mte1_stats,
            "mte2_stats": mte2_stats,
            "bandwidth_utilization": bw,
            "recommendations": recommendations,
        }

    # Veccore: MTE2 (GM→UB) + MTE3 (UB→GM)
    mte3_events = get_pipeline_events(pipelines, "MTE3")
    mte3_stats = _analyze_dma_pipeline(mte3_events, short_threshold_ps)

    bw = _estimate_bandwidth(mte2_stats, mte3_stats, total_dur, hw_params)

    # 建议
    recommendations = []
    if mte2_stats["verdict"] == "undersize_transfers":
        recommendations.append(
            f"MTE2 短搬运占 {mte2_stats['short_transfer_pct']}% → 增大 tile (P2) 可提升搬运效率"
        )
    if mte3_stats["verdict"] == "undersize_transfers":
        recommendations.append(
            f"MTE3 短搬运占 {mte3_stats['short_transfer_pct']}% → 增大 tile (P2) 可提升搬运效率"
        )
    if mte2_stats["count"] > 0 and mte3_stats["count"] > 0:
        ratio = mte2_stats["count"] / mte3_stats["count"]
        if ratio > 3:
            recommendations.append(
                f"MTE2 搬运次数是 MTE3 的 {ratio:.1f} 倍 → 可能存在冗余搬运"
            )
    if bw.get("estimated_pct") and bw["estimated_pct"] < 30:
        recommendations.append("DMA 带宽利用率低 → 考虑向量化搬运 (P10) 减少搬运次数")

    if not recommendations:
        recommendations.append("DMA 搬运效率正常，无明显优化空间")

    return {
        "tool": "T9",
        "core_id": core_id,
        "mte2_stats": mte2_stats,
        "mte3_stats": mte3_stats,
        "bandwidth_utilization": bw,
        "recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser(description="T9: DMA transfer efficiency analysis")
    parser.add_argument("--simulator-dir", required=True)
    parser.add_argument("--core-id", required=True)
    parser.add_argument("--hw-params", help="Path to hw_params JSON file")
    parser.add_argument("--short-threshold", type=float, default=DEFAULT_SHORT_THRESHOLD_PS,
                        help=f"Short transfer threshold in ps (default: {DEFAULT_SHORT_THRESHOLD_PS})")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    hw = None
    if args.hw_params:
        with open(args.hw_params) as f:
            hw = json.load(f)

    result = analyze_dma_efficiency(
        args.simulator_dir, args.core_id, hw, args.short_threshold
    )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T9 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
