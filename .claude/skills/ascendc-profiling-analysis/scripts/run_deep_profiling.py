#!/usr/bin/env python3
"""
run_deep_profiling.py — 深度 profiling 分析编排脚本

将 T1→T2→T3→T8→T9→profiling_evidence 的完整链路封装为单命令调用，
消除 agent 手动串联的脆弱性。

用法:
    python3 run_deep_profiling.py \
        --work-dir output/{op_name}_evo_{ts}/{solution_ref} \
        --op-name {op_name} \
        [--simulator-dir <path>]   # 可选：已有 trace 数据时跳过 profiling_runner
        [--output <path>]          # 可选：输出 JSON 到文件

输出:
    合并 JSON，含 profiling_analysis + profiling_evidence + 各 T 工具原始结果
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 确保同目录下的 T 工具可 import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# 引入 profiling_evidence（位于 .claude/skills/evolution-world-model/scripts/）
_EVOLUTION_WM_SCRIPTS = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "..", "evolution-world-model", "scripts")
)
if _EVOLUTION_WM_SCRIPTS not in sys.path:
    sys.path.insert(0, _EVOLUTION_WM_SCRIPTS)

from analyze_cross_core import analyze_cross_core_balance
from analyze_pipeline_bubbles import analyze_pipeline_bubbles
from analyze_vec_internal import analyze_vec_internal
from overlap_analyzer import analyze_overlap
from dma_efficiency import analyze_dma_efficiency
from profiling_runner import run_simulator_profiling

try:
    from profiling_evidence import extract_profiling_evidence
except ImportError:
    extract_profiling_evidence = None


def run_deep_profiling(
    work_dir: str,
    op_name: str,
    simulator_dir: str = None,
    timeout: int = 3600,
    test_case_csv: str = None,
    case_id: str = None,
) -> dict:
    """
    执行完整的深度 profiling 分析链路。

    链路: trace获取 → T1 → T2 → T3 → T8 → T9 → profiling_evidence

    Args:
        work_dir: 算子工作目录
        op_name: 算子名称
        simulator_dir: 已有 trace 目录（为 None 时自动运行 profiling_runner）
        timeout: msprof simulator 超时秒数 (默认 3600，即 1 小时；最大 21600 即 6 小时)
        test_case_csv: test_cases.csv 路径，提供时使用其中的 shape 替代 get_inputs()
        case_id: CSV 中的 case_id，未指定时选元素数最大的 case

    Returns:
        合并结果 dict，含 profiling_analysis / profiling_evidence / 各 T 原始结果
    """
    work_dir = str(Path(work_dir).resolve())
    result = {"success": False, "work_dir": work_dir, "op_name": op_name}

    # ── Step 0: 获取 simulator trace 目录 ──
    if simulator_dir and os.path.isdir(simulator_dir):
        result["simulator_dir"] = simulator_dir
        result["trace_source"] = "reused"
    else:
        runner_result = run_simulator_profiling(
            work_dir=work_dir,
            op_name=op_name,
            timeout=timeout,
            test_case_csv=test_case_csv,
            case_id=case_id,
        )
        if not runner_result.get("success"):
            result["error"] = f"profiling_runner failed: {runner_result.get('error', 'unknown')}"
            return result
        simulator_dir = runner_result["simulator_dir"]
        result["simulator_dir"] = simulator_dir
        result["trace_source"] = "generated"

    # ── Step 1: T1 跨核负载均衡 (必须成功) ──
    t1 = analyze_cross_core_balance(simulator_dir)
    result["t1_cross_core"] = t1
    if t1.get("error"):
        result["error"] = f"T1 failed: {t1['error']}"
        return result

    slowest_core = t1.get("slowest_core")
    if not slowest_core:
        result["error"] = "T1 did not identify slowest_core"
        return result

    imbalance_ratio = t1.get("imbalance_ratio", 1.0)

    # ── Step 2: T2 D类空泡检测 ──
    d_class_pct = 0.0
    primary_bottleneck = None
    dominant_subtype = None
    try:
        t2 = analyze_pipeline_bubbles(simulator_dir, slowest_core)
        result["t2_pipeline_bubbles"] = t2
        d_class_pct = t2.get("d_class_pct", 0.0)
        primary_bottleneck = t2.get("primary_bottleneck")
        # dominant_subtype: T2 sub_type_breakdown 中 total_us 最大的 key
        stb = t2.get("sub_type_breakdown", {})
        if stb:
            dominant_subtype = max(stb, key=lambda k: stb[k].get("total_us", 0))
    except Exception as e:
        result["t2_pipeline_bubbles"] = {"error": str(e)}

    # ── Step 3: T3 VEC内部空泡 ──
    c_class_pct = 0.0
    pure_compute_pct = 0.0
    c_class_primary_cause = None
    try:
        t3 = analyze_vec_internal(simulator_dir, slowest_core)
        result["t3_vec_internal"] = t3
        bs = t3.get("bubble_summary", {})
        c_class_pct = bs.get("c_class", {}).get("pct", 0.0)
        pure_compute_pct = bs.get("pure_compute_pct", 0.0)
        c_class_primary_cause = t3.get("c_class_detail", {}).get("primary_cause")
    except Exception as e:
        result["t3_vec_internal"] = {"error": str(e)}

    # ── Step 4: T8 流水线重叠度 ──
    overlap_status = None
    try:
        t8 = analyze_overlap(simulator_dir, slowest_core)
        result["t8_overlap"] = t8
        overlap_status = t8.get("overlap_status")
    except Exception as e:
        result["t8_overlap"] = {"error": str(e)}

    # ── Step 5: T9 DMA效率 ──
    dma_efficiency = {}
    try:
        t9 = analyze_dma_efficiency(simulator_dir, slowest_core)
        result["t9_dma_efficiency"] = t9
        mte2_info = t9.get("mte2", {})
        dma_efficiency = {
            "mte2_short_pct": mte2_info.get("short_transfer_pct", 0),
        }
    except Exception as e:
        result["t9_dma_efficiency"] = {"error": str(e)}

    # ── Step 6: 构建 profiling_analysis 字典 ──
    profiling_analysis = {
        "imbalance_ratio": imbalance_ratio,
        "d_class_pct": d_class_pct,
        "c_class_pct": c_class_pct,
        "primary_bottleneck": primary_bottleneck,
        "pure_compute_pct": pure_compute_pct,
        "c_class_primary_cause": c_class_primary_cause,
        "overlap_status": overlap_status,
        "dominant_subtype": dominant_subtype,
        "pattern_type": None,  # T7 不在核心链路
        "dma_efficiency": dma_efficiency,
    }
    result["profiling_analysis"] = profiling_analysis

    # ── Step 7: 调用 profiling_evidence 生成策略映射 ──
    if extract_profiling_evidence is not None:
        evidence = extract_profiling_evidence({"profiling_analysis": profiling_analysis})
        result["profiling_evidence"] = evidence
    else:
        result["profiling_evidence"] = None
        result["warning"] = "profiling_evidence module not available"

    result["success"] = True
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Deep profiling orchestrator: T1→T2→T3→T8→T9→evidence"
    )
    parser.add_argument("--work-dir", required=True, help="Operator work directory")
    parser.add_argument("--op-name", required=True, help="Operator name")
    parser.add_argument("--simulator-dir", help="Existing simulator trace directory (skip profiling_runner)")
    parser.add_argument("--output", help="Output JSON file path (default: stdout)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="msprof simulator timeout in seconds (default: 3600, i.e. 1 hour; max: 21600, i.e. 6 hours)")
    parser.add_argument("--test-case-csv", default=None,
                        help="Path to test_cases.csv; when provided, use its shapes for profiling instead of get_inputs()")
    parser.add_argument("--case-id", default=None,
                        help="Specific case_id from test_cases.csv (default: largest case by element count)")
    args = parser.parse_args()

    result = run_deep_profiling(
        work_dir=args.work_dir,
        op_name=args.op_name,
        simulator_dir=args.simulator_dir,
        timeout=args.timeout,
        test_case_csv=args.test_case_csv,
        case_id=args.case_id,
    )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Deep profiling result written to {args.output}")
    else:
        print(output_json)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()

