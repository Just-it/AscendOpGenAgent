#!/usr/bin/env python3
"""analyze_profiling.py — Analyze AscendC msprof op_summary CSV
and diagnose AIV/AIC pipeline bottlenecks.

Usage:
  python3 analyze_profiling.py <profiling_dir> [--task-type vector|cube|cv-mix] [--output result.json]
  python3 analyze_profiling.py --op-summary /path/to/op_summary_xxx.csv [--task-type ...] [--output ...]

The script searches for the latest op_summary CSV under:
  profiling_dir/ModelNew_*/PROF_*/mindstudio_profiler_output/op_summary_*.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def find_latest_csv(profiling_dir: str):
    """Find the latest op_summary or kernel_details CSV under profiling_dir.

    Search order:
    1. Primary: mindstudio_profiler_output/op_summary_*.csv (msprof native)
    2. Secondary: torch_npu.profiler ASCEND_PROFILER_OUTPUT/op_summary_*.csv
    3. Tertiary: recursive op_summary_*.csv search
    4. Fallback: kernel_details.csv (torch_npu.profiler alternative output)
    """
    # Primary: mindstudio_profiler_output directly (filtered by evaluate.py)
    pattern_primary = str(
        Path(profiling_dir) / "ModelNew_*" / "PROF_*"
        / "mindstudio_profiler_output" / "op_summary_*.csv"
    )
    csv_files = glob.glob(pattern_primary)

    # Secondary: torch_npu.profiler output (ASCEND_PROFILER_OUTPUT)
    if not csv_files:
        pattern_secondary = str(
            Path(profiling_dir) / "*_ascend_pt"
            / "ASCEND_PROFILER_OUTPUT" / "op_summary_*.csv"
        )
        csv_files = glob.glob(pattern_secondary)

    # Tertiary: recursive search, prefer non-origin_data files
    if not csv_files:
        all_files = glob.glob(
            str(Path(profiling_dir) / "**" / "op_summary_*.csv"), recursive=True
        )
        csv_files = [f for f in all_files if "origin_data" not in f] or all_files

    # Fallback: kernel_details.csv (torch_npu.profiler may produce this instead)
    if not csv_files:
        kd_files = glob.glob(
            str(Path(profiling_dir) / "**" / "kernel_details.csv"), recursive=True
        )
        csv_files = [f for f in kd_files if "origin_data" not in f] or kd_files

    if not csv_files:
        return None
    return max(csv_files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _safe_float(row: dict, key: str, default: float = 0.0) -> float:
    """Extract float value from row, trying multiple key variants."""
    val = row.get(key, "")
    if val is None or str(val).strip() in ("", "N/A", "nan"):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_float_multi(row: dict, *keys, default: float = 0.0) -> float:
    """Try multiple column name variants, return first valid float."""
    for key in keys:
        val = row.get(key, "")
        if val is not None and str(val).strip() not in ("", "N/A", "nan"):
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return default


def _read_csv(csv_path: str) -> list:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _pick_representative(rows: list) -> dict:
    """Return the median-duration row after filtering out AI_CPU tasks.

    Handles both op_summary_*.csv and kernel_details.csv column name formats.
    """
    valid = []
    for r in rows:
        task_type = str(
            r.get("Task Type", r.get("Task type", r.get("Type", "")))
        ).upper()
        if "AI_CPU" in task_type:
            continue
        dur = _safe_float_multi(r, "Task Duration(us)", "Duration(us)")
        if dur > 0:
            valid.append(r)
    if not valid:
        return None
    valid.sort(key=lambda r: _safe_float_multi(r, "Task Duration(us)", "Duration(us)"))
    return valid[len(valid) // 2]


def _detect_csv_format(rows: list) -> str:
    """Detect CSV format: 'op_summary' (has pipeline ratios) or 'kernel_details' (timing only)."""
    if not rows:
        return "unknown"
    sample = rows[0]
    # op_summary has aiv_*_ratio or aic_*_ratio columns
    if any(k.startswith("aiv_") or k.startswith("aic_") for k in sample.keys()):
        return "op_summary"
    # kernel_details has Duration(us) and Type columns
    if "Duration(us)" in sample or "Type" in sample:
        return "kernel_details"
    return "unknown"


# ---------------------------------------------------------------------------
# Optimization hints database
# ---------------------------------------------------------------------------

_VECTOR_HINTS = {
    "scalar_bound": [
        "减少 GetValue 调用，改用向量化批量处理 (P10)",
        "将标量归约循环改为 ReduceSum/ReduceMax 向量指令 (P68)",
        "改用 Vector Counter 模式消除 repeat/mask 标量控制 (P67/P84)",
        "避免在 Process 函数中使用分支逐元素处理",
    ],
    "memory_bound": [
        "增大 tile_size 减少分块次数，降低 MTE2 调用频率 (P2)",
        "启用双缓冲 (P1)，隐藏搬运延迟；Cube/CV 可用 Triple buffer (P20)",
        "向量化数据加载 (P10)，提升带宽利用率",
        "检查 DataCopy 偏移/对齐 (P25/P66)，保持 512B 对齐以发挥最大带宽",
        "权重/只读数据一次加载多次复用 (P34)，减少 MTE2 往返",
    ],
    "compute_bound": [
        "检查冗余计算，合并相邻 Add/Mul，形成 UB 融合链 (P69/P84)",
        "改用低延迟归约指令组合 (BlockReduce + WholeReduce, P68)",
        "考虑精度-性能折中，如近似激活函数 (P13)",
        "减少中间 tensor 分配，通过 TBuf/TQueBind 复用 UB (P22/P35)",
    ],
    "icache_bound": [
        "精简内核代码，减少模板实例化数量 (P8)",
        "合并处理逻辑，减少条件分支",
        "减少 kernel launch 次数，合并小 kernel (P54)",
    ],
    "output_bound": [
        "批量 DataCopy 输出减少搬运次数 (P56)",
        "输出转置融合，消除额外搬运 (P59)",
        "考虑 in-place 写出或双缓冲输出 (P1)",
        "增大 tile_size，减少 MTE3 调用次数 (P2)",
    ],
    "balanced": [
        "当前流水线均衡，可尝试增大 tile_size 提升整体吞吐",
        "考虑多核并行 (P4) 进一步提升性能",
        "检查 UB bank conflict (P65)，避免双 src 同 bank 下的停顿",
    ],
}

_CUBE_HINTS = {
    "memory_bound": [
        "增大 matmul 分块，减少数据搬运频率 (P2)",
        "启用 double buffer 缓解 MTE1/MTE2 延迟 (P1/P19)",
        "使用 MatmulImpl 高层 API 替代手写流水 (P46)",
        "多核场景试用 Matmul IBShare L1 共享 (P71)，避免重复搬运",
        "多核竞争下可用 L2 Cache Hint 禁用 (P52/P74)，缓解 cache 抖动",
    ],
    "compute_bound": [
        "检查是否有冗余 cube 调用，合并矩阵乘 (P13)",
        "调整 M/N/K 分块以最大化 MAC 利用率 (P2)",
        "大 K 轴切分多核累加 (Split-K, P72)",
        "对角分块调度优化 L2 命中 (P47)",
    ],
    "balanced": [
        "Cube 流水线相对均衡，尝试增大分块提升整体吞吐 (P2)",
        "可用 MatmulImpl 异步迭代接口隐藏搬运延迟 (P46/P63)",
    ],
}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _analyze_vector(row: dict):
    """Extract Vector (AIV) metrics and diagnose bottleneck."""
    mte2 = _safe_float(row, "aiv_mte2_ratio")
    vec  = _safe_float(row, "aiv_vec_ratio")
    mte3 = _safe_float(row, "aiv_mte3_ratio")
    sca  = _safe_float(row, "aiv_scalar_ratio")
    icm  = _safe_float(row, "aiv_icache_miss_rate")

    pipeline = {
        "mte2_pct":         round(mte2 * 100, 1),
        "vec_pct":          round(vec  * 100, 1),
        "mte3_pct":         round(mte3 * 100, 1),
        "scalar_pct":       round(sca  * 100, 1),
        "icache_miss_rate": round(icm, 4),
    }

    # Priority order: scalar > mte2 > vec > icache > mte3 > balanced
    if sca > 0.40:
        bn, desc, strats = (
            "scalar_bound",
            f"Scalar 占 {pipeline['scalar_pct']}%，大量 GetValue 或标量循环",
            ["P5", "P10", "P67", "P84"],
        )
    elif mte2 > 0.45:
        bn, desc, strats = (
            "memory_bound",
            f"MTE2 搬入占 {pipeline['mte2_pct']}%，带宽是主要瓶颈",
            ["P1", "P2", "P10", "P25", "P66"],
        )
    elif vec > 0.60:
        bn, desc, strats = (
            "compute_bound",
            f"Vector 计算占 {pipeline['vec_pct']}%，计算是主要瓶颈",
            ["P3", "P4", "P13", "P68", "P69", "P84"],
        )
    elif icm > 0.10:
        bn, desc, strats = (
            "icache_bound",
            f"ICache 缺失率 {icm:.1%}，指令缓存压力大",
            ["P8", "P54"],
        )
    elif mte3 > 0.35:
        bn, desc, strats = (
            "output_bound",
            f"MTE3 搬出占 {pipeline['mte3_pct']}%，输出带宽是瓶颈",
            ["P1", "P10", "P56", "P59"],
        )
    else:
        bn, desc, strats = (
            "balanced",
            "流水线相对均衡，无明显单一瓶颈",
            ["P4", "P7", "P65"],
        )

    return pipeline, bn, desc, strats, _VECTOR_HINTS.get(bn, [])


def _analyze_cube(row: dict):
    """Extract Cube (AIC) metrics and diagnose bottleneck."""
    mac  = _safe_float(row, "aic_mac_ratio")
    mte1 = _safe_float(row, "aic_mte1_ratio")
    mte2 = _safe_float(row, "aic_mte2_ratio")
    util = _safe_float(row, "cube_utilization(%)")

    pipeline = {
        "mac_pct":          round(mac  * 100, 1),
        "mte1_pct":         round(mte1 * 100, 1),
        "mte2_pct":         round(mte2 * 100, 1),
        "cube_utilization": round(util, 1),
    }

    if util < 50:
        bn, desc, strats = (
            "memory_bound",
            f"Cube 利用率 {util:.1f}% 偏低，内存搬运是瓶颈",
            ["P2", "P4", "P19", "P46", "P52", "P71"],
        )
    elif mac > 0.70:
        bn, desc, strats = (
            "compute_bound",
            f"MAC 占 {pipeline['mac_pct']}%，计算饱和",
            ["P13", "P2", "P47", "P72", "P78"],
        )
    else:
        bn, desc, strats = (
            "balanced",
            "Cube 流水线相对均衡",
            ["P46", "P63"],
        )

    return pipeline, bn, desc, strats, _CUBE_HINTS.get(bn, [])


def _one_liner(pipeline: dict, bn: str, strats: list, task_type: str) -> str:
    ss = ",".join(strats) if strats else "无"
    if task_type == "cube":
        return (
            f"MAC:{pipeline.get('mac_pct', 0)}% | "
            f"CubeUtil:{pipeline.get('cube_utilization', 0)}% | "
            f"瓶颈:{bn} → 推荐 {ss}"
        )
    else:
        return (
            f"MTE2:{pipeline.get('mte2_pct', 0)}% | "
            f"Vec:{pipeline.get('vec_pct', 0)}% | "
            f"MTE3:{pipeline.get('mte3_pct', 0)}% | "
            f"Scalar:{pipeline.get('scalar_pct', 0)}% | "
            f"瓶颈:{bn} → 推荐 {ss}"
        )


def analyze(csv_path: str, task_type: str) -> dict:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    csv_format = _detect_csv_format(rows)
    rep = _pick_representative(rows)
    if rep is None:
        raise ValueError("No valid AIV/AIC rows found after filtering AI_CPU tasks")

    dur = _safe_float_multi(rep, "Task Duration(us)", "Duration(us)")

    if csv_format == "kernel_details":
        # kernel_details.csv has no pipeline ratio columns — return limited analysis
        return {
            "status": "success",
            "csv_format": "kernel_details",
            "task_type": task_type,
            "task_duration_us": round(dur, 3),
            "pipeline": {},
            "bottleneck": "balanced",
            "bottleneck_description": "kernel_details.csv 无流水线占比数据，无法精确诊断瓶颈",
            "recommended_strategies": ["P4", "P7", "P65"],
            "optimization_hints": [
                "kernel_details.csv 不含流水线利用率数据，建议使用 op_summary_*.csv 获取更精确诊断",
                "可通过 msprof 指令级 profiling (run_deep_profiling.py) 获取详细流水线分析",
            ],
            "profiling_one_liner": f"Duration:{dur}us | CSV:kernel_details(无pipeline数据) → 建议深度profiling",
        }

    if task_type in ("vector", "cv-mix"):
        pipeline, bn, desc, strats, hints = _analyze_vector(rep)
    else:  # cube
        pipeline, bn, desc, strats, hints = _analyze_cube(rep)

    return {
        "status": "success",
        "csv_format": csv_format,
        "task_type": task_type,
        "task_duration_us": round(dur, 3),
        "pipeline": pipeline,
        "bottleneck": bn,
        "bottleneck_description": desc,
        "recommended_strategies": strats,
        "optimization_hints": hints,
        "profiling_one_liner": _one_liner(pipeline, bn, strats, task_type),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze AscendC msprof op_summary CSV for pipeline bottleneck diagnosis"
    )
    parser.add_argument(
        "profiling_dir", nargs="?", default=None,
        help="Profiling root directory (auto-searches for ModelNew op_summary CSV)",
    )
    parser.add_argument(
        "--op-summary", default=None,
        help="Direct path to an op_summary CSV file",
    )
    parser.add_argument(
        "--task-type", default="vector",
        choices=["vector", "cube", "cv-mix"],
        help="Task type for bottleneck analysis (default: vector)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: stdout)",
    )
    args = parser.parse_args()

    result: dict = {}
    try:
        if args.op_summary:
            csv_path = args.op_summary
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"op_summary CSV not found: {csv_path}")
        elif args.profiling_dir:
            csv_path = find_latest_csv(args.profiling_dir)
            if csv_path is None:
                raise FileNotFoundError(
                    f"No op_summary CSV found under: {args.profiling_dir}"
                )
        else:
            parser.error("Provide profiling_dir positional argument or --op-summary")

        result = analyze(csv_path, args.task_type)

    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "profiling_one_liner": "",
        }

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)

    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()
