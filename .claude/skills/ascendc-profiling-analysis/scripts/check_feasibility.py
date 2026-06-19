#!/usr/bin/env python3
"""
T5: check_feasibility.py — 优化可行性硬件约束检查

根据优化类型计算 UB 内存需求，对比 hw_params 约束，检查 32B 对齐。

用法:
    python3 check_feasibility.py --type add_double_buffer \
        --params '{"tile":4096,"dtype":"fp16"}' \
        [--hw-params <path_to_hw_params.json>]
"""

import argparse
import json
import math
import os
import sys

# 数据类型字节数
DTYPE_BYTES = {
    "fp16": 2, "half": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "fp32": 4, "float": 4, "float32": 4,
    "fp64": 8, "double": 8, "float64": 8,
    "int8": 1, "uint8": 1,
    "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4,
    "int64": 8, "uint64": 8,
}

# 默认硬件参数 (910B3)
DEFAULT_HW_PARAMS = {
    "chip_model": "910B3",
    "ub_size_bytes": 196608,  # 192KB
    "core_num": 40,
    "alignment_bytes": 32,
}


def _align_up(size: int, alignment: int) -> int:
    """向上对齐"""
    return ((size + alignment - 1) // alignment) * alignment


def check_double_buffer(params: dict, hw: dict) -> dict:
    """检查双缓冲可行性"""
    tile = params.get("tile", 4096)
    dtype = params.get("dtype", "fp16")
    pipe_count = params.get("pipe_count", 2)  # CopyIn + CopyOut
    buffer_num = 2  # 双缓冲

    elem_bytes = DTYPE_BYTES.get(dtype, 2)
    alignment = hw.get("alignment_bytes", 32)
    ub_size = hw.get("ub_size_bytes", 196608)

    # 每个 buffer: tile * elem_bytes, 对齐到 32B
    single_buf = _align_up(tile * elem_bytes, alignment)
    # 双缓冲 × pipe 数
    total_required = single_buf * buffer_num * pipe_count

    feasible = total_required <= ub_size
    utilization = total_required / ub_size * 100

    result = {
        "optimization": "add_double_buffer",
        "feasible": feasible,
        "ub_required": total_required,
        "ub_available": ub_size,
        "ub_utilization_pct": round(utilization, 1),
        "breakdown": {
            "single_buffer_bytes": single_buf,
            "buffer_num": buffer_num,
            "pipe_count": pipe_count,
            "total_buffers": buffer_num * pipe_count,
        },
    }

    if not feasible:
        # 计算最大可行 tile
        max_buf = ub_size // (buffer_num * pipe_count)
        max_tile = (max_buf // alignment) * alignment // elem_bytes
        result["max_feasible_tile"] = max_tile
        result["suggestion"] = f"Reduce tile to {max_tile} elements or fewer"

    return result


def check_increase_tile(params: dict, hw: dict) -> dict:
    """检查增大 tile 可行性"""
    current_tile = params.get("current_tile", 2048)
    target_tile = params.get("target_tile", 4096)
    dtype = params.get("dtype", "fp16")
    buffer_num = params.get("buffer_num", 1)
    pipe_count = params.get("pipe_count", 2)

    elem_bytes = DTYPE_BYTES.get(dtype, 2)
    alignment = hw.get("alignment_bytes", 32)
    ub_size = hw.get("ub_size_bytes", 196608)

    current_buf = _align_up(current_tile * elem_bytes, alignment)
    target_buf = _align_up(target_tile * elem_bytes, alignment)

    current_total = current_buf * buffer_num * pipe_count
    target_total = target_buf * buffer_num * pipe_count

    feasible = target_total <= ub_size

    return {
        "optimization": "increase_tile",
        "feasible": feasible,
        "current_ub_usage": current_total,
        "target_ub_usage": target_total,
        "ub_available": ub_size,
        "ub_utilization_pct": round(target_total / ub_size * 100, 1),
        "tile_increase_ratio": round(target_tile / current_tile, 2),
    }


def check_add_intermediate_buffer(params: dict, hw: dict) -> dict:
    """检查增加中间 buffer (如 FP32 中间计算) 可行性"""
    tile = params.get("tile", 4096)
    input_dtype = params.get("input_dtype", "fp16")
    intermediate_dtype = params.get("intermediate_dtype", "fp32")
    buffer_num = params.get("buffer_num", 1)
    existing_usage = params.get("existing_ub_usage", 0)

    input_bytes = DTYPE_BYTES.get(input_dtype, 2)
    inter_bytes = DTYPE_BYTES.get(intermediate_dtype, 4)
    alignment = hw.get("alignment_bytes", 32)
    ub_size = hw.get("ub_size_bytes", 196608)

    inter_buf = _align_up(tile * inter_bytes, alignment) * buffer_num
    total = existing_usage + inter_buf

    feasible = total <= ub_size

    return {
        "optimization": "add_intermediate_buffer",
        "feasible": feasible,
        "intermediate_buffer_bytes": inter_buf,
        "existing_usage": existing_usage,
        "total_required": total,
        "ub_available": ub_size,
        "ub_utilization_pct": round(total / ub_size * 100, 1),
    }


def check_alignment(params: dict, hw: dict) -> dict:
    """检查数据对齐"""
    tile = params.get("tile", 4096)
    dtype = params.get("dtype", "fp16")

    elem_bytes = DTYPE_BYTES.get(dtype, 2)
    alignment = hw.get("alignment_bytes", 32)

    total_bytes = tile * elem_bytes
    is_aligned = (total_bytes % alignment) == 0
    aligned_tile = _align_up(total_bytes, alignment) // elem_bytes
    waste = aligned_tile - tile

    return {
        "optimization": "check_alignment",
        "is_aligned": is_aligned,
        "tile_elements": tile,
        "tile_bytes": total_bytes,
        "alignment_bytes": alignment,
        "aligned_tile_elements": aligned_tile,
        "wasted_elements": waste,
    }


# 优化类型 → 检查函数
OPTIMIZATION_CHECKS = {
    "add_double_buffer": check_double_buffer,
    "increase_tile": check_increase_tile,
    "add_intermediate_buffer": check_add_intermediate_buffer,
    "check_alignment": check_alignment,
}


def check_feasibility(opt_type: str, params: dict, hw_params: dict = None) -> dict:
    """
    检查优化方案的硬件可行性。

    Args:
        opt_type: 优化类型 (add_double_buffer, increase_tile, etc.)
        params: 优化参数
        hw_params: 硬件参数 (None 则使用默认 910B3)

    Returns:
        可行性检查结果 dict
    """
    hw = hw_params or DEFAULT_HW_PARAMS

    check_fn = OPTIMIZATION_CHECKS.get(opt_type)
    if check_fn is None:
        return {
            "tool": "T5",
            "error": f"Unknown optimization type: {opt_type}. "
                     f"Available: {list(OPTIMIZATION_CHECKS.keys())}",
        }

    result = check_fn(params, hw)
    result["tool"] = "T5"
    result["hw_params"] = {
        "chip_model": hw.get("chip_model", "unknown"),
        "ub_size_bytes": hw.get("ub_size_bytes"),
        "alignment_bytes": hw.get("alignment_bytes", 32),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="T5: Optimization feasibility check")
    parser.add_argument("--type", required=True, choices=list(OPTIMIZATION_CHECKS.keys()),
                        help="Optimization type to check")
    parser.add_argument("--params", required=True, help="JSON string of optimization parameters")
    parser.add_argument("--hw-params", help="Path to hw_params JSON file")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    params = json.loads(args.params)

    hw = None
    if args.hw_params:
        with open(args.hw_params) as f:
            hw = json.load(f)

    result = check_feasibility(args.type, params, hw)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"T5 result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
