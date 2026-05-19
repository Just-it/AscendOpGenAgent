#!/usr/bin/env python3
"""AscendC 性能测试脚本 — 使用 torch_npu.profiler 测试算子性能表现。

参考 triton/kernel-verifier/scripts/benchmark.py 的 profiler 测性能方式，
支持解析 operator_details.csv 获取 device 侧算子级时延，并附带 time.perf_counter 兜底机制。
"""

import argparse
import copy
import importlib.util
import inspect
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50


# ============================================================================
# 模型加载与输入解析
# ============================================================================

def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class(module, preferred_name: str):
    candidate = getattr(module, preferred_name, None)
    if inspect.isclass(candidate) and issubclass(candidate, nn.Module):
        return candidate
    for _, value in vars(module).items():
        if inspect.isclass(value) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise AttributeError(f"No nn.Module subclass found in {module.__file__}")


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _extract_scalar_from_json(inp: dict):
    """从 JSON scalar/attr 描述中提取具体值（兼容 range_values 回退）。"""
    val = inp.get("value")
    if val is not None:
        return val
    rv = inp.get("range_values")
    if isinstance(rv, (int, float, bool, str)):
        return rv
    if isinstance(rv, list) and len(rv) > 0:
        return rv[0]
    dtype = inp.get("dtype", "")
    if dtype == "bool":
        return True
    if dtype.startswith("int") or dtype.startswith("uint"):
        return 1
    return 1.0


def _get_input_groups_from_json(output_dir: Path):
    """从 output_dir 下的 .json 文件读取输入 cases。"""
    json_files = sorted(output_dir.glob("*.json"))
    json_path = None
    for f in json_files:
        if not f.name.endswith("_all_case.json") and not f.name.endswith(".json.bak"):
            json_path = f
            break
    if json_path is None:
        raise FileNotFoundError(f"No suitable JSON case file found in {output_dir}")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "uint16": torch.uint16,
        "uint32": torch.uint32,
        "uint64": torch.uint64,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        group = []
        for inp in inputs:
            if inp["type"] == "tensor":
                dtype = dtype_map.get(inp["dtype"], torch.float32)
                shape = inp["shape"]
                if dtype == torch.bool:
                    t = torch.randint(0, 2, shape, dtype=dtype)
                elif dtype.is_floating_point:
                    t = torch.randn(shape, dtype=dtype)
                elif str(inp.get("dtype", "")).startswith("uint"):
                    t = torch.randint(0, 10, shape, dtype=torch.int64).to(dtype)
                elif str(inp.get("dtype", "")).startswith("int"):
                    t = torch.randint(-10, 10, shape, dtype=dtype)
                elif str(inp.get("dtype", "")).startswith("complex"):
                    t = torch.randn(shape, dtype=dtype)
                else:
                    t = torch.randn(shape, dtype=dtype)
                group.append(t)
            elif inp["type"] in ("attr", "scalar"):
                val = _extract_scalar_from_json(inp)
                dtype = inp.get("dtype", "")
                if dtype == "bool":
                    group.append(bool(val))
                elif dtype in ("float", "double", "fp32", "fp64", "float32", "float64"):
                    group.append(float(val))
                elif dtype in ("int", "int64", "int32", "int16", "int8", "uint8", "uint16", "uint32", "uint64"):
                    group.append(int(val))
                elif str(dtype).startswith("complex"):
                    group.append(complex(val))
                else:
                    group.append(val)
            else:
                group.append(inp.get("value"))
        input_groups.append(group)

    return input_groups, str(json_path)


def _get_input_groups_from_module(module):
    """优先使用 model.py 自带的 get_input_groups / get_inputs 生成输入。"""
    if hasattr(module, "get_input_groups"):
        groups = module.get_input_groups()
        if isinstance(groups, list) and groups:
            return groups
    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if isinstance(inputs, list) and inputs:
            return [inputs]
    return None


def _load_impl(output_dir: Path, impl: str):
    if impl == "reference":
        module_path = output_dir / "model.py"
        preferred_class = "Model"
    elif impl == "ascendc":
        module_path = output_dir / "model_new_ascendc.py"
        preferred_class = "ModelNew"
    else:
        raise ValueError(f"Unsupported impl: {impl}")

    if not module_path.is_file():
        raise FileNotFoundError(f"missing {impl} model: {module_path}")

    module = _load_module(module_path, f"perf_{impl}_model")
    model_cls = _find_model_class(module, preferred_class)
    return module, model_cls, module_path


# ============================================================================
# 性能分析逻辑（参考 triton benchmark.py）
# ============================================================================

def _find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _cleanup_profile_path(profile_path: str) -> None:
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path, ignore_errors=True)


def _parse_operator_latency(profile_path: str, active_count: int) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """从 profiling 结果文件中提取算子时延数据。"""
    try:
        import pandas as pd
    except ImportError:
        _cleanup_profile_path(profile_path)
        return None, None

    operator_details_file = _find_profile_file(profile_path, "operator_details.csv")
    if not operator_details_file or not os.path.exists(operator_details_file):
        _cleanup_profile_path(profile_path)
        return None, None

    try:
        df = pd.read_csv(operator_details_file)
    except Exception:
        _cleanup_profile_path(profile_path)
        return None, None

    required_columns = ["Name", "Device Self Duration(us)"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        _cleanup_profile_path(profile_path)
        return None, None

    if "Count" not in df.columns:
        return _parse_without_count(df, profile_path, active_count)
    return _parse_with_count(df, profile_path, active_count)


def _parse_without_count(df: Any, profile_path: str, active_count: int) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    operator_avg_times = {}
    grouped = df.groupby("Name")["Device Self Duration(us)"].sum()
    for op_name_str, total_us in grouped.items():
        operator_avg_times[op_name_str] = total_us / active_count
    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1000.0
    _cleanup_profile_path(profile_path)
    return operator_avg_times, round(total_avg_ms, 4)


def _parse_with_count(df: Any, profile_path: str, active_count: int) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    valid_ops = df[df["Count"] == active_count].copy()
    if valid_ops.empty:
        _cleanup_profile_path(profile_path)
        return None, None

    operator_avg_times = {}
    grouped = valid_ops.groupby("Name")
    for op_name_str, group in grouped:
        total_us = group["Device Self Duration(us)"].sum()
        avg_us = total_us / active_count
        operator_avg_times[op_name_str] = avg_us

    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1000.0
    _cleanup_profile_path(profile_path)
    return operator_avg_times, round(total_avg_ms, 4)


def _run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str) -> str:
    """运行 NPU profiler 并返回生成的性能分析目录路径。"""
    import torch_npu

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=None,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )

    test_fn()
    torch.npu.synchronize()

    skip_first = 1 + warmup
    total_steps = skip_first + repeats

    timestamp = int(time.time() * 1000)
    profile_path = os.path.join(os.getcwd(), f"{profile_name}_{timestamp}")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=repeats, repeat=1, skip_first=skip_first
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total_steps):
            test_fn()
            prof.step()
            torch.npu.synchronize()

    return profile_path


def _measure_single_with_profiler(model, inputs, warmup: int, repeats: int, profile_name: str, device) -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """使用 torch_npu.profiler 测量单次性能。"""
    import torch_npu

    # warmup + 同步
    with torch.no_grad():
        _ = model(*inputs)
    torch.npu.synchronize()

    def test_fn():
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()

    try:
        profile_path = _run_profiler_with_config(test_fn, warmup, repeats, profile_name)
        operators, latency_ms = _parse_operator_latency(profile_path, repeats)
    except Exception as e:
        print(f"  torch_npu.profiler 获取数据失败: {e}，使用兜底测试机制...")
        operators, latency_ms = None, None

    if operators is None or latency_ms is None or latency_ms <= 0.0001:
        print(f"  警告: profiler 无法获取有效时延数据（当前:{latency_ms} ms），将使用 time.perf_counter() 兜底...")
        return _measure_single_fallback(model, inputs, warmup, repeats, device)

    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return operators, latency_ms, round(peak_memory, 2)


def _measure_single_fallback(model, inputs, warmup: int, repeats: int, device) -> Tuple[Dict[str, float], float, float]:
    """使用 time.perf_counter() 的兜底测试机制。"""
    import torch_npu

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    torch.npu.synchronize()

    latencies = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    avg_latency_ms = statistics.mean(latencies)
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return {}, round(avg_latency_ms, 4), round(peak_memory, 2)


# ============================================================================
# 主测试逻辑
# ============================================================================

def run_performance(output_dir: str, warmup: int = WARMUP_DEFAULT, repeats: int = REPEATS_DEFAULT, seed: int = 0):
    """对指定 output_dir 进行 reference vs ascendc 性能测试。

    Returns:
        dict: 包含每个 case 的 latency、operators、speedup 等。
    """
    output_dir_path = Path(output_dir).resolve()
    device = _get_device()

    # 加载 reference 和 ascendc 实现
    ref_module, ref_cls, ref_path = _load_impl(output_dir_path, "reference")
    asc_module, asc_cls, asc_path = _load_impl(output_dir_path, "ascendc")

    init_inputs = getattr(ref_module, "get_init_inputs", lambda: [])()

    # 优先使用 model.py 自带的输入生成函数，否则回退到 JSON 解析
    input_groups = _get_input_groups_from_module(ref_module)
    if input_groups is not None:
        json_files = sorted(output_dir_path.glob("*.json"))
        json_path = None
        for f in json_files:
            if not f.name.endswith("_all_case.json") and not f.name.endswith(".json.bak"):
                json_path = str(f)
                break
        if json_path is None:
            json_path = str(output_dir_path / f"{output_dir_path.name}.json")
    else:
        input_groups, json_path = _get_input_groups_from_json(output_dir_path)

    report = {
        "op": output_dir_path.name,
        "output_dir": str(output_dir_path),
        "json_path": json_path,
        "device": str(device),
        "warmup": warmup,
        "repeats": repeats,
        "seed": seed,
        "reference": {
            "model_path": str(ref_path),
            "case_results": [],
            "ok": False,
            "error": "",
        },
        "ascendc": {
            "model_path": str(asc_path),
            "case_results": [],
            "ok": False,
            "error": "",
        },
        "per_case_speedup": [],
        "overall_speedup": None,
    }

    # 测试 reference
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed(seed)
        ref_model = ref_cls(*_clone_value(init_inputs)).to(device).eval()

        for idx, inputs in enumerate(input_groups):
            model_inputs = _move_to_device(_clone_value(inputs), device)
            operators, latency_ms, peak_mem = _measure_single_with_profiler(
                ref_model, model_inputs, warmup, repeats, f"ref_profile_case{idx}", device
            )
            report["reference"]["case_results"].append({
                "index": idx,
                "latency_ms": latency_ms,
                "peak_memory_mb": peak_mem,
                "operators": operators or {},
            })
        report["reference"]["ok"] = True
    except Exception as exc:
        report["reference"]["error"] = f"{type(exc).__name__}: {exc}"
        import traceback
        traceback.print_exc()

    # 测试 ascendc
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed(seed)
        asc_model = asc_cls(*_clone_value(init_inputs)).to(device).eval()

        for idx, inputs in enumerate(input_groups):
            model_inputs = _move_to_device(_clone_value(inputs), device)
            operators, latency_ms, peak_mem = _measure_single_with_profiler(
                asc_model, model_inputs, warmup, repeats, f"asc_profile_case{idx}", device
            )
            report["ascendc"]["case_results"].append({
                "index": idx,
                "latency_ms": latency_ms,
                "peak_memory_mb": peak_mem,
                "operators": operators or {},
            })
        report["ascendc"]["ok"] = True
    except Exception as exc:
        report["ascendc"]["error"] = f"{type(exc).__name__}: {exc}"
        import traceback
        traceback.print_exc()

    # 计算加速比
    if report["reference"]["ok"] and report["ascendc"]["ok"]:
        speedups = []
        for ref_case, asc_case in zip(report["reference"]["case_results"], report["ascendc"]["case_results"]):
            ref_lat = ref_case["latency_ms"]
            asc_lat = asc_case["latency_ms"]
            speedup = ref_lat / asc_lat if asc_lat and asc_lat > 0 else float("inf")
            speedups.append(speedup)
            report["per_case_speedup"].append({
                "index": ref_case["index"],
                "reference_ms": ref_lat,
                "ascendc_ms": asc_lat,
                "speedup": round(speedup, 4),
            })
        report["overall_speedup"] = round(statistics.mean(speedups), 4) if speedups else None

    return report


def _print_report(report: dict):
    print("=" * 88)
    print("Performance Report (AscendC)")
    print("=" * 88)
    print(f"Operator    : {report['op']}")
    print(f"Output Dir  : {report['output_dir']}")
    print(f"JSON Path   : {report['json_path']}")
    print(f"Device      : {report['device']}")
    print(f"Warmup      : {report['warmup']}")
    print(f"Repeat      : {report['repeats']}")
    print(f"Seed        : {report['seed']}")
    print("-" * 88)

    # Impl summary
    for impl in ("reference", "ascendc"):
        r = report[impl]
        status = "OK" if r["ok"] else "ERROR"
        print(f"{impl:<12} {status:<8} {r['model_path']}")
        if not r["ok"]:
            print(f"  error: {r['error']}")

    # Per-case speedup
    if report["per_case_speedup"]:
        print("-" * 88)
        print("Per-Case Speedup (reference / ascendc)")
        print("-" * 88)
        print(f"{'Case':<8} {'Ref(ms)':>12} {'AscendC(ms)':>14} {'Speedup':>10}")
        print("-" * 88)
        for case in report["per_case_speedup"]:
            print(
                f"[{case['index']:<5}] {case['reference_ms']:>12.4f} "
                f"{case['ascendc_ms']:>14.4f} {case['speedup']:>10.2f}x"
            )
        print("-" * 88)
        print(f"Overall speedup: {report['overall_speedup']:.2f}x")
        print("=" * 88)


def _report_to_markdown(report: dict) -> str:
    """将性能报告转为 markdown 格式，便于写入 trace.md。"""
    lines = []
    lines.append("## Performance Analysis")
    lines.append("")
    lines.append(f"- **Operator**: {report['op']}")
    lines.append(f"- **Device**: {report['device']}")
    lines.append(f"- **Warmup**: {report['warmup']}")
    lines.append(f"- **Repeat**: {report['repeats']}")
    lines.append("")

    for impl in ("reference", "ascendc"):
        r = report[impl]
        status = "OK" if r["ok"] else "ERROR"
        lines.append(f"### {impl.capitalize()} ({status})")
        lines.append(f"- Model: `{r['model_path']}`")
        if not r["ok"]:
            lines.append(f"- Error: `{r['error']}`")
        else:
            for case in r["case_results"]:
                idx = case["index"]
                lat = case["latency_ms"]
                mem = case["peak_memory_mb"]
                lines.append(f"- case[{idx}]: latency={lat:.4f} ms, peak_memory={mem:.2f} MB")
        lines.append("")

    if report["per_case_speedup"]:
        lines.append("### Per-Case Speedup")
        lines.append("")
        lines.append("| case | reference (ms) | ascendc (ms) | speedup |")
        lines.append("|------|---------------|-------------|---------|")
        for case in report["per_case_speedup"]:
            lines.append(
                f"| {case['index']} | {case['reference_ms']:.4f} | "
                f"{case['ascendc_ms']:.4f} | {case['speedup']:.2f}x |"
            )
        lines.append("")
        lines.append(f"**Overall speedup**: {report['overall_speedup']:.2f}x")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AscendC 性能测试脚本（基于 torch_npu.profiler）")
    parser.add_argument("--output_dir", required=True, help="算子输出目录（包含 model.py, model_new_ascendc.py, .json）")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT, help="warmup 次数（默认 5）")
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="正式测试次数（默认 50）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--output", help="输出 JSON 报告文件路径")
    parser.add_argument("--markdown", help="输出 Markdown 报告文件路径（用于 trace.md）")
    args = parser.parse_args()

    report = run_performance(args.output_dir, args.warmup, args.repeats, args.seed)
    _print_report(report)
    
    if args.output:
        # 1. 检查路径是否已经是一个存在的目录
        if os.path.isdir(args.output):
            # 2. 如果是目录，自动拼接一个默认的文件名 (例如 report.json)
            save_path = os.path.join(args.output, "preformance.json")
            print(f"提示: 检测到输出路径为目录，将自动保存为: {save_path}")
        else:
            # 3. 如果不是目录，则按原样处理（视为文件路径）
            # 注意：这里也可以增加检查父目录是否存在的逻辑，防止路径错误
            save_path = args.output

        # 4. 确保父目录存在（防止因为文件夹没创建而报错）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 5. 安全地写入文件
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved to: {save_path}")    
    

    if args.markdown:
        md = _report_to_markdown(report)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Markdown report saved to: {args.markdown}")


if __name__ == "__main__":
    main()
