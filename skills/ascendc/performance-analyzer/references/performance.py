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


def _get_input_groups(output_dir: Path):
    """从 output_dir 下的 .json 文件读取输入 cases。

    Returns:
        (input_groups, json_path, case_metas):
            input_groups: List[List[tensor / value]] —— 每个 case 的实参列表
            json_path: str —— 使用的 case 文件路径
            case_metas: List[dict] —— 每个 case 的元信息 {shape, dtype},
                取自该 case 的「主张量」(inputs 中第一个 type=tensor 的项),
                找不到主张量时 shape/dtype 取 "?",仅用于报告展示。
    """
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
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }

    input_groups = []
    case_metas = []
    for case in cases:
        inputs = case["inputs"]
        group = []
        meta_shape = "?"
        meta_dtype = "?"
        meta_filled = False
        for inp in inputs:
            if inp["type"] == "tensor":
                dtype = dtype_map.get(inp["dtype"], torch.float32)
                shape = inp["shape"]
                if not meta_filled:
                    meta_shape = list(shape)
                    meta_dtype = inp["dtype"]
                    meta_filled = True
                if dtype == torch.bool:
                    t = torch.randint(0, 2, shape, dtype=dtype)
                elif dtype.is_floating_point:
                    t = torch.randn(shape, dtype=dtype)
                else:
                    t = torch.randint(0, 10, shape, dtype=dtype)
                group.append(t)
            elif inp["type"] == "tensor_list":
                dtype = dtype_map.get(inp["dtype"], torch.float32)
                shapes = inp["shapes"]
                if not meta_filled and shapes:
                    meta_shape = list(shapes[0])
                    meta_dtype = inp["dtype"]
                    meta_filled = True
                tensors = []
                for shape in shapes:
                    if dtype == torch.bool:
                        t = torch.randint(0, 2, shape, dtype=dtype)
                    elif dtype.is_floating_point:
                        t = torch.randn(shape, dtype=dtype)
                    else:
                        t = torch.randint(0, 10, shape, dtype=dtype)
                    tensors.append(t)
                group.append(tensors)
            elif inp["type"] == "attr":
                if inp["dtype"] in ("float", "double"):
                    group.append(float(inp.get("value", 0.0)))
                elif inp["dtype"] in ("int", "int64", "int32"):
                    group.append(int(inp.get("value", 0)))
                else:
                    group.append(inp.get("value"))
            else:
                group.append(inp.get("value"))
        input_groups.append(group)
        case_metas.append({"shape": meta_shape, "dtype": meta_dtype})

    return input_groups, str(json_path), case_metas


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
    input_groups, json_path, case_metas = _get_input_groups(output_dir_path)

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
            meta = case_metas[idx] if idx < len(case_metas) else {"shape": "?", "dtype": "?"}
            report["reference"]["case_results"].append({
                "index": idx,
                "shape": meta["shape"],
                "dtype": meta["dtype"],
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
            meta = case_metas[idx] if idx < len(case_metas) else {"shape": "?", "dtype": "?"}
            report["ascendc"]["case_results"].append({
                "index": idx,
                "shape": meta["shape"],
                "dtype": meta["dtype"],
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
                "shape": ref_case.get("shape", "?"),
                "dtype": ref_case.get("dtype", "?"),
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


def _shape_str(shape) -> str:
    if isinstance(shape, (list, tuple)):
        return "[" + ", ".join(str(int(s)) for s in shape) + "]"
    return str(shape)


def _group_by_dtype(per_case: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """按 dtype 分组,稳定排序;preferred 排前。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for c in per_case:
        buckets.setdefault(str(c.get("dtype", "?")), []).append(c)
    preferred = ("float16", "bfloat16", "float32", "int8", "int16", "int32", "int64", "bool")
    out: List[Tuple[str, List[Dict[str, Any]]]] = []
    seen: set = set()
    for dt in preferred:
        if dt in buckets:
            out.append((dt, buckets[dt]))
            seen.add(dt)
    for dt in sorted(k for k in buckets if k not in seen):
        out.append((dt, buckets[dt]))
    return out


def _dtype_summary_rows(per_case: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for dtype, cases in _group_by_dtype(per_case):
        speedups = [c["speedup"] for c in cases if c.get("speedup") not in (None, float("inf"))]
        avg = statistics.mean(speedups) if speedups else None
        better = sum(1 for s in speedups if s > 1.0)
        worse = sum(1 for s in speedups if s < 1.0)
        rows.append({
            "dtype": dtype,
            "count": len(cases),
            "avg_speedup": avg,
            "ascendc_better": better,
            "reference_better": worse,
        })
    return rows


def _brief_analysis(report: dict) -> List[str]:
    """基于实际数据生成 ≥3 条简短分析。"""
    bullets: List[str] = []
    per_case = report.get("per_case_speedup", [])
    overall = report.get("overall_speedup")

    # 1. 整体趋势
    if overall is not None:
        if overall > 1.0:
            bullets.append(
                f"整体平均加速比 {overall:.2f}x(> 1),AscendC 实现整体快于参考实现。"
            )
        elif overall < 1.0:
            bullets.append(
                f"整体平均加速比 {overall:.2f}x(< 1),AscendC 实现整体慢于参考实现,需进一步优化。"
            )
        else:
            bullets.append(f"整体平均加速比 {overall:.2f}x,与参考实现持平。")
    else:
        bullets.append("整体加速比未能计算,可能存在参考或 AscendC 实现执行失败的情况。")

    # 2. dtype 差异
    dt_rows = _dtype_summary_rows(per_case)
    if len(dt_rows) >= 2:
        valid = [r for r in dt_rows if r["avg_speedup"] is not None]
        if valid:
            best = max(valid, key=lambda r: r["avg_speedup"])
            worst = min(valid, key=lambda r: r["avg_speedup"])
            if best["dtype"] != worst["dtype"]:
                bullets.append(
                    f"不同 dtype 表现差异:{best['dtype']} 平均 {best['avg_speedup']:.2f}x 最优,"
                    f"{worst['dtype']} 平均 {worst['avg_speedup']:.2f}x 相对劣势。"
                )
            else:
                bullets.append(
                    f"各 dtype 平均加速比相近,均约 {best['avg_speedup']:.2f}x 左右。"
                )
    elif len(dt_rows) == 1 and dt_rows[0]["avg_speedup"] is not None:
        bullets.append(
            f"全部用例采用 {dt_rows[0]['dtype']},平均加速比 {dt_rows[0]['avg_speedup']:.2f}x。"
        )

    # 3. shape 规模差异(按主张量元素数粗分大/小)
    sized_cases = []
    for c in per_case:
        sh = c.get("shape")
        if isinstance(sh, (list, tuple)) and sh:
            try:
                n = 1
                for v in sh:
                    n *= int(v)
                sized_cases.append((n, c["speedup"]))
            except Exception:
                continue
    if len(sized_cases) >= 2:
        sized_cases.sort(key=lambda p: p[0])
        half = len(sized_cases) // 2
        small_avg = statistics.mean([p[1] for p in sized_cases[:half]]) if half else None
        large_avg = statistics.mean([p[1] for p in sized_cases[half:]]) if len(sized_cases) - half else None
        if small_avg is not None and large_avg is not None:
            if abs(large_avg - small_avg) >= 0.1:
                trend = "大 shape 优势更明显" if large_avg > small_avg else "小 shape 表现更好"
                bullets.append(
                    f"shape 规模影响:小规模平均 {small_avg:.2f}x,大规模平均 {large_avg:.2f}x,{trend}。"
                )
            else:
                bullets.append(
                    f"不同 shape 规模下加速比稳定,小规模 {small_avg:.2f}x、大规模 {large_avg:.2f}x。"
                )

    # 4. 失败情况
    if not report["reference"]["ok"]:
        bullets.append(f"⚠ 参考实现执行失败:{report['reference']['error']}")
    if not report["ascendc"]["ok"]:
        bullets.append(f"⚠ AscendC 实现执行失败:{report['ascendc']['error']}")

    # 兜底:确保 ≥3 条
    while len(bullets) < 3:
        bullets.append("用例覆盖样本较少,建议补充更多 shape/dtype 组合后再下结论。")
        break

    return bullets


def _report_to_markdown(report: dict) -> str:
    """将性能报告转为 markdown 格式,便于写入 trace.md 或对话回显。

    结构(借鉴 ascendc-operator-performance-eval 外部 skill):
      ## Performance Analysis
      - 元信息
      ### 性能对比
      - 单表:Case | Shape | DType | 参考(ms) | AscendC(ms) | 加速比
      ### 全量汇总
      - 用例数 / 平均加速比 / 双方更优条数
      ### 按数据类型汇总
      - 分 dtype 的统计表
      ### 简短分析
      - ≥3 条结论
    """
    lines: List[str] = []
    lines.append("## Performance Analysis")
    lines.append("")
    lines.append(f"- **Operator**: {report['op']}")
    lines.append(f"- **Device**: {report['device']}")
    lines.append(f"- **Warmup**: {report['warmup']}")
    lines.append(f"- **Repeat**: {report['repeats']}")
    lines.append(f"- **Reference**: `{report['reference']['model_path']}` "
                 f"({'OK' if report['reference']['ok'] else 'ERROR'})")
    lines.append(f"- **AscendC**: `{report['ascendc']['model_path']}` "
                 f"({'OK' if report['ascendc']['ok'] else 'ERROR'})")
    if not report["reference"]["ok"]:
        lines.append(f"- Reference Error: `{report['reference']['error']}`")
    if not report["ascendc"]["ok"]:
        lines.append(f"- AscendC Error: `{report['ascendc']['error']}`")
    lines.append("")

    per_case = report.get("per_case_speedup", [])

    # 性能对比单表
    if per_case:
        lines.append("### 性能对比")
        lines.append("")
        lines.append("| Case | Shape | DType | 参考(ms) | AscendC(ms) | 加速比 |")
        lines.append("| ---- | ----- | ----- | -------- | ----------- | ------ |")
        for c in per_case:
            lines.append(
                f"| {c['index']} | {_shape_str(c.get('shape', '?'))} | {c.get('dtype', '?')} | "
                f"{c['reference_ms']:.4f} | {c['ascendc_ms']:.4f} | {c['speedup']:.3f}x |"
            )
        lines.append("")

        # 全量汇总
        valid_speedups = [c["speedup"] for c in per_case if c.get("speedup") not in (None, float("inf"))]
        avg = statistics.mean(valid_speedups) if valid_speedups else None
        ascendc_better = sum(1 for s in valid_speedups if s > 1.0)
        reference_better = sum(1 for s in valid_speedups if s < 1.0)
        lines.append("### 全量汇总")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("| ---- | -- |")
        lines.append(f"| 用例数 | {len(per_case)} |")
        if avg is not None:
            lines.append(f"| 平均加速比(>1 表示 AscendC 更快) | {avg:.3f}x |")
        else:
            lines.append("| 平均加速比 | n/a |")
        lines.append(f"| AscendC 更优(比值>1) | {ascendc_better} |")
        lines.append(f"| 参考更优(比值<1) | {reference_better} |")
        if report.get("overall_speedup") is not None:
            lines.append(f"| Overall speedup | {report['overall_speedup']:.3f}x |")
        lines.append("")

        # 按数据类型汇总
        dt_rows = _dtype_summary_rows(per_case)
        if dt_rows:
            lines.append("### 按数据类型汇总")
            lines.append("")
            lines.append("| DType | 用例数 | 平均加速比 | AscendC 更优 | 参考更优 |")
            lines.append("| ----- | ------ | ---------- | ------------ | -------- |")
            for r in dt_rows:
                avg_s = f"{r['avg_speedup']:.3f}x" if r["avg_speedup"] is not None else "n/a"
                lines.append(
                    f"| {r['dtype']} | {r['count']} | {avg_s} | "
                    f"{r['ascendc_better']} | {r['reference_better']} |"
                )
            lines.append("")

    # 简短分析
    lines.append("### 简短分析")
    lines.append("")
    for b in _brief_analysis(report):
        lines.append(f"- {b}")
    lines.append("")

    return "\n".join(lines)


def _display_console(report: dict) -> None:
    """在 stdout 直接 print 漂亮的 markdown 形式,供 agent 在对话中回显。"""
    print()
    print(_report_to_markdown(report))
    print()


def main():
    parser = argparse.ArgumentParser(description="AscendC 性能测试脚本（基于 torch_npu.profiler）")
    parser.add_argument("--output_dir", required=True, help="算子输出目录（包含 model.py, model_new_ascendc.py, .json）")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT, help="warmup 次数（默认 5）")
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="正式测试次数（默认 50）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--output", help="输出 JSON 报告文件路径")
    parser.add_argument("--markdown", help="输出 Markdown 报告文件路径（用于 trace.md）")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="关闭末尾的 markdown 形式对话回显（默认开启）",
    )
    args = parser.parse_args()

    report = run_performance(args.output_dir, args.warmup, args.repeats, args.seed)
    _print_report(report)

    if args.output:
        # 1. 检查路径是否已经是一个存在的目录
        if os.path.isdir(args.output):
            canonical_path = os.path.join(args.output, "performance.json")
            with open(canonical_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"JSON report saved to: {canonical_path} (alias)")
        else:
            # 3. 如果不是目录,按文件路径写入
            save_path = args.output
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nJSON report saved to: {save_path}")

    if args.markdown:
        md = _report_to_markdown(report)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Markdown report saved to: {args.markdown}")

    # 末尾默认在 stdout 输出 markdown 形式的报告,供 agent 在对话中直接回显
    if not args.no_display:
        _display_console(report)


if __name__ == "__main__":
    main()
