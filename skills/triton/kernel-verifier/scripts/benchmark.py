#!/usr/bin/env python3
# 性能测试脚本 — 使用 torch_npu.profiler 测试生成算子的性能表现

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50
TRITON_IMPL_NAME_DEFAULT = "triton_ascend_impl"
ERROR_MSG_LIMIT = 2000


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class BenchmarkConfig:
    """性能测试配置"""
    op_name: str
    verify_dir: str
    triton_impl_name: str = TRITON_IMPL_NAME_DEFAULT
    warmup: int = WARMUP_DEFAULT
    repeats: int = REPEATS_DEFAULT
    skip_framework: bool = False
    framework_latency_ms: float = 0.0


@dataclass
class PerformanceResult:
    """单次性能测试结果"""
    avg_latency_ms: float
    peak_memory_mb: float
    operators: Dict[str, float]


@dataclass
class SingleShapeResult:
    """单个 shape 的性能测试结果。

    失败用例时 framework / implementation / speedup_vs_torch 为 None，
    status="fail" 且附带 error_type / error_msg。
    通过用例但 speedup 异常（NaN/Inf/0/负数）时 status="pass"，
    speedup_vs_torch 落盘为 null，case_idx 收集到 BenchmarkResult 的对应分类列表。
    """
    case_idx: int
    input_desc: List[Dict[str, Any]]
    status: str = "pass"           # "pass" | "fail"
    framework: Optional[PerformanceResult] = None
    implementation: Optional[PerformanceResult] = None
    speedup_vs_torch: Optional[float] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None


@dataclass
class BenchmarkResult:
    """完整性能测试结果。

    speedup_vs_torch: 各通过 shape 加速比的几何平均(不含异常 shape)。
    *_indices: 五类异常 shape 的 case_idx 列表(从 1 开始),与 passed_cases 不冲突,
               异常 shape 仍计入 passed_cases,只是 s_i 不进入几何平均。
    """
    op_name: str
    warmup: int
    repeats: int
    framework: Optional[PerformanceResult]
    implementation: Optional[PerformanceResult]
    speedup_vs_torch: Optional[float]
    total_cases: int = 1
    passed_cases: int = 0
    failed_cases: int = 0
    nan_indices: List[int] = field(default_factory=list)
    inf_indices: List[int] = field(default_factory=list)
    zero_indices: List[int] = field(default_factory=list)
    negative_indices: List[int] = field(default_factory=list)
    none_indices: List[int] = field(default_factory=list)
    per_shape_results: List[SingleShapeResult] = field(default_factory=list)


# ============================================================================
# 通用辅助函数
# ============================================================================

def truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    """截断过长错误信息：保留头尾各 limit/2 字符。"""
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def describe_input(inputs: List[Any]) -> List[Dict[str, Any]]:
    """将输入列表描述为结构化字段，便于写入 JSON。

    - torch.Tensor → {"type": "tensor", "shape": [...], "dtype": "..."}
    - 其他标量/对象 → {"type": "scalar", "value": repr(x)}
    """
    try:
        import torch
    except Exception:
        torch = None

    descs: List[Dict[str, Any]] = []
    for x in inputs:
        if torch is not None and isinstance(x, torch.Tensor):
            descs.append({
                "type": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            })
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs


def cleanup_npu_memory() -> None:
    """清理 NPU 显存，避免单个 shape 失败后连锁 OOM。"""
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.empty_cache()
    except Exception:
        pass
    gc.collect()


# ============================================================================
# 输入解析
# ============================================================================

def resolve_inputs(op_name: str, verify_dir: str):
    """解析任务文件的输入提供方式。

    支持两种格式：
        - get_inputs(): 旧格式，返回单组输入
        - get_input_groups(): 新格式，返回多组输入列表

    Returns:
        输入组列表 (List[List[Any]])
    """
    import torch  # noqa: F401
    sys.path.insert(0, verify_dir)
    torch_module = __import__(f"{op_name}_torch")

    if hasattr(torch_module, "get_input_groups"):
        return torch_module.get_input_groups()
    elif hasattr(torch_module, "get_inputs"):
        return [torch_module.get_inputs()]
    else:
        raise AttributeError(
            "模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def prepare_model_fn(model: Any, inputs: List[Any], device: Any) -> callable:
    """准备模型用于性能测试，返回测试函数"""
    import torch
    import torch_npu  # noqa: F401

    with torch.no_grad():
        _ = model(*inputs)
    torch.npu.synchronize()

    def test_fn():
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()

    return test_fn


def find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def cleanup_profile_path(profile_path: str) -> None:
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path, ignore_errors=True)


# ============================================================================
# 性能分析逻辑
# ============================================================================

def parse_operator_latency(profile_path: str, active_count: int) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """从 profiling 结果文件中提取算子时延数据，计算平均执行时间。"""
    import pandas as pd

    operator_details_file = find_profile_file(profile_path, "operator_details.csv")

    if not operator_details_file or not os.path.exists(operator_details_file):
        cleanup_profile_path(profile_path)
        return None, None

    try:
        df = pd.read_csv(operator_details_file)
    except Exception:
        cleanup_profile_path(profile_path)
        return None, None

    required_columns = ["Name", "Device Self Duration(us)"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        cleanup_profile_path(profile_path)
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

    cleanup_profile_path(profile_path)
    return operator_avg_times, round(total_avg_ms, 4)


def _parse_with_count(df: Any, profile_path: str, active_count: int) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    valid_ops = df[df["Count"] == active_count].copy()

    if valid_ops.empty:
        cleanup_profile_path(profile_path)
        return None, None

    operator_avg_times = {}
    grouped = valid_ops.groupby("Name")
    for op_name_str, group in grouped:
        total_us = group["Device Self Duration(us)"].sum()
        avg_us = total_us / active_count
        operator_avg_times[op_name_str] = avg_us

    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1000.0

    cleanup_profile_path(profile_path)
    return operator_avg_times, round(total_avg_ms, 4)


def run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str) -> str:
    """运行NPU profiler并返回生成的性能分析目录路径。"""
    import torch
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


def measure_single(
        model: Any,
        inputs: List[Any],
        warmup: int,
        repeats: int,
        profile_name: str,
        device: Any
) -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """测量单次性能（warmup + profiling）"""
    import torch
    import torch_npu  # noqa: F401

    torch.npu.reset_peak_memory_stats()
    test_fn = prepare_model_fn(model, inputs, device)

    try:
        profile_path = run_profiler_with_config(test_fn, warmup, repeats, profile_name)
        operators, latency_ms = parse_operator_latency(profile_path, repeats)
    except Exception as e:
        print(f"torch_npu.profiler 获取数据失败: {e}，使用兜底测试机制...")
        operators, latency_ms = None, None

    if operators is None or latency_ms is None or latency_ms <= 0.0001:
        print(f"警告: profiler 无法获取有效时延数据（当前:{latency_ms} ms），将使用 time.perf_counter() 进行兜底测试...")
        return measure_single_fallback(model, inputs, warmup, repeats, device)

    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return operators, latency_ms, round(peak_memory, 2)


def measure_single_fallback(
        model: Any,
        inputs: List[Any],
        warmup: int,
        repeats: int,
        device: Any
) -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """使用time.perf_counter()的兜底测试机制"""
    import torch
    import torch_npu  # noqa: F401
    import statistics

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
        latencies.append((end - start) * 1000)

    avg_latency_ms = statistics.mean(latencies)
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)
    return {}, round(avg_latency_ms, 4), round(peak_memory, 2)


# ============================================================================
# 主测试逻辑
# ============================================================================

def run_single_benchmark(
    framework_model: Any,
    impl_model: Any,
    inputs: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
    total_cases: int
) -> Tuple[PerformanceResult, PerformanceResult, float]:
    """对单组输入进行性能测试（支持跳过 framework 测试）。

    Returns:
        (framework_result, implementation_result, speedup)
    """
    import torch

    print(f"  测试第 {case_idx}/{total_cases} 组输入...")

    inputs_impl = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    if config.skip_framework:
        print(f"    跳过 Framework 测试，使用参考延迟: {config.framework_latency_ms:.4f} ms")
        framework_operators: Dict[str, float] = {}
        framework_latency_ms = config.framework_latency_ms
        framework_peak_memory = 0.0
    else:
        inputs_framework = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        print(f"    测试 Framework (warmup={config.warmup}, active={config.repeats})...")
        framework_operators, framework_latency_ms, framework_peak_memory = measure_single(
            framework_model, inputs_framework, config.warmup, config.repeats,
            f"framework_profile_case{case_idx}", device
        )

    print(f"    测试 Implementation (warmup={config.warmup}, active={config.repeats})...")
    impl_operators, impl_latency_ms, impl_peak_memory = measure_single(
        impl_model, inputs_impl, config.warmup, config.repeats,
        f"impl_profile_case{case_idx}", device
    )

    if (not config.skip_framework and framework_latency_ms is None) or impl_latency_ms is None:
        raise RuntimeError(
            f"[用例 {case_idx}/{total_cases}] 无法从 profiler 提取有效时延数据"
        )

    speedup = (
        framework_latency_ms / impl_latency_ms
        if impl_latency_ms > 0 and framework_latency_ms > 0
        else 0
    )

    return (
        PerformanceResult(
            avg_latency_ms=round(framework_latency_ms, 4),
            peak_memory_mb=round(framework_peak_memory, 2),
            operators=framework_operators or {}
        ),
        PerformanceResult(
            avg_latency_ms=round(impl_latency_ms, 4),
            peak_memory_mb=round(impl_peak_memory, 2),
            operators=impl_operators or {}
        ),
        round(speedup, 4),
    )


def classify_speedup(s: Any) -> str:
    """对单个 shape 的 speedup 值分类。

    判定优先级：none → nan → inf → negative → zero → valid

    Returns:
        "none" | "nan" | "inf" | "negative" | "zero" | "valid"
    """
    if s is None:
        return "none"
    if not isinstance(s, (int, float)):
        return "none"
    if math.isnan(s):
        return "nan"
    if math.isinf(s):
        return "inf"
    if s < 0:
        return "negative"
    if s == 0:
        return "zero"
    return "valid"


def compute_overall(
    results: List[SingleShapeResult],
) -> Tuple[Optional[PerformanceResult], Optional[PerformanceResult], Optional[float],
           List[int], List[int], List[int], List[int], List[int]]:
    """基于通过的 shape 做几何平均聚合。

    异常 shape（speedup 为 None/NaN/Inf/负数/0）不进入几何平均，
    但其 case_idx 收集到对应类别列表中，供报告展示。
    异常 shape 仍计入 passed_cases（算子功能正常，只是测不准）。

    Returns:
        (overall_framework, overall_implementation, overall_speedup_geomean,
         nan_indices, inf_indices, zero_indices, negative_indices, none_indices)
        全部 shape 均失败时，前三个返回值为 None，索引列表为空。
    """
    passed = [r for r in results if r.status == "pass" and r.framework and r.implementation]

    nan_indices: List[int] = []
    inf_indices: List[int] = []
    zero_indices: List[int] = []
    negative_indices: List[int] = []
    none_indices: List[int] = []
    valid_speedups: List[float] = []

    for r in passed:
        category = classify_speedup(r.speedup_vs_torch)
        if category == "valid":
            valid_speedups.append(r.speedup_vs_torch)
        elif category == "nan":
            nan_indices.append(r.case_idx)
        elif category == "inf":
            inf_indices.append(r.case_idx)
        elif category == "negative":
            negative_indices.append(r.case_idx)
        elif category == "zero":
            zero_indices.append(r.case_idx)
        else:  # "none"
            none_indices.append(r.case_idx)

    if not passed:
        return (None, None, None,
                nan_indices, inf_indices, zero_indices, negative_indices, none_indices)

    n = len(passed)
    sum_fw = sum(r.framework.avg_latency_ms for r in passed)
    sum_impl = sum(r.implementation.avg_latency_ms for r in passed)

    # 兼容字段：avg_latency_ms = 各 shape 延时算术平均
    avg_fw = sum_fw / n
    avg_impl = sum_impl / n

    avg_fw_mem = sum(r.framework.peak_memory_mb for r in passed) / n
    avg_impl_mem = sum(r.implementation.peak_memory_mb for r in passed) / n

    fw_ops: Dict[str, float] = {}
    impl_ops: Dict[str, float] = {}
    for r in passed:
        for op, t in r.framework.operators.items():
            fw_ops[op] = fw_ops.get(op, 0) + t
        for op, t in r.implementation.operators.items():
            impl_ops[op] = impl_ops.get(op, 0) + t

    # 几何平均加速比（对数域，数值稳定），无有效 shape 则为 None
    if valid_speedups:
        overall_speedup: Optional[float] = round(
            math.exp(sum(math.log(s) for s in valid_speedups) / len(valid_speedups)),
            4,
        )
    else:
        overall_speedup = None

    return (
        PerformanceResult(
            avg_latency_ms=round(avg_fw, 4),
            peak_memory_mb=round(avg_fw_mem, 2),
            operators={k: round(v / n, 4) for k, v in fw_ops.items()},
        ),
        PerformanceResult(
            avg_latency_ms=round(avg_impl, 4),
            peak_memory_mb=round(avg_impl_mem, 2),
            operators={k: round(v / n, 4) for k, v in impl_ops.items()},
        ),
        overall_speedup,
        nan_indices,
        inf_indices,
        zero_indices,
        negative_indices,
        none_indices,
    )


def benchmark_implementations(config: BenchmarkConfig) -> BenchmarkResult:
    """执行完整的性能测试，支持多组输入。每个 shape 独立 try/except。"""
    import torch
    import torch_npu  # noqa: F401

    device = torch.device("npu")

    input_groups = resolve_inputs(config.op_name, config.verify_dir)
    total_cases = len(input_groups)

    sys.path.insert(0, config.verify_dir)
    torch_module = __import__(f"{config.op_name}_torch")
    impl_module = __import__(f"{config.op_name}_{config.triton_impl_name}")

    FrameworkModel = torch_module.Model
    ModelNew = impl_module.ModelNew
    get_init_inputs = torch_module.get_init_inputs

    per_shape_results: List[SingleShapeResult] = []

    for case_idx, inputs in enumerate(input_groups, start=1):
        input_desc = describe_input(inputs)
        try:
            init_params = get_init_inputs()
            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            framework_model = FrameworkModel(*init_params).to(device)

            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            impl_model = ModelNew(*init_params).to(device)

            fw_perf, impl_perf, speedup = run_single_benchmark(
                framework_model, impl_model, inputs, config, device, case_idx, total_cases
            )
            per_shape_results.append(SingleShapeResult(
                case_idx=case_idx,
                input_desc=input_desc,
                status="pass",
                framework=fw_perf,
                implementation=impl_perf,
                speedup_vs_torch=speedup,
            ))
        except Exception as e:
            err_detail = traceback.format_exc()
            print(f"  [用例 {case_idx}/{total_cases}] 失败: {type(e).__name__}: {e}", file=sys.stderr)
            per_shape_results.append(SingleShapeResult(
                case_idx=case_idx,
                input_desc=input_desc,
                status="fail",
                error_type=type(e).__name__,
                error_msg=truncate_error(err_detail),
            ))
        finally:
            # 清理显存，避免连锁 OOM
            try:
                del framework_model  # noqa: F821
            except Exception:
                pass
            try:
                del impl_model  # noqa: F821
            except Exception:
                pass
            cleanup_npu_memory()

    passed_cases = sum(1 for r in per_shape_results if r.status == "pass")
    failed_cases = total_cases - passed_cases

    overall_fw, overall_impl, overall_speedup, \
        nan_idxs, inf_idxs, zero_idxs, neg_idxs, none_idxs = compute_overall(per_shape_results)

    return BenchmarkResult(
        op_name=config.op_name,
        warmup=config.warmup,
        repeats=config.repeats,
        framework=overall_fw,
        implementation=overall_impl,
        speedup_vs_torch=overall_speedup,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        nan_indices=nan_idxs,
        inf_indices=inf_idxs,
        zero_indices=zero_idxs,
        negative_indices=neg_idxs,
        none_indices=none_idxs,
        per_shape_results=per_shape_results,
    )


def _perf_to_dict(p: Optional[PerformanceResult]) -> Optional[Dict[str, Any]]:
    if p is None:
        return None
    return {
        "avg_latency_ms": p.avg_latency_ms,
        "peak_memory_mb": p.peak_memory_mb,
        "operators": {name: round(avg_us, 4) for name, avg_us in p.operators.items()},
    }


def _normalize_shape_speedup(s: Optional[float]) -> Optional[float]:
    """落盘前规范单 shape speedup：异常值统一写为 None（JSON 中即 null）。"""
    return s if classify_speedup(s) == "valid" else None


def result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    """将 BenchmarkResult 转换为字典格式。"""
    base_dict: Dict[str, Any] = {
        "op_name": result.op_name,
        "warmup": result.warmup,
        "repeats": result.repeats,
        "total_cases": result.total_cases,
        "passed_cases": result.passed_cases,
        "failed_cases": result.failed_cases,
        "nan_indices": result.nan_indices,
        "inf_indices": result.inf_indices,
        "zero_indices": result.zero_indices,
        "negative_indices": result.negative_indices,
        "none_indices": result.none_indices,
        "framework": _perf_to_dict(result.framework),
        "implementation": _perf_to_dict(result.implementation),
        "speedup_vs_torch": result.speedup_vs_torch,
    }

    # per_shape_results 保留全量（含失败用例），带 status 列；
    # 异常 speedup（NaN/Inf/0/负数/None）落盘为 null
    base_dict["per_shape_results"] = [
        {
            "case_idx": r.case_idx,
            "input_desc": r.input_desc,
            "status": r.status,
            "framework": (
                {
                    "avg_latency_ms": r.framework.avg_latency_ms,
                    "peak_memory_mb": r.framework.peak_memory_mb,
                } if r.framework else None
            ),
            "implementation": (
                {
                    "avg_latency_ms": r.implementation.avg_latency_ms,
                    "peak_memory_mb": r.implementation.peak_memory_mb,
                } if r.implementation else None
            ),
            "speedup_vs_torch": _normalize_shape_speedup(r.speedup_vs_torch),
            "error_type": r.error_type,
            "error_msg": r.error_msg,
        }
        for r in result.per_shape_results
    ]

    return base_dict


# ============================================================================
# 命令行入口
# ============================================================================

VERIFY_GATE_FAILURES_TO_PRINT = 5


def resolve_verify_json_name(triton_impl_name: str) -> str:
    """按 impl_name 推导 verify_result json 文件名。

    - triton_ascend_impl（默认）→ verify_result.json（Phase 3）
    - triton_baseline / triton_optimized → verify_result_{suffix}.json（Phase 4）
    - 其他自定义名 → verify_result_{name 去掉 triton_ 前缀}.json
    """
    if triton_impl_name == TRITON_IMPL_NAME_DEFAULT:
        return "verify_result.json"
    suffix = triton_impl_name
    if suffix.startswith("triton_"):
        suffix = suffix[len("triton_"):]
    return f"verify_result_{suffix}.json"


def check_verify_gate(verify_dir: str, triton_impl_name: str) -> None:
    """L1 闸门：benchmark 启动前必须确认对应 verify_result 全过。

    不通过时直接 exit 2，stderr 打印路径 / 计数 / failures 摘要，
    便于上游 agent 把错误等价映射到 verify 失败处理路径。
    """
    verify_json_name = resolve_verify_json_name(triton_impl_name)
    verify_json_path = os.path.join(verify_dir, verify_json_name)

    if not os.path.isfile(verify_json_path):
        print(
            f"[L1 闸门] 拒绝执行 benchmark：未找到 verify_result 文件\n"
            f"  expected: {verify_json_path}\n"
            f"  triton_impl_name: {triton_impl_name}\n"
            f"  请先运行 verify.py，或在确实不需要精度校验的场景下传 --verify_not_required",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        with open(verify_json_path, "r", encoding="utf-8") as f:
            verify_data = json.load(f)
    except Exception as e:
        print(
            f"[L1 闸门] 拒绝执行 benchmark：verify_result 文件读取失败\n"
            f"  path: {verify_json_path}\n"
            f"  error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(2)

    total = verify_data.get("total_cases", 0)
    passed = verify_data.get("passed_cases", 0)
    failures = verify_data.get("failures", []) or []

    if total == 0:
        print(
            f"[L1 闸门] 拒绝执行 benchmark：verify_result 中 total_cases=0\n"
            f"  path: {verify_json_path}\n"
            f"  说明 verify.py 未实际跑任何 shape，benchmark 无意义",
            file=sys.stderr,
        )
        sys.exit(2)

    if passed != total:
        print(
            f"[L1 闸门] 拒绝执行 benchmark：精度校验未全通过\n"
            f"  path: {verify_json_path}\n"
            f"  passed_cases: {passed}/{total}\n"
            f"  triton_impl_name: {triton_impl_name}",
            file=sys.stderr,
        )
        if failures:
            print(
                f"  前 {min(VERIFY_GATE_FAILURES_TO_PRINT, len(failures))} "
                f"条 failures（共 {len(failures)} 条）：",
                file=sys.stderr,
            )
            for f_item in failures[:VERIFY_GATE_FAILURES_TO_PRINT]:
                print(
                    f"    - case_idx={f_item.get('case_idx')} "
                    f"error_type={f_item.get('error_type')} "
                    f"input_desc={f_item.get('input_desc')}",
                    file=sys.stderr,
                )
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="性能测试脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--verify_dir", default=".", help="验证目录路径（默认当前目录）")
    parser.add_argument("--triton_impl_name", default=TRITON_IMPL_NAME_DEFAULT,
                       help="Triton 实现模块名")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT, help="warmup 次数（默认 5）")
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="正式测试次数（默认 50）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    parser.add_argument("--skip_framework", action="store_true",
                       help="跳过 framework 性能测试（GPU Kernel 模式使用）")
    parser.add_argument("--framework_latency_ms", type=float, default=0.0,
                       help="预设的 framework 参考延迟（毫秒），用于计算 speedup")
    parser.add_argument("--verify_not_required", action="store_true",
                       help="跳过 L1 verify 闸门（默认强制要求 verify_result 全过）")
    args = parser.parse_args()

    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)

    if args.verify_not_required:
        print(
            f"[L1 闸门] 已通过 --verify_not_required 跳过 verify 闸门检查 "
            f"(triton_impl_name={args.triton_impl_name})",
            file=sys.stderr,
        )
    else:
        check_verify_gate(verify_dir, args.triton_impl_name)

    config = BenchmarkConfig(
        op_name=args.op_name,
        verify_dir=verify_dir,
        triton_impl_name=args.triton_impl_name,
        warmup=args.warmup,
        repeats=args.repeats,
        skip_framework=args.skip_framework,
        framework_latency_ms=args.framework_latency_ms,
    )

    try:
        result = benchmark_implementations(config)
        result_dict = result_to_dict(result)

        print("\n性能测试结果:")
        print(f"  通过率: {result_dict['passed_cases']}/{result_dict['total_cases']}")
        if result_dict["speedup_vs_torch"] is not None:
            print(f"  框架实现 - 平均延迟: {result_dict['framework']['avg_latency_ms']:.4f} ms")
            print(f"  生成实现 - 平均延迟: {result_dict['implementation']['avg_latency_ms']:.4f} ms")
            print(f"  加速比 (几何平均): {result_dict['speedup_vs_torch']:.4f}x")
        else:
            print("  无可用加速比数据（全部 shape 失败或 speedup 异常）")

        excluded_total = (
            len(result_dict["nan_indices"]) + len(result_dict["inf_indices"])
            + len(result_dict["zero_indices"]) + len(result_dict["negative_indices"])
            + len(result_dict["none_indices"])
        )
        if excluded_total > 0:
            print(
                f"  异常 shape (不计入几何平均): "
                f"nan={result_dict['nan_indices']}, "
                f"inf={result_dict['inf_indices']}, "
                f"zero={result_dict['zero_indices']}, "
                f"neg={result_dict['negative_indices']}, "
                f"none={result_dict['none_indices']}"
            )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
        else:
            print("\n结果:")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))

        # 只要脚本正常跑完就 exit 0（由 Agent 读 JSON 判断）
        sys.exit(0)

    except Exception as e:
        print(f"性能测试失败: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
