#!/usr/bin/env python3
# 性能测试脚本 — 使用 torch_npu.profiler 测试生成算子的性能表现

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50
TRITON_IMPL_NAME_DEFAULT = "triton_ascend_impl"


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
    continue_on_error: bool = False


@dataclass
class PerformanceResult:
    """单次性能测试结果"""
    avg_latency_ms: float
    peak_memory_mb: float
    operators: Dict[str, float]


@dataclass
class SingleShapeResult:
    """单个形状的性能测试结果"""
    shape: List[int]  # 主要输入的形状
    framework: PerformanceResult
    implementation: PerformanceResult
    speedup_vs_torch: float


@dataclass
class BenchmarkResult:
    """完整性能测试结果"""
    op_name: str
    warmup: int
    repeats: int
    framework: PerformanceResult
    implementation: PerformanceResult
    speedup_vs_torch: float
    total_cases: int = 1
    successful_cases: int = 1
    failed_cases: int = 0
    failed_case_details: List[Dict[str, Any]] = None
    per_shape_results: List[SingleShapeResult] = None

    def __post_init__(self):
        if self.per_shape_results is None:
            self.per_shape_results = []
        if self.failed_case_details is None:
            self.failed_case_details = []


# ============================================================================
# 辅助函数
# ============================================================================

def resolve_inputs(op_name: str, verify_dir: str):
    """解析任务文件的输入提供方式。

    支持两种格式：
        - get_inputs(): 旧格式，返回单组输入
        - get_input_groups(): 新格式，返回多组输入列表

    Returns:
        输入组列表 (List[List[Any]])
    """
    import torch
    sys.path.insert(0, verify_dir)
    torch_module = __import__(f"{op_name}_torch")
    
    if hasattr(torch_module, "get_input_groups"):
        # 新格式：多组输入（如 26_GELU_.py 有多个 shape 用例）
        return torch_module.get_input_groups()
    elif hasattr(torch_module, "get_inputs"):
        # 旧格式：单组输入（封装为列表）
        return [torch_module.get_inputs()]
    else:
        raise AttributeError(
            "模块必须提供 get_inputs() 或 get_input_groups() 方法"
        )


def load_models(op_name: str, verify_dir: str, triton_impl_name: str, device: Any, inputs: List[Any]):
    """加载框架实现和Triton实现模型，并迁移输入到设备。"""
    import torch
    import torch_npu
    
    sys.path.insert(0, verify_dir)
    
    torch_module = __import__(f"{op_name}_torch")
    impl_module = __import__(f"{op_name}_{triton_impl_name}")
    
    FrameworkModel = torch_module.Model
    get_init_inputs = torch_module.get_init_inputs
    ModelNew = impl_module.ModelNew
    
    init_params = get_init_inputs()
    
    # 创建模型（确保权重一致）
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    framework_model = FrameworkModel(*init_params).to(device)
    
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    impl_model = ModelNew(*init_params).to(device)
    
    # 迁移输入到设备
    inputs_device = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
    
    return framework_model, impl_model, inputs_device


def prepare_model_fn(model: Any, inputs: List[Any], device: Any) -> callable:
    """准备模型用于性能测试，返回测试函数"""
    import torch
    import torch_npu
    
    # 执行warmup
    with torch.no_grad():
        _ = model(*inputs)
    torch.npu.synchronize()
    
    # 返回测试函数
    def test_fn():
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()
    
    return test_fn


def find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    """在profile目录中查找指定文件"""
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def cleanup_profile_path(profile_path: str) -> None:
    """清理profile目录"""
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
    """处理没有 Count 列的情况：按操作名称直接累加计算。"""
    # 按算子名称分组，累加所有测量周期的 Device Self Duration
    operator_avg_times = {}
    grouped = df.groupby("Name")["Device Self Duration(us)"].sum()
    for op_name_str, total_us in grouped.items():
        # 平均到每次运行（微秒）
        operator_avg_times[op_name_str] = total_us / active_count
    
    # 汇总所有算子的平均时间，得到完整的 device 侧执行时间
    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1000.0
    
    cleanup_profile_path(profile_path)
    
    return operator_avg_times, round(total_avg_ms, 4)


def _parse_with_count(df: Any, profile_path: str, active_count: int) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """解析有 Count 列的情况：按操作名称分组，累加 Self Duration，计算每次运行的平均时间。"""
    # 筛选出 Count 等于 active_count 的记录（即正式测试阶段的算子）
    valid_ops = df[df["Count"] == active_count].copy()
    
    if valid_ops.empty:
        cleanup_profile_path(profile_path)
        return None, None
    
    # 按算子名称分组，累加 Device Self Duration
    operator_avg_times = {}
    grouped = valid_ops.groupby("Name")
    for op_name_str, group in grouped:
        total_us = group["Device Self Duration(us)"].sum()
        avg_us = total_us / active_count
        # 存储单位为微秒（us）
        operator_avg_times[op_name_str] = avg_us
    
    # 汇总所有算子的 Self Duration，得到一次完整运行的 device 侧总时间
    total_avg_us = sum(operator_avg_times.values())
    # 转换为毫秒
    total_avg_ms = total_avg_us / 1000.0
    
    cleanup_profile_path(profile_path)
    
    return operator_avg_times, round(total_avg_ms, 4)


def run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str) -> str:
    """运行NPU profiler并返回生成的性能分析目录路径。"""
    import torch
    import torch_npu
    
    # 实验性配置
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=None,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    
    # 预热一次确保模型准备就绪
    test_fn()
    torch.npu.synchronize()
    
    skip_first = 1 + warmup
    total_steps = skip_first + repeats
    
    # 生成唯一的profile路径
    timestamp = int(time.time() * 1000)
    profile_path = os.path.join(os.getcwd(), f"{profile_name}_{timestamp}")
    
    # 创建profiler
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
    import torch_npu

    # 重置峰值内存统计
    torch.npu.reset_peak_memory_stats()

    # 准备测试函数
    test_fn = prepare_model_fn(model, inputs, device)

    try:
        # 运行profiler
        profile_path = run_profiler_with_config(test_fn, warmup, repeats, profile_name)

        # 解析结果
        operators, latency_ms = parse_operator_latency(profile_path, repeats)
    except Exception as e:
        print(f"torch_npu.profiler 获取数据失败: {e}，使用兜底测试机制...")
        operators, latency_ms = None, None

    # 如果profiler获取不到数据或时延为0/无效，使用兜底机制
    if operators is None or latency_ms is None or latency_ms <= 0.0001:
        print(f"警告: profiler 无法获取有效时延数据（当前:{latency_ms} ms），将使用 time.perf_counter() 进行兜底测试...")
        return measure_single_fallback(model, inputs, warmup, repeats, device)

    # 获取峰值内存
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
    import torch_npu
    import time
    import statistics

    # 执行warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    torch.npu.synchronize()

    # 正式测试
    latencies = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # 转换为毫秒

    # 计算平均时延
    avg_latency_ms = statistics.mean(latencies)

    # 获取峰值内存
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)

    # 兜底机制不获取算子级别的时延，返回空字典
    return {}, round(avg_latency_ms, 4), round(peak_memory, 2)


# ============================================================================
# 主测试逻辑
# ============================================================================

def get_main_shape(inputs: List[Any]) -> List[int]:
    """获取主要输入张量的形状。

    注意: 此函数依赖 torch，应在已导入 torch 的上下文中调用，
    或由 run_single_benchmark() 调用（该函数所在调用链会导入 torch）。
    """
    import torch  # 延迟导入：确保 torch 在此作用域可用
    for x in inputs:
        if isinstance(x, torch.Tensor):
            return list(x.shape)
    return []


def run_single_benchmark(
    framework_model: Any,
    impl_model: Any,
    inputs: List[Any],
    config: BenchmarkConfig,
    device: Any,
    case_idx: int,
    total_cases: int
) -> SingleShapeResult:
    """对单组输入进行性能测试（支持跳过 framework 测试）。

    注意: 此函数依赖 torch 和 torch_npu，应在已导入这些模块的上下文中调用。

    Args:
        framework_model: 参考实现模型
        impl_model: 生成的实现模型
        inputs: 输入张量/参数列表
        config: 测试配置
        device: NPU 设备
        case_idx: 当前用例序号（从1开始）
        total_cases: 总用例数
    Returns:
        SingleShapeResult: 包含形状和性能数据的结果
    """
    import torch  # 延迟导入：函数内导入避免顶层依赖

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
    
    # 测试生成实现
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
    
    return SingleShapeResult(
        shape=get_main_shape(inputs),
        framework=PerformanceResult(
            avg_latency_ms=round(framework_latency_ms, 4),
            peak_memory_mb=round(framework_peak_memory, 2),
            operators=framework_operators or {}
        ),
        implementation=PerformanceResult(
            avg_latency_ms=round(impl_latency_ms, 4),
            peak_memory_mb=round(impl_peak_memory, 2),
            operators=impl_operators or {}
        ),
        speedup_vs_torch=round(speedup, 4)
    )


def compute_overall_average(results: List[SingleShapeResult]) -> Tuple[PerformanceResult, PerformanceResult, float]:
    """计算多组结果的总体平均值。

    Args:
        results: SingleShapeResult 列表
    Returns:
        (avg_framework, avg_implementation, avg_speedup)
    """
    import statistics
    
    if not results:
        raise RuntimeError("没有有效的测试结果")
    
    if len(results) == 1:
        return results[0].framework, results[0].implementation, results[0].speedup_vs_torch
    
    # 计算平均值
    fw_latencies = [r.framework.avg_latency_ms for r in results]
    impl_latencies = [r.implementation.avg_latency_ms for r in results]
    fw_memories = [r.framework.peak_memory_mb for r in results]
    impl_memories = [r.implementation.peak_memory_mb for r in results]
    speedups = [r.speedup_vs_torch for r in results]
    
    # 合并算子时间
    fw_ops: Dict[str, float] = {}
    impl_ops: Dict[str, float] = {}
    for r in results:
        for op, t in r.framework.operators.items():
            fw_ops[op] = fw_ops.get(op, 0) + t
        for op, t in r.implementation.operators.items():
            impl_ops[op] = impl_ops.get(op, 0) + t
    
    n = len(results)
    
    return (
        PerformanceResult(
            avg_latency_ms=round(statistics.mean(fw_latencies), 4),
            peak_memory_mb=round(statistics.mean(fw_memories), 2),
            operators={k: round(v / n, 4) for k, v in fw_ops.items()}
        ),
        PerformanceResult(
            avg_latency_ms=round(statistics.mean(impl_latencies), 4),
            peak_memory_mb=round(statistics.mean(impl_memories), 2),
            operators={k: round(v / n, 4) for k, v in impl_ops.items()}
        ),
        round(statistics.mean(speedups), 4)
    )


def _collect_case_meta(inputs):
    """收集当前 case 的元信息（shapes + dtypes），供失败汇报使用。"""
    import torch
    shapes = []
    dtypes = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            shapes.append(list(x.shape))
            dtypes.append(str(x.dtype).replace("torch.", ""))
    return shapes, dtypes


def benchmark_implementations(config: BenchmarkConfig) -> BenchmarkResult:
    """执行完整的性能测试，支持多组输入。"""
    import torch
    import torch_npu

    device = torch.device("npu")

    # 解析输入（支持单组/多组格式）
    input_groups = resolve_inputs(config.op_name, config.verify_dir)
    total_cases = len(input_groups)

    # 加载模块创建函数引用
    sys.path.insert(0, config.verify_dir)
    torch_module = __import__(f"{config.op_name}_torch")
    impl_module = __import__(f"{config.op_name}_{config.triton_impl_name}")

    FrameworkModel = torch_module.Model
    ModelNew = impl_module.ModelNew
    get_init_inputs = torch_module.get_init_inputs

    # 对每组输入进行测试
    per_shape_results: List[SingleShapeResult] = []
    failed_case_details: List[Dict[str, Any]] = []

    for case_idx, inputs in enumerate(input_groups, start=1):
        shapes, dtypes = _collect_case_meta(inputs)
        try:
            # 创建模型（确保权重一致）
            init_params = get_init_inputs()
            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            framework_model = FrameworkModel(*init_params).to(device)

            torch.manual_seed(0)
            torch.npu.manual_seed(0)
            impl_model = ModelNew(*init_params).to(device)

            # 测试该组输入
            shape_result = run_single_benchmark(
                framework_model, impl_model, inputs, config, device, case_idx, total_cases
            )
            per_shape_results.append(shape_result)
        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 500:
                err_msg = err_msg[:500] + "...(truncated)"
            failed_case_details.append({
                "case_idx": case_idx,
                "shapes": shapes,
                "dtypes": dtypes,
                "error": err_msg,
            })
            print(
                f"  [用例 {case_idx}/{total_cases}] 性能测试失败: "
                f"dtypes={dtypes}, shapes={shapes}, error={err_msg[:200]}",
                file=sys.stderr,
            )
            if not config.continue_on_error:
                raise

    successful_cases = len(per_shape_results)
    failed_cases_num = len(failed_case_details)

    if successful_cases == 0:
        raise RuntimeError(
            f"所有 {total_cases} 组用例全部测试失败，无有效性能数据"
        )

    # 计算总体平均值（仅基于成功的 case）
    if successful_cases == 1:
        overall_fw = per_shape_results[0].framework
        overall_impl = per_shape_results[0].implementation
        overall_speedup = per_shape_results[0].speedup_vs_torch
    else:
        overall_fw, overall_impl, overall_speedup = compute_overall_average(per_shape_results)

    return BenchmarkResult(
        op_name=config.op_name,
        warmup=config.warmup,
        repeats=config.repeats,
        framework=overall_fw,
        implementation=overall_impl,
        speedup_vs_torch=overall_speedup,
        total_cases=total_cases,
        successful_cases=successful_cases,
        failed_cases=failed_cases_num,
        failed_case_details=failed_case_details,
        per_shape_results=per_shape_results,
    )


def result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    """将BenchmarkResult转换为字典格式，多组输入时包含每个shape的详情"""
    compile_pass_rate = (
        result.successful_cases / result.total_cases
        if result.total_cases > 0 else 0.0
    )
    base_dict = {
        "op_name": result.op_name,
        "warmup": result.warmup,
        "repeats": result.repeats,
        "total_cases": result.total_cases,
        "successful_cases": result.successful_cases,
        "failed_cases": result.failed_cases,
        "compile_pass_rate": round(compile_pass_rate, 4),
        "framework": {
            "avg_latency_ms": result.framework.avg_latency_ms,
            "peak_memory_mb": result.framework.peak_memory_mb,
            "operators": {name: round(avg_us, 4) for name, avg_us in result.framework.operators.items()}
        },
        "implementation": {
            "avg_latency_ms": result.implementation.avg_latency_ms,
            "peak_memory_mb": result.implementation.peak_memory_mb,
            "operators": {name: round(avg_us, 4) for name, avg_us in result.implementation.operators.items()}
        },
        "speedup_vs_torch": result.speedup_vs_torch,
        "failed_case_details": result.failed_case_details,
    }

    # 多组输入时，记录每组的详细结果
    if result.total_cases > 1 and result.per_shape_results:
        base_dict["per_shape_results"] = [
            {
                "shape": r.shape,
                "framework": {
                    "avg_latency_ms": r.framework.avg_latency_ms,
                    "peak_memory_mb": r.framework.peak_memory_mb,
                },
                "implementation": {
                    "avg_latency_ms": r.implementation.avg_latency_ms,
                    "peak_memory_mb": r.implementation.peak_memory_mb,
                },
                "speedup_vs_torch": r.speedup_vs_torch
            }
            for r in result.per_shape_results
        ]

    return base_dict


# ============================================================================
# 命令行入口
# ============================================================================

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
    parser.add_argument("--continue_on_error", action="store_true",
                       help="单个 case 失败时继续跑其余 case（终评阶段使用）")
    args = parser.parse_args()
    
    # 验证目录
    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 构建配置
    config = BenchmarkConfig(
        op_name=args.op_name,
        verify_dir=verify_dir,
        triton_impl_name=args.triton_impl_name,
        warmup=args.warmup,
        repeats=args.repeats,
        skip_framework=args.skip_framework,
        framework_latency_ms=args.framework_latency_ms,
        continue_on_error=args.continue_on_error,
    )
    
    try:
        # 执行测试
        result = benchmark_implementations(config)
        result_dict = result_to_dict(result)
        
        # 输出结果
        print("\n性能测试结果:")
        print(f"  框架实现 - 平均延迟: {result_dict['framework']['avg_latency_ms']:.4f} ms")
        print(f"  生成实现 - 平均延迟: {result_dict['implementation']['avg_latency_ms']:.4f} ms")
        print(f"  加速比: {result_dict['speedup_vs_torch']:.4f}x")
        print(f"  生成实现 - 峰值内存: {result_dict['implementation']['peak_memory_mb']:.2f} MB")
        
        # 保存到文件或输出
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
        else:
            print("\n结果:")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
        
        sys.exit(0)
    
    except Exception as e:
        print(f"性能测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()