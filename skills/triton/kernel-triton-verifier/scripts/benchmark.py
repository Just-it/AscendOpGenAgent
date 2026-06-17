#!/usr/bin/env python3
"""算子验证脚本 — 算子对应.pt文件中包含输入以及预期的输出, 相同输入下, 对比生成算子输出与预期输出的一致性。

用法:
    python benchmark.py --op_name <算子名> [--verify_dir <验证目录>]  [--output <输出路径>] [--device_id <所用设备id>]

前置条件（验证目录下需存在以下文件）:
    {op_name}.pt            — 包含输入，预期输出
    {op_name}.py            — 包含生成算子的主要逻辑
    vllm_gpu_perf.csv        — 包含 Triton-GPU 实现的执行耗时（基准数据）
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import importlib
import gc
import torch
import pandas as pd

from test_common import convert_tensor_with_device_type

# ============================================================================
# 配置常量
# ============================================================================

WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class BenchmarkConfig:
    """性能测试配置"""
    op_name: str
    verify_dir: str
    warmup: int = WARMUP_DEFAULT
    repeats: int = REPEATS_DEFAULT


@dataclass
class PerformanceResult:
    """单次性能测试结果"""
    avg_latency_ms: float
    peak_memory_mb: float
    operators: Dict[str, float]


@dataclass
class BenchmarkResult:
    """完整性能测试结果"""
    op_name: str
    warmup: int
    repeats: int
    triton_gpu: PerformanceResult
    triton_ascend: PerformanceResult
    speedup_vs_gpu: float


# ============================================================================
# 辅助函数
# ============================================================================

def load_models(op_name: str, verify_dir: str, device: Any):
    """加载框架实现和Triton实现模型"""
    import torch
    import torch_npu
    
    spec = importlib.util.spec_from_file_location(op_name, f"{verify_dir}/{op_name}.py")
    triton_npu_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(triton_npu_module)

    # 获取 kernel 函数
    triton_npu_func = getattr(triton_npu_module, op_name)

    data = torch.load(f"{verify_dir}/{op_name}.pt", map_location=torch.device('cpu'), weights_only=False)
    
    return triton_npu_func, data, device

def prepare_triton_fn(triton_func: Any, grid: List[int], input_data: dict) -> callable:
    # 执行warmup
    with torch.no_grad():
        _ = triton_func[grid](**input_data)
    torch.npu.synchronize()
    
    # 返回测试函数
    def test_fn():
        with torch.no_grad():
            _ = triton_func[grid](**input_data)
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
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False
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
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total_steps):
            test_fn()
            prof.step()
            torch.npu.synchronize()
    
    return profile_path


def measure_single(
        model: Any,
        grid, 
        inputs: List[Any],
        warmup: int,
        repeats: int,
        profile_name: str,
        device: Any
) -> Tuple[Optional[Dict[str, float]], Optional[float], float]:
    """测量单次性能（warmup + profiling）"""
    import torch
    import torch_npu
    print(f"measure_single", flush=True)
    # 重置峰值内存统计
    torch.npu.reset_peak_memory_stats()

    # 准备测试函数
    test_fn = prepare_triton_fn(model, grid, inputs)

    try:
        # 运行profiler
        profile_path = run_profiler_with_config(test_fn, warmup, repeats, profile_name)

        # 解析结果
        operators, latency_ms = parse_operator_latency(profile_path, repeats)
    except Exception as e:
        print(f"torch_npu.profiler 获取数据失败: {e}，使用兜底测试机制...")
        operators, latency_ms = None, None

    # 如果profiler获取不到数据，使用兜底机制
    if operators is None or latency_ms is None:
        print(f"警告: profiler 无法获取时延数据，将使用 time.perf_counter() 进行兜底测试...")
        return measure_single_fallback(model, grid, inputs, warmup, repeats, device)

    # 获取峰值内存
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)

    return operators, latency_ms, round(peak_memory, 2)


def measure_single_fallback(
        model: Any,
        grid, 
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
            _ = model[grid](*inputs)
    torch.npu.synchronize()

    # 正式测试
    latencies = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model[grid](*inputs)
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

def benchmark_implementations(config: BenchmarkConfig) -> BenchmarkResult:
    """执行完整的性能测试"""
    import torch
    import torch_npu
    
    device = torch.device("npu")
    
    # 加载模型和输入
    triton_npu_func, data, device = load_models(
        config.op_name, 
        config.verify_dir, 
        device
    )
    print(f"load_models", flush=True)
    # 将输入移到设备上
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    grid = data['grid']
    # 测试框架实现
    df = pd.read_csv(f"{config.verify_dir}/vllm_gpu_perf.csv")
    framework_latency_us = df.loc[df["Name"] == config.op_name, "Duration(us)"].item()
    framework_latency_ms = framework_latency_us / 1000.0
    framework_operators, framework_peak_memory = {},1
    # 测试生成实现
    print(f"执行 Implementation warmup 和 profiler (warmup={config.warmup}, active={config.repeats})...")
    impl_operators, impl_latency_ms, impl_peak_memory = measure_single(
        triton_npu_func, grid, input_data, config.warmup, config.repeats, "impl_profile", device
    )
    
    # 验证结果
    if framework_latency_ms is None or impl_latency_ms is None:
        raise RuntimeError("无法从 profiler 结果中提取有效的时延数据")
    
    # 计算加速比
    speedup = (
        framework_latency_ms / impl_latency_ms 
        if impl_latency_ms > 0 and framework_latency_ms > 0 
        else 0
    )
    
    # 构建结果
    return BenchmarkResult(
        op_name=config.op_name,
        warmup=config.warmup,
        repeats=config.repeats,
        triton_gpu=PerformanceResult(
            avg_latency_ms=round(framework_latency_ms, 4),
            peak_memory_mb=round(framework_peak_memory, 2),
            operators=framework_operators or {}
        ),
        triton_ascend=PerformanceResult(
            avg_latency_ms=round(impl_latency_ms, 4),
            peak_memory_mb=round(impl_peak_memory, 2),
            operators=impl_operators or {}
        ),
        speedup_vs_gpu=round(speedup, 2)
    )


def result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    """将BenchmarkResult转换为字典格式"""
    return {
        "op_name": result.op_name,
        "warmup": result.warmup,
        "repeats": result.repeats,
        "triton_gpu": {
            "avg_latency_ms": result.triton_gpu.avg_latency_ms,
            "peak_memory_mb": result.triton_gpu.peak_memory_mb,
            "operators": {name: round(avg_us, 4) for name, avg_us in result.triton_gpu.operators.items()}
        },
        "triton_ascend": {
            "avg_latency_ms": result.triton_ascend.avg_latency_ms,
            "peak_memory_mb": result.triton_ascend.peak_memory_mb,
            "operators": {name: round(avg_us, 4) for name, avg_us in result.triton_ascend.operators.items()}
        },
        "speedup_vs_gpu": result.speedup_vs_gpu
    }


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="性能测试脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--verify_dir", default=".", help="验证目录路径（默认当前目录）")
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT, help="warmup 次数（默认 5）")
    parser.add_argument("--repeats", type=int, default=REPEATS_DEFAULT, help="正式测试次数（默认 50）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    parser.add_argument(
        "--device_id", required=True, type=int, default=0, help="指定npu卡"
    )

    args = parser.parse_args()
    torch.npu.set_device(args.device_id)
    
    # 验证目录
    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 构建配置
    config = BenchmarkConfig(
        op_name=args.op_name,
        verify_dir=verify_dir,
        warmup=args.warmup,
        repeats=args.repeats
    )
    
    try:
        # 执行测试
        result = benchmark_implementations(config)
        result_dict = result_to_dict(result)
        
        # 输出结果
        print("\n性能测试结果:")
        print(f"  Triton-GPU 实现 - 平均延迟: {result_dict['triton_gpu']['avg_latency_ms']:.4f} ms")
        print(f"  Triton-Ascend 实现 - 平均延迟: {result_dict['triton_ascend']['avg_latency_ms']:.4f} ms")
        print(f"  加速比 (Ascend vs GPU): {result_dict['speedup_vs_gpu']:.2f}x")
        print(f"  Triton-Ascend 实现 - 峰值内存: {result_dict['triton_ascend']['peak_memory_mb']:.2f} MB")
        
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