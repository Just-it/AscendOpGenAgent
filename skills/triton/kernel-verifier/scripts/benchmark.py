#!/usr/bin/env python3
"""性能测试脚本 — 测试生成算子的性能表现。

用法:
    python benchmark.py --op_name <算子名> [--verify_dir <目录>] 
                       [--warmup 5] [--repeats 50] [--output <路径>]

指标:
    - avg_latency_ms: 平均延迟
    - p50_latency_ms: P50 延迟  
    - p99_latency_ms: P99 延迟
    - peak_memory_mb: 峰值内存占用
    - speedup_vs_torch: 相比原生实现的加速比
"""
import argparse
import os
import sys
import json
import time
import statistics


def benchmark_implementations(op_name, verify_dir, warmup=5, repeats=50):
    """测试框架实现和生成实现的性能"""
    import torch
    import torch_npu  # noqa: F401
    
    sys.path.insert(0, verify_dir)
    
    # 加载模块
    torch_module = __import__(f"{op_name}_torch")
    FrameworkModel = torch_module.Model
    get_inputs = torch_module.get_inputs
    get_init_inputs = torch_module.get_init_inputs
    
    impl_module = __import__(f"{op_name}_triton_ascend_impl")
    ModelNew = impl_module.ModelNew
    
    device = torch.device("npu")
    
    init_params = get_init_inputs()

    # 分别在每个模型创建前重置种子，确保含随机权重的算子（如 Conv2d）
    # 在 Model 和 ModelNew 中获得完全一致的初始化参数
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    framework_model = FrameworkModel(*init_params).to(device)

    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    impl_model = ModelNew(*init_params).to(device)
    
    # 准备输入数据（预热时生成一次，正式测试时复用）
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    inputs_impl = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    inputs_framework = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    
    def measure_latency(model, inputs, n):
        """测量 n 次运行的延迟（毫秒）"""
        latencies = []
        for _ in range(n):
            torch.npu.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(*inputs)
            torch.npu.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转换为毫秒
        return latencies
    
    # 重置内存统计
    torch.npu.reset_peak_memory_stats()
    
    # Warmup
    print(f"执行 warmup ({warmup} 次)...")
    _ = measure_latency(impl_model, inputs_impl, warmup)
    torch.npu.synchronize()
    
    # 正式测试：生成实现
    print(f"测试生成实现 ({repeats} 次)...")
    impl_latencies = measure_latency(impl_model, inputs_impl, repeats)
    impl_peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)  # MB
    
    # 重置内存统计
    torch.npu.reset_peak_memory_stats()
    
    # Warmup 框架实现
    _ = measure_latency(framework_model, inputs_framework, warmup)
    torch.npu.synchronize()
    
    # 正式测试：框架实现
    print(f"测试框架实现 ({repeats} 次)...")
    framework_latencies = measure_latency(framework_model, inputs_framework, repeats)
    framework_peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)  # MB
    
    # 计算统计指标
    def calc_stats(latencies):
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {
            "avg": statistics.mean(latencies),
            "p50": sorted_lat[n // 2] if n % 2 == 1 else (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2,
            "p99": sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0]
        }
    
    impl_stats = calc_stats(impl_latencies)
    framework_stats = calc_stats(framework_latencies)
    
    # 计算加速比
    speedup = framework_stats["avg"] / impl_stats["avg"] if impl_stats["avg"] > 0 else 0
    
    result = {
        "op_name": op_name,
        "warmup": warmup,
        "repeats": repeats,
        "framework": {
            "avg_latency_ms": round(framework_stats["avg"], 4),
            "p50_latency_ms": round(framework_stats["p50"], 4),
            "p99_latency_ms": round(framework_stats["p99"], 4),
            "peak_memory_mb": round(framework_peak_memory, 2)
        },
        "implementation": {
            "avg_latency_ms": round(impl_stats["avg"], 4),
            "p50_latency_ms": round(impl_stats["p50"], 4),
            "p99_latency_ms": round(impl_stats["p99"], 4),
            "peak_memory_mb": round(impl_peak_memory, 2)
        },
        "speedup_vs_torch": round(speedup, 2)
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="性能测试脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--verify_dir", default=".", help="验证目录路径（默认当前目录）")
    parser.add_argument("--warmup", type=int, default=5, help="warmup 次数（默认 5）")
    parser.add_argument("--repeats", type=int, default=50, help="正式测试次数（默认 50）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    args = parser.parse_args()
    
    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        result = benchmark_implementations(
            args.op_name, 
            verify_dir, 
            warmup=args.warmup, 
            repeats=args.repeats
        )
        
        # 打印结果
        print("\n性能测试结果:")
        print(f"  框架实现 - 平均延迟: {result['framework']['avg_latency_ms']:.4f} ms")
        print(f"  生成实现 - 平均延迟: {result['implementation']['avg_latency_ms']:.4f} ms")
        print(f"  加速比: {result['speedup_vs_torch']:.2f}x")
        print(f"  生成实现 - 峰值内存: {result['implementation']['peak_memory_mb']:.2f} MB")
        
        # 保存到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
        else:
            print("\n结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        sys.exit(0)
        
    except Exception as e:
        print(f"性能测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
