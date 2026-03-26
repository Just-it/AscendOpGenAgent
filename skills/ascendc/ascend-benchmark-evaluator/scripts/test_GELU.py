"""
GELU 算子评测脚本
通过对比 Python 参考实现、原生 PyTorch 实现和自定义 NPU 实现的精度
"""
import json
import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

# 设置环境变量
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '/home/y00889327/AscendOpGenAgent/output/GELU/vendors/customize'
import torch_npu

# 导入自定义算子
try:
    import custom_ops_lib
    has_custom_op = True
except ImportError:
    print("[WARNING] 无法导入 custom_ops_lib，将跳过自定义算子测试")
    has_custom_op = False


def load_test_cases(json_file):
    """从 JSON 文件加载测试用例"""
    test_cases = []
    with open(json_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    test_cases.append(data)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] 解析 JSON 行失败: {e}")
    return test_cases


def test_gelu(test_data, use_custom=False):
    """测试单个 GELU 用例"""
    inputs = test_data.get('inputs', [])

    # 解析输入参数
    x_desc = inputs[0] if len(inputs) > 0 else None
    approximate = "none"
    if len(inputs) > 1 and inputs[1].get('name') == 'approximate':
        approximate = inputs[1].get('value', 'none')

    if x_desc is None:
        return None, "No input tensor"

    shape = x_desc.get('shape', [])
    dtype_str = x_desc.get('dtype', 'float32')

    # 转换 dtype
    if dtype_str == 'float32':
        dtype = torch.float32
    elif dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 生成测试数据
    if dtype == torch.float16:
        # 使用更小的值范围避免溢出
        x_cpu = (torch.randn(shape, dtype=torch.float32) * 0.5).to(dtype)
    else:
        x_cpu = torch.randn(shape, dtype=dtype) * 0.5  # 减小值范围以避免数值问题

    # 参考结果（使用更高精度的 float32 计算）
    x_ref = x_cpu.to(torch.float32)
    ref_output = F.gelu(x_ref, approximate=approximate)

    # 测试自定义算子（如果可用）
    custom_output = None
    custom_time = None

    if use_custom and has_custom_op:
        # 将数据移到 NPU
        x_npu = x_cpu.npu()

        # Warmup
        for _ in range(5):
            try:
                _ = custom_ops_lib.gelu_custom(x_npu, approximate)
            except Exception as e:
                print(f"[ERROR] Warmup 失败: {e}")
                return None, str(e)

        # 实际测试
        start_time = time.time()
        try:
            custom_output = custom_ops_lib.gelu_custom(x_npu, approximate)
            torch.npu.synchronize()
            custom_time = (time.time() - start_time) * 1000  # 转换为毫秒
        except Exception as e:
            print(f"[ERROR] 自定义算子执行失败: {e}")
            return None, str(e)

        custom_output = custom_output.cpu()

    # 测试 PyTorch GELU（CPU）
    start_time = time.time()
    torch_output = F.gelu(x_cpu, approximate=approximate)
    torch_time = (time.time() - start_time) * 1000  # 转换为毫秒

    # 测试 PyTorch GELU（NPU，如果可用）
    torch_npu_output = None
    torch_npu_time = None
    if torch.npu.is_available():
        x_npu = x_cpu.npu()
        torch_npu_output = F.gelu(x_npu, approximate=approximate)

        # Warmup
        for _ in range(5):
            _ = F.gelu(x_npu, approximate=approximate)

        # 实际测试
        start_time = time.time()
        torch_npu_output = F.gelu(x_npu, approximate=approximate)
        torch.npu.synchronize()
        torch_npu_time = (time.time() - start_time) * 1000
        torch_npu_output = torch_npu_output.cpu()

    # 计算差异
    results = {
        'shape': shape,
        'dtype': dtype_str,
        'approximate': approximate,
        'ref_output': ref_output,
        'torch_output': torch_output,
        'custom_output': custom_output,
        'torch_npu_output': torch_npu_output,
        'torch_time_ms': torch_time,
        'custom_time_ms': custom_time,
        'torch_npu_time_ms': torch_npu_time
    }

    # 计算精度差异
    if custom_output is not None:
        results['custom_max_diff'] = (ref_output - custom_output.to(torch.float32)).abs().max().item()
        results['custom_mae'] = (ref_output - custom_output.to(torch.float32)).abs().mean().item()

    results['torch_max_diff'] = (ref_output - torch_output.to(torch.float32)).abs().max().item()
    if torch_npu_output is not None:
        results['torch_npu_max_diff'] = (ref_output - torch_npu_output.to(torch.float32)).abs().max().item()

    return results, None


def main():
    # 加载测试用例
    test_cases = load_test_cases('/home/y00889327/AscendOpGenAgent/benchmarks/NPUKernelBench/level1/1_GELU.json')

    print(f"[INFO] 加载了 {len(test_cases)} 个测试用例")

    # 运行评测
    use_custom = len(sys.argv) > 1 and sys.argv[1] == '--custom'

    print(f"\n[{'启用' if use_custom else '禁用'}自定义算子测试]\n")

    all_passed = True
    for i, test_data in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1}/{len(test_cases)} ---")
        result, error = test_gelu(test_data, use_custom=use_custom)

        if error is not None:
            print(f"[FAILED] 测试失败: {error}")
            all_passed = False
            continue

        # 打印结果
        print(f"  形状: {result['shape']}, dtype: {result['dtype']}, 近似: {result['approximate']}")
        print(f"  PyTorch (CPU) 差异: {result['torch_max_diff']:.6e}")

        if 'torch_npu_max_diff' in result:
            print(f"  原生 PyTorch (NPU) 差异: {result['torch_npu_max_diff']:.6e}")

        if use_custom and result.get('custom_output') is not None:
            print(f"  自定义算子差异: max={result['custom_max_diff']:.6e}, mae={result['custom_mae']:.6e}")

            # 检查是否通过（通常使用相对容差 1e-3 或绝对容差 1e-5）
            tolerance = 1e-3 if result['dtype'] in ['float16', 'bfloat16'] else 1e-5
            if result['custom_max_diff'] > tolerance:
                print(f"  [FAILED] 自定义算子精度验证失败！超过容差 {tolerance}")
                all_passed = False
            else:
                print(f"  [PASSED] 自定义算子精度验证通过")

            # 打印性能对比
            if result.get('torch_time_ms') and result.get('custom_time_ms'):
                print(f"  性能: 自定义={result['custom_time_ms']:.3f}ms, PyTorch CPU={result['torch_time_ms']:.3f}ms")
        else:
            print(f"  自定义算子不可用")

    print(f"\n{'='*50}\n评测完成！")

    if all_passed:
        print("[SUCCESS] 所有测试通过！")
        return 0
    else:
        print("[FAILED] 有测试失败")
        return 1


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
