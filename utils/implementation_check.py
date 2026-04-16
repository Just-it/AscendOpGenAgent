#!/usr/bin/env python3
"""
实现方式校验脚本
在运行正确性验证之前，检查 model_new_*.py 是否使用自定义 kernel 实现，
而非 torch/torch_npu 替代。
"""

import re
import sys
import argparse


def check_implementation(file_path, file_type='tilelang'):
    """
    检查实现文件是否使用自定义 kernel，而非 torch/torch_npu 替代。

    Args:
        file_path: 要检查的文件路径
        file_type: 'tilelang' 或 'ascendc'

    Returns:
        (passed, errors): 是否通过检查，错误列表
    """
    with open(file_path, 'r') as f:
        content = f.read()

    errors = []

    # 禁止的模式：torch/torch_npu 计算操作
    forbidden_patterns = [
        # PyTorch 数学运算
        (r'torch\.(add|mul|div|sub|sum|mean|matmul|mm|bmm|dot|outer|tensordot)\s*\(', 'PyTorch math operation'),
        (r'torch\.(where|clamp|maximum|minimum|abs|sqrt|exp|log|pow|sin|cos|tan|sinh|cosh|tanh)\s*\(', 'PyTorch element-wise operation'),
        (r'torch\.(softmax|relu|gelu|silu|leaky_relu|elu|selu|hardtanh|hardshrink)\s*\(', 'PyTorch activation function'),
        (r'torch\.(max|min|argmax|argmin|sort|topk)\s*\(', 'PyTorch reduction/sort operation'),
        (r'torch\.(cat|stack|concat|repeat|tile|expand|permute|transpose|reshape|view|flatten|unflatten)\s*\(', 'PyTorch tensor manipulation'),
        # PyTorch NPU 接口
        (r'torch_npu\.[a-zA-Z_][a-zA-Z0-9_]*\s*\(', 'torch_npu API call - MUST use custom kernel instead'),
        # Tensor 方法调用（计算类）
        (r'\.(sum|mean|matmul|add|mul|div|sub|max|min|clamp|where|softmax|relu|gelu|silu|leaky_relu)\s*\([^)]*\)', 'Tensor method call for computation'),
        # 神经网络函数
        (r'torch\.nn\.functional\.[a-zA-Z_][a-zA-Z0-9_]*\s*\(', 'torch.nn.functional call'),
    ]

    for pattern, desc in forbidden_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            line_content = content.split('\n')[line_num - 1].strip()
            # 排除注释行、定义行、以及辅助函数（如 _make_boxes, _make_tensor 等）
            if (not line_content.startswith('#') and
                not line_content.startswith('def ') and
                not line_content.startswith('"""') and
                not line_content.startswith("'") and
                'boxes' not in line_content and  # 排除 _make_boxes 中的使用
                '_make_' not in line_content):   # 排除所有 _make_* 辅助函数
                errors.append(f"Line {line_num}: {desc} -> {match.group()}")

    # 检查是否有自定义 kernel 调用（必须存在）
    if file_type == 'tilelang':
        kernel_patterns = [
            r'advance_step_flashattn\s*\(',
            r'tl_kernel\s*\(',
            r'kernel\s*\(',
        ]
    else:  # ascendc
        kernel_patterns = [
            r'advance_step_flashattn\s*\(',
            r'ascendc_kernel\s*\(',
            r'kernel\s*\(',
            r'load\s*\(\s*[^)]*advance_step',
        ]

    has_kernel_call = False
    for pattern in kernel_patterns:
        if re.search(pattern, content):
            has_kernel_call = True
            break

    if not has_kernel_call:
        errors.append("No custom kernel call found - must call custom kernel for computation")

    # 检查是否导入了 TileLang 或 AscendC 模块（根据类型）
    if file_type == 'tilelang':
        if not re.search(r'(import tilelang|from tilelang|from design\.tile_level)', content):
            errors.append("No TileLang import found - should import TileLang kernel")
    else:  # ascendc
        if not re.search(r'(import torch_npu|from kernel|load\s*\()', content):
            errors.append("No AscendC kernel import/load found - should import AscendC kernel")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description='Check implementation uses custom kernel')
    parser.add_argument('file_path', help='Path to model_new_*.py file')
    parser.add_argument('--type', choices=['tilelang', 'ascendc'], default='tilelang',
                        help='Type of implementation')
    args = parser.parse_args()

    passed, errors = check_implementation(args.file_path, args.type)

    if passed:
        print(f"PASS: Implementation check passed for {args.file_path}")
        print(f"  - Using custom {args.type.upper()} kernel (no torch/torch_npu fallback detected)")
        return 0
    else:
        print(f"FAIL: Implementation check failed for {args.file_path}")
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nREQUIREMENT: Core computation MUST use custom kernel, NOT torch/torch_npu!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
