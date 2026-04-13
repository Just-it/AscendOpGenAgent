#!/usr/bin/env python3
"""算子验证脚本 — 算子对应.pt文件中包含输入以及预期的输出, 相同输入下, 对比生成算子输出与预期输出的一致性。

用法:
    python verify.py --op_name <算子名> [--verify_dir <验证目录>] [--timeout <超时秒数>] [--device_id <所用设备id>]

前置条件（验证目录下需存在以下文件）:
    {op_name}.pt            — 包含输入，预期输出
    {op_name}.py            — 包含生成算子的主要逻辑
"""
import argparse
import os
import sys
import torch
import importlib
import gc

from test_common import convert_tensor_with_device_type, compare_data_precision


# 🔥 强制清空 NPU 缓存 + 内存
def clear_npu_memory():
    try:
        torch.npu.empty_cache()        # 清空NPU缓存
        torch.npu.synchronize()        # 强制同步所有操作
        gc.collect()                   # 强制Python垃圾回收
    except:
        pass

# 🔥 安全卸载动态导入的模块（防止句柄泄漏卡死）
def unload_module(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    gc.collect()

def verify_implementations(op_name, verify_dir, triton_impl_name):
    """验证框架实现和生成实现的结果一致性"""
    try:
        spec = importlib.util.spec_from_file_location(op_name, f"{verify_dir}/{op_name}.py")
        triton_npu_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triton_npu_module)

        # 获取 kernel 函数
        triton_npu_func = getattr(triton_npu_module, op_name)

        data = torch.load(f"{verify_dir}/{op_name}.pt", map_location=torch.device('cpu'), weights_only=False)

        input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

        triton_npu_func[data["grid"]](**input_data)
        torch.npu.synchronize()

        compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
        print("验证成功")
    except BaseException as e:
        print(f"❌【失败】执行报错：{str(e)}")
    finally:
        # 🔥 终极清理：必须执行，否则连续跑必卡死
        try:
            # 清理变量
            locals().clear()
            gc.collect()
            
            # 卸载动态模块
            unload_module(op_name)
            
            # 强制清空NPU
            clear_npu_memory()
            
            # 移除所有临时引用
            module = None
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="算子验证脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument(
        "--verify_dir", default=".",
        help="验证目录，包含 {op_name}.pt算子输入 和 test_{op_name}.py算子的triton逻辑",
    )
    parser.add_argument("--timeout", type=int, default=900, help="超时秒数（默认 900）")
    parser.add_argument(
        "--triton_impl_name", default="triton_ascend_impl",
        help="Triton 实现模块名（不含 op_name 前缀，默认 triton_ascend_impl）",
    )
    parser.add_argument(
        "--_run", action="store_true",
        help=argparse.SUPPRESS,  # 内部参数：子进程模式，直接执行验证
    )
    parser.add_argument(
        "--device_id", required=True, type=int, default=0, help="指定npu卡"
    )

    args = parser.parse_args()

    torch.npu.set_device(args.device_id)
    
    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        verify_implementations(args.op_name, verify_dir, args.triton_impl_name)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)
