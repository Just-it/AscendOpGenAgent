#!/usr/bin/env python3
"""Triton 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测两种退化类型：
  Type 1: 无 @triton.jit kernel，全部使用 PyTorch
  Type 2: 代码中存在禁止的 PyTorch 计算操作

用法:
    python validate_triton_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import sys


# ---------------------------------------------------------------------------
# 白名单：允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor", "arange", "linspace",
    # 类型 / 设备
    "as_tensor",
}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    # 布局操作（不执行计算）
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda",
    "item", "tolist",
    # 原地标记
    "requires_grad_", "zero_",
    # 切片相关（一般通过 __getitem__ 而非方法，但以防万一）
    "index_select",
}

ALLOWED_TRITON_ATTRS = {
    "cdiv", "next_power_of_2",
}

FORBIDDEN_TENSOR_METHODS = {
    # 计算操作
    "sum", "mean", "max", "min", "softmax", "log_softmax",
    "matmul", "mm", "bmm", "addmm", "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "norm", "layer_norm", "batch_norm", "group_norm",
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    "dropout", "softplus", "hardtanh", "hardswish",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _decorator_is_triton_jit(decorator):
    """判断装饰器节点是否为 triton.jit 或 @jit（从 triton 导入）。"""
    # @triton.jit
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "jit"):
            return True
    # @jit（直接导入）
    if isinstance(decorator, ast.Name) and decorator.id == "jit":
        return True
    # @triton.jit 作为 Call（如 @triton.jit 带参数，虽然少见）
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_jit(decorator.func)
    return False


def _decorator_is_triton_autotune(decorator):
    """判断装饰器是否为 triton.autotune。"""
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "autotune"):
            return True
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_autotune(decorator.func)
    return False


def _has_triton_decorator(func_node):
    """检查函数是否有 @triton.jit（可能与 @triton.autotune 组合）。"""
    for dec in func_node.decorator_list:
        if _decorator_is_triton_jit(dec):
            return True
    return False


def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          my_func    -> (None, 'my_func')
          self.conv  -> ('self', 'conv')
          kernel[g]  -> 返回 None（kernel launch 通过 Subscript）
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        # 处理 torch.nn.functional.relu 形式
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def find_triton_kernels(tree):
    """查找所有 @triton.jit 装饰的函数名，及其是否使用了 tl.* API。"""
    kernels = {}  # name -> {"has_tl_usage": bool, "line": int, end_line: int}
    kernel_ranges = []  # 存储 (start_line, end_line)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node):
            # 检查函数体中是否使用 tl.* API
            has_tl = False
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute):
                    if isinstance(child.value, ast.Name) and child.value.id == "tl":
                        has_tl = True
                        break

            # 获取函数起止行
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
            kernels[node.name] = {
                "has_tl_usage": has_tl,
                "line": node.lineno,
                "end_line": end_line
            }
            kernel_ranges.append((start_line, end_line))

    return kernels, kernel_ranges


def check_forbidden_torch_ops(tree, kernel_ranges):
    """检查整个代码中是否使用了禁止的 torch 计算操作 —— 跳过 Triton 内核内部"""
    violations = []

    def is_inside_kernel(line):
        # 判断当前行是否在任意一个 @triton.jit 函数内部
        for (start, end) in kernel_ranges:
            if start <= line <= end:
                return True
        return False

    for node in ast.walk(tree):
        if not hasattr(node, 'lineno'):
            continue
        # 跳过在 kernel 内部的代码
        line = node.lineno
        if is_inside_kernel(line):
            continue

        # --- 检测 @ 运算符（矩阵乘法）---
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append({
                "line": node.lineno,
                "call": "@",
                "reason": "矩阵乘法 @ 运算符必须在 Triton kernel 中实现",
            })
            continue

        if not isinstance(node, ast.Call):
            continue

        # --- kernel launch 跳过检查 ---
        if isinstance(node.func, ast.Subscript):
            continue

        resolved = _resolve_call_name(node)
        if resolved is None:
            continue

        qual, attr = resolved

        # --- torch.xxx(...) ---
        if qual == "torch":
            if attr not in ALLOWED_TORCH_FUNCS:
                violations.append({
                    "line": node.lineno,
                    "call": f"torch.{attr}",
                    "reason": f"torch.{attr} 是计算操作，必须在 Triton kernel 中实现",
                })
            continue

        # --- F.xxx(...) / functional.xxx(...) ---
        if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
            violations.append({
                "line": node.lineno,
                "call": f"{qual}.{attr}",
                "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 Triton kernel 中实现",
            })
            continue

        # --- triton.cdiv 等 —— 允许 ---
        if qual == "triton" and attr in ALLOWED_TRITON_ATTRS:
            continue

        # --- tensor 方法计算操作 ---
        if attr in FORBIDDEN_TENSOR_METHODS:
            # 排除已知安全的 qual（torch/F/triton 已在上面处理）
            if qual not in ("torch", "F", "triton", "functional", "torch.nn.functional", "nn.functional"):
                violations.append({
                    "line": node.lineno,
                    "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                    "reason": f"{attr} 是计算操作，必须在 Triton kernel 中实现",
                })
            continue

    return violations


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = {
        "valid": False,
        "filepath": filepath,
        "checks": {
            "triton_kernel_exists": {"passed": False, "kernels": [], "error": None},
            "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
        },
        "regression_type": None,
        "suggestion": "",
    }

    # --- 解析 ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["triton_kernel_exists"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    # --- Check 1: kernel 存在性 ---
    kernels, kernel_ranges = find_triton_kernels(tree)
    kernel_names = set(kernels.keys())
    result["checks"]["triton_kernel_exists"]["kernels"] = [
        {"name": k, "line": v["line"], "has_tl_usage": v["has_tl_usage"]}
        for k, v in kernels.items()
    ]

    if not kernel_names:
        result["checks"]["triton_kernel_exists"]["error"] = "未找到任何 @triton.jit 装饰的 kernel 函数"
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有 Triton kernel。必须创建至少一个 @triton.jit 装饰的函数，"
            "在其中使用 tl.load/tl.store 实现核心计算逻辑。"
        )
        return result

    # 检查 kernel 是否使用了 tl API
    kernels_without_tl = [k for k, v in kernels.items() if not v["has_tl_usage"]]
    if len(kernels_without_tl) == len(kernels):
        result["checks"]["triton_kernel_exists"]["error"] = (
            f"kernel 函数 {kernels_without_tl} 未使用任何 tl.* API，"
            "可能是空壳 kernel"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "虽然存在 @triton.jit 装饰的函数，但没有使用 triton.language (tl) API。"
            "kernel 必须使用 tl.load/tl.store 等进行显式内存操作和计算。"
        )
        return result

    result["checks"]["triton_kernel_exists"]["passed"] = True

    # --- Check 2: 禁止的 torch 操作 ---
    violations = check_forbidden_torch_ops(tree, kernel_ranges)
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if violations:
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"代码中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
        )
        violation_details = "; ".join(
            f"第{v['line']}行 {v['call']}" for v in violations[:5]
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"代码中使用了禁止的 PyTorch 计算操作: {violation_details}。"
            "所有核心计算必须在 @triton.jit kernel 中完成，"
            "仅允许 buffer 分配和形状操作。"
        )
        return result

    result["checks"]["no_forbidden_torch_ops"]["passed"] = True

    # --- 全部通过 ---
    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="检查生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        if args.json:
            print(json.dumps({"valid": False, "error": f"文件不存在: {args.file}"}))
        else:
            print(f"[ERROR] 文件不存在: {args.file}")
        sys.exit(1)

    result = validate(code, filepath=args.file)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["valid"]:
            kernels = result["checks"]["triton_kernel_exists"]["kernels"]
            print("[PASS] Triton 实现验证通过")
            print(f"  - 发现 {len(kernels)} 个有效 @triton.jit kernel: {', '.join(k['name'] for k in kernels)}")
            print("  - 代码中无禁止的 PyTorch 计算操作")
        else:
            rtype = result["regression_type"]
            type_desc = {
                1: "完全无 Triton kernel（纯 PyTorch）",
                2: "部分计算使用 PyTorch（需全部移入 Triton kernel）",
            }
            print(f"[FAIL] 检测到 PyTorch 退化 — Type {rtype}: {type_desc.get(rtype, '未知')}")

            # 显示具体检查结果
            for check_name, check_result in result["checks"].items():
                status = "PASS" if check_result["passed"] else "FAIL"
                print(f"  [{status}] {check_name}")
                if check_result["error"]:
                    print(f"         {check_result['error']}")

            if result["checks"]["no_forbidden_torch_ops"]["violations"]:
                print("  违规详情:")
                for v in result["checks"]["no_forbidden_torch_ops"]["violations"]:
                    print(f"    第 {v['line']} 行: {v['call']} — {v['reason']}")

            print(f"\n  修复建议: {result['suggestion']}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()