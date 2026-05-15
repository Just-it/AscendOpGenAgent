#!/usr/bin/env python3
"""TileLang 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测四种退化类型：
  Type 1: 无 TileLang kernel 导入（纯 PyTorch）
  Type 2: 有 kernel 导入但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口
  Type 4: forward() 中存在逐元素 Python for 循环（标量写法退化）

TileLang 正确模式：
  1. 从 design.tile_level.xxx 导入 kernel builder 函数
  2. forward() 中调用 builder 获取 kernel 对象: kernel = builder(M, N, ...)
  3. 调用 kernel 对象执行计算: result = kernel(x, y)

用法:
    python validate_tilelang_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
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
    # 切片相关
    "index_select",
    # 设备检查
    "is_npu", "is_cuda",
}

ALLOWED_BUILTIN_FUNCS = {
    # Python 内建函数（非 tensor 方法）
    "min", "max", "abs", "len", "range", "int", "float", "bool",
    "list", "tuple", "str", "type", "isinstance", "print",
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "hasattr", "getattr", "setattr",
}

FORBIDDEN_TENSOR_METHODS = {
    # 归约操作
    "sum", "mean", "max", "min", "prod", "cumsum", "cumprod",
    "argmax", "argmin", "var", "std",
    # 矩阵 / 线性代数
    "matmul", "mm", "bmm", "addmm",
    # 逐元素算术
    "add", "sub", "mul", "div", "fmod", "remainder",
    "add_", "sub_", "mul_", "div_",
    # 激活函数
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "relu_", "sigmoid_", "tanh_",
    # 数学函数
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "sin", "cos", "clamp", "clamp_", "ceil", "floor", "round",
    "reciprocal", "neg", "sign",
    # softmax
    "softmax", "log_softmax",
    # 范数 / 归一化
    "norm", "layer_norm", "batch_norm", "group_norm",
    # 卷积 / 线性
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    # 其他
    "dropout", "softplus", "hardtanh", "hardswish",
    # 比较（用于计算，非条件判断时）
    "eq", "ne", "lt", "gt", "le", "ge", "where",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty      -> ('torch', 'empty')
          tl_rms_norm      -> (None, 'tl_rms_norm')
          self._build_kernel -> ('self', '_build_kernel')
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

def find_tilelang_kernel_imports(tree):
    """查找所有从 design.tile_level.* 导入的 TileLang kernel builder 函数。

    检测模式：
    1. from design.tile_level.xxx import yyy [as zzz]
    2. from design.tile_level.xxx import (a, b, c)

    返回 dict: {used_name: {"actual_name": str, "module": str,
                              "alias": str|None, "line": int}}
    """
    kernels = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # 匹配 design.tile_level.* 模块路径
            module = node.module
            if _is_tilelang_design_module(module):
                for alias in node.names:
                    used_name = alias.asname if alias.asname else alias.name
                    kernels[used_name] = {
                        "actual_name": alias.name,
                        "module": module,
                        "alias": alias.asname,
                        "line": node.lineno,
                    }

    return kernels


def _is_tilelang_design_module(module_path):
    """检查模块路径是否匹配 TileLang 设计模块的模式。

    匹配模式：
    - design.tile_level.xxx
    - design.tile_level.xxx.yyy
    """
    if not module_path:
        return False
    parts = module_path.split(".")
    if len(parts) >= 3 and parts[0] == "design" and parts[1] == "tile_level":
        return True
    return False


def find_model_forward(tree):
    """找到 ModelNew 或 Model 类的 forward 方法节点及其所属类。

    优先查找 ModelNew，若不存在则查找 Model。
    返回 (forward_node, class_name, class_node)。
    """
    model_new_forward = None
    model_new_class = None
    model_forward = None
    model_class = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "ModelNew":
                model_new_class = node
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "forward":
                            model_new_forward = item
            elif node.name == "Model":
                model_class = node
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "forward":
                            model_forward = item

    if model_new_forward:
        return model_new_forward, "ModelNew", model_new_class
    return model_forward, "Model", model_class


def find_build_kernel_methods(class_node, kernel_builder_names):
    """查找类中通过 kernel builder 构建 kernel 的辅助方法。

    典型模式：
      def _build_kernel(self, x):
          return tl_matmul_leakyrelu(m, n, k)

    返回方法名集合（这些方法返回可调用的 kernel 对象）。
    """
    builder_methods = set()
    if class_node is None:
        return builder_methods

    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name != "forward":
            for child in ast.walk(item):
                if isinstance(child, ast.Call):
                    resolved = _resolve_call_name(child)
                    if resolved:
                        qual, attr = resolved
                        # 直接调用 kernel builder: tl_xxx(M, N, ...)
                        if qual is None and attr in kernel_builder_names:
                            builder_methods.add(item.name)
                            break
                        # 通过模块调用（不太常见）
    return builder_methods


def find_module_wrapper_functions(tree, kernel_builder_names):
    """查找模块级别的辅助函数，它们调用 kernel builder 或 kernel 对象。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    resolved = _resolve_call_name(child)
                    if resolved:
                        qual, attr = resolved
                        if qual is None and attr in kernel_builder_names:
                            wrappers.add(node.name)
                            break
    return wrappers


def check_kernel_calls_in_forward(forward_node, kernel_builder_names,
                                  builder_method_names, wrapper_func_names):
    """检查 forward 中是否调用了 TileLang kernel。

    TileLang 调用模式：
    1. 直接模式:
       kernel = tl_builder(M, N, ...)  # 构建
       result = kernel(x, y)            # 调用
    2. builder 方法模式:
       kernel = self._build_kernel(x)   # 构建
       result = kernel(x, y)            # 调用
    3. 内联模式:
       result = tl_builder(M, N, ...)(x, y)  # 构建 + 调用
    4. wrapper 模式:
       result = wrapper_func(x, y)      # 通过 wrapper 调用

    返回被调用信息列表 [{"call": str, "line": int, "pattern": str}, ...]
    """
    called = []
    if forward_node is None:
        return called

    # 收集所有赋值目标中「由 kernel builder / builder method 返回的变量名」
    kernel_var_names = set()

    for node in ast.walk(forward_node):
        # --- 收集 kernel 变量：kernel = tl_builder(...) 或 kernel = self._build_kernel(...) ---
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                resolved = _resolve_call_name(node.value)
                if resolved:
                    qual, attr = resolved
                    # kernel = tl_builder(...)
                    if qual is None and attr in kernel_builder_names:
                        kernel_var_names.add(target.id)
                        called.append({
                            "call": attr,
                            "line": node.lineno,
                            "pattern": "builder_assign",
                        })
                    # kernel = self._build_kernel(...)
                    if qual == "self" and attr in builder_method_names:
                        kernel_var_names.add(target.id)
                        called.append({
                            "call": f"self.{attr}",
                            "line": node.lineno,
                            "pattern": "builder_method",
                        })

    # --- 检查 kernel 变量是否被调用: kernel(...) ---
    kernel_invoked = False
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call):
            resolved = _resolve_call_name(node)
            if resolved:
                qual, attr = resolved
                # kernel(x, y) — 调用由 builder 返回的 kernel 对象
                if qual is None and attr in kernel_var_names:
                    kernel_invoked = True
                    called.append({
                        "call": f"{attr}(...)",
                        "line": node.lineno,
                        "pattern": "kernel_invoke",
                    })
                # wrapper_func(x, y) — 通过 wrapper 调用
                if qual is None and attr in wrapper_func_names:
                    kernel_invoked = True
                    called.append({
                        "call": attr,
                        "line": node.lineno,
                        "pattern": "wrapper_call",
                    })
                # self.wrapper(x, y)
                if qual == "self" and attr in wrapper_func_names:
                    kernel_invoked = True
                    called.append({
                        "call": f"self.{attr}",
                        "line": node.lineno,
                        "pattern": "wrapper_call",
                    })

            # --- 内联模式: tl_builder(M, N)(x, y) ---
            if isinstance(node.func, ast.Call):
                inner_resolved = _resolve_call_name(node.func)
                if inner_resolved:
                    inner_qual, inner_attr = inner_resolved
                    if inner_qual is None and inner_attr in kernel_builder_names:
                        kernel_invoked = True
                        called.append({
                            "call": f"{inner_attr}(...)(...)",
                            "line": node.lineno,
                            "pattern": "inline_build_invoke",
                        })

    # 如果有 builder 赋值但没有 kernel 调用，标记不完整
    if called and not kernel_invoked:
        # 只有 builder assign，没有 invoke — 可能是未调用
        pass

    return called


def _has_kernel_invocation(called_list):
    """检查 called 列表中是否有实际的 kernel 调用（不仅仅是 builder 赋值）。"""
    invoke_patterns = {"kernel_invoke", "wrapper_call", "inline_build_invoke"}
    return any(c["pattern"] in invoke_patterns for c in called_list)


def check_forbidden_torch_ops(forward_node, kernel_builder_names,
                              builder_method_names):
    """检查 forward 中是否使用了禁止的 torch 计算操作。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        # --- 检测 @ 运算符（矩阵乘法）---
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append({
                "line": node.lineno,
                "call": "@",
                "reason": "矩阵乘法 @ 运算符必须在 TileLang kernel 中实现",
            })
            continue

        if not isinstance(node, ast.Call):
            continue

        resolved = _resolve_call_name(node)
        if resolved is None:
            continue

        qual, attr = resolved

        # --- TileLang kernel builder 调用 —— 允许 ---
        if qual is None and attr in kernel_builder_names:
            continue

        # --- kernel builder 方法调用 —— 允许 ---
        if qual == "self" and attr in builder_method_names:
            continue

        # --- torch.xxx(...) ---
        if qual == "torch":
            if attr not in ALLOWED_TORCH_FUNCS:
                violations.append({
                    "line": node.lineno,
                    "call": f"torch.{attr}",
                    "reason": f"torch.{attr} 是计算操作，必须在 TileLang kernel 中实现",
                })
            continue

        # --- F.xxx(...) / functional.xxx(...) ---
        if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
            violations.append({
                "line": node.lineno,
                "call": f"{qual}.{attr}",
                "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 TileLang kernel 中实现",
            })
            continue

        # --- Python 内建函数 —— 允许 ---
        if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
            continue

        # --- tensor 方法计算操作 ---
        if attr in FORBIDDEN_TENSOR_METHODS:
            if qual not in ("torch", "F", "functional",
                            "torch.nn.functional", "nn.functional"):
                violations.append({
                    "line": node.lineno,
                    "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                    "reason": f"{attr} 是计算操作，必须在 TileLang kernel 中实现",
                })
            continue

        # --- self.layer_name(x) —— 禁止 nn.Module 调用（但允许 builder 方法）---
        if qual == "self":
            if attr not in ("forward",) and attr not in builder_method_names:
                violations.append({
                    "line": node.lineno,
                    "call": f"self.{attr}(...)",
                    "reason": (
                        f"self.{attr}() 疑似 nn.Module 前向调用，"
                        "核心计算必须在 TileLang kernel 中实现"
                    ),
                })
            continue

    return violations


def check_for_loops_over_tensors(forward_node):
    """检查 forward 中是否存在用于计算的逐元素 Python for 循环（标量写法退化信号）。

    典型退化模式：
      for n in range(N):
          kernel(grad_output[n:n+1], ...)     # 逐批次调用
          grad_weight_per_n[n] = ...          # 逐元素收集
      grad_weight = grad_weight_per_n.sum(dim=0)  # 外部归约

    以下不视为退化：
      - 数据准备循环（仅做简单赋值 / 索引映射，无计算操作）

    返回违规列表 [{"line": N, "loop_var": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        if not isinstance(node, ast.For):
            continue

        # 检查是否是 for var in range(...)
        if isinstance(node.iter, ast.Call):
            resolved = _resolve_call_name(node.iter)
            if resolved and resolved == (None, "range"):
                loop_var = ""
                if isinstance(node.target, ast.Name):
                    loop_var = node.target.id

                has_tensor_indexing = _loop_has_tensor_indexing(node, loop_var)
                has_computation = _loop_has_computation(node)

                if has_tensor_indexing and has_computation:
                    violations.append({
                        "line": node.lineno,
                        "loop_var": loop_var,
                        "reason": (
                            f"for {loop_var} in range(...) 循环中存在 tensor 索引 + 计算操作，"
                            "这是逐元素标量写法，必须使用 TileLang kernel 的向量化操作替代"
                        ),
                    })

    return violations


def _loop_has_tensor_indexing(for_node, loop_var):
    """检查 for 循环体中是否存在使用循环变量的 tensor 索引。"""
    if not loop_var:
        return False
    for child in ast.walk(for_node):
        if isinstance(child, ast.Subscript):
            for sub_node in ast.walk(child.slice):
                if isinstance(sub_node, ast.Name) and sub_node.id == loop_var:
                    return True
    return False


def _loop_has_computation(for_node):
    """检查 for 循环体中是否包含实际的计算操作。

    计算操作包括：
    - 禁止的 tensor 方法（.sum(), .mul(), ...）
    - torch.xxx 计算调用
    - F.xxx 计算调用
    - 大量 BinOp 算术运算符
    - @ 矩阵乘法运算符
    """
    arithmetic_ops = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Pow, ast.Mod, ast.MatMult,
    )
    binop_count = 0

    for child in ast.walk(for_node):
        # 检查 @ 运算符
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            return True

        # 统计算术 BinOp
        if isinstance(child, ast.BinOp) and isinstance(child.op, arithmetic_ops):
            binop_count += 1

        # 检查函数调用
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if resolved is None:
                continue
            qual, attr = resolved

            # Python 内建函数 —— 允许
            if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
                continue

            # 禁止的 tensor 方法
            if attr in FORBIDDEN_TENSOR_METHODS:
                if qual not in ("torch", "F", "functional",
                                "torch.nn.functional", "nn.functional"):
                    return True

            # torch 计算调用
            if qual == "torch" and attr not in ALLOWED_TORCH_FUNCS:
                return True

            # F.xxx 计算调用
            if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
                return True

    # 阈值：5 个以上算术操作视为计算密集型循环
    if binop_count >= 5:
        return True

    return False


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
            "tilelang_kernel_imported": {
                "passed": False, "kernels": [], "error": None,
            },
            "kernel_called_from_forward": {
                "passed": False, "called": [], "error": None,
            },
            "no_forbidden_torch_ops": {
                "passed": False, "violations": [], "error": None,
            },
            "no_scalar_for_loops": {
                "passed": False, "violations": [], "error": None,
            },
        },
        "regression_type": None,
        "suggestion": "",
    }

    # --- 解析 ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["tilelang_kernel_imported"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    # --- Check 1: TileLang kernel 导入存在性 ---
    kernels = find_tilelang_kernel_imports(tree)
    kernel_builder_names = set(kernels.keys())

    result["checks"]["tilelang_kernel_imported"]["kernels"] = [
        {
            "used_name": k,
            "actual_name": v["actual_name"],
            "module": v["module"],
            "line": v["line"],
        }
        for k, v in kernels.items()
    ]

    if not kernel_builder_names:
        result["checks"]["tilelang_kernel_imported"]["error"] = (
            "未找到任何从 design.tile_level.* 导入的 TileLang kernel builder"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有从 design.tile_level.* 导入 TileLang kernel builder。"
            "model_new_tilelang.py 必须从 design/tile_level/ 导入 kernel builder 函数"
            "（如 from design.tile_level.xxx import xxx），"
            "并在 forward() 中调用 builder 构建 kernel 再执行计算。"
        )
        return result

    result["checks"]["tilelang_kernel_imported"]["passed"] = True

    # --- Check 2: forward 是否调用 kernel ---
    forward_node, class_name, class_node = find_model_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 或 Model.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew（或 Model）类或 forward 方法。"
        return result

    builder_method_names = find_build_kernel_methods(class_node, kernel_builder_names)
    wrapper_func_names = find_module_wrapper_functions(tree, kernel_builder_names)

    called = check_kernel_calls_in_forward(
        forward_node, kernel_builder_names,
        builder_method_names, wrapper_func_names,
    )
    result["checks"]["kernel_called_from_forward"]["called"] = [
        {"call": c["call"], "line": c["line"], "pattern": c["pattern"]}
        for c in called
    ]

    has_invocation = _has_kernel_invocation(called)

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"已导入 kernel builder {list(kernel_builder_names)} 但 "
            f"{class_name}.forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已导入 TileLang kernel builder {list(kernel_builder_names)} 但 "
            f"{class_name}.forward() 中未使用。"
            "forward() 必须通过 kernel = builder(M, N, ...); kernel(x, y) 模式调用。"
        )
        return result

    if not has_invocation:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"forward() 中调用了 kernel builder 但未执行返回的 kernel 对象"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            "forward() 构建了 kernel 对象但没有执行它。"
            "请确保在 kernel = builder(M, N, ...) 之后调用 kernel(x, y, ...)。"
        )
        return result

    result["checks"]["kernel_called_from_forward"]["passed"] = True

    # --- Check 3: 禁止的 torch 操作 ---
    violations = check_forbidden_torch_ops(
        forward_node, kernel_builder_names, builder_method_names,
    )
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if violations:
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"forward() 中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
        )
        violation_details = "; ".join(
            f"第{v['line']}行 {v['call']}" for v in violations[:5]
        )
        result["regression_type"] = 3
        result["suggestion"] = (
            f"forward() 调用了 TileLang kernel 但仍使用 PyTorch 进行部分计算: "
            f"{violation_details}。"
            "所有核心计算必须在 TileLang kernel 中完成，"
            "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
        )
        return result

    result["checks"]["no_forbidden_torch_ops"]["passed"] = True

    # --- Check 4: 标量 for 循环退化 ---
    loop_violations = check_for_loops_over_tensors(forward_node)
    result["checks"]["no_scalar_for_loops"]["violations"] = loop_violations

    if loop_violations:
        result["checks"]["no_scalar_for_loops"]["error"] = (
            f"forward() 中发现 {len(loop_violations)} 处逐元素 Python for 循环"
        )
        loop_details = "; ".join(
            f"第{v['line']}行 for {v['loop_var']} in range(...)"
            for v in loop_violations[:5]
        )
        result["regression_type"] = 4
        result["suggestion"] = (
            f"forward() 中存在逐元素 Python for 循环: {loop_details}。"
            "不能用标量逐元素写法，必须使用 TileLang kernel 的向量化 / 块级操作。"
        )
        return result

    result["checks"]["no_scalar_for_loops"]["passed"] = True

    # --- 全部通过 ---
    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="检查 TileLang 生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径或输出目录")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    file_path = args.file
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, "model_new_tilelang.py")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        if args.json:
            print(json.dumps({"valid": False, "error": f"文件不存在: {file_path}"}))
        else:
            print(f"[ERROR] 文件不存在: {file_path}")
        sys.exit(1)

    result = validate(code, filepath=file_path)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["valid"]:
            kerns = result["checks"]["tilelang_kernel_imported"]["kernels"]
            called = result["checks"]["kernel_called_from_forward"]["called"]
            print("[PASS] TileLang 实现验证通过")
            print(f"  - 导入 {len(kerns)} 个 kernel builder: "
                  f"{', '.join(k['used_name'] for k in kerns)}")
            print(f"  - forward() 调用: "
                  f"{', '.join(c['call'] for c in called)}")
            print("  - forward() 中无禁止的 PyTorch 计算操作")
            print("  - forward() 中无逐元素 Python for 循环")
        else:
            rtype = result["regression_type"]
            type_desc = {
                1: "无 TileLang kernel 导入（纯 PyTorch）",
                2: "有 kernel 导入但 forward() 未调用",
                3: "部分计算仍使用 PyTorch（需全部移入 TileLang kernel）",
                4: "存在逐元素 Python for 循环（需使用向量化操作）",
            }
            print(f"[FAIL] 检测到 PyTorch 退化 — Type {rtype}: "
                  f"{type_desc.get(rtype, '未知')}")

            for check_name, check_result in result["checks"].items():
                status = "PASS" if check_result["passed"] else "FAIL"
                print(f"  [{status}] {check_name}")
                if check_result["error"]:
                    print(f"         {check_result['error']}")

            # 显示 torch 操作违规详情
            torch_violations = result["checks"]["no_forbidden_torch_ops"]["violations"]
            if torch_violations:
                print("  torch 操作违规详情:")
                for v in torch_violations:
                    print(f"    第 {v['line']} 行: {v['call']} — {v['reason']}")

            # 显示 for 循环违规详情
            loop_violations = result["checks"]["no_scalar_for_loops"]["violations"]
            if loop_violations:
                print("  for 循环违规详情:")
                for v in loop_violations:
                    print(f"    第 {v['line']} 行: for {v['loop_var']} in range(...)"
                          f" — {v['reason']}")

            print(f"\n  修复建议: {result['suggestion']}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
