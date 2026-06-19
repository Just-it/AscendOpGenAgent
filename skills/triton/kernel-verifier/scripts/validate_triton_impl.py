#!/usr/bin/env python3
"""Triton 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测三种退化类型：
  Type 1: 无 @triton.jit kernel，全部使用 PyTorch
  Type 2: 有 @triton.jit kernel 定义但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口

用法:
    python validate_triton_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
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

# forward() 中禁止的 Python 控制流和结构
FORBIDDEN_PYTHON_STMTS = {
    "for": "Python for 循环",
    "while": "Python while 循环",
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


def _get_subscript_value_name(node):
    """从 kernel[grid](...) 的 Subscript 节点提取 kernel 名称。"""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                return f"{node.value.value.id}.{node.value.attr}"
    return None


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def find_triton_kernels(tree):
    """查找所有 @triton.jit 装饰的函数名，及其是否使用了 tl.* API。"""
    kernels = {}  # name -> {"has_tl_usage": bool, "line": int}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node):
            # 检查函数体中是否使用 tl.* API
            has_tl = False
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute):
                    if isinstance(child.value, ast.Name) and child.value.id == "tl":
                        has_tl = True
                        break
            kernels[node.name] = {"has_tl_usage": has_tl, "line": node.lineno}
    return kernels


def find_model_new_forward(tree):
    """找到 ModelNew 类的 forward 方法节点。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "forward":
                        return item
    return None


def find_wrapper_functions(tree, kernel_names):
    """找到模块级别或类级别的辅助函数，这些函数内部调用了 triton kernel。

    返回函数名集合（类方法以 "ClassName.method_name" 格式存储）。
    """
    wrappers = set()

    # --- 模块级别函数 ---
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Subscript):
                    name = _get_subscript_value_name(child.func)
                    if name in kernel_names:
                        wrappers.add(node.name)
                        break
                if isinstance(child, ast.Call):
                    resolved = _resolve_call_name(child)
                    if resolved and resolved[0] is None and resolved[1] in kernel_names:
                        wrappers.add(node.name)
                        break

    # --- 类级别方法（如 ModelNew._route） ---
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name not in kernel_names:
                    for child in ast.walk(item):
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Subscript):
                            name = _get_subscript_value_name(child.func)
                            if name in kernel_names:
                                wrappers.add(f"{class_name}.{item.name}")
                                break
                        if isinstance(child, ast.Call):
                            resolved = _resolve_call_name(child)
                            if resolved and resolved[0] is None and resolved[1] in kernel_names:
                                wrappers.add(f"{class_name}.{item.name}")
                                break
    return wrappers


def check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names, class_name="ModelNew"):
    """检查 forward 中是否调用了 triton kernel（直接或通过 wrapper）。

    返回被调用的 kernel/wrapper 名称集合。
    支持 self._route(...) 形式的类级别 wrapper 调用。
    """
    called = set()
    if forward_node is None:
        return called
    for node in ast.walk(forward_node):
        # kernel[grid](...) 形式
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            name = _get_subscript_value_name(node.func)
            if name in kernel_names:
                called.add(name)
        # 直接调用 kernel_name(...) — 虽然不标准但以防万一
        if isinstance(node, ast.Call):
            resolved = _resolve_call_name(node)
            if resolved:
                qual, attr = resolved
                if qual is None and attr in kernel_names:
                    called.add(attr)
                if qual is None and attr in wrapper_names:
                    called.add(attr)
                # self.wrapper_name(...)
                if qual == "self" and attr in wrapper_names:
                    called.add(attr)
                # self._route(...) -> 映射到 ClassName._route 格式
                if qual == "self" and f"{class_name}.{attr}" in wrapper_names:
                    called.add(f"{class_name}.{attr}")
    return called


def _count_kernel_launches_in_forward(forward_node):
    """统计 forward() 中 kernel 启动调用（kernel[grid](...)）的次数。"""
    count = 0
    if forward_node is None:
        return count
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            count += 1
    return count


def _is_loop_pure_kernel_launch(loop_node, kernel_names, wrapper_names):
    """检查循环体是否仅包含 kernel 启动和允许的 host 侧操作。

    允许的语句：kernel[grid](...)、赋值、条件判断（if/else）、
    torch.empty/empty_like、属性访问、方法调用（如 .contiguous/.numel 等）。
    禁止的语句：任何非 kernel 的 torch/F 计算操作、nn.Module 调用等。
    """
    allowed_expr_types = (
        ast.Call, ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.If, ast.Pass, ast.Expr, ast.Subscript, ast.Attribute,
        ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare,
        ast.BoolOp, ast.Tuple, ast.List, ast.Dict, ast.Return,
    )

    def _check_stmt(stmt):
        if isinstance(stmt, ast.If):
            for s in stmt.body + stmt.orelse:
                if not _check_stmt(s):
                    return False
            return True
        if isinstance(stmt, ast.For):
            return False  # 嵌套循环禁止
        if isinstance(stmt, ast.While):
            return False
        if isinstance(stmt, ast.With):
            return False
        if isinstance(stmt, ast.Try):
            return False
        if isinstance(stmt, ast.Expr):
            return _check_expr(stmt.value)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if not _check_expr(target):
                    return False
            return _check_expr(stmt.value)
        if isinstance(stmt, ast.AugAssign):
            return _check_expr(stmt.target) and _check_expr(stmt.value)
        if isinstance(stmt, ast.AnnAssign):
            return _check_expr(stmt.target) and (stmt.value is None or _check_expr(stmt.value))
        if isinstance(stmt, ast.Return):
            return stmt.value is None or _check_expr(stmt.value)
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, (ast.Continue, ast.Break)):
            return True
        return False

    def _check_expr(expr):
        if isinstance(expr, ast.Call):
            # kernel[grid](...) 允许
            if isinstance(expr.func, ast.Subscript):
                name = _get_subscript_value_name(expr.func)
                if name in kernel_names or name in wrapper_names:
                    return True
            resolved = _resolve_call_name(expr)
            if resolved:
                qual, attr = resolved
                # torch.empty / torch.empty_like 等 buffer 分配允许
                if qual == "torch" and attr in ALLOWED_TORCH_FUNCS:
                    return True
                # 允许的 tensor 方法
                if attr in ALLOWED_TENSOR_METHODS and qual is not None:
                    return True
                # list(...) / tuple(...) / range(...) / len(...) 等 Python 内置允许
                if qual is None and attr in ("list", "tuple", "range", "len", "min", "max", "int", "float", "str", "enumerate", "zip"):
                    return True
                # math.prod / math.ceil 等 math 模块函数允许
                if qual == "math" and attr in ("prod", "ceil", "floor", "log2", "pow"):
                    return True
                # self.xxx(...) 仅允许已知安全的方法（如 _get_block_size）
                if qual == "self":
                    return True  # 放宽对 helper 方法的限制，由后续 torch/F 检测兜底
            # 递归检查参数
            for kw in expr.keywords:
                if not _check_expr(kw.value):
                    return False
            for arg in expr.args:
                if not _check_expr(arg):
                    return False
            return True
        if isinstance(expr, ast.Subscript):
            return _check_expr(expr.value) and _check_expr(expr.slice)
        if isinstance(expr, ast.Attribute):
            return _check_expr(expr.value)
        if isinstance(expr, ast.Name):
            return True
        if isinstance(expr, ast.Constant):
            return True
        if isinstance(expr, ast.BinOp):
            return _check_expr(expr.left) and _check_expr(expr.right)
        if isinstance(expr, ast.UnaryOp):
            return _check_expr(expr.operand)
        if isinstance(expr, ast.Compare):
            return all(_check_expr(e) for e in [expr.left] + expr.comparators)
        if isinstance(expr, ast.BoolOp):
            return all(_check_expr(v) for v in expr.values)
        if isinstance(expr, ast.IfExp):
            return _check_expr(expr.test) and _check_expr(expr.body) and _check_expr(expr.orelse)
        if isinstance(expr, (ast.Tuple, ast.List)):
            return all(_check_expr(e) for e in expr.elts)
        if isinstance(expr, ast.Dict):
            return all(_check_expr(k) for k in expr.keys) and all(_check_expr(v) for v in expr.values)
        if isinstance(expr, ast.Starred):
            return _check_expr(expr.value)
        return False

    for stmt in loop_node.body:
        if not _check_stmt(stmt):
            return False
    return True


def check_forbidden_torch_ops(forward_node, kernel_names, wrapper_names):
    """检查 forward 中是否使用了禁止的 torch 计算操作或 Python 控制流。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    # --- 规则 A: forward() 中禁止 Python 循环（for/while）---
    # 例外：如果循环体仅包含 kernel 启动和允许的 host 侧操作
    #      （如 transformation-memory 类算子的逐维度串行 kernel 启动）
    kernel_launch_count = _count_kernel_launches_in_forward(forward_node)

    # 预先收集所有"纯 kernel 启动循环"的节点 ID，主循环中跳过这些循环的子节点
    skip_node_ids = set()
    for node in ast.walk(forward_node):
        if isinstance(node, ast.For):
            if _is_loop_pure_kernel_launch(node, kernel_names, wrapper_names):
                for child in ast.walk(node):
                    skip_node_ids.add(id(child))

    for node in ast.walk(forward_node):
        if id(node) in skip_node_ids:
            continue
        if isinstance(node, ast.For):
            violations.append({
                "line": node.lineno,
                "call": "for 循环",
                "reason": "forward() 中禁止 Python for 循环，核心计算必须在单个 Triton kernel 内完成",
            })
            continue
        if isinstance(node, ast.While):
            violations.append({
                "line": node.lineno,
                "call": "while 循环",
                "reason": "forward() 中禁止 Python while 循环，核心计算必须在单个 Triton kernel 内完成",
            })
            continue

        # --- 检测 @ 运算符（矩阵乘法）---
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append({
                "line": node.lineno,
                "call": "@",
                "reason": "矩阵乘法 @ 运算符必须在 Triton kernel 中实现",
            })
            continue

        # --- 检测 list.append — 表明在 host 端维护动态状态 ---
        if isinstance(node, ast.Call):
            resolved = _resolve_call_name(node)
            if resolved:
                qual, attr = resolved
                if attr == "append" and qual is not None:
                    violations.append({
                        "line": node.lineno,
                        "call": f"{qual}.append(...)",
                        "reason": "forward() 中禁止 list.append，动态状态维护必须在 Triton kernel 内完成",
                    })
                    continue

        if not isinstance(node, ast.Call):
            continue

        # --- kernel launch: kernel[grid](...) —— 允许 ---
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
        # `qual is None` 表示 bare-name 调用（如 `max(64, n)` 的 Python 内建），
        # 不是 tensor 方法 — 跳过避免误伤 BLOCK_SIZE 这类标量 shape 数学。
        if qual is not None and attr in FORBIDDEN_TENSOR_METHODS:
            # 排除已知安全的 qual（torch/F/triton 已在上面处理）
            if qual not in ("torch", "F", "triton", "functional", "torch.nn.functional", "nn.functional"):
                violations.append({
                    "line": node.lineno,
                    "call": f"{qual}.{attr}()",
                    "reason": f"{attr} 是计算操作，必须在 Triton kernel 中实现",
                })
            continue

        # --- self.layer_name(x) —— 禁止 nn.Module 调用 ---
        # 但允许 self._route()（kernel 分裂调度器）和 self.forward() 递归
        if qual == "self":
            if attr not in ("forward", "_route"):
                violations.append({
                    "line": node.lineno,
                    "call": f"self.{attr}(...)",
                    "reason": f"self.{attr}() 疑似 nn.Module 前向调用，核心计算必须在 Triton kernel 中实现",
                })
            continue

    # --- 规则 B: 如果 forward() 中 kernel 启动次数 > 1，视为 Type3 退化 ---
    # 例外：如果每次启动的 kernel 都是已定义的 @triton.jit kernel，且循环体纯为 kernel 启动
    #      （如 transformation-memory 类算子的逐维度串行处理），不视为退化。
    if kernel_launch_count > 1:
        # 检查是否存在 host 侧循环仅包含 kernel 启动
        has_pure_launch_loop = False
        for node in ast.walk(forward_node):
            if isinstance(node, ast.For):
                if _is_loop_pure_kernel_launch(node, kernel_names, wrapper_names):
                    has_pure_launch_loop = True
                    break
        if not has_pure_launch_loop:
            violations.append({
                "line": forward_node.lineno,
                "call": f"kernel 启动 {kernel_launch_count} 次",
                "reason": "forward() 中只能启动一次 Triton kernel，多次启动表明核心计算在 host 端循环中完成",
            })

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
            "kernel_called_from_forward": {"passed": False, "called": [], "error": None},
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
    kernels = find_triton_kernels(tree)
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

    # --- Check 2: forward 是否调用 kernel ---
    forward_node = find_model_new_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew 类或 forward 方法。"
        return result

    wrapper_names = find_wrapper_functions(tree, kernel_names)
    called = check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names, class_name="ModelNew")
    result["checks"]["kernel_called_from_forward"]["called"] = list(called)

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"@triton.jit kernel {list(kernel_names)} 已定义但 forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已定义 kernel {list(kernel_names)} 但 ModelNew.forward() 中未调用。"
            "forward() 必须通过 kernel_name[grid](...) 形式启动 kernel。"
            f"{'也存在 wrapper 函数 ' + str(list(wrapper_names)) + ' 但 forward 也未调用它们。' if wrapper_names else ''}"
        )
        return result

    result["checks"]["kernel_called_from_forward"]["passed"] = True

    # --- Check 3: 禁止的 torch 操作 ---
    violations = check_forbidden_torch_ops(forward_node, kernel_names, wrapper_names)
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
            f"forward() 调用了 Triton kernel 但仍使用 PyTorch 进行部分计算: "
            f"{violation_details}。"
            "所有核心计算必须在 @triton.jit kernel 中完成，"
            "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
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
            called = result["checks"]["kernel_called_from_forward"]["called"]
            print("[PASS] Triton 实现验证通过")
            print(f"  - 发现 {len(kernels)} 个 @triton.jit kernel: {', '.join(k['name'] for k in kernels)}")
            print(f"  - forward() 调用: {', '.join(called)}")
            print("  - forward() 中无禁止的 PyTorch 计算操作")
        else:
            rtype = result["regression_type"]
            type_desc = {
                1: "完全无 Triton kernel（纯 PyTorch）",
                2: "有 Triton kernel 但 forward() 未调用",
                3: "部分计算使用 PyTorch（需全部移入 Triton kernel）",
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
