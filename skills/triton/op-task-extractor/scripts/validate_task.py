#!/usr/bin/env python3
"""KernelBench 任务代码验证脚本

验证代码是否符合 KernelBench 格式并通过运行时检查。

检查项目:
1. 静态: class Model(nn.Module), forward, get_inputs, get_init_inputs
2. 运行时: exec → Model() → forward() → NaN/Inf 检查 → 一致性检查

用法:
    python validate_task.py /abs/path/task_desc.py
    python validate_task.py /abs/path/task_desc.py --json
    python validate_task.py /abs/path/task_desc.py --static-only

输出格式:
    [VALID] 代码符合 KernelBench 格式
    [INVALID] 代码不符合格式 + 原因 + 修复建议
"""
import ast
import sys
import argparse
import json


def check_static(code: str) -> dict:
    """静态检查: 验证 KernelBench 四大组件是否存在"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "passed": False,
            "found": [],
            "missing": ["Model", "forward", "get_inputs", "get_init_inputs"],
            "error": f"SyntaxError: {e}",
        }

    has = {
        "Model": False,
        "forward": False,
        "get_inputs": False,
        "get_init_inputs": False,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for base in node.bases:
                base_name = getattr(base, "attr", getattr(base, "id", ""))
                if base_name == "Module":
                    has["Model"] = True
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "forward":
                            has["forward"] = True

        if isinstance(node, ast.FunctionDef) and node.name in (
            "get_inputs",
            "get_init_inputs",
        ):
            has[node.name] = True

    found = [k for k, v in has.items() if v]
    missing = [k for k, v in has.items() if not v]
    return {"passed": len(missing) == 0, "found": found, "missing": missing, "error": None}


def check_runtime(code: str) -> dict:
    """运行时检查: exec → Model() → forward() → NaN/Inf → 一致性"""
    checks = []
    namespace = {}

    try:
        exec(code, namespace)
        checks.append({"name": "exec", "passed": True})
    except Exception as e:
        checks.append({"name": "exec", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"exec error: {e}"}

    try:
        init_inputs = namespace["get_init_inputs"]()
        checks.append({"name": "get_init_inputs()", "passed": True})
    except Exception as e:
        checks.append({"name": "get_init_inputs()", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"get_init_inputs() error: {e}"}

    try:
        model = namespace["Model"](*init_inputs)
        checks.append({"name": "Model(*init_inputs)", "passed": True})
    except Exception as e:
        checks.append({"name": "Model(*init_inputs)", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"Model() error: {e}"}

    try:
        inputs = namespace["get_inputs"]()
        checks.append({"name": "get_inputs()", "passed": True})
    except Exception as e:
        checks.append({"name": "get_inputs()", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"get_inputs() error: {e}"}

    try:
        output = model(*inputs)
        checks.append({"name": "model(*inputs)", "passed": True})
    except Exception as e:
        checks.append({"name": "model(*inputs)", "passed": False, "error": str(e)})
        return {"passed": False, "checks": checks, "error": f"forward() error: {e}"}

    try:
        import torch

        def _check_tensor(t, name="output"):
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any():
                    return f"{name} contains NaN"
                if torch.isinf(t).any():
                    return f"{name} contains Inf"
            return None

        issues = []
        if isinstance(output, (tuple, list)):
            for i, item in enumerate(output):
                issue = _check_tensor(item, f"output[{i}]")
                if issue:
                    issues.append(issue)
        else:
            issue = _check_tensor(output)
            if issue:
                issues.append(issue)

        if issues:
            checks.append({"name": "NaN/Inf check", "passed": False, "error": "; ".join(issues)})
            return {"passed": False, "checks": checks, "error": "; ".join(issues)}
        checks.append({"name": "NaN/Inf check", "passed": True})
    except ImportError:
        checks.append({"name": "NaN/Inf check", "passed": True, "note": "torch not available, skipped"})

    try:
        output2 = model(*inputs)
        import torch

        def _tensors_close(a, b, rtol=1e-5, atol=1e-6):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                return all(_tensors_close(x, y) for x, y in zip(a, b))
            return True

        if _tensors_close(output, output2):
            checks.append({"name": "consistency check", "passed": True})
        else:
            checks.append({"name": "consistency check", "passed": False, "error": "outputs differ between runs"})
            return {"passed": False, "checks": checks, "error": "consistency check failed"}
    except Exception:
        checks.append({"name": "consistency check", "passed": True, "note": "skipped"})

    return {"passed": True, "checks": checks, "error": None}


def main():
    parser = argparse.ArgumentParser(
        description="验证代码是否符合 KernelBench 任务格式"
    )
    parser.add_argument("file", help="要验证的 Python 文件路径")
    parser.add_argument("--static-only", action="store_true", help="只做静态检查")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")

    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        if args.json:
            print(json.dumps({"valid": False, "error": f"File not found: {args.file}"}))
        else:
            print(f"[ERROR] 文件不存在: {args.file}")
        sys.exit(1)

    static_result = check_static(code)
    result = {
        "valid": False,
        "static_check": static_result,
        "runtime_check": None,
        "suggestion": "",
    }

    if not static_result["passed"]:
        result["error"] = static_result.get("error") or f"缺少组件: {', '.join(static_result['missing'])}"
        result["suggestion"] = "检查代码结构，确保包含 Model(nn.Module)、forward、get_inputs、get_init_inputs"
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[INVALID] 代码不符合 KernelBench 格式")
            print(f"缺少: {', '.join(static_result['missing'])}")
            print(f"建议: {result['suggestion']}")
        sys.exit(1)

    if not args.static_only:
        runtime_result = check_runtime(code)
        result["runtime_check"] = runtime_result

        if not runtime_result["passed"]:
            result["error"] = runtime_result["error"]
            result["suggestion"] = "检查代码逻辑，修复后重新验证"
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"[INVALID] 运行时检查失败")
                print(f"错误: {runtime_result['error']}")
                for check in runtime_result["checks"]:
                    status = "PASS" if check["passed"] else "FAIL"
                    print(f"  [{status}] {check['name']}")
            sys.exit(1)

    result["valid"] = True
    check_type = "静态" if args.static_only else "静态+运行时"

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[VALID] 代码符合 KernelBench 格式（{check_type}检查通过）")
        print(f"包含组件: {', '.join(static_result['found'])}")
    sys.exit(0)


if __name__ == "__main__":
    main()
