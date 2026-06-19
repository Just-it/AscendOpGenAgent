#!/usr/bin/env python3
"""check_preconditions.py — 策略 Preconditions 执行器

读取策略的 Preconditions YAML，对当前算子代码 / profiling 数据执行硬性检查。
任一 check 失败则该策略被跳过。

用法:
    python3 check_preconditions.py \\
        --strategy-ids P1,P5,P14 \\
        --kernel-dir output/fastgelu/shared/original \\
        --baseline-eval output/fastgelu/baseline_evaluation.json \\
        --output /tmp/precond_result.json

输出 JSON:
    {
      "P1":  {"passed": true,  "failed_checks": []},
      "P5":  {"passed": true,  "failed_checks": []},
      "P14": {"passed": false, "failed_checks": [
         {"id": "has_cube_vector_fusion", "fail_msg": "算子无 Cube+Vector 双核结构..."}
      ]}
    }

设计要点:
- 无 PyYAML 依赖（手写轻量 parser）
- 无 Preconditions 文件时默认通过（向后兼容现有策略）
- grep 基于 Python re 实现，避免 shell 转义
"""
import argparse
import glob
import json
import os
import re
import sys


PRECONDITIONS_DIR = ".claude/skills/evolution-strategies/references/preconditions"


def _unquote_yaml_scalar(raw: str) -> str:
    """处理 YAML 标量：区分双引号字符串（支持 escape）和裸字符串。"""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        # 双引号字符串：处理 \\, \", \n, \t 等转义
        inner = raw[1:-1]
        result = []
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == "\\" and i + 1 < len(inner):
                nxt = inner[i + 1]
                if nxt == "\\":
                    result.append("\\")
                elif nxt == '"':
                    result.append('"')
                elif nxt == "n":
                    result.append("\n")
                elif nxt == "t":
                    result.append("\t")
                else:
                    # 保留原样（含 \s, \d, \w 等供正则使用）
                    result.append(ch)
                    result.append(nxt)
                i += 2
            else:
                result.append(ch)
                i += 1
        return "".join(result)
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        # 单引号字符串：内容原样，仅 '' 转义为 '
        return raw[1:-1].replace("''", "'")
    # 裸字符串：原样返回
    return raw


def parse_yaml_preconditions(content: str) -> dict:
    """解析 Preconditions YAML（轻量，只支持本项目的格式）。"""
    result = {"strategy_id": "", "description": "", "checks": []}
    current_check = None

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # 顶级字段: strategy_id / description
        m = re.match(r"^(strategy_id|description):\s*(.*)$", line)
        if m:
            key, raw_val = m.group(1), m.group(2)
            result[key] = _unquote_yaml_scalar(raw_val)
            i += 1
            continue

        # checks: 开始
        if stripped == "checks:":
            i += 1
            continue

        # 每个 check 项开头:  - id: xxx
        m = re.match(r"^\s+-\s+id:\s*(.*)$", line)
        if m:
            if current_check:
                result["checks"].append(current_check)
            current_check = {"id": _unquote_yaml_scalar(m.group(1))}
            i += 1
            continue

        # check 的属性: type / pattern / files / expected / fail_msg
        m = re.match(r"^\s+([a-z_]+):\s*(.*)$", line)
        if m and current_check is not None:
            key, raw_val = m.group(1), m.group(2)
            raw_val = raw_val.strip()
            if raw_val.startswith("[") and raw_val.endswith("]"):
                # 行内列表
                inner = raw_val[1:-1].strip()
                items = [_unquote_yaml_scalar(x) for x in inner.split(",")] if inner else []
                current_check[key] = [x for x in items if x]
            else:
                current_check[key] = _unquote_yaml_scalar(raw_val)
            i += 1
            continue

        i += 1

    if current_check:
        result["checks"].append(current_check)

    return result


# ---- 文件集解析 ----
FILE_GROUPS = {
    "kernel":  ["op_kernel/*.cpp", "op_kernel/*.h"],
    "header":  ["op_kernel/*.h", "op_host/*.h"],
    "tiling":  ["op_host/*_tiling.cpp", "op_host/*_tiling.h"],
    "all":     ["op_kernel/*.cpp", "op_kernel/*.h", "op_host/*.cpp", "op_host/*.h"],
}


def resolve_files(file_groups: list, root_dir: str) -> list:
    """将 files 关键字列表解析为实际文件路径列表。

    校验：root_dir 必须存在，且至少有 op_kernel 或 op_host 子目录之一。
    不满足抛 FileNotFoundError，避免"静默返回空 list 导致所有 check 失败"的误判。
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"kernel_dir 不存在: {root_dir}（请确认 --kernel-dir 参数是否正确）"
        )
    has_op_kernel = os.path.isdir(os.path.join(root_dir, "op_kernel"))
    has_op_host = os.path.isdir(os.path.join(root_dir, "op_host"))
    if not (has_op_kernel or has_op_host):
        raise FileNotFoundError(
            f"kernel_dir={root_dir} 下既无 op_kernel/ 也无 op_host/ 子目录 "
            f"（期望布局：{root_dir}/op_kernel/*.cpp, {root_dir}/op_host/*_tiling.cpp）"
        )

    patterns = []
    for g in file_groups:
        patterns.extend(FILE_GROUPS.get(g, []))
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(os.path.join(root_dir, pat)))
    return sorted(set(all_files))


# ---- Check 执行 ----
def exec_grep_count(check: dict, kernel_dir: str) -> tuple[bool, str]:
    pattern = check.get("pattern", "")
    files = resolve_files(check.get("files", []), kernel_dir)
    if not files:
        return False, f"grep_count: 无匹配文件集（files={check.get('files')}）"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return False, f"grep_count: 正则错误 '{pattern}': {e}"

    count = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if regex.search(line):
                        count += 1
        except OSError:
            pass

    return _assert_expr(count, check.get("expected", "")), f"count={count}"


def exec_grep_value(check: dict, kernel_dir: str) -> tuple[bool, str]:
    pattern = check.get("pattern", "")
    files = resolve_files(check.get("files", []), kernel_dir)
    expected_value = str(check.get("expected_value", ""))

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return False, f"grep_value: 正则错误 '{pattern}': {e}"

    first_val = None
    for fp in files:
        if first_val is not None:
            break
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = regex.search(line)
                    if m and m.groups():
                        first_val = m.group(1).strip()
                        break
        except OSError:
            pass

    if first_val is None:
        return False, f"grep_value: 未找到匹配（pattern={pattern}）"
    return first_val == expected_value, f"value='{first_val}', expected='{expected_value}'"


def exec_profiling_metric(check: dict, baseline_eval_path: str) -> tuple[bool, str]:
    if not baseline_eval_path or not os.path.isfile(baseline_eval_path):
        # 无 profiling 数据时默认通过
        return True, "skipped: no baseline_evaluation.json"

    try:
        with open(baseline_eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True, "skipped: baseline_evaluation.json unreadable"

    metric_path = check.get("metric", "")
    val = data
    for key in metric_path.split("."):
        if not isinstance(val, dict) or key not in val:
            return True, f"skipped: metric '{metric_path}' not in baseline_evaluation.json"
        val = val[key]

    return _assert_expr(val, check.get("expected", "")), f"metric={val}"


def _assert_expr(actual, expected_expr: str) -> bool:
    """解析 '== N' / '>= N' / '> N' / '<= N' / '< N' / 'in [a, b, c]' 。"""
    expr = expected_expr.strip()

    # 尝试数值比较
    m = re.match(r"^(==|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$", expr)
    if m:
        op, rhs_s = m.group(1), m.group(2)
        try:
            lhs = float(actual)
            rhs = float(rhs_s)
        except (TypeError, ValueError):
            return False
        return {
            "==": lhs == rhs,
            ">=": lhs >= rhs,
            "<=": lhs <= rhs,
            ">":  lhs > rhs,
            "<":  lhs < rhs,
        }[op]

    # 尝试 in [...]
    m = re.match(r"^in\s*\[(.+)\]\s*$", expr)
    if m:
        items = [x.strip().strip('"').strip("'") for x in m.group(1).split(",")]
        return str(actual) in items

    # 裸字符串精确匹配
    return str(actual) == expr


# ---- 主流程 ----
def check_strategy(strategy_id: str, precond_dir: str, kernel_dir: str,
                    baseline_eval_path: str) -> dict:
    yaml_path = os.path.join(precond_dir, f"{strategy_id}.yaml")
    if not os.path.isfile(yaml_path):
        # 无 Preconditions 文件 → 默认通过（向后兼容）
        return {"passed": True, "failed_checks": [], "note": "no_preconditions_file"}

    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = parse_yaml_preconditions(f.read())

    failed = []
    for ch in spec.get("checks", []):
        ch_type = ch.get("type", "")
        try:
            if ch_type == "grep_count":
                ok, detail = exec_grep_count(ch, kernel_dir)
            elif ch_type == "grep_value":
                ok, detail = exec_grep_value(ch, kernel_dir)
            elif ch_type == "profiling_metric":
                ok, detail = exec_profiling_metric(ch, baseline_eval_path)
            else:
                ok, detail = False, f"unknown check type: {ch_type}"
        except Exception as e:
            ok, detail = False, f"exception: {e}"

        if not ok:
            failed.append({
                "id": ch.get("id", "unknown"),
                "type": ch_type,
                "detail": detail,
                "fail_msg": ch.get("fail_msg", ""),
            })

    return {
        "passed": len(failed) == 0,
        "failed_checks": failed,
    }


def main():
    parser = argparse.ArgumentParser(description="策略 Preconditions 执行器")
    parser.add_argument("--strategy-ids", required=True,
                        help="逗号分隔的策略 ID 列表")
    parser.add_argument("--kernel-dir", required=True,
                        help="算子源码根目录（含 op_kernel/ 和 op_host/ 子目录）")
    parser.add_argument("--baseline-eval", default="",
                        help="baseline_evaluation.json 路径（profiling_metric 需要）")
    parser.add_argument("--precond-dir", default=PRECONDITIONS_DIR,
                        help="Preconditions YAML 目录")
    parser.add_argument("--output", default="-",
                        help="输出 JSON 路径（- 为 stdout）")
    parser.add_argument("--summary", action="store_true",
                        help="只打印简短摘要（passed ID 列表 + failed ID 列表）")
    args = parser.parse_args()

    # 预检 kernel-dir（任一 check 调用 resolve_files 前先整体校验，失败直接 FATAL 退出）
    if not os.path.isdir(args.kernel_dir):
        print(f"FATAL: --kernel-dir 不存在: {args.kernel_dir}", file=sys.stderr)
        sys.exit(2)
    has_op_kernel = os.path.isdir(os.path.join(args.kernel_dir, "op_kernel"))
    has_op_host = os.path.isdir(os.path.join(args.kernel_dir, "op_host"))
    if not (has_op_kernel or has_op_host):
        print(f"FATAL: --kernel-dir={args.kernel_dir} 下既无 op_kernel/ 也无 op_host/ 子目录",
              file=sys.stderr)
        print(f"       期望布局：{args.kernel_dir}/op_kernel/*.cpp, {args.kernel_dir}/op_host/*_tiling.cpp",
              file=sys.stderr)
        sys.exit(2)

    strategy_ids = [x.strip() for x in args.strategy_ids.split(",") if x.strip()]
    if not strategy_ids:
        print("FATAL: --strategy-ids 为空", file=sys.stderr)
        sys.exit(2)

    results = {}
    for sid in strategy_ids:
        try:
            results[sid] = check_strategy(sid, args.precond_dir, args.kernel_dir,
                                           args.baseline_eval)
        except FileNotFoundError as e:
            print(f"FATAL: check_strategy 内路径错误: {e}", file=sys.stderr)
            sys.exit(2)

    if args.summary:
        passed = [sid for sid, r in results.items() if r["passed"]]
        failed = [sid for sid, r in results.items() if not r["passed"]]
        print(f"passed: {passed}")
        print(f"failed: {failed}")
        for sid in failed:
            for fc in results[sid]["failed_checks"]:
                print(f"  {sid} / {fc['id']}: {fc['fail_msg']}")
        return

    out_json = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(out_json)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json)
        passed_count = sum(1 for r in results.values() if r["passed"])
        print(f"Preconditions checked: {passed_count}/{len(results)} passed. Written to {args.output}")


if __name__ == "__main__":
    main()
