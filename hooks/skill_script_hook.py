#!/usr/bin/env python3
"""
Skill 脚本执行 Hook (PreToolUse 协议)。

对 Bash 工具调用进行拦截：
- 命中 INTERCEPTED_PATTERNS: 直接由本 hook 执行脚本，并以 PreToolUse 协议返回
  permissionDecision=deny + additionalContext，把执行结果（exit_code/stdout/stderr）
  注入回 agent 的上下文，阻止 harness 二次执行。
- 不命中: 输出 permissionDecision=allow，让 harness 正常执行（不在 hook 中重复执行）。

输入：从 stdin 读取 harness PreToolUse JSON：
    {"tool_name": "Bash", "tool_input": {"command": "..."}, ...}

输出：单行 PreToolUse JSON 到 stdout。
"""

import json
import os
import re
import subprocess
import sys
import time


# 每个条目: (interpreter_regex, script_basename_regex)
# 只匹配“某个解释器 + 脚本路径”这种真正的脚本调用，
# 不匹配命令字符串里仅出现脚本名的情况（如 cat / grep / echo JSON 等）。
_PY_INTERP = r"(?:python|python3|python3\.\d+)"
_SH_INTERP = r"(?:bash|sh)"

INTERCEPTED_PATTERNS = [
    (_PY_INTERP, r"validate_tilelang_impl\.py"),
    (_PY_INTERP, r"validate_ascendc_impl\.py"),
    (_SH_INTERP, r"evaluate_tilelang\.sh"),
    (_SH_INTERP, r"evaluate_ascendc\.sh"),
    (_PY_INTERP, r"performance\.py"),
    (_SH_INTERP, r"batch_run_performance\.sh"),
    (_PY_INTERP, r"build_ascendc\.py"),
    (_PY_INTERP, r"verification_ascendc\.py"),
    (_PY_INTERP, r"verification_tilelang\.py"),
]

# 命令前缀允许的部分（环境变量赋值、cd ... &&、source ... &&）
# 这里只是用于推断"真实的脚本调用"，不需要覆盖所有 shell 语法。
_LEADING_PREFIX = (
    r"^\s*"
    r"(?:export\s+)?"                                 # 可选的 export 关键字
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"           # ENV=VAL ...
    r"(?:cd\s+\S+\s*&&\s*)?"                          # cd <dir> &&
    r"(?:export\s+)?"                                 # 可选的 export 关键字
    r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"           # 再次允许 ENV=VAL
    r"(?:&&\s+)*"                                     # 支持 && 链式连接
)

# 项目根目录：优先从环境变量获取，fallback 到 __file__ 计算
# 避免 hook 被以相对路径调用时 __file__ 解析错误
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def should_intercept(command: str) -> bool:
    # 仅当命令的"实际执行段"是 <interpreter> <path-to-script> 形式时才拦截。
    # 兼容：环境变量前缀、可选的 `cd <dir> &&` 前缀。
    for interp, script in INTERCEPTED_PATTERNS:
        # 路径里允许出现 / 和非空白字符；脚本名用 \b 边界
        pattern = (
            _LEADING_PREFIX
            + interp
            + r"\s+"
            + r"(?:\S*/)?"
            + script
            + r"(?:\s|$)"
        )
        if re.search(pattern, command):
            return True
    return False


def emit_pretooluse(permission_decision: str, *, reason: str = "", additional_context: str = ""):
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": permission_decision,
        }
    }
    if reason:
        output["hookSpecificOutput"]["permissionDecisionReason"] = reason
    if additional_context:
        output["hookSpecificOutput"]["additionalContext"] = additional_context
    print(json.dumps(output, ensure_ascii=False), flush=True)


def truncate(text: str, limit: int = 8000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n... [truncated {len(text) - limit} chars] ...\n\n{tail}"


def execute_intercepted(command: str) -> None:
    start_time = time.time()
    cwd = os.getcwd()

    # 解析路径：若是相对路径且在 cwd 不存在，则尝试 PROJECT_ROOT 下解析
    # 这里我们直接通过 shell 执行原命令，但先确保 cwd 是 PROJECT_ROOT，
    # 这样像 "python skills/..." 之类的相对路径能正确解析。
    if os.path.isdir(PROJECT_ROOT):
        cwd = PROJECT_ROOT

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        exit_code = 124
        stdout = (e.stdout.decode("utf-8", errors="replace") if e.stdout else "")
        stderr = (e.stderr.decode("utf-8", errors="replace") if e.stderr else "")
        stderr += "\n[HOOK] 命令执行超时（30 分钟）"
    except Exception as e:
        exit_code = 1
        stdout = ""
        stderr = f"[HOOK] 执行异常: {type(e).__name__}: {e}"

    duration_ms = int((time.time() - start_time) * 1000)

    additional_context = (
        f"[skill_script_hook intercepted execution]\n"
        f"command: {command}\n"
        f"cwd: {cwd}\n"
        f"exit_code: {exit_code}\n"
        f"duration_ms: {duration_ms}\n"
        f"--- stdout ---\n{truncate(stdout)}\n"
        f"--- stderr ---\n{truncate(stderr)}\n"
    )

    status = "成功" if exit_code == 0 else "失败"
    reason = (
        f"[Hook 拦截提示] 该命令命中 skill_script_hook 拦截规则，已由 hook 代为执行（{status}，exit_code={exit_code}）。"
        f"执行结果见下方 additionalContext，原命令已被替换为 no-op，不会重复执行。"
    )

    # 使用 allow + updatedInput 将原命令替换为 no-op，避免 harness 显示 "Error:" 前缀
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason,
            "additionalContext": additional_context,
            "updatedInput": {
                "command": f"echo '[hook-noop] original command was intercepted and executed by skill_script_hook'"
            },
        }
    }
    print(json.dumps(output, ensure_ascii=False), flush=True)


def read_stdin_command():
    try:
        raw = sys.stdin.read()
    except Exception:
        return None, None
    if not raw or not raw.strip():
        return None, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    tool_name = data.get("tool_name")
    tool_input = data.get("tool_input") or {}
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    return tool_name, command


def main():
    tool_name, command = read_stdin_command()

    # 没有命令或不是 Bash —— allow，由 harness 处理
    if not command or tool_name != "Bash":
        emit_pretooluse("allow")
        return

    if should_intercept(command):
        execute_intercepted(command)
    else:
        # 不拦截 —— 让 harness 正常处理（不在此重复执行）
        emit_pretooluse("allow")


if __name__ == "__main__":
    main()
