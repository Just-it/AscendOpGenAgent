#!/usr/bin/env python3
"""
prepare_strategy_resources.py — v3.2 Phase C7

为 partial-prompt 的 [STRATEGY RESOURCES] 段准备完整内容。
封装三步：filter-candidates → load_playbook → get-read-keys → 拼装。

接收一组策略 ID + 算子上下文，输出一段 markdown，可直接填入 partial-prompt
的 {strategy_resources_block} 变量。

用法（主 agent 启动 partial 前调用）：
    python3 .claude/skills/evolution-world-model/scripts/prepare_strategy_resources.py \\
        --strategy-ids "P1,P5,P10" \\
        --kernel-dir output/RmsNorm_ops-evo_<ts>/shared/original/ \\
        --evo-dir output/RmsNorm_ops-evo_<ts>/ \\
        --wm-path output/RmsNorm_ops-evo_<ts>/world_model.json \\
        --node-id n1 \\
        > /tmp/srblock_n1.md

  # 然后把 /tmp/srblock_n1.md 的内容填入 partial-prompt 的
  # {strategy_resources_block} 变量

退出码：
  0  success
  1  脚本调用失败（filter 或 load_playbook 任一失败）
  2  全部候选被 Preconditions 过滤掉（无 partial 任务可做）
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
SKILL_SCRIPTS = PROJECT_ROOT / ".claude/skills/evolution-strategies/scripts"
WM_OPS = SCRIPT_DIR / "wm_ops.py"
STATE_OPS = SCRIPT_DIR / "state_ops.py"


def _resolve_script(rel_path: str) -> Path:
    """Search project_root + cwd for the script."""
    for base in (PROJECT_ROOT, Path.cwd()):
        p = base / rel_path
        if p.exists():
            return p
    return Path(rel_path)


def step_1_filter(args, candidate_ids: list[str]) -> dict:
    """调用 wm_ops filter-candidates，返回 passed/failed/filtered_by_keys。"""
    cmd = [
        sys.executable, str(WM_OPS), "filter-candidates",
        "--candidate-ids", ",".join(candidate_ids),
        "--kernel-dir", args.kernel_dir,
    ]
    if args.baseline_eval:
        cmd.extend(["--baseline-eval", args.baseline_eval])
    if args.wm_path:
        cmd.extend(["--wm-path", args.wm_path])
    if args.node_id:
        cmd.extend(["--node-id", args.node_id])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"_error": "filter-candidates timed out", "passed": candidate_ids,
                "failed": [], "filtered_by_keys": []}

    if proc.returncode != 0:
        return {"_error": f"filter-candidates exit {proc.returncode}: {proc.stderr[:200]}",
                "passed": candidate_ids, "failed": [], "filtered_by_keys": []}

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"_error": "filter-candidates output not JSON",
                "passed": candidate_ids, "failed": [], "filtered_by_keys": []}


def step_2_load_playbook(passed_ids: list[str]) -> str:
    """调用 load_playbook 拿到 Playbook 全文。返回 markdown 字符串。"""
    if not passed_ids:
        return "（无采纳策略，跳过 Playbook 加载）"

    load_playbook_py = SKILL_SCRIPTS / "load_playbook.py"
    if not load_playbook_py.exists():
        return f"⚠️  load_playbook.py not found at {load_playbook_py}"

    cmd = [
        sys.executable, str(load_playbook_py),
        "--strategy-ids", ",".join(passed_ids),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "⚠️  load_playbook.py timed out"

    if proc.returncode != 0:
        return f"⚠️  load_playbook.py exit {proc.returncode}: {proc.stderr[:200]}"
    return proc.stdout.strip() or "（无对应 Playbook）"


def step_3_get_excluded(evo_dir: str) -> str:
    """调用 state_ops get-read-keys 拿 Excluded 段 markdown。"""
    if not evo_dir:
        return ""

    cmd = [
        sys.executable, str(STATE_OPS), "get-read-keys",
        "--evo-dir", evo_dir,
        "--format", "markdown",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def render_block(passed: list[str], failed: list[dict], filtered_by_keys: list[str],
                 playbook_md: str, excluded_md: str, node_id: str) -> str:
    """拼装最终 markdown。"""
    parts = []

    # 段 1: 采纳策略 + Preconditions 过滤结果
    parts.append("## Recommended Strategies（已通过 Preconditions 硬过滤）")
    parts.append("")
    if passed:
        parts.append(f"采纳策略 (Primary)：`{', '.join(passed)}`")
        parts.append("")
        parts.append("这些策略已通过适用性检查（BUFFER_NUM、shape、dtype 等条件），"
                     "你必须按下方 Playbook 严格执行，不要再质疑适用性。")
    else:
        parts.append("⚠️ **本变体所有候选策略均被 Preconditions 过滤掉**。")
        parts.append("可选行动：")
        parts.append("- 退回 open_exploration 模式（不依赖具体策略，从第一性原理推导）")
        parts.append("- 在 implementation_note.txt 报告原因，主 agent 在下一轮 refine 时决定")

    if filtered_by_keys:
        parts.append("")
        parts.append(f"被过滤的策略检查项（{len(filtered_by_keys)} 项，记录到 node.filtered_by 供事后分析）：")
        for k in filtered_by_keys:
            parts.append(f"- `{k}`")

    # 段 2: Playbook SOP 全文
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## Playbooks（每个 has_playbook=true 的策略附完整 SOP）")
    parts.append("")
    parts.append(playbook_md)

    # 段 3: Excluded（本 session 已读）
    if excluded_md and "暂无已读" not in excluded_md:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append(excluded_md)
        parts.append("")
        parts.append("**说明**：上述 source_key 你不需要再 Read，主 agent 已通过 prompt 注入相关内容。")

    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare {strategy_resources_block} for partial-prompt injection"
    )
    parser.add_argument("--strategy-ids", required=True,
                        help="逗号分隔的候选策略 ID（如 'P1,P5,P10'）")
    parser.add_argument("--kernel-dir", required=True,
                        help="算子源码目录（含 op_kernel/ 和 op_host/）")
    parser.add_argument("--evo-dir", default=None,
                        help="evo 输出根目录（用于读 state.read_keys）")
    parser.add_argument("--baseline-eval", default=None,
                        help="baseline_evaluation.json 路径")
    parser.add_argument("--wm-path", default=None,
                        help="world_model.json (配合 --node-id 写 filtered_by)")
    parser.add_argument("--node-id", default=None,
                        help="节点 ID（写 filtered_by）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出文件（默认 stdout）")
    args = parser.parse_args()

    candidate_ids = [s.strip() for s in args.strategy_ids.split(",") if s.strip()]
    if not candidate_ids:
        print("ERROR: --strategy-ids cannot be empty", file=sys.stderr)
        return 1

    # Step 1: filter
    filter_result = step_1_filter(args, candidate_ids)
    if "_error" in filter_result:
        print(f"WARN: filter step degraded: {filter_result['_error']}", file=sys.stderr)
    passed = filter_result.get("passed", candidate_ids)
    failed = filter_result.get("failed", [])
    filtered_by_keys = filter_result.get("filtered_by_keys", [])

    # Step 2: load playbooks
    playbook_md = step_2_load_playbook(passed)

    # Step 3: excluded
    excluded_md = step_3_get_excluded(args.evo_dir) if args.evo_dir else ""

    # Render
    block = render_block(passed, failed, filtered_by_keys,
                        playbook_md, excluded_md, args.node_id or "?")

    # Output
    if args.output:
        args.output.write_text(block, encoding="utf-8")
        print(f"Wrote {args.output} ({len(block)} chars)", file=sys.stderr)
    else:
        print(block)

    # Exit code
    if not passed:
        return 2  # all candidates filtered
    return 0


if __name__ == "__main__":
    sys.exit(main())
