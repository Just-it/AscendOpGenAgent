#!/usr/bin/env python3
"""load_playbook.py — 按 ID 加载 Playbook markdown，拼接为纯文本

用途:
    主 agent 填充 ops-partial prompt 前，按 node.strategy_combination 加载
    命中的 Playbook，注入到子 agent prompt 的 [STRATEGY PLAYBOOKS] 段落。

用法:
    python3 load_playbook.py \\
        --strategy-ids P1,P14,P99 \\
        --output /tmp/playbooks.txt

    # P99 无 Playbook 时不报错（只加载已存在的）
    # 输出格式：各 Playbook 之间用 '\n\n---\n\n' 分隔

输出:
    如果至少加载了一个 Playbook：写入 output，stdout 报告加载数
    如果全部都无 Playbook：output 为空文件，stdout 报告 "no_playbooks_loaded"
"""
import argparse
import glob
import os
import sys


PLAYBOOK_DIR = ".claude/skills/evolution-strategies/references/playbooks"


def find_playbook(strategy_id: str, playbook_dir: str) -> str | None:
    """查找策略对应的 Playbook 文件。

    支持：
    1. 精确前缀匹配: {ID}_*.md
    2. 兼容: {ID}.md
    """
    for pattern in (f"{strategy_id}_*.md", f"{strategy_id}.md"):
        matches = glob.glob(os.path.join(playbook_dir, pattern))
        if matches:
            return matches[0]
    return None


def load_playbook_text(strategy_id: str, playbook_dir: str) -> tuple[str, str]:
    """加载单个 Playbook 的内容。

    返回 (content, source_path)，若无 Playbook 则返回 ("", "")。
    """
    path = find_playbook(strategy_id, playbook_dir)
    if not path:
        return "", ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), path
    except OSError:
        return "", ""


def main():
    parser = argparse.ArgumentParser(description="按 ID 加载 Playbook")
    parser.add_argument("--strategy-ids", required=True,
                        help="逗号分隔的策略 ID 列表")
    parser.add_argument("--playbook-dir", default=PLAYBOOK_DIR,
                        help="Playbook 目录")
    parser.add_argument("--output", default="-",
                        help="输出文件路径（- 为 stdout）")
    parser.add_argument("--list-only", action="store_true",
                        help="只列出命中的 Playbook ID，不输出内容")
    args = parser.parse_args()

    strategy_ids = [x.strip() for x in args.strategy_ids.split(",") if x.strip()]

    loaded = []
    missing = []
    chunks = []
    for sid in strategy_ids:
        content, path = load_playbook_text(sid, args.playbook_dir)
        if content:
            loaded.append((sid, path))
            # 用显式 BEGIN/END 标记包裹（Playbook 正文含 '---' 用作分隔线，避免歧义）
            chunks.append(
                f"===== PLAYBOOK {sid} BEGIN =====\n\n"
                f"{content.strip()}\n\n"
                f"===== PLAYBOOK {sid} END ====="
            )
        else:
            missing.append(sid)

    if args.list_only:
        print(f"loaded: {[sid for sid, _ in loaded]}")
        print(f"missing: {missing}")
        return

    if not chunks:
        # 无任何 Playbook 命中：输出空文件 + stderr 说明
        if args.output != "-":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("")
        print(f"no_playbooks_loaded (requested: {strategy_ids})", file=sys.stderr)
        return

    combined = "\n\n".join(chunks)

    if args.output == "-":
        print(combined)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(combined)

    loaded_ids = [sid for sid, _ in loaded]
    print(f"Loaded {len(loaded)} playbooks: {loaded_ids}. Missing: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
