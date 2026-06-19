#!/usr/bin/env python3
"""
bottleneck_to_sources.py — Stage 3 CLI 入口

把 Stage 2 LLM 给出的 bottleneck_labels 列表反查 INDEX.json，
输出 candidate_source_keys 供 wm_ops select 使用。

设计文档：docs/design/knowledge-strategy-architecture-v3.2.md §3.4 Stage 3
关联库函数：profiling_evidence.match_strategies_by_labels

用法：
    # 从 JSON 文件读 labels
    python3 bottleneck_to_sources.py --labels-file diagnosis.json

    # 命令行直接传 labels
    python3 bottleneck_to_sources.py --labels mte2_stall no_overlap

    # 限制候选数量
    python3 bottleneck_to_sources.py --labels mte2_stall --limit 10

    # 输出到指定文件（默认 stdout）
    python3 bottleneck_to_sources.py --labels mte2_stall --output candidates.json
"""

import argparse
import json
import sys
from pathlib import Path

# 让 sibling import 能 work（脚本既可以在 .claude/skills/evolution-world-model/scripts/ 跑，也可以在 skill 副本跑）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from profiling_evidence import match_strategies_by_labels, validate_labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按 bottleneck_labels 反查 evolution-strategies INDEX.json",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--labels",
        nargs="+",
        help="直接给出 bottleneck_labels 列表（空格分隔）",
    )
    src.add_argument(
        "--labels-file",
        type=Path,
        help="JSON 文件，含 bottleneck_labels 字段（list of str）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="candidate_source_keys 返回数量上限（默认全返回）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径（默认 stdout）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：unknown labels 时退出码 2（默认 0 但在输出中标注）",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="人类友好的摘要（默认输出完整 JSON）",
    )
    args = parser.parse_args()

    # 解析 labels
    if args.labels:
        labels = args.labels
    else:
        try:
            data = json.loads(args.labels_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR: cannot read labels-file: {e}", file=sys.stderr)
            return 1
        # 支持两种字段名
        labels = data.get("bottleneck_labels") or data.get("labels") or []
        if not labels:
            print("ERROR: labels-file must contain non-empty 'bottleneck_labels' or 'labels' field",
                  file=sys.stderr)
            return 1

    # 校验
    valid = validate_labels(labels)
    if not valid["valid"] and args.strict:
        print(f"ERROR: unknown labels in strict mode: {valid['unknown']}", file=sys.stderr)
        print(f"       valid vocabulary size = {valid['vocabulary_size']}", file=sys.stderr)
        return 2

    # 反查
    result = match_strategies_by_labels(labels, limit=args.limit, include_unknown=True)

    # 输出
    if args.summary:
        text = (
            f"Input labels: {labels}\n"
            f"Unknown labels: {result.get('unknown_labels', [])}\n"
            f"Candidate count: {len(result['candidate_source_keys'])}\n"
            f"Top candidates (by hit count):\n"
        )
        for sid, sk in zip(result["candidate_ids"], result["candidate_source_keys"]):
            text += f"  - {sid:5}  {sk}\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)
    else:
        output = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
