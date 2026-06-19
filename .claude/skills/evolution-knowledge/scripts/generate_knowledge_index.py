#!/usr/bin/env python3
"""
generate_knowledge_index.py — 生成 evolution-knowledge skill 的 INDEX.json

不像 evolution-strategies 有 frontmatter，evolution-knowledge 是纯文档库，
INDEX.json 仅记录路径、kind、所属 category，方便 source_key 反查和 LLM 导航。
"""

import json
from pathlib import Path

SKILL_ROOT = Path(".claude/skills/evolution-knowledge/references")
INDEX_PATH = SKILL_ROOT / "INDEX.json"

SKILL = "evolution-knowledge"

KIND_DIRS = {
    "a3": SKILL_ROOT / "a3",
    "a5": SKILL_ROOT / "a5",
}


def discover():
    entries = []
    for kind, root in KIND_DIRS.items():
        if not root.exists():
            continue
        for md in sorted(root.rglob("*.md")):
            if md.name == "SOURCE_KEY.md":
                continue
            rel = md.relative_to(SKILL_ROOT).as_posix()
            # source_key 用不带后缀的相对路径作为标识
            key_path = rel[:-3]  # strip ".md"
            # 跳过 a3 / a5 顶层 INDEX.md 自身
            entry = {
                "source_key": f"{SKILL}#{key_path}",
                "kind": kind,
                "category": md.parent.name if md.parent != root else None,
                "is_index": md.name == "INDEX.md",
                "is_guide": md.name == "guide.md",
                "doc_path": rel,
            }
            entries.append(entry)
    return entries


def main():
    entries = discover()
    by_kind = {}
    for e in entries:
        by_kind.setdefault(e["kind"], 0)
        by_kind[e["kind"]] += 1

    index = {
        "skill": SKILL,
        "version": "1.0",
        "architectures": ["a3", "a5"],
        "categories": {
            "a3": ["hardware", "algorithm_insights", "ascendc_api",
                   "optimization_patterns", "proven_solutions", "profiling_reference"],
            "a5": ["hardware", "regbase_api", "vf_programming",
                   "optimization_patterns", "translation_rules"],
        },
        "stats": {**by_kind, "total": len(entries)},
        "entries": entries,
    }
    INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {INDEX_PATH}: total={len(entries)} ({by_kind})")


if __name__ == "__main__":
    main()
