#!/usr/bin/env python3
"""
backfill_resource_flags.py — 一次性脚本：给所有策略卡补全 has_preconditions / has_playbook 字段

扫描 preconditions/ 和 playbooks/ 目录，根据策略 ID 判断对应资源是否存在，
然后在每张 card 的 frontmatter 的 synergizes_with 后插入这两个字段。

幂等：如果字段已存在则跳过。
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

SKILL_ROOT = Path(".claude/skills/evolution-strategies/references")
CARDS_DIR = SKILL_ROOT / "cards"
PRECOND_DIR = SKILL_ROOT / "preconditions"
PLAYBOOK_DIR = SKILL_ROOT / "playbooks"

ID_RE = re.compile(r"^id:\s*(\S+)\s*$", re.MULTILINE)
SYNERGIZES_LINE_RE = re.compile(r"^synergizes_with:.*$", re.MULTILINE)
HAS_PRECOND_RE = re.compile(r"^has_preconditions:", re.MULTILINE)
HAS_PB_RE = re.compile(r"^has_playbook:", re.MULTILINE)


def discover_resources():
    precond_ids = {p.stem for p in PRECOND_DIR.glob("*.yaml")}
    playbook_ids = set()
    for p in PLAYBOOK_DIR.glob("*.md"):
        # 文件名形如 P1_double_buffering.md → 提取 P1
        stem = p.stem
        if "_" in stem:
            playbook_ids.add(stem.split("_", 1)[0])
        else:
            playbook_ids.add(stem)
    return precond_ids, playbook_ids


def split_frontmatter(text: str):
    """返回 (frontmatter, body)。无 frontmatter 返回 (None, text)。"""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    fm = text[4:end]
    body = text[end + 5 :]
    return fm, body


def patch_card(card_path: Path, precond_ids: set[str], playbook_ids: set[str],
               force: bool = False) -> bool:
    text = card_path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(text)
    if fm is None:
        print(f"  SKIP {card_path.name}: no frontmatter")
        return False

    # 跳过元数据文件（CONTRIBUTING/SCHEMA）
    id_match = ID_RE.search(fm)
    if not id_match:
        return False
    sid = id_match.group(1).strip()

    has_precond_existed = bool(HAS_PRECOND_RE.search(fm))
    has_pb_existed = bool(HAS_PB_RE.search(fm))

    if has_precond_existed and has_pb_existed and not force:
        return False  # already done

    has_precond = sid in precond_ids
    has_pb = sid in playbook_ids

    if force and (has_precond_existed or has_pb_existed):
        # Force mode: 替换现有字段值（用 sed-like 替换）
        if has_precond_existed:
            fm = re.sub(
                r"^has_preconditions:.*$",
                f"has_preconditions: {str(has_precond).lower()}",
                fm,
                flags=re.MULTILINE,
            )
        else:
            # 字段不存在，落回原插入逻辑
            synergizes_match = SYNERGIZES_LINE_RE.search(fm)
            line = f"has_preconditions: {str(has_precond).lower()}"
            if synergizes_match:
                fm = fm[: synergizes_match.end()] + "\n" + line + fm[synergizes_match.end() :]
            else:
                fm = fm.rstrip("\n") + "\n" + line + "\n"

        if has_pb_existed:
            fm = re.sub(
                r"^has_playbook:.*$",
                f"has_playbook: {str(has_pb).lower()}",
                fm,
                flags=re.MULTILINE,
            )
        else:
            synergizes_match = SYNERGIZES_LINE_RE.search(fm)
            line = f"has_playbook: {str(has_pb).lower()}"
            if synergizes_match:
                fm = fm[: synergizes_match.end()] + "\n" + line + fm[synergizes_match.end() :]
            else:
                fm = fm.rstrip("\n") + "\n" + line + "\n"

        new_text = f"---\n{fm}\n---\n{body}"
        card_path.write_text(new_text, encoding="utf-8")
        return True

    # Non-force original logic
    inserts = []
    if not has_precond_existed:
        inserts.append(f"has_preconditions: {str(has_precond).lower()}")
    if not has_pb_existed:
        inserts.append(f"has_playbook: {str(has_pb).lower()}")

    inserts_block = "\n".join(inserts)

    synergizes_match = SYNERGIZES_LINE_RE.search(fm)
    if synergizes_match:
        new_fm = (
            fm[: synergizes_match.end()]
            + "\n"
            + inserts_block
            + fm[synergizes_match.end() :]
        )
    else:
        new_fm = fm.rstrip("\n") + "\n" + inserts_block + "\n"

    new_text = f"---\n{new_fm}\n---\n{body}"
    card_path.write_text(new_text, encoding="utf-8")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill has_preconditions / has_playbook")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖现有 has_* 字段（基于当前文件系统）")
    args = parser.parse_args()

    precond_ids, playbook_ids = discover_resources()
    print(f"Preconditions IDs: {sorted(precond_ids)}")
    print(f"Playbook IDs:      {sorted(playbook_ids)}")
    print()

    cards = sorted(CARDS_DIR.glob("*.md"))
    # 排除非策略卡
    cards = [c for c in cards if c.name not in {"CONTRIBUTING.md", "SCHEMA.md"}]

    patched = 0
    for card in cards:
        if patch_card(card, precond_ids, playbook_ids, force=args.force):
            patched += 1

    print(f"Patched {patched}/{len(cards)} cards (force={args.force})")


if __name__ == "__main__":
    main()
