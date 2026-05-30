#!/usr/bin/env python3
"""
历史经验增删改查工具（exp-edit.py）。

对 .claude/memory/kernel-opt-{category}.md 中的 Layer 1-3 经验条目进行管理。

用法:
    python3 utils/exp-edit.py list
    python3 utils/exp-edit.py show <category> [--layer N] [--id ID]
    python3 utils/exp-edit.py add <category> --layer N --title TITLE [--content TEXT] [--why TEXT] [--apply TEXT]
    python3 utils/exp-edit.py update <category> --id ID [--title TEXT] [--content TEXT] [--why TEXT] [--apply TEXT]
    python3 utils/exp-edit.py remove <category> --id ID
    python3 utils/exp-edit.py undo <category>
    python3 utils/exp-edit.py history <category>

高危操作（update/remove）默认交互确认；Agent 可附加 --yes 跳过。
"""

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class Entry:
    id: str
    layer: int
    title: str
    raw_title_line: str
    body_lines: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0


class ExperienceStore:
    def __init__(self, category: str):
        self.category = category.lower()
        self.root = get_project_root()
        self.memory_dir = self.root / '.claude' / 'memory'
        self.file_path = self.memory_dir / f'kernel-opt-{self.category}.md'
        self.backup_dir = self.memory_dir / '.backup'
        self.history_path = self.backup_dir / f'{self.category}_history.jsonl'
        self.lines: List[str] = []
        self.frontmatter_end = -1
        self.entries: List[Entry] = []
        self._load()

    def exists(self) -> bool:
        return self.file_path.exists()

    def _load(self):
        if not self.exists():
            return
        text = self.file_path.read_text(encoding='utf-8')
        self.lines = text.splitlines()
        self._parse()

    def _parse(self):
        self.entries = []
        self.frontmatter_end = -1
        if len(self.lines) >= 2 and self.lines[0].strip() == '---':
            for i in range(1, len(self.lines)):
                if self.lines[i].strip() == '---':
                    self.frontmatter_end = i
                    break

        current_layer = 0
        i = self.frontmatter_end + 1 if self.frontmatter_end >= 0 else 0
        while i < len(self.lines):
            line = self.lines[i]
            # Detect Layer section
            m = re.match(r'^## Layer\s+(\d+)\s*:', line)
            if m:
                current_layer = int(m.group(1))
                i += 1
                continue

            # Detect entry
            if current_layer in (1, 2, 3) and line.startswith('### '):
                entry = self._parse_entry(i, current_layer)
                if entry:
                    self.entries.append(entry)
                    i = entry.end_line
                    continue
            i += 1

    def _parse_entry(self, start_idx: int, layer: int) -> Optional[Entry]:
        title_line = self.lines[start_idx]
        # Extract ID from <!-- exp-id:xxx -->
        id_match = re.search(r'<!--\s*exp-id:([\w\-]+)\s*-->', title_line)
        entry_id = id_match.group(1) if id_match else None

        # Extract title text
        title_clean = re.sub(r'<!--.*?-->', '', title_line).strip()
        title_clean = re.sub(r'^###\s+', '', title_clean)
        title_clean = re.sub(r'^L\d+\.\d+\s+', '', title_clean)

        end_idx = start_idx + 1
        while end_idx < len(self.lines):
            nxt = self.lines[end_idx]
            if nxt.startswith('### ') or nxt.startswith('## '):
                break
            end_idx += 1

        if entry_id is None:
            # Auto-assign ID based on layer max sequence
            seq = self._next_seq(layer)
            entry_id = f'l{layer}-{seq:03d}'
            # Inject ID into title line
            new_title = title_line.rstrip() + f' <!-- exp-id:{entry_id} -->'
            self.lines[start_idx] = new_title
            title_line = new_title

        body = self.lines[start_idx + 1:end_idx]
        return Entry(
            id=entry_id,
            layer=layer,
            title=title_clean,
            raw_title_line=title_line,
            body_lines=body,
            start_line=start_idx,
            end_line=end_idx,
        )

    def _next_seq(self, layer: int) -> int:
        seqs = []
        for e in self.entries:
            if e.layer == layer:
                m = re.match(rf'^l{layer}-(\d+)$', e.id)
                if m:
                    seqs.append(int(m.group(1)))
        return max(seqs, default=0) + 1

    def _find_entry(self, entry_id: str) -> Optional[Entry]:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def _ensure_backup_dir(self):
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _backup(self, reason: str):
        self._ensure_backup_dir()
        ts = time.strftime('%Y%m%d_%H%M%S')
        bak = self.backup_dir / f'kernel-opt-{self.category}_{ts}.md'
        shutil.copy2(self.file_path, bak)
        self._log_history({'action': 'backup', 'reason': reason, 'file': str(bak), 'time': ts})
        return bak

    def _log_history(self, record: dict):
        self._ensure_backup_dir()
        with open(self.history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def get_last_backup(self) -> Optional[Path]:
        if not self.backup_dir.exists():
            return None
        files = sorted(
            self.backup_dir.glob(f'kernel-opt-{self.category}_*.md'),
            key=lambda p: p.stat().st_mtime
        )
        return files[-1] if files else None

    def _save(self):
        self.file_path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8')
        self._load()  # re-parse

    # ------------------ Read operations ------------------

    def list_categories(self) -> List[Tuple[str, int, str]]:
        """Return [(category, entry_count, last_update)]"""
        results = []
        for f in sorted(self.memory_dir.glob('kernel-opt-*.md')):
            cat = f.stem.replace('kernel-opt-', '')
            store = ExperienceStore(cat)
            count = len(store.entries)
            mtime = time.strftime('%Y-%m-%d', time.localtime(f.stat().st_mtime))
            results.append((cat, count, mtime))
        return results

    def show(self, layer: Optional[int] = None, entry_id: Optional[str] = None) -> str:
        if not self.exists():
            return f'[ERROR] 经验文件不存在: {self.file_path}'

        if entry_id:
            e = self._find_entry(entry_id)
            if not e:
                return f'[ERROR] 未找到 ID={entry_id}，可用条目: {", ".join(e.id for e in self.entries)}'
            lines = [f'ID: {e.id} | Layer {e.layer} | {e.title}', '-' * 40]
            lines.extend(e.body_lines)
            return '\n'.join(lines)

        if layer:
            filtered = [e for e in self.entries if e.layer == layer]
            if not filtered:
                return f'[INFO] Layer {layer} 下暂无条目'
            lines = [f'Layer {layer} 条目列表 ({len(filtered)} 条):', '']
            for e in filtered:
                preview = ' '.join(e.body_lines)[:60].replace('\n', ' ')
                lines.append(f'  {e.id}: {e.title}')
                lines.append(f'       {preview}...')
            return '\n'.join(lines)

        # Show full file summary
        lines = [f'# {self.category.capitalize()} 算子经验摘要', '']
        for lnum in (1, 2, 3):
            filtered = [e for e in self.entries if e.layer == lnum]
            lines.append(f'Layer {lnum}: {len(filtered)} 条')
            for e in filtered:
                lines.append(f'  {e.id}: {e.title}')
            lines.append('')
        return '\n'.join(lines)

    # ------------------ Write operations ------------------

    def add(self, layer: int, title: str, content: Optional[str] = None,
            why: Optional[str] = None, apply: Optional[str] = None) -> str:
        if not self.exists():
            return f'[ERROR] 经验文件不存在，请先运行: python3 utils/exp-init.py {self.category}'

        seq = self._next_seq(layer)
        entry_id = f'l{layer}-{seq:03d}'

        # Build body
        body_lines = []
        if content:
            body_lines.append(f'- **必须** {content}' if layer == 1 else f'- {content}')
        if why:
            body_lines.append(f'- **Why:** {why}')
        if apply:
            body_lines.append(f'- **How to apply:** {apply}')
        if not body_lines:
            body_lines.append('- 待补充')

        # Build title line with L numbering
        existing = [e for e in self.entries if e.layer == layer]
        subnum = len(existing) + 1
        title_line = f'### L{layer}.{subnum} {title} <!-- exp-id:{entry_id} -->'

        # Find insert position: end of target layer
        insert_pos = len(self.lines)
        in_target_layer = False
        for i, line in enumerate(self.lines):
            m = re.match(r'^## Layer\s+(\d+)\s*:', line)
            if m:
                if in_target_layer and int(m.group(1)) != layer:
                    insert_pos = i
                    break
                if int(m.group(1)) == layer:
                    in_target_layer = True
            if in_target_layer and line.startswith('### '):
                insert_pos = i + 1
                # skip to end of this entry
                while insert_pos < len(self.lines) and not self.lines[insert_pos].startswith('### ') and not self.lines[insert_pos].startswith('## '):
                    insert_pos += 1

        self._backup(f'add {entry_id}')
        new_block = [title_line] + body_lines
        self.lines = self.lines[:insert_pos] + new_block + self.lines[insert_pos:]
        self._save()
        self._log_history({'action': 'add', 'id': entry_id, 'layer': layer, 'title': title})
        return f'[ADD] {entry_id}: {title} (Layer {layer})'

    def update(self, entry_id: str, title: Optional[str] = None,
               content: Optional[str] = None, why: Optional[str] = None,
               apply: Optional[str] = None, dry_run: bool = False) -> Tuple[str, str, str]:
        """Returns (diff_str, before_str, after_str). If dry_run=True, does not write."""
        e = self._find_entry(entry_id)
        if not e:
            available = ', '.join(e.id for e in self.entries)
            raise ValueError(f'未找到 ID={entry_id}，可用条目: {available}')

        before_lines = [e.raw_title_line] + e.body_lines
        after_lines = before_lines.copy()

        if title:
            new_title = re.sub(r'^###\s+', '', after_lines[0])
            new_title = re.sub(r'<!--.*?-->', '', new_title).strip()
            # preserve L numbering
            prefix = re.match(r'^(L\d+\.\d+\s+)', new_title)
            prefix_str = prefix.group(1) if prefix else ''
            after_lines[0] = f'### {prefix_str}{title} <!-- exp-id:{e.id} -->'

        # Rebuild body if any content fields provided
        if content or why or apply:
            new_body = []
            if content:
                new_body.append(f'- **必须** {content}' if e.layer == 1 else f'- {content}')
            if why:
                new_body.append(f'- **Why:** {why}')
            if apply:
                new_body.append(f'- **How to apply:** {apply}')
            after_lines = [after_lines[0]] + new_body

        diff = '\n'.join(difflib.unified_diff(
            before_lines, after_lines,
            fromfile=f'{entry_id} (before)', tofile=f'{entry_id} (after)',
            lineterm=''
        ))

        if not dry_run:
            self._backup(f'update {entry_id}')
            self.lines = (
                self.lines[:e.start_line] +
                after_lines +
                self.lines[e.end_line:]
            )
            self._save()
            self._log_history({'action': 'update', 'id': entry_id})
        return diff, '\n'.join(before_lines), '\n'.join(after_lines)

    def remove(self, entry_id: str, dry_run: bool = False) -> Tuple[str, str]:
        """Returns (removed_title, removed_body). If dry_run=True, does not write."""
        e = self._find_entry(entry_id)
        if not e:
            available = ', '.join(e.id for e in self.entries)
            raise ValueError(f'未找到 ID={entry_id}，可用条目: {available}')

        removed = '\n'.join(self.lines[e.start_line:e.end_line])
        if not dry_run:
            self._backup(f'remove {entry_id}')
            self.lines = self.lines[:e.start_line] + self.lines[e.end_line:]
            self._save()
            self._log_history({'action': 'remove', 'id': entry_id, 'removed': removed})
        return e.title, removed

    def undo(self) -> Optional[Path]:
        last = self.get_last_backup()
        if not last:
            return None
        shutil.copy2(last, self.file_path)
        self._load()
        self._log_history({'action': 'undo', 'restored_from': str(last)})
        return last

    def history(self) -> str:
        if not self.history_path.exists():
            return '[INFO] 暂无修改历史'
        lines = []
        with open(self.history_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line.strip())
                ts = rec.get('time', '?')
                act = rec.get('action', '?')
                detail = ''
                if act == 'backup':
                    detail = f"reason={rec.get('reason')}"
                elif act in ('add', 'update', 'remove'):
                    detail = f"id={rec.get('id')}"
                elif act == 'undo':
                    detail = f"from={rec.get('restored_from')}"
                lines.append(f'  [{ts}] {act} {detail}')
        return '\n'.join(lines)


# ------------------ CLI ------------------

def confirm(prompt: str, diff: str = '') -> bool:
    if diff:
        print('\n--- 变更 diff ---')
        print(diff)
        print('--- end diff ---\n')
    resp = input(f'{prompt} (y/n/dry-run): ').strip().lower()
    if resp == 'y':
        return True
    if resp == 'dry-run':
        print('[DRY-RUN] 未执行写入')
        return False
    print('[CANCEL] 操作已取消')
    return False


def main():
    parser = argparse.ArgumentParser(description='历史经验增删改查工具')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # list
    sub.add_parser('list', help='列出所有经验类别')

    # show
    p_show = sub.add_parser('show', help='查看经验内容')
    p_show.add_argument('category', help='算子类别')
    p_show.add_argument('--layer', type=int, choices=(1, 2, 3), help='仅查看指定 Layer')
    p_show.add_argument('--id', help='查看指定条目')

    # add
    p_add = sub.add_parser('add', help='新增经验条目')
    p_add.add_argument('category', help='算子类别')
    p_add.add_argument('--layer', type=int, required=True, choices=(1, 2, 3))
    p_add.add_argument('--title', required=True)
    p_add.add_argument('--content')
    p_add.add_argument('--why')
    p_add.add_argument('--apply')

    # update
    p_up = sub.add_parser('update', help='修改经验条目（高危）')
    p_up.add_argument('category', help='算子类别')
    p_up.add_argument('--id', required=True, help='条目 ID，如 l1-002')
    p_up.add_argument('--title')
    p_up.add_argument('--content')
    p_up.add_argument('--why')
    p_up.add_argument('--apply')
    p_up.add_argument('--yes', action='store_true', help='跳过交互确认（Agent 模式）')

    # remove
    p_rm = sub.add_parser('remove', help='删除经验条目（高危）')
    p_rm.add_argument('category', help='算子类别')
    p_rm.add_argument('--id', required=True, help='条目 ID')
    p_rm.add_argument('--yes', action='store_true', help='跳过交互确认（Agent 模式）')

    # undo
    p_undo = sub.add_parser('undo', help='回退上一次写操作')
    p_undo.add_argument('category', help='算子类别')

    # history
    p_hist = sub.add_parser('history', help='查看修改历史')
    p_hist.add_argument('category', help='算子类别')

    args = parser.parse_args()

    if args.cmd == 'list':
        store = ExperienceStore('dummy')
        results = store.list_categories()
        print(f'{"类别":<12} {"条目数":>6} {"最近更新":>12}')
        print('-' * 32)
        for cat, cnt, mtime in results:
            print(f'{cat:<12} {cnt:>6} {mtime:>12}')
        return

    store = ExperienceStore(args.category)

    if args.cmd == 'show':
        print(store.show(layer=args.layer, entry_id=args.id))
        return

    if args.cmd == 'add':
        result = store.add(
            layer=args.layer, title=args.title,
            content=args.content, why=args.why, apply=args.apply
        )
        print(result)
        return

    if args.cmd == 'update':
        try:
            diff, before, after = store.update(
                args.id, title=args.title,
                content=args.content, why=args.why, apply=args.apply,
                dry_run=True
            )
        except ValueError as e:
            print(f'[ERROR] {e}', file=sys.stderr)
            sys.exit(1)
        if not args.yes and not confirm(f'确认更新 {args.id}?', diff=diff):
            sys.exit(0)
        # Execute for real
        store.update(
            args.id, title=args.title,
            content=args.content, why=args.why, apply=args.apply,
            dry_run=False
        )
        print(f'[UPDATE] {args.id} 已更新')
        print(f'[BACKUP] 原文件已备份')
        return

    if args.cmd == 'remove':
        try:
            title, removed = store.remove(args.id, dry_run=True)
        except ValueError as e:
            print(f'[ERROR] {e}', file=sys.stderr)
            sys.exit(1)
        preview = removed[:200].replace('\n', ' ')
        if not args.yes and not confirm(
            f'确认删除 {args.id} ({title})?\n预览: {preview}...'
        ):
            sys.exit(0)
        store.remove(args.id, dry_run=False)
        print(f'[REMOVE] {args.id} ({title}) 已删除')
        print(f'[BACKUP] 内容已备份')
        return

    if args.cmd == 'undo':
        bak = store.undo()
        if bak:
            print(f'[UNDO] 已恢复至备份: {bak.name}')
        else:
            print('[WARN] 未找到可回退的备份')
        return

    if args.cmd == 'history':
        print(store.history())
        return


if __name__ == '__main__':
    main()
