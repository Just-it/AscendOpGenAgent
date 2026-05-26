#!/usr/bin/env python3
"""
归档与记忆规范检查工具：检查 archive 目录结构、MEMORY.md 索引、
kernel-opt 经验文件是否符合四层隔离模型的规范要求。

用法:
    python3 utils/exp-check.py [--root <project_root>]

退出码:
    0  全部通过
    1  存在失败项
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


class CheckReporter:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    def ok(self, msg: str):
        self.passed += 1
        self.details.append(f'  [PASS] {msg}')

    def fail(self, msg: str):
        self.failed += 1
        self.details.append(f'  [FAIL] {msg}')

    def warn(self, msg: str):
        self.warnings += 1
        self.details.append(f'  [WARN] {msg}')

    def section(self, title: str):
        self.details.append(f'\n{title}')

    def print_report(self):
        for line in self.details:
            print(line)
        print('\n' + '='*60)
        print(f'总计: {self.passed} 通过, {self.failed} 失败, {self.warnings} 警告')
        if self.failed > 0:
            print('结论: 不符合规范，请修复失败项')
        elif self.warnings > 0:
            print('结论: 基本符合规范，但存在警告')
        else:
            print('结论: 完全符合规范')


def check_archive_structure(root: Path, reporter: CheckReporter) -> dict:
    """
    检查 archive/ 目录结构，返回按类别组织的归档文件映射。
    返回: {category: [{'py': Path, 'report': Path, 'summary': Path, 'ver': int, 'date': str}]}
    """
    archive_root = root / '.claude' / 'memory' / 'archive'
    reporter.section('## 检查 archive/ 目录结构')

    if not archive_root.exists():
        reporter.fail(f'archive 根目录不存在: {archive_root}')
        return {}

    readme = archive_root / 'README.md'
    if readme.exists():
        reporter.ok('archive/README.md 存在')
    else:
        reporter.fail('archive/README.md 缺失')

    # 遍历子目录
    categories = {}
    for entry in sorted(archive_root.iterdir()):
        if not entry.is_dir():
            continue
        cat = entry.name
        if cat.startswith('.') or cat == 'archive':
            continue

        reporter.section(f'### 类别: {cat}')
        if not re.match(r'^[a-z][a-z0-9_]*$', cat):
            reporter.warn(f'目录名 "{cat}" 建议全小写且无特殊字符')

        py_files = list(entry.glob('*_v*_*.py'))
        if not py_files:
            reporter.fail(f'{cat}/ 下未找到符合命名规范的 .py 归档文件')
            continue

        categories[cat] = []
        for py_file in sorted(py_files):
            m = re.match(re.escape(cat) + r'_v(\d+)_(\d{8})\.py$', py_file.name)
            if not m:
                reporter.warn(f'文件名不符合规范: {py_file.name}')
                continue
            ver = int(m.group(1))
            date = m.group(2)

            report_file = py_file.with_suffix('').with_name(py_file.stem + '_report.md')
            summary_file = py_file.with_suffix('').with_name(py_file.stem + '_summary.json')

            has_report = report_file.exists()
            has_summary = summary_file.exists()

            if has_report and has_summary:
                reporter.ok(f'{py_file.name} 配套文件完整 (v{ver}, {date})')
            else:
                if not has_report:
                    reporter.fail(f'{py_file.name} 缺少配套报告: {report_file.name}')
                if not has_summary:
                    reporter.fail(f'{py_file.name} 缺少配套摘要: {summary_file.name}')

            categories[cat].append({
                'py': py_file,
                'report': report_file if has_report else None,
                'summary': summary_file if has_summary else None,
                'ver': ver,
                'date': date,
            })

    return categories


def check_summary_json(categories: dict, reporter: CheckReporter):
    """检查每个 summary.json 的字段与数值规范。"""
    reporter.section('\n## 检查 summary.json 字段规范')
    for cat, items in categories.items():
        for item in items:
            sf = item['summary']
            if sf is None:
                continue
            try:
                with open(sf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                reporter.fail(f'{cat}/{sf.name} JSON 解析失败: {e}')
                continue

            # 关键字段
            for key in ('success', 'perf_data'):
                if key not in data:
                    reporter.fail(f'{cat}/{sf.name} 缺少顶层字段: {key}')
                    break
            else:
                reporter.ok(f'{cat}/{sf.name} 包含必要顶层字段')

            perf = data.get('perf_data', {})
            total = perf.get('total_cases', 0)
            passed = perf.get('passed_cases', 0)
            speedup = perf.get('speedup_vs_torch')

            if passed == total and total > 0:
                reporter.ok(f'{cat}/{sf.name} 精度全通过 ({passed}/{total})')
            else:
                reporter.fail(f'{cat}/{sf.name} 精度未全通过: {passed}/{total}')

            if speedup is not None and isinstance(speedup, (int, float)) and speedup > 0.8:
                reporter.ok(f'{cat}/{sf.name} 加速比达标: {speedup:.4f}x')
            else:
                reporter.fail(f'{cat}/{sf.name} 加速比不达标: {speedup} (需 > 0.8x)')


def check_memory_md(root: Path, categories: dict, reporter: CheckReporter):
    """检查 MEMORY.md 索引是否与 archive 目录一致。"""
    reporter.section('\n## 检查 MEMORY.md 索引一致性')
    memory_path = root / '.claude' / 'memory' / 'MEMORY.md'
    if not memory_path.exists():
        reporter.fail(f'MEMORY.md 不存在: {memory_path}')
        return

    content = memory_path.read_text(encoding='utf-8')
    lines = content.splitlines()

    # 收集 archive 下所有预期条目
    indexed_cats = set()
    for cat in categories:
        # 查找是否包含指向该 category 的链接
        found = False
        for line in lines:
            if f'/{cat}/' in line:
                found = True
                # 检查行长度
                if len(line) > 150:
                    reporter.warn(f'MEMORY.md 行超长 ({len(line)} > 150): {line[:80]}...')
                # 检查链接文件是否存在
                for link_match in re.finditer(r'\]\(([^)]+)\)', line):
                    link = link_match.group(1)
                    linked_path = (root / '.claude' / 'memory' / link).resolve()
                    if not linked_path.exists():
                        reporter.fail(f'MEMORY.md 链接指向的文件不存在: {link}')
                break
        if found:
            reporter.ok(f'MEMORY.md 包含 {cat} 的索引条目')
            indexed_cats.add(cat)
        else:
            reporter.fail(f'MEMORY.md 缺少 {cat} 的索引条目')

    # 反向检查：MEMORY.md 中是否有指向不存在的 archive 条目的链接
    for line in lines:
        if '/archive/' in line and line.strip().startswith('- '):
            m = re.search(r'archive/([a-z][a-z0-9_]*)', line)
            if m:
                cat = m.group(1)
                if cat not in categories:
                    reporter.warn(f'MEMORY.md 引用了 archive 中不存在的类别: {cat}')


def check_kernel_opt_refs(root: Path, categories: dict, reporter: CheckReporter):
    """检查 kernel-opt-*.md 中 Layer 4 引用是否有效。"""
    reporter.section('\n## 检查 kernel-opt-*.md 引用有效性')
    memory_dir = root / '.claude' / 'memory'
    for md_file in sorted(memory_dir.glob('kernel-opt-*.md')):
        # 从文件名推断类别，如 kernel-opt-pad.md -> pad
        cat = md_file.stem.replace('kernel-opt-', '')
        if cat not in categories:
            continue

        content = md_file.read_text(encoding='utf-8')
        # 查找 Layer 4 或 archive 路径引用
        refs = re.findall(r'archive/[a-z][a-z0-9_/\-\.]+', content)
        if not refs:
            reporter.warn(f'{md_file.name} 的 Layer 4 部分未引用任何 archive 路径')
            continue

        valid = True
        for ref in refs:
            full = memory_dir / ref
            if not full.exists():
                reporter.fail(f'{md_file.name} 引用的路径不存在: {ref}')
                valid = False
        if valid:
            reporter.ok(f'{md_file.name} Layer 4 引用全部有效 ({len(refs)} 处)')


def main():
    parser = argparse.ArgumentParser(description='检查归档与记忆规范')
    parser.add_argument('--root', default='.',
                        help='项目根目录（默认当前目录）')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    reporter = CheckReporter()

    categories = check_archive_structure(root, reporter)
    if categories:
        check_summary_json(categories, reporter)
        check_memory_md(root, categories, reporter)
        check_kernel_opt_refs(root, categories, reporter)
    else:
        reporter.fail('未检测到任何有效的归档类别，跳过后续检查')

    reporter.print_report()
    sys.exit(1 if reporter.failed > 0 else 0)


if __name__ == '__main__':
    main()
