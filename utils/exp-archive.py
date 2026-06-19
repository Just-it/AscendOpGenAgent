#!/usr/bin/env python3
"""
经验归档工具：将已探索成功的算子工作目录归档到 memory/archive/，并更新 MEMORY.md 索引。

用法:
    python3 utils/exp-archive.py <work_dir> [--category <category>] [--force]

示例:
    python3 utils/exp-archive.py ./triton_ascend_output/op_1_15_Pad_20260522_0509_7731
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """以脚本所在目录的父目录作为项目根目录。"""
    return Path(__file__).resolve().parent.parent


def parse_op_name_from_dir(work_dir: Path) -> str:
    """从工作目录名解析算子名称，如 op_1_15_Pad_20260522_0509_7731 -> 15_Pad。"""
    name = work_dir.name
    m = re.match(r'op_\d+_(.+?)_\d{8}_\d{4}_\d+', name)
    if m:
        return m.group(1)
    # fallback: 查找目录下 *_generated.py
    for f in work_dir.glob('*_generated.py'):
        return f.stem.replace('_generated', '')
    return name


def parse_date_from_dir(work_dir: Path) -> str:
    """从工作目录名提取日期，如 op_1_15_Pad_20260522_0509_7731 -> 20260522。"""
    name = work_dir.name
    m = re.search(r'(\d{8})_\d{4}_\d+', name)
    if m:
        return m.group(1)
    return datetime.now().strftime('%Y%m%d')


def infer_category(op_name: str) -> str:
    """从算子名称推断类别，如 15_Pad -> pad，0_Softmax -> softmax。"""
    category = re.sub(r'^\d+_', '', op_name).lower()
    return category


def get_next_version(archive_dir: Path, category: str) -> int:
    """扫描 archive 目录下已有版本，返回下一个版本号。"""
    max_ver = 0
    if not archive_dir.exists():
        return 1
    pattern = re.compile(re.escape(category) + r'_v(\d+)_(\d{8})\.py')
    for f in archive_dir.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                max_ver = max(max_ver, int(m.group(1)))
    return max_ver + 1


def find_best_version(archive_dir: Path, category: str) -> tuple:
    """
    扫描 archive 目录下已有版本，返回最优版本的 (speedup, item)。
    item: {'py': Path, 'report': Path, 'summary': Path, 'ver': int}
    无已有版本时返回 (0.0, None)。
    """
    if not archive_dir.exists():
        return 0.0, None

    best_speedup = 0.0
    best_item = None
    pattern = re.compile(re.escape(category) + r'_v(\d+)_(\d{8})\.py$')

    for f in archive_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if not m:
            continue
        ver = int(m.group(1))
        summary_file = f.with_suffix('').with_name(f.stem + '_summary.json')
        if not summary_file.exists():
            continue
        try:
            with open(summary_file, 'r', encoding='utf-8') as sf:
                data = json.load(sf)
            speedup = data.get('perf_data', {}).get('speedup_vs_torch', 0.0)
            if isinstance(speedup, (int, float)) and speedup > best_speedup:
                best_speedup = speedup
                report_file = f.with_suffix('').with_name(f.stem + '_report.md')
                best_item = {
                    'py': f,
                    'report': report_file,
                    'summary': summary_file,
                    'ver': ver,
                }
        except Exception:
            continue

    return best_speedup, best_item


def remove_all_versions(archive_dir: Path, category: str) -> None:
    """删除该类别下所有历史版本文件（.py + _report.md + _summary.json）。"""
    if not archive_dir.exists():
        return
    py_pattern = re.compile(re.escape(category) + r'_v\d+_\d{8}\.py$')
    for f in list(archive_dir.iterdir()):
        if f.is_file() and py_pattern.match(f.name):
            stem = f.stem
            for suffix in ('.py', '_report.md', '_summary.json'):
                to_remove = f.parent / (stem + suffix)
                if to_remove.exists():
                    to_remove.unlink()
                    print(f'[REMOVE] {to_remove}')


def validate_work_dir(work_dir: Path) -> dict:
    """验证工作目录是否符合归档条件，返回 summary dict。"""
    summary_path = work_dir / 'summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'工作目录缺少 summary.json: {work_dir}')

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    if not summary.get('success'):
        raise ValueError('summary.json 中 success 为 false，不可归档')

    perf = summary.get('perf_data', {})
    total = perf.get('total_cases', 0)
    passed = perf.get('passed_cases', 0)
    if passed != total or total == 0:
        raise ValueError(f'精度未全通过: {passed}/{total}')

    speedup = perf.get('speedup_vs_torch')
    if speedup is None or (isinstance(speedup, (int, float)) and speedup <= 0.8):
        raise ValueError(f'加速比不满足归档条件: {speedup} (需 > 0.8x)')

    # 检查必要文件是否存在
    op_name = parse_op_name_from_dir(work_dir)
    required = [
        work_dir / f'{op_name}_generated.py',
        work_dir / 'report.md',
        work_dir / 'summary.json',
    ]
    for rp in required:
        if not rp.exists():
            raise FileNotFoundError(f'工作目录缺少必要文件: {rp.name}')

    return summary


def update_memory_md(memory_path: Path, category: str, ver: int, date: str,
                     speedup: float, passed: int, total: int) -> None:
    """更新 MEMORY.md 索引，如果不存在该类别的条目则追加。"""
    category_cap = category.capitalize()
    py_file = f'archive/{category}/{category}_v{ver}_{date}.py'
    report_file = f'archive/{category}/{category}_v{ver}_{date}_report.md'
    summary_file = f'archive/{category}/{category}_v{ver}_{date}_summary.json'

    new_line = (
        f'- [{category_cap} 算子最佳实现]({py_file}) — '
        f'几何平均加速比 {speedup:.2f}x，{passed}/{total} cases 通过；'
        f'配套 [report]({report_file}) / [summary]({summary_file})'
    )

    if not memory_path.exists():
        memory_path.write_text('# Memory Index\n\n', encoding='utf-8')

    content = memory_path.read_text(encoding='utf-8')
    lines = content.splitlines()

    # 查找是否已有该类别的归档条目
    section_idx = None
    existing_idx = None
    for i, line in enumerate(lines):
        if '完整代码归档' in line or 'Layer 4' in line:
            section_idx = i
        if f'/{category}/' in line and f'{category}_v' in line:
            existing_idx = i

    if existing_idx is not None:
        # 替换旧条目
        lines[existing_idx] = new_line
    else:
        # 在 Layer 4 节末尾追加，或在文件末尾追加
        if section_idx is not None:
            # 找到该节最后一个非空行，在其后插入
            insert_pos = section_idx + 1
            for j in range(section_idx + 1, len(lines)):
                if lines[j].strip().startswith('- '):
                    insert_pos = j + 1
                elif lines[j].strip().startswith('## ') and j > section_idx:
                    break
            lines.insert(insert_pos, new_line)
        else:
            lines.append('')
            lines.append('## 完整代码归档（Layer 4，Agent 默认不可读）')
            lines.append(new_line)

    memory_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='归档已探索成功的算子经验')
    parser.add_argument('work_dir', help='算子工作目录绝对或相对路径')
    parser.add_argument('--category', default=None,
                        help='手动指定算子类别（默认从目录名自动推断）')
    parser.add_argument('--force', action='store_true',
                        help='跳过归档条件校验（仅复制文件）')
    parser.add_argument('--create-experience', action='store_true',
                        help='若该算子类别尚无经验文件，自动基于模板创建')
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    if not work_dir.is_dir():
        print(f'[ERROR] 工作目录不存在: {work_dir}', file=sys.stderr)
        sys.exit(1)

    root = get_project_root()
    archive_root = root / '.claude' / 'memory' / 'archive'
    memory_path = root / '.claude' / 'memory' / 'MEMORY.md'

    # 1. 验证
    try:
        summary = validate_work_dir(work_dir)
    except Exception as e:
        if args.force:
            summary_path = work_dir / 'summary.json'
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            print(f'[WARN] 强制跳过校验: {e}')
        else:
            print(f'[ERROR] 归档校验失败: {e}', file=sys.stderr)
            sys.exit(1)

    op_name = parse_op_name_from_dir(work_dir)
    category = args.category or infer_category(op_name)
    date_str = parse_date_from_dir(work_dir)
    version = get_next_version(archive_root / category, category)

    perf = summary.get('perf_data', {})
    passed_cases = perf.get('passed_cases', 0)
    total_cases = perf.get('total_cases', 0)
    speedup = perf.get('speedup_vs_torch', 0.0)

    # 2. 版本比较与旧版本清理（仅存最优方案）
    dest_dir = archive_root / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    best_speedup, best_item = find_best_version(dest_dir, category)
    if best_item:
        if speedup <= best_speedup and not args.force:
            print(f'[ERROR] 当前加速比 {speedup:.4f}x 不优于已有最优版本 '
                  f'v{best_item["ver"]} ({best_speedup:.4f}x)。'
                  f'如需强制覆盖，请加 --force', file=sys.stderr)
            sys.exit(1)
        # 新版本更优，清理所有旧版本物理文件
        remove_all_versions(dest_dir, category)

    # 3. 创建归档并复制文件
    base_name = f'{category}_v{version}_{date_str}'
    src_files = {
        work_dir / f'{op_name}_generated.py': dest_dir / f'{base_name}.py',
        work_dir / 'report.md': dest_dir / f'{base_name}_report.md',
        work_dir / 'summary.json': dest_dir / f'{base_name}_summary.json',
    }

    for src, dst in src_files.items():
        shutil.copy2(src, dst)
        print(f'[COPY] {src} -> {dst}')

    # 3. 更新 MEMORY.md
    update_memory_md(memory_path, category, version, date_str,
                     speedup, passed_cases, total_cases)
    print(f'[UPDATE] {memory_path}')

    print(f'\n[SUCCESS] {op_name} 已归档为 {base_name}.* (speedup={speedup:.4f}x)')

    # 4. 可选：自动创建经验文件模板
    if args.create_experience:
        exp_path = root / '.claude' / 'memory' / f'kernel-opt-{category}.md'
        if not exp_path.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(root / 'utils' / 'exp-init.py'),
                 category, '--op-name', op_name],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr, file=sys.stderr)
        else:
            print(f'[INFO] 经验文件已存在: {exp_path}，请手动更新 Layer 1-3 内容。')


if __name__ == '__main__':
    main()
