#!/usr/bin/env python3
"""
新算子类别经验文件初始化工具：基于模板创建 kernel-opt-{category}.md 并更新 MEMORY.md 索引。

用法:
    python3 utils/init_experience.py <category> [--op-name <op_name>]

示例:
    python3 utils/init_experience.py pad --op-name 15_Pad
"""

import argparse
import re
import sys
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def render_template(category: str, op_name: str) -> str:
    """渲染 kernel-opt 模板。"""
    category_cap = category.capitalize()
    return f"""---
name: kernel-opt-{category}
description: {category_cap} 算子的 Triton Ascend 四层隔离优化经验
metadata:
  type: reference
---

# {category_cap} 算子优化经验

**算子类别**: `待补充`
**典型特征**: 待补充
**性能基准**: 待补充

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 约束名称
- **必须** / **禁止** 做某事
- **Why:** 原因说明
- **How to apply:** 适用场景

> 提示：每次算子探索后，将本次验证过的硬性约束追加到本层。

---

## Layer 2: 算法骨架（Agent 可参考架构）

### L2.1 Host 侧分支决策树（伪代码）

```
待补充
```

### L2.2 多核并行骨架模式

**模式 A - 按元素分配**:
```
elements_per_core = cdiv(total_elements, num_cores)
core_start = pid * elements_per_core
...
```

**模式 B - 按行分配**:
```
rows_per_core = cdiv(total_rows, num_cores)
row_start = pid * rows_per_core
...
```

---

## Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

### L3.1 技巧名称
```python
# 代码片段
```

**可替代方向**: 说明其他可能的实现方式。

---

## Layer 4: 完整归档（Agent 默认不读取，仅人工复盘）

> ⚠️ **Agent 注意**：以下仅为历史实现的路径记录。你**禁止**直接复制其代码结构、变量命名或 kernel 组织方式。

### 历史实现归档

| 版本 | 代码 | 报告 | 摘要 | 性能 | 特点 |
|------|------|------|------|------|------|
| 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

### 完整归档路径（Layer 4）
```
/home/zmm/OpAgent-Pad/.claude/memory/archive/{category}/
```

### 原始工作目录
```
待补充
```

### 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 待补充 | 待补充 | 待补充 |

**关键结论**：待补充

---

## 常见陷阱与避免方法

### 陷阱 1: 名称
- **问题**: 描述
- **解决**: 方案
"""


def update_memory_md(memory_path: Path, category: str) -> None:
    """在 MEMORY.md 中添加该算子类别的经验文件索引。"""
    category_cap = category.capitalize()
    exp_line = f'- [{category_cap} 算子优化经验](kernel-opt-{category}.md) — 待补充'

    if not memory_path.exists():
        memory_path.write_text('# Memory Index\n\n', encoding='utf-8')

    content = memory_path.read_text(encoding='utf-8')
    lines = content.splitlines()

    # 查找是否已有该类别
    for line in lines:
        if f'kernel-opt-{category}.md' in line:
            return  # 已存在

    # 在 "算子优化经验" 节追加
    section_idx = None
    insert_pos = len(lines)
    for i, line in enumerate(lines):
        if '算子优化经验' in line or '经验' in line:
            section_idx = i
    if section_idx is not None:
        for j in range(section_idx + 1, len(lines)):
            if lines[j].strip().startswith('- ') and 'kernel-opt-' in lines[j]:
                insert_pos = j + 1
            elif lines[j].strip().startswith('## ') and j > section_idx:
                insert_pos = j
                break
        lines.insert(insert_pos, exp_line)
    else:
        lines.append('')
        lines.append('## 算子优化经验')
        lines.append(exp_line)

    memory_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='初始化新算子类别经验文件')
    parser.add_argument('category', help='算子类别名（小写，如 pad, softmax）')
    parser.add_argument('--op-name', default=None, help='原始算子名称（如 15_Pad）')
    args = parser.parse_args()

    category = args.category.lower()
    op_name = args.op_name or category.capitalize()
    root = get_project_root()

    exp_path = root / '.claude' / 'memory' / f'kernel-opt-{category}.md'
    memory_path = root / '.claude' / 'memory' / 'MEMORY.md'

    if exp_path.exists():
        print(f'[SKIP] 经验文件已存在: {exp_path}')
        print('       如需更新，请直接编辑该文件。')
        sys.exit(0)

    exp_path.write_text(render_template(category, op_name), encoding='utf-8')
    print(f'[CREATE] {exp_path}')

    update_memory_md(memory_path, category)
    print(f'[UPDATE] {memory_path}')

    print(f'\n[SUCCESS] {category} 类别经验文件已初始化。')
    print('          下一步：将本次探索提炼的 Layer 1-3 内容填入该文件。')


if __name__ == '__main__':
    main()
