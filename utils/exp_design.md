# 历史经验管理方案设计（增删改查）

## 1. 设计目标

让 Agent 和人类用户都能对 `.claude/memory/kernel-opt-{category}.md` 中的历史经验进行增删改查，同时保证：
- **高危操作可回退**：更新/删除前必须确认，并保留备份
- **多次操作一致性**：所有写操作后自动校验四层模型格式
- **界面简洁**：命令语义清晰，输出聚焦核心信息

## 2. 命令接口设计

统一入口：`utils/exp-edit.py`

```bash
# 查
python3 utils/exp-edit.py list                          # 列出所有经验类别及条目数
python3 utils/exp-edit.py show <category>               # 查看完整经验文件
python3 utils/exp-edit.py show <category> --layer 1     # 仅查看 Layer 1
python3 utils/exp-edit.py show <category> --id l1-002   # 查看指定条目

# 增
python3 utils/exp-edit.py add <category> --layer 1 \
  --title "坐标比较必须用 float32" \
  --content "禁止直接对整数坐标使用 tl.where(coord < 0, ...)" \
  --why "Ascend 后端整数比较可能行为异常" \
  --apply "所有边界映射 kernel"

# 改（高危）
python3 utils/exp-edit.py update <category> --id l1-002 \
  --content "必须先 .to(tl.float32) 再比较"

# 删（高危）
python3 utils/exp-edit.py remove <category> --id l1-002

# 工具
python3 utils/exp-edit.py undo <category>               # 回退上一次写操作
python3 utils/exp-edit.py history <category>            # 查看该类别的修改历史
```

## 3. 数据模型与 ID 方案

### 3.1 Markdown 内嵌 ID

在现有 `kernel-opt-*.md` 的 Markdown 结构中，为每个可独立操作的条目（L1.x、L3.x 等）末尾注入不可见的 HTML 注释 ID：

```markdown
### L1.2 坐标比较必须用 float32 <!-- exp-id:l1-002 -->
- **必须**先 `.to(tl.float32)` 再比较
- **Why:** Ascend 后端整数比较可能行为异常
- **How to apply:** 所有边界映射 kernel
```

- `l1-002`：`l` + 层级 + `-` + 该层内序号
- 序号按该层现有最大序号 + 1 自动生成
- 注释对人类阅读完全透明，但对程序提供精确定位锚点

### 3.2 条目定位规则

| 场景 | 定位方式 |
|------|---------|
| 有 ID | 精确匹配 `<!-- exp-id:{id} -->` |
| 无 ID（历史遗留）| 按子标题 `### L{N}.{M} ...` 定位，操作后自动补 ID |
| 新增 | 在目标 Layer 最后一个条目后追加，自动生成 ID |

## 4. 高危操作确认机制

### 4.1 交互确认（人类用户）

`update` / `remove` 默认进入交互确认：

```
[UPDATE] pad / l1-002
--- 当前内容 ---
- **必须**先 `.to(tl.float32)` 再比较
--- 变更后内容 ---
- **必须**先 `.to(tl.float32)` 再比较，禁止直接对整数坐标使用 tl.where

确认执行？ (y/n/dry-run): 
```

- `y`：执行
- `n`：取消
- `dry-run`：仅打印 diff，不写入

### 4.2 Agent 非交互模式

Agent 调用时附加 `--yes` 参数跳过确认：

```bash
python3 utils/exp-edit.py update pad --id l1-002 --content "..." --yes
```

**约束**：Agent 必须在调用前向用户展示变更摘要（diff），获得用户文字确认后方可加 `--yes` 执行。

### 4.3 删除特别保护

`remove` 操作无论是否 `--yes`，都会：
1. 将被删条目完整内容写入 `.claude/memory/.backup/{category}_removed_{timestamp}.md`
2. 在 `history` 中记录删除事件（含被删内容摘要）
3. 输出 `[REMOVED] ID={id} 内容已备份至 ...`

## 5. 一致性保障机制

### 5.1 自动备份

每次写操作（add/update/remove）前：
- 自动将原文件复制到 `.claude/memory/.backup/kernel-opt-{category}_{timestamp}.md`
- 保留最近 20 个备份，超出的自动清理

### 5.2 自动校验

每次写操作后：
- 校验 Markdown 结构完整性（frontmatter、Layer 1-4 标题是否齐全）
- 校验所有条目是否都有 `exp-id`
- 若有缺失，自动补 ID 并告警
- 调用 `exp-check.py` 做全局一致性检查

### 5.3 回退机制

```bash
python3 utils/exp-edit.py undo <category>
```

- 将文件恢复为最近一次备份状态
- 仅能回退一次（最近一次写操作）
- 回退后删除对应备份，防止连环 undo

## 6. 简洁性设计

### 6.1 输出格式

- **查询**：仅显示条目标题 + ID + 前 80 字符摘要，不显示完整 Markdown
- **变更**：统一用 unified diff 格式展示，3 行上下文
- **列表**：一行为一个类别，显示"类别名 / 总条目数 / 最近更新日期"

### 6.2 错误提示

- 命令参数错误：直接输出用法示例（`Usage: ...`）
- ID 不存在：`[ERROR] pad 中不存在 ID=l1-099，可用条目: l1-001, l1-002, l3-001`
- 文件不存在：`[ERROR] kernel-opt-foo.md 不存在。如需创建，请先运行: python3 utils/exp-init.py foo`

## 7. 实施计划

1. **Phase A**：实现 `show` / `list`（只读，无风险）
2. **Phase B**：实现 `add`（写操作，有备份和校验）
3. **Phase C**：实现 `update` / `remove` + 交互确认（高危操作）
4. **Phase D**：实现 `undo` / `history`（回退与审计）
5. **Phase E**：对现有 `kernel-opt-pad.md` 自动补全 `exp-id`

## 8. 与现有工具的协作关系

```
exp-init.py  ---->  首次创建 kernel-opt-{category}.md（含模板结构）
       |
       v
exp-archive.py  ->  归档代码到 Layer 4（更新 Layer 4 表格）
       |
       v
   exp-edit.py  ------------>  日常增删改查 Layer 1-3 经验条目
       |
       v
exp-check.py -> 全局一致性最终校验
```

---

**请确认本方案后，我按 Phase A→E 逐步实现。**
