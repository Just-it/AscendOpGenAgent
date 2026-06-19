---
name: kernel-opt-framework
description: Triton Ascend 算子历史探索经验积累方案框架，定义四层隔离分类体系、存储结构、复用机制和防依赖策略
metadata:
  type: reference
---

# Triton Ascend 算子历史探索经验积累方案

## 1. 核心设计原则

**经验 ≠ 模板**。历史代码的探索价值在于提取"设计决策"和"验证过的技巧"，而非提供可直接复制的答案。

**目标**：让 Agent 站在历史肩膀上，而非躺在历史温床上。

## 2. 算子分类体系

按计算特征分为 6 类，每类维护独立的经验文件：

| 类别 | 特征 | 典型算子 |
|------|------|---------|
| `element-wise` | 逐元素独立计算，无跨元素依赖 | add, mul, relu, gelu, sigmoid |
| `reduction` | 沿某维度聚合，输出维度降低 | sum, mean, max, softmax, layernorm |
| `transformation-memory` | 主要是数据重排/搬运，计算简单 | pad, permute, slice, repeat, tile |
| `transformation-compute` | 数据重排伴随计算 | conv, matmul, attention |
| `indexing-gather` | 按索引收集/散射 | embedding, index_select, scatter |
| `sort-topk` | 排序/选择类 | sort, topk, argsort |

## 3. 四层隔离存储模型（关键设计）

每个算子类别的经验按**四层隔离**存储，不同 Skill/阶段只能访问对应层级，防止直接复制代码。

### Layer 1: 设计约束层 (Constraints)
- **内容**：必须遵守的设计原则、禁止的反模式、已验证的无效方向
- **受众**：`kernel-designer`（Phase 2）
- **形式**：文字描述 + 伪代码，无具体实现
- **示例**："constant padding 禁止在 kernel 内逐元素判断边界，必须在 host 侧拆分为 fill + copy"

### Layer 2: 算法骨架层 (Skeleton)
- **内容**：核心并行策略的抽象描述、分块逻辑、grid 分配模式
- **受众**：`kernel-designer` + `kernel-generator`（Phase 2/3）
- **形式**：极简伪代码或文字流程，不含具体变量名和完整边界处理
- **示例**："1D 并行模板：elements_per_core = cdiv(total, num_cores)，每个 core 内部 for-block 循环"

### Layer 3: 关键技巧层 (Snippets)
- **内容**：5-15 行最具技巧性的代码片段，已验证有效
- **受众**：`kernel-generator` + `latency-optimizer`（Phase 3/4）
- **形式**：带注释的代码片段，明确标注"此为已知有效技巧，实现方式可不同"
- **示例**：坐标压缩公式、边界映射模式、block size 选择逻辑

### Layer 4: 完整归档层 (Archive)
- **内容**：完整的历史实现代码、任务工作目录路径、性能数据
- **受众**：**默认对 Agent 不可见**，仅在明确需要对比时才提供路径
- **形式**：仅记录路径引用（如 `triton_ascend_output/op_0_15_Pad_*/15_Pad_generated.py`）
- **强制规则**：Prompt 中必须包含"禁止直接复制历史代码结构，必须根据当前任务重新设计"

## 4. 经验存储结构

每个算子类别的经验文件（`kernel-opt-{category}.md`）包含以下章节：

```
## Layer 1: 设计约束
- 必须做/禁止做的事
- 已验证的无效方向（避免重复踩坑）

## Layer 2: 算法骨架
- 核心并行策略（grid 维度选择、元素分配方式）
- 分块/tiling 策略
- 多核分配模式

## Layer 3: 关键技巧
- 代码片段（带"可替代"标注）
- tl.load/tl.store 的 mask 处理模式
- 边界条件处理技巧

## Layer 4: 完整归档（Agent 默认不读取）
- 历史实现路径
- 性能基准
- 备注：完整代码仅用于人工复盘，Agent 禁止直接引用

## 常见陷阱与避免方法
- 精度问题来源
- 性能退化场景
- 边界 case 处理遗漏
```

## 5. 复用机制与防依赖策略

### 5.1 阶段化访问控制

| 工作流阶段 | 可读取层级 | 必须遵守的规则 |
|-----------|-----------|---------------|
| Phase 2 (kernel-designer) | Layer 1 + Layer 2 | 仅作为设计约束参考，输出必须是全新草图 |
| Phase 3 (kernel-generator) | Layer 1 + Layer 2 + Layer 3 | 技巧"可参考但不可复制"，变量名/结构必须重新设计 |
| Phase 4 (latency-optimizer) | 全部（含 Archive 路径） | 优先尝试历史未使用过的优化方向；若复用技巧需明确说明来源 |
| Conductor 修复 | Layer 1 + 常见陷阱 | 禁止直接复制历史代码修复问题 |

### 5.2 多样性保护机制

- **并列记录**：若新实现采用与历史完全不同的思路且通过验证，将该思路**并列记录**，而非覆盖旧经验
- **版本标注**：每条经验标注首次验证通过的日期和算子版本，过期经验标注 `[DEPRECATED]`
- **探索配额**：Phase 4 优化时，latency-optimizer 必须至少尝试 1 个历史未记录过的优化方向，再考虑历史技巧

### 5.3 新任务启动时查询

```
1. 从任务描述提取算子类别
2. 读取对应类别的经验文件（仅 Layer 1-3）
3. 将 Layer 1 约束作为 kernel-designer 的 negative_prompt（禁止事项）
4. 将 Layer 2 骨架作为 kernel-designer 的参考方向（可选策略）
5. 将 Layer 3 技巧作为 kernel-generator 的 "known-good patterns"（仅供参考）
6. 明确注入提示："历史经验仅供启发，不得复制代码结构"
```

### 5.4 经验更新规则

- 每次算子任务完成后，提取至少 1 条可复用经验（优先提取 Layer 1 约束或 Layer 3 新技巧）
- 若同类经验已存在且新经验与之等价，不重复记录
- 若新经验与旧经验矛盾，并列记录并标注各自适用条件
- 删除已被证伪的优化策略（标注 `[DEPRECATED]` 保留追溯）
