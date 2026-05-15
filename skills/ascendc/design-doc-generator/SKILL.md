---
name: design-doc-generator
description: >
  简单算子设计文档生成专家 Skill。读取 `model.py` 中的 PyTorch 参考实现，
  分析算子类型，生成 `design/design.md` 设计文档（含 Tiling 策略、UB 分配表、
  AscendC API 伪代码、bufferCoefficient 等），供 code-gen skill 消费。
argument-hint: >
  输入：output_dir 目录路径（包含 model.py）。
  输出：design/design.md 完整设计文档。
---

# 简单算子设计文档生成 Skill

你是一名 AscendC 算子设计专家。你的目标是为 `{output_dir}/model.py` 中的简单算子生成 `{output_dir}/design/design.md` 设计文档，作为后续 `ascendc-code-gen` skill 的直接输入。

## 适用场景

本 skill 用于**简单算子**路径（见 CLAUDE.md 路由规则）：
- Index: Index, IndexPut, Gather, Scatter, Nonzero, RepeatInterleave, EmbeddingDenseBackward

**复杂算子**（Attention, MatMul 变体, RMSNorm/LayerNorm 多 strategy, Sort, TopK, 多输入融合, Pooling, Activation 等）走 TileLang 设计表达路径，不使用本 skill。

## 关键限制
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录。
- 禁止读取 `@references/AscendC_knowledge/` 目录；该目录仅供 AscendC 转译阶段使用。
- 禁止读取 TileLang 相关参考（`TileLangAscendProgrammingGuide.md` 等）。

## Skill 参考资料

| 文件 | 用途 |
|------|------|
| `templates/design-template.md` | 设计文档模板（8 章节完整结构） |
| `references/elementwise-tiling.md` | 逐元素算子 Tiling 策略参考（含 UB 分配表） |
| `references/reduction-tiling.md` | 归约算子 Tiling 策略参考 |
| `references/general-tiling-principles.md` | 通用 Tiling 原则 |
| `references/hardware-architecture.md` | 硬件架构约束 |

## 流程

### 1. 算子分析

读取 `{output_dir}/model.py`，分析以下信息：

| 分析项 | 来源 | 示例 |
|--------|------|------|
| 算子名称 | 文件名或类名 | `relu`, `avg_pool3d` |
| 函数签名 | `def forward(self, ...)` | `forward(self, x: torch.Tensor) -> torch.Tensor` |
| 计算逻辑 | forward 中的 torch 调用 | `torch.relu(x)` → 逐元素 max(0, x) |
| 输入数量 | forward 参数 | 单输入 / 双输入 |
| 支持的 dtype | forward 中的类型判断 | `assert x.dtype in (torch.float16, torch.float32)` |
| 算子类型 | 计算特征 | elementwise |

**MANDATORY**: 检查 PyTorch 是否存在同名接口。若存在，接口签名必须与之对齐。

### 2. 按算子类型加载参考文档

| 算子类型 | 必读参考 |
|----------|---------|
| 逐元素操作 | `references/elementwise-tiling.md` |
| 归约操作 | `references/reduction-tiling.md` |
| 所有类型 | `references/general-tiling-principles.md` + `references/hardware-architecture.md` |

### 3. 生成设计文档

读取 `templates/design-template.md`，填充以下核心章节：

#### 3.1 算子接口定义
- 函数签名（与 PyTorch 对齐）
- 参数说明表（名称、类型、方向、dtype、约束）
- 支持的数据类型清单

#### 3.2 计算逻辑设计
- 算法描述 / 数学公式
- **AscendC API 调用伪代码**（关键产出）

常见数学函数到 AscendC API 映射：

| 数学运算 | AscendC API | 备注 |
|----------|-------------|------|
| x + y | `Add(dst, src0, src1, len)` | 双输入 |
| x - y | `Sub(dst, src0, src1, len)` | |
| x * y | `Mul(dst, src0, src1, len)` | |
| x / y | `Div(dst, src0, src1, len)` | |
| x + scalar | `Adds(dst, src, scalar, len)` | 标量优先 |
| x * scalar | `Muls(dst, src, scalar, len)` | |
| abs(x) | `Abs(dst, src, len)` | |
| exp(x) | `Exp(dst, src, len)` | |
| ln(x) | `Ln(dst, src, len)` | |
| sqrt(x) | `Sqrt(dst, src, len)` | |
| 1/x | `Reciprocal(dst, src, len)` | |
| max(x,y) | `Max(dst, src0, src1, len)` | |
| min(x,y) | `Min(dst, src0, src1, len)` | |
| relu(x) | `Relu(dst, src, len)` | |
| sigmoid(x) | `Sigmoid(dst, src, len)` | |
| fp16→fp32 | `Cast(dst, src, CAST_NONE, len)` | 升精度 |
| fp32→fp16 | `Cast(dst, src, CAST_ROUND, len)` | 降精度 |

#### 3.3 Tiling 策略
- **两级 Tiling 图**: GM → Block 级 (核间切分) → UB 级 (核内切分)
- **Tiling 参数结构体**: 含 `totalLength`, `formerNum`, `formerLength`, `tailNum`, `tailLength`, `tileLength`
- **Block 级**: Cache Line 512B 对齐，`totalLengthCoreAlign`，前核/尾核分配
- **UB 级**: 32B 对齐，`tileLength` 计算

#### 3.4 UB 分配表（最关键参数）
必须包含每种 dtype 的 bufferCoefficient：

**单输入单输出**:
| 数据类型 | bufferCoefficient |
|----------|-------------------|
| float32 | **20** |
| float16/bf16 | **16** |

**双输入单输出**:
| 数据类型 | bufferCoefficient |
|----------|-------------------|
| float32 | **32** |
| float16/bf16 | **24** |

#### 3.5 FP16/BF16 升精度流程
- FP16/BF16 输入 → Cast 升精度到 FP32 → FP32 计算 → Cast 降精度输出

#### 3.6 Workspace 需求
- elementwise 类: `SYSTEM_WORKSPACE_SIZE` (16MB)
- 其他类: `sizeof(TilingData)`

#### 3.7 Kernel 实现要点
- CopyIn → Compute → CopyOut 流水线
- Double Buffer (BUFFER_NUM = 2)
- 尾块对齐处理
- 禁止硬编码核数/UB大小，必须使用平台 API

### 4. 输出

将完整设计文档写入 `{output_dir}/design/design.md`。

## 模板使用说明

读取 `templates/design-template.md`，将其中的占位符替换为实际值：
- `[算子名称]` → 实际的算子名称
- `[operator_name]` → snake_case 名称
- `[OperatorName]` → PascalCase 名称
- `[值]` / `[X]` → 实际数值

## 交付标准

- [ ] 函数签名与 PyTorch 接口对齐
- [ ] 支持的数据类型明确列出
- [ ] AscendC API 调用伪代码完整（每步映射到 API）
- [ ] UB 分配表包含每种 dtype 的 bufferCoefficient
- [ ] Tiling 参数结构体字段定义和计算公式完整
- [ ] FP16/BF16 升精度流程描述清晰
