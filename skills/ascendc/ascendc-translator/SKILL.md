---
name: ascendc-translator
description: >
  AscendC kernel 转译与实现专家 Skill。将已通过验证的 TileLang 设计转译为 AscendC kernel，
  并生成 model_new_ascendc.py 调用 AscendC kernel。
argument-hint: >
  输入：output_dir 目录路径（包含已通过验证的 tile_level/ 和 model_new_tilelang.py）。
  输出：kernel/ 下的 AscendC 实现、model_new_ascendc.py。
---

# AscendC Kernel 转译 Skill

你是一名 AscendC kernel 转译与实现专家。你的目标是将已通过验证的 TileLang 设计转译为 AscendC kernel，并生成 `{output_dir}/model_new_ascendc.py` 调用 AscendC kernel，最终通过验证。

## 前置条件
本阶段开始前，以下产物必须已经存在且 TileLang 验证已通过：
- `{output_dir}/design/tile_level/` — 完整可执行的 TileLang kernel
- `{output_dir}/model_new_tilelang.py` — TileLang 优化实现

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_ascendc.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 AscendC 实现中应尽可能避免标量逐元素写法，优先使用块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。
- 禁止读取 `@references/TileLangAscendProgrammingGuide.md`；该文档是 TileLang 编程指南，仅供 TileLang 阶段使用，与本阶段无关。

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── block_level/  # TileLang block-level 设计（已由上一阶段完成）
│   │   └── tile_level/   # TileLang tile-level 设计（已由上一阶段完成，作为转译输入）
│   ├── kernel/           # 你的主要实现位置，放置 AscendC kernel
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── model_new_tilelang.py # 上一阶段产物，可参考但不要修改
│   └── model_new_ascendc.py  # 你的 AscendC 优化实现，调用 AscendC kernel
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料
本 skill 提供以下参考资料（位于 `@references/` 目录）：
- `@references/dsl2Ascendc.md` — TileLang 转 AscendC 指南
- `@references/TileLang-AscendC-API-Mapping.md` — TileLang 与 AscendC API 映射表
- `@references/AscendC_knowledge/` — AscendC 知识库目录
- `@references/AscendCVerification.md` — AscendC 验证指南
- `@references/evaluate_ascendc.sh` — AscendC 评测脚本

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

## 流程
执行以下各步骤前，必须先阅读对应的参考文档，再开始实现、验证与迭代。

1. `TileLang 转译成 AscendC`
   将 `{output_dir}/design/tile_level/` 下的 TileLang 设计转译为对应的 AscendC 实现，在 `{output_dir}/kernel/` 中生成 AscendC kernel 文件。
   参考文档：`@references/dsl2Ascendc.md`
   **实施转译前必须先阅读 `@references/TileLang-AscendC-API-Mapping.md`，逐一确认每个 TileLang API 对应的 AscendC API 映射关系，再根据映射查阅 `@references/AscendC_knowledge/` 下的具体 API 文档。禁止跳过 Mapping 直接编写 AscendC 代码。**
2. `生成 model_new_ascendc.py`
   编写 `{output_dir}/model_new_ascendc.py`，调用自定义 AscendC kernel 实现算子逻辑。
3. `实现方式校验（验证前强制检查）`
   在运行正确性验证之前，必须先校验 `model_new_ascendc.py`，确保使用自定义 AscendC kernel 实现，而非 torch/torch_npu 替代。
   **校验命令**：`python utils/implementation_check.py {output_dir}/model_new_ascendc.py --type ascendc`
   - 若返回 PASS：继续执行 AscendC 验证
   - 若返回 FAIL：**立即停止，禁止运行验证脚本**，返回本 skill 重新实现 AscendC kernel（计入迭代次数）
   详见下方「实现方式校验」章节。
4. `AscendC 验证与迭代`
   调用 `@references/evaluate_ascendc.sh {output_dir}` 验证 AscendC；如果结果不正确，继续迭代修改直到通过验证。迭代次数上限为 3 次，若 3 次迭代后仍未通过验证，停止迭代并报告当前状态。
   参考文档：`@references/AscendCVerification.md`

## 实现方式校验（验证前强制检查）

**⚠️ 重要：此校验必须在运行正确性验证之前执行！**

在 `model_new_ascendc.py` 生成后、运行验证脚本前，必须执行以下校验，确保使用自定义 AscendC kernel 实现，而非 torch/torch_npu 替代：

### 禁止的实现方式（严格禁止）

| 类别 | 禁止模式 | 示例 | 说明 |
|------|----------|------|------|
| PyTorch 函数调用 | `torch.*` | `torch.add`, `torch.mul`, `torch.sum`, `torch.mean`, `torch.matmul` 等 | 禁止用 PyTorch 计算 |
| PyTorch 神经网络函数 | `torch.nn.functional.*` | `F.relu`, `F.softmax`, `F.linear` 等 | 禁止用 PyTorch 计算 |
| **PyTorch NPU 接口** | **`torch_npu.*`** | **`torch_npu.npu_xxx`**, `torch_npu.npu_add` 等 | **禁止用 NPU 原生 API 替代自定义 kernel** |
| Tensor 计算方法 | `tensor.计算方法()` | `tensor.sum()`, `tensor.mean()`, `tensor.matmul()` 等 | 禁止用 PyTorch 计算 |
| 其他计算函数 | `torch.where`, `torch.clamp`, `torch.maximum`, `torch.minimum` 等 | 禁止用 PyTorch 计算 |

### 允许的操作（仅限以下操作）

| 类别 | 允许模式 | 示例 | 说明 |
|------|----------|------|------|
| 张量创建 | `torch.empty`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.tensor` | 仅用于创建输入/输出张量 | 数据准备 |
| 张量变换 | `.to()`, `.view()`, `.reshape()`, `.permute()`, `.contiguous()` | 仅用于调整张量布局 | 数据格式转换 |
| 类型/设备查询 | `.dtype`, `.device`, `.shape` | 用于获取张量元信息 | 信息查询 |
| **自定义 AscendC kernel** | **`ascendc_kernel(...)`**, **`kernel(...)`** | 调用 AscendC 实现的自定义算子 | **唯一允许的计算方式** |
| Python 标准库 | `import`, 控制流等 | 非 PyTorch 计算操作 | 辅助代码 |

### 核心要求

**`model_new_ascendc.py` 中的核心计算必须调用自定义 AscendC kernel，禁止以下替代方案：**

1. **禁止直接调用 `torch_npu.npu_xxx` 接口** - 即使是 NPU 原生接口也不允许，必须使用自定义 AscendC kernel
2. **禁止用 PyTorch 运算组合实现** - 如 `torch.add`, `torch.mul` 等
3. **禁止用 Python 循环 + PyTorch 标量运算实现** - 必须将计算下放到 AscendC kernel

### 校验方法

```bash
# 步骤1: 检查是否包含禁止的 torch.* 计算操作
grep -n "torch\.[a-zA-Z_]*\s*(" model_new_ascendc.py | grep -vE "torch\.(empty|zeros|ones|randn|arange|tensor|as_tensor|from_numpy|int32|int64|float16|float32|bfloat16|bool|nn\.Module|Tensor)\s*("

# 步骤2: 检查是否包含 torch_npu.* 接口（严格禁止）
grep -n "torch_npu\.[a-zA-Z_]*\s*(" model_new_ascendc.py

# 步骤3: 检查是否包含 tensor.计算方法() 调用
grep -n "\.[a-z_]*\s*(" model_new_ascendc.py | grep -E "\.(sum|mean|matmul|add|mul|div|sub|max|min|clamp|where|softmax|relu|linear)\s*(" | grep -v "def\|#"

# 步骤4: 检查是否调用了自定义 AscendC kernel（必须存在）
grep -n "kernel\s*(" model_new_ascendc.py | grep -v "def\|#"
```

### 校验命令

```bash
python utils/implementation_check.py {output_dir}/model_new_ascendc.py --type ascendc
```

### 处理规则

- **若发现使用 torch/torch_npu 实现替代**：
  1. **立即停止**，标记 Phase 4 失败，**禁止运行验证脚本**
  2. 向 ascendc-translator 提供具体违规代码行号和内容
  3. **要求重新实现 AscendC kernel**，将计算逻辑完整下放到 AscendC，而非在 Python 层用 torch/torch_npu 实现
  4. 重新进行实现方式校验，直至通过
  5. 校验通过后才允许运行正确性验证（计入 5 次迭代限制内）
