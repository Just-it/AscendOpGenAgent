---
name: op-task-extractor
description: >
  从用户 PyTorch/Python 代码中提取算子实现，构建为 KernelBench 格式的标准化
  单文件自包含任务（task_desc.py）。支持从独立的文件中提取 shape/dtype 信息。
argument-hint: >
  需要提供：1) 待优化的代码文件路径；
  2) 可选：shape/dtype 信息来源文件路径
---

# 算子任务提取 Skill

<role>
你是一个算子任务提取专家。你的任务是从用户提供的代码中提取出可优化的
算子部分，并将其构建为 KernelBench 格式的 task_desc.py 文件。
</role>

## 目标格式

最终生成的文件必须是 **单一自包含 Python 文件**，**仅包含以下 4 个部分**：

1. `import` 区：只允许 torch / torch.nn / 标准库
2. `class Model(nn.Module)`：包装待优化算子逻辑（含 `__init__` 和 `forward`）
3. `def get_inputs()`：返回 `forward()` 的输入参数列表
4. `def get_init_inputs()`：返回 `__init__()` 的初始化参数列表

详细格式规范见 `@references/kernelbench-format.md`

---

## 提取流程

### Step 1: 代码分析

- 读取用户提供的源代码文件
- 读取 `arch` 配置（`framework=torch`、`backend=ascend`、`dsl=triton_ascend` 为固定值）
- 识别用户指定的待优化部分——即待优化的算子/计算逻辑
- 从用户的描述或提供的文件中提取 shape, dtype 等信息
- 确定算子的输入/输出签名

### Step 2: 依赖追踪

- 分析目标代码段的依赖关系（AST 级别）
- 追踪所有被调用的自定义函数/类
- 确定需要内联的外部依赖
- 识别 import 依赖链，区分标准库/PyTorch 与自定义模块

### Step 3: 构建 task_desc.py

- 将目标算子逻辑包装到 `Model.forward()` 中
- 如果算子有初始化状态（如权重、参数），放入 `Model.__init__()`
- 将所有依赖的自定义函数内联到文件中（禁止 import 外部模块）
- 根据 shape/dtype 信息构建 `get_inputs()` 和 `get_init_inputs()`
- 如果用户未提供 shape/dtype，从代码上下文推断合理默认值

### Step 4: 验证（必须执行，禁止跳过）

**使用本 skill 自带的验证脚本**进行静态检查和运行时检查。

**验证命令**（使用 `bash` 工具执行）：

```bash
python3 <skill-path>/scripts/validate_task.py /abs/path/task_desc.py --json
```

其中 `<skill-path>` 为本 skill 所在目录的绝对路径。

**验证脚本检查项**：
1. **静态检查**：`class Model(nn.Module)`、`forward`、`get_inputs`、`get_init_inputs` 是否齐全
2. **运行时检查**：`exec` → `Model(*get_init_inputs())` → `model(*get_inputs())` → NaN/Inf 检查 → 两次运行一致性检查

**结果处理**：
- 脚本退出码为 0 且输出 `[VALID]` → 验证通过，进入 Step 5
- 脚本退出码非 0 且输出 `[INVALID]` → 根据错误信息修复代码，**重新运行验证脚本**（最多重试 2 次）
- 重试 2 次仍失败 → 向用户报告错误，请求协助

**⛔ 禁止事项**：
- 禁止跳过验证直接进入 Step 5
- 禁止用自创的方法替代验证脚本
- 禁止仅做静态检查就认为验证通过（必须同时通过运行时检查）

### Step 5: 用户确认（必须执行，禁止跳过）

验证通过后，**必须使用 `question` 工具**将完整的 task_desc.py 内容展示给用户，请求确认。

1. 展示任务代码内容
2. 询问用户
> 算子生成完成，请查看生成代码：
>
> 请选择：
> 1. 接受
> 2. <让用户输入修改要求>

**具体要求**：
- **必须调用 `question` 工具**，不能只打印文本让用户选择
- 在 `question` 的描述中展示 task_desc.py 的完整内容
- 提供"确认"和"需要修改"两个选项

**处理回复**：
- 用户确认 → 算子提取任务完成
- 用户要求修改 → 结合用户反馈返回 Step 3 重新生成

**⛔ 禁止事项**：
- 禁止不经用户确认就结束此 skill
- 禁止用普通文本消息替代 `question` 工具调用

---

## 关键约束

| 约束 | 说明 |
|------|------|
| 自包含 | 所有依赖函数必须内联到文件中，禁止 import 项目内模块 |
| 可执行 | `Model(*get_init_inputs()).forward(*get_inputs())` 必须直接运行 |
| 确定性 | 给定相同输入，输出必须一致 |
| 无 NaN/Inf | forward 输出不能包含 NaN 或 Inf |
| 禁止重写 | 原始函数可运行就直接复用，一行都不改 |
| 返回一致 | 返回类型/形状必须与原始实现一致 |
| 合理输入 | get_inputs 应提供合理大小的输入（不能过小或过大） |

---

## 示例

### 输入

用户说："`/path/to/model.py` 的 `matmul_with_bias` 函数有优化空间，shape 信息在 `/path/to/config.py`"

### 输出 task_desc.py

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t()) + self.bias


def get_inputs():
    batch_size = 32
    in_features = 1024
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    in_features = 1024
    out_features = 512
    return [in_features, out_features]
```
