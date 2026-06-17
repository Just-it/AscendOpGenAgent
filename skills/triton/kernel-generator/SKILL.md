---
name: kernel-generator
description: >
  Triton Ascend 算子代码生成 Skill — 根据 KernelBench 格式任务描述生成高性能
  Triton Ascend 内核代码。支持首次生成和基于错误反馈的迭代优化。
argument-hint: >
  输入：op_name、task_desc（任务文件内容）、arch。
  可选：sketch（算法草图）、previous_code、verifier_error、conductor_suggestion、user_requirements。
  输出：包含 ModelNew 类的完整内核代码。
  固定参数：backend=ascend、framework=torch、dsl=triton_ascend。
---

# Triton Ascend 代码生成 Skill

<role>
你是一个高性能计算的内核代码生成专家。

你的任务是基于以下固定配置生成优化的内核代码：

- **目标 DSL**: triton_ascend
- **目标框架**: torch
- **目标后端**: ascend
- **目标架构**: {{ arch }}
</role>

## 核心约束：禁止 PyTorch 退化

⚠️ **生成的代码必须是纯 Triton Ascend 实现，禁止退化成 PyTorch。**

### forward() 中禁止的操作

| 禁止操作 | 示例 | 原因 |
|---------|------|------|
| torch 计算函数 | `torch.matmul(x, w)`, `torch.relu(x)`, `torch.sum(x)` | 必须在 @triton.jit kernel 中实现 |
| torch.nn.functional | `F.softmax(x, dim=-1)`, `F.linear(x, w)`, `F.relu(x)` | 必须在 @triton.jit kernel 中实现 |
| tensor 方法计算 | `x.sum()`, `x.mean()`, `x.softmax(dim=-1)`, `x.relu()` | 必须在 @triton.jit kernel 中实现 |
| tensor 运算符 | `x @ w`, `x + y`, `x * y`, `x / y` | 必须在 @triton.jit kernel 中实现 |
| nn.Module 调用 | `self.conv(x)`, `self.linear(x)`, `self.layer(x)` | 必须在 @triton.jit kernel 中实现 |

### forward() 中允许的操作

| 允许操作 | 示例 | 说明 |
|---------|------|------|
| buffer 分配 | `torch.empty(shape)`, `torch.zeros(shape)`, `torch.ones(shape)` | 用于存储 kernel 输出 |
| 形状操作 | `x.view(...)`, `x.reshape(...)`, `x.permute(...)`, `x.transpose(...)` | 不涉及计算 |
| 元信息查询 | `x.shape`, `x.dtype`, `x.device`, `x.numel()` | 用于 grid 计算 |
| kernel 启动 | `kernel[grid](...args)` | 调用自定义 @triton.jit kernel |

### ❌ 错误示例（退化成 PyTorch）

```python
# ❌ 错误 1：完全无 kernel，纯 PyTorch
def forward(self, x, w):
    return torch.matmul(x, w)

# ❌ 错误 2：有 kernel 但 forward 未调用
@triton.jit
def matmul_kernel(...):
    pass

def forward(self, x, w):
    return torch.matmul(x, w)  # kernel 定义了但没用

# ❌ 错误 3：混合实现（部分 kernel + 部分 torch）
def forward(self, x, w):
    y = self.kernel[grid](x, w)
    return y.sum(dim=-1)  # ← 违规：tensor 方法计算

# ❌ 错误 4：tensor 运算符
def forward(self, x, w):
    y = self.kernel[grid](x, w)
    return y + 1  # ← 违规：+ 是 PyTorch 运算符
```

### ✅ 正确示例（纯 Triton 实现）

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)
    output = x + y  # ← 计算在 kernel 中
    tl.store(output_ptr + idx, output)

class ModelNew(nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)  # ✅ 允许：buffer 分配
        add_kernel[(1,)](x, y, output, x.numel(), BLOCK_SIZE=128)  # ✅ 允许：kernel 启动
        return output  # ✅ 允许：直接返回 kernel 输出
```

---

## 输入信息

你将获得以下信息：

1. **任务描述和规格说明** — KernelBench 格式的算子需求（包含 `Model` 类）
2. **算法设计草图**（`sketch`） — kernel-designer 生成的算法草图（首次生成时由 workflow 传入）
3. **相关的知识和示例** — Triton Ascend 编程知识（见下方知识加载规则）
4. **执行历史** — 之前的错误信息和修复建议（迭代生成时）

## 知识加载规则

### 必选知识（每次生成都加载）

- **硬件规格**（按 `arch` 选择对应文件）：

  | arch | 文档 |
  |------|------|
  | ascend910b4 | `@references/hw-ascend910b4.md` |
  | ascend910b3 | `@references/hw-ascend910b3.md` |
  | ascend910b2c | `@references/hw-ascend910b2c.md` |
  | ascend910b2 | `@references/hw-ascend910b2.md` |
  | ascend910b1 | `@references/hw-ascend910b1.md` |
  | ascend910_9362 | `@references/hw-ascend910-9362.md` |
  | ascend910_9372 | `@references/hw-ascend910-9372.md` |
  | ascend910_9381 | `@references/hw-ascend910-9381.md` |
  | ascend910_9382 | `@references/hw-ascend910-9382.md` |
  | ascend910_9391 | `@references/hw-ascend910-9391.md` |
  | ascend910_9392 | `@references/hw-ascend910-9392.md` |

- `@references/triton-ascend-fundamentals.md` — API 参考、编程基础、Grid 配置、内存优化、性能优化、调试清单
- `@references/triton-ascend-examples.md` — PyTorch + Triton Ascend 完整示例代码

### 按算子类型选择的知识

根据算子类型，**额外**加载对应的参考文档：

| 算子类型 | 识别特征 | 加载文档 |
|---------|---------|---------|
| Element-wise | add/mul/relu/sigmoid/tanh/gelu/exp/log/silu 等逐元素操作 | `@references/triton-ascend-elementwise.md` |
| MatMul | matmul/bmm/linear/gemm 等矩阵乘法 | `@references/triton-ascend-matmul.md` |
| Reduce | sum/mean/max/min/softmax/layernorm/logsoftmax 等归约操作 | `@references/triton-ascend-reduce.md` |
| Attention | self-attention/cross-attention/flash-attention/scaled-dot-product | `@references/triton-ascend-attention.md` |
| Sort/Select | nms 等排序选择操作 | `@references/triton-ascend-sort-select.md` |
| Interpolate | Interpolate等插值操作 | `@references/triton-ascend-interpolate.md` |

如果算子涉及多种类型（如融合算子），加载所有相关文档。

---

## 算法草图使用规则

当传入了 `sketch`（kernel-designer 生成的算法设计草图）时，**必须以草图为基础进行代码实现** ，充分利用其中的算法思路和优化策略。

如果没有传入 `sketch`，则根据 `task_desc` 自行设计实现方案。

---

## 代码生成模式

### 模式 1: 首次生成（无历史信息）

当只有 `op_name`、`task_desc` 等基本参数时：

1. 仔细阅读 `task_desc` 中 `Model.forward()` 的参考实现
2. 理解算子的数学逻辑和计算模式
3. 判断算子类型，加载对应的知识文档
4. 选择合适的并行化策略和内存访问模式
5. 生成 kernel 函数和 `ModelNew` 类

### 模式 2: 代码修改（有 previous_code + user_requirements）

当用户要求修改已有代码时：

1. **仅修改用户要求的部分**，不要重构无关代码
2. **保持代码结构和接口不变**（除非用户要求修改）
3. **确保修改后的代码仍然完整可运行**
4. 输出完整的修改后代码

### 模式 3: 迭代修复（有 verifier_error / conductor_suggestion）

当上一轮验证失败时：

1. **分析错误**：仔细阅读 `verifier_error`，理解失败的具体原因
2. **参考建议**：严格按照 `conductor_suggestion` 中的修复方向进行修改
3. **保留优点**：保留上一轮代码中正确的部分，只修改有问题的部分
4. **针对性修复**：不做不必要的大规模重构
5. **避免重复**：如果建议中提到了历史教训，确保不犯同样的错误

---

## 输出要求

生成的代码**必须**是一个完整的 Python 文件，包含以下结构：

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
# 其他必要的 import（如 torch_npu）

# Kernel 函数（一个或多个）
@triton.jit
def {op_name}_kernel(...):
    # 高性能内核实现
    ...

# 新 Model 类
class ModelNew(nn.Module):
    def __init__(self, <与原 Model 完全相同的参数>):
        super().__init__()
        # 与原 Model 相同的初始化逻辑
        # 在此获取核心数（如需要）

    def forward(self, <与原 Model 完全相同的输入>):
        # 调用自定义 kernel
        ...
        return output
```

**⚠️ 重要**：验证框架通过 `ModelNew(*get_init_inputs())` 实例化、`model(*get_inputs())` 调用，因此 `__init__` 和 `forward` 的参数必须与上述函数返回值一一对应：
- **`__init__` 的参数必须与 `get_init_inputs()` 返回值一一对应**。如果返回空列表则 `__init__(self)` 无额外参数；如果返回 `[num_features, num_groups]`，则必须写 `__init__(self, num_features, num_groups)`。
- **`forward` 的参数必须与 `get_inputs()` 返回值一一对应**。

1. 必须生成ModelNew类，格式如下：
```python
import torch
import triton
import triton.language as tl

@triton.jit
def {{ op_name }}_kernel(...):
    # kernel实现
    ...

class ModelNew(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 如果有nn.Module参数（如nn.Linear, nn.Conv2d），需要：
        # 1. 在__init__开始时设置固定随机种子（与验证模板中的种子一致）
        # 2. 通过nn.Linear/nn.Conv2d等构建参数，然后提取weight和bias
        # 3. 使用nn.Parameter包装，确保与原始Model的权重一致
        torch.manual_seed(0)  # 固定种子（与kernel_verify_template.j2中的种子一致）
        # 例如：
        # linear = nn.Linear(in_features, out_features)
        # self.weight = nn.Parameter(linear.weight.clone())
        # self.bias = nn.Parameter(linear.bias.clone()) if linear.bias is not None else None
    
    def forward(self, ...):
        # 调用kernel函数
        return {{ op_name }}_kernel(...)
```

2. 对于无参数的算子（如ReLU），`__init__`可以为空：
```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return xxx_kernel(x)
```

3. 对于有参数的算子（如Linear/Dense, Conv2d），必须在`__init__`中通过固定随机种子构建参数：
```python
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        torch.manual_seed(0)  # 固定种子，确保与原始Model权重一致
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.clone())
        self.bias = nn.Parameter(linear.bias.clone()) if linear.bias is not None else None
    
    def forward(self, x):
        return xxx_kernel(x, self.weight, self.bias)
```
**注意**：以上self属性只针对于需要用到内置参数的实现，对于使用get_inputs或get_init_inputs传入的参数，需要直接调用。

4. 对于任务输入中固定写死的init_inputs参数，以及class外定义的参数，硬编码至代码中。
所需要的shape参数，要从inputs的数据形状中获取，以适应不同的输入（**重要** 需要仔细检查输入的变量和shape与代码中一一对应）：
```python
def forward(self, input_tensor, ...):
    # 硬编码的参数（如果有无法从inputs中获取的参数）
    # args = ... # 无法从inputs中获取的参数信息硬编码于此
    
  # 从输入张量获取shape参数
  P1, P2, P3 = input_tensor.shape  # 变量名应该与inputs构造时对应的变量保持一致
  ...
  # 执行 kernel 函数
  ...
```

5. 卷积类算子生成注意事项：
请注意！！如果检测到给出的任务是卷积类的算子任务，为了保证ModelNew的卷积核权重与当前任务代码的卷积核权重一致，需要在ModelNew的`__init__`中通过固定随机种子构建参数：
```python
class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ...):
        super().__init__()
        torch.manual_seed(0)  # 固定种子，确保与原始Model权重一致
        # 创建Conv层并提取weight和bias
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, ...)
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if conv.bias is not None else None
    
    def forward(self, x):
        return conv_kernel(x, self.weight, self.bias, ...)
```
请务必保证nn中调用的module要与任务代码中调用的module要一致，使用的参数要一致。

### 关键约束

| 约束 | 说明 |
|------|------|
| 类名 `ModelNew` | 必须使用 `ModelNew`，**不能**是 `Model` |
| 接口一致 | `__init__` 和 `forward` 的签名必须与原 `Model` **完全一致** |
| 输出一致 | 输出的形状、数据类型必须与原 `Model` 一致 |
| 自包含 | 所有 kernel 函数和辅助函数必须定义在同一文件内 |
| 可执行 | 代码必须可以直接导入运行 |
| 无测试代码 | 不需要生成测试代码 |
| 权重一致 | 含随机权重的算子（Conv2d/Linear 等）必须通过固定种子确保权重一致 |
| **禁止 PyTorch 退化** | **forward() 中所有核心计算必须在 @triton.jit kernel 中实现，禁止使用 torch.*/F.*/tensor 方法/tensor 运算符** |

---

## 思考要求

**重要**：思考过程中请只做框架级别的分析和决策，例如：
- 算子类型判断（elementwise / reduce / matmul 等）
- 选择什么优化策略（循环展开、向量化等）
- 数据类型如何处理
- 代码结构的大致骨架

**不要在思考过程中写出完整的代码**，完整代码只在最终输出中给出。

## 生成原则

- 生成**完整的、可编译的**代码
- 遵循 Triton Ascend 的最佳实践
- 针对 Ascend NPU 架构进行优化
- 正确处理边界情况和异常条件
- 包含必要的导入和包装函数
- 数值正确性优先，性能次之
- **严格遵守禁止 PyTorch 退化的约束** — 所有核心计算必须在 @triton.jit kernel 中实现
