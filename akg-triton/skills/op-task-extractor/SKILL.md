---
name: op-task-extractor
description: >
  算子任务提取Skill - 从用户代码中提取算子逻辑，生成KernelBench格式任务描述。
argument-hint: >
  输入：用户代码文件路径。
  输出：KernelBench格式的{op_name}.py任务文件。
---

# 算子任务提取 Skill

## 功能概述

Op Task Extractor负责：
1. **代码分析**：分析用户提供的算子代码
2. **逻辑提取**：提取算子的核心计算逻辑
3. **格式转换**：转换为KernelBench标准格式
4. **输入生成**：自动生成测试输入函数

## 工作流程

```
输入：用户代码文件
    ↓
[代码解析] → 解析代码结构
    ↓
[算子识别] → 识别算子类型和接口
    ↓
[逻辑提取] → 提取forward实现
    ↓
[输入分析] → 分析输入形状和dtype
    ↓
[格式生成] → 生成KernelBench格式
    ↓
[验证检查] → 验证生成的任务文件
    ↓
输出：{op_name}.py任务文件
```

## KernelBench格式

生成的任务文件包含：

```python
import torch
import torch.nn as nn

# 算子实现
class Model(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 初始化参数
    
    def forward(self, x, ...):
        # 算子逻辑
        return output

# 输入生成函数
def get_inputs():
    # 生成测试输入
    return [input1, input2, ...]

# 初始化输入（可选）
def get_init_inputs():
    # 生成初始化输入
    return [init_input1, ...]
```

## 提取内容

### 1. 算子类定义

提取`nn.Module`子类的定义：
- 类名
- `__init__`方法参数
- `forward`方法签名

### 2. 计算逻辑

提取`forward`方法的实现：
- 完整的计算逻辑
- 所有PyTorch操作
- 控制流（if/for/while）

### 3. 输入规格

分析输入张量的规格：
- 形状（Shape）
- 数据类型（dtype）
- 设备（device）

### 4. 参数规格

提取模型参数：
- 权重张量
- 偏置张量
- 超参数

## 使用方式

### 在op-optimizer中调用

Phase 2自动调用：

```
Phase 2: 构建任务描述代码
    ↓
加载 op-task-extractor skill
    ↓
提取算子逻辑
    ↓
生成{op_name}.py
```

### 独立调用

```bash
skill op-task-extractor \
  --input-file ./user_relu.py \
  --output-path ${pwd}/triton_ascend_output/tasks/ \
  --op-name relu
```

### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| --input-file | string | 是 | 用户代码文件路径 |
| --output-path | string | 否 | 输出目录，默认${pwd}/triton_ascend_output |
| --op-name | string | 否 | 算子名称，自动推断 |

## 支持的输入格式

### 格式1：完整nn.Module

```python
import torch.nn as nn

class MyReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.relu(x)
```

### 格式2：函数式实现

```python
import torch

def my_relu(x):
    return torch.relu(x)
```

### 格式3：融合算子

```python
class FusedOp(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        # LayerNorm + GELU融合
        normalized = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        return self.weight * normalized + self.bias
```

## 输出示例

### 输入：用户代码

```python
# user_layernorm.py
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
```

### 输出：任务文件

```python
# layernorm.py
import torch
import torch.nn as nn
from typing import Tuple

class Model(nn.Module):
    def __init__(self, hidden_size: int = 768, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

def get_inputs() -> Tuple[torch.Tensor]:
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    x = torch.randn(batch_size, seq_len, hidden_size)
    return (x,)

def get_init_inputs() -> Tuple[int, float]:
    return (768, 1e-5)
```

## 验证检查

生成后会自动验证：

```bash
python validate_kernelbench_task.py layernorm.py --json
```

**验证内容**：
- 语法正确性
- 类名是否为`Model`
- 是否包含`get_inputs`函数
- 输入输出类型是否匹配

## 错误处理

| 错误类型 | 原因 | 处理 |
|---------|------|------|
| 解析失败 | 代码语法错误 | 提示用户修复代码 |
| 未找到算子 | 没有nn.Module | 提示提供完整类定义 |
| 输入推断失败 | 形状不确定 | 使用默认形状或提示用户 |
| 验证失败 | 格式不符合 | 自动修复或提示 |

## 最佳实践

1. **提供完整代码**：包含所有import和类定义
2. **明确形状**：在代码中体现张量形状
3. **避免外部依赖**：只使用标准PyTorch操作
4. **检查生成结果**：确认任务文件符合预期
