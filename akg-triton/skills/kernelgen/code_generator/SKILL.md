---
name: kernelgen-code-generator
description: >
  KernelGen Code Generator Skill - 基于Skill系统生成高性能内核代码。
  支持动态Skill选择、多轮迭代优化、多DSL支持。
argument-hint: >
  输入：task描述、DSL类型、历史错误、修复建议。
  输出：生成的内核代码。
---

# KernelGen Code Generator Skill

## 功能概述

Code Generator是KernelGen的核心代码生成组件，负责：
1. **Skill选择**：基于任务动态选择相关Skill知识
2. **代码生成**：根据任务描述生成高性能内核代码
3. **多DSL支持**：支持Triton CUDA、Triton Ascend、C++等
4. **迭代优化**：基于错误反馈优化代码

## 工作流程

```
输入：Task描述 + DSL + Framework + Backend
    ↓
[Skill选择] → 粗筛 + 精筛(LLM)
    ↓
[Prompt构建] → System Prompt + User Prompt + Skill内容
    ↓
[代码生成] → LLM生成代码
    ↓
[代码解析] → 提取可执行代码
    ↓
输出：生成的内核代码
```

## Skill系统

### Skill选择流程

**阶段1：粗筛（Coarse Filter）**
- 基于DSL过滤：只选择目标DSL相关的Skill
- 基于Backend过滤：匹配后端类型
- 基于Category过滤：选择knowledge类别

**阶段2：精筛（Fine Filter）**
- LLM根据任务描述选择最相关的Skill
- 返回排序后的Skill列表

### Skill类别

| Category | 说明 | 示例 |
|----------|------|------|
| fundamental | 基础知识 | Triton编程基础、API参考 |
| method | 方法论 | 优化技巧、调试方法 |
| implementation | 实现参考 | 典型算子实现示例 |
| example | 完整示例 | 端到端代码示例 |

## 代码生成策略

### 首次生成

基于任务描述和选择的Skill生成初始代码：

```python
# 输入
op_name = "layernorm"
task_desc = "LayerNorm算子实现..."
dsl = "triton_cuda"
framework = "torch"

# 输出
code = generate_code(
    op_name=op_name,
    task_desc=task_desc,
    dsl=dsl,
    framework=framework,
    skills=selected_skills
)
```

### 迭代优化

基于Verifier错误和Conductor建议优化代码：

```python
# 输入
previous_code = generated_code
verifier_error = "数值验证失败，最大差异0.05"
conductor_suggestion = "检查reduce操作维度"

# 输出
optimized_code = regenerate_code(
    previous_code=previous_code,
    verifier_error=verifier_error,
    conductor_suggestion=conductor_suggestion,
    iteration=current_iteration
)
```

## Prompt模板

### System Prompt

```
你是一个专业的内核代码生成专家，精通{DSL}编程。

目标环境：
- DSL: {dsl}
- 后端: {backend}
- 框架: {framework}
- 架构: {arch}

要求：
1. 生成高性能、正确的内核代码
2. 遵循最佳实践和优化模式
3. 代码需包含完整的class ModelNew定义
4. 确保数值精度符合要求
```

### User Prompt

```
## 任务描述

算子名称：{op_name}

功能描述：
{task_desc}

## 参考知识

{skill_contents}

## 历史信息

{history_section}

## 输出要求

请生成完整的内核代码，包含：
1. class ModelNew(nn.Module)定义
2. kernel函数实现
3. forward函数调用kernel

{format_instructions}
```

## 支持的DSL

| DSL | 后端 | 说明 |
|-----|------|------|
| triton_cuda | cuda | Triton GPU编程 |
| triton_ascend | ascend | Triton Ascend NPU |
| cuda_c | cuda | CUDA C |
| cpp | cpu | C++ CPU实现 |
| tilelang_cuda | cuda | TileLang GPU |
| ascendc | ascend | AscendC |

## 使用方式

### 在KernelGen中调用

Code Generator作为KernelGen的内部组件自动调用。

### 独立调用

```bash
skill kernelgen/code_generator \
  --op-name layernorm \
  --task-file ./layernorm.py \
  --dsl triton_cuda \
  --framework torch \
  --backend cuda \
  --arch a100
```

## 输出格式

生成的代码必须包含：

```python
import torch
import triton
import triton.language as tl

# Kernel函数
@triton.jit
def layernorm_kernel(...):
    # 实现
    pass

# Model类
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, weight, bias):
        # 调用kernel
        layernorm_kernel[grid](...)
        return output
```

## 优化策略

### 自动优化

基于Skill知识自动应用优化：
- **内存访问优化**：合并访存、避免bank conflict
- **并行度优化**：合理设置block/thread配置
- **计算优化**：使用高效算法、减少冗余计算

### 用户指定优化

通过user_requirements传递优化需求：

```bash
--user-requirements "使用向量化加载，block大小设为128"
```

## 配置选项

```yaml
code_generator:
  model_level: "standard"  # standard/fast/complex
  
  skill_selection:
    coarse_filter: true
    fine_filter: true
    max_skills: 10
  
  generation:
    temperature: 0.1
    max_tokens: 4096
    
  optimization:
    enable_auto_optimize: true
    vectorize_load_store: true
```

## 与其他组件的关系

```
Skill系统
    ↓ 提供知识
Code Generator (本Skill)
    ↓ 生成代码
Verifier
    ↓ 验证
Conductor
    ↓ 分析 + 建议
Code Generator (重新生成)
```
