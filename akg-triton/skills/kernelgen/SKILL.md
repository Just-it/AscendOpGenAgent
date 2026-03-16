---
name: kernelgen
description: >
  KernelGen Agent - 基于Skill系统的快速迭代算子生成。
  流程：KernelGen → CodeChecker → Verifier → Conductor（失败时）→ 重新生成
mode: subagent
temperature: 0.1
tools:
  write: true
  edit: true
  bash: true
  skill: true
  read: true
argument-hint: >
  必需：task-file、framework、backend、arch、dsl。
  可选：max-iterations、enable-code-checker、user-requirements。
---

# KernelGen Agent

## 功能概述

KernelGen是基于Skill系统的快速迭代算子生成Agent，特点：
- **快速迭代**：1-5分钟完成代码生成和验证
- **智能修复**：Conductor自动分析失败原因并指导修复
- **Skill驱动**：动态加载相关Skill知识
- **默认首选**：需求明确场景下的默认选择

## 工作流程

```
┌─────────────┐
│  KernelGen  │ ← 基于Skill生成初始代码
└──────┬──────┘
       ↓
┌─────────────┐
│ CodeChecker │ ← 代码静态检查（可选）
└──────┬──────┘
       ↓（通过）
┌─────────────┐
│  Verifier   │ ← 编译、运行、验证正确性
└──────┬──────┘
       ↓（失败）
┌─────────────┐
│  Conductor  │ ← 分析失败原因，生成修复建议
└──────┬──────┘
       ↓（未达最大迭代）
    [重新生成]
       ↓
   [完成]
```

## 核心组件

### 1. Code Generator (代码生成器)
- 基于Skill系统动态选择知识
- 支持多种DSL（Triton CUDA、Triton Ascend、C++等）
- 支持多轮迭代优化

### 2. Code Checker (代码检查器)
- 静态代码分析
- 语法和风格检查
- 可配置启用/禁用

### 3. Verifier (验证器)
- 代码编译验证
- 数值正确性验证
- 性能分析（可选）

### 4. Conductor (编排器)
- **结果分析**：查看Verifier运行结果
- **错误分类**：识别错误类型（语法/逻辑/性能）
- **修复决策**：判断是否重新生成代码
- **建议生成**：为下一轮生成提供修复建议

## 使用方式

### 作为Skill调用

```bash
skill kernelgen \
  --task-file /path/to/relu.py \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --output-path ${pwd}/triton_ascend_output/
```

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| --task-file | string | 是 | - | KernelBench格式任务文件 |
| --framework | string | 是 | - | torch/mindspore |
| --backend | string | 是 | - | cuda/ascend/cpu |
| --arch | string | 是 | - | a100/ascend910b4等 |
| --dsl | string | 是 | - | triton_cuda/triton_ascend等 |
| --output-path | string | 否 | ${pwd}/triton_ascend_output | 输出目录 |
| --max-iterations | int | 否 | 5 | 最大迭代次数 |
| --enable-code-checker | bool | 否 | false | 启用代码检查 |
| --user-requirements | string | 否 | "" | 用户额外需求 |

## Conductor功能详解

### 结果查看

Conductor会分析Verifier的输出：
- 编译日志
- 运行时错误
- 数值对比结果
- 性能数据（如启用）

### 错误分类

| 错误类别 | 说明 | 处理策略 |
|---------|------|---------|
| SyntaxError | 语法错误 | 指出具体语法问题 |
| ImportError | 导入错误 | 检查依赖和模块路径 |
| LogicError | 逻辑错误 | 分析数值差异原因 |
| ShapeError | 形状不匹配 | 检查tensor维度处理 |
| DeviceError | 设备错误 | 检查设备访问代码 |
| TimeoutError | 超时 | 优化代码或调整配置 |

### 重新生成决策

Conductor根据以下因素决策：
- 当前迭代次数 < max_iterations
- 错误是否可修复
- 历史修复成功率
- 错误复杂度评估

### 修复建议生成

为下一轮KernelGen生成具体建议：
```
错误分析：
- 类型：LogicError
- 位置：forward函数第23行
- 原因：reduce操作维度处理错误

修复建议：
1. 检查tl.sum的axis参数
2. 确保输出形状与预期一致
3. 参考elementwise reduction模板
```

## 输出结果

```
${pwd}/triton_ascend_output/
├── generated_code.py          # 生成的算子代码
├── summary.json               # 执行摘要
│   {
│     "subagent": "kernelgen",
│     "success": true,
│     "iterations": 3,
│     "final_status": "success",
│     "error_history": [...]
│   }
├── logs/
│   ├── kernelgen.log          # 代码生成日志
│   ├── verifier.log           # 验证日志
│   └── conductor.log          # Conductor决策日志
└── iterations/                # 每轮迭代记录
    ├── iter_0/
    ├── iter_1/
    └── iter_2/
```

## 适用场景

✅ **推荐使用**：
- 需求明确的算子生成
- 快速原型验证
- 标准算子实现
- 时间敏感场景

❌ **不推荐使用**：
- 需要极致性能优化（考虑adaptive_search或evolve）
- 复杂融合算子（可能需要更多探索）

## 性能指标

- **典型耗时**：1-5分钟
- **成功率**：> 85%（标准算子）
- **平均迭代次数**：2-3次
- **并发支持**：单设备串行执行

## 依赖Skills

- `kernelgen/conductor` - Conductor功能
- `kernelgen/code_generator` - 代码生成
- `kernelgen/verifier_wrapper` - 验证包装

## 示例

### 示例1：生成ReLU算子

```bash
skill kernelgen \
  --task-file ./tasks/relu.py \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --output-path ${pwd}/triton_ascend_output/relu/
```

### 示例2：带代码检查的生成

```bash
skill kernelgen \
  --task-file ./tasks/matmul.py \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --enable-code-checker true \
  --max-iterations 10 \
  --output-path ${pwd}/triton_ascend_output/matmul/
```
