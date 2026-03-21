---
name: benchmark-evaluator
description: >
  Benchmark Evaluator Skill — 串行执行算子评测，调用用户指定的 Agent 生成代码并验证。
  支持多数据集选择（KernelBench、NPUKernelBench），支持 per-level problem 选择，始终生成评测报告。
argument-hint: >
  必需：agent_name、agent_workspace、level_problems。
  可选：benchmark_name、benchmark_path、output_root、arch、resume、timeout_per_task、warmup、repeats。
  level_problems 格式：{1: [1,2], 2: [1,3], 3: None}，None 表示该 level 全选。
  benchmark_name 支持 KernelBench（默认）、NPUKernelBench。
---

# Benchmark Evaluator Skill

<role>
你是一个自动化评测框架执行器。你的任务是串行执行 KernelBench 评测，调用用户指定的 Agent 生成代码，验证正确性，测试性能，并生成详细报告。
</role>

## 输入参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `agent_name` | str | 用户指定的 Agent 名称 | `"my-agent"` |
| `agent_workspace` | str | Agent 工作区路径（包含 agents/ 和 skills/） | `"/path/to/.opencode"` |
| `level_problems` | dict | 每个 level 的 problem 选择 | `{1: [1,2], 2: None}` |

### level_problems 格式说明

```python
# 选择 Level 1 的 problem 1,2
# 选择 Level 2 的 problem 1,3
# Level 3 全选
# Level 4 不选（不传入）
{
    1: [1, 2],        # 只选指定 problems
    2: [1, 3],        
    3: None,          # None 表示该 level 全部
    # 4: ...          # 不传入表示不评测该 level
}
```

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `benchmark_name` | str | `KernelBench` | 数据集名称，支持 KernelBench、NPUKernelBench |
| `benchmark_path` | str | 根据 benchmark_name 自动推导 | Benchmark 根目录，详见 datasets.md |
| `output_root` | str | `./benchmark_results` | 输出根目录 |
| `arch` | str | `ascend910b1` | 目标硬件架构 |
| `resume` | bool | `true` | 是否断点续跑 |
| `timeout_per_task` | int | 2400 | 单任务超时（秒）|
| `warmup` | int | 5 | 性能测试 warmup 次数 |
| `repeats` | int | 50 | 性能测试重复次数 |

### benchmark_name 到 benchmark_path 的自动映射

| benchmark_name | benchmark_path | levels | 说明 |
|----------------|----------------|--------|------|
| `KernelBench` | `{repo_root}/benchmark/KernelBench` | 1-4 | 标准 KernelBench，每级 100 个算子 |
| `NPUKernelBench` | `{repo_root}/benchmarks/NPUKernelBench` | 1 | NPU 专用数据集，当前 3 个算子（GELU、SwiGLU、Add） |

> **优先级**：`benchmark_path` > `benchmark_name` 自动推导
> 即如果显式指定了 `benchmark_path`，则忽略 `benchmark_name` 的自动推导。

## 工作流程

```
Phase 1: 初始化
  ├── 解析用户输入（自然语言 → 结构化参数）
  ├── 加载配置（默认值 ← 用户输入）
  ├── 解析 benchmark_name 和 benchmark_path
  │   ├── 如果指定了 benchmark_path，直接使用
  │   └── 否则根据 benchmark_name 自动推导 benchmark_path
  ├── 验证 agent_workspace 存在且有效
  ├── 从 agent 文件 frontmatter 解析 skills
  └── 恢复断点状态（resume=true 时）

Phase 2: 任务扫描
  ├── 根据 level_problems 确定要评测的 (level, problem) 组合
  ├── 遍历指定 levels 的目录
  ├── 根据 problem_ids 过滤
  ├── 解析每个 task 文件，提取元数据
  │   ├── KernelBench：解析 .py 文件中的 Model 类和 get_inputs()
  │   └── NPUKernelBench：解析 .py 文件 + .json 测试用例配置
  ├── 过滤已完成的任务
  └── 构建任务队列 [(level, problem_id, task_file)]

Phase 3: 串行评测
  └── 串行执行每个任务：
      ├── 调用用户指定的 Agent 生成代码
      ├── 正确性验证（使用 agent 指定的 kernel-verifier skill）
      ├── 性能测试（调用 benchmark.py）
      └── 保存结果

Phase 4: 报告生成
  ├── 汇总所有结果
  ├── 按 Level / 算子类型统计
  └── 生成 agent_report.md（⚠️ 始终生成）
```

## Agent Skills 解析

从 agent 文件的 frontmatter 中解析 skills：

```markdown
---
name: my-agent
skills:
  - kernel-verifier    # 用于验证
  - code-generator     # 用于代码生成
---
```

验证时将使用 `kernel-verifier` skill 中的 `scripts/verify.py` 和 `scripts/benchmark.py`。

## 输出目录结构

```
{output_root}/
└── {timestamp}_{run_id}/
    └── agent_{name}/
        ├── level_{n}/              # 只包含用户指定的 levels
        │   ├── {problem_id}_{op_name}/
        │   │   ├── generated_code.py      # 生成的代码
        │   │   ├── verify_result.json     # 验证结果
        │   │   └── perf_result.json       # 性能结果
        │   └── ...
        └── agent_report.md          # ⚠️ 始终生成评测报告
```

## 报告内容

### agent_report.md 包含

1. **执行摘要** - 时间、硬件、评测范围
2. **总体统计** - 表格形式展示各 Level 和总体的任务数、编译成功率、正确率、优于PyTorch比例、平均加速比
3. **按算子类型统计** - 分类统计
4. **编译失败列表** - 按 Level 组织的编译失败详情
5. **数值验证失败列表** - 按 Level 组织的数值错误详情（含 Max Diff）
6. **性能劣化列表** - 按 Level 组织的性能低于 PyTorch 的算子（含劣化倍数）
7. **详细结果表** - 每个 problem 的完整结果

## 使用示例

### 示例 1: 基础使用

```
使用 agent "my-agent" 评测 Level 1 的 problems 1-10
参数：
- agent_name: my-agent
- agent_workspace: /path/to/.opencode
- level_problems: {1: [1,2,3,4,5,6,7,8,9,10]}
```

### 示例 2: 多 Level 部分选择

```
评测 Level 1 的 [1,2] 和 Level 2 的 [1,3]，Level 3 全选
参数：
- agent_name: my-agent
- agent_workspace: /path/to/.opencode
- level_problems: {1: [1,2], 2: [1,3], 3: null}
- arch: ascend910b2
```

### 示例 3: 指定范围

```
评测 Level 1 的 1-20 号问题
参数：
- agent_name: my-agent
- agent_workspace: /path/to/.opencode
- level_problems: {1: "1-20"}
```

### 示例 4: 使用 NPUKernelBench 数据集

```
评测 NPUKernelBench 的所有算子（当前仅 Level 1）
参数：
- agent_name: my-agent
- agent_workspace: /path/to/.opencode
- level_problems: {1: null}
- benchmark_name: NPUKernelBench
```

### 示例 5: 显式指定 benchmark_path

```
使用自定义路径评测
参数：
- agent_name: my-agent
- agent_workspace: /path/to/.opencode
- level_problems: {1: [1, 2, 3]}
- benchmark_path: /custom/path/to/my-benchmark
```

## Agent 要求

**实现方式**：此 skill **直接调用 kernelgen-workflow subagent**，无需通过 AKG-triton primary agent。

### 为什么直接调用 kernelgen-workflow？

1. **非阻塞执行**：AKG-triton 是交互式 agent，有多个强制确认点（question 工具），无法自动化执行
2. **效率更高**：跳过 AKG-triton 的编排层，直接执行代码生成和验证
3. **结果一致**：kernelgen-workflow 内部已经包含完整的生成-验证-测试流程

### Agent 配置

需要指定 agent_workspace（包含 kernelgen-workflow 的工作目录）：

```markdown
---
name: kernelgen-workflow
mode: subagent
skills:
  - code-generator
  - kernel-verifier
---
```

### 调用方式

benchmark-evaluator 会直接调用：
```bash
opencode run --agent kernelgen-workflow "生成并验证算子代码..."
```

### 工作流程

```
benchmark-evaluator
    ↓
直接调用 kernelgen-workflow subagent
    ↓
内部执行：
    ├── 代码生成（code-generator skill）
    ├── 正确性验证（kernel-verifier skill）
    ├── 性能测试（benchmark.py）
    └── 输出结果
    ↓
benchmark-evaluator 收集结果并生成报告
```

## 支持的数据集

### 数据集映射表

| benchmark_name | benchmark_path | levels | 说明 |
|----------------|----------------|--------|------|
| `KernelBench` | `{repo_root}/benchmark/KernelBench` | 1-4 | 标准 KernelBench，每级 100 个算子 |
| `NPUKernelBench` | `{repo_root}/benchmarks/NPUKernelBench` | 1 | NPU 专用数据集，当前 3 个算子（GELU、SwiGLU、Add） |

> **优先级**：`benchmark_path` > `benchmark_name` 自动推导
> 即如果显式指定了 `benchmark_path`，则忽略 `benchmark_name` 的自动推导。

### 数据集格式说明

#### KernelBench

- **目录结构**：`level1/` ~ `level4/`
- **文件格式**：每个 problem 一个 `.py` 文件，包含 `Model` 类和 `get_inputs()` 函数
- **文件名**：`{problem_id}_{op_name}.py`

#### NPUKernelBench

- **目录结构**：`level1/`
- **文件格式**：每个 problem 两个文件
  - `.py` 文件：包含算子实现代码
  - `.json` 文件：包含测试用例配置（dtype、shape 等）
- **解析方式**：解析 `.py` 文件 + `.json` 测试用例配置
- **文件名**：`{problem_id}_{op_name}.py` 和 `{problem_id}_{op_name}.json`


## 依赖

- Python 3.8+
- opencode Agent 调用机制
- 支持的数据集（KernelBench 或 NPUKernelBench）
- NPU 设备（用于验证和性能测试）

## 注意事项

1. **串行执行**: 任务按顺序逐个执行，不进行并行化
2. **始终生成报告**: 无论评测成功与否，都会生成 `agent_report.md`
3. **断点续跑**: 基于 `(level, problem_id)` 去重，支持中断后恢复
4. **Agent Skills**: 必须包含 `kernel-verifier` skill 用于验证
5. **超时处理**: 单任务超时不影响整体流程，记录为失败
6. **错误隔离**: 单任务失败会记录但继续执行其他任务
7. **数据集兼容性**: 不同数据集的文件解析逻辑可能不同，请确保使用对应的格式
8. **benchmark_path 优先级**: 显式指定的 `benchmark_path` 会覆盖 `benchmark_name` 的自动推导
