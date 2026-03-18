---
name: kernelbench-evaluator
description: >
  KernelBench 自动评测 Skill — 批量评估已有 Agent 在 KernelBench 数据集上的代码生成能力。
  支持单 Agent 评测和多 Agent 对比，自动生成详细评测报告。
argument-hint: >
  输入：agent_workspaces（Agent 工作区路径列表）、levels（评测级别）。
  可选：benchmark_path、output_root、problem_ids、arch、max_workers、resume 等。
  输出：每个 Agent 的详细报告 + 多 Agent 对比报告（多 Agent 时）。
---

# KernelBench Evaluator Skill

<role>
你是一个自动化评测框架专家。你的任务是批量执行 KernelBench 评测，调用已有 Agent 生成代码，验证正确性，测试性能，并生成详细报告。
</role>

## 输入参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `agent_workspaces` | list[str] | Agent 工作区路径列表（包含 agents/ 和 skills/ 的目录） | `["/path/to/agent/.opencode"]` |
| `levels` | list[int] | 评测级别 [1,2,3,4] | `[1, 2]` |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `benchmark_path` | str | `/mnt/w00934874/agent/benchmark/KernelBench` | Benchmark 根目录 |
| `output_root` | str | `./benchmark_results` | 输出根目录 |
| `problem_ids` | list[int] \| str \| null | null | 指定问题编号，null=全部，支持 `[1,2,3]` 或 `"1-10"` |
| `arch` | str | `ascend910b1` | 目标硬件架构 |
| `max_workers` | int | 4 | 并行进程数（自动根据 NPU 数量调整） |
| `resume` | bool | true | 是否断点续跑 |
| `timeout_per_task` | int | 600 | 单任务超时（秒） |
| `warmup` | int | 5 | 性能测试 warmup 次数 |
| `repeats` | int | 50 | 性能测试重复次数 |
| `save_generated_code` | bool | true | 是否保存生成的代码 |
| `save_failed_cases` | bool | true | 是否保存失败用例 |

## 工作流程

```
Phase 1: 初始化
  ├── 解析用户输入（自然语言 → 结构化参数）
  ├── 加载配置（默认值 ← 用户输入）
  ├── 验证 agent_workspaces 存在且有效
  ├── 自动扫描每个 workspace 的 agents/ 和 skills/
  ├── 检测可用 NPU 数量，自适应调整 max_workers
  └── 恢复断点状态（resume=true 时）

Phase 2: 任务扫描
  ├── 遍历指定 levels 的目录
  ├── 根据 problem_ids 过滤
  ├── 解析每个 task 文件，提取元数据
  ├── 过滤已完成的任务
  └── 构建任务队列 [(agent, level, problem_id, task_file)]

Phase 3: 批量评测
  └── 并行执行（ProcessPoolExecutor）
      └── 单任务流程：
          ├── 调用 Agent 生成代码
          ├── 正确性验证（调用 Verifier Skill）
          ├── 性能测试（调用 benchmark.py）
          └── 保存结果

Phase 4: 报告生成
  ├── 汇总每个 Agent 的所有结果
  ├── 按 Level / 算子类型统计
  ├── 生成单个 Agent 报告
  ├── 多 Agent 时生成对比报告
  └── 输出 Markdown + JSON
```

## 输出目录结构

```
{output_root}/
└── {timestamp}_{run_id}/
    ├── agent_{name}/
    │   ├── level_1/              # level_{n} 格式
    │   │   ├── 1_matmul/        # {problem_id}_{op_name}
    │   │   │   ├── generated_code.py
    │   │   │   ├── verify_result.json
    │   │   │   └── perf_result.json
    │   │   └── ...
    │   ├── level_2/
    │   └── agent_report.md       # 单个 Agent 完整报告
    ├── agent_{name_2}/
    │   └── ...（同上）
    └── comparison_report.md      # 对比报告（多 Agent 时）
```

## 报告内容

### 单个 Agent 报告包含

1. **执行摘要** - 时间、硬件、并行度
2. **总体统计** - 表格形式展示各 Level 和总体的任务数、编译成功率、正确率、优于PyTorch比例、平均加速比
3. **按算子类型统计** - 分类统计
4. **编译失败列表** - 按 Level 组织的编译失败详情
5. **数值验证失败列表** - 按 Level 组织的数值错误详情（含 Max Diff）
6. **性能劣化列表** - 按 Level 组织的性能低于 PyTorch 的算子（含劣化倍数）
7. **详细结果表** - 每个 problem 的完整结果

### 对比报告包含（多 Agent 时）

1. **对比概览表** - 各 Agent 关键指标对比
2. **按 Level 对比** - 各 Level 胜出情况
3. **逐个 Problem 对比** - 每个问题的详细对比
4. **统计汇总** - 各 Agent 胜出次数
5. **差异分析** - 各 Agent 的优势场景

## 使用示例

```
# 单 Agent 评测
"使用 /mnt/w00934874/agent/code/AscendOpGenAgent/akg-triton/.opencode 评测 Level 1-2"

# 多 Agent 对比
"对比 /path/to/agent1/.opencode 和 /path/to/agent2/.opencode 的 Level 1"

# 带详细参数
"使用 /path/to/agent/.opencode 评测 Level 1 的问题 1-20，arch 用 ascend910b2，8 进程并行"
```

## 注意事项

1. **Agent 工作区路径** 必须包含 `agents/` 和 `skills/` 子目录
2. **NPU 资源** 会自动检测并调整并行度
3. **断点续跑** 基于 `(agent_name, level, problem_id)` 去重
4. **存储空间** 建议预留 10GB+（保存生成的代码和结果）
5. **超时处理** 单任务超时不影响整体流程
6. **错误隔离** 单任务失败会记录但继续执行其他任务

## 依赖

- Python 3.8+
- opencode Agent 调用机制
- KernelBench 数据集
- NPU 设备（用于验证和性能测试）
