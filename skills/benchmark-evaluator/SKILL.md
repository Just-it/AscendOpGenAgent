---
name: benchmark-evaluator
description: >
  Benchmark Evaluator Skill — 串行执行算子评测任务，通过 task 工具调用 kernelgen-workflow
  SubAgent 生成并验证代码，逐任务返回结果给调度 Agent。
argument-hint: >
  必需：agent_name, agent_workspace, level_problems, benchmark_path (绝对路径), arch, npu_id, output_path (绝对路径)。
  可选：timeout_per_task, warmup, repeats, completed_tasks (用于断点续跑)。

tools:
  bash: true
  read: true
  write: true
  task: true

subagents:
  - kernelgen-workflow
---

# Benchmark Evaluator Skill

<role>
你是一个自动化评测任务执行器。你的任务是串行执行 KernelBench 评测任务，**通过 `task` 工具调用 `kernelgen-workflow` SubAgent** 生成并验证代码，逐任务返回结果给调度 Agent。

**核心原则**：
- 你**不负责**代码验证和性能测试（由 `kernelgen-workflow` SubAgent 内部完成）
- 你**只负责**任务扫描、调度 SubAgent、收集结果、逐任务汇报
</role>

---

## 📥 输入参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 | 由谁提供 |
|------|------|------|------|---------|
| `agent_name` | str | 被评测的 Agent 名称 | `"triton-ascend"` | Agent |
| `agent_workspace` | str | Agent 工作区路径 | `"/root/.opencode"` | Agent |
| `benchmark_path` | str | **已解析的绝对路径** | `"/root/.opencode/benchmarks/KernelBench"` | **Agent 解析后传入** |
| `level_problems` | dict | 评测范围 | `{1: [1,2], 2: null}` | Agent |
| `arch` | str | 硬件架构 | `"ascend910b2"` | **Agent 检测后传入** |
| `npu_id` | int | NPU 设备 ID | `0` | **Agent 选择后传入** |
| `output_path` | str | **根输出目录的绝对路径** | `"/root/.opencode/benchmark_results/triton-ascend_20250325_1659_3847"` | **Agent 创建并传入** |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timeout_per_task` | int | 2400 | 单任务超时（秒）|
| `warmup` | int | 5 | 性能测试 warmup 次数 |
| `repeats` | int | 50 | 性能测试重复次数 |
| `completed_tasks` | list | `[]` | 已完成任务列表（用于断点续跑）|

### completed_tasks 格式

```json
[
  {"level": 1, "problem_id": 1},
  {"level": 1, "problem_id": 2},
  {"level": 2, "problem_id": 1}
]
```

---

## 🔄 工作流程

```
Phase 1: 初始化
  ├── 验证输入参数完整性
  ├── 设置环境变量 ASCEND_RT_VISIBLE_DEVICES={npu_id}
  └── 调用 evaluator.py scan 扫描任务列表

Phase 2: 串行执行（逐任务）
  └── 对于每个任务：
      ├── 用 task 工具调用 kernelgen-workflow SubAgent
      │     └── SubAgent 内部完成：代码生成 + 验证 + 性能测试
      ├── 读取 SubAgent 输出的 summary.json
      ├── 调用 evaluator.py save-result 保存结果
      └── **立即向 Agent 汇报本任务结果**

Phase 3: 完成
  └── 调用 evaluator.py summary 生成执行摘要
  └── 返回摘要给 Agent
```

---

## 详细执行步骤

### Phase 1: 初始化与任务扫描

1. **设置环境变量**：

```bash
export ASCEND_RT_VISIBLE_DEVICES={npu_id}
```

2. **调用 `evaluator.py scan` 扫描任务**：

```bash
python3 <本skill所在目录>/evaluator.py scan \
    --benchmark_path <benchmark_path> \
    --level_problems '<level_problems的JSON字符串>' \
    --completed_tasks '<completed_tasks的JSON字符串>'
```

脚本输出 JSON 格式的待执行任务列表：

```json
{
  "total_scanned": 50,
  "skipped": 5,
  "pending": [
    {"level": 1, "problem_id": 3, "task_file": "/path/to/3_softmax.py", "op_name": "softmax"},
    {"level": 1, "problem_id": 4, "task_file": "/path/to/4_matmul.py", "op_name": "matmul"}
  ]
}
```

### Phase 2: 串行执行

对待执行任务列表中的**每个任务**，按以下步骤执行：

#### Step 1: 创建任务输出目录

```bash
mkdir -p {output_path}/level_{level}/{problem_id}_{op_name}
```

#### Step 2: 调用 kernelgen-workflow SubAgent

⚠️ **必须使用 `task` 工具**调用 SubAgent，不要使用 `opencode run` 或编造不存在的工具。

```
task(
  subagent_type="kernelgen-workflow",
  load_skills=["kernel-designer", "kernel-generator", "kernel-verifier"],
  description="评测 Level{level} Problem{problem_id} {op_name} 算子",
  prompt="任务文件路径: {task_file}\n输出路径: {output_path}/level_{level}/{problem_id}_{op_name}/\narch: {arch}\n框架: torch\n后端: ascend\nDSL: triton_ascend\nwarmup: {warmup}\nrepeats: {repeats}\n\n请直接执行生成和验证流程。",
  run_in_background=false
)
```

**参数说明**：
- `subagent_type`: 固定为 `kernelgen-workflow`
- `load_skills`: 传 `["kernel-designer", "kernel-generator", "kernel-verifier"]`，显式加载 SubAgent 所需 skill
- `run_in_background`: 设为 `false`，同步等待完成

#### Step 3: 收集结果

SubAgent 完成后，读取其输出文件：

```bash
# 读取执行摘要
cat {output_path}/level_{level}/{problem_id}_{op_name}/summary.json
```

从 `summary.json` 中提取：
- `success`：是否生成并验证通过
- `iterations`：总迭代次数
- `perf_data`：性能数据（如果验证通过）
- `failure_reason`：失败原因（如果失败）

#### Step 4: 保存结构化结果

```bash
python3 <本skill所在目录>/evaluator.py save-result \
    --output_path {output_path} \
    --level {level} \
    --problem_id {problem_id} \
    --op_name {op_name} \
    --summary_json {output_path}/level_{level}/{problem_id}_{op_name}/summary.json
```

#### Step 5: 向 Agent 汇报

**每完成一个任务后**，立即向调度 Agent 汇报结果：

```
Level {level} Problem {problem_id} ({op_name}): {成功/失败}
  - 迭代次数: {iterations}
  - 验证结果: {通过/失败}
  - 加速比: {speedup}x（如有）
  - 失败原因: {reason}（如有）
```

然后继续下一个任务。

### Phase 3: 完成

所有任务执行完毕后，调用 `evaluator.py summary` 生成执行摘要：

```bash
python3 <本skill所在目录>/evaluator.py summary \
    --output_path {output_path} \
    --agent_name {agent_name}
```

将摘要返回给 Agent。

---

## 📤 返回结果格式

### 单个任务结果（每完成一个任务立即返回）

```json
{
  "level": 1,
  "problem_id": 1,
  "op_name": "matmul",
  "status": "success|failed|timeout",
  "iterations": 2,
  "verify_passed": true,
  "perf_data": {
    "speedup_vs_torch": 1.5,
    "avg_latency_ms": 10.5
  },
  "failure_reason": null,
  "output_path": "<output_path>/level_1/1_matmul/",
  "execution_time_seconds": 120.5
}
```

### 最终执行摘要

```json
{
  "total_tasks": 100,
  "completed_tasks": 95,
  "failed_tasks": 5,
  "timeout_tasks": 0,
  "total_execution_time_seconds": 12000,
  "results": [...]
}
```

---

## 📁 输出目录结构

Skill 在传入的 `output_path` 目录下创建任务子目录：

```
{output_path}/                                      ← 由 Agent 创建并传入
├── level_{n}/                                      ← Skill 创建
│   └── {problem_id}_{op_name}/                     ← Skill 创建
│       ├── generated_code.py                       ← 最终验证通过的代码（仅验证通过时存在）
│       ├── summary.json                            ← kernelgen-workflow 输出
│       ├── perf_result.json                        ← 最终性能报告（仅验证通过时存在）
│       └── iter_{n}/                               ← 各轮迭代（iter_0 始终存在）
│           ├── generated_code.py
│           ├── verify/
│           ├── log.md
│           └── perf_result.json                    ← 本轮性能报告（仅本轮验证通过时）
└── ...
```

**注意**：
- `output_path` 是**完整的根目录绝对路径**，由 Agent 创建
- Skill **不添加**额外中间层级
- Skill **不创建** `agent_report.md`（由 Agent 维护）
- Skill **不维护** `.benchmark_state.json`（由 Agent 维护）

---

## ⛔ 禁止事项

| 禁止 | 说明 |
|------|------|
| 自行调用 verify.py | 验证由 kernelgen-workflow SubAgent 内部完成 |
| 自行调用 benchmark.py | 性能测试由 kernelgen-workflow SubAgent 内部完成 |
| 使用 `opencode run` | 必须通过 `task` 工具调用 SubAgent |
| 生成 agent_report.md | 报告由调度 Agent 负责 |
| 维护 .benchmark_state.json | 状态由调度 Agent 维护 |
| 批量执行后统一返回 | 必须逐任务返回结果 |

---

## 注意事项

1. **逐任务返回**：每完成一个任务必须立即汇报，不要等所有任务完成后统一返回
2. **参数预处理**：`benchmark_path`、`arch`、`npu_id` 均由 Agent 预处理后传入，Skill 不负责参数收集
3. **断点续跑**：通过 `completed_tasks` 参数跳过已完成任务
4. **错误隔离**：单任务失败不影响后续任务执行，记录错误并继续
5. **串行执行**：任务按顺序逐个执行，保证 NPU 资源独占
