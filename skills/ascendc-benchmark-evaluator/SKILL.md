---
name: ascendc-benchmark-evaluator
description: >
  AscendC 算子评测 Skill — 串行执行 AscendC 算子评测任务，调用 lingxi-code subagent 生成代码并验证。
  接收已解析的参数，返回每个任务的结构化结果。
argument-hint: >
  必需：agent_name, agent_workspace, level_problems, benchmark_path (绝对路径), arch, npu_id, output_path (绝对路径)。
  可选：timeout_per_task, warmup, repeats, completed_tasks (用于断点续跑)。
---

# AscendC Op Evaluator Skill

<role>
你是一个自动化评测任务执行器。你的任务是串行执行 AscendC 算子评测任务，调用 lingxi-code subagent 生成代码，编译安装，验证正确性，测试性能，并返回每个任务的结构化结果。
</role>

---

## 📥 输入参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 | 由谁提供 |
|------|------|------|------|---------|
| `agent_name` | str | 被评测的 Agent 名称 | `"lingxi-code"` | Agent |
| `agent_workspace` | str | Agent 工作区路径 | `"/root/.opencode"` | Agent |
| `benchmark_path` | str | **已解析的绝对路径** | `"/root/.opencode/benchmarks/KernelBench"` | **Agent 解析后传入** |
| `level_problems` | dict | 评测范围 | `{1: [1,2], 2: null}` | Agent |
| `arch` | str | 硬件架构 | `"ascend910b1"` | **Agent 检测后传入** |
| `npu_id` | int | NPU 设备 ID | `0` | **Agent 选择后传入** |
| `output_path` | str | **根输出目录的绝对路径** | `"/root/.opencode/benchmark_results/lingxi-code_20250324_103000_1234"` | **Agent 创建并传入** |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timeout_per_task` | int | 3600 | 单任务超时（秒），AscendC 编译时间较长 |
| `warmup` | int | 5 | 性能测试 warmup 次数 |
| `repeats` | int | 50 | 性能测试重复次数 |
| `completed_tasks` | list | `[]` | 已完成任务列表（用于断点续跑）|
| `model_name` | str | `""` | subagent 模型名称 |
| `log_level` | str | `"INFO"` | subagent 日志级别 |

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
  ├── 验证 benchmark_path 存在且有效
  ├── 设置环境变量 ASCEND_RT_VISIBLE_DEVICES={npu_id}
  └── 创建输出目录结构

Phase 2: 任务扫描
  ├── 根据 level_problems 扫描 benchmark_path
  ├── 构建任务列表 [(level, problem_id, task_file, op_name)]
  ├── 根据 completed_tasks 过滤已完成任务
  └── 确定待执行任务队列

Phase 3: 串行执行
  └── 对于每个任务：
      ├── 调用 lingxi-code subagent 生成 AscendC 代码
      ├── 调用 scripts/evaluate.py 执行正确性验证和性能测试（不采纳subagent的测试结果，必须根据该 skill 的脚本测试）
      ├── 保存结果到 test_report.json
      └── **返回任务结果给 Agent**

Phase 4: 完成
  └── 返回执行摘要
```

---

## 📤 返回结果格式

### 单个任务结果

```json
{
  "level": 1,
  "problem_id": 1,
  "op_name": "gelu",
  "status": "success|failed|timeout",
  "error_type": null|"compilation"|"verification"|"performance"|"codegen",
  "error_message": "具体错误信息",
  "error_stage": null|"codegen"|"install_run"|"pybind"|"evaluation",
  "test_report": {
    "op_name": "gelu",
    "total_cases": 10,
    "passed": 8,
    "failed": 2,
    "pass_rate": 0.8,
    "avg_speedup": 2.5,
    "details": [
      {
        "case_id": 1,
        "shape_info": "shapes=[[128]], dtypes=['float32'], attrs={}",
        "correctness": true,
        "error_message": null,
        "ref_time_ms": 0.112,
        "custom_time_ms": 0.045,
        "speedup": 2.49,
        "ref_avg_ms": 0.115,
        "ref_p50_ms": 0.112,
        "ref_p99_ms": 0.130,
        "custom_avg_ms": 0.048,
        "custom_p50_ms": 0.045,
        "custom_p99_ms": 0.055,
        "peak_memory_mb": 1.2
      }
    ]
  },
  "output_path": "<output_path>/level_1/1_gelu/",
  "execution_time_seconds": 180.5
}
```

### 最终执行摘要

```json
{
  "total_tasks": 100,
  "completed_tasks": 95,
  "failed_tasks": 5,
  "timeout_tasks": 0,
  "total_execution_time_seconds": 18000,
  "results": [
    {/* 单个任务结果 */},
    {/* 单个任务结果 */},
    ...
  ]
}
```

---

## 🎯 核心职责

### 1. 任务扫描

- 根据 `level_problems` 扫描 `benchmark_path` 目录
- 解析每个任务文件的元数据（从文件名提取 op_name）
- 根据 `completed_tasks` 过滤已完成任务
- 构建待执行任务队列

### 2. 代码生成

**命令示例**：

命令中，必须包含OPENCODE_PERMISSION的设置！
```bash
source ~/.bashrc && conda activate py39 && \
export ASCEND_RT_VISIBLE_DEVICES=7 && \
export OPENCODE_PERMISSION='"allow"' && \  # ⚠ 单双引号均不可缺少！
opencode run --model {model_name} --agent {agent_name} --log-level DEBUG "{prompt}"
```
其中`--log-level DEBUG`在用户不指定时不必设置。

**prompt设置**：

你需要根据与任务文件同名的json文件（只随机挑其中一行进行描述！）描述算子的输入填入`{input_desc}`。
```
生成{op_name}算子，它的输入需要满足：{input_desc}

**模式**: benchmark 模式

任务文件路径: {benchmark_path}/level{n}/{problem_id}_{op_name}.py
目标架构: {arch}
输出目录: {output_path}/level_{n}/{problem_id}_{op_name}/{op_name}/

请直接执行完整流程，无需用户确认。
```

### 3. 正确性验证与性能测试

调用 `evaluate.py` 脚本一次性完成正确性验证和性能测试：

```bash
python3 scripts/evaluate.py \
    --op_name {op_name} \
    --benchmark_file {benchmark_path}/level_{n}/{problem_id}_{op_name}.py \
    --benchmark_json {benchmark_path}/level_{n}/{problem_id}_{op_name}.json \
    --output_dir {output_path}/level_{n}/{problem_id}_{op_name}/{op_name}/ \
    --warmup {warmup} \
    --repeats {repeats} \
    &> {output_path}/level_{n}/{problem_id}_{op_name}/{op_name}/evaluate.log
```
输出可能会很长，需要重定向后对文件分析。

**evaluate.py 功能**：
- 自动安装 run 文件
- 自动生成 PyBind 绑定
- 遍历所有测试用例（从 .json 文件加载）
- 对每个测试用例执行正确性验证
- 对通过正确性验证的用例执行性能测试
- 生成完整的测试报告（test_report.json）

### 4. 结果返回

- **每完成一个任务，立即返回结果给 Agent**
- 不生成报告（报告由 Agent 负责）
- 不维护状态文件（状态由 Agent 维护）

---

## 📁 输出目录结构

Skill 在传入的 `output_path` 目录下创建任务子目录：

```
{output_path}/                                      ← 由 Agent 创建并传入
├── level_{n}/                                      ← Skill 创建
│   └── {problem_id}_{op_name}/                     ← Skill 创建
│       ├── {op_name}/                              ← lingxi-code subagent 创建
│       │   └── ...                                 ← SubAgent 生成的所有文件
│       ├── verify_result.json                      ← Skill 保存（汇总）
│       └── perf_result.json                        ← Skill 保存（汇总）
└── ...
```

**lingxi-code subagent 生成的文件**（在 `{op_name}/` 目录下）：
- `{op_name}_functional.py` - functional_conversion 输出
- `{op_name}_ascend_call.py` - ascend_call_generation 输出
- `{op_name}_dsl.py` - dsl_baseline_generation 输出
- `{op_name}.cpp` - dsl_lowering 输出
- `{op_name}Custom/` - 编译产物目录
- `vendors/` - 安装后的算子库
- `{op_name}_custom.py` - 自定义算子调用代码
- `test_report.json` - evaluate.py 输出

**注意**：
- `output_path` 是**完整的根目录绝对路径**，由 Agent 创建
- Skill **不添加** `run_{timestamp}/agent_{agent_name}/` 等中间层级
- Skill **不创建** `agent_report.md`（由 Agent 维护）
- Skill **不维护** `.benchmark_state.json`（由 Agent 维护）

---

## 💡 使用示例

### 示例 1: 基础调用

```python
{
  "agent_name": "lingxi-code",
  "agent_workspace": "/root/.opencode",
  "benchmark_path": "/root/.opencode/benchmarks/KernelBench",  # 已解析的绝对路径
  "output_path": "/root/.opencode/benchmark_results/lingxi-code_20250324_103000_1234",  # Agent 创建的根目录绝对路径
  "level_problems": {1: [1, 2, 3]},
  "arch": "ascend910b1",  # Agent 检测后传入
  "npu_id": 0  # Agent 选择后传入
}
```

### 示例 2: 断点续跑

```python
{
  "agent_name": "lingxi-code",
  "agent_workspace": "/root/.opencode",
  "benchmark_path": "/root/.opencode/benchmarks/KernelBench",
  "output_path": "/root/.opencode/benchmark_results/lingxi-code_20250324_103000_1234",  # Agent 创建的根目录
  "level_problems": {1: [1, 2, 3, 4, 5]},
  "arch": "ascend910b1",
  "npu_id": 0,
  "completed_tasks": [  # Agent 从状态文件加载后传入
    {"level": 1, "problem_id": 1},
    {"level": 1, "problem_id": 2}
  ]
}
```

---

## 📋 Benchmark 文件格式

### `.py` 文件 - PyTorch Model 定义

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, approximate='none') -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=approximate)
```

### `.json` 文件 - 测试输入配置（每行一个 JSON 对象）

```json
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float32", "shape": [128]}, {"name": "approximate", "type": "attr", "required": false, "dtype": "str", "value": "none"}]}
{"inputs": [{"name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [256, 256]}, {"name": "approximate", "type": "attr", "required": false, "dtype": "str", "value": "tanh"}]}
```

### 输入类型说明

| Type | Description | Generated Value |
|------|-------------|-----------------|
| `tensor` | 输入张量 | `torch.randn(shape, dtype=dtype)` |
| `attr` | 属性参数 | 直接使用 JSON 中的 `value` 值 |

---

## 🔧 脚本说明

### evaluate.py

一次性完成正确性验证和性能测试：

```bash
python3 scripts/evaluate.py \
    --op_name <op_name> \
    --benchmark_file <benchmark .py 文件路径> \
    --benchmark_json <benchmark .json 文件路径> \
    --output_dir <输出目录> \
    --warmup 5 \
    --repeats 50 \
    --skip_setup  # 可选，跳过环境设置
```

**输出文件**：`{output_dir}/test_report.json`

**test_report.json 格式**：
```json
{
  "op_name": "gelu",
  "total_cases": 10,
  "passed": 8,
  "failed": 2,
  "pass_rate": 0.8,
  "avg_speedup": 2.5,
  "details": [
    {
      "case_id": 1,
      "shape_info": "shapes=[[128]], dtypes=['float32'], attrs={}",
      "correctness": true,
      "error_message": null,
      "ref_time_ms": 0.112,
      "custom_time_ms": 0.045,
      "speedup": 2.49,
      "ref_avg_ms": 0.115,
      "ref_p50_ms": 0.112,
      "ref_p99_ms": 0.130,
      "custom_avg_ms": 0.048,
      "custom_p50_ms": 0.045,
      "custom_p99_ms": 0.055,
      "peak_memory_mb": 1.2
    }
  ]
}
```

### generate_pybind.py

单独生成 PyBind 绑定（通常由 evaluate.py 自动调用）：

```bash
python3 scripts/generate_pybind.py <op_name> --output_dir <目录>
```

---

## 📊 性能指标

| 指标 | 说明 |
|------|------|
| `ref_time_ms` | 参考实现（Model）的中位数延迟 |
| `custom_time_ms` | 自定义实现（ModelNew）的中位数延迟 |
| `speedup` | 加速比 = ref_time / custom_time |
| `avg_latency_ms` | 平均延迟 |
| `p50_latency_ms` | P50 延迟 |
| `p99_latency_ms` | P99 延迟 |
| `peak_memory_mb` | 峰值内存占用（MB） |

---

## ⚠️ 注意事项

1. **参数预处理**：
   - `benchmark_path` 必须是**已解析的绝对路径**
   - `arch` 和 `npu_id` 由 Agent 检测/选择后传入
   - Skill 不负责参数收集和解析

2. **断点续跑**：
   - Skill 根据 `completed_tasks` 跳过已完成任务
   - 状态文件由 Agent 维护，Skill 不操作

3. **结果返回**：
   - 每完成一个任务，立即返回结果给 Agent
   - 不生成报告，不维护状态文件

4. **错误处理**：
   - 单任务失败不影响整体流程
   - 记录错误信息和错误阶段（error_stage）
   - 超时任务标记为 `timeout` 状态

5. **串行执行**：
   - 任务按顺序逐个执行，不进行并行化
   - 保证资源独占，避免 NPU 冲突

6. **环境隔离**：
   - 设置 `ASCEND_RT_VISIBLE_DEVICES={npu_id}` 确保 NPU 独占
   - 每个任务独立的输出目录

7. **超时设置**：
   - AscendC 编译时间较长，建议 `timeout_per_task` 设置为 3600 秒（1小时）
   - 包含代码生成、编译、安装、验证、性能测试全流程

---

## 依赖

- Python 3.8+
- lingxi-code subagent
- torch, torch_npu
- NPU 设备（用于验证和性能测试）
- msopgen 工具（用于创建 AscendC 项目）
- CANN 环境
