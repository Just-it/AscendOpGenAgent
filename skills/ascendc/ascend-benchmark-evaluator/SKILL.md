---
name: ascend-benchmark-evaluator
description: >
  Ascend Benchmark Evaluation Skill — 专门用于测评 lingxi-code agent 生成算子的能力，包括精度和性能。
  支持遍历数据集、调用 lingxi-code agent 生成算子、测评精度和性能，并生成详细报告。
argument-hint: >
  必需：benchmark_path、agent_workspace、level_problems。
  可选：output_dir、arch、npu_id、resume、timeout_per_task、warmup、repeats。
  level_problems 格式：{1: [1,2], 2: [1,3], 3: None}，None 表示该 level 全选。
  benchmark_path 支持：1) 绝对路径；2) 相对路径（基于当前工作目录）；3) 数据集名称（如 "NPUKernelBench"）。
---

# Ascend Benchmark Evaluation Skill

<role>
你是一个自动化评测框架执行器，专门用于测评 lingxi-code agent 生成 AscendC 算子的能力。你的任务是遍历指定的算子数据集，调用 lingxi-code agent 生成算子代码，验证其正确性，测试性能，并生成详细的评测报告。
</role>

## 输入参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `agent_workspace` | str | Agent 工作区路径（包含 agents/ 和 skills/） | `"/path/to/AscendOpGenAgent"` |
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
| `benchmark_path` | str | 见下方详细说明 | Benchmark 路径，支持多种指定方式 |
| `output_dir` | str | `"benchmark_results"` | 输出目录名 |
| `arch` | str | 首次运行时询问 | 目标硬件架构（ascend910b2 等） |
| `npu_id` | int | 首次运行时询问 | 目标 NPU 设备序号（如 0, 1, 2...） |
| `resume` | bool | `true` | 是否断点续跑 |
| `timeout_per_task` | int | 2400 | 单任务超时（秒）|
| `warmup` | int | 5 | 性能测试 warmup 次数 |
| `repeats` | int | 50 | 性能测试重复次数 |

### benchmark_path 解析规则

**优先级（从高到低）：**

1. **绝对路径**：以 `/` 开头，直接使用
   - 示例：`/data/custom_benchmark`
   - 结果：`/data/custom_benchmark`

2. **相对路径**：包含 `/` 但不以 `/` 开头，基于 `agent_workspace` 解析
   - 示例：`custom/benchmarks/A`
   - 结果：`{agent_workspace}/custom/benchmarks/A`

3. **Benchmark 名称**：不包含 `/`，视为 `agent_workspace/benchmarks/{名称}`
   - 示例：`NPUKernelBench`
   - 结果：`{agent_workspace}/benchmarks/NPUKernelBench`

4. **未指定**：默认使用 `{agent_workspace}/benchmarks/NPUKernelBench`

**路径解析伪代码：**

```python
def resolve_benchmark_path(agent_workspace, benchmark_path=None):
    """
    解析 benchmark 路径
    
    规则：
    1. 绝对路径（以/开头）：直接使用
    2. 相对路径（包含/）：基于 agent_workspace 拼接
    3. 名称（不含/）：视为 agent_workspace/benchmarks/{名称}
    4. 未指定：默认使用 agent_workspace/benchmarks/NPUKernelBench
    """
    if benchmark_path is None:
        return f"{agent_workspace}/benchmarks/NPUKernelBench"
    
    if benchmark_path.startswith('/'):
        # 绝对路径
        return benchmark_path
    elif '/' in benchmark_path:
        # 相对路径，基于 agent_workspace
        return f"{agent_workspace}/{benchmark_path}"
    else:
        # 纯名称，视为 benchmarks/{名称}
        return f"{agent_workspace}/benchmarks/{benchmark_path}"
```

## 工作流程

```
Phase 1: 初始化
  ├── 解析用户输入（自然语言 → 结构化参数）
  ├── 加载配置（默认值 ← 用户输入）
  ├── 验证 agent_workspace 存在且有效
  ├── 从 agent 文件 frontmatter 解析 skills
  ├── 检查/询问目标硬件架构（arch）
  │   └── 保存 arch 到状态文件供后续任务使用
  ├── 检查/询问目标 NPU 序号（npu_id）
  │   └── 保存 npu_id 到状态文件供后续任务使用
  │   └── 设置环境变量 ASCEND_RT_VISIBLE_DEVICES={npu_id}
  ├── 创建 tmp/ 目录用于存放中间文件
  └── 恢复断点状态（resume=true 时）

Phase 2: 任务扫描
  ├── 根据 level_problems 确定要评测的 (level, problem) 组合
  ├── 遍历指定 levels 的目录
  ├── 根据 problem_ids 过滤
  ├── 解析每个 task 文件，提取元数据
  ├── 过滤已完成的任务
  └── 构建任务队列 [(level, problem_id, task_file)]

Phase 3: 串行评测
  └── 串行执行每个任务：
      ├── 调用lingxi-code 生成代码、验证正确性、测试性能（显示详细步骤，由于生成时间较长，期间不能跳过生成算子往下走）
      ├── 保存结果
      └── 增量更新 agent_report.md（追加本次任务结果）

Phase 4: 报告完成
  └── 所有任务完成后，生成最终汇总报告
```

## NPU 设备选择

**npu_id 参数处理逻辑：**

1. **首次运行**：如果未指定 `npu_id`，使用 `question` 工具询问用户选择目标 NPU 设备序号
2. **保存选择**：将用户选择的 npu_id 保存到状态文件（`{output_dir}/.benchmark_state.json`）
3. **后续运行**：从状态文件读取已保存的 npu_id，不再询问
4. **显式指定**：用户可以通过参数直接指定 npu_id，跳过询问

**如何查看可用 NPU：**

```bash
# 查看可用的 NPU 设备
npu-smi info
```

**NPU 序号说明：**
- `0` - 第 0 号 NPU 设备
- `1` - 第 1 号 NPU 设备
- 以此类推...

**环境变量设置：**

选定 NPU 后，在执行任务前设置环境变量：
```bash
export ASCEND_RT_VISIBLE_DEVICES={npu_id}
```



## 输出目录结构

```
output/
├── {op_name}/
│   ├── {op_name}_functional.py   # Functional PyTorch 代码
│   ├── {op_name}_dsl.py          # DSL 代码
│   ├── {op_name}Custom/          # AscendC 项目目录
│   │   └── vendors/customize/
│   ├── {op_name}_custom.py       # 自定义算子代码
│   └── {op_name}_reference.py    # 参考代码
└── agent_report.md               # ⚠️ 始终生成评测报告
```

## 报告内容

### agent_report.md 结构

**增量更新机制：**
- 每个任务完成后立即更新报告，追加新结果
- 报告头部始终显示最新汇总统计
- 支持断点续跑时恢复和继续更新

**报告包含内容：**

1. **执行摘要** - 时间、硬件、评测范围
2. **总体统计** - 表格形式展示各 Level 和总体的任务数、编译成功率、正确率、优于PyTorch比例、平均加速比
3. **按算子类型统计** - 分类统计
4. **编译失败列表** - 按 Level 组织的编译失败详情
5. **数值验证失败列表** - 按 Level 组织的数值错误详情（含 Max Diff）
6. **性能劣化列表** - 按 Level 组织的性能低于 PyTorch 的算子（含劣化倍数）
7. **详细结果表** - 每个 problem 的完整结果

### 增量更新实现

```python
# 伪代码示例
def update_report_incrementally(task_result, report_path):
    """每次任务完成后增量更新报告"""
    # 1. 读取现有报告（如果存在）
    # 2. 更新统计数据
    # 3. 追加新任务结果到详细结果表
    # 4. 重新生成报告文件
    pass
```

## 使用示例

### 示例 1: 基础使用

```
测评 NPUKernelBench 数据集 Level 1 的 problems 1-10
参数：
- agent_name: lingxi-code
- agent_workspace: /path/to/AscendOpGenAgent
- level_problems: {1: [1,2,3,4,5,6,7,8,9,10]}
```

**执行流程：**
1. 首次运行会询问目标硬件架构和 NPU 设备序号
2. 结果保存到 `output`
3. 报告增量更新

### 示例 2: 多 Level 部分选择

```
评测 Level 1 的 [1,2] 和 Level 2 的 [1,3]，Level 3 全选
参数：
- agent_name: lingxi-code
- agent_workspace: /path/to/AscendOpGenAgent
- level_problems: {1: [1,2], 2: [1,3], 3: null}
```

### 示例 3: 指定范围

```
评测 Level 1 的 1-20 号问题
参数：
- agent_name: lingxi-code
- agent_workspace: /path/to/AscendOpGenAgent
- level_problems: {1: "1-20"}
```

### 示例 4: 自定义输出目录

```
评测 Level 1-3 全部问题，自定义输出目录
参数：
- agent_name: lingxi-code
- agent_workspace: /path/to/AscendOpGenAgent
- level_problems: {1: null, 2: null, 3: null}
- output_dir: my_results
```

**输出路径：** `/path/to/AscendOpGenAgent/my_results/20250321_143022_001_lingxi-code/`

### 示例 5: 指定 Benchmark 名称

```
使用名称为 "MyBench" 的 benchmark（路径：{agent_workspace}/benchmarks/MyBench）
参数：
- agent_name: lingxi-code
- agent_workspace: /path/to/AscendOpGenAgent
- level_problems: {1: [1,2,3]}
- benchmark_path: MyBench
```

### 示例 6: 指定相对路径

```
使用 agent_workspace 下的相对路径
参数：
- agent_name: lingxi-code
- agent_workspace: /path/to/AscendOpGenAgent
- level_problems: {1: [1,2,3]}
- benchmark_path: data/custom_benchmarks/NPUKernelBench
# 解析为：/path/to/AscendOpGenAgent/data/custom_benchmarks/NPUKernelBench
```

### 示例 7: 指定绝对路径

```
使用绝对路径指定 benchmark
参数：
- agent_name: lingxi-code
- agent_workspace: /path/to/AscendOpGenAgent
- level_problems: {1: [1,2,3]}
- benchmark_path: /data/shared/benchmarks/NPUKernelBench
# 直接使用：/data/shared/benchmarks/NPUKernelBench

### Agent 配置

需要指定 lingxi-code：

```markdown
---
name: lingxi-code
mode: primary
temperature: 0.1
skills:
  - op_desc_generation
  - reference_generation
  - functional_conversion
  - ascend_call_generation
  - dsl_baseline_generation
  - dsl_lowering
  - ascendc_evalution
---

```


### 调用方式

ascend-benchmark-evaluator 调用步骤：
1.用户输入：
```
示例:
串行生成/home/y00889327/AscendOpGenAgent/benchmarks/NPUKernelBench/level1 前两个算子，并将测试结果汇总形成表格
```

2.解析数据集需要生成的算子
假设/home/y00889327/AscendOpGenAgent/benchmarks/NPUKernelBench/level1 前两个算子是GELU和SwiGLU
3.切换到Lingxi-Code Agent
4.生成一个GELU算子
5.调用 `ascendc_evalution` skill 验证GELU算子精度和算子性能
6 生成一个SwiGLU算子
7.调用 `ascendc_evalution` skill 验证SwiGLU算子精度和算子性能
8.生成agent_report.md评测报告



### 工作流程
```
ascend-benchmark-evaluator
    ↓
直接调用 `lingxi-code` 
    ↓
内部执行：
    | Stage | Skill | Output |
    |-------|-------|--------|
    | 1 | `op_desc_generation` | Operator JSON descriptor |
    | 2 | `reference_generation` | Reference PyTorch implementation |
    | 3 | `functional_conversion` | Functional PyTorch API |
    | 4 | `ascend_call_generation` | Ascend function bindings |
    | 5 | `dsl_baseline_generation` | Baseline DSL code |
    | 6 | `dsl_lowering` | Lowered AscendC Code & Compilation artifacts |
    | 7 | `ascendc_evalution` | Deployment & Evalution result |
    | 8 | - | Final summary & status report |
    | 9 | - | Final benchmark summary & status report |
    ↓
ascend-benchmark-evaluator 收集结果并生成报告
```

## 依赖

- Python 3.8+
- opencode Agent 调用机制
- lingxi-code agent（用于生成 AscendC 算子）
- ascendc_evalution skill（用于安装、pybind、评测）
- NPU 设备（用于验证和性能测试）

## 注意事项

1. **串行执行**: 任务按顺序逐个执行，不进行并行化
2. **始终生成报告**: 无论评测成功与否，都会生成 `agent_report.md`
3. **增量报告**: 每个任务完成后立即更新报告，支持实时查看进度
4. **断点续跑**: 基于 `(level, problem_id)` 去重，支持中断后恢复
5. **超时处理**: 单任务超时不影响整体流程，记录为失败
6. **错误隔离**: 单任务失败会记录但继续执行其他任务
7. **架构选择**: 首次运行需选择目标硬件架构，选择后保存到状态文件
8. **NPU 选择**: 首次运行需选择目标 NPU 设备序号，选择后保存到状态文件
