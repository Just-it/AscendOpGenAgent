---
name: op-gen
description: >
  调用akg_agents workflow生成优化算子。
  支持三种Agent：@kernelgen、@adaptive-search、@evolve。
argument-hint: >
  必需：agent类型、任务文件路径、framework、backend、arch、dsl。
  可选：output-path、devices、agent特定参数。
---

# 算子生成 (Operator Generation)

## 功能概述

本Skill负责调用底层的算子生成Agent，支持三种不同的生成策略：
- **@kernelgen**: 快速迭代生成（默认）
- **@adaptive-search**: UCB自适应搜索
- **@evolve**: 进化算法优化

## 调用方式

### 方式A: 直接调用Agent（推荐）

根据需求直接调用对应的Agent：

```bash
# @kernelgen - 快速生成
@kernelgen --task-file /path/to/op.py --framework torch --backend cuda --arch a100 --dsl triton_cuda

# @adaptive-search - 自适应搜索
@adaptive-search --task-file /path/to/op.py --framework torch --backend cuda --arch a100 --dsl triton_cuda

# @evolve - 进化优化
@evolve --task-file /path/to/op.py --framework torch --backend cuda --arch a100 --dsl triton_cuda
```

### 方式B: 脚本调用

脚本路径：`@scripts/run_agent.py`

```bash
python @scripts/run_agent.py \
  --agent <kernelgen|adaptive-search|evolve> \
  --task-file /abs/path/{op_name}.py \
  --framework <framework> --backend <backend> \
  --arch <arch> --dsl <dsl> \
  --devices <device_ids> \
  --output-path /abs/path/output_dir
```

## Agent选择指南

| 场景 | 推荐Agent | 理由 |
|------|-----------|------|
| 快速原型验证 | @kernelgen | 1-5分钟，快速迭代 |
| 标准算子生成 | @kernelgen | 默认选择，平衡速度和质量 |
| 高性能要求 | @adaptive-search | UCB策略，收敛快 |
| 极致性能 | @evolve | 多轮进化，探索充分 |
| 时间敏感 | @kernelgen | 最快完成 |

## 参数说明

### 通用参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| --agent | string | 是 | Agent类型：kernelgen/adaptive-search/evolve |
| --task-file | string | 是 | KernelBench格式的任务文件路径 |
| --framework | string | 是 | 框架：torch/mindspore |
| --backend | string | 是 | 后端：cuda/ascend/cpu |
| --arch | string | 是 | 架构：a100/v100/ascend910b4等 |
| --dsl | string | 是 | DSL：triton_cuda/triton_ascend/cpp等 |
| --devices | string | 否 | 设备ID，如"0,1,2,3" |
| --output-path | string | 否 | 输出目录路径 |

### kernelgen特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --max-iterations | int | 5 | 最大迭代次数 |
| --enable-code-checker | bool | false | 是否启用代码检查 |

### adaptive-search特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --max-concurrent | int | 2 | 最大并发数 |
| --initial-tasks | int | 2 | 初始任务数 |
| --max-tasks | int | 10 | 最大总任务数 |

### evolve特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --max-rounds | int | 3 | 进化轮数 |
| --parallel-num | int | 4 | 每轮并行数 |
| --num-islands | int | 2 | 岛屿数量 |

## 输出结构

```
${pwd}/triton_ascend_output/
├── generated_code.py      # 最佳生成代码
├── summary.json           # 执行摘要
├── logs/                  # 详细日志
│   ├── kernelgen.log
│   ├── conductor.log
│   └── verifier.log
└── run.log               # 运行日志
```

## 长时间运行处理

Agent运行时间较长，需要特殊处理：

| Agent | 推荐timeout |
|-------|-------------|
| @kernelgen | 3600000ms (60分钟) |
| @adaptive-search | 7200000ms (120分钟) |
| @evolve | 14400000ms (240分钟) |

**执行方式**：前台执行 + 延长timeout + tee记录日志

```bash
# 示例（conda环境）
conda run -n $CONDA_ENV --no-capture-output bash -c \
  "cd $AKG_AGENTS_DIR && source env.sh && \
   python @scripts/run_agent.py \
   --agent kernelgen \
   --task-file /abs/path/op.py \
   --framework torch --backend cuda \
   --arch a100 --dsl triton_cuda \
   --output-path ${pwd}/triton_ascend_output/ \
   2>&1 | tee ${pwd}/triton_ascend_output/run.log"
```

## 结果判定

- **成功**: 命令正常退出 + `summary.json`存在 + `success=true`
- **失败**: 命令异常退出或无`summary.json`或`success=false`
- **中断**: 用户按Esc终止，可查看`run.log`了解进度

## 错误处理

| 错误类型 | 处理方式 |
|---------|---------|
| 环境/安装失败 | 由`akg-env-setup`报告 |
| 任务文件验证失败 | 修复重试（最多2次） |
| 算子生成失败 | 输出失败报告，该任务结束 |
| 超时 | 检查`run.log`，可选择重试 |
