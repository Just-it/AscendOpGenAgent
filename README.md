# AscendOpGenAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

中文 | [English](README.en.md)

**AscendOpGenAgent** 是一个面向 Ascend NPU 的自动化算子生成与评测框架。本项目基于 Triton/AscendC 自动生成并验证高性能算子代码，旨在大幅提升 Ascend 架构下的算子开发效率与质量。

## 目录

- [AscendOpGenAgent](#ascendopgenagent)
  - [目录](#目录)
  - [核心功能](#核心功能)
  - [快速开始](#快速开始)
    - [1. 环境要求](#1-环境要求)
    - [2. 安装与配置](#2-安装与配置)
    - [3. 使用场景指南](#3-使用场景指南)
      - [**3.1 Triton**](#31-triton)
      - [场景一：单算子生成](#场景一单算子生成)
      - [场景二：Benchmark 批量评测](#场景二benchmark-批量评测)
      - [**3.2 AscendC**](#32-ascendc)
      - [场景一：单算子生成 (Lingxi-code Agent)](#场景一单算子生成-lingxi-code-agent)
      - [场景二：Benchmark 批量评测 (Ascend-Benchmark-Evaluator)](#场景二benchmark-批量评测-ascend-benchmark-evaluator)
      - [场景三：A5 单算子生成（预览版）(ascendc-a5-op-gen Skill)](#场景三a5-单算子生成-ascendc-a5-op-gen-skill)
    - [评测基线](#评测基线)
      - [Triton](#triton)
      - [AscendC](#ascendc)
  - [项目结构](#项目结构)
  - [许可证](#许可证)

## 核心功能

| 算子类型 | 模块 | 定位 | 核心能力 |
|------|------|------|----------|
| **Triton** | **AKG-Triton Agent** | 单算子交互式生成 | 任务提取 → 代码生成 → 评测验证（精度对齐与性能测试） |
| **Triton**  | **Benchmark-Evaluator** | 一键批量评测 | 执行指定 Benchmark 评测，自动总结并生成详细报告 |
| **AscendC** | **Lingxi_code Agent** | AscendC 单算子交互式生成 | 代码生成 → 评测验证（精度对齐与性能测试） |
| **AscendC** | **Ascend-Benchmark-Evaluator** | AscendC 算子一键批量评测 | 执行指定 Benchmark 评测，自动总结并生成详细报告 |
| **AscendC (A5)** | **ascendc-a5-op-gen** | A5 单算子生成与优化（预览版） | 支持 Benchmark / Op-gen / Optimize 三种模式，自动生成并优化 AscendC SIMT/SIMD kernel |

>  **共享内核**：AKG-Triton Agent、Benchmark-Evaluator两者底层共用代码生成 Agent，统一处理“代码生成 → 验证 → 性能测试”的核心工作流，确保生成逻辑的一致性与高复用性。

##  快速开始

### 1. 环境要求

在运行本项目之前，请确保您的环境满足以下要求：
- Python 3.8+
- Ascend CANN 8.0+
- Triton Ascend
- PyTorch 2.0+
- Claude Code CLI (请确保已正确安装并配置)
- tilelang-ascend (参考https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/README.md#method-3-compile-and-install-from-source 安装)

### 2. 安装与配置

克隆本项目并配置 Claude Code 环境：

```bash
# 1. 克隆项目并进入目录
git clone https://github.com/your-repo/AscendOpGenAgent.git
cd AscendOpGenAgent

# 2. 配置 Claude Code（可选，如需自定义配置）
# Claude Code 会自动识别项目中的 .claude/CLAUDE.md 配置文件
```

完成后，即可在项目目录中使用 Claude Code 进行开发。

### 3. 使用场景指南

本项目主要提供两个核心使用场景，请根据需求选择对应的 Agent 或 Skill。
#### **3.1 Triton**

#### 场景一：单算子生成

适用于开发者需要快速生成、验证某个特定算子的 Triton 实现。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下配置 Agent和skills：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/triton-ascend-coder.md .claude/CLAUDE.md
mv skills/triton/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，启动 claude：
```bash
claude
```

3. 输入算子生成 Prompt：
```text
生成一个基于 Triton-Ascend 框架的 softmax 算子实现。目标设备架构为 ascend910b1，请将生成的代码文件输出至 /path/to/output/ 目录下。
```

**执行流程**：Agent 自动执行 Phase 0-5：参数确认 → 任务构建 → 算法设计 → 代码生成与验证（迭代） → 性能优化与验证（迭代） → 输出报告。

---

#### 场景二：Benchmark 批量评测

适用于批量评测算子的生成效果，支持单 NPU 串行或多 NPU 并行执行。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下创建 `.claude` 目录并配置 Agent：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/triton-ascend-coder.md .claude/CLAUDE.md
mv skills/triton/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，执行批量调度脚本：

**单 NPU 串行模式**：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_triton.sh \
    --benchmark-dir /path/to/KernelBench \
    --level 1 \
    --range 1-30 \
    --npu 0 \
    --output /path/to/output
```

**多 NPU 并行模式**（推荐）：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_triton.sh \
    --benchmark-dir /path/to/KernelBench \
    --level 1 \
    --range 1-30 \
    --npu-list "0,1,2,3,4,5" \
    --output /path/to/output
```

**参数说明**：
- `--benchmark-dir`: Benchmark 根目录路径（必填）
- `--level`: Level 编号，如 1, 2, 3, 4（必填）
- `--range`: 算子范围，如 `1-30`（与 `--ids` 二选一）
- `--ids`: 指定算子编号列表，逗号分隔，如 `3,7,15`（与 `--range` 二选一）
- `--npu`: 单 NPU 设备 ID，如 0（默认 0，与 `--npu-list` 互斥）
- `--npu-list`: 多 NPU 列表，逗号分隔，如 `0,1,2,3,4,5`（与 `--npu` 互斥，优先级更高）
- `--output`: 输出目录（必填）


#### **3.2 AscendC**

#### 场景一：单算子生成 (Lingxi-code Agent)

适用于开发者需要快速生成、验证某个特定算子的 AscendC 实现。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下配置 Agent 和 skills：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/ascend-kernel-developer.md .claude/CLAUDE.md
mv skills/ascendc/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，启动 claude：
```bash
claude
```

3. 输入算子生成 Prompt：
```text
生成一个基于 AscendC 框架的 softmax 算子实现。目标设备架构为 ascend910b2，请将生成的代码文件输出至 /path/to/output/ 目录下。
```

**执行流程**：Agent 自动执行：确认参数 → 提取任务描述 → 生成代码 → 验证精度与性能 → 输出最终报告。

---

#### 场景二：Benchmark 批量评测 (Ascend-Benchmark-Evaluator)

适用于批量评测算子的生成效果，支持单 NPU 串行或多 NPU 并行执行。

**操作步骤**：

1. 在 AscendOpGenAgent 目录下创建 `.claude` 目录并配置 Agent：
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/ascend-kernel-developer.md .claude/CLAUDE.md
mv skills/ascendc/* .claude/skills/
```

2. 进入 AscendOpGenAgent 目录，执行批量调度脚本：

**单 NPU 串行模式**：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --range 1-30 \
    --npu 0 \
    --output /path/to/output
```

**多 NPU 并行模式**（推荐）：
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --range 1-30 \
    --npu-list "0,1,2,3,4,5" \
    --output /path/to/output
```

**参数说明**：
- `--benchmark-dir`: Benchmark 根目录路径（必填）
- `--level`: Level 编号，如 1, 2, 3（必填）
- `--range`: 算子范围，如 `1-30`（与 `--ids` 二选一）
- `--ids`: 指定算子编号列表，逗号分隔，如 `3,7,15`（与 `--range` 二选一）
- `--npu`: 单 NPU 设备 ID，如 0（默认 0，与 `--npu-list` 互斥）
- `--npu-list`: 多 NPU 列表，逗号分隔，如 `0,1,2,3,4,5`（与 `--npu` 互斥，优先级更高）
- `--output`: 输出目录（必填）

---

#### 场景三：A5 单算子生成（预览版）(ascendc-a5-op-gen Skill)

> **注意**：A5 生成能力目前处于预览/测试阶段，生成质量和成功率有限，遇到复杂算子时可能需要人工介入调整。

适用于在 A5 架构上快速生成、验证并优化单个 AscendC 算子。支持三种模式：
- **Benchmark 模式**：从 NPUKernelBench 源文件自动生成 kernel
- **Op-gen 模式**：从任意 CUDA/PyTorch 源文件生成 AscendC 实现
- **Optimize 模式**：对已有归档 kernel 进行性能优化

**操作步骤**：

1. 在 AscendOpGenAgent 目录下配置 A5 Agent 和 skills：
```bash
mkdir -p .claude
mkdir -p .claude/skills
cp agents/ascend-a5-kernel-developer.md .claude/CLAUDE.md
cp -r skills/ascendc/* .claude/skills/
```

2. 启动 Claude 并输入 Skill 命令：

**Benchmark 模式**（从 NPUKernelBench 生成）：
```text
/ascendc-a5-op-gen 13_Cat
```

**Op-gen 模式**（从源文件生成）：
```text
/ascendc-a5-op-gen path/to/source.py
```

**Optimize 模式**（优化已有算子）：
```text
/ascendc-a5-op-gen --optimize 12_Permute
```

**执行流程**：
1. 自动识别模式并创建输出目录（如 `output/npukernelbench/13_Cat`）
2. 生成 AscendC kernel 文件（`.h`、`.cpp`、`pybind11.cpp`、`model_new_ascendc.py`）
3. 静态检查 + 编译（最多 5 轮迭代修复）
4. 精度验证（所有 case 必须 PASS）
5. 性能测试并计算 ratio（低于 0.6x 时自动进入优化阶段）
6. 输出最终报告和 `PROGRESS.md`

算子生成完成后，可进一步使用 `a5-perf-summary` 进行统一的性能汇总与归档：

```text
/a5-perf-summary 1_GELU    # 单算子汇总
/a5-perf-summary all       # 批量汇总所有已完成算子
```

该步骤会自动验证精度、检测 kernel 编程模式（SIMT / SIMD）、计算性能 ratio，并更新 `benchmarks/NPUKernelBench/A5_RESULTS.md`。

### 评测基线

#### Triton
关于 Triton 的相关数据，请参阅[`benchmarks/BASELINE_0408.md`](benchmarks/BASELINE_0408.md)

#### AscendC
关于 AscendC 的相关数据，请参阅[`benchmarks/BASELINE_0408.md`](benchmarks/BASELINE_0408.md) 



## 项目结构

```text
AscendOpGenAgent/
├── .gitignore
├── LICENSE
├── README.en.md
├── README.md
├── agents/                     # Agent 定义目录
│   ├── AKG-triton.md           # 主编排 Agent
│   ├── benchmark-scheduler.md
│   ├── kernelgen-workflow.md   # 子 Agent（代码生成工作流）
│   ├── ascend-kernel-developer.md
│   └── performance-optimizer.md
├── benchmarks/                 # 评测数据集存放目录
│   ├── KernelBench/
│   │   ├── level1/             # Level 1 测试用例 (100个)
│   │   ├── level2/             # Level 2 测试用例 (99个)
│   │   ├── level3/             # Level 3 测试用例 (52个)
│   │   └── level4/             # Level 4 测试用例 (20个)
│   └── NPUKernelBench/
│       └── level1/             # NPU KernelBench Level 1 测试用例 (31个)
└── skills/                     # Skill 实现目录
    ├── ascendc_evalution/
    ├── ascend_benchmark_evaluator/
    ├── ascendc/
    ├── benchmark-evaluator/    # 批量评测 Skill
    ├── dsl_baseline_generation/
    ├── dsl_lowering/
    ├── functional_conversion/
    ├── kernel-designer/
    ├── kernel-generator/       # 代码生成 Skill
    ├── kernel-verifier/        # 验证与性能测试 Skill
    ├── latency-optimizer/
    ├── op-task-extractor/      # 任务提取 Skill
    ├── op_desc_generation/
    └── reference_generation/

```


## 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。