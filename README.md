# AscendOpGenAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

中文 | [English](README.en.md)

**AscendOpGenAgent** 是一个面向 Ascend NPU 的自动化算子生成与评测框架。本项目基于 Triton 自动生成并验证高性能算子代码，旨在大幅提升 Ascend 架构下的算子开发效率与质量。

## 目录

- [核心功能](#核心功能)
- [快速开始](#快速开始)
  - [环境要求](#1-环境要求)
  - [安装与配置](#2-安装与配置)
  - [使用场景指南](#3-使用场景指南)
    - [单算子生成](#场景一单算子生成-akg-triton-agent)
    - [Benchmark 批量评测](#场景二benchmark-批量评测-benchmark-evaluator)
  - [评测基线（更新于 2026-03-20）](#评测基线更新于-2026-03-20)
- [项目结构](#项目结构)
- [许可证](#许可证)

## 核心功能

| 模块 | 定位 | 核心能力 |
|------|------|----------|
| **AKG-Triton Agent** | 单算子交互式生成 | 任务提取 → 代码生成 → 评测验证（精度对齐与性能测试） |
| **Benchmark-Evaluator** | 一键批量评测 | 执行指定 Benchmark 评测，自动总结并生成详细报告 |

>  **共享内核**：两者底层共用代码生成 Agent，统一处理“代码生成 → 验证 → 性能测试”的核心工作流，确保生成逻辑的一致性与高复用性。

##  快速开始

### 1. 环境要求

在运行本项目之前，请确保您的环境满足以下要求：
- Python 3.8+
- Ascend CANN 8.0+
- Triton Ascend
- PyTorch 2.0+
- [OpenCode](https://opencode.ai/) (请确保已正确安装并配置)

### 2. 安装与配置

首先，克隆本项目并将其配置到 OpenCode 的工作环境中：

```bash
# 1. 克隆项目并进入目录
git clone https://github.com/your-repo/AscendOpGenAgent.git
cd AscendOpGenAgent

# 2. 部署 Agent 和 Skills 到 OpenCode 默认配置路径
mkdir -p ~/.config/opencode/
cp -r agents/ ~/.config/opencode/
cp -r skills/ ~/.config/opencode/
```

完成后，启动 OpenCode，即可在界面或命令行中选择对应的 Agents 和 Skills。

### 3. 使用场景指南

本项目主要提供两个核心使用场景，请根据需求选择对应的 Agent 或 Skill。

#### 场景一：单算子生成 (AKG-Triton Agent)
适用于开发者需要快速生成、验证某个特定算子的 Triton 实现。

**操作步骤**：
1. 在 OpenCode 中，通过 `/agents` 命令切换至 `AKG-Triton`。
2. 输入算子生成 Prompt。

**Prompt 示例**：
```text
/AKG-Triton
生成一个基于 Triton-Ascend 框架的 softmax_mat 算子实现。目标设备架构为 ascend910b2，请将生成的代码文件输出至 /path/to/output/ 目录下。
```

**执行流程**：
Agent 接收到指令后，将自动执行以下流程：确认参数 → 提取任务描述 → 生成代码 → 验证精度与性能 → 输出最终报告。

#### 场景二：Benchmark 批量评测 (Benchmark-Evaluator)
适用于评估 Agent 在标准数据集（如 KernelBench）上的整体代码生成能力。

**操作步骤**：
1. 在 OpenCode 中，通过 `/skills` 命令切换至 `benchmark-evaluator`。
2. 输入评测 Prompt。

**Prompt 示例 1：基础评测**（仅指定目标与测试范围）
```text
评测KernelBench中level1的[20,30]任务,agent_workspace是<path/to/your/AscendOpGenAgent>,使用<AKG-triton> agent
```

**Prompt 示例 2：进阶评测**（指定输出路径、运行设备及权限）
```text
执行KernelBench评测,agent_workspace是<path/to/your/AscendOpGenAgent>,使用<AKG-triton> agent。
评测 Level 1 的 problem_id=[6] 和 Level 2 的 problem_id=[2]。
请将生成的代码和评测结果输出到 /path/to/output 目录下。
执行期间默认同意所有权限，并指定设备 ASCEND_RT_VISIBLE_DEVICES=10。
```

**参数说明**：
- `<agent_path>`: 本项目的工作目录路径（需包含 `agents/` 和 `skills/`）。
- `<benchmark_path>`: 评测数据集（如 KernelBench）的本地路径。
- `<output_path>`: **[可选]** 评测结果与生成代码的输出目录。
- `ASCEND_RT_VISIBLE_DEVICES`: **[可选]** 指定使用的 NPU 设备 ID。

### 评测基线（更新于 2026-03-20）

- **测试设备**：Ascend 910B2
- **总任务数**：12

| Level | Problem ID | 算子名称 | 编译通过 | 精度正确 | PyTorch 延迟 | 生成代码延迟 | 加速比 | 最终状态 |
|:---:|:---:|---|:---:|:---:|---:|---:|---:|:---:|
| 1 | 1 | `Square_matrix_multiplication_` | ✅ | ✅ | 1.65 ms | 2.95 ms | 0.56x | 成功 |
| 1 | 2 | `Standard_matrix_multiplication_` | ✅ | ✅ | 1.65 ms | 7.82 ms | 0.21x | 成功 |
| 1 | 3 | `Batched_matrix_multiplication` | ✅ | ✅ | 3.64 ms | 9.70 ms | 0.38x | 成功 |
| 1 | 4 | `Matrix_vector_multiplication_` | ✅ | ✅ | 36.26 ms | 162.41 ms | 0.22x | 成功 |
| 1 | 5 | `Matrix_scalar_multiplication` | ✅ | ✅ | 6.80 ms | 7.70 ms | 0.88x | 成功 |
| 1 | 6 | `Matmul_with_large_K_dimension_` | ✅ | ✅ | 2.35 ms | 2.35 ms | 1.00x | 成功 |
| 1 | 7 | `Matmul_with_small_K_dimension_` | ✅ | ✅ | 3.34 ms | 4.07 ms | 0.82x | 成功 |
| 1 | 8 | `Matmul_with_irregular_shapes_` | ✅ | ✅ | 4.24 ms | 4.28 ms | 0.99x | 成功 |
| 1 | 9 | `Tall_skinny_matrix_multiplication_` | ✅ | ✅ | 3.20 ms | 4.02 ms | 0.79x | 成功 |
| 2 | 3 | `ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU` | ✅ | ✅ | 16.11 ms | 16.99 ms | 0.95x | 成功 |
| 3 | 4 | `LeNet5` | ✅ | ✅ | 1.72 ms | 113.54 ms | 0.02x | 成功 |


## 项目结构

```text
AscendOpGenAgent/
├── agents/                     # Agent 定义目录
│   ├── AKG-triton.md           # 主编排 Agent
│   └── kernelgen-workflow.md   # 子 Agent（代码生成工作流）
├── skills/                     # Skill 实现目录
│   ├── op-task-extractor/      # 任务提取 Skill
│   ├── kernel-generator/       # 代码生成 Skill
│   ├── kernel-verifier/        # 验证与性能测试 Skill
│   └── benchmark-evaluator/    # 批量评测 Skill
├── benchmarks/                 # 评测数据集存放目录
│   └── KernelBench/
└── README.md
```


## 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。