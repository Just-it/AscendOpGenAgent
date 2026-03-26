# AscendOpGenAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[中文](README.md) | English

**AscendOpGenAgent** is an automated operator generation and evaluation framework for Ascend NPUs. Based on Triton, this project automatically generates and verifies high-performance operator code, aiming to significantly improve the efficiency and quality of operator development on the Ascend architecture.

## Table of Contents

- [Core Features](#core-features)
- [Quick Start](#quick-start)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Installation & Configuration](#2-installation--configuration)
  - [3. Usage Scenarios](#3-usage-scenarios)
    - [Scenario 1: Single Operator Generation](#scenario-1-single-operator-generation-akg-triton-agent)
    - [Scenario 2: Batch Benchmark Evaluation](#scenario-2-batch-benchmark-evaluation-benchmark-evaluator)
  - [Evaluation Baseline (Updated 2026-03-20)](#evaluation-baseline-updated-2026-03-20)
- [Project Structure](#project-structure)
- [License](#license)

## Core Features

| Module | Positioning | Core Capabilities |
|------|------|----------|
| **AKG-Triton Agent** | Single operator interactive generation | Task extraction → Code generation → Evaluation & Verification (Accuracy alignment & Performance testing) |
| **Benchmark-Evaluator** | One-click batch evaluation | Execute specified Benchmark evaluation, automatically summarize and generate detailed reports |

> **Shared Kernel**: Both share the underlying code generation Agent, uniformly handling the core workflow of "Code Generation → Verification → Performance Testing" to ensure consistency and high reusability of the generation logic.

## Quick Start

### 1. Prerequisites

Before running this project, please ensure your environment meets the following requirements:
- Python 3.8+
- Ascend CANN 8.0+
- Triton Ascend
- PyTorch 2.0+
- [OpenCode](https://opencode.ai/) (Please ensure it is correctly installed and configured)

### 2. Installation & Configuration

First, clone this project and configure it into your OpenCode workspace:

```bash
# 1. Clone the project and enter the directory
git clone https://github.com/your-repo/AscendOpGenAgent.git
cd AscendOpGenAgent

# 2. Deploy Agents and Skills to the default OpenCode configuration path
mkdir -p ~/.config/opencode/
cp -r agents/ ~/.config/opencode/
cp -r skills/ ~/.config/opencode/
```

After completion, start OpenCode, and you can select the corresponding Agents and Skills in the UI or command line.

### 3. Usage Scenarios

This project mainly provides two core usage scenarios. Please select the corresponding Agent or Skill according to your needs.

#### Scenario 1: Single Operator Generation (AKG-Triton Agent)
Suitable for developers who need to quickly generate and verify the Triton implementation of a specific operator.

**Steps**:
1. In OpenCode, switch to `AKG-Triton` via the `/agents` command.
2. Enter the operator generation Prompt.

**Prompt Example**:
```text
/AKG-Triton
Generate a softmax_mat operator implementation based on the Triton-Ascend framework. The target device architecture is ascend910b2. Please output the generated code files to the /path/to/output/ directory.
```

**Execution Flow**:
After receiving the instruction, the Agent will automatically execute the following workflow: Confirm parameters → Extract task description → Generate code → Verify accuracy and performance → Output final report.

#### Scenario 2: Batch Benchmark Evaluation (Benchmark-Evaluator)
Suitable for evaluating the overall code generation capability of the Agent on standard datasets (e.g., KernelBench).

**Steps**:
1. In OpenCode, switch to `benchmark-evaluator` via the `/skills` command.
2. Enter the evaluation Prompt.

**Prompt Example 1: Basic Evaluation** (Only specify target and test scope)
```text
Evaluate tasks [20,30] of level 1 in KernelBench, with agent_workspace set to <path/to/your/AscendOpGenAgent>, using the <AKG-triton> agent.
```

**Prompt Example 2: Advanced Evaluation** (Specify output path, running device, and permissions)
```text
Run KernelBench evaluation with the <AKG-triton> agent (workspace: <path/to/your/AscendOpGenAgent>). Target Level 1 problem_id=[6] and Level 2 problem_id=[2]. Save the generated code and results to /path/to/output. Automatically approve all permissions during execution, and specify the device ASCEND_RT_VISIBLE_DEVICES=10.
```

**Parameter Description**:
- `<agent_path>`: The working directory path of this project (must contain `agents/` and `skills/`).
- `<benchmark_path>`: The local path of the evaluation dataset (e.g., KernelBench).
- `<output_path>`: **[Optional]** Output directory for evaluation results and generated code.
- `ASCEND_RT_VISIBLE_DEVICES`: **[Optional]** Specify the NPU device ID to use.

### Evaluation Baseline (Updated 2026-03-20)

- **Test Device**: Ascend 910B2
- **Total Tasks**: 12

| Level | Problem ID | Operator Name | Compilation | Accuracy | PyTorch Latency | Generated Code Latency | Speedup | Final Status |
|:---:|:---:|---|:---:|:---:|---:|---:|---:|:---:|
| 1 | 1 | `Square_matrix_multiplication_` | ✅ | ✅ | 1.65 ms | 2.95 ms | 0.56x | Success |
| 1 | 2 | `Standard_matrix_multiplication_` | ✅ | ✅ | 1.65 ms | 7.82 ms | 0.21x | Success |
| 1 | 3 | `Batched_matrix_multiplication` | ✅ | ✅ | 3.64 ms | 9.70 ms | 0.38x | Success |
| 1 | 4 | `Matrix_vector_multiplication_` | ✅ | ✅ | 36.26 ms | 162.41 ms | 0.22x | Success |
| 1 | 5 | `Matrix_scalar_multiplication` | ✅ | ✅ | 6.80 ms | 7.70 ms | 0.88x | Success |
| 1 | 6 | `Matmul_with_large_K_dimension_` | ✅ | ✅ | 2.35 ms | 2.35 ms | 1.00x | Success |
| 1 | 7 | `Matmul_with_small_K_dimension_` | ✅ | ✅ | 3.34 ms | 4.07 ms | 0.82x | Success |
| 1 | 8 | `Matmul_with_irregular_shapes_` | ✅ | ✅ | 4.24 ms | 4.28 ms | 0.99x | Success |
| 1 | 9 | `Tall_skinny_matrix_multiplication_` | ✅ | ✅ | 3.20 ms | 4.02 ms | 0.79x | Success |
| 2 | 3 | `ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU` | ✅ | ✅ | 16.11 ms | 16.99 ms | 0.95x | Success |
| 3 | 4 | `LeNet5` | ✅ | ✅ | 1.72 ms | 113.54 ms | 0.02x | Success |


## Project Structure

```text
AscendOpGenAgent/
├── agents/                     # Agent definition directory
│   ├── AKG-triton.md           # Main orchestration Agent
│   └── kernelgen-workflow.md   # Sub-Agent (Code generation workflow)
├── skills/                     # Skill implementation directory
│   ├── op-task-extractor/      # Task extraction Skill
│   ├── kernel-generator/       # Code generation Skill
│   ├── kernel-verifier/        # Verification and performance testing Skill
│   └── benchmark-evaluator/    # Batch evaluation Skill
├── benchmarks/                 # Evaluation dataset storage directory
│   └── KernelBench/
└── README.md
```


## License

This project is licensed under the [Apache 2.0 License](LICENSE).