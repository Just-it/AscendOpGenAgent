# AscendOpGenAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[中文](README.md) | English

**AscendOpGenAgent** is an automated operator generation and evaluation framework for Ascend NPUs. Based on Triton/AscendC, this project automatically generates and verifies high-performance operator code, aiming to significantly improve the efficiency and quality of operator development on the Ascend architecture.

## Table of Contents

- [AscendOpGenAgent](#ascendopgenagent)
  - [Table of Contents](#table-of-contents)
  - [Core Features](#core-features)
  - [Quick Start](#quick-start)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Installation \& Configuration](#2-installation--configuration)
    - [3. Usage Scenarios](#3-usage-scenarios)
      - [**3.1 Triton**](#31-triton)
      - [Scenario 1: Single Operator Generation (AKG-Triton Agent)](#scenario-1-single-operator-generation-akg-triton-agent)
      - [Scenario 2: Batch Benchmark Evaluation (Benchmark-Evaluator)](#scenario-2-batch-benchmark-evaluation-benchmark-evaluator)
      - [**3.2 AscendC**](#32-ascendc)
      - [Scenario 1: Single Operator Generation (Lingxi-code Agent)](#scenario-1-single-operator-generation-lingxi-code-agent)
      - [Scenario 2: Batch Benchmark Evaluation (Ascend-Benchmark-Evaluator)](#scenario-2-batch-benchmark-evaluation-ascend-benchmark-evaluator)
      - [Scenario 3: A5 Single Operator Generation (Preview) (ascendc-a5-op-gen Skill)](#scenario-3-a5-single-operator-generation-ascendc-a5-op-gen-skill)
    - [Evaluation Baseline](#evaluation-baseline)
      - [Triton](#triton)
      - [AscendC](#ascendc)
  - [Project Structure](#project-structure)
  - [License](#license)

## Core Features

| Operator Type | Module | Positioning | Core Capabilities |
|------|------|------|----------|
| **Triton** | **AKG-Triton Agent** | Single operator interactive generation | Task extraction → Code generation → Evaluation & Verification (Accuracy alignment & Performance testing) |
| **Triton** | **Benchmark-Evaluator** | One-click batch evaluation | Execute specified Benchmark evaluation, automatically summarize and generate detailed reports |
| **AscendC** | **Lingxi_code Agent** | AscendC single operator interactive generation | Code generation → Evaluation & Verification (Accuracy alignment & Performance testing) |
| **AscendC** | **Ascend-Benchmark-Evaluator** | AscendC operator one-click batch evaluation | Execute specified Benchmark evaluation, automatically summarize and generate detailed reports |
| **AscendC (A5)** | **ascendc-a5-op-gen** | A5 single operator generation and optimization (Preview) | Supports Benchmark / Op-gen / Optimize modes, automatically generates and optimizes AscendC SIMT/SIMD kernels |

> **Shared Kernel**: AKG-Triton Agent and Benchmark-Evaluator share the underlying code generation Agent, uniformly handling the core workflow of "Code Generation → Verification → Performance Testing" to ensure consistency and high reusability of the generation logic.

## Quick Start

### 1. Prerequisites

Before running this project, please ensure your environment meets the following requirements:
- Python 3.8+
- Ascend CANN 8.0+
- Triton Ascend
- PyTorch 2.0+
- Claude Code CLI (Please ensure it is correctly installed and configured)

### 2. Installation & Configuration

Clone this project and configure the Claude Code environment:

```bash
# 1. Clone the project and enter the directory
git clone https://github.com/your-repo/AscendOpGenAgent.git
cd AscendOpGenAgent

# 2. Configure Claude Code (Optional, if custom configuration is needed)
# Claude Code will automatically recognize the .claude/CLAUDE.md configuration file in the project
```

After completion, you can use Claude Code for development in the project directory.

### 3. Usage Scenarios

This project mainly provides two core usage scenarios. Please select the corresponding Agent or Skill according to your needs.

#### **3.1 Triton**

#### Scenario 1: Single Operator Generation

Suitable for developers who need to quickly generate and verify the Triton implementation of a specific operator.

**Steps**:

1. Configure the Agent and skills in the AscendOpGenAgent directory:
```bash
mkdir -p .claude
mkdir -p .claude/skills
cp agents/triton-ascend-coder.md .claude/CLAUDE.md
cp -r skills/triton/* .claude/skills/
```

2. Enter the AscendOpGenAgent directory and start Claude:
```bash
claude
```

3. Enter the operator generation Prompt:
```text
Generate a softmax operator implementation based on the Triton-Ascend framework. The target device architecture is ascend910b1. Please output the generated code files to the /path/to/output/ directory.
```

**Execution Flow**: Agent automatically executes Phase 0-5: Parameter confirmation → Task construction → Algorithm design → Code generation & verification (iterative) → Performance optimization & verification (iterative) → Output report.

---

#### Scenario 2: Batch Benchmark Evaluation

Suitable for batch generation and evaluation of multiple operators with support for single NPU serial or multi-NPU parallel execution.

**Steps**:

1. Create the `.claude` directory in the AscendOpGenAgent directory and configure the Agent:
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/triton-ascend-coder.md .claude/CLAUDE.md
mv skills/triton/* .claude/skills/
```

2. Enter the AscendOpGenAgent directory and execute the batch scheduling script:

**Single NPU Serial Mode** (backward compatible):
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_triton.sh \
    --benchmark-dir /path/to/KernelBench \
    --level 1 \
    --range 1-30 \
    --npu 0 \
    --output /path/to/output
```

**Multi-NPU Parallel Mode** (recommended for better hardware utilization):
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_triton.sh \
    --benchmark-dir /path/to/KernelBench \
    --level 1 \
    --range 1-30 \
    --npu-list "0,1,2,3,4,5" \
    --output /path/to/output
```

**Parameter Description**:
- `--benchmark-dir`: Path to KernelBench root directory (required)
- `--level`: Level number, e.g., 1, 2, 3, 4 (required)
- `--range`: Operator range, e.g., `1-30` (mutually exclusive with `--ids`)
- `--ids`: Comma-separated operator IDs, e.g., `3,7,15` (mutually exclusive with `--range`)
- `--npu`: Single NPU device ID, e.g., 0 (default 0, mutually exclusive with `--npu-list`)
- `--npu-list`: Multi-NPU list, comma-separated, e.g., `0,1,2,3,4,5` (mutually exclusive with `--npu`, higher priority)
- `--output`: Output directory (required)


#### **3.2 AscendC**

#### Scenario 1: Single Operator Generation (Lingxi-code Agent)

Suitable for developers who need to quickly generate and verify the AscendC implementation of a specific operator.

**Steps**:

1. Configure the Agent and skills in the AscendOpGenAgent directory:
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/lingxi_code.md .claude/CLAUDE.md
mv skills/ascend_call_generation/* .claude/skills/
```

2. Enter the AscendOpGenAgent directory and start Claude:
```bash
claude
```

3. Enter the operator generation Prompt:
```text
Generate a softmax operator implementation based on the AscendC framework. The target device architecture is ascend910b2. Please output the generated code files to the /path/to/output/ directory.
```

**Execution Flow**: Agent automatically executes: Confirm parameters → Extract task description → Generate code → Verify accuracy and performance → Output final report.

---

#### Scenario 2: Batch Benchmark Evaluation (Ascend-Benchmark-Evaluator)

Suitable for batch generation and evaluation of multiple operators with support for single NPU serial or multi-NPU parallel execution.

**Steps**:

1. Create the `.claude` directory in the AscendOpGenAgent directory and configure the Agent:
```bash
mkdir -p .claude
mkdir -p .claude/skills
mv agents/lingxi_code.md .claude/CLAUDE.md
mv skills/ascend_call_generation/* .claude/skills/
```

2. Enter the AscendOpGenAgent directory and execute the batch scheduling script:

**Single NPU Serial Mode**:
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --range 1-30 \
    --npu 0 \
    --output /path/to/output
```

**Multi-NPU Parallel Mode** (recommended):
```bash
cd /path/to/AscendOpGenAgent
bash utils/run_benchmark_ascendc.sh \
    --benchmark-dir /path/to/NPUKernelBench \
    --level 1 \
    --range 1-30 \
    --npu-list "0,1,2,3,4,5" \
    --output /path/to/output
```

**Parameter Description**:
- `--benchmark-dir`: Benchmark root directory path (required)
- `--level`: Level number, e.g., 1, 2, 3 (required)
- `--range`: Operator range, e.g., `1-30` (mutually exclusive with `--ids`)
- `--ids`: Comma-separated operator ID list, e.g., `3,7,15` (mutually exclusive with `--range`)
- `--npu`: Single NPU device ID, e.g., 0 (default 0, mutually exclusive with `--npu-list`)
- `--npu-list`: Multi-NPU list, comma-separated, e.g., `0,1,2,3,4,5` (mutually exclusive with `--npu`, higher priority)
- `--output`: Output directory (required)

---

#### Scenario 3: A5 Single Operator Generation (Preview) (ascendc-a5-op-gen Skill)

> **Note**: The A5 generation capability is currently in preview / testing phase. Generation quality and success rate are limited, and complex operators may require manual intervention.

Suitable for quickly generating, verifying, and optimizing a single AscendC operator on A5 architecture. Supports three modes:
- **Benchmark mode**: Automatically generate a kernel from an NPUKernelBench source file
- **Op-gen mode**: Generate an AscendC implementation from any CUDA/PyTorch source file
- **Optimize mode**: Optimize performance of an already archived kernel

**Steps**:

1. Configure the A5 Agent and skills in the AscendOpGenAgent directory:
```bash
mkdir -p .claude
mkdir -p .claude/skills
cp agents/ascend-a5-kernel-developer.md .claude/CLAUDE.md
cp -r skills/ascendc/* .claude/skills/
```

2. Start Claude and enter the Skill command:

**Benchmark mode** (generate from NPUKernelBench):
```text
/ascendc-a5-op-gen 13_Cat
```

**Op-gen mode** (generate from source file):
```text
/ascendc-a5-op-gen path/to/source.py
```

**Optimize mode** (optimize existing operator):
```text
/ascendc-a5-op-gen --optimize 12_Permute
```

**Execution Flow**:
1. Automatically detect mode and create output directory (e.g., `output/npukernelbench/13_Cat`)
2. Generate AscendC kernel files (`.h`, `.cpp`, `pybind11.cpp`, `model_new_ascendc.py`)
3. Static check + compilation (up to 5 iterative fix rounds)
4. Accuracy verification (all cases must PASS)
5. Performance test and compute ratio (auto-enters optimization if below 0.6x)
6. Output final report and `PROGRESS.md`

After operator generation, you can use `a5-perf-summary` for unified performance summary and archiving:

```text
/a5-perf-summary 1_GELU    # single operator summary
/a5-perf-summary all       # batch summary for all completed operators
```

This step automatically verifies accuracy, detects the kernel programming mode (SIMT / SIMD), computes performance ratios, and updates `benchmarks/NPUKernelBench/A5_RESULTS.md`.

### Evaluation Baseline

#### Triton

Please refer to [`benchmarks/BASELINE_0408.md`](benchmarks/BASELINE_0408.md) for Triton-related data.

#### AscendC

Please refer to [`benchmarks/BASELINE_0408.md`](benchmarks/BASELINE_0408.md) for AscendC-related data.

## Project Structure

```text
AscendOpGenAgent/
├── .gitignore
├── LICENSE
├── README.en.md
├── README.md
├── agents/                     # Agent definition directory
│   ├── AKG-triton.md           # Main orchestration Agent
│   ├── benchmark-scheduler.md
│   ├── kernelgen-workflow.md   # Sub-Agent (Code generation workflow)
│   ├── lingxi_code.md
│   └── performance-optimizer.md
├── benchmarks/                 # Evaluation dataset storage directory
│   ├── KernelBench/
│   │   ├── level1/             # Level 1 test cases (100 tasks)
│   │   ├── level2/             # Level 2 test cases (99 tasks)
│   │   ├── level3/             # Level 3 test cases (52 tasks)
│   │   └── level4/             # Level 4 test cases (20 tasks)
│   └── NPUKernelBench/
│       └── level1/             # NPU KernelBench Level 1 test cases (31 tasks)
└── skills/                     # Skill implementation directory
    ├── ascendc_evalution/
    ├── ascend_benchmark_evaluator/
    ├── ascend_call_generation/
    ├── benchmark-evaluator/    # Batch evaluation Skill
    ├── dsl_baseline_generation/
    ├── dsl_lowering/
    ├── functional_conversion/
    ├── kernel-designer/
    ├── kernel-generator/       # Code generation Skill
    ├── kernel-verifier/        # Verification and performance testing Skill
    ├── latency-optimizer/
    ├── op-task-extractor/      # Task extraction Skill
    ├── op_desc_generation/
    └── reference_generation/
```


## License

This project is licensed under the [Apache 2.0 License](LICENSE).
