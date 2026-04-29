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

**Two input modes are supported:**
- **Standard Mode**: Using KernelBench (PyTorch Model)
- **GPU Migration Mode**: Using TritonNPUKernelBench (GPU Triton Code → NPU Triton Code)

---

##### Sub-mode A: Standard Mode (KernelBench)

Suitable for standard PyTorch operators batch generation and evaluation.

**Steps**:

1. Create the `.claude` directory in the AscendOpGenAgent directory and configure the Agent:
```bash
mkdir -p .claude
mkdir -p .claude/skills
cp agents/triton-ascend-coder.md .claude/CLAUDE.md
cp -r skills/triton/* .claude/skills/
```

2. Enter the AscendOpGenAgent directory and execute the batch scheduling script:

**Single NPU Serial Mode**:
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
- `--benchmark-dir`: Path to Benchmark root directory (required)
- `--level`: Level number, e.g., 1, 2, 3, 4 (required)
- `--range`: Operator range, e.g., `1-30` (mutually exclusive with `--ids`)
- `--ids`: Comma-separated operator IDs, e.g., `3,7,15` (mutually exclusive with `--range`)
- `--npu`: Single NPU device ID, e.g., 0 (default 0, mutually exclusive with `--npu-list`)
- `--npu-list`: Multi-NPU list, comma-separated, e.g., `0,1,2,3,4,5` (mutually exclusive with `--npu`, higher priority)
- `--output`: Output directory (required)

---

##### Sub-mode B: GPU Triton Code → NPU (TritonNPUKernelBench)

Suitable for migrating existing GPU Triton kernels to NPU Triton implementations with direct performance comparison against GPU baseline.

**Prerequisites**:
Upload the following files to `benchmarks/TritonNPUKernelBench/` directory (files must share the same base name):
- `{op_name}.pt` - Contains `input_data` (required) and optional `gpu_output`
- `vllm_gpu_perf.csv` - GPU performance baseline data (for comparison)

**Steps**:

1. Configure the Agent in the AscendOpGenAgent directory:
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
Generate triton operator,
Description file path: benchmarks/TritonNPUKernelBench/${operator}.py,
arch is ascend910b2, ASCEND_RT_VISIBLE_DEVICES=1
Output directory is /path/to/output
```

> **Note**: Although the prompt includes the `.py` file path, the Agent will automatically detect the TritonNPUKernelBench path and enter **GPU Kernel Input Mode**, automatically looking for the same-named `.pt` file and `vllm_gpu_perf.csv` file. The `.py` file is used to understand the operator logic, while actual data is loaded from `.pt`.

**Execution Flow**:
- **Phase 0**: Auto-detects TritonNPUKernelBench path, enters GPU Kernel Input Mode
- **Phase 1**: Builds task description from `.pt` file (does not call op-task-extractor skill, built by Agent itself)
- **Phase 2-5**: Standard workflow to generate NPU Triton code
- **Performance Comparison**: Auto-comparison of NPU implementation vs GPU baseline performance

**Output Features** (GPU Migration Mode only):
- `report.md` will additionally display **"GPU Reference Performance"** section:
  - GPU reference latency (from `vllm_gpu_perf.csv`)
  - Ascend Triton latency
  - Ascend/GPU ratio
- `summary.json` will contain extended fields:
  - `gpu_mode: true`
  - `perf_data.gpu_reference_ms`
  - `perf_data.ascend_vs_gpu_ratio`
  - `per_shape_results[].gpu_reference_ms`
  - `per_shape_results[].ascend_vs_gpu_ratio`


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

### Evaluation Baseline

#### Triton

Please refer to [`benchmarks/BASELINE_latest.md`](benchmarks/BASELINE_latest.md) for Triton-related data.

#### AscendC

Please refer to [`benchmarks/BASELINE_latest.md`](benchmarks/BASELINE_latest.md) for AscendC-related data.

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
│   ├── NPUKernelBench/
│   │   └── level1/             # NPU KernelBench Level 1 test cases (31 tasks)
│   └── TritonNPUKernelBench/   # GPU Triton → NPU migration dataset
│       ├── {op_name}.pt        # Contains input_data and optional gpu_output
│       ├── {op_name}.py        # GPU Triton kernel source code
│       └── vllm_gpu_perf.csv   # GPU performance baseline data
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


## Single Case Multi-Shape Support

This framework supports defining multiple Shape configurations within a single operator case for batch verification and performance evaluation, suitable for scenarios requiring performance testing across different input scales.

### Input Specifications (Operator Description File)

#### Single Shape Format (Backward Compatible)

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

def get_inputs():
    """Return single input group as List[Tensor/...]"""
    return [torch.randn(128, 128, dtype=torch.float16)]

def get_init_inputs():
    """Return initialization parameter list"""
    return []
```

**Specifications**:
- `get_inputs()`: Returns `List[Tensor/...]` representing a single input group
- For single-shape scenarios only
- `get_init_inputs()`: Returns parameter list for `Model.__init__`

#### Multi-Shape Format

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x: torch.Tensor, approximate='none') -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=approximate)

# Multi-Shape configuration list
INPUT_CASES = [
    {'inputs': [{'dtype': 'float32', 'name': 'x', 'shape': [128, 128], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'none'}]},
    {'inputs': [{'dtype': 'float32', 'name': 'x', 'shape': [256, 256], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'tanh'}]},
    {'inputs': [{'dtype': 'float16', 'name': 'x', 'shape': [1024, 1024], 'type': 'tensor'},
                 {'dtype': 'str', 'name': 'approximate', 'type': 'attr', 'value': 'none'}]},
]

# Required: returns List[List[Tensor/...]]
def get_input_groups():
    """Return multiple input groups, each corresponding to a Shape configuration"""
    input_groups = []
    for case in INPUT_CASES:
        group = []
        for spec in case['inputs']:
            if spec['type'] == 'tensor':
                dtype = {'float16': torch.float16, 'float32': torch.float32}[spec['dtype']]
                group.append(torch.randn(*spec['shape'], dtype=dtype))
            elif spec['type'] == 'attr':
                group.append(spec['value'])
        input_groups.append(group)
    return input_groups

# Optional for backward compatibility
def get_inputs():
    """Return single input group, using the first group"""
    return get_input_groups()[0]

def get_init_inputs():
    """Return initialization parameter list"""
    return []
```

**Input Specifications**:

| Function | Return Type | Purpose | Required |
|----------|-------------|---------|----------|
| `get_input_groups()` | `List[List[Tensor/...]]` | Multi-Shape entry, each group for a test configuration | ✅ Required for multi-shape |
| `get_inputs()` | `List[Tensor/...]` | Single-Shape entry, returns first or single group | Recommended (backward compatible) |
| `get_init_inputs()` | `List[Any]` | Initialization parameters for `Model.__init__` | ✅ Required |

**Input Configuration Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `dtype` | `str` | Data type: float16/float32/float64/bfloat16/int8/int16/int32/int64/bool |
| `shape` | `List[int]` | Tensor shape, e.g., `[128, 256]` |
| `name` | `str` | Parameter name |
| `type` | `str` | Type: "tensor", "attr" (attribute value), or "tensor_list" |
| `value` | `Any` | Attribute value when `type="attr"` |

### Output Specifications (Performance Report)

#### Single Shape Performance Report

```json
{
  "op_name": "gelu",
  "warmup": 5,
  "repeats": 50,
  "total_cases": 1,
  "framework": {
    "avg_latency_ms": 0.2345,
    "peak_memory_mb": 2.50
  },
  "implementation": {
    "avg_latency_ms": 0.1567,
    "peak_memory_mb": 1.25
  },
  "speedup_vs_torch": 1.5000,
  "perf_method": "profiler",
  "skill_path": "/path/to/.claude/skills/kernel-verifier"
}
```

#### Multi-Shape Performance Report

```json
{
  "op_name": "gelu",
  "warmup": 5,
  "repeats": 50,
  "total_cases": 3,
  "framework": {
    "avg_latency_ms": 0.4567,
    "peak_memory_mb": 8.50
  },
  "implementation": {
    "avg_latency_ms": 0.3123,
    "peak_memory_mb": 4.25
  },
  "speedup_vs_torch": 1.4600,
  "perf_method": "profiler",
  "skill_path": "/path/to/.claude/skills/kernel-verifier",
  "per_shape_results": [
    {
      "shape": [128, 128],
      "framework": {
        "avg_latency_ms": 0.0234,
        "peak_memory_mb": 0.50
      },
      "implementation": {
        "avg_latency_ms": 0.0156,
        "peak_memory_mb": 0.25
      },
      "speedup_vs_torch": 1.5000
    },
    {
      "shape": [256, 256],
      "framework": {
        "avg_latency_ms": 0.0891,
        "peak_memory_mb": 2.00
      },
      "implementation": {
        "avg_latency_ms": 0.0588,
        "peak_memory_mb": 1.00
      },
      "speedup_vs_torch": 1.5200
    },
    {
      "shape": [1024, 1024],
      "framework": {
        "avg_latency_ms": 1.2577,
        "peak_memory_mb": 8.00
      },
      "implementation": {
        "avg_latency_ms": 0.8625,
        "peak_memory_mb": 12.50
      },
      "speedup_vs_torch": 1.4600
    }
  ]
}
```

**Output Field Description**:

| Field | Type | Description |
|-------|------|-------------|
| `op_name` | `str` | Operator name |
| `warmup` | `int` | Warmup iterations |
| `repeats` | `int` | Test iterations |
| `total_cases` | `int` | Number of Shape tests (1 for single, ≥2 for multi) |
| `framework.avg_latency_ms` | `float` | PyTorch average latency (ms), average across all Shapes |
| `framework.peak_memory_mb` | `float` | PyTorch peak memory (MB), average across all Shapes |
| `implementation.avg_latency_ms` | `float` | Implementation average latency (ms), average across all Shapes |
| `implementation.peak_memory_mb` | `float` | Implementation peak memory (MB), average across all Shapes |
| `speedup_vs_torch` | `float\|null` | Geometric mean speedup `(∏ s_i)^(1/n)` over PyTorch (across pass shapes with finite positive `s_i`); `null` when all shapes are abnormal |
| `perf_method` | `str` | Profiling method: "profiler" (torch_npu.profiler) or "fallback" (time.perf_counter) |
| `skill_path` | `str` | Path to the benchmark skill used |
| `per_shape_results` | `List[Dict]` | Multi-Shape details (present when `total_cases > 1`) |

**per_shape_results Elements**:

| Field | Type | Description |
|-------|------|-------------|
| `shape` | `List[int]` | Main input tensor shape |
| `framework.avg_latency_ms` | `float` | PyTorch latency for this Shape |
| `implementation.avg_latency_ms` | `float` | Implementation latency for this Shape |
| `speedup_vs_torch` | `float` | Speedup for this Shape |

### Applicable Scenarios

1. **Operator Generalization Testing**: Verify correctness and stability of generated Triton operators across various input scales
2. **Performance Trend Analysis**: Identify operator advantages and limitations by comparing speedups across different Shapes
3. **AI Model Scenario Reproduction**: Simulate typical input Shape distributions in real models (e.g., multiple sequence lengths in LLMs)
4. **Automated Benchmark Evaluation**: Automatically cover multiple Shapes during batch evaluation to reduce repetitive work

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
