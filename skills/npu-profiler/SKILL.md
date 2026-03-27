---
name: npu-profiler
description: >
  算子性能分析 Skill — 使用 torch_npu.profiler 对算子进行详细性能分析，
  生成 profiling 报告并提取 Device Self/Total Duration 等关键指标，
  支持框架实现与生成实现的性能对比。
argument-hint: >
  输入：op_name、verify_dir、num_iterations、warm_up。
  输出：包含 device_self_duration_us、device_total_duration_us、
  operator_details、speedup 等的详细性能分析报告。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# NPU Profiler Skill

## 功能

使用 `torch_npu.profiler` 对算子进行详细性能分析，生成 profiling 报告并提取关键指标。

## 指标

- `device_self_duration_us`: 算子在 Device 侧的耗时（除去内部调用的其他算子），单位 us
- `device_total_duration_us`: 算子在 Device 侧的耗时（包含内部调用的所有算子），单位 us
- `operator_details`: 按算子名称聚合的详细统计，包括：
  - `name`: 算子名称
  - `avg_device_self_duration_us`: 该算子的平均 Self Duration
  - `avg_device_total_duration_us`: 该算子的平均 Total Duration
- `speedup`: 生成实现相对于框架实现的加速比

## 用法

执行 `scripts/performance-test.py` 脚本：

```bash
python3 <npu-profiler路径>/scripts/performance-test.py \
    --op_name <算子名> \
    --verify_dir <验证目录> \
    --num_iterations <迭代次数> \
    --warm_up <warmup次数> \
    --output <输出文件路径>
```

**参数说明**：

| 参数 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `--op_name` | 是 | 算子名称 | - |
| `--verify_dir` | 否 | 验证目录路径 | 当前目录 |
| `--num_iterations` | 否 | 迭代次数 | 50 |
| `--warm_up` | 否 | warmup 次数 | 5 |
| `--output` | 否 | 输出文件路径 | 打印到 stdout |

## 输出

生成包含以下字段的 JSON 文件：

```json
{
  "op_name": "Softmax",
  "num_iterations": 50,
  "warm_up": 5,
  "timestamp": 1714473600,
  "profiling": {
    "framework": {
      "op_name": "Softmax_torch",
      "avg_device_self_duration_us": 12.3456,
      "avg_device_total_duration_us": 15.6789,
      "device_self_duration_us": [12.1, 12.3, ...],
      "device_total_duration_us": [15.2, 15.4, ...],
      "operator_details": [
        {"name": "Softmax", "avg_device_self_duration_us": 12.34, "avg_device_total_duration_us": 15.67},
        {"name": "Gelu", "avg_device_self_duration_us": 0.12, "avg_device_total_duration_us": 0.15}
      ]
    },
    "implementation": {
      "op_name": "Softmax_triton_ascend_impl",
      "avg_device_self_duration_us": 8.0000,
      "avg_device_total_duration_us": 9.5000,
      "device_self_duration_us": [7.8, 8.1, ...],
      "device_total_duration_us": [9.3, 9.7, ...],
      "operator_details": [...]
    }
  },
  "speedup": {
    "device_self_duration_speedup": 1.54,
    "device_total_duration_speedup": 1.65
  }
}
```

**字段说明**：

- `profiling.framework`: 框架实现（Torch）的详细性能数据
- `profiling.implementation`: 生成实现（Triton Ascend）的详细性能数据
- `speedup.device_self_duration_speedup`: 基于 Self Duration 的加速比
- `speedup.device_total_duration_speedup`: 基于 Total Duration 的加速比

## 数据来源

从 `torch_npu.profiler` 生成的 `operator_details.csv` 文件中提取数据，包含：
- `Name`: 算子名称
- `Device Self Duration(us)`: 算子在 Device 侧的实际计算耗时
- `Device Total Duration(us)`: 算子在 Device 侧的总耗时，包含调用的子算子

## 执行示例

```bash
python3 /path/to/npu-profiler/scripts/performance-test.py \
    --op_name Softmax \
    --verify_dir ./output/kernelgen-workflow_0/iter_0/verify \
    --num_iterations 50 \
    --warm_up 5 \
    --output ./profiling_result.json
```
