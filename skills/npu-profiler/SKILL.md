# NPU Profiler Skill

## 功能

使用 `torch_npu.profiler` 对算子进行详细性能分析，生成 profiling 报告并提取关键指标。

## 指标

- `device_self_duration_us`: 算子在 Device 侧的耗时（除去内部调用的其他算子），单位 us
- `device_total_duration_us`: 算子在 Device 侧的耗时，单位 us
- `operator_details`: 每个算子的详细统计，包括平均 Duration

## 用法

```python
# 执行 performance-test.py 脚本
python3 <npu-profiler路径>/scripts/performance-test.py \
    --op_name <算子名> \
    --verify_dir <验证目录> \
    --num_iterations <迭代次数> \
    --output <输出文件路径>
```

## 输出

生成包含以下字段的 JSON 文件：
```json
{
  "profiling": {
    "framework": {
      "op_name": "...",
      "device_self_duration_us": [...],
      "device_total_duration_us": [...],
      "operator_details": [
        {"name": "...", "avg_device_self_duration_us": xxx, "avg_device_total_duration_us": xxx}
      ]
    },
    "implementation": {
      "op_name": "...",
      "device_self_duration_us": [...],
      "device_total_duration_us": [...],
      "operator_details": [...]
    }
  },
  "speedup": {
    "device_self_duration_speedup": 1.5,
    "device_total_duration_speedup": 1.8
  }
}
```
