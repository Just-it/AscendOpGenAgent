---
name: a5-perf-summary
description: >
  A5 性能汇总 Skill。对指定 op 运行 performance.py，按 case 计算 ratio（ref_mean / asc_mean），
  汇总 best/worst/mean 及 >=0.6x、>=1.0x 统计，更新 A5_RESULTS.md Summary Table。
argument-hint: >
  输入：problem ID（如 1_GELU）、op 目录路径、或 "all" 扫描所有已完成 op。
  示例：/a5-perf-summary 1_GELU  |  /a5-perf-summary all  |  /a5-perf-summary output/npukernelbench/1_GELU
---

# A5 性能汇总 Skill

你的任务是对已完成的 AscendC 算子运行性能测试，计算 per-case ratio，并更新 A5_RESULTS.md。

## 核心脚本

`utils/perf_ratio.py` — 封装了全部逻辑，直接调用即可。

## 流程

### Step 1: 解析参数

```
arg = 用户输入

if arg == "all":
    OP_DIR = "output/npukernelbench"    # 批量扫描
elif arg matches /^\d+_\w+$/:          # e.g. "1_GELU"
    OP_DIR = "output/npukernelbench/{arg}"
else:
    OP_DIR = arg                        # 直接路径
```

### Step 2: 运行 perf_ratio.py

```bash
python3 utils/perf_ratio.py {OP_DIR} --update-results
```

这一条命令会：
1. 对每个 op 执行 `performance.py`（warmup=5, repeat=10）
2. 计算 per-case ratio = `ref_case_mean / asc_case_mean`
3. 汇总 best / worst / mean / >=0.6x / >=1.0x
4. 打印详细的 per-case 表格
5. 自动更新 `benchmarks/NPUKernelBench/A5_RESULTS.md` 中对应行

### Step 3: 输出结果

将 perf_ratio.py 的摘要输出展示给用户，包括：
- 每个 case 的 ref_mean / asc_mean / ratio
- 汇总统计：best, worst, mean, >=0.6x count, >=1.0x count
- A5_RESULTS.md 更新状态

## 高级用法

```bash
# 只打印，不更新 A5_RESULTS.md
python3 utils/perf_ratio.py output/npukernelbench/1_GELU

# 自定义 warmup/repeat
python3 utils/perf_ratio.py output/npukernelbench/1_GELU --update-results --warmup 10 --repeat 20

# 输出 JSON 格式
python3 utils/perf_ratio.py output/npukernelbench/1_GELU --json
```

## 口径说明

- **ratio**: `ref_case_mean_ms / asc_case_mean_ms`（每个 case 各自的 mean latency 之比）
- **Mean**: `mean(ratio_i)`（所有 case 的 ratio 取等权平均，非 time-weighted）
- **Best**: `max(ratio_i)`
- **Worst**: `min(ratio_i)`
- 与参考 RESULTS.md 口径一致
