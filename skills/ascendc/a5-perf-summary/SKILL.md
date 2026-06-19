---
name: a5-perf-summary
description: >
  A5 性能汇总 Skill。对指定 op 验证正确性、检测 kernel 模式（SIMT/SIMD）、运行 performance.py 计算 ratio，
  汇总 best/worst/mean 及 >=0.6x、>=1.0x 统计，更新 A5_RESULTS.md Summary Table。
argument-hint: >
  输入：problem ID（如 1_GELU）、op 目录路径、或 "all" 扫描所有已完成 op。
  示例：/a5-perf-summary 1_GELU  |  /a5-perf-summary all  |  /a5-perf-summary output/npukernelbench/1_GELU
---

# A5 性能汇总 Skill

你的任务是对已完成的 AscendC 算子验证正确性、检测编程模式、运行性能测试，并更新 A5_RESULTS.md。

## 核心脚本

- `utils/verification_ascendc.py` — 精度验证
- `utils/performance.py` — 性能测试
- `utils/perf_ratio.py` — ratio 计算 + A5_RESULTS.md 更新

## 流程

### Step 1: 解析参数

```
arg = 用户输入

if arg == "all":
    扫描 output/npukernelbench/ 下所有包含 model_new_ascendc.py 的子目录
    对每个子目录依次执行 Step 2 ~ Step 5
elif arg matches /^\d+_\w+$/:          # e.g. "1_GELU"
    OP_DIR = "output/npukernelbench/{arg}"
else:
    OP_DIR = arg                        # 直接路径

从 OP_DIR 提取 PROBLEM_ID（如 "1_GELU"）和 OP_NAME（如 "gelu"）。
```

### Step 2: 验证正确性

在运行性能测试之前，必须先确认精度。

```bash
python3 utils/verification_ascendc.py {OP_DIR}
```

**解析输出**:
- 统计总 case 数和 PASS 数
- 如果全部 PASS：记录 `PRECISION = "PASS {N}/{N}"`，继续 Step 3
- 如果有 FAIL：记录 `PRECISION = "FAIL {pass}/{total}"`，打印失败 case 详情
  - **仍然继续** Step 3~5（填表时标记为 FAIL），不中断流程

### Step 3: 检测 Kernel 模式（SIMT / SIMD）

检查 kernel 源文件判断编程模式：

```bash
# 查找 kernel header 文件
KERNEL_H=$(find {OP_DIR}/kernel/ -name "*_kernel.h" | head -1)
```

**判断规则**（按优先级）：

1. `grep -c "__simt_vf__\|Simt::VF_CALL\|Simt::GetThreadIdx\|VF_CALL" {KERNEL_H}`
   - 命中 > 0 → `MODE = "SIMT"`
2. `grep -c "TQue\|TPipe\|DataCopy\|EnQue\|DeQue" {KERNEL_H}`
   - 命中 > 0 → `MODE = "SIMD"`
3. 两者都命中 → `MODE = "SIMT+SIMD"`（混合模式）
4. 都不命中 → `MODE = "Unknown"`

### Step 4: 运行性能测试 + 计算 Ratio

```bash
python3 utils/perf_ratio.py {OP_DIR} --update-results
```

这一条命令会：
1. 对 op 执行 `performance.py`（warmup=5, repeat=10）
2. 计算 per-case ratio = `ref_case_mean / asc_case_mean`
3. 汇总 best / worst / mean / >=0.6x / >=1.0x
4. 打印详细的 per-case 表格
5. 自动更新 `benchmarks/NPUKernelBench/A5_RESULTS.md` 中对应的性能列

### Step 5: 更新 A5_RESULTS.md（补充 Mode 和 Precision 列）

`perf_ratio.py` 只更新性能数据。你还需要手动更新 **Mode** 和 **Precision** 列：

读取 `benchmarks/NPUKernelBench/A5_RESULTS.md`，找到 PROBLEM_ID 对应的行，用 Edit 工具更新：

- **Mode** 列：填入 Step 3 检测到的 `SIMT` / `SIMD` / `SIMT+SIMD`
- **Precision** 列：填入 Step 2 的结果，格式为 `**PASS** N/N` 或 `**FAIL** pass/total`
- **Status** 列：
  - 精度全 PASS 且 mean ≥ 0.6x → `Verified`
  - 精度全 PASS 但 mean < 0.6x → `Verified ⚠️`
  - 精度有 FAIL → `Precision FAIL`

> **特殊标注规则**：若算子精度未通过，需在 **Precision** 列和 **Status** 列均添加醒目标注 `❌`，例如 `❌ **FAIL** 32/46` 和 `❌ Precision FAIL`，以便一眼识别问题算子。

### Step 6: 输出结果

将汇总展示给用户：

```
## {PROBLEM_ID} Summary

Precision: {PASS N/N | FAIL pass/total}
Mode:      {SIMT | SIMD | SIMT+SIMD}
Best:      {X.XXx}
Worst:     {X.XXx}
Mean:      {X.XXx}
≥0.6x:    {N}/{total}
≥1.0x:    {N}/{total}
Status:    {Verified | Verified ⚠️ | Precision FAIL}

A5_RESULTS.md updated ✓
```

## 高级用法

```bash
# 只打印，不更新 A5_RESULTS.md
python3 utils/perf_ratio.py output/npukernelbench/1_GELU

# 自定义 warmup/repeat
python3 utils/perf_ratio.py output/npukernelbench/1_GELU --update-results --warmup 10 --repeat 20

# 输出 JSON 格式
python3 utils/perf_ratio.py output/npukernelbench/1_GELU --json

# 只验证精度不跑性能
python3 utils/verification_ascendc.py output/npukernelbench/1_GELU
```

## 口径说明

- **ratio**: `ref_case_mean_ms / asc_case_mean_ms`（每个 case 各自的 mean latency 之比）
- **Mean**: `mean(ratio_i)`（所有 case 的 ratio 取等权平均，非 time-weighted）
- **Best**: `max(ratio_i)`
- **Worst**: `min(ratio_i)`
- 与参考 RESULTS.md 口径一致

## SIMT / SIMD 判断参考

| 模式 | 关键特征 | 典型 API |
|------|---------|---------|
| SIMT | 线程级并行，每线程处理单个元素 | `__simt_vf__`, `Simt::VF_CALL`, `Simt::GetThreadIdx`, `__gm__` 直接读写 |
| SIMD | 流水线并行，DMA + VEC + 写回 | `TQue`, `TPipe`, `DataCopy`, `EnQue`/`DeQue`, `TBuf` |
| SIMT+SIMD | 混合：SIMD 搬数据到 UB，SIMT 做不规则计算 | 同时包含 TQue 和 VF_CALL |
