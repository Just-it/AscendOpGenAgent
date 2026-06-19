---
name: ascendc-profiling
description: 读取 msprof op_summary_*.csv，诊断 Vector/Cube 算子性能瓶颈，
             输出结构化诊断 JSON 和 profiling_one_liner，供世界模型节点和子 agent 使用
---

## What I do

自动读取 msprof 产出的 `op_summary_*.csv`，提取 AIV/AIC 流水线各阶段占比，按优先级判断性能瓶颈类型（`scalar_bound` / `memory_bound` / `compute_bound` / `icache_bound` / `output_bound` / `balanced`），输出结构化 JSON 和单行摘要（`profiling_one_liner`），供世界模型节点更新和子 agent prompt 注入使用。

## When to use

- **lingxi-evo 步骤 4.4.P**（在 Refine 完成、Strategy Discovery 之前）：对每个精度通过的节点调用，将 `bottleneck` 和 `profiling_one_liner` 写入世界模型节点的 `profiling_insight` 字段
- **lingxi-partial 阶段5 步骤A-pre**（Local Refinement 内层循环之前）：调用一次，将诊断结果作为步骤A生成新内核的核心参考

## 调用方式

### 从 profiling 目录自动搜索

```bash
python3 .claude/skills/ascendc-profiling/scripts/analyze_profiling.py \
    <profiling_dir> \
    --task-type vector \
    --output result.json
```

脚本自动搜索 CSV 文件，按以下优先级：
1. `profiling_dir/ModelNew_*/PROF_*/mindstudio_profiler_output/op_summary_*.csv` (msprof native)
2. `profiling_dir/*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_summary_*.csv` (torch_npu.profiler)
3. 递归搜索 `op_summary_*.csv`
4. **Fallback**: `kernel_details.csv` (torch_npu.profiler 有时只产出此格式)

### 直接指定 CSV 文件

```bash
python3 .claude/skills/ascendc-profiling/scripts/analyze_profiling.py \
    --op-summary /path/to/op_summary_xxx.csv \
    --task-type vector \
    --output result.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `profiling_dir` | (与 --op-summary 二选一) | profiling 根目录 |
| `--op-summary` | None | 直接指定 CSV 文件路径 |
| `--task-type` | `vector` | `vector` / `cube` / `cv-mix` |
| `--output` | stdout | 输出 JSON 路径 |

## 输出字段说明

```json
{
  "status": "success",
  "task_type": "vector",
  "task_duration_us": 5.36,
  "pipeline": {
    "mte2_pct": 45.0,
    "vec_pct": 37.0,
    "mte3_pct": 18.0,
    "scalar_pct": 12.0,
    "icache_miss_rate": 0.05
  },
  "bottleneck": "memory_bound",
  "bottleneck_description": "MTE2 搬入占 45.0%，带宽是主要瓶颈",
  "recommended_strategies": ["P1", "P2", "P10", "P25", "P66"],
  "optimization_hints": [
    "增大 tile_size 减少分块次数，降低 MTE2 调用频率",
    "启用双缓冲 (P1)，隐藏搬运延迟",
    "向量化数据加载 (P10)，提升带宽利用率",
    "检查 DataCopy 对齐与偏移 (P25/P66)，避免非对齐带来的带宽折损"
  ],
  "profiling_one_liner": "MTE2:45.0% | Vec:37.0% | MTE3:18.0% | Scalar:12.0% | 瓶颈:memory_bound → 推荐 P1,P2,P10,P25,P66"
}
```

失败时：
```json
{
  "status": "error",
  "error": "No op_summary CSV found under: ./profiling",
  "profiling_one_liner": ""
}
```

## 瓶颈诊断规则

> 策略 ID 取自 `.claude/skills/evolution-strategies/references/strategy-index.md`（当前覆盖 D1-D5、P1-P88、A1-A8）。
> 推荐集保持"少而准"：首列给出最通用的 L0 策略，再按需补充 L1 精修策略。

### Vector 算子（aiv_* 列，按优先级判断）

| 条件 | 瓶颈类型 | 推荐策略 | 说明 |
|------|---------|---------|------|
| `aiv_scalar_ratio > 0.40` | `scalar_bound` | P5, P10, P67, P84 | Counter 模式 / 低延迟归约替代标量循环 |
| `aiv_mte2_ratio > 0.45` | `memory_bound` | P1, P2, P10, P25, P66 | 双缓冲 + 增大 tile + 对齐搬运 |
| `aiv_vec_ratio > 0.60` | `compute_bound` | P3, P4, P13, P68, P69, P84 | UB 融合链 / 低延迟归约消除中间 GM 往返 |
| `aiv_icache_miss_rate > 0.10` | `icache_bound` | P8, P54 | 精简内核 + 减少 kernel launch |
| `aiv_mte3_ratio > 0.35` | `output_bound` | P1, P10, P56, P59 | 批量写出 + 输出转置融合 |
| 其余 | `balanced` | P4, P7, P65 | 负载均衡 + 对齐 + UB bank 冲突规避 |

注：`aiv_*_ratio` 列为 0–1 之间的浮点数（不是百分比），直接与阈值比较。

### Cube 算子（aic_* 列）

| 条件 | 瓶颈类型 | 推荐策略 | 说明 |
|------|---------|---------|------|
| `cube_utilization(%) < 50` | `memory_bound` | P2, P4, P19, P46, P52, P71 | IBShare L1 共享 + L2 hint + MatmulImpl |
| `aic_mac_ratio > 0.70` | `compute_bound` | P13, P2, P47, P72, P78 | 对角分块 / Split-K / 片上缓存加速 |
| 其余 | `balanced` | P46, P63 | MatmulImpl + 异步迭代隐藏延迟 |

> 推荐集是"候选清单"，上游 agent 应结合算子类别（cv_fusion / flash_attention / quantization / matmul）从 strategy-index 的 Tags 字段进一步筛选。

## 兜底策略

- CSV 文件缺失 → `status: "error"`，`profiling_one_liner: ""`
- 列名不存在或值为空 → 该列按 0.0 处理，不抛异常
- 所有 AI_CPU 行被过滤后无剩余行 → `status: "error"`
- **kernel_details.csv 格式** → 返回 `csv_format: "kernel_details"`，`bottleneck: "balanced"`，建议使用深度 profiling 获取精确诊断
- **列名兼容**: 自动处理 `Task Duration(us)` / `Duration(us)` 和 `Task Type` / `Type` 两种列名格式
- **调用方检查 `status` 字段，失败时静默跳过，不阻断主流程**
