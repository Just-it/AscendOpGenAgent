---
name: ascendc-profiling-analysis
description: AscendC算子指令级流水空泡分析 — 基于 msprof simulator trace 的自动诊断与优化建议
---

## What I do

解析 msprof simulator 产出的 trace.json (Chrome Trace Format)，自动检测流水空泡并进行两级分类
（7 大类 / ~30 子类型），进行因果归因，生成可操作的优化建议。输出结构化 JSON 诊断报告。

## When to use me

- 算子性能不达预期，需要定位流水瓶颈
- 进化优化中需要 profiling 证据指导策略选择
- 修改代码后需要验证优化效果

## Lingxi Profiling 脚本

用于 lingxi-evo 进化优化中的 Profiling 数据采集：

| 脚本 | 说明 |
|------|------|
| `lingxi_profiling_collector.py` | CSV profiling 数据收集：对变体目录自动执行 build + profiling，输出结构化 CSV 数据 |
| `lingxi_profiling_runner.py` | ACL-native msprof simulator profiling：脱离 libtorch_npu 依赖，通过独立 .so 进行 msprof 仿真 |
| `build_standalone_kernel.sh` | 从 libkernels.a 构建独立 libkernel_standalone.so，供 lingxi_profiling_runner.py 使用 |

## 空泡分类体系

### 一级分类 (BubbleCategory)

| 类别 | 含义 | 可优化 |
|------|------|--------|
| NORMAL | 正常发射间距 | No |
| STRUCTURAL | 流水线结构性开销 (drain/barrier/icache) | No |
| DATA_STALL | 数据搬运等待 (MTE2/MTE3) | ★★ |
| SCALAR_OVERHEAD | 标量计算/加载阻塞 | ★ |
| RESOURCE_CONTENTION | 资源竞争 (UB/icache/bus) | △ |
| CROSS_CORE | 跨核不均衡 | ★ |
| CUBE_VECTOR | Cube-Vector 协同 | ★ |

### 二级分类 (BubbleSubType) — 关键子类型

| 子类型 | 一级分类 | 旧分类 | 含义 |
|--------|----------|--------|------|
| N_ISSUE_GAP | NORMAL | A | ≤1ps 正常发射间距 |
| S_DRAIN | STRUCTURAL | B | drain 指令后排空 |
| S_BARRIER | STRUCTURAL | B | PipeBarrier 同步 |
| S_COLD_START | STRUCTURAL | B | 首 tile 流水线填充 |
| S_TAIL_DRAIN | STRUCTURAL | B | 末 tile 流水线排空 |
| S_ICACHE_MISS | STRUCTURAL | B | 指令缓存未命中 |
| S_FLOWCTRL | STRUCTURAL | B | 流控指令开销 |
| D_MTE2_WAIT | DATA_STALL | D | VEC 等 MTE2 (显式 WAIT_FLAG) |
| D_MTE2_IMPLICIT | DATA_STALL | D | VEC 等 MTE2 (隐式) |
| D_MTE3_WAIT | DATA_STALL | D | VEC 等 MTE3 (显式) |
| D_MTE3_IMPLICIT | DATA_STALL | D | VEC 等 MTE3 (隐式) |
| D_NO_OVERLAP | DATA_STALL | D | 缺少双缓冲无法重叠 |
| D_PARTIAL_OVERLAP | DATA_STALL | D | 双缓冲重叠不充分 |
| SC_LDST_BLOCK | SCALAR_OVERHEAD | C | SCALARLDST 参数加载阻塞 |
| SC_COMPUTE_BLOCK | SCALAR_OVERHEAD | C | SCALAR 地址计算阻塞 |
| R_BUS_CONTENTION | RESOURCE_CONTENTION | B | MTE2+MTE3 总线竞争 |

### 向后兼容

旧四类 (A/B/C/D) 仍保留在 `BubbleClassification.bubble_type` 中，新分类通过 `category` 和 `sub_type` 字段访问。

## 工具清单

| 工具 | 脚本 | 功能 |
|------|------|------|
| T1 | `analyze_cross_core.py` | 跨核负载均衡分析 (含每核 pipeline 利用率，支持 veccore + cubecore) |
| T2 | `analyze_pipeline_bubbles.py` | D类空泡检测 (自动识别 veccore→VECTOR / cubecore→CUBE 主 pipeline) |
| T3 | `analyze_vec_internal.py` | 主 pipeline 内部空泡分析 (veccore: VEC / cubecore: CUBE) |
| T4 | `concurrent_pipeline_view.py` | 时间窗口并发状态快照 (含 cubecore MTE1/CUBE/FIXPIPE) |
| T5 | `check_feasibility.py` | 优化可行性硬件约束检查 |
| T6 | `diff_profiling.py` | 修改前后对比 (含 T7/T8 结果对比) |
| T7 | `pattern_detector.py` | 周期性模式检测 (支持 veccore + cubecore) |
| T8 | `overlap_analyzer.py` | 流水线重叠度分析 (veccore: VEC/MTE2/MTE3 / cubecore: CUBE/MTE1/MTE2/FIXPIPE) |
| T9 | `dma_efficiency.py` | DMA搬运效率分析 (veccore: MTE2/MTE3 / cubecore: MTE1/MTE2) |

### Cubecore 支持

所有分析工具自动检测核类型 (`is_cubecore(core_id)`)，针对不同核类型使用对应的 pipeline:

| 核类型 | 主 pipeline | DMA pipeline | 格式转换 |
|--------|-------------|-------------|----------|
| veccore | VECTOR (PID 30) | MTE2 (GM→UB), MTE3 (UB→GM) | - |
| cubecore | CUBE (PID 50) | MTE1 (L1→L0), MTE2 (GM→L1) | FIXPIPE (L0C→UB, PID 80) |

Cubecore PID 映射:
- PID 40 = MTE1 (L1→L0A/L0B 数据加载)
- PID 50 = CUBE (MatMul 计算)
- PID 80 = FIXPIPE (L0C→UB 格式转换)

## Workflow

执行此 skill 时，按以下步骤顺序执行。所有脚本位于 `.claude/skills/ascendc-profiling-analysis/scripts/`。

---

### Step 0: 确认 simulator profiling 数据

检查是否已有 simulator trace 数据:

```bash
ls output/{op_name}/simulator_prof/
```

**如果不存在**: 运行 profiling_runner.py 生成。该脚本会自动:
1. 从 `{op_name}_custom.py` 生成临时运行脚本
2. 调用 `msprof op simulator --application="python3 {script}" --output=...` 执行 simulator profiling
3. 返回生成的 simulator 目录路径

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/profiling_runner.py \
    --work-dir output/{op_name} \
    --op-name {op_name}
```

前置条件:
- `output/{op_name}/{op_name}_custom.py` 必须存在 (含 ModelNew, get_inputs, get_init_inputs)
- `output/{op_name}/pybind_lib/` 必须存在 (编译好的 .so)
- `output/{op_name}/vendors/customize/` 应存在 (部署好的自定义算子包)；脚本会自动设置 `ASCEND_CUSTOM_OPP_PATH` 和扩展 `LD_LIBRARY_PATH`
- CANN 环境已 source (msprof 在 PATH 中)
- 脚本会自动检测 `_custom.py` 中的 `permute().contiguous()` 调用，在 CPU 上执行 permute 以避免 msprof 捕获 Transpose 内核而非自定义算子内核
- 如果 `get_inputs`/`get_init_inputs` 不在 `_custom.py` 中，脚本会自动从 `_functional.py` 加载

可选参数:
- `--output-dir`: 自定义输出目录 (默认: work_dir/simulator_prof)
- `--soc-version`: 指定 SoC 版本 (如 Ascend910B3，默认自动检测)
- `--aic-metrics`: 采集指标 (默认 PipeUtilization)
- `--timeout`: 超时秒数 (默认 3600 即 1 小时；最大 21600 即 6 小时)
- `--test-case-csv`: test_cases.csv 路径；提供时使用其中的 shape 构造输入张量，替代 `get_inputs()`。自动选择总元素数最大的 case，确保 profiling 反映真实负载
- `--case-id`: 指定 CSV 中的 case_id（默认: 元素数最大的 case）

**如果已存在**: 找到最新的 simulator 目录:

```bash
# 找到包含 coreN.veccoreM/trace.json 的 simulator 目录
find output/{op_name}/simulator_prof -type d -name "simulator" | head -1
```

将此路径记为 `{simulator_dir}`。

---

### Step 1 (T1): 跨核均衡分析

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/analyze_cross_core.py \
    --simulator-dir {simulator_dir}
```

**决策逻辑**:
- `is_imbalanced = true` (ratio > 1.3 或 CV > 0.15):
  → 主瓶颈 = tiling 不均衡
  → 记录诊断: "tiling_imbalance"
  → 建议策略: P4 (负载均衡) + P2 (自适应分块)
  → 检查 `per_core_pipeline_utilization` 判断不均衡是 tiling 问题还是搬运效率问题
  → 跳到 Step 5 检查可行性

- `is_imbalanced = false`:
  → 记录 `slowest_core` 作为后续分析目标
  → 进入 Step 2

---

### Step 2 (T2): D 类空泡检测

使用 Step 1 输出的 `slowest_core`:

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/analyze_pipeline_bubbles.py \
    --simulator-dir {simulator_dir} \
    --core-id {slowest_core} \
    --top-n 5
```

**决策逻辑**:
- `d_class_pct > 50%`:
  → 主瓶颈 = 跨pipeline同步等待
  → 记录 `primary_bottleneck` (MTE2/MTE3)
  → 查看 `sub_type_breakdown` 确认具体子类型分布
  → 进入 Step 2a 和 Step 4

- `d_class_pct 30-50%`:
  → 混合瓶颈，同时进入 Step 2a、Step 3 和 Step 4

- `d_class_pct < 30%`:
  → D类不是主要问题
  → 进入 Step 3 看 VEC 内部

---

### Step 2a (T8): 流水线重叠度分析

在 T2 之后立即运行（可与 T3 并行）:

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/overlap_analyzer.py \
    --simulator-dir {simulator_dir} \
    --core-id {slowest_core}
```

**决策逻辑**:
- `overlap_status = "no_overlap"`:
  → 未启用双缓冲或双缓冲无效
  → 建议 P1 (双缓冲) 为首要优化
- `overlap_status = "partial_overlap"`:
  → 双缓冲已启用但效果不充分
  → 建议调整 tile 大小 (P2) 或 UB 分区 (P8)
- `overlap_status = "good_overlap"`:
  → 双缓冲效果良好，瓶颈在其他方面
- `mte2_mte3 overlap > 30%`:
  → 存在总线竞争，建议错开搬运时序

---

### Step 3 (T3): VEC 内部空泡分析

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/analyze_vec_internal.py \
    --simulator-dir {simulator_dir} \
    --core-id {slowest_core}
```

**决策逻辑**:
- `c_class.pct > 30%`:
  → 标量参数加载是瓶颈
  → 检查 `c_class_detail.primary_cause`:
    - "SCALARLDST" → 建议 P5 (参数预取)
    - "SCALAR" → 建议简化地址计算
  → 查看 `sub_type_breakdown` 确认子类型分布

- `pure_compute_pct > 70%`:
  → 计算密集型，空泡已很少
  → 建议算法级优化 (P13) 或混合精度 (D1)

- 均正常 (c_class < 30%, compute > 50%):
  → 内核已接近最优

---

### Step 3a (T9): DMA 搬运效率分析

在 T3 之后运行:

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/dma_efficiency.py \
    --simulator-dir {simulator_dir} \
    --core-id {slowest_core}
```

可选: 如果有 hw_params:
```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/dma_efficiency.py \
    --simulator-dir {simulator_dir} \
    --core-id {slowest_core} \
    --hw-params path/to/hw_params.json
```

**决策逻辑**:
- `mte2_stats.verdict = "undersize_transfers"`:
  → MTE2 搬运粒度过小，tile 太小
  → 建议 P2 (增大 tile) + P10 (向量化搬运)
- `mte3_stats.verdict = "undersize_transfers"`:
  → MTE3 搬运粒度过小
  → 同上
- `bandwidth_utilization.estimated_pct < 30%`:
  → 带宽利用率低，需优化搬运策略

---

### Step 4 (T4): 因果链调查

对 Step 2 输出的 Top 空泡，调用 T4 查看并发状态:

```bash
# 对每个 Top D类空泡的时间窗口
python3 .claude/skills/ascendc-profiling-analysis/scripts/concurrent_pipeline_view.py \
    --simulator-dir {simulator_dir} \
    --core-id {slowest_core} \
    --start-us {bubble_start_us} \
    --end-us {bubble_end_us}
```

**分析要点**:
- VEC idle 时谁在 busy? → 确认根因
- MTE2 busy 做什么操作? → 判断是否可通过双缓冲重叠
- 多个 pipeline 同时 busy? → 可能是资源竞争
- CACHEMISS active? → 指令缓存未命中
- FLOWCTRL active? → 流控开销
- 查看各 pipeline 的 `coverage_pct` 量化忙碌程度

---

### Step 4a (T7): 周期性模式检测

在 T4 因果链调查后运行:

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/pattern_detector.py \
    --simulator-dir {simulator_dir} \
    --core-id {slowest_core}
```

**决策逻辑**:
- `pattern_type = "periodic"`:
  → 空泡是系统性的，每个 tile 重复出现
  → 对应优化可系统性消除（如双缓冲消除周期性 MTE2 等待）
- `pattern_type = "sporadic"`:
  → 空泡是偶发的，非主要瓶颈
- `pattern_type = "cold_start_dominant"`:
  → 冷启动空泡占比高，可通过预取首 tile 数据缓解
- `pattern_type = "tail_dominant"`:
  → 尾部空泡占比高，检查尾块处理逻辑

---

### Step 5 (T5): 优化可行性检查

根据前面步骤的诊断结果，验证提出的优化方案:

**双缓冲可行性** (当建议 P1 时):
```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/check_feasibility.py \
    --type add_double_buffer \
    --params '{"tile": {current_tile}, "dtype": "{dtype}", "pipe_count": 2}'
```

**增大 tile 可行性** (当建议增大 tile 时):
```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/check_feasibility.py \
    --type increase_tile \
    --params '{"current_tile": {current}, "target_tile": {target}, "dtype": "{dtype}", "buffer_num": 2, "pipe_count": 2}'
```

如果有 `hw_params` (从世界模型或 evaluation_results.json):
```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/check_feasibility.py \
    --type add_double_buffer \
    --params '{"tile": 4096, "dtype": "fp16"}' \
    --hw-params path/to/hw_params.json
```

---

### Step 6: 综合诊断报告

汇总 Step 1-5 + T7/T8/T9 的结果，生成 `profiling_diagnosis.json`:

```json
{
  "op_name": "{op_name}",
  "simulator_dir": "{simulator_dir}",
  "diagnosis": {
    "primary_bottleneck": "mte2_stall | mte3_stall | tiling_imbalance | scalar_loading | compute_bound | near_optimal | no_overlap | partial_overlap | undersize_transfer | icache_miss | bus_contention",
    "t1_summary": { "imbalance_ratio": 1.02, "is_imbalanced": false },
    "t2_summary": { "d_class_pct": 56.0, "primary_bottleneck": "MTE2", "sub_type_breakdown": {} },
    "t3_summary": { "c_class_pct": 8.1, "pure_compute_pct": 2.7, "sub_type_breakdown": {} },
    "t7_summary": { "pattern_type": "periodic", "dominant_subtype": "d_mte2_wait" },
    "t8_summary": { "overlap_status": "no_overlap", "vec_mte2_overlap_pct": 5.2 },
    "t9_summary": { "mte2_short_pct": 35.0, "mte3_short_pct": 20.0 },
    "causal_chain": "VEC idle → MTE2 busy (data load) → no overlap detected → need double buffering"
  },
  "recommendations": [
    {
      "priority": 1,
      "strategy": "P1",
      "action": "Enable double buffering to overlap MTE2 data load with VEC compute",
      "feasible": true,
      "ub_utilization_pct": 50.0
    }
  ],
  "profiling_analysis": {
    "imbalance_ratio": 1.02,
    "d_class_pct": 56.0,
    "c_class_pct": 8.1,
    "primary_bottleneck": "mte2_stall",
    "top_recommendation": "Enable double buffering (P1)",
    "pattern_type": "periodic",
    "overlap_status": "no_overlap",
    "dominant_subtype": "d_mte2_wait",
    "dma_efficiency": {
      "mte2_short_pct": 35.0,
      "mte3_short_pct": 20.0
    }
  }
}
```

将 `profiling_analysis` 子字段写入 `evaluation_results.json` (如果存在)。

将完整报告保存到:
```
output/{op_name}/profiling_diagnosis.json
```

---

### Step 7 (T6): [可选] 修改后对比验证

当用户修改代码并重新 profiling 后，对比前后效果:

```bash
python3 .claude/skills/ascendc-profiling-analysis/scripts/diff_profiling.py \
    --before-dir {before_simulator_dir} \
    --after-dir {after_simulator_dir}
```

**判定标准**:
- `time_improvement_pct > 10%` → significant_improvement
- `d_class_reduction_pct > 20%` → D类空泡显著减少
- `verdict = "structural_improvement"` → 时间未显著变化但重叠度/模式改善
- `verdict = "significant_regression"` → 优化方向错误，需回退
- `overlap_change.improved = true` → 流水线重叠度改善
- `pattern_change.improved = true` → 周期性空泡模式消除

---

## 与进化系统集成

当此 skill 在 lingxi-evo 上下文中使用时:

1. 评估通过后自动运行 T1+T2+T3+T8+T9
2. 将 `profiling_analysis` (含 pattern_type, overlap_status, dominant_subtype, dma_efficiency) 写入 `evaluation_results.json`
3. lingxi-evo Refine 步骤读取 `profiling_analysis`
4. 调用 `.claude/skills/evolution-world-model/scripts/profiling_evidence.py` 的 `extract_profiling_evidence()`
5. 将 `suggested_strategies` 注入子节点的 `strategy_combination`

## Key Notes

1. 所有时间单位: trace.json 中为 **皮秒 (ps)**，输出报告中转换为 **微秒 (us)** 或 **纳秒 (ns)**
2. PID 映射: 10=SCALAR, 20=SCALARLDST, 30=VECTOR, 40=MTE1, 50=CUBE, 60=MTE2, 70=MTE3, 80=FIXPIPE, 90=FLOWCTRL, 110=CACHEMISS
3. 只分析 `ph="X"` 的 duration 事件，跳过 metadata (`ph="M"`)
4. 32B 对齐是硬约束，T5 会自动检查
5. 脚本间通过 `sys.path.insert` 共享 `trace_parser.py` 和 `bubble_classifier.py`
6. 空泡分类向后兼容: 旧 A/B/C/D 四类通过 `bubble_type` 字段保留，新分类通过 `category` + `sub_type` 访问
