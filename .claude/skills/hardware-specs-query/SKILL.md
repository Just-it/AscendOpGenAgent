---
name: hardware-specs-query
description: 查询目标昇腾芯片的硬件规格参数（UB大小、核数、带宽、向量宽度等），
             计算 derived_params，为进化流程的世界模型初始化提供硬件约束依据。
             输出结果写入 world_model.json 的 hw_params 字段。
---

# System Prompt

查询目标昇腾芯片的硬件规格，计算 derived_params，为进化优化的 Tiling 决策和 Roofline 分析提供基础。

## 执行纪律

### 语言约束
- **所有思考、分析、推理、解释必须使用中文**
- **代码、技术标识符、JSON键、文件路径、技术术语（如UB、GM、TFLOPS）除外**

### 精确查询原则
- **必须根据芯片型号选择正确的JSON文件**
- **必须区分shared通用参数和variants特有参数**
- **必须计算derived_params（带宽、算力）**

### 检查点机制
查询完成后必须：
1. 验证返回的参数完整性
2. 标注精确值vs估算值

## Workflow

### Step 1: 芯片检测

若调用方未指定芯片型号，自动检测：

```bash
npu-smi info 2>/dev/null | grep -E "^\| [0-9]+" | head -1
```

从输出中提取芯片型号（如 910B3、910B4、310P3 等）。

---

### Step 2: 查询规格参数

根据芯片型号前缀，读取对应的芯片规格 JSON 文件：

| 芯片型号前缀 | JSON 文件 |
|-------------|---------|
| 910B / 910B1~B4 | `references/chips/ascend_910b.json` |
| 910A / 910PremiumA / 910ProA/B | `references/chips/ascend_910.json` |
| 910_93xx | `references/chips/ascend_910_93xx.json` |
| 910_95xx / 910_950z | `references/chips/ascend_910_95xx.json` |
| 910_55xx | `references/chips/ascend_910_55xx.json` |
| 310B1~B4 | `references/chips/ascend_310b.json` |
| 310P1/P3/P5/P7 | `references/chips/ascend_310p.json` |
| 610 / 610Lite | `references/chips/ascend_610.json` |
| BS9SX* | `references/chips/bs9.json` |
| Kirin* | `references/chips/kirin.json` |
| Hi3796* | `references/chips/hi3796.json` |
| SD3403 | `references/chips/sd3403.json` |
| Ascend031 | `references/chips/ascend_031.json` |
| Ascend035* | `references/chips/ascend_035.json` |
| AS31XM1X | `references/chips/as31xm1x.json` |
| MC6* | `references/chips/mc6x.json` |

**参数来源**:
1. 从 `shared` 部分获取通用参数（UB、对齐、memory_rates、流水线）
2. 从 `variants[检测到的型号]` 获取型号特有参数（核数、频率、UB/L2大小、GM容量）

---

### Step 3: 计算 derived_params

使用芯片 JSON 中的参数计算：

**带宽计算**:
```
peak_bw_gbps = ddr_rate (Bytes/cycle) × freq_mhz / 1000
```

**Vector 算力计算**:
```
peak_vector_tflops_per_core = vec_calc_size / dtype_size × freq_mhz × 2(FMA) / 1e6
```

**Cube 算力计算**:
```
peak_cube_tflops_per_core = cube_m × cube_n × cube_k × 2 × freq_mhz / 1e6
```

**Tiling 参数**:
```
max_tile_fp16_double_buf = (ub_size // (2 × 3 × 2)) // 16 × 16   # 双缓冲·3级流水·fp16·32B对齐
```

计算说明：
- `ddr_rate`：来自 JSON 的 `memory_rates.ddr_rate`（单位：Bytes/cycle）
- `freq_mhz`：来自 JSON 的 `variants[型号].cube_freq_mhz`（单位：MHz）
- `vec_calc_size`：来自 JSON 的 `shared.vector_unit.vec_calc_size_bytes`（单位：Bytes）
- `cube_m/n/k`：来自 JSON 的 `shared.cube_unit.cube_m/n/k_size`
- `ub_size`：来自 JSON 的 `variants[型号].ub_size_bytes`（单位：Bytes）

**910_95xx 系列特殊处理**:
910_95xx 系列 CubeCore 和 VectorCore 有独立的 ddr_rate：
- CubeCore 带宽：使用 `variants[型号].ddr_rate_cube`
- VectorCore 带宽：使用 `variants[型号].ddr_rate_vector`（约为 CubeCore 的一半）

对于 Vector 算子优化场景，应使用 `ddr_rate_vector` 计算带宽上界。

---

### Step 4: 输出 hw_params

返回以下结构化参数，供调用方写入 `world_model.json` 的 `hw_params` 字段：

```json
{
  "chip_model": "910B3",
  "ub_size_bytes": 196608,
  "core_num": 40,
  "peak_bw_gbps": 57.6,
  "peak_vector_tflops_per_core": 0.2304,
  "peak_cube_tflops_per_core": 14.746,
  "alignment_bytes": 32,
  "max_tile_fp16_double_buf": 16384
}
```

同时输出 `hw_params_one_liner`（供子agent prompt注入使用）：

```
Chip: 910B3 | UB: 192KB | Cores: 40 | Peak BW: 57.6GB/s | Max tile(FP16,2buf): 16384 elems
```

## 输入规范

```yaml
chip_model: string|null               # 芯片型号，如"910B3"，不填则自动检测
query_type: string|null               # 查询类别
  # full: 返回全部规格（默认）
  # memory_hierarchy: 内存层次参数
  # compute_capability: 计算能力参数
  # roofline_params: Roofline建模所需参数
  # alignment: 对齐要求
```

## 输出规范

```yaml
chip_model: string                    # 实际芯片型号
params:                               # 结构化的硬件参数对象
  # 通用参数（来自shared）
  ub_size_bytes: int                  # UB大小（bytes）
  alignment_bytes: int                # 对齐要求（bytes）
  vec_calc_size_bytes: int            # 向量宽度（bytes）

  # 型号特有参数（来自variants）
  core_num: int                       # 核数（Vector算子用 vector_core_cnt）
  cube_core_cnt: int                  # Cube核数
  freq_mhz: int                      # 频率（MHz）
  l2_size_bytes: int|null             # L2大小（bytes）
  gm_size_bytes: int|null             # GM容量（bytes）

  # 计算参数（derived_params）
  peak_bw_gbps: float                 # 峰值带宽（GB/s per core）
  peak_vector_tflops_per_core: float  # Vector峰值算力（TFLOPS/core）
  peak_cube_tflops_per_core: float    # Cube峰值算力（TFLOPS/core）
  max_tile_fp16_double_buf: int       # FP16双缓冲最大tile元素数

  # 参数来源标注
  _annotations:
    ub_size_bytes: exact              # exact | derived | estimated
    peak_bw_gbps: derived
    peak_cube_tflops_per_core: derived
```

## 错误处理

| 错误场景 | 处理策略 | 返回值 |
|---------|---------|--------|
| 芯片型号未指定且自动检测失败 | 明确报错，列出支持的芯片型号 | `{status: "error", reason: "无法检测芯片型号", supported_chips: [...]}` |
| 指定的芯片型号不在支持列表中 | 提示错误，建议相近型号 | `{status: "error", reason: "不支持的芯片型号", suggestion: "..."}` |
| 芯片JSON文件读取失败 | 尝试3次，仍失败则使用默认值并标注 | `{status: "partial", params: {...}, _annotations: {estimated: true}}` |
| shared/variants参数缺失 | 使用默认值，明确标注为estimated | `{status: "partial", params: {...}, missing_params: [...]}` |
| derived_params计算异常 | 记录异常，跳过计算，使用原始参数 | `{status: "partial", params: {...}, derivation_failed: true}` |
| npu-smi命令执行失败 | 降级到仅返回通用知识 | `{status: "degraded", params: {shared_only: true}, reason: "..."}` |

兜底时，所有依赖 `hw_params` 的后续步骤**静默跳过**，不影响进化流程继续运行。

## 芯片规格文件索引

| 系列 | 文件 | 典型型号 |
|------|------|---------|
| 910B | `references/chips/ascend_910b.json` | 910B, 910B1, 910B2, 910B2C, 910B3, 910B4, 910B4-1 |
| 910初代 | `references/chips/ascend_910.json` | 910A, 910PremiumA, 910ProA, 910ProB |
| 910_93xx | `references/chips/ascend_910_93xx.json` | 910_9351, 910_9352, 910_9354, 910_9356 |
| 910_95xx | `references/chips/ascend_910_95xx.json` | 910_950z, 910_9572~9599 (8/28/32/36核) |
| 910_55xx | `references/chips/ascend_910_55xx.json` | 910_5561, 910_5562, 910_5564 |
| 310B | `references/chips/ascend_310b.json` | 310B1, 310B2, 310B3, 310B4 |
| 310P | `references/chips/ascend_310p.json` | 310P1, 310P3, 310P5, 310P7 |
| 610 | `references/chips/ascend_610.json` | 610, 610Lite |
| BS9 | `references/chips/bs9.json` | BS9SX系列 |
| Kirin | `references/chips/kirin.json` | Kirin系列 |
| Hi3796 | `references/chips/hi3796.json` | Hi3796CV300系列 |
| SD3403 | `references/chips/sd3403.json` | SD3403 |
| Ascend031 | `references/chips/ascend_031.json` | Ascend 031 |
| Ascend035 | `references/chips/ascend_035.json` | Ascend 035系列 |
| AS31XM1X | `references/chips/as31xm1x.json` | AS31XM1X |
| MC6x | `references/chips/mc6x.json` | MC6x系列 |

## 与进化流程的集成点

- **调用时机**：`ops-evo` 的世界模型初始化之前调用一次
- **输出去向**：写入 `world_model.json` 顶层 `hw_params` 字段
- **后续使用**：
  - 世界模型节点设计：Roofline 分析 + 具体 tile_size 建议
  - 子agent prompt `[Hardware Specs]` 块：为 DSL 生成和 Local Refinement 提供硬件约束
  - Profiling 分析：峰值带宽/算力用于利用率计算
