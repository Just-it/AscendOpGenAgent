# LayerNorm 计算分解示例

## 算子信息
- op_name: layer_norm
- category: normalization
- 计算模式: **单行归约 (3-pass tiled)**
- 归约维度: normalized_dim=1 (即 features 维), normalized_shape=[768]
- 输入 shape: [32, 768], dtype: float32 (另有 weight [768], bias [768])
- 输出 shape: [32, 768], dtype: float32
- 特殊属性: eps=1e-5, elementwise_affine=true

## 来源文件
- reference: `layer_norm_reference.py` → `F.layer_norm(x, normalized_shape, weight, bias, eps)`
- DSL: `layer_norm_dsl.py` → 3-pass kernel (mean → variance → normalize+affine)
- op_desc: `layer_norm_op_desc.json` → `attributes.normalized_shape=[768], eps=1e-5`

## 计算链分解

### Step 0: 输入
- tensor: x shape [32, 768], weight shape [768], bias shape [768]
- dtype: float32
- x 数值范围: torch.rand → [0, 1)
- weight 初始值: torch.ones → 全 1
- bias 初始值: torch.zeros → 全 0

### Step 1: ReduceSum → Mean (Pass 1)
- 操作: `row_sum = sum(x, dim=-1)`, `row_mean = row_sum / norm_size`
- DSL 对应: Pass 1 循环 — 每 tile `tl.reduce_sum(shared_ub, row_tile_ub, shared_ub)`, 跨 tile 累加 `row_sum += tile_sum`, 最后 `row_mean = row_sum / norm_size`
- 输入: Step 0 的 x, shape [32, 768]
- 输出 shape: [32, 1] (每行一个标量 mean)
- 数值范围预期: ~0.5 (torch.rand 均值约 0.5)
- **精度风险点**:
  - ReduceSum 的 count 参数必须 64 倍数对齐
  - 多 tile 累加时, 跨 tile 的 partial sum 在标量变量中累加, 精度取决于累加顺序
  - Padding 区域值为 0 时: Sum 不受影响 (0 是加法单位元), 但 **分母 norm_size 必须是原始维度长度而非对齐后长度**
  - 若 norm_size 不等于 tileLength × n_tiles (最后一个 tile 有 padding), 分母仍为 norm_size 是正确的
- **知识库关联**: #7 Reduce Count 未对齐, #13 Normalization CHECKLIST

### Step 2: Sub (去中心化)
- 操作: `centered = x - row_mean` (广播减法)
- DSL 对应: Pass 2 中 `tl.vsub_scalar(temp_ub, row_tile_ub, row_mean)`
- 输入: Step 0 的 x + Step 1 的 row_mean
- 输出 shape: [32, 768]
- 数值范围预期: ~[-0.5, 0.5] (以均值为中心)
- **精度风险点**:
  - 此步本身安全, 但若 Step 1 的 mean 有误差, 会传播到所有后续步骤

### Step 3: Square → ReduceSum → Variance (Pass 2)
- 操作: `sq = centered^2`, `var_sum = sum(sq, dim=-1)`, `row_var = var_sum / norm_size`
- DSL 对应: Pass 2 循环 — `tl.vmul(temp_ub, temp_ub, temp_ub)` + `tl.reduce_sum(shared_ub, temp_ub, shared_ub)`, 跨 tile 累加, 最后 `row_var = row_var_sum / norm_size`
- 输入: Step 2 的 centered, shape [32, 768]
- 输出 shape: [32, 1] (每行一个标量 variance)
- 数值范围预期: ~0.083 (uniform [0,1) 的方差 = 1/12 ≈ 0.083)
- **精度风险点**:
  - 使用 `E[(x-mean)^2]` 直接计算方差 (DSL 中的做法), 数值稳定
  - 如果改用 `E[x^2] - (E[x])^2`, 大值时不稳定 — 检查 Kernel 是否保持了 DSL 的计算方式
  - Padding 区域: centered padding = (0 - mean), 其平方不为 0, **会污染 variance**
  - 正确做法: padding 区域应在 Sub 前初始化为 mean (使 centered=0), 或在 ReduceSum 时排除 padding
- **知识库关联**: #2 归约操作 Padding 值语义不匹配, #13 Normalization CHECKLIST 项 (1)(4)

### Step 4: Sqrt (标准差)
- 操作: `row_std = sqrt(row_var + eps)`
- DSL 对应: `row_std = tl.sqrt(row_var + eps)` (标量运算)
- 输入: Step 3 的 row_var + eps=1e-5
- 输出 shape: [32, 1] (每行一个标量 std)
- 数值范围预期: ~0.288 (sqrt(0.083 + 1e-5))
- **精度风险点**:
  - eps 必须在 sqrt **之前**加入 (即 sqrt(var + eps)), 而非 sqrt(var) + eps
  - 若 variance 为 0 且无 eps, sqrt(0)=0 → 后续 Div 除零
  - 此步为标量运算, AscendC 中可能用 CPU 侧计算或 UB 标量操作
- **知识库关联**: #13 Normalization CHECKLIST 项 (2)

### Step 5: Div (归一化) — Pass 3 前半
- 操作: `normalized = centered / row_std` (广播除法)
- DSL 对应: Pass 3 中 `tl.vsub_scalar(temp_ub, row_tile_ub, row_mean)` + `tl.vdiv_scalar(temp_ub, temp_ub, row_std)`
- 输入: Step 0 的 x (重新加载) + Step 1 的 mean + Step 4 的 std
- 输出 shape: [32, 768]
- 数值范围预期: ~N(0, 1) 分布 (均值≈0, 标准差≈1)
- **精度风险点**:
  - DSL Pass 3 中重新加载了输入 x 并重新做 Sub, 而非复用 Pass 2 的 centered — 这是因为 UB 空间不够保存中间结果
  - Kernel 是否也做了同样的重新加载? 如果 Kernel 试图复用已被覆盖的 buffer, 会读到错误数据
- **注意**: DSL 中 Pass 3 重新对 x 做 Sub, 这意味着 **同一行的 x 被加载了 3 次** (Pass 1 + Pass 2 + Pass 3)

### Step 6: Mul + Add (Affine 变换) — Pass 3 后半
- 操作: `output = normalized * weight + bias`
- DSL 对应: Pass 3 中 `tl.vmul(out_ub, temp_ub, weight_tile_ub)` + `tl.vadd(out_ub, out_ub, bias_tile_ub)`
- 输入: Step 5 的 normalized + weight [768] + bias [768]
- 输出 shape: [32, 768]
- 数值范围预期: 与 normalized 相同 (因为 weight=1, bias=0 是初始值)
- **精度风险点**:
  - weight 和 bias 是 per-element 的, 需要正确的 GM 偏移 (按 tile 对齐加载)
  - weight/bias 的 GM 偏移应为 `col_start + arange(0, tile_length)`, 不带 row_idx 乘积 (它们是共享参数)
- **知识库关联**: #13 Normalization CHECKLIST 项 (3)

## 误差传播链

```
Step 1 (Mean) 误差
  → Step 2 (Sub) 所有元素中心化偏移
    → Step 3 (Var) 方差计算偏差
      → Step 4 (Std) 标准差偏差
        → Step 5 (Div) 归一化偏差
          → Step 6 (Affine) 最终输出偏差

独立风险: Step 3 中 Padding 污染 Variance
  → Step 4 std 偏大
    → Step 5 归一化值偏小
      → 典型 pattern: uniform_offset (所有值偏小)
```

## DSL tiling 策略要点

从 `layer_norm_dsl.py` 提取:
- `n_cores = 32`, 按行分配 (`rows_per_core = rows // n_cores`)
- `tile_length = min(4096, norm_size)`, 若 norm_size=768 则 tile_length=768 (单 tile)
- `n_tiles = ceil(norm_size / tile_length)` — 768 / 768 = 1 tile
- 3 个 Pass 分别遍历: 每 Pass 都从 GM 重新加载 x (无法复用, UB 不够存全行)
- 6 个 UB buffer: row_tile_ub, weight_tile_ub, bias_tile_ub, temp_ub, shared_ub, out_ub

## 与 AscendC Kernel 的对照要点

1. Pass 1 (Mean): ReduceSum count 是否对齐? 分母是否 = norm_size (不是 tileLength)?
2. Pass 2 (Var): 是否使用 `(x-mean)^2` 而非 `x^2 - mean^2`? Padding 区域是否正确处理?
3. Pass 2→3 之间: `sqrt(var + eps)` 中 eps 是否在 sqrt 之前加入?
4. Pass 3 (Normalize): 是否重新加载了 x 并重新 Sub? (与 DSL 一致)
5. Pass 3 (Affine): weight/bias 的 GM 偏移是否只用 col_start (不含 row_idx)?
6. Host TilingFunc: tileLength 是否正确传递? norm_size 是否作为 TilingData 字段?
