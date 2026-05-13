# ReduceSum 计算分解示例

## 算子信息
- op_name: reduce_sum
- category: reduction
- 计算模式: **单步归约 + 跨步内存访问**
- 归约维度: dim=1, keepdim=false
- 输入 shape: [16, 32, 64], dtype: float32
- 输出 shape: [16, 64], dtype: float32

## 来源文件
- reference: `reduce_sum_reference.py` → `torch.sum(x, dim=self.dim, keepdim=self.keepdim)`
- DSL: `reduce_sum_dsl.py` → 按 (batch, feature) 维度分配任务, 沿 dim=1 归约
- op_desc: `reduce_sum_op_desc.json` → `attributes.dim=1, keepdim=false`

## 计算链分解

### Step 0: 输入
- tensor: x
- shape: [16, 32, 64] (B=16, D=32, F=64)
- dtype: float32
- 数值范围: torch.rand → [0, 1)
- 关键: 归约维 D=32 不是最后一维, 内存中沿 D 方向的元素**不连续** (stride=F=64)

### Step 1: Gather (跨步加载归约维数据)
- 操作: 对每个 (b, f) 位置, 收集 x[b, 0:D, f] 共 D=32 个元素到连续 buffer
- DSL 对应: `offsets = batch_idx * D * F + feature_idx + tl.arange(0, D) * F` → `tl.load(input_ptr + offsets, x_ub)`
- 输入: GM 中 shape [16, 32, 64] 的 tensor
- 输出: UB 中 shape [D] = [32] 的连续 buffer
- **精度风险点**:
  - 非连续访存: stride=F=64, 不是元素级连续的 DataCopy
  - AscendC DataCopy 要求连续内存, 非连续数据需要 strided DataCopy 或多次 DataCopy
  - 如果 Kernel 错误地假设数据连续 (使用简单 DataCopy), 会加载到错误数据
  - D=32 × sizeof(float32) = 128 bytes, 满足 32-byte 对齐
- **知识库关联**: #5 数据布局不一致导致维度级错误

### Step 2: ReduceSum
- 操作: `row_sum = sum(buffer, dim=0)` → 一个标量
- DSL 对应: `tl.reduce_sum(accum_ub, x_ub, shared_ub)` + `extract_scalar(accum_ub, 0)`
- 输入: Step 1 的连续 buffer, shape [32]
- 输出: 标量 (写入 output[b, f])
- 数值范围预期: ~16 (32 个 [0,1) 的和, 期望值 = 32 × 0.5 = 16)
- **精度风险点**:
  - D=32 个 float32 相加, 精度损失很小 (float32 有足够精度)
  - ReduceSum count=32 → 不满足 64 倍数对齐, 需要对齐到 64
  - Padding 区域 (32→64 的填充) 值必须为 0, 否则 sum 结果偏大
- **知识库关联**: #7 Reduce Count 未对齐, #2 归约操作 Padding 值语义不匹配

## 误差传播链

```
Step 1 (Gather) 数据加载错误
  → Step 2 (Sum) 完全错误的求和结果
  → pattern: all_wrong 或 dimension_concentration

Step 2 (Sum) Padding/Count 问题
  → 每个输出元素偏大 (padding 非 0 参与求和)
  → pattern: uniform_offset (所有值偏大)
```

## DSL tiling 策略要点

从 `reduce_sum_dsl.py` 提取:
- 核心分配: 按输出元素 (B×F = 16×64 = 1024) 分配, `n_cores = min(16, total_output_elems)`
- 每核处理: `rows_per_core` 个 (b, f) 位置
- tile_size = min(D, 2048) = 32 (整个归约维放入 UB)
- 3 个 UB buffer: x_ub, accum_ub, shared_ub

## 与 AscendC Kernel 的对照要点

1. **GM 偏移计算**: 是否正确实现跨步访存 `batch_idx * D * F + feature_idx + i * F`?
2. **DataCopy 方式**: 是否使用 strided DataCopy 而非简单连续 DataCopy?
3. **ReduceSum count**: D=32 不满足 64 倍数, 是否对齐到 64? 对齐后 padding 是否为 0?
4. **输出偏移**: output[b, f] 的 GM 偏移是否 = `batch_idx * F + feature_idx`?
5. **keepdim 处理**: keepdim=false 时输出 shape [16, 64], 不需要保留 dim=1 — Kernel 无需特殊处理
