# 混合策略自动选择

## 概述

同一算子在不同 shape、数据类型或内存布局下，可能需要不同的优化策略才能获得最优性能。混合策略自动选择通过在 host 端（`forward()` 中）根据运行时条件选择不同的 kernel 变体或参数配置，实现对多样化输入的自适应优化。

## 适用场景

当算子满足以下任一条件时，应考虑混合策略：

1. **不同 shape 范围性能瓶颈不同**：小 batch 受限于并行度，大 batch 受限于内存带宽
2. **不同数据类型精度要求不同**：fp32 对数值稳定性敏感，fp16/bf16 可容忍更多近似优化
3. **不同维度配置访存模式不同**：如 `inner_size == 1` vs `inner_size > 1` 的向量化策略差异

## 典型模式

### 模式 1：Shape 自适应策略选择

**问题**：单一 kernel 无法在所有 shape 下达到最优性能。

```python
# 问题：单一策略无法覆盖所有 shape
# 小 n_cols 时：需要 noloop 减少循环开销
# 大 n_cols 时：需要 loop 避免 UB 溢出
```

**优化方案**：根据 shape 动态选择 kernel 变体。

```python
class ModelNew:
    def forward(self, x, normalized_shape, weight=None, bias=None):
        n_rows, n_cols = ...
        
        # 混合策略：根据 n_cols 选择不同 kernel
        if n_cols <= 128 and n_rows >= 256:
            # 小 n_cols + 足够行数：使用 noloop kernel
            kernel = layer_norm_kernel_fp16_noloop
            ROWS_PER_BLOCK = 32
        else:
            # 大 n_cols：使用 loop kernel
            kernel = layer_norm_kernel_fp16_loop
            ROWS_PER_BLOCK = 8
        
        grid = (min(triton.cdiv(n_rows, ROWS_PER_BLOCK), self.VEC_CORE_NUM),)
        kernel[grid](..., ROWS_PER_BLOCK=ROWS_PER_BLOCK)
```

**关键约束**：
- noloop 需满足 `ROWS_PER_BLOCK >= 4`（避免 grid 过大导致 hang）
- noloop 需满足 `n_rows >= 256` 或 `grid >= 16`（保证足够并行度）
- noloop 需满足 UB 安全：`ROWS_PER_BLOCK * n_cols * dtype_size <= 98304`

### 模式 2：数据类型自适应策略选择

**问题**：不同数据类型对并行优化的容忍度不同。

```python
# 问题：fp32 对求和顺序敏感，fp16 可容忍
```

**优化方案**：根据数据类型启用/禁用特定优化。

```python
class ModelNew:
    def forward(self, x, ...):
        # fp32：禁用改变求和顺序的优化（精度敏感）
        # fp16/bf16：可启用并行优化
        use_parallel_stats = (x.dtype != torch.float32) and (total_groups < num_cores // 2)
        
        if use_parallel_stats:
            # 并行 stats：多个 group 同时统计，用 atomic_add 合并
            stats_kernel = parallel_stats_kernel
        else:
            # 原始 stats：单 group 串行统计
            stats_kernel = serial_stats_kernel
        
        stats_kernel[grid](...)
```

### 模式 3：Grid 并行度自适应策略选择

**问题**：小 batch / small groups 时 grid 不足，无法充分利用核数。

```python
# 问题：total_groups = 4，num_cores = 48，grid 过小
```

**优化方案**：根据 grid 大小选择串行或并行策略。

```python
# 判断条件
total_groups = N * num_groups
use_parallel = (total_groups < num_cores // 2) and (x.dtype != torch.float32)

if use_parallel:
    # 并行策略：每个 program 处理一个 group，用 atomic_add 合并
    grid = (min(total_groups, num_cores),)
    parallel_stats_kernel[grid](...)
else:
    # 串行策略：每个 program 处理多个 group
    groups_per_core = triton.cdiv(total_groups, num_cores)
    grid = (num_cores,)
    serial_stats_kernel[grid](...)
```

## 策略选择决策树

```
开始
│
├─ 检查数据类型
│   ├─ fp32 → 禁用改变求和顺序的优化
│   └─ fp16/bf16 → 可启用并行优化
│
├─ 检查 shape 大小
│   ├─ 小 n_cols (<=128) + 大 n_rows (>=256) + UB 安全
│   │   → 考虑 noloop kernel
│   └─ 其他 → 使用 loop kernel
│
├─ 检查 grid 并行度
│   ├─ total_work_items < num_cores // 2
│   │   → 考虑并行策略（atomic_add）
│   └─ total_work_items >= num_cores // 2
│       → 使用串行策略
│
└─ 检查内存布局
    ├─ inner_size == 1（操作维度为最后一维）
    │   → 直接向量化操作维度
    └─ inner_size > 1
        → 向量化 inner 维度
```

## 常见错误

### 错误 1：策略切换条件过于粗糙

```python
# ❌ 错误：只根据单一条件判断
if n_cols < 256:
    use_noloop = True  # 未考虑 n_rows 和 UB

# ✅ 正确：综合考虑多个条件
use_noloop = (
    n_cols <= 128 and           # noloop 仅适合小 n_cols
    n_rows >= 256 and           # 保证足够并行度
    ROWS_PER_BLOCK * n_cols * dtype_size <= 98304  # UB 安全
)
```

### 错误 2：策略切换引入额外开销

```python
# ❌ 错误：在 kernel 内做条件分支
@triton.jit
def kernel(...):
    if some_condition:  # kernel 内分支性能差
        ...

# ✅ 正确：在 host 端选择 kernel
if some_condition:
    kernel_a[grid](...)
else:
    kernel_b[grid](...)
```

### 错误 3：忽略精度影响

```python
# ❌ 错误：fp32 也启用并行规约
use_parallel = total_groups < num_cores // 2  # 未检查数据类型

# ✅ 正确：fp32 禁用改变求和顺序的优化
use_parallel = (total_groups < num_cores // 2) and (x.dtype != torch.float32)
```

## 性能收益

| 场景 | 单一策略 | 混合策略 | 收益 |
|------|---------|---------|------|
| LayerNorm (小 n_cols) | loop kernel | noloop kernel | 减少循环开销 |
| GroupNorm (小 groups) | serial stats | parallel stats + atomic | 提升并行度 |
| fp32 数据 | 并行规约 | 串行规约 | 保证精度 |

## 总结

| 维度 | 小/敏感 | 大/容忍 |
|------|--------|--------|
| n_cols | noloop (<=128) | loop |
| n_rows / grid | parallel (atomic) | serial |
| 数据类型 | fp32 → 保守 | fp16/bf16 → 激进 |
| inner_size | 向量化操作维度 | 向量化 inner 维度 |

**核心原则**：
- 策略选择在 host 端（`forward()`）完成，不在 kernel 内分支
- 切换条件需综合考虑 shape、数据类型、UB 安全、并行度
- fp32 优先保证精度，fp16/bf16 可尝试更多优化
