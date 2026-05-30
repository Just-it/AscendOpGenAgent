# Reduce 类算子 Kernel 分裂经验

## 1. 分组维度

| 维度 | 判定条件 | 分组 |
|------|---------|------|
| **Reduce 轴位置** | `inner_size == 1` vs `inner_size > 1` | reduce-last / reduce-non-last |
| **Reduce 规模** | `reduce_size < 256` / `256~4096` / `>4096` | 小/中/大 |
| **数据类型** | fp16/bf16/fp32 | 不同精度需不同向量化策略 |

---

## 2. Reduce-Last 特化（`inner_size == 1`）

### 2.1 适用场景
Reduce 轴在最后维度，数据在内存中连续，可直接线性访存。

### 2.2 Kernel 实现

```python
@triton.jit
def sum_kernel_reduce_last(
    input_ptr, output_ptr,
    outer_size, reduce_size,
    BLOCK_RED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Grid 动态分配：按 VEC_CORE_NUM 限制
    tiles_per_pid = (outer_size + num_pids - 1) // num_pids
    tile_start = pid * tiles_per_pid
    tile_end = tl.minimum(tile_start + tiles_per_pid, outer_size)

    for outer_idx in range(tile_start, tile_end):
        acc = tl.full((), 0.0, tl.float32)
        base_offset = outer_idx * reduce_size

        # 沿 reduce 轴分块累加
        for r_start in range(0, reduce_size, BLOCK_RED):
            r_end = tl.minimum(r_start + BLOCK_RED, reduce_size)
            r_offsets = r_start + tl.arange(0, BLOCK_RED)
            r_mask = r_offsets < r_end

            vals = tl.load(input_ptr + base_offset + r_offsets, mask=r_mask, other=0.0)
            vals = vals.to(tl.float32)
            acc += tl.sum(vals, axis=0)

        tl.store(output_ptr + outer_idx, acc.to(output_ptr.dtype.element_ty))
```

### 2.3 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **连续访存** | `base_offset + r_offsets` 线性偏移，无 stride | 最大化内存带宽利用率 |
| **固定 constexpr** | `BLOCK_RED` 声明为 constexpr | 编译期优化，消除运行时分支 |
| **Grid 限制** | `grid = (min(outer_size, VEC_CORE_NUM),)` | 避免过度并行导致调度开销 |
| **标量累加器** | `acc = tl.full((), 0.0, tl.float32)` | 单值累加，减少寄存器压力 |

### 2.4 Grid 配置

```python
grid = (min(outer_size, self.VEC_CORE_NUM),)
sum_kernel_reduce_last[grid](
    x, output,
    outer_size, reduce_size,
    BLOCK_RED=1024,
)
```

---

## 3. Reduce-Non-Last 特化（`inner_size > 1`）

### 3.1 适用场景
Reduce 轴不在最后维度，需要 2D tile 策略同时处理 reduce 轴和 inner 轴。

### 3.2 Kernel 实现

```python
@triton.jit
def sum_kernel_reduce_non_last(
    input_ptr, output_ptr,
    outer_size, reduce_size, inner_size,
    BLOCK_RED: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # 2D tile 映射：outer × inner 展平为 1D grid
    inner_tiles = (inner_size + BLOCK_INNER - 1) // BLOCK_INNER
    total_tiles = outer_size * inner_tiles

    tiles_per_pid = (total_tiles + num_pids - 1) // num_pids
    tile_start = pid * tiles_per_pid
    tile_end = tl.minimum(tile_start + tiles_per_pid, total_tiles)

    for tile_idx in range(tile_start, tile_end):
        outer_idx = tile_idx // inner_tiles
        inner_tile = tile_idx % inner_tiles
        inner_start = inner_tile * BLOCK_INNER
        inner_end = tl.minimum(inner_start + BLOCK_INNER, inner_size)
        inner_len = inner_end - inner_start

        # 向量累加器：shape = (BLOCK_INNER,)
        acc = tl.zeros((BLOCK_INNER,), dtype=tl.float32)
        base_offset = outer_idx * reduce_size * inner_size + inner_start

        for r_start in range(0, reduce_size, BLOCK_RED):
            r_end = tl.minimum(r_start + BLOCK_RED, reduce_size)

            # 2D offset 计算
            r_offs = r_start + tl.arange(0, BLOCK_RED)[:, None]
            i_offs = tl.arange(0, BLOCK_INNER)[None, :]
            in_offsets = base_offset + r_offs * inner_size + i_offs

            mask_r = r_offs < r_end
            mask_i = i_offs < inner_len
            mask = mask_r & mask_i

            vals = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)
            vals = vals.to(tl.float32)
            acc += tl.sum(vals, axis=0)

        # 2D store
        out_base = outer_idx * inner_size + inner_start
        out_offsets = out_base + tl.arange(0, BLOCK_INNER)
        out_mask = tl.arange(0, BLOCK_INNER) < inner_len
        tl.store(output_ptr + out_offsets, acc.to(output_ptr.dtype.element_ty), mask=out_mask)
```

### 3.3 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **2D tiling** | 同时 tile reduce 轴和 inner 轴 | 充分利用并行度 |
| **向量累加器** | `acc = tl.zeros((BLOCK_INNER,), dtype=tl.float32)` | 批量累加，减少循环次数 |
| **Offset 计算** | `r_offs * inner_size + i_offs` 二维广播 | 精确映射 2D 内存布局 |
| **Grid 计算** | `total_tiles = outer_size * inner_tiles` | 按 `VEC_CORE_NUM` 限制 |

### 3.4 Grid 配置

```python
inner_tiles = (inner_size + 63) // 64
total_tiles = outer_size * inner_tiles
grid = (min(total_tiles, self.VEC_CORE_NUM),)
sum_kernel_reduce_non_last[grid](
    x, output,
    outer_size, reduce_size, inner_size,
    BLOCK_RED=64,
    BLOCK_INNER=64,
)
```

---

## 4. 分组建议

| Case 特征 | 推荐 Kernel | BLOCK 配置 |
|----------|------------|-----------|
| `inner_size == 1`, reduce_size 任意 | reduce-last | `BLOCK_RED=1024` |
| `inner_size > 1`, reduce_size ≤ 256 | reduce-non-last | `BLOCK_RED=64, BLOCK_INNER=64` |
| `inner_size > 1`, reduce_size > 256 | reduce-non-last | `BLOCK_RED=128, BLOCK_INNER=32` |

---

## 5. 调度器实现

```python
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        return self._route(x, dim, keepdim)

    def _route(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        original_dim = dim
        if dim is None:
            x = x.reshape(1, -1)
            dim = 1
        if dim < 0:
            dim = x.ndim + dim

        shape = x.shape
        outer_size = math.prod(shape[:dim]) if dim > 0 else 1
        reduce_size = shape[dim]
        inner_size = math.prod(shape[dim + 1:]) if dim + 1 < x.ndim else 1

        if keepdim:
            out_shape = list(shape)
            out_shape[dim] = 1
        else:
            out_shape = list(shape[:dim]) + list(shape[dim + 1:])

        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        # === 路由决策 ===
        if inner_size == 1:
            # reduce-last: 直接沿 reduce 轴连续访存
            grid = (min(outer_size, self.VEC_CORE_NUM),)
            sum_kernel_reduce_last[grid](
                x, output,
                outer_size, reduce_size,
                BLOCK_RED=1024,
            )
        else:
            # reduce-non-last: 2D tile 策略
            inner_tiles = (inner_size + 63) // 64
            total_tiles = outer_size * inner_tiles
            grid = (min(total_tiles, self.VEC_CORE_NUM),)
            sum_kernel_reduce_non_last[grid](
                x, output,
                outer_size, reduce_size, inner_size,
                BLOCK_RED=64,
                BLOCK_INNER=64,
            )

        if original_dim is None and not keepdim:
            output = output.squeeze()

        return output
```

**关键设计**：
- **无字典映射**：直接用 `if-else` 分支，零开销
- **内联 grid 计算**：每个分支独立计算 grid
- **constexpr 固定**：BLOCK 值硬编码
