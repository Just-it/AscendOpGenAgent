# 广播逐元素算子 Kernel 分裂经验

## 1. 分组维度

| 维度 | 判定条件 | 分组 |
|------|---------|------|
| **是否广播** | `x.shape == y.shape` | 无广播 / 有广播 |
| **广播维度位置** | `broadcast_dims` 位置 | dim0 / dim1 / 多维 |
| **输出维度数** | `out_ndim` | 2D / 3D / 4D |

---

## 2. 无广播特化

### 2.1 适用场景
`x.shape == y.shape`，纯逐元素操作，无广播开销。

### 2.2 Kernel 实现

```python
@triton.jit
def add_no_broadcast_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, tl.num_programs(0))
    start_block = pid * blocks_per_core
    end_block = min(start_block + blocks_per_core, num_blocks)

    for block_idx in range(start_block, end_block):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        out = x + alpha * y

        tl.store(out_ptr + offsets, out, mask=mask)
```

### 2.3 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **大 BLOCK** | 推荐范围 `4096~16384`，通过 autotune 选择 | 充分利用带宽 |
| **Grid 动态** | `grid = (min(tl.cdiv(n_elements, BLOCK_SIZE), VEC_CORE_NUM),)` | 避免过度并行 |
| **循环分块** | `blocks_per_core` 分配 | 每个 core 处理多个 block |

---

## 3. 2D 广播特化

### 3.1 适用场景
2D 张量，`y` 在某一维度上广播（如 `y=[M,1]` 或 `y=[1,N]`）。

### 3.2 Broadcast Dim1（`y=[M,1]`）

```python
@triton.jit
def add_broadcast_2d_dim1_kernel(
    x_ptr, y_ptr, out_ptr, M, N, alpha,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n
    blocks_per_core = tl.cdiv(num_blocks, tl.num_programs(0))
    start_block = pid * blocks_per_core
    end_block = min(start_block + blocks_per_core, num_blocks)

    for block_idx in range(start_block, end_block):
        bm = block_idx // num_blocks_n
        bn = block_idx % num_blocks_n
        row_offs = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offs = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = row_offs < M
        col_mask = col_offs < N
        mask_2d = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + row_offs[:, None] * N + col_offs[None, :], mask=mask_2d, other=0.0)
        y = tl.load(y_ptr + row_offs[:, None], mask=row_mask[:, None], other=0.0)
        out = x + alpha * y

        tl.store(out_ptr + row_offs[:, None] * N + col_offs[None, :], out, mask=mask_2d)
```

### 3.3 Broadcast Dim0（`y=[1,N]`）

```python
# y load 改为 1D 列维度
y = tl.load(y_ptr + col_offs, mask=col_mask, other=0.0)
```

### 3.4 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **2D tiling** | `BLOCK_M` 推荐 `4~16`，`BLOCK_N` 推荐 `512~2048`，通过 autotune 选择 | 匹配广播模式 |
| **广播 load** | `y` 仅加载一次，利用 broadcast 语义 | 减少内存访问 |
| **2D mask** | `row_mask[:, None] & col_mask[None, :]` | 精确边界处理 |

---

## 4. 通用高维广播特化

### 4.1 适用场景
3D/4D 张量，广播模式复杂，不适合在 kernel 内处理。

### 4.2 特化策略
Host 端 `expand + contiguous + view(-1)`，退化为 1D kernel。

### 4.3 Kernel 实现

```python
@triton.jit
def add_broadcast_1d_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, tl.num_programs(0))
    start_block = pid * blocks_per_core
    end_block = min(start_block + blocks_per_core, num_blocks)

    for block_idx in range(start_block, end_block):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        out = x + alpha * y

        tl.store(out_ptr + offsets, out, mask=mask)
```

### 4.4 Host 端预处理

```python
out_shape = torch.broadcast_shapes(x.shape, y.shape)
x_expanded = x.expand(out_shape).contiguous().view(-1)
y_expanded = y.expand(out_shape).contiguous().view(-1)
output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
out_flat = output.view(-1)

n_elements = out_flat.numel()
# BLOCK_SIZE 和 grid 通过 autotune 或动态计算
BLOCK_SIZE = 8192  # 推荐初始值，可 autotune
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
grid_size = min(num_blocks, self.VEC_CORE_NUM)
add_broadcast_1d_kernel[grid_size](
    x_expanded, y_expanded, out_flat, n_elements, alpha, BLOCK_SIZE
)
```

### 4.5 关键优化点

| 优化点 | 做法 | 收益 |
|-------|------|------|
| **Host 端展平** | 利用 PyTorch 的 `expand` 处理广播 | kernel 简化为 1D |
| **内存连续化** | `contiguous()` 确保线性访存 | 避免跨步访存 |
| **复用 1D kernel** | 与无广播路径共享 kernel 代码 | 减少代码量 |

---

## 5. 广播信息提取

```python
def _get_broadcast_info(self, x_shape, y_shape):
    if x_shape == y_shape:
        return len(x_shape), [], False

    max_ndim = max(len(x_shape), len(y_shape))
    x_padded = [1] * (max_ndim - len(x_shape)) + list(x_shape)
    y_padded = [1] * (max_ndim - len(y_shape)) + list(y_shape)

    out_shape = []
    broadcast_dims = []
    for i in range(max_ndim):
        if x_padded[i] == y_padded[i]:
            out_shape.append(x_padded[i])
        elif x_padded[i] == 1:
            out_shape.append(y_padded[i])
        elif y_padded[i] == 1:
            out_shape.append(x_padded[i])
            broadcast_dims.append(i)
        else:
            raise ValueError(f"Incompatible shapes: {x_shape}, {y_shape}")

    return len(out_shape), tuple(broadcast_dims), True
```

---

## 6. 分组建议与参数范围

| Case 特征 | 推荐 Kernel | BLOCK 范围 | 推荐 autotune 配置 |
|----------|------------|-----------|-------------------|
| `x.shape == y.shape` | no-broadcast | `BLOCK_SIZE: [4096, 8192, 16384]` | `@triton.autotune(configs=[...], key=['n_elements'])` |
| 2D, `y` 在 dim1 广播 | broadcast-2d-dim1 | `BLOCK_M: [4, 8, 16], BLOCK_N: [512, 1024, 2048]` | `@triton.autotune(configs=[...], key=['M', 'N'])` |
| 2D, `y` 在 dim0 广播 | broadcast-2d-dim0 | `BLOCK_M: [4, 8, 16], BLOCK_N: [512, 1024, 2048]` | `@triton.autotune(configs=[...], key=['M', 'N'])` |
| 3D/4D 任意广播 | generic-1d | `BLOCK_SIZE: [4096, 8192, 16384]` | `@triton.autotune(configs=[...], key=['n_elements'])` |

---

## 7. 调度器实现

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def _get_broadcast_info(self, x_shape, y_shape):
        # ... 同上 ...

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self._route(x, y, alpha)

    def _route(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        out_ndim, broadcast_dims, is_broadcast = self._get_broadcast_info(x.shape, y.shape)

        if not is_broadcast:
            if not x.is_contiguous(): x = x.contiguous()
            if not y.is_contiguous(): y = y.contiguous()
            output = torch.empty_like(x)
            n_elements = x.numel()

            # BLOCK_SIZE 可通过 autotune 选择，此处为推荐初始值
            BLOCK_SIZE = 8192
            grid = (min(triton.cdiv(n_elements, BLOCK_SIZE), self.VEC_CORE_NUM),)
            add_no_broadcast_kernel[grid](x, y, output, n_elements, alpha, BLOCK_SIZE)
            return output

        # === 广播路由 ===
        key = (out_ndim, broadcast_dims)

        if key == (2, (1,)):
            output = torch.empty_like(x)
            M, N = x.shape
            BLOCK_M, BLOCK_N = 8, 1024  # 推荐初始值，可 autotune
            num_blocks_m = triton.cdiv(M, BLOCK_M)
            num_blocks_n = triton.cdiv(N, BLOCK_N)
            grid = (min(num_blocks_m * num_blocks_n, self.VEC_CORE_NUM),)
            add_broadcast_2d_dim1_kernel[grid](x, y, output, M, N, alpha, BLOCK_M, BLOCK_N)
            return output

        elif key == (2, (0,)):
            output = torch.empty_like(x)
            M, N = x.shape
            BLOCK_M, BLOCK_N = 8, 1024
            num_blocks_m = triton.cdiv(M, BLOCK_M)
            num_blocks_n = triton.cdiv(N, BLOCK_N)
            grid = (min(num_blocks_m * num_blocks_n, self.VEC_CORE_NUM),)
            add_broadcast_2d_dim0_kernel[grid](x, y, output, M, N, alpha, BLOCK_M, BLOCK_N)
            return output

        else:
            out_shape = torch.broadcast_shapes(x.shape, y.shape)
            x_expanded = x.expand(out_shape).contiguous().view(-1)
            y_expanded = y.expand(out_shape).contiguous().view(-1)
            output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            out_flat = output.view(-1)

            n_elements = out_flat.numel()
            BLOCK_SIZE = 8192
            grid = (min(triton.cdiv(n_elements, BLOCK_SIZE), self.VEC_CORE_NUM),)
            add_broadcast_1d_kernel[grid](x_expanded, y_expanded, out_flat, n_elements, alpha, BLOCK_SIZE)
            return output
```

**关键设计**：
- **Key 匹配**：`(out_ndim, broadcast_dims)` 元组作为路由 key
- **内联分支**：每个分支直接写 kernel 调用
- **Host 预处理**：3D/4D 走 `expand + contiguous + view(-1)` 路径
- **参数可调**：BLOCK 值为推荐初始值，支持 `@triton.autotune` 自动调优
