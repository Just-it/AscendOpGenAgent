# Grid 形状与多路径特化 优化模式

## 概述

**核心方法论：动态 Host Dispatch 是 Triton 性能调优的核心手段。**

Triton kernel 一旦被 `@triton.jit` 编译，其内部的控制流结构（循环、分支）和 grid 拓扑即被固定。然而，同一算子在不同输入 shape 下往往呈现截然不同的 workload 特征：
- 小 tensor：`total_blocks <= num_cores`，每个 program 只需处理 1 个 block，任何分区循环都是纯开销
- 大 tensor：`total_blocks >> num_cores`，必须通过标量循环将 block 均匀分配给各 program

单一 kernel 实现无法在所有场景下同时最优。此时，**在 Host 侧（Python `forward()`）根据运行时 workload 特征动态选择不同 kernel 路径**，是突破性能瓶颈的关键手段。

## 为什么 Host Dispatch 有效

| 层面 | 限制 | Host Dispatch 的优势 |
|------|------|---------------------|
| **Kernel 内部** | 编译后控制流固定，无法根据 block 数量动态调整 | Kernel 侧只写最优路径，不加兼容逻辑 |
| **Grid 拓扑** | 1D/2D grid 形状在启动时确定，kernel 内无法变更 | Host 侧为不同场景选择最适合的 grid 形状 |
| **编译特化** | 同一 kernel 源码只能生成一种机器码 | 不同路径可分别编译，各自特化 |
| **调度开销** | 小 grid 时 program 启动/同步开销占比高 | Host 侧避免不必要的多 program 调度 |

**关键洞察**：kernel 的通用性越强，单个场景的性能越差；Host 侧做动态 dispatch，让每个 kernel 只做一件事并把这件事做到极致。

## 典型多路径设计模式

### 模式 A：小 Grid 直接映射路径（Direct Mapping Path）

**适用场景**：`total_blocks <= num_cores`（或略大于核数）

**策略**：
- `grid` 直接映射到 workload 拓扑，如 `grid = (outer_size, num_inner_blocks)`
- Kernel 内**无任何标量分区循环**，每个 program 直接处理 1 个 block
- `tl.program_id(0)` / `tl.program_id(1)` 直接定位到具体 block

**代码骨架**：
```python
@triton.jit
def kernel_direct(x_ptr, out_ptr, outer_size, inner_size,
                  r: tl.constexpr, BLOCK: tl.constexpr):
    outer_idx = tl.program_id(0)
    local_block = tl.program_id(1)

    block_start = local_block * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < inner_size

    in_offset = outer_idx * inner_size
    val = tl.load(x_ptr + in_offset + offs, mask=mask)

    for repeat_idx in range(r):
        out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
        tl.store(out_ptr + out_offset + offs, val, mask=mask)
```

### 模式 B：大 Grid 多核分区路径（Partition Loop Path）

**适用场景**：`total_blocks > num_cores`

**策略**：
- `grid = (num_cores,)` 或 `grid = (min(total_blocks, num_cores),)`
- Kernel 内通过标量循环将 block 均匀分配给各 program
- 保证负载均衡，充分利用所有物理核

**代码骨架**：
```python
@triton.jit
def kernel_partition(x_ptr, out_ptr, outer_size, inner_size, num_inner_blocks,
                     num_cores: tl.constexpr, r: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    total_blocks = outer_size * num_inner_blocks

    blocks_per_core = total_blocks // num_cores
    remainder = total_blocks - blocks_per_core * num_cores

    if pid < remainder:
        my_blocks = blocks_per_core + 1
        start_block = pid * (blocks_per_core + 1)
    else:
        my_blocks = blocks_per_core
        start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core

    for block_idx in range(start_block, start_block + my_blocks):
        outer_idx = block_idx // num_inner_blocks
        local_block = block_idx - outer_idx * num_inner_blocks

        block_start = local_block * BLOCK
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < inner_size

        in_offset = outer_idx * inner_size
        val = tl.load(x_ptr + in_offset + offs, mask=mask)

        for repeat_idx in range(r):
            out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
            tl.store(out_ptr + out_offset + offs, val, mask=mask)
```

### 模式 C：按维度数特化路径（Dimension Specialization）

**适用场景**：算子支持 1D/2D/3D/4D 等多种维度，不同维度的最优 grid 策略不同

**策略**：
- Host 侧根据 `len(shape)` 或具体维度大小选择不同 kernel
- 例如 1D 小 tensor 用单 program 处理，4D 大 tensor 用 2D grid

## Host Dispatch 决策框架

```python
def forward(self, x, repeats):
    # 1. 计算 workload 特征
    outer_size = ...
    inner_size = ...
    BLOCK = get_block_size(inner_size)
    num_inner_blocks = (inner_size + BLOCK - 1) // BLOCK
    total_blocks = outer_size * num_inner_blocks

    # 2. 根据特征选择路径
    if total_blocks <= self.VEC_CORE_NUM:
        # 小 Grid 路径：直接映射，无分区循环
        grid = (outer_size, num_inner_blocks)
        kernel_direct[grid](x, out, outer_size, inner_size, r=r, BLOCK=BLOCK)
    else:
        # 大 Grid 路径：多核分区，带标量循环
        grid = (min(total_blocks, self.VEC_CORE_NUM),)
        kernel_partition[grid](x, out, outer_size, inner_size, num_inner_blocks,
                               num_cores=grid[0], r=r, BLOCK=BLOCK)
```

## 完整示例 — Repeat 算子的多路径特化

**背景**：`torch.repeat(*repeats)` 对不同 shape 的输入，其 `total_blocks` 分布跨度极大（从 1 到 8192+）。

**未优化前**：单一 kernel + `grid = (min(total_blocks, VEC_CORE_NUM),)`
- 小 grid 时：分区计算（`blocks_per_core = total_blocks // num_cores`）的标量分支和循环完全无意义
- 大 grid 时：分区循环是必要的

**优化后**：双 kernel + Host dispatch
```python
class ModelNew(nn.Module):
    def forward(self, x, repeats):
        # ... 维度处理 ...
        BLOCK = get_block_size(inner_size)
        num_inner_blocks = (inner_size + BLOCK - 1) // BLOCK
        total_blocks = outer_size * num_inner_blocks

        if total_blocks <= self.VEC_CORE_NUM:
            # 小 Grid 路径：2D 精确映射，kernel 内无循环
            grid = (outer_size, num_inner_blocks)
            repeat_small_kernel[grid](out, output, outer_size, inner_size, r=r, BLOCK=BLOCK)
        else:
            # 大 Grid 路径：多核分区负载均衡
            grid = (min(total_blocks, self.VEC_CORE_NUM),)
            repeat_large_kernel[grid](out, output, outer_size, inner_size, num_inner_blocks,
                                      num_cores=grid[0], r=r, BLOCK=BLOCK)
```

**收益**：
- 小 grid 场景消除标量分区开销，schedule 效率提升
- 大 grid 场景保持原有负载均衡，无性能退化
- 综合 geomean 从 0.933x 提升至 0.992x

## 关键原则

1. **路径越少越好**：不要为了 dispatch 而 dispatch。通常 2 个路径即可覆盖绝大多数场景（小/大）。超过 3 个路径会增加维护成本和编译缓存压力。

2. **条件判断必须廉价**：Host 侧的 dispatch 条件（如 `total_blocks <= num_cores`）必须是纯 Python 标量比较，不能涉及 tensor 运算。

3. **语义严格一致**：所有路径的输出必须逐元素相等，不能因路径不同而引入精度差异或布局差异。

4. **grid 形状即策略**：选择 grid 形状时要考虑：
   - 1D grid `(N,)`：简单，适合 block 编号可线性映射的场景
   - 2D grid `(M, N)`：适合 outer/inner 两级分解，可直接用 `tl.program_id(0)` / `tl.program_id(1)` 定位
   - 避免 3D grid，Triton Ascend 对 3D grid 支持有限

5. **与 constexpr 配合**：多路径特化常与「入参静态化」协同使用。将路径相关的参数（如 `r`）设为 `tl.constexpr`，让每个路径在编译期获得最大优化。

## 常见陷阱

| 陷阱 | 说明 | 避免方法 |
|------|------|---------|
| **路径边界性能跳变** | 在 dispatch 阈值（如 `total_blocks == num_cores`）附近，两种路径性能差异过大 | 阈值选择要留有余量，或让两种路径在边界处性能接近 |
| **小路径grid过大** | 小 grid 路径的 grid 超过核数，退化成调度开销 | 确保小路径的 grid `total_blocks <= num_cores` |
| **2D grid索引错误** | 使用 `tl.program_id(0)` 线性化 2D grid，导致索引越界或重复 | 2D grid 应直接用 `program_id(0)` / `program_id(1)`，不要从一维 pid 推导 |
| **路径间代码复制** | 两个路径的 kernel 逻辑大量重复，维护困难 | 提取公共逻辑到 Python helper，或接受少量重复以保证性能 |
| **忽略编译缓存** | 过多路径导致 Triton 编译缓存膨胀，首次启动变慢 | 控制路径数量（<=3），并确保路径条件覆盖合理 |

## 与其他优化点的关系

- **与优化点 1（入参静态化）协同**：将 `r`、`num_cores` 等声明为 `tl.constexpr`，让每个路径编译出最优代码
- **与优化点 3（分核优化）互补**：分核优化关注「grid 是否合理」；本优化点关注「单一 grid 策略是否够用，是否需要多策略 dispatch」
- **与优化点 12（Autotune）的区别**：Autotune 是在同一 kernel 上尝试不同 `constexpr` 配置；本优化点是切换完全不同的 kernel 实现
