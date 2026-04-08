# Core Partition 分核优化模式

## 概述

在 Triton NPU kernel 中，**分核策略直接影响硬件利用率和性能**。Ascend 910B4 有 20 个 AI Core（共 40 个 VEC），选择合适的核数、分核维度和任务分配方式是性能优化的关键。

## 触发条件

**当 Triton 代码中存在以下情况时，应考虑优化分核策略**：

1. **核数不合理**：grid 大小与数据规模不匹配（过多或过少）
2. **负载不均衡**：某些核空闲或处理数据量差异大
3. **极端数据形状**：M << N 或 N << M 的场景
4. **小数据量**：数据量 < 100KB 时使用过多核
5. **UB 溢出风险**：tile 大小超过 192KB UB 容量

## 硬件约束

| 硬件资源 | 数量      | 说明                                    |
| -------- | --------- | --------------------------------------- |
| AI Core  | 20        | 每个 AI Core 包含 2 VEC + 1 CUBE + 1 SU |
| VEC      | 40        | 向量计算单元，可并行执行                |
| UB       | 192KB/VEC | Unified Buffer，每个 VEC 独立           |

## 优化方法

### 1. 核数选择优化

#### 原始代码（核数过多）

```python
# 小数据量使用过多核
N = 1024
BLOCK_SIZE = 64
grid = (triton.cdiv(N, BLOCK_SIZE),)  # grid = 16，调度开销大
```

#### 优化后代码（合理核数）

```python
# 小数据量使用少量核
N = 1024
BLOCK_SIZE = 256
grid = (triton.cdiv(N, BLOCK_SIZE),)  # grid = 4，调度开销合理
```

### 2. Reduction 分核维度优化

#### 原始代码（非 reduce 轴分核，负载不均）

```python
# 数据形状：(16, 262144)，沿 axis=1 归约
M, N = 16, 262144
BLOCK_M = 16
grid = (triton.cdiv(M, BLOCK_M),)  # grid = 1，只用 1 个核！
```

#### 优化后代码（reduce 轴分核 + 原子操作）

```python
# 数据形状：(16, 262144)，沿 axis=1 归约
M, N = 16, 262144
BLOCK_N = 8192
SUB_BLOCK_N = 1024  # 二次切分避免 UB 溢出
grid = (triton.cdiv(N, BLOCK_N),)  # grid = 32，充分利用多核

@triton.jit
def reduce_min_kernel(x_ptr, y_ptr, M, N, BLOCK_N: tl.constexpr, SUB_BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    
    row_min = tl.full([M], float('inf'), dtype=tl.float32)
    
    # 二次切分
    for sub_start in range(0, BLOCK_N, SUB_BLOCK_N):
        n_offsets = n_start + sub_start + tl.arange(0, SUB_BLOCK_N)
        mask = n_offsets < N
        x = tl.load(x_ptr + tl.arange(0, M)[:, None] * N + n_offsets[None, :], mask=mask)
        local_min = tl.min(x, axis=1)
        row_min = tl.minimum(row_min, local_min)
    
    # 原子操作跨核归约
    for m in range(M):
        tl.atomic_min(y_ptr + m, row_min[m])
```

### 3. 任务分配优化（非均匀分配）

#### 原始代码（简单除法，边界处理不当）

```python
n_cores = 32
total_tasks = 1000
tasks_per_core = total_tasks // n_cores  # 31，剩余 8 个任务无人处理

# Kernel 内
pid = tl.program_id(0)
start = pid * tasks_per_core
end = start + tasks_per_core  # 最后 8 个任务丢失！
```

#### 优化后代码（base + rem 模式）

```python
n_cores = 32
total_tasks = 1000
base = total_tasks // n_cores  # 31
rem = total_tasks % n_cores    # 8

# Kernel 内
pid = tl.program_id(0)
if pid < rem:
    task_start = pid * (base + 1)
    task_end = task_start + (base + 1)
else:
    task_start = rem * (base + 1) + (pid - rem) * base
    task_end = task_start + base
# 总计：8*32 + 24*31 = 256 + 744 = 1000 ✓
```

## 核数选择建议

| 数据规模    | 推荐核数 | 理由           |
| ----------- | -------- | -------------- |
| < 1KB       | 1        | 调度开销主导   |
| 1KB - 100KB | 1-8      | 平衡调度和计算 |
| 100KB - 1MB | 8-16     | 开始受益于并行 |
| > 1MB       | 16-40    | 充分利用硬件   |

## 分核维度决策

| 算子类型    | 数据特征 | 分核维度                    |
| ----------- | -------- | --------------------------- |
| Elementwise | 任意     | 按数据量均分                |
| Reduction   | M >> N   | 按非 reduce 轴分核          |
| Reduction   | M << N   | 按 reduce 轴分核 + 原子操作 |
| MatMul      | 通用     | 按 M-N 维度 2D 分核         |

## 性能收益

- **核数优化**：小数据量场景可提升 2-5x 性能
- **分核维度优化**：极端形状场景可提升 2-3x 性能
- **负载均衡**：避免部分核空闲，提升整体吞吐

## 注意事项

1. **UB 容量约束**：确保 tile_size * dtype_size * buffers <= 192KB
2. **原子操作开销**：原子操作有额外开销，核数过多时可能成为瓶颈
3. **对齐要求**：数据传输需 256B 对齐
