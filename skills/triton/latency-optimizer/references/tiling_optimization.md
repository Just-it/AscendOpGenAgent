# Tiling 优化

## 概述

Tiling（分块）优化是通过将大尺寸数据划分为更小的块（tile），使每个块能够更好地利用 Triton NPU 的局部性原理，提升缓存命中率和计算效率。

## 适用条件

代码中存在可优化的循环分块策略，例如：
- 矩阵乘法中的 BLOCK_M、BLOCK_N、BLOCK_K 分块
- 归约操作中的分块归约
- 元素级操作中的数据分块

## 优化方法

### 原始代码

```python
@triton.jit
def kernel(A, C, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    # ... 直接访问大块内存
```

### 优化后代码

```python
@triton.jit
def kernel(A, C, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    # 将大块划分为更小的 tile
    tile_m = BLOCK_M // 2
    tile_n = BLOCK_N // 2
    m_offsets = pid * BLOCK_M + tl.arange(0, tile_m)
    n_offsets = tl.arange(0, tile_n)
    # ... 逐 tile 处理，提升缓存效率
```

## 关键点

1. **分块大小**：分块大小需要根据硬件特性调整，过小导致并行度不足，过大导致缓存失效
2. **嵌套分块**：对于大尺寸操作，可采用多层分块策略
3. **向量化访问**：配合向量化内存访问可以进一步提升性能
