# 标量操作优化

## 概述

在 Triton NPU kernel 中，标量计算通常在 CPU 上执行，然后将结果传递给 kernel。将标量计算移入 kernel 或使用更高效的计算方式，可以减少数据传输开销并提升性能。

## 适用条件

代码中存在可优化的标量计算，例如：
- 循环内的标量算术运算
- 可在编译时计算的常量表达式
- 标量与向量的混合运算

## 优化方法

### 原始代码

```python
@triton.jit
def kernel(A, C, M, N, scale):
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    # scale 是标量，每次循环都需要传递
    a = tl.load(A + m_offsets[:, None] * stride_am + n_offsets[None, :])
    c = a * scale  # 标量乘法在 kernel 内执行
```

### 优化后代码

```python
@triton.jit
def kernel(A, C, M, N, scale: tl.constexpr):
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    # 将 scale 声明为 constexpr，编译器可进行常量折叠
    a = tl.load(A + m_offsets[:, None] * stride_am + n_offsets[None, :])
    c = a * scale
```

## 关键点

1. **常量折叠**：将可以在编译时计算的表达式声明为 constexpr
2. **减少数据传输**：将标量计算移入 kernel，减少 Host-Device 数据传输
3. **向量化**：将多个标量操作向量化，利用 NPU 的并行计算能力
