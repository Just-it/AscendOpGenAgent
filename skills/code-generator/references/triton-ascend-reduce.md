# Reduce 算子优化

> 适用于需要聚合多个值的归约操作

## 适用算子

**基础归约**: sum, mean, max, min, prod
**归一化**: softmax, logsoftmax, layernorm, batchnorm
**统计**: variance, std

## 通用归约策略

### 1. 块内归约 + 原子操作

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 块内归约
    block_sum = tl.sum(data, axis=0)
    
    # 原子操作写回全局内存
    tl.atomic_add(output_ptr, block_sum)
```

### 2. 数值稳定性处理

**关键**: 对于涉及 exp 的操作（softmax、logsoftmax），必须减去最大值防止溢出。

```python
# 错误：错误：直接 exp 可能溢出
scores = tl.math.exp2(x)

# 正确：正确：减去最大值
max_val = tl.max(x, axis=0)
scores = tl.math.exp2(x - max_val)
```
