# BLOCK_SIZE 调优

## 概述

BLOCK_SIZE（分块大小）是影响 Triton 算子性能的关键参数之一。合理的 BLOCK_SIZE 配置可以充分利用硬件的并行计算能力和缓存局部性，从而显著提升性能。

## 适用条件

代码中存在可调整的 BLOCK_SIZE 参数，且 BLOCK_SIZE 未经过充分调优。

## 调优原则

### 1. 考虑内存带宽和缓存

- 较大的 BLOCK_SIZE 可以更好地利用数据局部性，减少重复加载
- 但过大的 BLOCK_SIZE 会导致缓存竞争和并行度下降
- 需要在带宽利用率和并行度之间取得平衡

### 2. 考虑数据类型

- 不同数据类型的向量宽度不同，BLOCK_SIZE 应与向量宽度匹配
- fp32、fp16、bfp16 等类型可能有不同的最优 BLOCK_SIZE

## 调优方法

通过对不同的 BLOCK_SIZE 配置进行性能测试，找到最优值

### 3. 遵循经验规则

常见的经验规则：
- BLOCK_SIZE 通常选择 2 的幂次方


## 关键点

1. **避免过小 BLOCK_SIZE**：过小的 BLOCK_SIZE 会导致并行度不足，无法充分利用硬件

2. **避免过大 BLOCK_SIZE**：过大的 BLOCK_SIZE 会导致缓存竞争和资源浪费

3. **多维度协调**：BLOCK_M、BLOCK_N、BLOCK_K 需要联合调优，找到最优组合

4. **验证调优结果**：调优后必须进行功能和性能验证，确保优化有效

## 关键约束

### 核心规则

```
BLOCK_SIZE 必须 <= 被分块的维度大小
```

### 原因分析

当 BLOCK_SIZE > 维度大小时，padding 的 0.0 会被 `tl.sum()` 累加，污染统计结果。

```python
# ❌ 错误示例
W = 512
BLOCK_SIZE = 4096  # > W

for w_start in range(0, W, BLOCK_SIZE):  # 只循环一次
    w_offs = w_start + tl.arange(0, BLOCK_SIZE)  # [0, 1, ..., 4095]
    mask = w_offs < W  # 只有前 512 个是 True

    data = tl.load(..., mask=mask, other=0.0)
    # data = [有效数据×512, padding 0.0×3584]

    sum_val += tl.sum(data)  # 累加了 3584 个 0.0！
    # 结果：sum 被污染，后续计算全部错误
```

### padding 比例

```
padding 比例 = (BLOCK_SIZE - 维度大小) / BLOCK_SIZE
```

| BLOCK_SIZE | 维度大小 | padding 比例 | 结果 |
|-----------|---------|-------------|------|
| 512 | 512 | 0% | ✅ 正确 |
| 1024 | 512 | 50% | ❌ 错误 |
| 4096 | 512 | 87.5% | ❌ 错误 |
| 8192 | 262144 | 0.03% | ✅ 正确 |

## BLOCK_SIZE 的影响

### 优点：增大 BLOCK_SIZE

```python
# BLOCK_SIZE = 512
for start in range(0, 262144, 512):  # 循环 512 次
    ...

# BLOCK_SIZE = 8192
for start in range(0, 262144, 8192):  # 循环 32 次
    ...
```

- ✅ 减少循环迭代次数
- ✅ 减少循环开销
- ✅ 更大的数据块，更好的内存带宽利用

### 缺点：增大 BLOCK_SIZE

```python
# BLOCK_SIZE = 512
data = tl.load(..., other=0.0)  # 加载 512 个元素

# BLOCK_SIZE = 16384
data = tl.load(..., other=0.0)  # 加载 16384 个元素
```

- ❌ 更高的寄存器压力
- ❌ 可能降低并行度（SM 资源有限）
- ❌ 可能触发寄存器溢出

### 最优值需要实测

| BLOCK_SIZE | 循环次数 | 寄存器压力 | 性能 |
|-----------|---------|-----------|------|
| 512 | 多 | 低 | 基准 |
| 2048 | 中 | 中 | 较好 |
| 4096 | 少 | 中 | 好 |
| **8192** | **更少** | **中高** | **最优** |
| 16384 | 最少 | 高 | 可能更慢 |

## 案例：BatchNorm2d

### 测试配置

- 数据尺寸：(64, 64, 512, 512)
- 分块维度：H×W = 262144
- 可用 BLOCK_SIZE 范围：[1, 262144]

### 测试结果

| BLOCK_SIZE | 循环迭代次数 | 平均延迟 | 加速比 |
|-----------|-------------|---------|--------|
| 512 | 512 | ~20 ms | 基准 |
| 2048 | 128 | ~14 ms | 1.43x |
| 4096 | 64 | ~14 ms | 1.43x |
| **8192** | **32** | **11.67 ms** | **1.71x** |

### 代码示例

```python
@triton.jit
def batchnorm_kernel(
    input_ptr, output_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, stride_n, stride_c, stride_h, stride_w,
    eps: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    HW = H * W  # 262144
    channel_base = c * stride_c

    sum_val = 0.0
    sum_sq = 0.0

    for n in range(N):
        batch_offset = n * stride_n + channel_base
        for hw_start in range(0, HW, BLOCK_SIZE):
            hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
            mask = hw_offs < HW

            offsets = batch_offset + hw_offs
            data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

            sum_val += tl.sum(data)
            sum_sq += tl.sum(data * data)

    # ... 后续计算
```

### 调用代码

```python
class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        output = torch.empty_like(x)
        grid = (C,)

        # 不同 BLOCK_SIZE 测试
        BLOCK_SIZE = 8192  # 实测最优

        batchnorm_kernel[grid](
            x, output, self.gamma, self.beta,
            N, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
```

## 常见错误

### 错误 1：BLOCK_SIZE 超过分块维度

```python
# ❌ 错误：分块 W 维度 (512)，BLOCK_SIZE > W
W = 512
BLOCK_SIZE = 4096  # > W

for w_start in range(0, W, BLOCK_SIZE):
    # padding 87.5%，结果错误
```

### 错误 2：认为越大越好

```python
# ❌ 错误：盲目增大
BLOCK_SIZE = 16384  # 可能寄存器溢出

# ✅ 正确：实测确定
# 在 BatchNorm 中，8192 > 16384 性能更好
```

### 错误 3：忽略硬件限制

```python
# ❌ 错误：超过硬件限制
BLOCK_SIZE = 65536  # 可能超过共享内存或寄存器限制

# ✅ 正确：考虑硬件限制
# 最大 BLOCK_SIZE 通常受限于：
# - 寄存器数量
# - 共享内存大小
# - warp 大小（通常 32 或 64）
```

## 调优策略

### 方法 1：二分搜索

```python
# 从小到大测试
block_sizes = [512, 1024, 2048, 4096, 8192, 16384]
for bs in block_sizes:
    if bs > dim_size:
        break
    latency = benchmark(bs)
    print(f"BLOCK_SIZE={bs}: {latency} ms")
```

### 方法 2：Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['N', 'H', 'W'],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

### 经验值

| 场景 | 推荐 BLOCK_SIZE |
|------|----------------|
| 小维度 (< 1024) | 维度大小或 512 |
| 中维度 (1K-64K) | 2048 或 4096 |
| 大维度 (> 64K) | 4096 或 8192 |
| 统计算子 (mean, sum) | 可偏大 (8192) |
| 复杂算子 (conv, matmul) | 偏小 (256-1024) |

## 维度合并与 BLOCK_SIZE 的关系

### 维度合并扩展可用范围

```python
# 原始：分块 W 维度
W = 512
# 可用 BLOCK_SIZE: [1, 512]

# 维度合并：分块 H×W 维度
HW = H * W = 262144
# 可用 BLOCK_SIZE: [1, 262144]
```

**注意：能用更大的 BLOCK_SIZE 是维度合并的副作用，不是主要目的！**

维度合并的主要收益：
- 减少循环层数
- 减少重复计算
- 提高内存连续性

## 总结

| 优化 | 约束 | 方法 | 收益 |
|------|------|------|------|
| 调整 BLOCK_SIZE | BLOCK_SIZE <= 分块维度 | 实测确定最优值 | 减少循环迭代次数 |

**核心原则：**
1. 确保 `BLOCK_SIZE <= 分块维度大小`
2. BLOCK_SIZE 不是越大越好
3. 通过实测或 autotune 确定最优值
4. 考虑寄存器压力和并
