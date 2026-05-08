# Libdevice 函数使用 Skill for Triton-Ascend

<skill>
<name>libdevice-usage</name>
<description>Triton-Ascend libdevice 函数使用指南 — 提供已优化的基础函数库，避免重复造轮子，提升代码性能。</description>
<trigger>
当用户需要实现数学函数（如 round, sqrt, exp, sin 等）或激活函数时触发：
- 需要四舍五入、取整等操作
- 需要三角函数、双曲函数
- 需要指数、对数函数
- 需要特殊函数（gamma, erf 等）
</trigger>
</skill>

<role>
你是 Triton-Ascend libdevice 函数使用专家。你的任务是帮助用户找到并使用 `tl.extra.cann.libdevice` 中已有的优化函数，
避免从头实现，提升代码性能和可维护性。
</role>

## 核心原则

**优先使用 `tl.extra.cann.libdevice` 中已有的基础函数，不要从头实现。**

**Why:** libdevice 中已提供经过优化的基础函数，重新实现既浪费时间又可能性能更差。

**How to apply:**
1. 需要数学函数时，**先检查** `tl.extra.cann.libdevice` 是否已有实现
2. 找到后直接调用，再组合类型转换等操作
3. **不要从头造轮子**

> 完整函数列表源码：`triton-ascend/ascend/language/ascend/libdevice.py`

## 错误 vs 正确示例

### 示例 1: round 函数

```python
# ❌ 错误：从头实现 round
@triton.jit
def round_int8(x):
    return (x + 0.5).to(tl.int8)  # 重复造轮子，且逻辑不完整

# ✅ 正确：复用 libdevice 中的 round
@triton.jit
def round_int8(x):
    return tl.extra.cann.libdevice.round(x).to(tl.int8)
```

### 示例 2: relu 激活

```python
# ❌ 错误：手写 relu
@triton.jit
def relu_kernel(x_ptr, out_ptr, ...):
    x = tl.load(x_ptr + ...)
    out = tl.maximum(x, 0.0)  # 手写实现

# ✅ 正确：使用 libdevice relu
@triton.jit
def relu_kernel(x_ptr, out_ptr, ...):
    x = tl.load(x_ptr + ...)
    out = tl.extra.cann.libdevice.relu(x)  # 更高效
    tl.store(out_ptr + ..., out)
```

## 函数分类速查表

### 数学运算函数

| 函数 | libdevice 路径 | 用途 | 支持类型 |
|-----|---------------|------|---------|
| `round` | `tl.extra.cann.libdevice.round` | 四舍五入 | fp32 |
| `trunc` | `tl.extra.cann.libdevice.trunc` | 向零截断 | fp32 |
| `nearbyint` | `tl.extra.cann.libdevice.nearbyint` | 就近整数（银行家舍入） | fp32 |
| `pow` | `tl.extra.cann.libdevice.pow` | 幂运算 x^y | fp32, fp16, bf16, int |
| `reciprocal` | `tl.extra.cann.libdevice.reciprocal` | 倒数 1/x | fp32, fp16 |

### 三角函数

| 函数 | libdevice 路径 | 用途 | 支持类型 |
|-----|---------------|------|---------|
| `tan` | `tl.extra.cann.libdevice.tan` | 正切 | fp32, fp16 |
| `atan` | `tl.extra.cann.libdevice.atan` | 反正切 | fp32, fp16 |
| `atan2` | `tl.extra.cann.libdevice.atan2` | 双参数反正切 | fp32, fp16 |
| `acos` | `tl.extra.cann.libdevice.acos` | 反余弦 | fp32, fp16, bf16 |
| `asin` | `tl.extra.cann.libdevice.asin` | 反正弦 | fp32 |

### 双曲函数

| 函数 | libdevice 路径 | 用途 | 支持类型 |
|-----|---------------|------|---------|
| `sinh` | `tl.extra.cann.libdevice.sinh` | 双曲正弦 | fp32, fp16, bf16 |
| `cosh` | `tl.extra.cann.libdevice.cosh` | 双曲余弦 | fp32, fp16, bf16 |
| `tanh` | `tl.extra.cann.libdevice.tanh` | 双曲正切 | fp32, fp16 |
| `asinh` | `tl.extra.cann.libdevice.asinh` | 反双曲正弦 | fp32, fp16, bf16 |
| `acosh` | `tl.extra.cann.libdevice.acosh` | 反双曲余弦 | fp32, fp16, bf16 |
| `atanh` | `tl.extra.cann.libdevice.atanh` | 反双曲正切 | fp32, fp16, bf16 |

### 指数与对数函数

| 函数 | libdevice 路径 | 用途 | 支持类型 |
|-----|---------------|------|---------|
| `exp` | `tl.exp` (内置) | 指数函数 e^x | - |
| `expm1` | `tl.extra.cann.libdevice.expm1` | e^x - 1 | fp32, fp16, bf16 |
| `log` | `tl.log` (内置) | 自然对数 | - |
| `log1p` | `tl.extra.cann.libdevice.log1p` | ln(1+x) | fp32, fp16 |
| `log10` | `tl.extra.cann.libdevice.log10` | 以10为底对数 | fp32 |

### 激活与工具函数

| 函数 | libdevice 路径 | 用途 | 支持类型 |
|-----|---------------|------|---------|
| `relu` | `tl.extra.cann.libdevice.relu` | ReLU 激活 | fp32, fp16 |
| `abs` | `tl.abs` (内置) | 绝对值 | - |
| `sqrt` | `tl.sqrt` (内置) | 平方根 | - |
| `rsqrt` | `tl.rsqrt` (内置) | 逆平方根 | - |
| `hypot` | `tl.extra.cann.libdevice.hypot` | 欧氏距离 √(x²+y²) | fp32, fp16, bf16 |
| `copysign` | `tl.extra.cann.libdevice.copysign` | 复制符号 | fp32 |

### 特殊函数

| 函数 | libdevice 路径 | 用途 | 支持类型 |
|-----|---------------|------|---------|
| `erfinv` | `tl.extra.cann.libdevice.erfinv` | 逆误差函数 | fp32 |
| `gamma` | `tl.extra.cann.libdevice.gamma` | 伽马函数 Γ(x) | fp32 |
| `lgamma` | `tl.extra.cann.libdevice.lgamma` | 对数伽马函数 ln\|Γ(x)\| | fp32 |
| `cyl_bessel_i0` | `tl.extra.cann.libdevice.cyl_bessel_i0` | 修正贝塞尔函数 I₀ | fp16, fp32 |

### 浮点判断与操作

| 函数 | libdevice 路径 | 用途 | 支持类型 |
|-----|---------------|------|---------|
| `isnan` | `tl.extra.cann.libdevice.isnan` | 判断是否为 NaN | fp32, fp16, bf16 |
| `isinf` | `tl.extra.cann.libdevice.isinf` | 判断是否为无穷 | fp32, fp16, bf16 |
| `signbit` | `tl.extra.cann.libdevice.signbit` | 获取符号位 | fp16, fp32 |
| `nextafter` | `tl.extra.cann.libdevice.nextafter` | 下一可表示浮点数 | fp32, fp16, bf16 |
| `ldexp` | `tl.extra.cann.libdevice.ldexp` | ldexp(x, exp) | fp32, fp16 |
| `ilogb` | `tl.extra.cann.libdevice.ilogb` | 整数对数 | fp32, fp16 |

## 常用工具函数封装

### round_int8 - 四舍五入到 int8

**用途：** 将浮点数四舍五入后转换为 int8，常用于量化算子

```python
@triton.jit
def round_int8(x):
    """四舍五入到 int8，使用 libdevice 的 round 函数"""
    return tl.extra.cann.libdevice.round(x).to(tl.int8)
```

### clamp_int8 - 截断到 int8 范围

```python
@triton.jit
def clamp_int8(x):
    """将值截断到 int8 范围 [-128, 127]"""
    return tl.maximum(tl.minimum(x, 127), -128).to(tl.int8)
```

### safe_div - 安全除法

```python
@triton.jit
def safe_div(a, b, eps=1e-10):
    """安全除法，避免除零"""
    return a / (b + eps)
```

### safe_inv - 安全求逆

```python
@triton.jit
def safe_inv(x, eps=1e-10):
    """安全求逆，避免除零"""
    return 1.0 / (x + eps)
```

## 完整使用示例

### 量化场景

```python
@triton.jit
def quantize_round(x, scale):
    """量化：round + 类型转换"""
    x_scaled = x * scale
    return tl.extra.cann.libdevice.round(x_scaled).to(tl.int8)
```

### 激活函数场景

```python
@triton.jit
def activations_kernel(x_ptr, out_ptr, mode, ...):
    """多种激活函数"""
    x = tl.load(x_ptr + ...)

    if mode == "relu":
        out = tl.extra.cann.libdevice.relu(x)
    elif mode == "tanh":
        out = tl.extra.cann.libdevice.tanh(x)
    elif mode == "sigmoid":
        out = 1.0 / (1.0 + tl.exp(-x))  # sigmoid 无 libdevice 版本
    else:
        out = x

    tl.store(out_ptr + ..., out)
```

### 数学运算场景

```python
@triton.jit
def math_kernel(x_ptr, y_ptr, out_ptr, op, ...):
    """数学运算"""
    x = tl.load(x_ptr + ...)
    y = tl.load(y_ptr + ...)

    if op == "pow":
        out = tl.extra.cann.libdevice.pow(x, y)
    elif op == "hypot":
        out = tl.extra.cann.libdevice.hypot(x, y)
    elif op == "atan2":
        out = tl.extra.cann.libdevice.atan2(x, y)
    else:
        out = x

    tl.store(out_ptr + ..., out)
```

## 检查清单

在使用数学函数时，检查以下项目：

- [ ] 是否先检查 `tl.extra.cann.libdevice` 中是否有该函数
- [ ] 是否使用了正确的函数路径 (`tl.extra.cann.libdevice.xxx`)
- [ ] 是否注意了支持的类型（fp32/fp16/bf16）
- [ ] 是否需要组合类型转换 (`.to(tl.int8)` 等)
- [ ] 对于内置函数 (`tl.exp`, `tl.log`, `tl.abs` 等)，是否直接使用内置版本
