# 入参静态化 优化模式

## 概述

在 Triton NPU kernel 中，将固定数值的入参声明为 `tl.constexpr`，可以让编译器在编译时进行更多的常量折叠和常量传播优化，从而提升 kernel 的执行效率。

**关键洞察 — 启动级特化（Launch-Level Specialization）**：

Triton Ascend 的编译机制允许在**每次 `kernel[grid](...)` 启动时**，根据传入的 `tl.constexpr` 值生成特化代码。这意味着：
- 即使某个参数在 `forward()` 内的多次启动之间变化（如 dim0 启动时 `r=2`，dim1 启动时 `r=4`），只要它在**单次启动内固定**，就值得声明为 `tl.constexpr`
- 编译器会为每次启动生成对应常量值的特化机器码，带来 loop unroll、分支消除等激进优化

## 触发条件

**当代码中存在以下固定数值参数时，应考虑将其声明为 `tl.constexpr`**：

1. **固定的 BLOCK_SIZE**：如 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 等
2. **固定的 STRIDE**：如 `stride_m`、`stride_n` 等
3. **模型配置超参数**：如 MoE 场景中的 `num_experts`、`topk_numel`、`seq_len` 等。这些值在模型训练/推理过程中通常是固定配置（如 `num_experts=128`），不应仅凭变量名判断为运行时变量。若该参数来自 Python 层的固定配置，应优先尝试声明为 `tl.constexpr`
4. **启动级常量**（重点新增）：在单次 `kernel[grid]()` 调用期间不变的标量参数
   - 典型例子：`repeat` 次数 `r`、操作轴 `axis`、窗口大小 `window_size`、头数 `head_num`
   - 判断标准：该参数作为**关键字参数**以 Python 标量形式传入 kernel，且在 kernel 执行期间不变化
5. **其他在 kernel 生命周期内不会变化的常量参数**

**反例 — 不应声明为 `tl.constexpr` 的参数**：
- 张量数据指针（`input_ptr`, `output_ptr`）
- 动态维度（`M`, `N`, `K`, `batch_size`）
- 在 kernel 执行期间逐 thread 变化的标量（如每个 program 的独立缩放因子）

如果已有入参中的某个参数对性能影响很大，且在kernel生命周期内不会变化，如若不确定则应该**询问用户是否可以将该参数设置为 `tl.constexpr`**。

## 优化方法

### 原始代码

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_an,  # 这些是入参，但实际运行时是固定值
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ...
```

### 优化后代码

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am: tl.constexpr,  # 声明为 constexpr
    stride_an: tl.constexpr,  # 声明为 constexpr
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ...
```

### 启动级特化示例 — Repeat 算子

**场景**：`torch.repeat(*repeats)` 需要在 `forward()` 中对每个维度分别启动 kernel，各维度的 repeat 次数不同。

**原始代码（未优化）**：
```python
@triton.jit
def repeat_dim_kernel(x_ptr, out_ptr, outer_size, inner_size, num_inner_blocks, r, BLOCK: tl.constexpr):
    # r 作为普通入参
    for repeat_idx in range(r):  # 标量循环，无法 unroll
        ...

# forward 中多次启动，每次 r 不同
repeat_dim_kernel[grid](..., r=2)  # dim3
repeat_dim_kernel[grid](..., r=4)  # dim2
```

**优化后代码**：
```python
@triton.jit
def repeat_dim_kernel(x_ptr, out_ptr, outer_size, inner_size, num_inner_blocks, r: tl.constexpr, BLOCK: tl.constexpr):
    # r 声明为 constexpr
    for repeat_idx in range(r):  # 编译器根据每次启动的 r 值进行 loop unroll
        ...

# forward 中多次启动，每次传入不同的 constexpr r
repeat_dim_kernel[grid](..., r=2)  # 编译器特化为 r=2 的版本
repeat_dim_kernel[grid](..., r=4)  # 编译器特化为 r=4 的版本
```

**核心原理**：虽然 `r` 在两次 `kernel[grid]()` 之间变化，但每次启动时 `r` 是固定值。声明为 `tl.constexpr` 后，Triton Ascend 编译器在每次启动时生成对应 `r` 值的特化代码，将 `for repeat_idx in range(r)` 完全展开为顺序指令，消除循环计数器和分支判断的标量开销。

## 关键点

1. **常量性质**：只有那些在 kernel 单次启动执行期间不会变化的参数才适合声明为 `tl.constexpr`
2. **启动级变化不影响**：参数在 `forward()` 内的多次 `kernel[grid]()` 之间变化是允许的，因为每次启动都会重新编译/特化
3. **性能影响**：对于性能敏感的参数（如 BLOCK_SIZE），应优先考虑声明为 `tl.constexpr`
4. **循环 unroll 机会**：特别检查 kernel 内是否存在 `for i in range(some_param)` 的模式 —— 若 `some_param` 在单次启动内固定，将其设为 `tl.constexpr` 可直接触发编译期 loop unroll
5. **用户确认**：如果不确定某个参数是否可以设为 constexpr，应询问用户

## 性能收益

将固定参数声明为 `tl.constexpr` 可以：
- 启用编译时常量折叠
- 帮助编译器进行更 aggressive 的常量传播
- 减少运行时分支判断开销
- **触发循环 unroll**：将 `for i in range(constexpr_param)` 展开为顺序指令，消除标量循环开销（如 Repeat 算子中 `r: tl.constexpr` 带来的显著收益）
- **启动级特化**：为不同启动参数生成最优机器码，避免运行时动态分支
