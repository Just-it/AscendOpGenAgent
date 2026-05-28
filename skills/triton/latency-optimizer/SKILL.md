---
name: latency-optimizer
description: >
  擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
  按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点，
  确保优化前后功能一致、精度一致。
  ⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。
argument-hint: >
  输入：code-file-path（代码文件路径）。
  输出：优化后的 Triton 代码、功能一致性说明、精度一致性说明。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# Latency Optimizer Skill

<role>
你是一个擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
你的任务是按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点。
**必须确保优化前后的功能一致性和精度一致性。**
**⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。**
</role>

## 优化点执行顺序

Agent 必须严格按照以下顺序逐一检查优化点，**每次只能尝试一个优化点，命中后参考对应文档**。

⚠️ **前置要求**：必须先命中某个优化点的「命中条件」（代码特征满足典型代码特征之一且适用条件成立），才能加载对应的参考文档。未命中则跳过，禁止加载参考文档。

---

### 优化点 1：入参静态化优化

**适用条件**：代码中存在可声明为 `tl.constexpr` 的固定参数

**典型代码特征**：
```python
@triton.jit
def kernel(A, B, C, M, N,
            stride_am, stride_an,  # 运行时不变化的固定值，但未声明为 constexpr
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr):
```

**判断逻辑**：
1. 遍历 kernel 参数列表，排除明确属于运行时变量的参数：
  - 张量数据指针（如 input_ptr, output_ptr）
  - 动态维度（如 batch size M/N/K、序列长度 seq_len）
  - **仅在单次 kernel 执行期间变化**的标量动态值（如逐元素的缩放因子，每个 thread 的值都不同）
2. 对剩余参数逐一检查是否满足"单次 kernel 启动后不变"（即该次 `kernel[grid](...)` 调用传入后，在整个 grid 执行期间不变）：
  - stride 参数（stride_am, stride_bn 等）→ 涉及
  - 固定索引（如 lse_idx, head_idx_offset）→ 涉及
  - BLOCK_SIZE / HEAD_DIM / N_ROUNDED 等配置参数 → 涉及
  - **启动级常量**（如 repeat 次数 `r`、操作轴 `axis`、reduce 维度 `dim`）→ **涉及**
    - 此类参数虽然在 `forward()` 内的多次 `kernel[grid]()` 之间可能变化，但在**单次启动内固定**
    - Triton Ascend 编译器会在每次启动时根据传入的 `constexpr` 值进行**启动级特化**（launch-level specialization），生成特化代码
    - 典型收益：触发 `for i in range(r)` 等循环的编译期 unroll，消除标量循环开销
3. 若第2步中任一参数未声明 `tl.constexpr` → 命中，进入参考文档
4. 若第2步中无参数或已全部声明 `tl.constexpr` → 不涉及，跳过

**命中条件**：代码特征满足上述典型代码特征之一，且适用条件成立

**参考文档**：`references/constexpr_parameters.md`

---

### 优化点 2：Tiling 优化（连续轴向量化）

**适用条件**：处理多维张量（3D 及以上）的规约类或归一化算子，且规约轴并非内存布局中的最连续轴

**典型代码特征**：
```python
@triton.jit
def kernel(input_ptr, output_ptr, dim1, dim2, ...):
    # 特征 1：向量化偏移 tl.arange 作用在非连续轴（如 dim1/M 轴）
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    # 特征 2：访存偏移计算中，向量化部分乘上了较大的 stride
    input_offset = m_offsets * stride_m + n_idx * stride_n
    # 特征 3：循环内部频繁进行还原操作（如 tl.sum）将向量压缩为标量
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    ...
    total_sum = tl.sum(acc, axis=0)
```

**判断逻辑**：
- 检查 `tl.load` 的偏移量计算：如果 `tl.arange` 产生的向量偏移量作用于 `stride > 1` 的轴，而存在 `stride = 1` 的轴仅被当作标量索引处理 → 涉及
- 检查循环累加器：如果累加器在还原轴上分块，但访存模式导致了非连续内存读取 → 涉及
- 如果 `tl.arange` 已经作用于内存最连续的轴（通常是最后一张量的最后一维），且实现了合并访存 → 不涉及，跳过

**命中条件**：代码逻辑旨在对某维度进行还原，但其分块策略导致硬件执行了跨步访存

**参考文档**：`references/tiling_optimization.md`

---

### 优化点 3：分核优化

**适用条件**：代码中 Grid 大小设置不合理，或未充分利用 NPU 硬件资源

**典型代码特征**：
```python
# 特征 1：Grid 远大于物理核数
grid = (batch_size,)  # 如果 batch_size=128，远超 48 核

# 特征 2：Grid 远小于物理核数
grid = (batch_size // 64,)  # 如果 batch_size=128，只有 2 核

# 特征 3：每个 program 只处理 1 行数据
row_idx = tl.program_id(0)
x = tl.load(ptr + row_idx * stride + cols, mask=mask)

# 特征 4：未使用编译优化选项（multibuffer、unit_flag）
kernel[grid](...)  # 未传入 multibuffer、unit_flag

```

**判断逻辑**：
- 检查 Grid 大小是否接近物理核数（40-48）
  - 如果 Grid >> 48 或 Grid << 48 或者 Grid值无从判断 → 涉及
- 检查每个 program 处理的数据量
  - 如果每个 program 只处理少量数据（如 1 行）→ 涉及
- 检查是否使用了编译优化选项
  - 如果未使用 multibuffer 且是内存密集型算子 → 涉及
- 如果 Grid 合理且已使用优化选项 → 不涉及，跳过

**命中条件**：代码中 Grid 大小设置不合理，或未充分利用 NPU 硬件资源

**参考文档**：`references/vector_core_partition.md`

---

### 优化点 4：离散访存优化

**适用条件**：代码中存在通过随机/不可预测索引访问全局内存

**典型代码特征**：
```python
# 索引来源于 tl.load 加载的值（随机性）
idx = tl.load(indices_ptr + offset)  # idx 是运行时确定的随机值
val = tl.load(data_ptr + idx)        # 通过随机索引访问

# 或者索引来源于 kernel 入参（可能是随机值）
val = tl.load(ptr + random_index)
```

**判断逻辑**：
- 检查 `tl.load` 的索引来源：
  - 如果索引是 `tl.program_id` 线性变换 → 确定性连续，不涉及
  - 如果索引是循环变量线性变换 → 确定性步长，不涉及
  - 如果索引来源于 `tl.load` 加载的值或 kernel 入参 → 潜在随机，涉及
- 如果所有访存索引都是确定性连续/步长模式 → 不涉及，跳过

**命中条件**：代码中存在通过随机/不可预测索引访问全局内存

**参考文档**：`references/discrete_memory_access.md`

---

### 优化点 5：Scalar 转 Vector 优化

**适用条件**：代码中存在标量操作，可转换为向量操作以充分利用 NPU Vector 计算单元

**典型代码特征**：
```python
# 特征 1：标量广播操作
scalar_val = 0.5  # Python 标量
result = x * scalar_val  # scalar 广播，无法启用 vector 加速

# 特征 2：标量规约操作
sum_val = 0.0  # 标量累加器
for n in range(N):
    val = tl.load(x_ptr + n)
    sum_val += val  # 标量加法

# 特征 3：标量控制流
if x > 0:  # 标量条件，导致 warp divergence
    result = tl.exp(x)
else:
    result = tl.cos(x)

# 特征 4：int 类型比较/除法/取余
is_invalid = tok < 0  # int 类型比较，退化为标量
c = a // b  # int 类型除法，退化为标量
d = a % b   # int 类型取余，退化为标量

# 特征 5：atomic_* 标量操作
for idx in range(0, BLOCK_SIZE):
    tl.atomic_add(output_ptr + idx, block_sum)  # 标量的原子加
```

**判断逻辑**：
- 检查是否存在 Python 标量与向量数据的计算（标量广播）
- 检查是否存在标量累加器（如 `sum_val = 0.0`）
- 检查是否存在 `if-else` 控制流处理向量数据
- 检查是否存在 `int32/int64` 类型的比较、除法、取余操作
- 检查是否存在 `atomic_add` 这一类的 `atomic_*` 标量操作
- 如果存在以上任一情况 → 涉及
- 如果所有操作都已使用向量形式 → 不涉及，跳过

**命中条件**：代码中存在标量操作，可转换为向量操作

**参考文档**：`references/scalar_to_vector.md`

---

### 优化点 6：避免向量API标量降级

**适用条件**：代码中存在可能被编译器降级为标量循环的向量操作，包括通用算术操作、比较操作、扩展乘法、累积操作（cumsum/cumprod）或归约操作（reduce）

**典型代码特征**：
```python
# 特征 1：通用算术操作使用 i64，或者满足降级条件
z = x + y  # x/y 为 i64
z = x % y  # x/y 为 i32且执行取余计算

# 特征 2：整数比较操作（非 i32 EQ/NE，或非浮点比较）
mask = x < y  # i8/i16/i32/i64 的 LT/GT/LE/GE 比较

# 特征 3：扩展乘法
z = x * y  # 触发 vmulext，始终降级

# 特征 4：cumsum/cumprod 在最后一个维度上操作
x_cumsum = tl.cumsum(x_1d, axis=0)  # 一维张量，或 cumDim 是 lastDim

# 特征 5：reduce 操作在特定条件下
# i64 类型的 sum/prod/max/min
# 整数类型的 argmax/argmin
# 浮点类型 argmax/argmin 且 flatten 后维度 > 2
```

**判断逻辑**：
- 检查通用算术操作（add/sub/mul/min/max/abs/shl/shr/interleave/deinterleave）：如果数据类型为 i64
- 检查比较操作：如果数据类型为 i8/i16/i64（所有比较），或 i32 的 LT/GT/LE/GE → 涉及
- 检查取余操作：如果数据类型是任何int类型 → 涉及
- 检查扩展乘法（vmulext）：任何扩展乘法 → 涉及
- 检查 cumsum/cumprod：如果累积维度是输入张量的最后一个维度（一维时 axis=0 即最后维度），或数据类型为 i64 → 涉及
- 检查 reduce 操作：如果是 i64 类型的 sum/prod/max/min；整数类型的 argmax/argmin；浮点类型 argmax/argmin 且 flatten 后维度 > 2 → 涉及
- 如果以上情况均不存在 → 不涉及，跳过

**命中条件**：代码中存在上述任一向量操作，且满足对应的标量降级条件

**参考文档**：`references/avoid_scalar_lowering.md`

---

### 优化点 7：Pass 合并优化

**适用条件**：代码中存在多次遍历相同数据计算不同统计量

**典型代码特征**：
```python
# Pass 1: 计算 mean
for ...:
    data = tl.load(...)
    mean += tl.sum(data)

# Pass 2: 计算 variance（再次遍历！）
for ...:
    data = tl.load(...)  # 重复加载
    var += tl.sum((data - mean) ** 2)

# Pass 3: 归一化（第三次遍历！）
for ...:
    data = tl.load(...)  # 第三次加载
    tl.store(...)
```

**判断逻辑**：
- 检查是否存在多个独立的循环遍历相同数据
- 检查是否可以同时计算多个统计量（如 sum + sum_sq 可同时计算 mean + var）
- 如果存在多次遍历且可合并 → 涉及
- 如果只有单次遍历，或统计量之间有依赖无法合并 → 不涉及，跳过

**命中条件**：代码中存在多次遍历相同数据，且可以合并计算

**参考文档**：`references/pass-merge.md`

---

### 优化点 8：维度合并优化

**适用条件**：代码中存在多层嵌套循环处理连续维度，且维度间无依赖关系

**典型代码特征**：
```python
# 问题代码：3层循环处理 NCHW 布局
for n in range(N):           # 64 次
    for h in range(H):       # 512 次
        for w_start in range(0, W, BLOCK_SIZE):  # 循环层数过多
            base_offset = n * stride_n + c * stride_c + h * stride_h
            data = tl.load(input_ptr + base_offset + ...)
```

**判断逻辑**：
- 检查是否存在多层嵌套循环（3层及以上）
- 检查循环维度是否为连续内存布局（如 NCHW 的 H×W）
- 检查维度间是否有依赖关系
- 如果存在多层循环且维度连续、无依赖 → 涉及
- 如果循环层数较少，或维度间有依赖 → 不涉及，跳过

**命中条件**：代码中存在多层嵌套循环处理连续维度，且可合并

**参考文档**：`references/dimension-merge.md`

---

### 优化点 9：Libdevice 函数使用

**适用条件**：代码中存在手动实现的数学函数，而 `tl.extra.cann.libdevice` 中已有优化版本

**典型代码特征**：
```python
# 手动实现 round
return (x + 0.5).to(tl.int8)

# 手动实现 relu
out = tl.maximum(x, 0.0)

# 手动实现 tanh、sinh、pow 等数学函数
```

**判断逻辑**：
- 检查代码中是否手动实现了以下函数：round、trunc、relu、tanh、sinh、cosh、pow、atan、acos、asin、expm1、log1p、hypot 等
- 如果存在手动实现且 `tl.extra.cann.libdevice` 中有对应函数 → 涉及
- 如果代码中没有数学函数实现，或已使用 libdevice 版本 → 不涉及，跳过

**命中条件**：代码中存在手动实现的数学函数，且 libdevice 中有优化版本

**参考文档**：`references/libdevice-usage.md`

---

### 优化点 10：循环不变量外提

**适用条件**：代码中存在嵌套循环，且内层循环中有只依赖外层变量的 `tl.load`

**典型代码特征**：
```python
# 问题代码：内层循环重复加载相同值
for outer_idx in range(outer_size):
    for inner_idx in range(inner_size):
        param_idx = outer_idx  # 只依赖外层变量
        val = tl.load(param_ptr + param_idx)  # 重复加载相同值
        ...

# 或者通过整除映射到更粗粒度
for block in range(num_blocks):
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channel = offsets // spatial_size
    w = tl.load(weight_ptr + channel)  # 相同 channel 重复加载
```

**判断逻辑**：
- 检查是否存在嵌套循环结构
- 检查内层循环中是否有 `tl.load(param_ptr + index_expr)`
- 检查 `index_expr` 是否只依赖外层循环变量，不依赖内层循环变量
- 如果存在且内层循环次数 >> 外层循环次数 → 涉及
- 如果没有嵌套循环，或所有 load 都依赖内层变量 → 不涉及，跳过

**命中条件**：代码中存在嵌套循环，且内层循环中有只依赖外层变量的 `tl.load`

**参考文档**：`references/loop-invariant-hoisting.md`

---

### 优化点 11：Load 指令重排序

**适用条件**：代码中存在循环，且循环内有多个 `tl.load` 和 `tl.store`，存在数据依赖导致的阻塞

**典型代码特征**：
```python
for i in range(HEAD_NUM):
    # load B 在前，会等待上一次循环的 store B
    idx_B = tl.load(p_B_index)
    b_B = tl.load(p_B)
    
    # load A 在后，必须等 load B 完成
    b_A = tl.load(p_A)
    
    # calculation
    b_O = b_A * b_B
    
    # store
    tl.store(p_O, b_O)
    tl.store(p_B, b_B)  # store B 会阻塞下一次循环的 load B
```

**判断逻辑**：
- 检查是否存在循环结构
- 检查循环内是否有多个 `tl.load` 和 `tl.store`
- 检查是否存在 `load A` 与 `store B` 之间没有数据依赖，但被其他依赖阻塞的情况
- 如果存在可重排序的 load 指令 → 涉及
- 如果循环内只有一个 load，或所有 load 都有依赖关系 → 不涉及，跳过

**命中条件**：代码中存在循环，且有 load 指令可以通过重排序提前发射

**参考文档**：`references/load-order.md`

---

### 优化点 12：Grid 形状与多路径特化

**适用条件**：单一 kernel 实现无法在不同 workload 规模下同时达到最优，且 Host 侧可在运行时根据 workload 特征选择不同 kernel 路径

**典型代码特征**：
```python
# 特征 1：grid 被钳制到核数，导致小 workload 时调度开销占比高
grid = (min(total_blocks, num_cores),)

# 特征 2：kernel 内存在兼容大小 grid 的通用循环结构
blocks_per_core = total_blocks // num_cores
remainder = total_blocks % num_cores
if pid < remainder:
    my_blocks = blocks_per_core + 1
    ...
for block_idx in range(start_block, start_block + my_blocks):
    ...  # 小 grid 时循环只执行 1 次，但分区计算无法消除

# 特征 3：同一算子同时存在 total_blocks << num_cores 和 total_blocks >> num_cores 两种 workload
```

**判断逻辑**：
1. 检查 grid 计算逻辑：是否存在 `min(total_blocks, num_cores)`、`clamp(grid, ...)` 等钳制逻辑
2. 检查 kernel 内部：是否存在为了兼容“program 可能处理多 block”而引入的标量分区循环、分支判断
3. 检查 workload 分布：同一算子在不同 shape 下是否同时出现以下两种场景：
   - `total_blocks <= num_cores`（小 grid，每个 program 本可直接映射 1 个 block）
   - `total_blocks > num_cores`（大 grid，必须进行多核分区）
4. 如果以上任一成立 → 涉及

**命中条件**：单一 kernel 无法同时最优覆盖小 grid 和大 grid 场景，且 Host 侧有条件做动态 dispatch

**参考文档**：`references/grid-dispatch-specialization.md`

---

### 优化点 13：Autotune 自动调优

**适用条件**：代码中存在一个或者多个可调参数（例如BLOCK_SIZE、BLOCK_M等），且这些参数未经过充分调优，考虑到其他优化点可能引入可调超参数，最后再优化该优化点

**典型代码特征**：
```python
# 未使用 autotune，手动指定固定参数
@triton.jit
def kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...

# 调用时固定参数
kernel[grid](..., BLOCK_M=128, BLOCK_N=128)
```

**判断逻辑**：
- 检查是否已使用 `@triton.autotune` 装饰器
- 检查是否存在多个可调的 `tl.constexpr` 参数
- 如果未使用 autotune 且存在可调参数 → 涉及
- 如果已使用 autotune → 不涉及，跳过

**命中条件**：代码中存在多个可调参数，且未使用 autotune

**参考文档**：`references/autotune.md`

---

### 优化点 14：混合策略自动选择

**适用条件**：同一算子在不同 shape 或数据类型下需要不同优化策略

**典型代码特征**：
```python
# 问题：单一策略无法覆盖所有 shape
if some_condition:
    # 策略 A: 适合小 shape
    kernel_a[grid](...)
else:
    # 策略 B: 适合大 shape
    kernel_b[grid](...)
```

**判断逻辑**：
- 检查是否存在 shape 相关的条件分支选择不同 kernel
- 检查是否存在数据类型相关的条件分支选择不同策略
- 检查不同策略是否针对不同的性能瓶颈（如 small grid vs large grid）
- 若存在 → 涉及

**参考策略**：
- small batch / small groups → 并行规约（atomic_add）
- large batch / large groups → 原始规约（避免 atomic 开销）
- fp32 → 禁用改变求和顺序的优化
- fp16/bf16 → 可启用并行优化

**命中条件**：代码中存在 shape 或数据类型相关的条件分支选择不同 kernel 或策略

**参考文档**：`references/mixed_strategy.md`

---

### 优化点 14：维度合并与大 BLOCK 累加（归一化算子专用）

**适用条件**：
- 算子类型为 BatchNorm / LayerNorm / GroupNorm / InstanceNorm / RMSNorm / Softmax
- 代码中存在对 stats unit（group / row / channel）内元素的归约操作
- 当前实现使用嵌套循环或多通道分块累加

**典型代码特征（问题模式）**：
```python
# 特征 1：嵌套循环处理连续维度
for c in range(c_start, c_end):
    for hw_block in range(0, L, BLOCK_HW):
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(vals)  # 小量多次标量累加

# 特征 2：mask 覆盖率过低
BLOCK_HW = 256
L = H * W  # 若 L=16，mask 覆盖率仅 6.25%

# 特征 3：标量累加次数远大于向量化加载次数
# 如：3584 次标量累加 vs 14 次向量化加载
```

**判断逻辑**：
1. 检查 stats kernel 中是否存在嵌套循环处理连续维度
2. 检查 `tl.load` 的 mask 覆盖率是否 < 50%
3. 检查标量累加次数是否 > `max(16, total_elements / 4096)`
4. 若任一条件满足 → 命中

**优化动作**：
1. 将 stats unit 内所有元素展平为一维连续块：
   `group_elements = channels_per_group * HW`
2. 基地址直接定位到 stats unit 起始：
   `x_base = x_ptr + n * CHW + g * channels_per_group * HW`
3. 使用单循环大 BLOCK 遍历：
   ```python
   for offset in range(0, group_elements, BLOCK_SIZE):
       idx = offset + tl.arange(0, BLOCK_SIZE)
       mask = idx < group_elements
       val = tl.load(x_base + idx, mask=mask, other=0.0).to(tl.float32)
       mean_acc += tl.sum(val, axis=0)
       var_acc += tl.sum(val * val, axis=0)
   ```
4. BLOCK_SIZE 自适应选择：
   | group_elements | fp32 | fp16/bf16 |
   |---------------|------|-----------|
   | < 1024 | 向上取整到 2^n | 向上取整到 2^n |
   | 1024 ~ 8191 | 1024 | 1024 |
   | 8192 ~ 32767 | 1024 | 2048 |
   | >= 32768 | 1024 | 4096 |

**预期收益**：
- 性能：减少循环开销，提高向量利用率，减少 mask 浪费
- 精度：减少标量累加次数（从数千次降到十几次），避免 float16 累积误差
- 典型提升：0.3x → 0.8x（同时解决精度失败）

**验证要求**：
- 精度验证必须通过（特别关注 `num_groups=1, C` 很大、`HW` 很小的 case）
- 性能不劣化

**参考文档**：`../kernel-generator/references/triton-ascend-reduce.md`（"Stats Kernel 精度保障：累加模式规范"章节）

---

## 优化流程
```
1. 按顺序检查优化点 1 → 2 → 3 → ... → 13 → 14
2. 对于当前优化点，先判断是否命中（代码特征满足 + 适用条件成立）：
   - 未命中 → 跳过，检查下一优化点
   - 命中 → 参考对应文档，应用优化策略
3. 应用优化后，必须加载 references/checklist.md 检查代码规范
4. 如果代码规范不满足 → 修改代码直到满足规范
5. 代码规范满足后 → 返回优化后的代码，回到1继续检查优化点
```

**重要约束**：
- ⚠️ **只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式**
- ⚠️ **必须先命中优化点的「命中条件」，才能加载参考文档；未命中则跳过**
- 一次优化迭代只能使用一个优化点，可以有多轮优化，示例：
```
  第一轮：检查 1→2→3→...，命中优化点 X，应用后验证
  第二轮：检查 1→2→...，命中优化点 Y，应用后验证
  第三轮：检查 1→2→...，命中优化点 Z，应用后验证
  ...
  直到所有优化点都不命中
```
- 一次只能参考一个文档

## 优化验证规则

**⚠️ 强制要求：在进行任何精度验证或性能验证之前，必须先执行 checklist 检查，确保所有代码规范都已满足。验证流程如下：**

1. **Checklist 检查**：加载 `references/checklist.md`，逐项检查代码是否满足所有规范要求
2. **不满足规范** → 修改代码直到满足所有规范要求，然后重新执行 checklist 检查确认
3. **满足规范后** → 执行精度验证和性能验证

- **成功**：优化后的性能不劣化（speedup ≥ 1.0），该优化结果作为下一次优化迭代的基线
- **失败**：优化后的性能劣化（speedup < 1.0），放弃本次优化结果，以优化前的代码作为下一次优化迭代的基线

## 参考资料索引

| 文档类型 | 文档路径 |
|----------|----------|
| 入参静态化优化 | `references/constexpr_parameters.md` |
| Tiling 优化 | `references/tiling_optimization.md` |
| 分核优化 | `references/vector_core_partition.md` |
| 离散访存优化 | `references/discrete_memory_access.md` |
| Scalar 转 Vector 优化 | `references/scalar_to_vector.md` |
| 避免向量API标量降级 | `references/avoid_scalar_lowering.md` |
| Pass 合并优化 | `references/pass-merge.md` |
| 维度合并优化 | `references/dimension-merge.md` |
| Libdevice 函数使用 | `references/libdevice-usage.md` |
| 循环不变量外提 | `references/loop-invariant-hoisting.md` |
| Load 指令重排序 | `references/load-order.md` |
| Grid 形状与多路径特化 | `references/grid-dispatch-specialization.md` |
| Autotune 自动调优 | `references/autotune.md` |
| 混合策略自动选择 | `references/mixed_strategy.md` |
| 维度合并与大 BLOCK 累加 | `../kernel-generator/references/triton-ascend-reduce.md` |
| 代码规范检查 | `references/checklist.md` |
