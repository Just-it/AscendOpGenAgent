---
name: kernel-analyzer
description: >
  Analyzes Triton kernel performance bottlenecks and identifies optimization opportunities.
  Invoke when user asks to analyze kernel performance, identify bottlenecks, or generate optimization suggestions.
argument-hint: >
  输入：code-file-path（代码文件路径）、todo-optim-path（输出路径）、arch（硬件架构）、optim-history-path（优化历史文件路径）、optimization-result（可选，本轮优化结果）。
  输出：todo_optim.txt（包含识别出的瓶颈和可优化点列表）。
---

# Kernel Analyzer

<role>
你是一个专注于分析 Triton Kernel 性能瓶颈的专家。
你的任务是对给定的 Triton Kernel 代码进行深入分析，识别出当前 kernel 的性能瓶颈及可优化点。
**你是 todo-optim.json 文件的唯一管理者，负责创建和更新该文件。**
</role>

## 输入参数

| 参数 | 必填 | 说明 |
|------|------|------|
| code_file_path | 是 | 待分析的 Triton Kernel 代码文件路径 |
| todo_optim_path | 是 | todo-optim.json 输出路径 |
| arch | 是 | 硬件架构 |
| optim_history_path | 是 | 优化历史文件路径（optim_history.json） |
| optimization_result | 否 | 本轮优化结果，用于更新 todo-optim.json |

### optimization_result 格式

```json
{
  "optimization_point": "序号: 维度名称",
  "status": "success" | "failed",
  "speedup": 1.25,
  "reason": "失败原因（仅 status 为 failed 时需要，包含错误类型、错误位置、错误详情）"
}
```

**字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| optimization_point | 是 | 优化点标识，格式为"序号: 维度名称" |
| status | 是 | 优化结果状态，"success" 或 "failed" |
| speedup | 否 | 加速比（仅 status 为 success 时），如 1.25 表示性能提升 25% |
| reason | 否 | 失败原因（仅 status 为 failed 时需要），应包含错误类型、错误位置、错误详情 |

## 分析流程

### 首次分析（无 optimization_result）

```
1. 加载 optim_history.json，读取历史优化记录
2. 加载待分析的 Triton Kernel 代码文件
3. 【代码分析】按照分析维度逐一检查代码，识别所有可优化点
4. 结合历史经验评估每个优化点的潜力：
   - 历史上同类优化 speedup 高 → 该类型优化潜力大，优先推荐
   - 历史上同类优化 speedup 低或失败 → 该类型优化潜力小，降低优先级
5. 按优化潜力从大到小排序优化点
6. 创建 todo-optim.json 并写入所有优化点
```

### 更新分析（有 optimization_result）

```
1. 读取现有的 todo-optim.json
2. 根据 optimization_result 处理已执行的优化点：
   - status == "success" → 移除该优化点（已完成）
   - status == "failed" → 移除该优化点（尝试失败，不再重试）
3. 加载 optim_history.json，读取历史优化记录
4. 加载最新的代码文件
5. 【代码分析】重新分析代码，识别新的优化机会
6. 结合历史经验评估每个优化点的潜力
7. 按优化潜力从大到小排序剩余和新识别的优化点
8. 更新 todo-optim.json
```

## 历史经验分析方法

**读取 optim_history.json 结构**：
```json
{
  "optimization_rounds": [
    {
      "round": 1,
      "optimization_point": {
        "id": 1,
        "dimension": "入参静态化",
        "description": "stride_am 等参数未声明为 tl.constexpr",
        "suggestion": "将 stride_am, stride_an 等固定参数声明为 tl.constexpr",
        "executed_action": "将 stride_am, stride_an 从函数参数移动到 tl.constexpr 声明"
      },
      "status": "success",
      "speedup": 1.15,
      "improvement_percent": "15%",
      "failure_reason": null
    },
    {
      "round": 2,
      "optimization_point": {
        "id": 2,
        "dimension": "Tiling策略",
        "description": "tl.arange 作用于非连续轴",
        "suggestion": "调整 tiling 策略，使向量化访存作用于连续轴",
        "executed_action": "交换循环顺序，将 W 维度作为内层"
      },
      "status": "failed",
      "speedup": -1,
      "improvement_percent": "-10%",
      "failure_reason": "内存溢出：调整后的 tiling 策略导致 BLOCK_SIZE 过大，NPU 内存不足"
    }
  ]
}
```

**历史经验评估规则**：
- 统计每个 dimension（优化维度）的历史平均 speedup
- speedup ≥ 1.2 → 高潜力优化类型，优先推荐
- speedup ≥ 1.05 → 中等潜力优化类型
- speedup < 1.0 或 failed → 低潜力优化类型，延后推荐或跳过
- 从未尝试过的维度 → 根据维度通用优化潜力评估
- **分析 failure_reason**：了解之前失败的具体原因，避免重复失败的尝试
- **参考 executed_action**：了解之前成功优化的具体做法，作为后续优化的参考

## 分析维度

### 维度 1：入参静态化检查

**检查内容**：代码中是否存在可声明为 `tl.constexpr` 但未声明的固定参数

**典型问题特征**：
```python
@triton.jit
def kernel(A, B, C, M, N,
            stride_am, stride_an,  # 运行时不变化的固定值，但未声明为 tl.constexpr
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr):
```

**判断逻辑**：
- 如果代码中存在运行时不变化的固定参数（如 stride、固定数值、BLOCK_SIZE等）未声明为 `tl.constexpr` → 标记为可优化点
- 如果所有固定参数都已正确声明为 `tl.constexpr` → 此维度无问题

---

### 维度 2：Tiling 策略检查

**检查内容**：检查是否存在 Tiling 不当导致的非连续内存访问

**典型问题特征**：
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
- 如果 `tl.load` 的偏移量计算中，`tl.arange` 产生的向量偏移量作用于 `stride > 1` 的轴，而存在 `stride = 1` 的轴仅被当作标量索引处理 → 标记为可优化点
- 如果 `tl.arange` 已经作用于内存最连续的轴（通常是最后一张量的最后一维），且实现了合并访存 → 此维度无问题

---

### 维度 3：BLOCK_SIZE 配置检查

**检查内容**：检查 BLOCK_SIZE 参数是否经过充分调优

**典型问题特征**：
```python
@triton.jit
def kernel(A, C, M, N,
            BLOCK_M: tl.constexpr = 128,  # BLOCK_SIZE 可能需要调优
            BLOCK_N: tl.constexpr = 128):
```

**判断逻辑**：
- 如果代码中存在 BLOCK_SIZE 参数（BLOCK_M、BLOCK_N、BLOCK_K 等）且未进行系统性调优 → 标记为可优化点
- 如果 BLOCK_SIZE 已经过充分调优（如通过 benchmark 确定了最优值）→ 此维度无问题

---

### 维度 4：向量化检查（Scalar 转 Vector）

**检查内容**：检查是否存在 scalar 操作可以转换为 vector 操作

**典型问题特征**：
```python
# 问题1：标量广播
scalar_val = 0.5
result = x * scalar_val  # scalar 广播，无法启用 vector 加速

# 问题2：标量规约
sum_val = 0.0
for n in range(N):
    val = tl.load(x_ptr + row_offset + n)
    sum_val += val  # 标量加法，循环依赖

# 问题3：int类型比较
is_invalid_tok = tok < 0  # i64/i32类型，退化为标量

# 问题4：int类型除法/取余
c = a // b  # i32标量除法
d = a % b   # i32标量取余
```

**判断逻辑**：
- 如果存在标量广播操作未使用 `tl.full` 转为 vector → 标记为可优化点
- 如果存在标量规约循环未使用 vector 分块规约 → 标记为可优化点
- 如果存在 int 类型比较未转为 float32 → 标记为可优化点
- 如果存在 int 类型除法/取余未优化 → 标记为可优化点

---

### 维度 5：循环不变式外提检查

**检查内容**：检查循环内部是否存在可以外提的计算

**典型问题特征**：
```python
# 问题：循环内重复加载相同的值
for outer_idx in range(outer_size):
    for inner_idx in range(inner_size):
        param_idx = outer_idx  # 只依赖外层变量
        val = tl.load(param_ptr + param_idx)  # 重复加载相同值
        ...

# 问题：索引通过整除映射
for block in range(num_blocks):
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    param_idx = offsets // SPATIAL_SIZE  # 映射到更粗粒度的索引
    val = tl.load(param_ptr + base + param_idx)  # 同一 param_idx 的多个元素重复加载
```

**判断逻辑**：
- 如果循环内存在索引不依赖内层变量的 `tl.load` → 标记为可优化点
- 如果存在内外层循环次数比例大且有重复加载 → 标记为可优化点

---

### 维度 6：维度合并检查

**检查内容**：检查是否存在可以合并的多层嵌套循环

**典型问题特征**：
```python
# 问题：3层循环
for n in range(N):
    for h in range(H):
        for w_start in range(0, W, BLOCK_SIZE):
            base_offset = n * stride_n + c * stride_c + h * stride_h
            data = tl.load(input_ptr + base_offset + ...)
```

**判断逻辑**：
- 如果存在多层嵌套循环处理连续维度（如 H×W）未合并 → 标记为可优化点
- 合并后可以减少外层循环次数、减少重复计算、提高内存连续性

---

### 维度 7：Pass 合并检查

**检查内容**：检查是否存在可以合并的多次遍历

**典型问题特征**：
```python
# 问题：多次遍历
for ...:  # Pass 1
    data = tl.load(...)
    mean += tl.sum(data)

for ...:  # Pass 2 - 再次遍历
    data = tl.load(...)
    var += tl.sum((data - mean) ** 2)

for ...:  # Pass 3 - 第三次遍历
    data = tl.load(...)
    tl.store(...)
```

**判断逻辑**：
- 如果存在多个统计量计算（mean + variance）需要多次遍历数据 → 标记为可优化点
- 可以利用数学公式同时计算多个统计量，减少遍历次数

---

### 维度 8：Load 指令重排序检查

**检查内容**：检查是否存在可以重排序以提高并行度的 load 指令

**典型问题特征**：
```python
# 问题：load B 在前，阻塞了 load A
for i in range(HEAD_NUM):
    p_B_index = B_index + i
    idx_B = tl.load(p_B_index)  # 在前，会阻塞
    p_B = B + idx_B
    b_B = tl.load(p_B)
    b_A = tl.load(p_A)  # 必须等 load B 完成
```

**判断逻辑**：
- 如果存在因依赖关系导致串行执行的 load 指令 → 标记为可优化点
- 调整 load 顺序可让无依赖的 load 与其他指令并行执行

---

### 维度 9：分核策略检查

**检查内容**：检查 Grid 大小是否与物理核数匹配

**典型问题特征**：
```python
# 问题1：Grid 远超物理核数
grid = (batch_size,)  # batch_size=128，远超 48 核

# 问题2：Grid 远小于物理核数
grid = (batch_size // 64,)  # 只有 2 核

# 问题3：Tile 过小
BLOCK_SIZE = 64  # UB 利用率低
```

**判断逻辑**：
- 如果 Grid 大小远大于或远小于物理核数（40-48 核）→ 标记为可优化点
- 如果 BLOCK_SIZE 过小（小于 1024）导致 UB 利用率低 → 标记为可优化点
- 如果 BLOCK_SIZE 过大导致 UB 溢出风险 → 标记为可优化点

---

### 维度 10：离散访存检查

**检查内容**：检查是否存在非连续或不可预测的索引访存

**典型问题特征**：
```python
# 问题：随机索引导致离散访存
offset = tl.load(offset_ptr)  # 随机标量
idx = tl.load(idx_ptr + rn * stride_idx)  # 随机向量
val = tl.load(x_ptr + offset + idx * stride_x)  # 直接从GM离散访问
```

**判断逻辑**：
- 如果存在完全无法预测的随机索引导致的离散访存 → 标记为可优化点
- 可考虑先整块读取到 UB，再使用 `tl.gather` 收集

---

### 维度 11：Libdevice 函数检查

**检查内容**：检查是否重复实现了已有的 libdevice 函数

**典型问题特征**：
```python
# 问题：重复造轮子
@triton.jit
def round_int8(x):
    return (x + 0.5).to(tl.int8)  # 逻辑不完整

# 问题：手写激活函数
out = tl.maximum(x, 0.0)  # 手写 relu
```

**判断逻辑**：
- 如果存在 round、trunc、pow 等数学函数的手写实现 → 标记为可优化点
- 如果存在激活函数（如 relu）的手写实现 → 标记为可优化点
- 应优先使用 `tl.extra.cann.libdevice` 中已有的优化函数

---

### 维度 12：自动调优（Autotune）检查

**检查内容**：检查是否使用了 `@triton.autotune` 装饰器

**典型问题特征**：
```python
# 问题：硬编码的 BLOCK_SIZE
@triton.jit
def kernel(A, C, M, N, BLOCK_SIZE: tl.constexpr = 128):
    ...

# 优化：使用 autotune 自动搜索最优配置
@triton.autotune(configs=[...], key=['M', 'N'])
@triton.jit
def kernel(A, C, M, N, BLOCK_SIZE: tl.constexpr):
    ...
```

**判断逻辑**：
- 如果存在可调整的参数（如 BLOCK_SIZE）但未使用 autotune → 标记为可优化点
- 如果已经使用 autotune 或参数已通过其他方式充分调优 → 此维度无问题

---

## 输出格式

**输出文件**：`todo-optim.json`

**文件格式要求**：JSON 结构化格式

```json
{
  "optimization_points": [
    {
      "id": 1,
      "dimension": "入参静态化",
      "description": "stride_am 等参数未声明为 tl.constexpr",
      "suggestion": "将 stride_am, stride_an 等固定参数声明为 tl.constexpr"
    },
    {
      "id": 2,
      "dimension": "Tiling策略",
      "description": "tl.arange 作用于非连续轴，导致向量化访存效率低",
      "suggestion": "调整 tiling 策略，使向量化访存作用于连续轴"
    }
  ]
}
```

**字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| id | 是 | 优化点的唯一标识，从1开始递增 |
| dimension | 是 | 分析维度名称 |
| description | 是 | 清晰描述当前代码存在的问题 |
| suggestion | 是 | 具体可执行的优化方案 |

**注意**：todo-optim.json 只保留未完成的优化点。已完成或失败的优化点会被移除。

## 重要约束

- ⚠️ **必须对所有 12 个维度逐一进行分析，不得遗漏**
- ⚠️ **每个发现的优化点都必须写入 todo-optim.json**
- ⚠️ **优化建议必须具体、可执行**
- ⚠️ **只能使用本 skill 规定的优化方式进行识别，不要使用任何超出本 skill 之外的优化方式**
- ⚠️ **优化点必须按优化潜力从大到小排序，优化潜力高的排在前面**
- ⚠️ **必须结合历史经验（optim_history.json）评估每个优化点的潜力**
- ⚠️ **todo-optim.json 只保留未完成的优化点，已完成或失败的优化点必须移除**
- 如果某个维度没有发现问题，仍需在报告中注明"该维度无明显问题"
