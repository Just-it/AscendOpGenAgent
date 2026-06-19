---
name: kernel-opt-repeat
description: Repeat 算子（transformation-memory 类）的 Triton Ascend 四层隔离优化经验
metadata:
  type: reference
---

# Repeat 算子优化经验

> ⚠️ **Agent 必读**：本文件 Layer 1 约束为**硬性规则**，非建议。设计 Repeat 类算子时，草图和代码必须逐条核对以下约束，任何冲突都必须在输出前修正。若草图架构与 Layer 1 冲突，不得进入代码生成阶段。

**算子类别**: `transformation-memory`
**典型特征**: 数据搬运为主无复杂计算，沿各维度复制张量，输出 shape = 输入 shape * repeats
**性能基准**: 49 cases 全过，几何平均加速比 **0.8785x**（部分大 shape 可达 9.4x，小 shape 受 kernel launch overhead 影响显著）

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 `r` 必须声明为 `tl.constexpr` <!-- exp-id:l1-001 -->
- **必须**将 repeat 次数 `r` 声明为 `tl.constexpr`
- **Why:** 触发编译器对 `for repeat_idx in range(r)` 的 loop unroll，消除动态循环开销；若 `r` 为运行时变量，循环无法展开，性能下降显著
- **How to apply:** 所有含固定次数循环的 Triton kernel，若循环次数来自 host 侧且在每个 kernel 实例中不变，均应声明为 constexpr

### L1.2 必须做多 kernel 分支（small vs large grid） <!-- exp-id:l1-002 -->
- **禁止**用单一通用 kernel 处理所有 grid 规模
- **Why:** 小 grid（total_blocks <= VEC_CORE_NUM）可直接用 2D grid 映射，每个 program 处理一个 block，无标量分区循环开销；大 grid 若仍用 2D grid 会超出硬件 program 限制，必须用 1D grid + 标量循环分配
- **How to apply:** 启动 kernel 前计算 `total_blocks = outer_size * num_inner_blocks`，与动态读取的 VEC_CORE_NUM 比较后分支

### L1.3 禁止硬编码 `num_cores` <!-- exp-id:l1-003 -->
- **必须**动态读取实际 Vector Core 数量，禁止硬编码固定值
- **Why:** 硬编码 num_cores 会导致实际利用的 core 数与硬件不匹配，小 grid 时调度不均，大 grid 时无法充分利用算力
- **How to apply:** `torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)`，所有使用多核并行的 Triton kernel 启动代码

### L1.4 处理顺序必须从最低维到最高维 <!-- exp-id:l1-004 -->
- **必须**按 r3（最低维）→ r2 → r1 → r0（最高维）的顺序逐维度处理
- **Why:** 低维处理后 inner_size 增大，后续高维处理时每个 block 可处理更多连续数据，提升内存带宽利用率和并行度分布；反向处理会导致早期阶段 inner_size 过小，block 利用率低
- **How to apply:** Host 侧 forward() 中按维度索引降序（shape 索引升序）依次判断并启动 kernel

### L1.5 输入必须确保 contiguous <!-- exp-id:l1-005 -->
- **必须**在 kernel 启动前检查并保证输入 tensor 为 contiguous
- **Why:** Triton 的 `tl.load`/`tl.store` 依赖连续内存访问模式获取最佳带宽；非 contiguous 输入会导致跨步访存，性能急剧下降甚至语义错误
- **How to apply:** Host 侧 `if not x.is_contiguous(): x = x.contiguous()`，或在 kernel 内使用 stride 参数（但通常 host 侧处理更简洁）

### L1.6 禁止直接使用模运算 `a % b` <!-- exp-id:l1-006 -->
- **必须**使用 `a - (a // b) * b` 替代 `a % b`
- **Why:** Triton Ascend 上整数类型的 `%` 会导致标量降级（scalar lowering），编译器将其展开为标量循环，严重损失 SIMD 并行度
- **How to apply:** 所有 kernel 内涉及周期性回绕的索引计算，一律改写为减法形式；若已计算 `coord = a // b`，则直接用 `a - coord * b`

### L1.7 禁止交织划分（interleaved partition）<!-- exp-id:l1-007 -->
- **必须**使用连续分块，每个 program 处理的数据在全局内存中连续
- **Why:** 交织划分（如 `range(pid, total, num_cores)`）会破坏内存访问的局部性和缓存行利用率，Ascend NPU 的预取机制对连续地址更有效
- **How to apply:** 计算 `blocks_per_core = cdiv(total_blocks, num_cores)`，`start = pid * blocks_per_core`，`end = min(start + blocks_per_core, total_blocks)`，然后 `for i in range(start, end)`

### L1.8 索引计算必须使用 int32 <!-- exp-id:l1-008 -->
- **必须**将索引张量显式转换为 `tl.int32`，禁止依赖默认的 int64
- **Why:** int64 类型的算术和比较操作在 Ascend Vector 单元上会被降级为标量循环；int32 可保持向量化执行
- **How to apply:** `offsets = (... + tl.arange(0, BLOCK)).to(tl.int32)`，所有 strides/shapes 也以 int32 传入

---

## Layer 2: 算法骨架（Agent 可参考架构）

### L2.1 Host 侧分支决策树（伪代码） <!-- exp-id:l2-001 -->

```python
for dim_idx in [3, 2, 1, 0]:  # 从最低维到最高维
    r = repeats[dim_idx] if dim_idx < len(repeats) else 1
    if r <= 1:
        continue

    shape = list(out.shape)
    outer_size = prod(shape[:dim_idx])  # 外层元素个数
    inner_size = out.numel() // outer_size  # 内层元素个数（含当前维）
    out_shape[dim_idx] *= r
    output = torch.empty(out_shape, dtype=out.dtype, device=out.device)

    BLOCK = get_block_size(inner_size)  # 按 2 的幂次向上取整
    num_inner_blocks = (inner_size + BLOCK - 1) // BLOCK
    total_blocks = outer_size * num_inner_blocks

    if total_blocks <= VEC_CORE_NUM:
        grid = (outer_size, num_inner_blocks)  # 2D 精确映射
        repeat_small_kernel[grid](...)
    else:
        grid = (min(total_blocks, VEC_CORE_NUM),)  # 1D 循环分配
        repeat_large_kernel[grid](..., num_cores=grid[0])

    out = output
```

### L2.2 多核并行骨架模式 <!-- exp-id:l2-002 -->

**模式 A - Small Grid（2D 精确映射）**:  
适合 `total_blocks <= VEC_CORE_NUM`
```python
outer_idx = tl.program_id(0)    # 对应 outer slice
local_block = tl.program_id(1)  # 对应 inner block

block_start = local_block * BLOCK
offs = block_start + tl.arange(0, BLOCK)
mask = offs < inner_size

in_offset = outer_idx * inner_size
val = tl.load(x_ptr + in_offset + offs, mask=mask)

for repeat_idx in range(r):     # r 为 constexpr，编译期展开
    out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
    tl.store(out_ptr + out_offset + offs, val, mask=mask)
```

**模式 B - Large Grid（1D 循环分配）**:  
适合 `total_blocks > VEC_CORE_NUM`
```python
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
    # ... 同模式 A 的 load/store
```

### L2.3 BLOCK 大小选择策略 <!-- exp-id:l2-003 -->

按 inner_size 向上取 2 的幂次，目标使 `num_inner_blocks` 尽量小（理想为 1）：
```
inner_size <= 64   -> 64
inner_size <= 128  -> 128
inner_size <= 256  -> 256
inner_size <= 512  -> 512
inner_size <= 1024 -> 1024
inner_size <= 2048 -> 2048
inner_size <= 4096 -> 4096
else               -> 8192
```

**可替代方向**：也可固定使用较大 BLOCK（如 1024）并用 mask 处理余量，但可能导致小 inner_size 时 mask 比例过高；按幂次分级可在各种尺寸下取得平衡。

---

## Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

### L3.1 逐维度处理时的 shape 同步更新 <!-- exp-id:l3-001 -->

```python
# 技巧：每处理完一个维度，立即用输出 shape 作为下一轮输入 shape
shape = list(out.shape)
outer_size = prod(shape[:dim_idx])  # 当前维度之前所有维度的乘积
inner_size = out.numel() // outer_size  # 包含当前维度及之后的所有元素
out_shape = shape[:]
out_shape[dim_idx] *= r
output = torch.empty(out_shape, dtype=out.dtype, device=out.device)
# ... launch kernel ...
out = output  # 关键：更新 out 供下一轮使用
```

**可替代方向**：也可以在 kernel 内处理多个维度，但会急剧增加编译期分支复杂度，且难以针对每个维度选择最优 BLOCK；逐维度串行处理是更稳健的方案。

### L3.2 多核分区循环的负载均衡公式 <!-- exp-id:l3-002 -->

```python
blocks_per_core = total_blocks // num_cores
remainder = total_blocks - blocks_per_core * num_cores  # 等价于 total_blocks % num_cores

if pid < remainder:
    my_blocks = blocks_per_core + 1
    start_block = pid * (blocks_per_core + 1)
else:
    my_blocks = blocks_per_core
    start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core
```

**可替代方向**：也可使用 `elements_per_core` 按元素分配，但 repeat 算子的天然计算单元是 "outer slice + inner block"，按 block 分配更符合数据局部性。

### L3.3 输入 contiguous 的防御性处理 <!-- exp-id:l3-003 -->

```python
if not x.is_contiguous():
    x = x.contiguous()
```

**可替代方向**：对于确定总是 contiguous 的场景（如上一 Triton kernel 的输出），可省略此检查以节省 host 侧开销；但从通用性和正确性角度，保留检查更安全。

---

## Layer 4: 完整归档（Agent 默认不读取，仅人工复盘）

> ⚠️ **Agent 注意**：以下仅为历史实现的路径记录。你**禁止**直接复制其代码结构、变量命名或 kernel 组织方式。若需参考，仅可借鉴其设计思想，必须根据当前任务重新设计。

### 历史实现归档

| 版本 | 代码 | 报告 | 摘要 | 性能 | 特点 |
|------|------|------|------|------|------|
| v2 (current best) | `repeat_v2_20260526.py` | `repeat_v2_20260526_report.md` | `repeat_v2_20260526_summary.json` | **0.8785x** | r-constexpr + 多版本 dispatch + 反向维度处理 |

**反面教材（flat-kernel 尝试）**: 曾尝试不逐维度串行启动 kernel，而是将整个多维 repeat 展平为 1D，在单个 element-wise kernel 中通过取模运算将输出线性索引映射回输入线性索引。该思路验证通过（49/49 cases），但性能仅 0.0306x，严重劣于逐维度方案。主要原因：(1) 每个元素的索引计算开销高（多维取模/除法）；(2) 输入访存高度离散，无法利用连续内存带宽；(3) 小 shape 上 kernel launch overhead 占比大。此思路**未归档**，仅作为教训记录。

### 完整归档路径（Layer 4）
```
/home/zmm/OpAgent-Pad/.claude/memory/archive/repeat/
├── repeat_v2_20260526.py           # 完整实现代码
├── repeat_v2_20260526_report.md    # 生成报告（含逐 shape 性能明细）
└── repeat_v2_20260526_summary.json # 性能摘要（JSON，含 per_shape_results）
```

### 原始工作目录
```
/home/zmm/OpAgent-Pad/triton_ascend_output/op_0_16_Repeat_20260526_0607_4528/
```

### 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 1D 小向量 | 0.3x - 3.3x | kernel launch overhead 主导，波动大 |
| 2D 大矩阵 | 0.5x - 9.8x | 大 shape 带宽利用率高，[5120,13824] 可达 9.79x |
| 3D/4D 特征图 | 0.02x - 2.5x | 小 batch / 大 spatial 时 overhead 显著，部分 shape 劣化严重 |
| 非对齐 shape | 0.1x - 2.2x | 奇数维度导致 num_inner_blocks > 1，效率下降 |

**关键结论**：Repeat 算子在 Ascend Triton 上的整体加速比未超过 PyTorch（0.8676x），主要原因包括：(1) 逐维度串行启动多个 kernel 带来多次 launch overhead；(2) 小 tensor 上 Triton kernel 的固定开销远大于 `torch.repeat` 的底层优化；(3) 仅在超大连续内存块（如 [5120, 13824]）上显著优于 PyTorch。未来同类算子可尝试：合并多维度处理到单个 kernel、或使用 AscendC 替代 Triton 以降低 launch 开销。

---

## 常见陷阱与避免方法

### 陷阱 1: `r` 未声明为 constexpr 导致循环无法展开
- **问题**: `for repeat_idx in range(r)` 中 `r` 为运行时变量，编译器不做 loop unroll，每次迭代有额外分支开销
- **解决**: 严格声明为 `r: tl.constexpr`

### 陷阱 2: 维度处理顺序从高维到低维
- **问题**: 先处理 r0（最高维）会导致前几个 kernel 的 inner_size 极大（等于整个 tensor 元素数），outer_size = 1，grid 过小无法充分利用多核；后续低维处理时数据已被打散，cache 局部性差
- **解决**: 严格按 r3→r2→r1→r0（最低维到最高维）顺序处理

### 陷阱 3: 硬编码 `num_cores` 导致调度不均
- **问题**: 如写死 `num_cores = 8`，在 40 core 的 ascend910b1 上仅利用 20% 算力
- **解决**: 运行时动态读取 `torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)`

### 陷阱 4: 非 contiguous 输入导致性能劣化或错误
- **问题**: stride tensor 传入 Triton kernel 后，`tl.load` 按连续地址访问，实际读取到错误数据
- **解决**: Host 侧强制 `.contiguous()` 后再启动 kernel
