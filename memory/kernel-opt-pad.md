---
name: kernel-opt-pad
description: Pad 算子（transformation-memory 类）的 Triton Ascend 四层隔离优化经验
metadata:
  type: reference
---

# Pad 算子优化经验

**算子类别**: `transformation-memory`
**典型特征**: 数据搬运为主，计算极简（仅边界坐标映射），输出 shape != 输入 shape
**性能基准**: 51 cases 全过，几何平均加速比 **1.68x**（大矩阵可达 18x+）

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 必须做多 kernel 分支 <!-- exp-id:l1-001 -->
- **禁止**用单一通用 kernel 处理所有 (ndim, mode) 组合
- 通用 4D kernel 的逐元素坐标解码 overhead 极大，仅作为兜底方案
- **必须**为高频场景（2D/3D constant）写特化 kernel

### L1.2 constant 模式必须拆分为 fill + copy <!-- exp-id:l1-002 -->
- **禁止**在 kernel 内逐元素判断 `if in_bounds else fill_value`
- **必须**先 `output.fill_(value)`，再用 copy kernel 搬运有效数据

### L1.3 Host 侧必须做维度压缩 <!-- exp-id:l1-003 -->
- **必须**在调用 kernel 前 squeeze 前导 size-1 维度
- 压缩后需同步调整 pad_list 的维度对应关系

### L1.4 坐标比较必须用 float32 <!-- exp-id:l1-004 -->
- **禁止**直接对整数坐标使用 `tl.where(coord < 0, ...)`
- **必须**先 `.to(tl.float32)` 再比较

---

### L1.5 禁止硬编码 num_cores <!-- exp-id:l1-005 -->
- **必须** 必须动态读取实际 Vector Core 数量，禁止硬编码 num_cores。正确做法：torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)
- **Why:** 硬编码 num_cores=8 仅利用 20% Vector Core，导致加速比从 ~1.3x 跌至 0.67x（慢于 PyTorch）
- **How to apply:** 所有使用多核并行的 Triton kernel 启动代码
## Layer 2: 算法骨架（Agent 可参考架构）

### L2.1 Host 侧分支决策树（伪代码） <!-- exp-id:l2-001 -->

```
ndim = squeeze(x) 后的维度
mode = constant/reflect/replicate/circular

if ndim == 2:
    if mode == constant:
        output.fill_(value)
        launch copy_kernel_2d
    else:
        launch pad_kernel_2d  # 逐行边界映射
elif ndim == 3:
    if mode == constant:
        output.fill_(value)
        launch copy_kernel_3d
    else:
        if D0 * D1_out > THRESHOLD:  # THRESHOLD ~ 3000
            launch pad_kernel_3d_nonconstant_v2  # 1D grid
        else:
            launch pad_kernel_3d_nonconstant_2d   # 2D grid
else:
    pad_to_4d()
    launch pad_kernel_4d  # 通用兜底
```

### L2.2 多核并行骨架模式 <!-- exp-id:l2-002 -->

**模式 A - 按元素分配（适合通用/1D 场景）**:
```
elements_per_core = cdiv(total_elements, num_cores)
core_start = pid * elements_per_core
core_end = min(core_start + elements_per_core, total_elements)
for block_idx in range(cdiv(core_end - core_start, BLOCK_SIZE)):
    # 处理一个 block
```

**模式 B - 按行分配（适合 2D/3D 场景）**:
```
rows_per_core = cdiv(total_rows, num_cores)
row_start = pid * rows_per_core
row_end = min(row_start + rows_per_core, total_rows)
for row_idx in range(row_end - row_start):
    # 处理一行，内部按 block 遍历列
```

### L2.3 Block Size 选择策略 <!-- exp-id:l2-003 -->

根据最后一维宽度选择（向上取 2 的幂次）：
```
width <= 64   -> 64
width <= 128  -> 128
width <= 256  -> 256
width <= 512  -> 512
width <= 1024 -> 1024
width <= 2048 -> 2048
else          -> 4096
```

---

## Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

### L3.1 维度压缩与 pad_list 同步调整 <!-- exp-id:l3-001 -->

```python
# 技巧：squeeze 后需裁剪 pad_list
num_pad_pairs = len(pad_list) // 2
implicit_pad_dims = ndim_orig - num_pad_pairs
if squeeze_count > implicit_pad_dims:
    entries_to_remove = 2 * (squeeze_count - implicit_pad_dims)
    pad_list_kernel = pad_list[:-entries_to_remove]
```

**可替代方向**：也可以在 kernel 内处理维度映射，但 host 侧预处理通常更清晰。

### L3.2 2D Copy Kernel 核心结构（constant 模式） <!-- exp-id:l3-002 -->

```python
pid = tl.program_id(0)
rows_per_core = tl.cdiv(H, num_cores)
row_start = pid * rows_per_core
row_end = tl.minimum(row_start + rows_per_core, H)

for row_idx in range(row_end - row_start):
    in_row = row_start + row_idx
    out_row = in_row + pad_t
    base_in = in_row * W
    base_out = out_row * W_out + pad_l

    for block_idx in range(tl.cdiv(W, BLOCK_SIZE)):
        cols = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < W
        data = tl.load(in_ptr + base_in + cols, mask=mask)
        tl.store(out_ptr + base_out + cols, data, mask=mask)
```

**可替代方向**：可以用 1D 元素级分配代替按行分配，但按行更利于利用行内连续性。

### L3.3 边界映射公式（四种模式） <!-- exp-id:l3-003 -->

```python
# reflect: 镜像反射
coord = tl.where(coord_f < 0.0, -coord, coord)
coord = tl.where(coord_f >= N, 2*(N-1) - coord, coord)

# replicate: 钳制到边界
coord = tl.maximum(0, tl.minimum(coord, N - 1))
# 或等价的 tl.where 版本（推荐，精度更稳）

# circular: 循环取模
coord = tl.where(coord_f < 0.0, coord + N, coord)
coord = tl.where(coord_f >= N, coord - N, coord)
```

**可替代方向**：循环取模可用 `coord % N`，但 `%` 在 Triton Ascend 后端可能 slower，tl.where 通常更优。

### L3.4 3D non-constant Grid 切换阈值 <!-- exp-id:l3-004 -->

```python
# 经验阈值：约 3000 行时切换
if D0 * D1_out > 3000:
    # 1D grid：每个 core 处理多行，内部 loop
    launch v2_kernel[(num_cores,)]
else:
    # 2D grid：每个 program 处理一个 (d0, d1) 平面位置
    launch kernel_2d[(D0_out, D1_out)]
```

**可替代方向**：阈值可调整，也可基于输出元素总数而非行数做决策。

---

## Layer 4: 完整归档（Agent 默认不读取，仅人工复盘）

> ⚠️ **Agent 注意**：以下仅为历史实现的路径记录。你**禁止**直接复制其代码结构、变量命名或 kernel 组织方式。若需参考，仅可借鉴其设计思想，必须根据当前任务重新设计。

### 历史实现归档

| 版本 | 代码 | 报告 | 摘要 | 性能 | 特点 |
|------|------|------|------|------|------|
| v1 (baseline) | `pad_v1_20260522.py` | `pad_v1_20260522_report.md` | `pad_v1_20260522_summary.json` | 1.68x | 多 kernel 分支、维度压缩、constant 特化 |

### 完整归档路径（Layer 4）
```
/home/zmm/OpAgent-Pad/.claude/memory/archive/pad/
├── pad_v1_20260522.py           # 完整实现代码
├── pad_v1_20260522_report.md    # 生成报告（含逐 shape 性能明细）
└── pad_v1_20260522_summary.json # 性能摘要（JSON，含 per_shape_results）
```

### 原始工作目录
```
/home/zmm/OpAgent-Pad/triton_ascend_output/op_0_15_Pad_20260522_0423_8652/
```

### 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 1D 小向量 | 1.5x - 2.2x | kernel launch overhead 主导 |
| 2D 大矩阵 | 2.0x - 18.6x | copy kernel 高效利用带宽 |
| 3D/4D constant | 0.5x - 4.3x | fill_ overhead，小 tensor 时劣化 |
| 3D/4D non-constant | 0.1x - 7.4x | 边界判断 overhead |

**关键结论**：该实现对大 tensor（> 2048x2048）表现极佳；小 tensor（< 128x128）因 kernel launch overhead 可能略慢于 torch。这是 Triton kernel 的普遍特征。

---

## 常见陷阱与避免方法

### 陷阱 1: 整数比较的隐式行为
- **问题**: `tl.where(coord < 0, ..., ...)` 中 `coord` 为整数类型时，在 Ascend 后端可能不正确
- **解决**: 统一先 `.to(tl.float32)` 再比较

### 陷阱 2: 前导 1 维未压缩导致 4D kernel 性能劣化
- **问题**: `[1, 3, 224, 224]` 走 4D 通用 kernel，加速比可能 < 1x
- **解决**: Host 侧 squeeze + 同步调整 pad_list

### 陷阱 3: constant 模式混用边界判断 kernel
- **问题**: 同一个 kernel 处理 constant 和 non-constant，constant 时每个元素都判断 `valid & mask`
- **解决**: constant 严格拆分为 `fill_` + `copy_kernel`

### 陷阱 4: replicate 模式用 tl.maximum/tl.minimum 的精度问题
- **问题**: `tl.maximum(0, tl.minimum(coord, N-1))` 对负整数可能异常
- **解决**: 优先用 `tl.where` + float32 比较
