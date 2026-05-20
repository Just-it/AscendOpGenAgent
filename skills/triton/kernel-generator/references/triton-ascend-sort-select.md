# Sort/Select 算子优化

> 适用于需要迭代选择元素的算子：NMS、TopK、ArgSort 等

## 核心约束

Triton Ascend 不支持 `break`/`continue`/`return` 和 Python `if` 分支，必须用 `tl.where` + mask 实现条件逻辑。

| 禁止语法 | 替代方案 | 说明 |
|---------|---------|------|
| `if cond:` | `tl.where(cond, a, b)` | 所有条件必须用 SIMD 友好的方式表达 |
| `break`/`continue` | 用循环变量 + mask 控制 | 循环次数固定，用 mask 跳过无效迭代 |
| `return` | 无法提前返回 | 所有路径必须执行到函数末尾 |
| 标量条件赋值 `x = y if cond` | `x = tl.where(cond, y, x)` | 标量变量更新必须用 `tl.where` |

### 1.2 迭代选择的标准模式

对于需要"每次从剩余元素中选一个最优"的算法（如NMS），标准模式是：

```python
# 模式：selection-sort 风格的迭代选择
for step in range(max_select):
    # 1. 线性扫描找最优候选
    best_idx = -1
    best_score = threshold
    for i in range(n_elements):
        score = tl.load(scores_ptr + i)
        higher = (score > best_score) & active
        best_idx = tl.where(higher, i, best_idx)
        best_score = tl.where(higher, score, best_score)

    # 2. 检查是否找到有效候选
    found = (best_idx != -1) & active

    # 3. 记录结果（仅当 found 时）
    tl.store(output_ptr + count, best_idx.to(tl.int32), mask=found)
    count = tl.where(found, count + 1, count)

    # 4. 标记已选（通过修改内存状态）
    tl.store(scores_ptr + best_idx, sentinel_value, mask=found)

    # 5. 根据选中元素更新其他元素状态（算子特定逻辑）
    # ... 例如 NMS 中计算 IoU 并抑制重叠 box
```

**关键原则**：
- 用**内存值**（如将 score 设为哨兵值）表示"已选/已抑制"状态，而非标量 flag
- 用 `tl.where` 做所有条件选择，不用 Python `if`
- 用 `mask=` 参数控制 `tl.load`/`tl.store` 的执行

---

## 2. 算子特定实现

### 2.1 NMS (Non-Maximum Suppression)

#### 算法语义

验证框架对比的是 PyTorch 参考实现（如 `30_NMS.py`），其语义通常包含：

1. **先过滤**：只保留满足门槛条件的元素（如 `score > scores_threshold`）
2. **再降序排序**：参考实现通常用 `torch.argsort(..., descending=True, stable=True)` 确定顺序
3. **迭代选择**：按排序后的顺序遍历，若当前元素未被抑制则选中
4. **依赖抑制**：选中后，根据算子特定规则抑制其他元素（如 NMS 的 IoU 阈值）
5. **数量限制**：最多输出 `max_output_size` 个，达到即停止
6. **输出格式**：输出张量前 `num_selected` 个有效，其余为 0 或哨兵值

**关键：降序关系来自参考实现的排序步骤**。Triton kernel 中没有显式排序，而是通过迭代选择最高分来隐式复现降序语义。

#### 参考实现

```python
@triton.jit
def select_kernel(
    values_ptr,           # 用于比较的值
    selected_indices_ptr, # 输出：选中的原始索引
    num_selected_ptr,     # 输出：实际选中数量
    n_elements,
    max_output_size: tl.constexpr,
    threshold: tl.constexpr,
):
    pid = tl.program_id(0)
    active = (pid == 0)
    selected_count = 0

    for step in range(max_output_size):
        # 1. 线性扫描找最优候选
        best_idx = -1
        best_val = threshold
        for i in range(n_elements):
            val = tl.load(values_ptr + i)
            better = (val > best_val) & active
            best_idx = tl.where(better, i, best_idx)
            best_val = tl.where(better, val, best_val)

        # 2. 检查是否找到有效候选
        found = (best_idx != -1) & active

        # 3. 记录结果
        tl.store(selected_indices_ptr + selected_count,
                 best_idx.to(tl.int32), mask=found)
        selected_count = tl.where(found, selected_count + 1, selected_count)

        # 4. 标记已选，防止重复
        tl.store(values_ptr + best_idx, sentinel_value, mask=found)

        # 5. 算子特定逻辑：根据选中元素更新其他元素状态
        #    - NMS：读取选中元素的数据，计算与其他元素的关系（如 IoU），
        #            将满足条件的其他元素标记为已选/已抑制
        #    - TopK：无需此步骤
        #    - 其他算子：根据业务规则更新其他元素的值或标记

    tl.store(num_selected_ptr, selected_count, mask=active)
```

**关键点**:
- `grid=(1,)` 单核执行，顺序依赖算法天然难以并行
- `best_idx = -1` 初始值，配合 `found = (best_idx != -1)` 判断是否找到有效元素
- `mask=found` 保护所有依赖 `best_idx` 的 load/store，避免 -1 越界
- 写入顺序自然为降序，与参考实现 `argsort(descending=True)` 语义一致

## 算子特定扩展

### NMS

在通用模式阶段5加入：读取选中 box 坐标，计算与其他 box 的 IoU，将 IoU >= threshold 的 box 的 score 设为哨兵值（抑制）。

**关键点**:
- `scores_f32 = scores.float().contiguous()` 保证连续内存访问
- 输出前 `num_selected` 个为原始索引（按 score 降序），其余为 0

### TopK

无抑制逻辑，阶段5为空。将哨兵值设为 `-float('inf')`。

## 常见错误

```python
# 错误：Python if 分支
if score > best_score:
    best_idx = i

# 正确：tl.where
best_idx = tl.where(score > best_score, i, best_idx)
```

```python
# 错误：标量 flag 累积
keep = True
for j in range(n):
    if iou >= threshold:
        keep = False

# 正确：通过内存状态传递
tl.store(scores_ptr + j, -1.0, mask=suppress)
```

```python
# 错误：先收集所有保留元素再截断（破坏降序）
# 正确：每次迭代只选一个，天然满足降序和数量限制
```

```python
# 错误：用 binary search 找 top-k 阈值 + dynamic eps tie-breaking
# 原因：低精度 dtype 下 eps 无法覆盖所有 tied 边界，stable sort 语义无法复现
# 正确：参考实现含 stable=True 时必须显式排序（如 bitonic sort）
```

---

## 3. Bitonic Sort + 排序后处理范式

### 3.1 范式适用判定

| 模式 | 适用场景 | 不适用场景 |
|------|---------|-----------|
| **selection-sort**（第1节） | 只要 TopK 的索引/值集合，不关心排序后顺序 | 需要排序后顺序做 cumsum / mask / scatter |
| **bitonic-sort**（本节） | 需要**排序后的完整数组**用于后续 cumsum / scatter back | 纯 TopK 取前 K 个即可 |

**强制使用 bitonic-sort 的信号**：
- 参考实现含 `torch.sort(..., stable=True)` 且后续有 `cumsum`、`masked_fill`、`scatter_`
- 算子名含 `topk` + `topp` 组合
- 需要按排序后顺序做前缀和再映射回原始索引

### 3.2 Bitonic Sort 核心模板

```python
@triton.jit
def bitonic_sort_kernel(
    vals_ptr, idxs_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tile-based bitonic sort for Triton Ascend.
    同时维护值和原始索引，支持 stable sort tie-breaking.
    """
    NEG_INF_VAL = -10000.0
    pid = tl.program_id(0)

    # 加载数据到 temp buffer（padding 用 NEG_INF_VAL）
    for v_start in range(0, n_elements, BLOCK_SIZE):
        v_offsets = v_start + tl.arange(0, BLOCK_SIZE)
        vmask = v_offsets < n_elements
        vals = tl.load(vals_ptr + v_offsets, mask=vmask, other=NEG_INF_VAL)
        idxs = v_offsets
        tl.store(vals_ptr + v_offsets, tl.where(vmask, vals, NEG_INF_VAL))
        tl.store(idxs_ptr + v_offsets, idxs)

    # Bitonic sort: 16 outer * 16 inner = log2(65536)^2
    stride = 2
    for outer_iter in range(0, 16):
        do_sort = stride <= n_elements
        dist = stride // 2

        for inner_iter in range(0, 16):
            do_inner = dist >= 1
            tile_dist = dist // BLOCK_SIZE

            if dist < BLOCK_SIZE:
                # Same-tile: partner = i ^ dist
                for v_start in range(0, n_elements, BLOCK_SIZE):
                    tile_offsets = v_start + tl.arange(0, BLOCK_SIZE)
                    vmask_tile = tile_offsets < n_elements

                    tile_vals = tl.load(vals_ptr + tile_offsets, mask=vmask_tile, other=NEG_INF_VAL)
                    tile_idxs = tl.load(idxs_ptr + tile_offsets, mask=vmask_tile, other=0)

                    i = tl.arange(0, BLOCK_SIZE)
                    partner = i ^ dist
                    partner_offsets = v_start + partner
                    partner_mask = partner_offsets < n_elements

                    partner_vals = tl.load(vals_ptr + partner_offsets, mask=partner_mask, other=NEG_INF_VAL)
                    partner_idxs = tl.load(idxs_ptr + partner_offsets, mask=partner_mask, other=0)

                    j = tile_offsets
                    direction_asc = (j & stride) == 0
                    is_smaller = j < partner_offsets

                    # Tie-breaking: equal 时原始索引小者优先
                    is_less = tile_vals < partner_vals
                    is_equal = tile_vals == partner_vals
                    use_tile = is_less | (is_equal & (tile_idxs < partner_idxs))

                    min_val = tl.where(use_tile, tile_vals, partner_vals)
                    max_val = tl.where(use_tile, partner_vals, tile_vals)
                    min_idx = tl.where(use_tile, tile_idxs, partner_idxs)
                    max_idx = tl.where(use_tile, partner_idxs, tile_idxs)

                    final_val = tl.where(
                        is_smaller,
                        tl.where(direction_asc, min_val, max_val),
                        tl.where(direction_asc, max_val, min_val)
                    )
                    final_idx = tl.where(
                        is_smaller,
                        tl.where(direction_asc, min_idx, max_idx),
                        tl.where(direction_asc, max_idx, min_idx)
                    )

                    do_swap = do_inner & do_sort
                    tl.store(vals_ptr + tile_offsets, tl.where(do_swap, final_val, tile_vals), mask=vmask_tile)
                    tl.store(idxs_ptr + tile_offsets, tl.where(do_swap, final_idx, tile_idxs), mask=vmask_tile)
            else:
                # Cross-tile: partner_t = t ^ tile_dist
                for v_start in range(0, n_elements, BLOCK_SIZE):
                    t = v_start // BLOCK_SIZE
                    partner_t = t ^ tile_dist

                    tile_start = v_start
                    partner_start = partner_t * BLOCK_SIZE

                    tile_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
                    partner_offsets = partner_start + tl.arange(0, BLOCK_SIZE)
                    vmask_tile = tile_offsets < n_elements
                    vmask_partner = partner_offsets < n_elements

                    tile_vals = tl.load(vals_ptr + tile_offsets, mask=vmask_tile, other=NEG_INF_VAL)
                    tile_idxs = tl.load(idxs_ptr + tile_offsets, mask=vmask_tile, other=0)
                    partner_vals = tl.load(vals_ptr + partner_offsets, mask=vmask_partner, other=NEG_INF_VAL)
                    partner_idxs = tl.load(idxs_ptr + partner_offsets, mask=vmask_partner, other=0)

                    j = tile_offsets
                    partner_j = partner_offsets
                    direction_asc = (j & stride) == 0
                    is_smaller = j < partner_j

                    is_less = tile_vals < partner_vals
                    is_equal = tile_vals == partner_vals
                    use_tile = is_less | (is_equal & (tile_idxs < partner_idxs))

                    min_val = tl.where(use_tile, tile_vals, partner_vals)
                    max_val = tl.where(use_tile, partner_vals, tile_vals)
                    min_idx = tl.where(use_tile, tile_idxs, partner_idxs)
                    max_idx = tl.where(use_tile, partner_idxs, tile_idxs)

                    final_val = tl.where(
                        is_smaller,
                        tl.where(direction_asc, min_val, max_val),
                        tl.where(direction_asc, max_val, min_val)
                    )
                    final_partner_val = tl.where(
                        ~is_smaller,
                        tl.where(direction_asc, min_val, max_val),
                        tl.where(direction_asc, max_val, min_val)
                    )
                    final_idx = tl.where(
                        is_smaller,
                        tl.where(direction_asc, min_idx, max_idx),
                        tl.where(direction_asc, max_idx, min_idx)
                    )
                    final_partner_idx = tl.where(
                        ~is_smaller,
                        tl.where(direction_asc, min_idx, max_idx),
                        tl.where(direction_asc, max_idx, min_idx)
                    )

                    do_swap = do_inner & do_sort
                    tl.store(vals_ptr + tile_offsets, tl.where(do_swap, final_val, tile_vals), mask=vmask_tile)
                    tl.store(idxs_ptr + tile_offsets, tl.where(do_swap, final_idx, tile_idxs), mask=vmask_tile)

                    store_partner_mask = vmask_partner & (partner_t != t)
                    tl.store(vals_ptr + partner_offsets, tl.where(do_swap, final_partner_val, partner_vals), mask=store_partner_mask)
                    tl.store(idxs_ptr + partner_offsets, tl.where(do_swap, final_partner_idx, partner_idxs), mask=store_partner_mask)

            dist = dist // 2
        stride = stride * 2
```

**关键点**：
- `BLOCK_SIZE=256`，`sort_size` 向上取整到 2 的幂
- 同时维护 `temp_vals` 和 `temp_idxs` 两个 buffer
- `use_tile = is_less | (is_equal & (tile_idxs < partner_idxs))` —— stable sort 的 tie-breaking 核心
- same-tile（`dist < BLOCK_SIZE`，`partner = i ^ dist`）vs cross-tile（`partner_t = t ^ tile_dist`）
- Triton Ascend 不能 `break`，循环次数必须固定（16×16 覆盖到 65536）

### 3.3 排序后处理三件套（TopK + TopP 范式）

排序完成后，按排序后顺序执行：

```python
# Step 1: 直接取 kth_value（零误差）
kth_idx = sort_size - k_int
kth_value = sorted_vals[kth_idx]  # 通过 tl.load + mask 实现

# Step 2: Two-pass softmax（跨 tile 全局 max + sum_exp）
#   Pass 1: 遍历所有 tile 找 global_max（排除被 top-k mask 的元素）
#   Pass 2: 遍历所有 tile 累加 exp(sorted_val - global_max)

# Step 3: Cumsum with offset（跨 tile 全局前缀和）
cumsum_offset = 0.0
for v_start in range(0, sort_size, BLOCK_SIZE):
    tile_vals = tl.load(sorted_vals_ptr + v_offsets)
    softmax_vals = exp(tile_vals - global_max) / sum_exp
    cumsum_vals = tl.cumsum(softmax_vals, axis=0) + cumsum_offset
    cumsum_offset += tl.sum(softmax_vals)

# Step 4: Top-p mask（零误差，无需 eps）
threshold_p = 1.0 - p_val
top_p_mask = cumsum_vals <= threshold_p
is_last = v_offsets == (sort_size - 1)
top_p_mask = top_p_mask & (~is_last)  # 强制保留最后一个元素

# Step 5: Scatter back 到原始索引
out_ptrs = out_ptr + sorted_idxs * stride
 tl.store(out_ptrs, sorted_vals, mask=sorted_idxs < V)
```

**与 selection-sort 的本质区别**：
| 步骤 | selection-sort | bitonic-sort |
|------|---------------|--------------|
| top-k 阈值 | 线性扫描找第 K 大 | 直接取 `sorted[sort_size - k]` |
| top-p 判定 | 无法做（没有排序后顺序） | `cumsum_vals <= 1 - p` |
| tie-breaking | 依赖迭代选择的顺序 | 由 sort 的 `use_tile` 精确控制 |
| scatter | 通常不需要 | 通过 `sorted_idxs` 写回 |

### 3.4 TopK + TopP 完整 ModelNew 结构

```python
class ModelNew(nn.Module):
    def forward(self, logits, p, k):
        B, V = logits.shape
        BLOCK_SIZE_V = 256
        sort_size = 1 << (max(V, BLOCK_SIZE_V) - 1).bit_length()

        output = torch.empty_like(logits)
        temp_vals = torch.empty((B, sort_size), dtype=torch.float32, device=logits.device)
        temp_idxs = torch.empty((B, sort_size), dtype=torch.int32, device=logits.device)

        grid = (VEC_CORE_NUM,)
        topk_topp_kernel[grid](
            logits, p, k, output,
            temp_vals, temp_idxs,
            B, V, sort_size,
            # strides...
            VEC_CORE_NUM=VEC_CORE_NUM,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )
        return output
```
