# MatMul 计算分解示例

## 算子信息
- op_name: matmul
- category: matmul
- 计算模式: **分块累加 (M×N×K 三重循环)**
- 输入 shape: input [1024, 512], other [512, 256], dtype: float32
- 输出 shape: [1024, 256], dtype: float32
- 特殊属性: transpose_a=false, transpose_b=false

## 来源文件
- reference: `matmul_reference.py` → `torch.matmul(a, b)`
- DSL: `matmul_dsl.py` → 三重循环 M×N(tiled)×K(tiled), 逐行处理
- op_desc: `matmul_op_desc.json` → `attributes.transpose_a=false, transpose_b=false`

## 计算链分解

### Step 0: 输入
- input (A): shape [M, K] = [1024, 512], dtype: float32, 数值范围 [0, 1)
- other (B): shape [K, N] = [512, 256], dtype: float32, 数值范围 [0, 1)
- 计算: C[m, n] = Σ_k A[m, k] × B[k, n]

### Step 1: 累加器初始化
- 操作: `C_tile[n_start:n_end] = 0` (对每个 m 行, 每个 N-tile)
- DSL 对应: `tl.duplicate(c_ub, 0.0, count=actual_n)`
- 输出 shape: [tile_n] = [128]
- **精度风险点**:
  - 累加器必须初始化为 **精确的 0.0**, 否则结果有常数偏移
  - AscendC 中 AllocTensor 后 buffer 内容不确定, **必须显式 Duplicate 初始化**
  - 若忘记初始化, pattern = uniform_offset
- **知识库关联**: #12 MatMul CHECKLIST 项 (1)

### Step 2: 分块矩阵乘 (K 维内积累加)
- 操作: 对每个 K-tile, 加载 A[m, k_start:k_end] 和 B[k_start:k_end, n_start:n_end], 累加到 C_tile
- DSL 对应: k_tile 循环中的 `tl.load(A)`, `tl.load(B)`, 然后逐元素乘加
- 输入: A 的一行片段 [tile_k=128] + B 的一块 [tile_k=128, tile_n=128]
- 输出: 累加到 c_ub [tile_n=128]
- **精度风险点**:
  - **K 维分块边界**: K=512, tile_k=128, k_tiles=4 — 整除, 无边界问题
  - 若 K 不整除 tile_k, 最后一个 K-tile 有 padding — padding 值必须为 0
  - **累加精度**: 512 次 float32 乘加, 累积误差可能显著 (特别是大值场景)
  - **B 矩阵访存**: B 是按行存储 (row-major), 但 K 维迭代需要按列访问 B — 需要正确的 stride 计算
  - B 偏移: `k_start * n_size + n_start`, 每行 stride = n_size = 256
- **知识库关联**: #12 MatMul CHECKLIST 项 (2)(3)

### Step 3: N 维分块 → 输出写回
- 操作: 对每个 N-tile, 将累加完成的 C_tile 写回 GM
- DSL 对应: `tl.store(output_ptr + c_offsets, c_ub)`
- 输出偏移: `m_idx * n_size + n_start + arange(0, actual_n)`
- **精度风险点**:
  - N=256, tile_n=128, n_tiles=2 — 整除, 无边界问题
  - 若 N 不整除 tile_n, 最后一个 N-tile 写回时必须只写 actual_n 个元素
  - 超出范围的 GM 写入会覆盖其他行数据 → 数据污染

### Step 4: M 维迭代 (外层循环)
- 操作: 对每个 m 行重复 Step 1-3
- DSL 对应: `for m_idx in range(m_start, m_end)`
- 核心分配: `n_cores = 32`, `m_per_core = ceil(1024/32) = 32`
- **精度风险点**:
  - M 维是按核分配的, 每核处理 32 行 — 各核独立, 无跨核通信
  - 若 M 不整除 n_cores, 最后一个核可能处理超出范围的行

## 误差传播链

```
Step 1 累加器未初始化
  → 结果有常数偏移 → pattern: uniform_offset

Step 2 K-tile 边界 padding 非 0
  → 每行结果偏大 → pattern: uniform_offset

Step 2 B 矩阵 stride 计算错误
  → 完全错误的数据参与计算 → pattern: all_wrong

Step 3 N-tile 边界写入越界
  → 相邻行数据被覆盖 → pattern: dimension_concentration
```

## DSL tiling 策略要点

从 `matmul_dsl.py` 提取:
- `n_cores = 32`, M 维按核分配: `m_per_core = ceil(M / n_cores)`
- `tile_k = 128`, `tile_n = 128` (K 和 N 维分块)
- `k_tiles = ceil(512/128) = 4`, `n_tiles = ceil(256/128) = 2`
- 三重循环: m_idx → n_tile_id → k_tile_id
- 4 个 UB buffer: a_ub [tile_k×tile_n], b_ub [tile_k×tile_n], c_ub [tile_n], temp_ub

**注意**: DSL 中 a_ub 分配大小为 `tile_k * tile_n` 但实际只需 `tile_k` (一行的 K 片段), 可能是预留空间。

## 与 AscendC Kernel 的对照要点

1. **累加器初始化**: 每个 (m, n_tile) 组合是否 Duplicate(0.0)?
2. **B 矩阵 GM 偏移**: `k_start * n_size + n_start` 是否正确?
3. **K-tile 边界**: K=512 整除 tile_k=128, 但其他 shape 不保证 — Kernel 是否处理余数?
4. **N-tile 边界**: 同上, actual_n 是否正确?
5. **累加精度**: 是否使用 float32 累加? (float16 输入需要提升)
6. **M 维边界**: m_per_core × n_cores 可能 > M, Kernel 是否有越界保护?
