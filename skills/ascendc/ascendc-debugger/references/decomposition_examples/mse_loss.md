# MSELoss 计算分解示例

## 算子信息
- op_name: mse_loss
- category: loss
- 计算模式: **跨核两阶段归约**
- 归约方式: reduction="mean", 全局归约到标量
- 输入 shape: predictions [32, 768], targets [32, 768], dtype: float32
- 输出 shape: [1], dtype: float32 (标量 loss)

## 来源文件
- reference: `mse_loss_reference.py` → `F.mse_loss(predictions, targets, reduction='mean')`
- DSL: `mse_loss_dsl.py` → Phase 1 (各核局部归约) + Phase 2 (Core 0 全局汇总)
- op_desc: `mse_loss_op_desc.json` → `attributes.reduction="mean"`

## 计算链分解

### Step 0: 输入
- predictions: shape [32, 768], dtype: float32, 数值范围 [0, 1)
- targets: shape [32, 768], dtype: float32, 数值范围 [0, 1)
- total_elems = 32 × 768 = 24576

---

### Phase 1: 各核局部归约 (所有核并行执行)

### Step 1: Sub (差值)
- 操作: `diff = predictions - targets`
- DSL 对应: `tl.vsub(diff_ub, pred_ub, target_ub)`
- 输入: predictions 和 targets 的当前 tile
- 输出 shape: [tile_size] (每次处理 2048 个元素)
- 数值范围预期: (-1, 1) (两个 [0,1) 之差)
- **精度风险点**: 无 (逐元素减法, 精度安全)

### Step 2: Square (平方)
- 操作: `sq = diff * diff`
- DSL 对应: `tl.vmul(sq_ub, diff_ub, diff_ub)`
- 输入: Step 1 的 diff
- 输出 shape: [tile_size]
- 数值范围预期: [0, 1) (平方后非负, 且 < 1)
- **精度风险点**: 无 (逐元素乘法)

### Step 3: ReduceSum (tile 内归约)
- 操作: `tile_sum = sum(sq)`
- DSL 对应: `tl.reduce_sum(sq_ub, sq_ub, shared_ub)` + `extract_scalar(sq_ub, 0)`
- 输入: Step 2 的 sq, shape [tile_size=2048]
- 输出: 标量 tile_sum
- **精度风险点**:
  - tile_size=2048 满足 64 倍数对齐 ✓
  - 但最后一个 tile 可能不满 (若 elems_per_core 不能被 tile_size 整除)
  - Padding 区域值应为 0 (平方值的 padding)
- **知识库关联**: #7 Reduce Count 未对齐

### Step 4: 跨 tile 累加 → partial_sum
- 操作: `partial_sum += tile_sum` (标量循环累加, inner_loops 次)
- DSL 对应: `partial_sum = partial_sum + tile_sum` (Python 标量运算)
- 输入: 每个 tile 的 tile_sum
- 输出: 单核的 partial_sum (标量)
- **精度风险点**:
  - inner_loops = elems_per_core / tile_size = (24576/16) / 2048 ≈ 0.75 → **不足 1 次循环!**
  - 实际: elems_per_core = 24576/16 = 1536, tile_size=2048 → inner_loops = 0
  - 这说明 24576 个元素分给 16 核, 每核 1536 个, 但 tile_size=2048 > 1536
  - **DSL 在此场景下 inner_loops=0, 不会执行任何计算** — 这是一个潜在的 DSL bug
  - AscendC Kernel 需要正确处理 elems_per_core < tile_size 的情况

---

### Phase 2: Core 0 全局汇总 (仅 Core 0 执行)

### Step 5: 写入 → 读取 workspace (跨核通信)
- 操作: 各核将 partial_sum 写入 `workspace[pid]`, Core 0 读取所有 n_cores 个值
- DSL 对应:
  - 写入: `tl.store(workspace + tl.arange(pid, pid+1), workspace_out_ub)`
  - 读取: `tl.load(workspace + tl.arange(0, n_cores), workspace_in_ub)` (仅 Core 0)
- **精度风险点 (关键)**:
  - **同步问题**: 各核 DataCopy(UB→GM) 是异步的, Core 0 读取时其他核可能尚未完成写入
  - **workspace 初始化**: 若某核因 inner_loops=0 未写入, workspace 中残留脏数据
  - AscendC 中需要 pipe_barrier 或等效同步机制确保所有核写完后 Core 0 才读取
- **知识库关联**: #3 TBuf 异步 DataCopy 数据竞争 (类似机制)

### Step 6: ReduceSum → Div (最终 mean)
- 操作: `total_sq = sum(workspace[0:n_cores])`, `mse = total_sq / total_elems`
- DSL 对应: `tl.reduce_sum(shared_ub, workspace_in_ub, shared_ub)` → `mse = sum_sq / float(total_elems)`
- 输入: workspace 中 n_cores=16 个 partial_sum
- 输出: 标量 mse
- **精度风险点**:
  - ReduceSum count = n_cores = 16, 不满足 64 倍数 → 需对齐, padding 为 0
  - **分母必须是 total_elems (24576) 而非 elems_per_core 或 n_cores**
  - 如果 Kernel 使用了错误的分母, 输出会有固定比例偏差 → uniform_offset 模式
- **知识库关联**: #11 Loss CHECKLIST 项 (1): mean 分母是否正确

## 误差传播链

```
Phase 1 (各核独立, 误差独立):
  Step 3 Padding 问题 → partial_sum 偏大 → Step 6 结果偏大

Phase 2 (全局汇总, 误差集中):
  Step 5 同步问题 → workspace 数据不完整 → Step 6 结果随机偏差
  Step 6 分母错误 → 结果有固定比例偏差

特殊风险: inner_loops 计算不正确 → Phase 1 根本不执行 → partial_sum=0 → MSE=0
```

## DSL tiling 策略要点

从 `mse_loss_dsl.py` 提取:
- `n_cores = 16`, 按元素总数平分: `elems_per_core = total_elems // n_cores`
- `tile_size = 2048`, `inner_loops = elems_per_core // tile_size`
- **workspace**: GM 中 shape [n_cores] 的 buffer, 用于跨核通信
- Phase 2 仅在 `pid == 0` 时执行
- 8 个 UB buffer (非常多): pred_ub, target_ub, diff_ub, sq_ub, shared_ub, workspace_out_ub, workspace_in_ub, output_ub

## 与 AscendC Kernel 的对照要点

1. **inner_loops 计算**: `elems_per_core // tile_size` 是否正确? 余数怎么处理?
2. **workspace GM 分配**: Host 是否在 GM 中分配了 n_cores 大小的 workspace?
3. **跨核同步**: 各核写 workspace 后是否有 barrier? Core 0 读取前是否等待?
4. **Phase 2 ReduceSum**: count=n_cores=16, 是否对齐到 64? padding 是否为 0?
5. **最终除法分母**: 是否 = total_elems (24576)? 不是 elems_per_core (1536)?
6. **输出写入**: Core 0 将标量写入 output_ptr, GM 偏移是否正确?

## AscendC 实现约束

### 1. 非对齐处理
- **CopyOut1 非对齐 (关键)**: 每核将 `partial_sum` (1 个 float = 4 字节) 写入 workspace GM — **4 字节不是 32 字节的倍数**, 必须使用 DataCopyPad: `DataCopyPad(workspaceGm[programId], partialLocal, {1, sizeof(float), 0, 0})`
- **CopyIn2 对齐**: Core 0 读取 n_cores=16 个 float (64 字节) = 32 × 2, 满足对齐 ✓, 可使用普通 DataCopy
- **尾块非对齐**: 若 `elems_per_core % tile_size ≠ 0`, 尾块 DataCopy 同样需要检查对齐, 不满足时使用 DataCopyPad

### 2. 数据类型泛化
- **精度阈值**: float32 max_diff > 1e-4 = 逻辑错误 (标量 loss 输出, 不接受固定比例偏差)
- **float16 中间累加**: Phase 1 的 tile_sum 累加建议在 float32 精度下执行以避免 float16 累加精度损失

### 3. Tiling 优化与 Reduce 对齐
- **Phase 2 ReduceSum count=16 不足 64 倍数**: n_cores=16 < 64, 须将 workspace_in_ub padding 到 64 个元素, 多余部分 `Duplicate(0.0f, 64)` 后再调用 `ReduceSum(result, workspace_in_ub, work_buf, 64)`
- **Phase 2 work_buf 初始化**: 调用 ReduceSum 前必须 `Duplicate(workLocal, 0.0f, 64)`
- **inner_loops 为 0 的边界情况**: 当 `elems_per_core < tile_size` 时 inner_loops=0, Kernel 必须有尾块处理逻辑 (使用 actualLength = elems_per_core) 确保 partial_sum 被正确计算, 否则 partial_sum=0 → MSE=0

### 4. 同步指令插入
- **SyncAll() 是必须的 (关键)**: Phase 1 各核 DataCopyPad 写入 workspace GM 后, Phase 2 Core 0 DataCopy 读取 workspace 前, **必须插入 `AscendC::SyncAll()`**:
  ```cpp
  // Phase 1 末尾 (所有核执行)
  DataCopyPad(workspaceGm[programId], partialLocal, {1, sizeof(float), 0, 0});
  AscendC::SyncAll();  // ← 必须在此处同步
  // Phase 2 (仅 Core 0 执行)
  if (programId == 0) {
      DataCopy(workspaceInLocal, workspaceGm[0], 16);
      // ...
  }
  ```
- **缺少 SyncAll 的后果**: Core 0 在其他核写完前读取 workspace, 读到旧值 (0 或上轮残留), 导致 MSE 结果偏小或随机错误
