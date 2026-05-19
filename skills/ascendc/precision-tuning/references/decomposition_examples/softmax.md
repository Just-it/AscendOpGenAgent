# Softmax 计算分解示例

## 算子信息
- op_name: softmax
- category: activation (含归约)
- 计算模式: **单行归约**
- 归约维度: dim=-1 (最后一维, 即 features 维)
- 输入 shape: [32, 768], dtype: float32
- 输出 shape: [32, 768], dtype: float32

## 来源文件
- reference: `softmax_reference.py` → `F.softmax(x, dim=self.dim)`
- DSL: `softmax_dsl.py` → `softmax_kernel` 中的 5 步计算
- op_desc: `softmax_op_desc.json` → `attributes.dim = -1`

## 计算链分解

### Step 0: 输入
- tensor: x
- shape: [32, 768]
- dtype: float32
- 数值范围: 取决于 `get_inputs()` 中的 `torch.rand`, 通常 [0, 1)

### Step 1: ReduceMax (数值稳定性)
- 操作: `row_max = max(x, dim=-1, keepdim=True)`
- DSL 对应: `tl.reduce_max(shared_ub, row_ub, shared_ub)` + `extract_scalar(shared_ub, 0)`
- 输入: Step 0 的输出, shape [32, 768]
- 输出 shape: [32, 1] (每行一个标量)
- 数值范围预期: [0, 1) (因为输入是 torch.rand)
- **精度风险点**:
  - Padding 值为 0 时参与 Max: 若输入全负则 max 被错误抬高到 0
  - ReduceMax 的 count 参数必须为 64 的倍数, 不对齐时硬件行为未定义
  - Buffer 未用 Duplicate(-INF) 初始化时, padding 区域的默认值可能干扰结果
- **知识库关联**: #0 尾块 Padding 值污染, #7 ReduceMax count 未对齐, #9 Reduction CHECKLIST

### Step 2: Sub (去中心化)
- 操作: `x_shifted = x - row_max` (广播减法)
- DSL 对应: `tl.vsub_scalar(exp_ub, row_ub, row_max)`
- 输入: Step 0 的 x + Step 1 的 row_max
- 输出 shape: [32, 768]
- 数值范围预期: (-∞, 0] (最大值变为 0, 其余为负)
- **精度风险点**:
  - 如果 Step 1 的 max 不正确 (如被 padding 抬高), x_shifted 中将出现正值
  - 正值会导致 Step 3 的 Exp 溢出
  - 此步本身不引入精度误差, 但会传播 Step 1 的误差

### Step 3: Exp
- 操作: `exp_val = exp(x_shifted)`
- DSL 对应: `tl.vexp(exp_ub, exp_ub)`
- 输入: Step 2 的 x_shifted, shape [32, 768]
- 输出 shape: [32, 768]
- 数值范围预期: (0, 1] (因为 x_shifted ≤ 0, exp(0)=1, exp(负数)<1)
- **精度风险点**:
  - 若 Step 2 输入有正值 (Step 1 错误导致), exp 可能溢出 (float32 下 exp(88) ≈ 1.65e38)
  - **Padding 区域**: exp(0) = 1 (若 padding 为 0), 这些虚假的 1 会污染后续 ReduceSum
  - 若 padding 初始化为 -INF (ReduceMax 正确做法), 则 Sub 后 padding 仍为 -INF, exp(-INF)=0, 不污染 — 这是理想情况
- **知识库关联**: #0 尾块 Padding 值污染, #6 除零或 Exp 溢出

### Step 4: ReduceSum
- 操作: `exp_sum = sum(exp_val, dim=-1, keepdim=True)`
- DSL 对应: `tl.reduce_sum(shared_ub, exp_ub, shared_ub)` + `extract_scalar(shared_ub, 0)`
- 输入: Step 3 的 exp_val, shape [32, 768]
- 输出 shape: [32, 1] (每行一个标量)
- 数值范围预期: (0, 768] (每行 exp 值之和, 最大 768 × 1 = 768)
- **精度风险点**:
  - Padding 区域 exp(0)=1 会增大 sum, 导致最终 softmax 结果偏小
  - ReduceSum 的 count 参数同样要求 64 倍数对齐
  - Padding 区域对 Sum 的语义: 正确值应为 0 (即 padding 元素不贡献到 sum)
- **知识库关联**: #2 归约操作 Padding 值语义不匹配, #7 Reduce Count 未对齐

### Step 5: Div (归一化)
- 操作: `output = exp_val / exp_sum` (广播除法)
- DSL 对应: `tl.vdiv_scalar(out_ub, exp_ub, row_sum)`
- 输入: Step 3 的 exp_val + Step 4 的 exp_sum
- 输出 shape: [32, 768]
- 数值范围预期: [0, 1], 每行和为 1
- **精度风险点**:
  - 若 exp_sum 接近 0 (不应该在正常情况发生, 但 NaN 输入可能导致)
  - 若 exp_sum 被 padding 污染而偏大, 所有输出值都会偏小 → uniform_offset 模式

## 误差传播链

```
Step 1 (Max) 错误
  → Step 2 (Sub) 所有元素偏移不正确
    → Step 3 (Exp) 部分值溢出或偏大
      → Step 4 (Sum) 值偏大
        → Step 5 (Div) 所有值偏小
```

典型表现: pattern = uniform_offset 或 all_wrong

## DSL tiling 策略要点

从 `softmax_dsl.py` 提取:
- `n_cores = 32`, 按行分配 (`rows_per_core = rows // n_cores`)
- `tile_length = cols = 768` (整行放入 UB, 无需多 tile)
- 归约维 (dim=-1) 在单核内完整处理, **不跨核切分**
- 4 个 UB buffer: row_ub, exp_ub, shared_ub, out_ub

## 与 AscendC Kernel 的对照要点

1. Compute() 中 ReduceMax 的 count 参数是否 = tileLength (而非动态 actualLength)
2. ReduceMax 前 buffer 是否用 Duplicate(-INF) 初始化
3. Exp 前 Sub 的标量来源是否正确 (从 sharedLocal.GetValue(0) 提取)
4. ReduceSum 前 padding 区域是否已被正确初始化为 0
5. ReduceSum 的 count 参数是否与 ReduceMax 一致
6. Div 的除数来源是否正确 (sharedLocal.GetValue(0) 在 ReduceSum 之后)
7. Host TilingFunc 中 tileLength 是否 = cols (保持归约维完整)

## AscendC 实现约束

### 1. 非对齐处理
- **触发条件**: `tileLength × sizeof(float32) = tileLength × 4` 字节, 若不是 32 的倍数则须使用 DataCopyPad
- **float32 对齐要求**: tileLength 必须是 8 的倍数 (8 × 4 = 32 字节); float16 要求 16 的倍数
- **典型场景**: cols=768 时, 768 × 4 = 3072 = 32 × 96, 对齐 ✓; cols=100 时, 400 = 32 × 12.5, 需 DataCopyPad
- **DataCopyPad 用法 (GM→UB)**: `DataCopyPad(dstLocal, srcGlobal, {1, tileLength*sizeof(float), 0, 0}, {false, 0, 0, 0})`
- **DataCopyPad 用法 (UB→GM)**: `DataCopyPad(dstGlobal, srcLocal, {1, tileLength*sizeof(float), 0, 0})`

### 2. 数据类型泛化
- **精度阈值 (AbsoluteThreshConfig)**: float32 max_diff > 1e-4 = 逻辑错误; float16 max_diff > 1e-2 = 逻辑错误, 1e-3~1e-2 = float16 精度损失
- **float16 归约精度**: ReduceMax/ReduceSum 的中间计算在 float16 下精度较低, 出现 float16 精度损失时检查是否需要在 float32 下执行归约再转回 float16
- **负无穷常量**: float32 用 `-3.402823466e+38f` 或 `(float)(-INFINITY)`; float16 用 `-65504.0f`; **禁止使用 `AscendC::INFINITY`**

### 3. Tiling 优化与 Reduce 对齐
- **ReduceMax/ReduceSum count 必须为 64 的倍数**: 若 tileLength 不满足, 须将 count 向上对齐到 64 的倍数, padding 区域预先用 Duplicate 初始化为安全值再调用 Reduce
- **ReduceMax work_buf 初始化 (强制)**: 调用 ReduceMax 前必须 `Duplicate(workLocal, -3.402823466e+38f, count)`, 否则 work buffer 残留值会使 max 结果偏高
- **ReduceSum work_buf 初始化**: 调用 ReduceSum 前必须 `Duplicate(workLocal, 0.0f, count)`, 避免残留累加
- **padding 区域初始化**: ReduceMax 前, padding 区域须 Duplicate(-INF) 使其不影响 max; ReduceSum 前, exp(padding) 区域须为 0

### 4. 同步指令插入
- **单行归约无需 SyncAll**: Softmax 各核独立处理各自行, 无跨核共享 GM 写 → **不需要 SyncAll()**
- **TBuf 不可直接 DataCopy 写 GM**: sharedBuf (TBuf<VECCALC>) 中的 row_max/row_sum 等标量用 `.GetValue(0)` 提取, 最终输出结果必须经 outQueue 路径写出
- **outQueue 路径 (强制)**: AllocTensor(outLocal) → 计算写入 outLocal → outQueue.EnQue(outLocal) → outQueue.DeQue() → DataCopy(outputGm[], outLocal, tileLength)
- **绕过 outQueue 的后果**: 直接 `DataCopy(outputGm[], tbufLocal, count)` 会因绕过 VECOUT 同步导致数据未写出 (输出全零)
