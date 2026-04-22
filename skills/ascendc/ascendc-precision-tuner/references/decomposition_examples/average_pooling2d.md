# AvgPool2d 计算分解示例

## 算子信息
- op_name: average_pooling2d
- category: pooling
- 计算模式: **滑窗累加**
- 输入 shape: [1, 3, 224, 224] (NCHW), dtype: float32
- 输出 shape: [1, 3, 111, 111] (NCHW), dtype: float32
- 特殊属性: kernel_size=[3,3], stride=[2,2], padding=[0,0], ceil_mode=false, count_include_pad=true

## 来源文件
- reference: `average_pooling2d_reference.py` → `nn.AvgPool2d(kernel_size, stride, padding, ...)`
- DSL: `average_pooling2d_dsl.py` → 逐输出位置遍历 kernel window, 累加 → 除以面积
- op_desc: `average_pooling2d_op_desc.json` → 完整的 pooling 参数

## 计算链分解

### Step 0: 输入
- x: shape [N=1, C=3, H=224, W=224], dtype: float32, layout: NCHW
- 数值范围: torch.rand → [0, 1)
- 输出尺寸计算: out_h = (224 + 0 - 3) / 2 + 1 = 111, out_w = 同理 = 111

### Step 1: 窗口定位 + 元素加载
- 操作: 对每个输出位置 (b, c, oh, ow), 计算对应输入窗口的起始坐标 `h0 = oh * stride_h`, `w0 = ow * stride_w`
- DSL 对应: 循环 `for kh ... for kw ...`, 逐元素从 GM 加载
- 输入: GM 中的单个元素
- **精度风险点**:
  - 窗口可能部分超出输入边界 (当 padding > 0 时), 需要边界检查
  - 本例 padding=[0,0], 无边界越界风险
  - GM 偏移计算: `b * C * H * W + c * H * W + ih * W + iw`, 涉及 4 维索引, 容易出错
  - DSL 中逐元素加载 (`tl.load(... x_ub[0:1])`), 效率低但逻辑简单
- **知识库关联**: #5 数据布局不一致

### Step 2: 累加
- 操作: `sum_ub += x_element`, 对窗口内所有有效元素累加
- DSL 对应: `tl.vadd(sum_ub, sum_ub, x_ub[0:1])`, 循环 kernel_h × kernel_w 次
- 累加次数: 最多 3 × 3 = 9 次
- 累加器初始化: `tl.vconst(sum_ub, 0.0)` (每个输出位置重新初始化)
- **精度风险点**:
  - 9 次 float32 累加, 精度损失可忽略
  - **count_include_pad=true**: 有效窗口面积始终 = kernel_h × kernel_w = 9 (即使边缘位置窗口部分越界)
  - **count_include_pad=false**: 有效面积 = 实际覆盖的输入元素数, 边缘位置 < 9
  - 当 padding > 0 且 count_include_pad=false 时, 除数计算是高频精度错误点
- **知识库关联**: #10 Pooling CHECKLIST 项 (1)(3): padding 区域是否正确参与/排除计算

### Step 3: 除以面积 (均值)
- 操作: `output = sum_ub / effective_kernel_area`
- DSL 对应: `tl.vdiv_scalar(out_ub, sum_ub, float(effective_kernel_area))`
- 输入: Step 2 的 sum_ub + 面积标量
- 输出: 单个输出元素值
- 数值范围预期: [0, 1) (输入 [0,1) 的均值仍在 [0,1))
- **精度风险点**:
  - **effective_kernel_area 计算**: 本例 count_include_pad=true, 所以恒等于 9
  - 若 Kernel 错误地计算了 effective_kernel_area (如使用边缘实际面积), 结果会偏大或偏小
  - 特别是 padding>0 + count_include_pad=true 的组合: 即使窗口越界, 分母仍为 kernel_h × kernel_w
  - 若 Kernel 用 count_include_pad=false 的逻辑处理 count_include_pad=true, 边缘输出值会偏大
- **知识库关联**: #10 Pooling CHECKLIST 项 (3): AvgPool 除数是否排除 padding 元素

## 误差传播链

```
Step 1 GM 偏移计算错误 (NCHW 4 维索引)
  → 加载错误数据 → pattern: all_wrong 或 dimension_concentration

Step 2 累加器未重新初始化
  → 前一个输出位置的残留值累加到当前位置 → pattern: scattered

Step 3 effective_kernel_area 计算错误
  → 边缘位置输出偏大/偏小 → pattern: boundary_concentration
  → 所有位置偏差 (若全局面积错误) → pattern: uniform_offset
```

## DSL tiling 策略要点

从 `average_pooling2d_dsl.py` 提取:
- `n_cores = 16`, 按输出元素 (N×C×out_h×out_w) 分配 tasks
- `tasks_per_core = total_tasks // n_cores`
- **逐元素处理**: 每个 task 处理一个输出位置, tile_h=1, tile_w=1
- 任务解码: `task_id → (b, c, oh, ow)` 通过除法和取余
- 3 个 UB buffer: x_ub, sum_ub, out_ub (极小, 只需 1 个元素)

**注意**: DSL 中的逐元素加载 (`tl.load(... x_ub[0:1])`) 效率极低, AscendC 实现可能会优化为按行加载整个窗口, 这种优化可能引入新的对齐和边界问题。

## 与 AscendC Kernel 的对照要点

1. **4 维 GM 偏移**: `b * C * H * W + c * H * W + ih * W + iw` 是否正确?
2. **窗口边界**: h0 + kh 是否越界? 是否有 `if 0 <= ih < H and 0 <= iw < W` 保护?
3. **累加器初始化**: 每个输出位置是否重新 Duplicate(0)?
4. **effective_kernel_area**: count_include_pad=true 时是否恒等于 kernel_h × kernel_w?
5. **AscendC 优化**: 是否从逐元素加载优化为按行 DataCopy? 优化后 padding 和边界是否正确?
6. **输出 GM 偏移**: `b * C * out_h * out_w + c * out_h * out_w + oh * out_w + ow` 是否正确?
7. **输出尺寸计算**: Host TilingFunc 中 out_h/out_w 公式与 reference 是否一致?

## AscendC 实现约束

### 1. 非对齐处理
- **channels 维度不保证对齐**: C=3 时 3 × 4 = 12 字节, 不是 32 字节的倍数; C=5 时 20 字节, 同样不对齐
- **每次 DataCopy 都需检查对齐**: CopyIn 读取 `channels` 个 float 时, `channels × sizeof(float)` 若不是 32 的倍数, 必须用 DataCopyPad
- **优化后的按行加载**: 若 Kernel 将逐元素加载优化为按行加载整个 kernel window, 每行 `kernel_w × channels` 个元素 (如 3 × 3 = 9, 9 × 4 = 36 字节 = 32 × 1.125, 非对齐), 同样须 DataCopyPad
- **DataCopyPad 用法 (GM→UB)**: `DataCopyPad(dstLocal, srcGlobal, {1, count*sizeof(float), 0, 0}, {false, 0, 0, 0})`

### 2. 数据类型泛化
- **精度阈值**: float32 max_diff > 1e-4 = 逻辑错误 (Pooling 均值结果应高精度)
- **累加器 dtype 一致性**: `Duplicate(sumLocal, 0.0f, count)` 的字面量类型须与计算 dtype 一致; float16 下用 `Duplicate(sumLocal, (half)0.0f, count)`

### 3. Tiling 优化与 Buffer 初始化
- **sum_buf 初始化 (强制)**: 每个输出位置计算开始前必须 `Duplicate(sumLocal, 0.0f, count)` 重置累加器; 未初始化时前一位置的残留值会污染当前位置 → pattern: scattered
- **tasks_per_core 尾块处理**: 若 `total_tasks % n_cores ≠ 0`, 最后一核的实际 task 数少于 tasks_per_core, 须正确处理边界不越界
- **windowsPerLoad 对齐检查**: 若批量处理多个 window 行时, `windowsPerLoad × channels × sizeof(float)` 须满足 32 字节对齐, 否则用 DataCopyPad

### 4. 同步指令插入
- **无跨核通信, 无需 SyncAll**: AvgPool2d 按输出位置分配任务, 各核独立计算 → **不需要 SyncAll()**
- **TBuf 不可直接 DataCopy 写 GM**: sum_buf (TBuf<VECCALC>) 中的累加结果不可直接 `DataCopy(outputGm[], sumLocal, channels)`; 必须经 outQueue 路径写出
- **outQueue 路径 (强制)**: AllocTensor(outLocal) → 除法结果写入 outLocal → outQueue.EnQue(outLocal) → outQueue.DeQue() → DataCopy(outputGm[outBase], outLocal, channels)
- **绕过 outQueue 的后果**: 直接用 TBuf 的 sumLocal 作为 DataCopy 源会因绕过 VECOUT 同步导致输出全零
