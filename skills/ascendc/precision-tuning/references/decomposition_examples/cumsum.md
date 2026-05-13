# CumSum 计算分解示例

## 算子信息
- op_name: cumsum
- category: reduction (前缀和, 有顺序依赖)
- 计算模式: **前缀累加**
- 归约维度: dim=2, 输入先 permute 使 scan axis 到 axis 0
- 输入 shape: [16, 32, 64] → permute → [64, 16, 32], dtype: float32
- 输出 shape: [16, 32, 64] (与输入同 shape), dtype: float32

## 来源文件
- reference: `cumsum_reference.py` → `torch.cumsum(x, dim=self.dim)`
- functional: `cumsum_functional.py` → permute(dim→0) → `torch.cumsum(xt, dim=0)` → permute back
- DSL: `cumsum_dsl.py` → 顺序扫描 scan_len 步, 每步 acc_ub += x_ub
- op_desc: `cumsum_op_desc.json` → `attributes.dim=2`

## 计算链分解

### Step 0: 输入 (permute 后)
- 原始 tensor: shape [16, 32, 64], dim=2
- permute 后: shape [64, 16, 32] (scan axis 在 axis 0)
- scan_len = 64 (扫描长度)
- inner_size = 16 × 32 = 512 (每个 scan 位置的切片大小)
- dtype: float32, 数值范围: torch.rand → [0, 1)

### Step 1: 累加器初始化
- 操作: `acc_ub = 0` (每个 inner tile 的前缀累加器归零)
- DSL 对应: `tl.duplicate(acc_ub, 0.0)`
- 输出 shape: [tile_inner] = [min(1024, 512)] = [512]
- **精度风险点**:
  - 必须在每个 task (inner tile) 开始时初始化, 不能跨 task 残留
  - 若未初始化, 前一个 task 的累加值会混入当前 task → 数据污染
  - AscendC 中 AllocTensor 后内容不确定, 必须显式 Duplicate

### Step 2: 顺序扫描 (scan_len 步)
- 操作: 对 i = 0, 1, ..., scan_len-1:
  - 加载 `x[i, inner_slice]` (shape [tile_inner])
  - `acc_ub += x[i]`
  - 输出 `output[i, inner_slice] = acc_ub`
- DSL 对应:
  ```
  for i in range(scan_len):
      tl.load(input_ptr + offsets, x_ub)      # copyin
      tl.vadd(acc_ub, acc_ub, x_ub)           # compute: 前缀累加
      tl.vadd_scalar(out_ub, acc_ub, 0.0)     # compute: 复制到输出 buffer
      tl.store(output_ptr + offsets, out_ub)   # copyout
  ```
- 输入: 每步加载 x[i] 的 tile_inner 个元素
- 输出: 每步写出 acc 的当前值 (前缀和)
- scan_len = 64, 每步的 GM 偏移: `i * inner_size + inner_base`
- **精度风险点**:
  - **顺序依赖**: 步骤 i 的输出依赖步骤 0..i-1 的累加, **不可并行化**
  - 64 次 float32 累加, 精度损失: 最后一步累积了 64 个 [0,1) 的和, 期望值 ~32, 精度足够
  - **tile_inner 对齐**: tile_inner=512, sizeof(float32)=4, 512×4=2048 bytes, 满足 32-byte 对齐 ✓
  - **inner_size 与 tile_inner 的关系**: inner_size=512, tile_inner=512, 恰好 1 个 task 覆盖所有 inner 元素
  - 若 inner_size > tile_inner (如更大的 tensor), 多个 task 独立处理各自的 inner slice — 各 task 之间无依赖
  - **GM 偏移计算**: `base = i * inner_size + inner_base`, 其中 inner_base = task_id × tile_inner — 需确保 inner_base 正确
  - **DSL 中的 vadd_scalar(out_ub, acc_ub, 0.0)**: 这是将 acc_ub 复制到 out_ub 的方式 (加 0 = 复制), AscendC 中应使用 DataCopy(UB→UB) 或等效操作
- **知识库关联**: 无直接匹配 (cumsum 不是标准归约), 但 count 对齐问题同 #7

## 误差传播链

```
Step 1 累加器未初始化
  → 所有前缀和都有常数偏移 → pattern: uniform_offset (偏移量恒定)

Step 2 GM 偏移计算错误
  → 加载了错误位置的数据 → pattern: all_wrong

Step 2 顺序被打乱 (如并行化了不应并行的循环)
  → 前缀和不满足 output[i] = sum(x[0:i+1]) → 随机偏差
```

## DSL tiling 策略要点

从 `cumsum_dsl.py` 提取:
- **scan axis permute**: Host 将 dim=2 permute 到 axis 0, Kernel 始终沿 axis 0 扫描
- `n_cores = 16`, 按 inner tile 分配任务
- `tile_inner = min(1024, inner_size)` = min(1024, 512) = 512
- `total_tasks = ceil(inner_size / tile_inner)` = 1
- `tasks_per_core = total_tasks / n_cores` — 需要 total_tasks 能被 n_cores 整除
- **注意**: 本例 total_tasks=1 而 n_cores=16, 意味着只有 1 个核实际执行, 其余 15 个核空转
- 3 个 UB buffer: x_ub, acc_ub, out_ub (均为 tile_inner 大小)

## 与 AscendC Kernel 的对照要点

1. **permute 处理**: 输入是否已在 Python 层完成 permute? Kernel 是否假设 scan axis 在 axis 0?
2. **累加器初始化**: 每个 task 开始时是否 Duplicate(acc_ub, 0)?
3. **scan 循环顺序**: for i in range(scan_len) **必须串行执行**, 不能被优化为并行
4. **GM 偏移**: `i * inner_size + inner_base` 中 inner_size 和 inner_base 是否正确?
5. **acc → out 复制**: DSL 用 `vadd_scalar(out_ub, acc_ub, 0.0)` 复制, AscendC 是否用等效方式?
6. **每步都 copyout**: 前缀和要求每步都写出结果, 不能只在最后一步写出
7. **tasks_per_core 越界**: 若 total_tasks < n_cores, 部分核的 task_start ≥ total_tasks, 需要跳过
