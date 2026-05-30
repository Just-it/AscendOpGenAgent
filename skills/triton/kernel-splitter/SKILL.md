---
name: kernel-splitter
description: >
  多 Case 专用 Kernel 分裂 Skill — 在泛用 Kernel 优化完成后，针对不同 Shape/Case 特征
  生成专用 Kernel，构建智能调度器，实现性能最大化。失败自动回退到泛用 Kernel。
argument-hint: >
  输入：base_kernel_path（泛用 Kernel）、task_desc_path（任务描述）、baseline_perf（泛用性能数据）。
  输出：分裂后的专用 Kernel 集合 + 调度器代码。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# Kernel Splitter Skill

<role>
你是一个高性能算子分裂优化专家。你的任务是在泛用 Kernel 性能未达标（`speedup_vs_torch < 0.8`）时，
针对不同 Case 的 Shape/Dtype 特征生成专用 Kernel，并构建智能调度器，
确保每个 Case 都能获得最优性能，同时保持精度一致性和接口兼容性。
</role>

## 触发条件

同时满足以下两项才执行，否则直接跳过：
1. `total_cases > 1`（多 case 场景）
2. Phase 4 最终 `speedup_vs_torch < 0.8`（性能未达标）

## 核心原则

1. **精度优先**：专用 Kernel 必须 100% 通过精度验证，否则回退到泛用 Kernel
2. **性能底线**：专用 Kernel 性能必须 ≥ 泛用 Kernel，否则不采用
3. **调度透明**：对外接口不变，内部自动路由到最优 Kernel
4. **渐进分裂**：按 Case 特征分组，不盲目一对一，优先合并相似特征

## 工作流程

```
输入：base_kernel.py + task_desc.py + baseline_perf.json
    ↓
[1. Case 特征分析] → 按经验以及特征计算模式分组
    ↓
[2. 专用 Kernel 生成] → 为每组生成特化实现
    ↓
[3. 精度验证] → verify.py 独立验证每组
    │   ├─ 通过 → 继续
    │   └─ 失败 → 标记该组使用泛用 Kernel
    ↓
[4. 性能测试] → benchmark.py 测试每组
    │   ├─ 优于基线 → 采纳
    │   └─ 劣于/等于基线 → 标记该组使用泛用 Kernel
    ↓
[5. 调度器构建] → 构建 ModelNew 路由逻辑
    ↓
[6. 集成验证] → 全量 Case 端到端测试
    ↓
输出：split_kernel.py（含调度器 + 多个专用 Kernel）
```

## Step 1: Case 特征分析与分组

### 1.1 加载 Reference 经验

首先加载对应的参考文档，根据算子类型选择：

| 算子类型 | 识别特征 | 加载文档 |
|---------|---------|---------|
| **Reduce** | `sum/mean/max/min/softmax/layernorm` 等归约操作 | `references/reduce-split.md` |
| **广播逐元素** | `add/sub/mul/div` + 存在 shape 不等 | `references/broadcast-elemwise-split.md` |

### 1.2 经验命中判定

读取参考文档中的分组维度表，检查当前任务是否命中：

**Reduce 类命中条件**：
- 任务包含归约操作，且存在多个 Case
- 按 `inner_size`（reduce 轴位置）分组：`==1` → reduce-last，`>1` → reduce-non-last
- 若命中，必须使用文档中的分组建议和 BLOCK 配置

**广播逐元素命中条件**：
- 任务为逐元素操作，且 Case 间存在 shape 不等（需要广播）
- 按 `(out_ndim, broadcast_dims)` 分组：无广播 / 2D dim0/dim1 / 3D/4D
- 若命中，必须使用文档中的分组建议和 BLOCK 配置

### 1.3 未命中时的性能瓶颈分析与分组

若未命中任何参考文档，执行以下步骤：

#### 1.3.1 读取 Phase 4 性能文件

从工作目录读取 Phase 4 最终性能数据：
- 优先读取：`{工作目录}/output/optimized_perf_result.json`
- 备选读取：`{工作目录}/output/perf_result.json`

#### 1.3.2 筛选性能瓶颈用例

从 `per_shape_results` 中提取 `speedup_vs_torch < 0.3` 的用例（即加速比低于 0.3，性能显著劣于 PyTorch 参考实现）。

```python
# 伪代码示例
bottleneck_cases = []
for shape_result in perf_data["per_shape_results"]:
    if shape_result["speedup_vs_torch"] is not None and shape_result["speedup_vs_torch"] < 0.3:
        bottleneck_cases.append(shape_result)
```

#### 1.3.3 分类归因

对筛选出的瓶颈用例，按以下维度分类归因：

| 归因维度 | 判定条件 | 典型原因 |
|---------|---------|---------|
| **Shape 过小** | 元素数 < 1024 | Kernel 启动开销占比过高 |
| **Shape 过大** | 元素数 > 10M | 寄存器溢出、缓存未命中 |
| **非对齐访问** | shape 非 2 的幂次 | mask 分支导致性能下降 |
| **跨步访存** | stride > 1 且非连续 | 内存带宽利用率低 |
| **特殊 dtype** | bf16/int8 等低精度 | 向量化策略不匹配 |

#### 1.3.4 生成分组

为每个归因类别生成专属分组：
- 同一归因类别的 Case 合并为一组
- 每组记录 `case_indices`、`bottleneck_reason`、`baseline_perf`
- 输出分组清单：`[{group_id, cases, features, bottleneck_reason, baseline_perf}]`

**示例输出**：
```json
[
  {
    "group_id": "grp_small_shape",
    "case_indices": [1, 3, 5],
    "features": {"element_count": "<1024", "dtype": "float16"},
    "bottleneck_reason": "kernel_launch_overhead",
    "baseline_perf": {"speedup_vs_torch": 0.15}
  },
  {
    "group_id": "grp_strided_access",
    "case_indices": [7, 9],
    "features": {"stride": "non_contiguous", "dtype": "float16"},
    "bottleneck_reason": "strided_memory_access",
    "baseline_perf": {"speedup_vs_torch": 0.22}
  }
]
```

## Step 2: 专用 Kernel 生成

对每个分组，基于泛用 Kernel 进行特化：

### 特化策略

| 策略 | 适用场景 | 优化方向 |
|------|---------|---------|
| **固定 constexpr** | Shape 固定或范围极小 | 将 BLOCK_SIZE、grid 等硬编码为 constexpr |
| **展开循环** | 小 Shape 场景 | 消除循环开销，完全展开 |
| **调整 Tiling** | 特定 Shape 比例 | 优化 tile 尺寸匹配 UB 容量 |
| **简化边界** | 尺寸对齐的 Case | 移除 mask 检查，使用无分支 load/store |
| **专用规约** | 特定规约轴 | 选择最优的 reduce 策略（原子/二分/树形） |

### 生成要求

- 每个专用 Kernel 命名格式：`{op_name}_kernel_{group_id}`
- 保持输入输出签名与泛用 Kernel 一致
- 代码必须完整可编译，禁止占位符

### 性能优化（可选）

生成专用 Kernel 后，**必须调用 `latency-optimizer` skill** 对每个专用 Kernel 进行进一步优化：

```
对每个专用 Kernel:
  1. 调用 latency-optimizer skill
  2. 按顺序检查 13 个优化点（constexpr/tiling/分核/...）
  3. 命中则应用优化策略
  4. 执行 checklist 检查确保代码规范
  5. 输出优化后的专用 Kernel
```

**注意**：
- 优化是可选步骤，若 latency-optimizer 报告无更多优化点则跳过
- 优化后必须保持精度一致性，不得改变算子功能
- 每个专用 Kernel 独立优化，互不影响

## Step 3: 精度验证

**必须使用** `kernel-verifier` 的 `verify.py` 脚本，对每个专用 Kernel 独立验证：

```bash
python3 <kernel-verifier-path>/scripts/verify.py \
    --op_name <op_name> \
    --verify_dir <split_verify_dir>/<group_id> \
    --triton_impl_name <group_kernel_name> \
    --timeout 300
```

**判定**：
- `passed_cases == total_cases` → 通过，进入 Step 4
- 任何失败 → 该组标记为 `fallback_to_base`，跳过后续步骤

## Step 4: 性能测试

**必须使用** `kernel-verifier` 的 `benchmark.py` 脚本：

```bash
python3 <kernel-verifier-path>/scripts/benchmark.py \
    --op_name <op_name> \
    --verify_dir <split_verify_dir>/<group_id> \
    --triton_impl_name <group_kernel_name> \
    --warmup 5 --repeats 50 \
    --output <split_perf_dir>/<group_id>_perf.json
```

**判定**：
- `speedup_vs_torch >= baseline_speedup` → 采纳
- 否则 → 标记为 `fallback_to_base`

## Step 5: 调度器构建

构建统一的 `ModelNew` 类，**必须**将路由逻辑封装在独立的 `_route` 方法中，`forward()` 仅负责调用该方法：

```python
class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 初始化所有专用 Kernel 和泛用 Kernel
        self.base_kernel = BaseKernel(...)
        self.specialized_kernels = {
            group_id: SpecializedKernel(group_id, ...)
            for group_id, adopted in adopted_groups.items()
        }

    def forward(self, *args):
        # forward 保持极简，仅调用一次路由函数
        return self._route(*args)

    def _route(self, *args):
        # 路由逻辑全部在此
        # 1. 提取输入 shape/dtype 特征
        # 2. 匹配分组规则
        # 3. 返回对应 kernel 启动结果，若无匹配则返回 base_kernel 结果
        if condition_1:
            return kernel_1[grid](...)
        elif condition_2:
            return kernel_2[grid](...)
        else:
            return self.base_kernel[grid](...)
```

**关键约束**：
- **禁止**在 `forward()` 中直接编写 `if-elif-else` 路由分支。
- **必须**使用 `_route` 方法封装路由逻辑。
- 路由开销必须 < 0.1ms（使用简单的 shape 比较，禁止复杂计算）。

## Step 6: 集成验证

对分裂后的完整代码执行全量验证：

1. 使用 `verify.py` 验证所有 Case 精度通过
2. 使用 `benchmark.py` 测试整体性能
3. 生成 `split_summary.json`，包含：
   - 每组采用的 Kernel 类型（specialized / base）
   - 每组性能对比（vs baseline）
   - 整体几何平均加速比

## 关键约束

| 约束 | 说明 |
|------|------|
| **触发条件** | 仅 `total_cases > 1` 且 `speedup_vs_torch < 0.8` 时执行，否则跳过 |
| **精度零妥协** | 任何专用 Kernel 精度不通过，立即回退到泛用 Kernel |
| **性能底线** | 专用 Kernel 必须 ≥ 泛用 Kernel 性能，否则不采用 |
| **路由封装** | 路由逻辑必须封装在 `_route` 方法中，`forward` 仅调用该方法 |
| **代码自包含** | 所有 Kernel 和调度逻辑必须在同一文件内 |
| **禁止过度分裂** | 相似 Case 必须合并分组，禁止 1-to-1 无意义分裂 |
| **回退安全** | 路由逻辑必须包含兜底机制，确保 100% 覆盖所有 Case |

## 输出格式

最终输出 `split_kernel.py`，结构如下：

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

# === 泛用 Kernel（保留原样） ===
@triton.jit
def {op_name}_base_kernel(...): ...

# === 专用 Kernel 1 ===
@triton.jit
def {op_name}_kernel_grp1(...): ...

# === 专用 Kernel 2 ===
@triton.jit
def {op_name}_kernel_grp2(...): ...

# === 调度器 ===
class ModelNew(nn.Module):
    def __init__(self, ...): ...
    def _route(self, ...): ...
    def forward(self, ...): ...
```

## 参考资料

| 文档 | 用途 |
|------|------|
| `references/reduce-split.md` | Reduce 类算子 Kernel 分裂经验 |
| `references/broadcast-elemwise-split.md` | 广播逐元素算子 Kernel 分裂经验 |
