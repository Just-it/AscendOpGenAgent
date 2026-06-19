# 基准测试方法论

> 说明性能数据如何产生、如何采集、如何对比精度。
> 所有 benchmark 代码位于 `tests/npu/` 和 `tests/gpu/` 目录，精度测试代码位于 `tests/`。

---

## 零、Profiling-First 工作流（强制规则）

> **每次性能优化前必须先用 msprof 采集 profiling 数据，确认瓶颈后再行动。**

### 为什么需要这条规则

Session 1 中的教训：SIMT 参数调优（TI=1024、1024 线程、block 超订）全部无效——因为瓶颈是 atomicAdd 序列化（HBM 带宽仅 0.12%），不是线程数或 tile 大小。如果先 profile，可以直接发现 atomicAdd 是瓶颈，避免浪费 3+ 轮无效调优。

### 标准流程

```
1. msprof 采集基线数据
   └─ 确认瓶颈（带宽？计算？原子操作？pipeline stall？）
2. 设计优化方案（针对瓶颈）
3. 实现优化
4. msprof 采集优化后数据
   └─ 对比：瓶颈指标是否改善？总时间是否减少？
5. 若无改善 → 回到步骤 1 重新分析
```

### msprof 关键指标

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `aiv_vec_ratio` | AIV vector pipe 利用率 | 接近 1.0 = 计算密集 |
| `aiv_mte2_ratio` | DMA pipe 利用率 | SIMT 模式下通常 0（GM 访问走 vector pipe） |
| HBM bandwidth utilization | 实际带宽/理论带宽 | < 1% 说明不是带宽瓶颈 |
| per-atomicAdd cycles | 每次 atomicAdd 延迟 | 3~4 cycles = 接近硬件极限 |

### msprof 命令示例

```bash
# 在 A5 容器内执行
msprof --output=/tmp/msprof_out -- ./npu_prod_benchmark
# 分析结果
msprof --export=summary --output=/tmp/msprof_out
```

---

## 零.一、精度保持规则（强制）

> **每次代码修改后必须先验证精度，性能优化不允许引入精度损失。**

### 工作流

```
1. 修改代码（优化、重构、新功能）
2. 编译
3. 运行精度测试（GPU golden ref 逐元素对比）
   └─ 必须 PASS 所有 dtype + 所有 cluster/case
4. 只有精度 PASS 后才能运行性能 benchmark
5. 如果精度 FAIL → 定位根因 → 修复 → 回到步骤 3
```

### 精度阈值（不允许放宽）

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| fp32 | 1e-4 | 1e-4 | 严格匹配 |
| fp16 | 1e-2 | 1e-2 | fp16 精度范围 |
| bf16 | 2e-2 | 2e-2 | scatter-add 累加顺序差异 |

### 不允许的做法

- ❌ 放宽阈值让测试通过
- ❌ 新增 waiver 掩盖不匹配（除非确认是 scatter-add 累加顺序导致）
- ❌ 跳过精度测试直接看性能
- ❌ 在精度 FAIL 的情况下提交代码

---

## 零.二、性能 A/B 对比规则（强制，OL-27）

> **任何性能声明必须基于同条件 A/B 数据。违反此规则等同于发布虚假数据。**

### 硬性要求

1. **同一 NPU**：before/after 必须在同一个物理 NPU 上运行（不同 NPU 有硬件差异）
2. **同一 session**：不重启容器、不切换 NPU，背靠背运行 A 和 B
3. **A/B 方法**：`checkout old → build → benchmark → checkout new → build → benchmark`
4. **全覆盖**：每个被修改的 kernel 都必须有对应的性能数据行。如果 benchmark 不覆盖某 kernel，标注 **"性能未验证"**
5. **NPU 空闲确认**：benchmark 前必须 `npu-smi info` 确认无其他进程

### 不允许的做法

- ❌ 用不同 NPU 的数据声称"性能无退步"
- ❌ 用"趋势一致"、"数值接近"替代同条件 A/B
- ❌ benchmark 不覆盖修改的 kernel 却声称"全部验证"
- ❌ 默认"应该没问题"——没有数据就是"未验证"
- ❌ 跨 session 数据用于性能声明（只能标注为"参考"）

### 教训

E14: 在 NPU 1 跑 before、NPU 0 跑 after，声称"性能无退步"发布到文档。F16/BF16 kernel 完全没有性能数据却被包含在"ALL PASS"结论中。用户三次指正才纠正。

---

## 零.三、CANN Baseline 获取方法（NPUKernelBench）

> **CANN 内置算子 = 答案/baseline。** 性能报告必须对比 CANN，不是只报绝对数字。

### 获取 CANN baseline

```python
# 在 A5 容器内（需要 torch_npu + CANN 环境变量）
import torch, torch_npu, time
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp

x = torch.randn(n, dtype=torch.float32).npu()
# CANN 算子通过 torch_npu 自动调用
output = torch.nn.functional.gelu(x)  # → 调用 CANN 内置 Gelu kernel
```

### 报告格式

| n | CANN (ms) | 我们 (ms) | ratio (CANN/我们) | 精度 |
|---|-----------|-----------|:-----------------:|------|

- ratio > 1.0 = 我们比 CANN 快
- ratio < 1.0 = 我们比 CANN 慢
- 精度: 我们 vs CPU PyTorch (ground truth)，同时报告 CANN vs CPU 精度

### 注意事项
- CANN 通过 torch_npu graph mode 有更低的 launch overhead → 小 tensor CANN 通常更快
- 大 tensor 才能看到算法差异
- 精度对比必须三方: CANN vs CPU, 我们 vs CPU, 我们 vs CANN

### 教训 (GELU, 2026-04-08)

**裸调用 vs 框架调用差异巨大**：
用 C++ `aclrtlaunch` 裸调用对比 Python `torch_npu` 调用是**不公平的**。
裸调用 CANN 在小 tensor 上显示 ~4us（实际是 PyTorch dispatch overhead），
让我们的 ~10us kernel 看起来慢 2-4x。
用框架标准评测（两者都经 Python dispatch），差距从 "2-4x" 变成 "0.83-1.11x"。

**必须使用 NPUKernelBench 框架的标准工具评测**：
- Build: `python3 utils/build_ascendc.py <task> -v Ascend950PR_9589`
- Precision: `python3 utils/verification_ascendc.py <task>`
- Performance: `python3 utils/performance.py <task> all`
- 两者经过相同的 Python → PyTorch → NPU dispatch 路径

**精度声明不能跨实现比较**：
之前声称"我们精度比 CANN 高 1000x"是错误的——两者用不同的 erf 近似，
对比的 ground truth 也不同（C++ std::erf vs NPU hardware erf）。
正确做法：用框架的 `torch.allclose(ref, cand, atol=1e-2, rtol=1e-2)`。

---

## 一、测试数据

### 1.1 生产数据（Pooling）

**来源**：客户生产环境 assign data dump（step 47）。
**格式**：每个 cluster 的 `edge_in`(int32)、`edge_out`(int32)、`grad_table`(float32)、`pooling_table`(float32) 等二进制文件。
**规模**：61 clusters，最大 14M edges，82M 全局嵌入表，emb_dim 1-358。
**dtype**：仅 fp32（客户数据原生格式）。
**数据位置**：见 [BENCHMARK_RESULTS.md](../../../output/docs/BENCHMARK_RESULTS.md#测试数据位置)。

文件名格式：`pooling_{forward|backward}_dump_step_47_{tensor_name}_shape_{shape}_dtype_{dtype}_rank_0.bin`

**注意**：NPU benchmark 对 edge 索引做了 compact remapping（[0, N-1]），减少内存占用但保留 edge 数量和 dim（性能主要因素）。GPU 侧做相同 remapping 保证公平。

### 1.2 合成数据（Pooling fp16/bf16 + SG 全 dtype）

**生成方式**：GPU 端 [`tests/gpu/gpu_benchmark.py`](../../../tests/gpu/gpu_benchmark.py) 生成确定性随机数据（`torch.manual_seed(42)`），按 dtype cast 后 dump 为二进制文件。

**Pooling 合成 shapes**：
| Case | src | dst | edges | dim |
|------|-----|-----|-------|-----|
| small | 100 | 200 | 500 | 32 |
| medium | 1000 | 2000 | 50000 | 128 |

**SG 合成 shapes**：
| Case | experts | hidden_dim | token_num | top_k |
|------|---------|-----------|-----------|-------|
| small | 8 | 64 | 32 | 2 |
| medium | 64 | 256 | 512 | 4 |
| large | 128 | 1024 | 2048 | 4 |
| xlarge | 256 | 4096 | 4096 | 8 |
| prod_a | 64 | 256 | 8192 | 4 |
| prod_b | 128 | 512 | 4096 | 8 |

GPU 参考输出（forward reference）同时 dump，用于精度验证。

---

## 二、性能采集方法

### 2.1 GPU 侧（A100）

**工具**：CUDA Event（device 时钟，不含 host 开销）。
**代码**：[`tests/gpu/gpu_cuda_benchmark.cu`](../../../tests/gpu/gpu_cuda_benchmark.cu)

```cpp
// 正确做法：memset 在 event 外，只测 kernel
cudaMemset(d_out, 0, out_bytes);          // 清零（不计入时间）
cudaEventRecord(start_ev);                // GPU 时钟起点
kernel<<<grid, block>>>(...);             // 只测 kernel
cudaEventRecord(stop_ev);                 // GPU 时钟终点
cudaEventSynchronize(stop_ev);            // 等待完成
cudaEventElapsedTime(&ms, start_ev, stop_ev);  // device 侧时间差
```

**关键**：
- `cudaEventRecord` 是 stream-ordered 的，start/stop 之间只包含 kernel 执行时间
- **不包含** host 侧 launch overhead、cudaMemset、数据传输
- Warmup 3 次后取 10 次平均

### 2.2 NPU 侧（Ascend950PR）

**工具**：ACL Event（device 时钟，不含 host 开销）。
**代码**：[`tests/npu/npu_prod_benchmark.cpp`](../../../tests/npu/npu_prod_benchmark.cpp)

```cpp
// 正确做法：memset 在 event 外，用 aclrtEvent 测 device 时间
aclrtMemset(d_out, out_bytes, 0, out_bytes);   // 清零（不计入时间）
aclrtSynchronizeStream(stream);                 // 确保清零完成
aclrtRecordEvent(start_ev, stream);             // device 时钟起点
kernel_launch(nblk, stream, ...);               // 只测 kernel
aclrtRecordEvent(stop_ev, stream);              // device 时钟终点
aclrtSynchronizeEvent(stop_ev);                 // 等待完成
float ms;
aclrtEventElapsedTime(&ms, start_ev, stop_ev);  // device 侧时间差
```

**关键**：
- 使用 `aclrtEvent`（非 host chrono），与 GPU 侧 cudaEvent 对等
- **不包含** host 侧 aclrtSynchronizeStream 等待时间、launch overhead
- Warmup 3 次后取 10 次平均

### 2.3 与旧方法的差异

| 方面 | 旧方法（Batch 4 前） | 新方法（Batch 5 后） |
|------|---------------------|---------------------|
| GPU 计时 | cudaEvent（正确），但 memset 在 event 内 | cudaEvent，memset 移到 event 外 |
| NPU 计时 | `chrono::high_resolution_clock`（host 时钟） | `aclrtEvent`（device 时钟） |
| NPU 测量内容 | kernel + memset + launch overhead + sync wait | 纯 kernel 执行时间 |
| 影响 | NPU 数据偏大（特别是小 kernel），GPU/NPU 对比不完全公平 | 两侧都是 device 时钟，公平对比 |

> **注意**：切换到新方法后，所有历史基线数据需要重新采集。新旧数据不能直接混用对比。
> **规则**：必须使用 `aclrtEvent` device 时钟。如果 API 不可用，定位根因修复，**不允许 fallback 到 host chrono**。

---

## 三、精度验证方法

详见 [BUILD_AND_TEST_GUIDE.md](BUILD_AND_TEST_GUIDE.md) Section 4.2。

### 3.1 流程

```
GPU (A100)                          NPU (Ascend950PR)
  │                                     │
  ├─ 生成输入数据                        │
  ├─ 运行 CUDA kernel                   │
  ├─ dump 输入 + 输出 (.bin)    ──传输──→ 加载输入
  │                                     ├─ 运行 AscendC kernel
  │                                     ├─ 读回输出
  │                                     └─ 逐元素对比 NPU vs GPU
```

### 3.2 精度阈值

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| fp32 | 1e-4 | 1e-4 | 严格匹配，scatter-add 类允许累加顺序差异 |
| fp16 | 1e-2 | 1e-2 | fp16 精度范围 |
| bf16 | 2e-2 | 2e-2 | bf16 尾数更短，scatter-add 累加顺序差异更大 |

判定逻辑：`|npu[i] - gpu[i]| <= atol + rtol * |gpu[i]|`

### 3.3 已知精度特征

- **确定性算子**（SG forward, SG SIMD forward）：所有 dtype 零不匹配
- **scatter-add 算子**（Pooling fwd/bwd, SG bwd grad_in）：atomicAdd 累加顺序不确定，fp16/bf16 可能有少量不匹配，属预期行为
- **bf16 SIMD**：全程 bf16 累加（bisheng 不支持 bf16↔float Cast），精度低于 SIMT（scalar float 累加）。详见 [Pattern 14](ASCENDC_PATTERN_LIBRARY.md)

---

## 四、结果展示方法

### 4.1 总结表（比值法）

用 GPU/NPU 比值突出差距：

| 指标 | Forward 比值均值 | Backward 比值均值 | Forward 求和比值 | Backward 求和比值 |
|------|:---:|:---:|:---:|:---:|
| Batch N | x.xx | x.xx | x.xx | x.xx |

- **比值均值** (mean-of-ratios)：每个 cluster/case 的 `GPU_time / NPU_time` 取算术平均
- **求和比值** (ratio-of-sums)：`sum(GPU_time) / sum(NPU_time)`
- GPU/NPU < 1.0 = NPU 慢于 GPU，= 1.0 持平，> 1.0 = NPU 快于 GPU。越接近 1.0 越好。

### 4.2 详细数据（后续章节）

Per-cluster / per-case 的具体毫秒数，按 Batch 版本分列。
