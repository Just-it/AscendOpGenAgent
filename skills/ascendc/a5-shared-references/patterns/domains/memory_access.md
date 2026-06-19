# Domain: Memory Access Optimization
> Patterns for vectorized loads, adaptive tiling, and inner-loop variable sizing.
> Load when: Analyzer detects strided memory access, tile size selection, or loop variable types.

---

## Patterns

### P-P3: vec4 向量化路径启用条件

**严重度**: 中

**反模式**: `if (hidden_dim % 4 == 0 && grid_y > 1)` — grid_y > 1 是多余门控。

**正确**: `if (hidden_dim % 4 == 0)` — grid_y==1 时 block_y=0，完全正确。

---

### P-P11: 自适应 Tile 大小

**严重度**: **高**

```
dim ≤ 256:  <BRE=32,  TI=16>    16 edges × 32 emb threads
dim ≤ 512:  <BRE=64,  TI=8>     8 edges × 64 emb threads
dim > 512:  <BRE=512, TI=1>     1 edge × 512 emb threads
```

NPU 56 blocks × TI=16 = 896 edges/step → TI=1024 → 57344 edges/step (接近 GPU 的 65536)。

---

### P-P12: 内层循环 int32 替代 int64

**严重度**: 中

值域在 int32 范围内的变量用 int（循环计数器 j、thread_idx_emb 等）。`edge_in[i] * emb_dim` 必须保留 int64。效果: fwd -3%, bwd -2%。

---

### P-P23: 连续块分配 vs Grid-Stride（相邻元素比较场景）

**严重度**: **高** | **来源**: E10-1 assign_edges 优化 (2026-03-30) | **平台**: Ascend950PR

**问题**: 需要比较相邻元素（`arr[i] vs arr[i-1]`）时，grid-stride 循环导致每次读取 `arr[i-1]` 都是 cache miss（间隔 `total_threads` 个元素，通常 28672）。

**反模式**（grid-stride，cache miss）:
```cpp
for (int64_t i = tid; i < n; i += total_threads) {
    if (i == 0 || arr[i] != arr[i - 1]) { ... }  // arr[i-1] 在 HBM 中距 arr[i] 28672 个元素
}
```

**正确模式**（连续块，cache 友好）:
```cpp
int64_t chunk = (n + total_threads - 1) / total_threads;
int64_t start = tid * chunk;
int64_t end = min(start + chunk, n);
for (int64_t i = start; i < end; i++) {
    if (i == 0 || arr[i] != arr[i - 1]) { ... }  // arr[i-1] 紧邻 arr[i]
}
```

**实测**: assign_edges sorted scan 从 123ms → 10ms (**12x**)。总优化（含避免 atomicCAS）: 259ms → 10ms (**25.6x**)。

**触发条件**: 看到循环内 `arr[i-1]` 或 `arr[i+1]` 的相邻访问 + grid-stride 循环 → 改为连续块。

**注意**: 连续块分配不适用于所有场景。如果每次迭代的数据完全独立（无相邻依赖），grid-stride 的 coalesced 访问反而更优。只有**需要相邻元素比较/依赖**时才用连续块。

---

### P-P24: Sort-to-Reuse — 消除间接寻址 GM 读放大

**严重度**: **CRITICAL** | **来源**: E11 msprof 分析 (2026-03-31) | **平台**: Ascend950PR (无 L2 cache)

**问题**: 多个 work item 通过间接索引读同一 GM 数据，每次都是独立 HBM 访问。

```
// 反模式: per-token 遍历，input[expert] 被 N 个 token 重复读 N 次
for token in all_tokens:
    expert = index[token]
    for d in hidden_dim:
        val = input[expert * hdim + d]  // ← 128 个 token 共享同一 expert, 读 128 次!
```

**根因**: Ascend NPU SIMT 模式无 L2 cache。GPU (A100 40MB L2) 自动缓存热数据，NPU 每次标量 GM 读都走 HBM。当 N 个 work item 通过间接索引 `arr[index[i]]` 读同一行时，实际 HBM 读量 = N × row_size（而非 1 × row_size）。

**msprof 验证** (E11, SG backward xlarge):
- `vec_ratio = 0.99` **不等于** 99% atomicAdd
- 去掉 atomicAdd 后仅省 4.4%（158us / 3605us）
- **95%+ 的 VEC 时间是标量 GM 随机读**（SIMT 标量读走 VEC pipe）
- `msprof vec_ratio` 在 SIMT 模式下 = GM读 + 计算 + atomicAdd 的总和，不能分拆

**正确模式**: Sort-to-Reuse — 按间接索引排序，共享数据只加载一次

```
// Step 1: counting sort edges by expert_index → sorted_edges[], expert_offsets[]
// Step 2: per-expert 处理
for expert in all_experts:
    DataCopy(local_buf, input[expert * hdim], hdim)  // ← 只读 1 次, 不是 128 次
    for edge in expert_run:
        token = sorted_edges[edge]
        // 用 local_buf 计算, 无需再读 input[expert]
```

**效果 (实测)**:
| | GM 读量 | 耗时 |
|---|:-:|:-:|
| per-token SIMT | 32K × 4096 = **134M** 次 | 3447us |
| per-expert SIMD sorted | 256 × 4096 = **1M** 次 | 265us |
| **读量减少** | **128x** | **13x 加速** |

**触发条件**: 看到 `arr[index[i]]` 形式的间接 GM 读 + 多 work item 共享同一 index 值 → 考虑 sort-to-reuse。

**适用场景**:
- MoE scatter-gather: `input[expert_index]` 被 N tokens 共享
- GNN message passing: `features[neighbor_id]` 被多条边共享
- Embedding lookup: `embedding[token_id]` 被多个位置共享
- 任何有 fan-out 的间接寻址模式

**不适用**:
- index 完全唯一（无共享）→ 排序无复用收益
- GPU 有 L2 cache → 硬件自动缓存，排序收益小
- 数据量 < UB 容量 → 一次性加载，无需排序

**与 P-P21 的关系**: P-P21 (sorted-edge accumulation) 关注消除 atomicAdd 写冲突。P-P24 关注消除 GM 读放大。两者通常一起出现（同一排序同时解决读和写），但 P-P24 收益远大于 P-P21 (95% vs 4.4% for SG backward)。

**与 msprof 解读的关系**: SIMT 模式 `vec_ratio` 高不一定是 atomicAdd 瓶颈。必须对比"有 atomicAdd"和"无 atomicAdd"两个 kernel 的耗时差才能确认。如果差值 <10%，真正瓶颈是 GM 读。

---

### P-P25: SetAtomicAdd + DataCopyPad — SIMD 模式的硬件 atomic 写

**严重度**: **CRITICAL** | **来源**: E12 专家 SIMD backward (2026-03-31) | **平台**: Ascend950PR

**问题**: SIMD 模式下 scatter-add (如 `grad_in[expert] += weight * grad_out`) 需要 atomic 写，但 SIMT 的 `atomicAdd` 走 VEC pipe CAS 循环（慢），`SetValue` 在 AIV 模式不可靠 (OL-19)。

**正确模式**: `SetAtomicAdd<T>()` + `DataCopyPad` — 走 MTE3 pipe 的硬件 atomic

```cpp
// SIMD backward: 每个 token 的 grad_in 贡献写回到 expert 位置
Muls(gradInLocal, gradOutLocal, expertWeight, hdim);
// EnQue + DeQue for pipeline sync
gradInOutQue_.EnQue(gradInLocal);
LocalTensor<float> gradInOut = gradInOutQue_.DeQue<float>();
// MTE3 atomic add: 硬件保证原子性，不走 VEC CAS
SetAtomicAdd<float>();
DataCopyPad(gradInGm_[expertIdx * hdim], gradInOut, copyParams);
SetAtomicNone();
gradInOutQue_.FreeTensor(gradInOut);
```

**对比**:
| 方法 | 管线 | 机制 | 速度 |
|------|------|------|------|
| SIMT `atomicAdd(ptr, val)` | VEC | CAS 循环 | 慢 (竞争序列化) |
| SIMD `SetAtomicAdd` + `DataCopyPad` | **MTE3** | 硬件 atomic DMA | **快** (批量原子加) |
| SIMD `SetValue` (GM) | 标量 | — | ❌ 不可靠 (OL-19) |

**关键优势**: DataCopyPad 一次传输整个 hdim 向量的 atomic add，不是逐元素 CAS。
**不需要排序**: per-token SIMD + SetAtomicAdd 直接写回，无需 counting sort 预处理。

---

### P-P26: SetFlag/WaitFlag 精细事件同步（替代 PipeBarrier<PIPE_ALL>）

**严重度**: 高 | **来源**: E12 专家代码 (2026-03-31) | **平台**: Ascend950PR

**问题**: `PipeBarrier<PIPE_ALL>()` 阻塞所有管线，阻止 MTE2/VEC/MTE3 并行。

**正确模式**: 用 `SetFlag`/`WaitFlag` + `HardEvent` 精确指定管线间依赖

```cpp
// MTE2→Scalar: DataCopy 完成后才能 GetValue 读 UB
event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
SetFlag<HardEvent::MTE2_S>(id);
WaitFlag<HardEvent::MTE2_S>(id);

// VEC→Scalar: ReduceSum 完成后才能 GetValue 读结果
event_t id2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(id2);
WaitFlag<HardEvent::V_S>(id2);

// Scalar→VEC: SetValue 完成后才能 Muls
event_t id3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(id3);
WaitFlag<HardEvent::S_V>(id3);

// Scalar→MTE3: SetValue 完成后才能 DataCopyPad 写回
event_t id4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
SetFlag<HardEvent::S_MTE3>(id4);
WaitFlag<HardEvent::S_MTE3>(id4);
```

**与 TQue 的配合**: TQue<VECIN,2> 的 AllocTensor/EnQue/DeQue/FreeTensor 自动管理双缓冲。SetFlag/WaitFlag 用于 TQue 之外的标量↔向量同步。

**注意**: OL-4 (TQue 数据损坏) 可能是特定 buffer 大小/配置问题。专家代码在 E12 中用 TQue<VECIN,2> + SetFlag 正常工作。

---

### P-P28: TQue 管线自动重叠（替代 PipeBarrier + 手动 Ping-Pong）

**严重度**: **CRITICAL** | **来源**: 专家 E13 (2026-04-01) | **效果**: SG fwd **1.6-2.3x** over PipeBarrier

**场景**: SIMD kernel 循环处理多个数据块（如 top_k 个 expert），每轮需要 DataCopy(MTE2) 读 + Muls/Add(VEC) 算。

**反模式 1** (PipeBarrier<PIPE_ALL> 串行 — 旧方案，已淘汰):
```cpp
// ❌ PipeBarrier<PIPE_ALL> 同步所有管线，MTE2/VEC 无法并行
for (int k = 0; k < top_k; k++) {
  DataCopy(buf, inGm_[expert[k] * hdim], hdim);  // MTE2
  PipeBarrier<PIPE_ALL>();                         // 等所有管线 → 串行!
  Muls(tmp, buf, w[k], hdim);                     // VEC
  Add(accum, accum, tmp, hdim);                    // VEC
  PipeBarrier<PIPE_ALL>();                         // 又等所有管线 → 串行!
}
```

**正确模式** (Ping-Pong 流水):
```cpp
// 两个独立 TBuf: ping 和 pong
DataCopy(ping, inGm_[expert[0] * hdim], hdim);   // prolog: 加载第一个
PipeBarrier<PIPE_ALL>();

for (int k = 0; k < top_k - 1; k++) {
  int cur = k % 2, nxt = 1 - cur;
  // MTE2: 预加载下一个到另一个 buffer（与 VEC 并行）
  DataCopy(nxt==0 ? ping : pong, inGm_[expert[k+1] * hdim], hdim);
  // VEC: 计算当前 buffer（与 MTE2 并行）
  Cast(expertF, cur==0 ? ping : pong, RoundMode::CAST_NONE, hdim);
  Muls(tmp, expertF, w[k], hdim);
  Add(accum, accum, tmp, hdim);
  PipeBarrier<PIPE_ALL>();  // 同步: 下一轮需要两个 buffer 都就绪
}
// epilog: 处理最后一个
```

**反模式 2** (手动 Ping-Pong + PipeBarrier — 旧方案 E10-3):
```cpp
// ⚠️ 比反模式1好，但 PipeBarrier<PIPE_ALL> 仍然同步所有管线
DataCopy(ping, inGm_[expert[0]], hdim);
PipeBarrier<PIPE_ALL>();
for (int k = 0; k < top_k - 1; k++) {
  DataCopy(pong, inGm_[expert[k+1]], hdim);  // MTE2: 加载下一个
  Muls(tmp, ping, w[k], hdim);               // VEC: 计算当前
  Add(accum, accum, tmp, hdim);
  PipeBarrier<PIPE_ALL>();                    // 等全部管线 — 包括不需要等的
  swap(ping, pong);
}
```

**正确模式** (TQue<VECIN,4> 自动管线重叠 — E13):
```cpp
// ✅ TQue 的 EnQue/DeQue 只同步 MTE2→VEC，不阻塞其他管线
// depth=4 允许 MTE2 提前 prefetch，VEC 从不等待
pipe_.InitBuffer(xQueue_, 4, bufBytes);  // depth 4
pipe_.InitBuffer(yQueue_, 2, bufBytes);  // output depth 2

LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
Duplicate(yLocal, 0.0f, hdim);
for (int k = 0; k < top_k; k++) {
  LocalTensor<T> x = xQueue_.AllocTensor<T>();
  DataCopy(x, inGm_[expert[k] * hdim], hdim);   // MTE2
  xQueue_.EnQue(x);                              // MTE2 完成时自动入队
  LocalTensor<T> xComp = xQueue_.DeQue<T>();     // 等 MTE2 完成（仅此管线）
  Muls(xComp, xComp, w[k], hdim);               // VEC（与下一轮 MTE2 并行）
  Add(yLocal, yLocal, xComp, hdim);
  xQueue_.FreeTensor(xComp);
}
yQueue_.EnQue(yLocal);
LocalTensor<T> yOut = yQueue_.DeQue<T>();
DataCopy(outGm_[dst], yOut, hdim);               // MTE3
yQueue_.FreeTensor(yOut);
```

**关键差异**: TQue 的 EnQue/DeQue 只在 MTE2→VEC 之间同步。PipeBarrier<PIPE_ALL> 同步 MTE2+VEC+MTE3+Scalar 全部管线。当 depth=4 时 MTE2 可以提前加载 3 个 buffer，VEC 从不空等。

**实测效果 (SG forward, 2026-04-01)**:
- PipeBarrier → TQue: **1.6-2.3x** 加速（6 cases）
- xlarge GPU/NPU: 0.73 → **1.18**（NPU 比 GPU 快 18%）
- OL-4 TQue bug 已解决（CANN 9.0.0，backward 早已用 TQue 验证）

**适用条件**:
- SIMD kernel（有 DataCopy + VEC 计算循环）
- 循环迭代 >= 2
- **强烈推荐 TQue 方案**，只在 TQue 有已知 bug 时才用 PipeBarrier fallback
- **accum 必须也在 TQue 管理下** — E14 实测 TBuf accum + TQue input 精度 FAIL (max_diff=0.76)。根因：TBuf **没有自动同步**（官方文档确认），VEC 写 accum(TBuf) 和 MTE2 写 input(TQue) 之间无 sync → UB bus contention。**修复方案**：accum 移入 TQue<VECOUT>（Pattern B in ASCENDC_LANGUAGE_REFERENCE.md），与 forward yQueue_ 模式一致
- 详细同步语义参考: `src/skills/references/ASCENDC_LANGUAGE_REFERENCE.md` §2-3

**与 P-P22 (Persistent) 组合**: TQue 在内循环重叠 MTE2/VEC，Persistent 在外循环减少调度开销。两者正交可叠加。

---

### P-P29: Batch Preload Cache — 消除标量 GM 读瓶颈

**严重度**: CRITICAL | **来源**: 专家 E12 (2026-03-31) | **效果**: scalar pipe 42% → 预估 <10%

**场景**: SIMD kernel 循环中需要读取少量标量数据（index/weight），每次 `GetValue()` 从 GM 读取 ~100 cycle。

**msprof 证据**: scalar=42%（SG backward），大部分是 GetValue 的 GM 标量读。

**反模式** (逐元素 GM 读):
```cpp
for (int i = 0; i < actual_k; i++) {
  local_index[i] = indexGm_.GetValue(index_base + i);   // ~100 cycle per read
  local_weight[i] = weightGm_.GetValue(index_base + i); // ~100 cycle per read
}
```

**正确模式** (批量预加载到 UB 缓存):
```cpp
// Init: 分配缓存 buffer
static constexpr uint32_t CACHE_SIZE = 1024;
pipe_.InitBuffer(idxCacheBuf_, CACHE_SIZE * sizeof(int32_t));
pipe_.InitBuffer(wtCacheBuf_, CACHE_SIZE * sizeof(float));

// 缓存访问器: cache miss 时批量加载
__aicore__ inline int32_t GetIndexCached(int64_t idx, int64_t endIdx) {
  if (idx >= idxCacheBase_ + idxCacheLen_) {
    uint32_t copyLen = min(endIdx - idx, CACHE_SIZE);
    DataCopyPad(cache, indexGm_[idx], {1, copyLen * sizeof(int32_t), 0, 0}, padNone);
    idxCacheBase_ = idx; idxCacheLen_ = copyLen;
    // MTE2→Scalar 同步: 等 DataCopyPad 完成后才能 GetValue
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(ev);
    WaitFlag<HardEvent::MTE2_S>(ev);
  }
  return cache.GetValue(idx - idxCacheBase_);  // ~1 cycle from UB
}
```

**原理**: 1 次 GM DMA 加载 1024 元素 (amortized ~0.1 cycle/element) 替代 1024 次标量 GM 读 (~100 cycle/each)。命中率取决于循环中连续访问的 index 是否落在同一缓存块。

**适用条件**:
- 循环中连续读 GM 标量（index, weight, offset 等）
- 总访问量 > 缓存大小时分块加载，访问必须递增
- 非递增访问（如 indirect indexing）不适用此 pattern

**与 P-P28 (Ping-Pong) 组合**: P-P29 消除标量读瓶颈 (scalar pipe)，P-P28 重叠 MTE2/VEC。两者正交。

**缓存大小选择**: 1024 是专家经验值。过大浪费 UB 空间，过小增加 miss 频率。可根据 UB 剩余空间和 top_k 调整。

---

### P-P33: SIMT→SIMD 转换（memory-bound elementwise kernel）

**严重度**: **HIGH** | **来源**: MXFP4 迁移 (2026-04-07) | **状态**: 候选（待 SIMD 实现验证）

**触发条件**: msprof 显示 MTE2=0% AND 吞吐 < 理论带宽 50%

**场景**: SIMT kernel 处理 elementwise/per-group 操作，所有 GM 访问走 dcache（VEC pipe），MTE2 DMA 引擎完全空闲。

**诊断**:
```
msprof 数据:
  aiv_vec_ratio: 高 (>70% 大 tensor)
  aiv_mte2_ratio: ~0%
  aiv_mte3_ratio: ~0%
  吞吐: 实际 125 GB/s vs 理论 400 GB/s (31%)
```

**原因**: SIMT 模式下 GM 读写走 dcache（128B cacheline），不走 MTE2 DMA。VEC pipe 同时承载计算和内存访问，无法并行。

**优化**: 转为 SIMD 模式或混合模式：
```
方案 A (纯 SIMD): DataCopy(MTE2) → VEC compute → DataCopy(MTE3)
  - 适用于计算可用 SIMD 向量指令表达的场景
  - TQue<VECIN,4> + TQue<VECOUT,2> 自动管线重叠

方案 B (混合 SIMT+SIMD):
  - SIMD DataCopy 批量加载到 UB
  - GetPhyAddr() 获取 UB 物理地址
  - SIMT VF_CALL 做不规则计算（如位操作）
  - SIMD DataCopy 写回 GM
  - 适用于计算包含 SIMD 不支持的操作（如 reinterpret float↔int）
```

**预期效果**: 大 tensor 2-3x 吞吐提升（MTE2+VEC 双管线并行）

**约束**:
- 需要 MXFP4 特定分析：PyTorch 版算法用 float 数学（log2, floor, pow2），可能完全用 SIMD 表达
- CUDA 版算法用位操作（reinterpret cast, bit shift），必须用混合模式

**重要限制 (2026-04-07 验证)**:
SIMD 版本在 MXFP4 上比 SIMT **慢 4-20x**。原因: MXFP4 quantization 需要 per-element 的 x_exp 和 shift amount，无法用 SIMD 向量指令表达（每个元素不同的 shift）。SIMD 退化为逐元素 GetValue/SetValue 标量操作，比 SIMT 128 线程并行慢得多。

**P-P33 适用条件更新**:
- ✅ 适用: 计算完全可用 SIMD 向量指令表达（Add, Muls, Cast — 所有元素同一操作）
- ✅ 适用: SG forward/backward（连续 DataCopy + Muls + Add 全是向量操作）
- ❌ 不适用: per-element 异构计算（如 MXFP4 的 per-element log2/pow2/shift）
- ❌ 不适用: 需要 per-element 条件分支的量化/位操作

**判断标准**: 看内循环是否每个元素执行**完全相同的指令序列**（同一个 Muls/Add/Cast）。如果每个元素需要不同的操作（不同 shift amount, 不同 branch），SIMT 更优。

**SIMD V4 "fast" 实验 (2026-04-07)**:
tile-wide shared exponent（不做 per-group 循环）在小 tensor 上 **比 SIMT 快 1.08x**，证明 SIMD 本身不慢。
但 **精度有问题**: 使用 tile-level exponent 代替 per-32-group exponent 导致量化精度下降（不符合 MXFP4 spec，无法作为 production 使用）。

**完整对比 (同 NPU A/B)**:
| 版本 | 4K(ms) | 4M(ms) | 精度 | 可否 production |
|:---:|:---:|:---:|:---:|:---:|
| SIMT (128 threads) | 0.018 | **0.253** | ✅ PyTorch exact | ✅ **production** |
| SIMD V3 (per-group vectorized) | 0.029 | 1.724 | ✅ PyTorch exact | ❌ 比 SIMT 慢 |
| SIMD V4 fast (tile-wide) | **0.017** | 0.813 | ⚠️ **精度降级** | ❌ 不符 spec |

**⚠️ 精度警告**: SIMD V4 fast 用 tile-wide shared exponent 替代 per-32-group exponent。
这意味着 1024 元素共享一个 exponent，而 MXFP4 spec 要求每 32 元素一个。
当 tile 内数值差异大时（如部分元素接近 0，部分很大），小值会被 underflow 到 0。
**A3 手写 SIMD 实现有同样的问题——这就是其精度 bug 的根本原因。**

**P-P33 最终结论**:
1. SIMD 性能瓶颈不是 SIMD 模式本身，而是 **per-group 串行循环**
2. 消除 per-group 循环（tile-wide 处理）SIMD 可以比 SIMT 快
3. 但消除 per-group 循环 = 放弃 per-group precision = **不符合 spec**
4. 对于 group-local 量化算子，**SIMT 是唯一同时满足精度和性能的方案**
5. SIMD 适用于 group_size >= tile_size 或不需要 per-group precision 的场景

**证据**: MXFP4 全链路验证 (2026-04-07): msprof + SIMD V1/V2/V3/V4 A/B + PyTorch spec 对比
