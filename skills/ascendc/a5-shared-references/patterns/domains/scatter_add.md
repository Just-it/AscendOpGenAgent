# Domain: Scatter-Add Optimization
> Patterns for kernels with atomicAdd scatter-write patterns.
> Load when: Analyzer detects atomicAdd in loop with indirect write target.

---

## Patterns

### P-P2: WarpReduceAddSync + warp-lane-0 原子操作

**严重度**: 高 (7x) | **来源**: Codex

```cpp
// 反模式: 每线程 atomicAdd → 512 次/block
// 正确: warp 规约后 lane-0 atomicAdd → 16 次/block
float warp_sum = Simt::WarpReduceAddSync(partial_sum);
Simt::ThreadBarrier();
if (threadIdx.x % 32 == 0 && warp_sum != 0.0f)
    atomicAdd(dst, warp_sum);
```

**前提**: dst 必须 host 端预清零。实测 SG backward: 0.865ms → 0.121ms (7.1x)。

---

### P-P10: Block 超订（Oversubscription）

**严重度**: **高** | **平台**: A5 验证 | **适用**: scatter-add 类算子

nblk > 物理 AIV 核心数（56）→ 分散 atomicAdd 竞争。

**适用条件（关键）**: 仅当 kernel 有 atomicAdd 竞争时有效。
- **未排序 backward** (有 atomicAdd 竞争): nblk=224 → bwd 2.0x 加速 ✅
- **已排序 backward** (寄存器累加，无 atomicAdd): nblk=56 严格最优。超订 monotonically 变差: 112(+3.8%), 224(+9.8%), 448(+20.8%) ❌
- **Forward**: nblk=56 始终最优（pipe-bound，超订无收益）

**根因**: 排序+寄存器累加消除了 atomicAdd 竞争 → 超订分散竞争的收益消失 → 只剩寄存器压力代价。多个 block 竞争同一 AIV core 的寄存器文件，累加器变量被 spill 到 HBM。

**E9-2 实测确认** (2026-03-30, 61 clusters, NPU 空闲): 见 E10 exploration results。

---

### P-P17: 前缀和 + Block 级 atomicAdd 聚合

**严重度**: 高 | **来源**: HKV 手写版，已验证 | **适用**: scatter-add 类算子

三级聚合减少全局原子操作:
1. **组内前缀和** (`__shfl_up`): 每线程得到 local_offset
2. **组 leader → UB atomicAdd** (`__ubuf__`): 每组 1 次 (512/32=16 次)
3. **Block leader → 全局 atomicAdd**: 整个 block 1 次

512 次全局 atomicAdd → 1 次。**直接适用于 Pooling backward 的 atomicAdd 优化。**

---

### P-P21: Scatter-add 排序+寄存器累加（减少 atomicAdd 次数）

**严重度**: **高** | **来源**: AI 自研 + msprof 驱动 (Batch 6, 2026-03-26)
**适用**: 任何 scatter-add kernel（多个输入写同一输出地址）

**问题**: scatter-add 中每条 edge 都做一次 atomicAdd。当多条 edge 指向同一目标时（高 fan-in），atomicAdd 串行排队成为瓶颈。msprof 特征：`aiv_vec_ratio=1.0` 但 HBM 带宽利用率 < 1%（pipe 被 atomicAdd 堵死）。

**优化思路**: 预排序 edges 使同一目标的 writes 连续 → 寄存器累加 → 一次 atomicAdd。

**反模式**（每条 edge 一次 atomicAdd）:
```cpp
for (int i = 0; i < edge_length; i++) {
    atomicAdd(&output[edge_out[i] * dim + d], input[edge_in[i] * dim + d]);
    // 10万条 edge 指向同一 target → 10万次 atomicAdd 排队
}
```

**正确模式**（排序后寄存器累加）:
```cpp
// 前提: edges 已按 edge_out 排序
float accum = 0;
int prev_target = -1;
for (int i = 0; i < edge_length; i++) {
    int target = edge_out[i];
    if (target != prev_target) {
        if (prev_target >= 0) atomicAdd(&output[prev_target * dim + d], accum);  // 只在目标变化时写
        accum = input[edge_in[i] * dim + d];
        prev_target = target;
    } else {
        accum += input[edge_in[i] * dim + d];  // 寄存器累加，无 atomicAdd
    }
}
// flush final
if (prev_target >= 0) atomicAdd(&output[prev_target * dim + d], accum);
```

**效果**: atomicAdd 次数从 `edge_length` 降至 `unique_targets`。平均 fan-in=100 时减少 100x。

**A5 实测**: Pooling backward 100.53ms → 14.53ms（**-86%**），forward 15.66ms → 11.29ms（**-28%**）。Bwd 改善更大因 atomicAdd 竞争更严重（scatter-write vs scatter-read）。Fwd 改善因排序后 register accum 减少 atomicAdd 次数（从 edge_length 次降至 unique_targets 次）。

**排序开销**: host `std::sort` 1147ms, NPU counting sort 405ms。排序是一次性预处理（图结构不变时不需要重排）。

**触发条件（generator 必检）**:
- 看到 `atomicAdd` 在循环内 → 检查是否是 scatter-add 模式（多输入写同一输出）
- msprof 显示 `HBM_util < 1%` 但 `aiv_vec_ratio = 1.0` → atomicAdd 瓶颈确认
- **生成阶段就应提示"此 kernel 有 scatter-add，建议提供 sorted variant"**，不要等到优化阶段

**注意**:
- ~~排序只对 backward 有效~~ **更正 (E9-1 实测 61 clusters)**: forward 同样受益。D_Fwd（sorted+BRE=emb_dim+register accum）比 B_Fwd（unsorted）总计快 **1.39x**（15.66ms→11.29ms）。小 dim（≤33）时快 **1.45x**，dim=1+高 fan-in 时快 **7x**（cluster 5: 0.507→0.072ms）。但 dim>128 时 D 反而变慢（BRE=emb_dim 导致 index_threads 太少），此时应 fallback 到 C（template sorted, BRE=32）。
- 需要配合 P-P20（BRE=emb_dim）使 iter_emb=1，使单标量 accum 足够
- 精度：float 寄存器累加的顺序与逐次 atomicAdd 不同，但差异在 atol 范围内（61/61 PASS）
- **dim 分界点**: dim ≤ ~128 用 D variant（BRE=emb_dim）; dim > 128 用 C variant（BRE=32, template sorted）

#### Multi-Dim Accumulator Array (iter_emb > 1)

When BRE < emb_dim (large dims where BRE=emb_dim is not feasible), `iter_emb > 1` and a single scalar `accum` is insufficient. Use an accumulator array:

```cpp
constexpr int MAX_ACCUM = 16;  // max dim positions per thread
float accum[MAX_ACCUM] = {0};
int prev_target = -1;
int my_emb_lane = threadIdx.x % BRE;  // cache once

// Loop nesting: INDEX OUTER, EMBEDDING INNER (CRITICAL — never invert!)
for (int i = iter_start; i < iter_end; i++) {
    int target = edge_out[sorted_idx[i]];
    if (target != prev_target) {
        // Flush all accumulators for previous target
        if (prev_target >= 0) {
            for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
                if (accum[j] != 0.0f)
                    atomicAdd(&output[prev_target * dim + my_emb_lane + j * BRE], accum[j]);
                accum[j] = 0.0f;
            }
        }
        prev_target = target;
    }
    // Accumulate ALL embedding positions for this edge
    for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
        int64_t src_offset = static_cast<int64_t>(edge_in[sorted_idx[i]]) * dim + my_emb_lane + j * BRE;
        accum[j] += simt_to_float_generic<DATA_TYPE>(input[src_offset]);
    }
}
// Final flush
for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
    if (prev_target >= 0 && accum[j] != 0.0f)
        atomicAdd(&output[prev_target * dim + my_emb_lane + j * BRE], accum[j]);
}

// FALLBACK: when iter_emb > MAX_ACCUM, fall back to per-element atomicAdd
if (iter_emb > MAX_ACCUM) {
    // Use baseline atomicAdd path for the overflow portion
}
```

**CRITICAL loop nesting rule**: Index MUST be the outer loop, embedding inner. Inverting causes:
1. `prev_target` state reset between embedding iterations → redundant atomicAdd
2. iter_emb × redundant GM reads on edge_in/edge_out arrays
3. Defeats the entire purpose of sorted accumulation for multi-dim

---

### P-P32: Sorted-Edge First-Occurrence Dedup (atomicCAS-Free)

**严重度**: 高 | **来源**: E10-1 手工优化 (2026-03-30) | **适用**: 去重/首次出现检测

**问题**: `generate_assign_edges` 类 kernel 使用 `atomicCAS` 检测首次出现——每条 edge 一次原子操作。当 edge 数量大时，atomicCAS 串行成为瓶颈。

**前提**: edges 已按目标排序（配合 P-P21 的排序预处理）。

**反模式** (atomicCAS per edge):
```cpp
// 每条 edge 一次 atomicCAS，检查是否是第一个写入该目标的
int old = Simt::AtomicCas(&assign_edges[target], INVALID, source);
if (old == INVALID) {
    // 首次出现
}
```

**正确模式** (相邻比较):
```cpp
// 前提: edges 已按 edge_out 排序
// 比较相邻 edge 的目标——目标变化时即为"首次出现"
int prev_target = (tid > 0) ? edge_out[tid - 1] : -1;
int cur_target = edge_out[tid];
if (cur_target != prev_target) {
    // 首次出现该目标——无 atomic 操作！
    assign_edges[cur_target] = edge_in[tid];
}
```

**优势**:
- 零原子操作（pure load/store，无 CAS 竞争）
- O(1) per element（vs atomicCAS 的 O(contention)）
- 适配块分界: 第一个线程需要与前一个块的最后元素比较（跨块边界）

**实测**: Pooling assign_edges: 259ms → 10ms (**25.6x**), 配合整体 sorted pipeline

**触发条件**: 任何 first-occurrence / dedup 操作，当输入已排序或可以预排序时
