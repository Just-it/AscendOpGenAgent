# Domain: Thread Utilization & Scheduling
> Patterns for block/thread count tuning, work distribution, and kernel scheduling.
> Load when: Analyzer detects numBlocks assignment, blockDim tuning, or token_num >> core count.

---

## Patterns

### P-P1: numBlocks 动态获取

**严重度**: 高

```cpp
// Pooling: 用满所有 AIV 核心
constexpr uint32_t MAX_AIV_CORES = 56;  // 28 AICore x 2 AIV

// SG: 每个 block 处理一个 token
uint32_t fwd_blk = token_num * grid_y;
```

**区分**: Pooling 按 stride 共享工作 → 用满核心。SG 每 block 独占输出切片 → numBlocks = 工作项数。

---

### P-P4: 动态 block size

**严重度**: 中

```cpp
__aicore__ inline uint32_t calc_block_size(int dim, int divisor) {
    int raw = std::min(dim / divisor, 1024);
    return ((std::max(raw, 1) + 31) / 32) * 32;  // round to warp
}
```

**注意**: 函数必须标注 `__aicore__`。

---

### P-P20: 线程利用率再平衡（Thread Utilization Analysis）

**严重度**: **高** | **来源**: 专家手工优化 E7-1 (2026-03-27) | **平台**: 全平台 AscendC
**通用策略**: [SKILLS_DESIGN.md §6.2](../../SKILLS_DESIGN.md) — 可静态分析自动发现，不需要 msprof

**核心原则**: 多维工作分解中，每个维度分配的线程数应匹配实际工作量。`utilization = min(actual_work, threads) / threads < 50%` 时需要再平衡。

**具体实例 (Pooling BRE=emb_dim)**: 将 BLOCK_READ_EMB 设为 emb_dim（而非固定 32 或 512），使 iter_emb=1，最大化 index 线程数。

**反模式**（固定 BRE=32）:
```cpp
// dim=9 时: 512/32=16 个 index 线程，23/32 emb 线程浪费
constexpr int BRE = 32;
// dim=1 时: 16 个 index 线程处理 16 条 edge/tile（严重低效）
```

**正确模式**（BRE=emb_dim, 运行时参数）:
```cpp
int BRE = (emb_dim <= 512) ? static_cast<int>(emb_dim) : 512;
int block_read_indices = THREAD_NUM / BRE;
// dim=9 时: 512/9=56 个 index 线程（3.5x 更多 edge/tile）
// dim=1 时: 512/1=512 个 index 线程（32x 更多 edge/tile）
```

**A5 实测效果**（61 clusters, fp32, sorted edges）:

| dim 范围 | BRE=32 Fwd | BRE=dim Fwd | 加速 |
|---------|:---:|:---:|:---:|
| dim=1~9 (clusters 0-6) | 0.64ms | 0.39ms | **1.65x** |
| dim=33 (clusters 13-16) | 16.51ms | 11.38ms | **1.45x** |
| dim>256 (clusters 52+) | 基本持平 | 基本持平 | 1.0x |
| **总计** | 15.43ms | 10.70ms | **1.44x** |

**注意**:
- BRE=emb_dim 要求模板参数改为运行时参数（无法为每个 dim 值实例化模板）
- **替代 P-P11 的 BRE 选择策略**: dim ≤ 512 用 BRE=emb_dim，dim > 512 用 BRE=512
- 与 sorted-edge 寄存器累加兼容：iter_emb=1 时 accum 退化为单标量

**翻译时触发规则（generator 必检）**:
- 从 CUDA 翻译时，凡是看到 `blockDim.x / CONSTANT` 的线程分解，**必须**检查 CONSTANT 在 NPU 上是否合理
- CUDA 用 `BRE=32` 是因为 GPU warp=32 + 4096 blocks。NPU 只有 56 blocks，BRE=32 浪费严重
- **不要照搬 CUDA 的固定常量**——在 generator 阶段就改为 runtime dispatch

#### Runtime-BRE Variant (_rt_vf) 实现模式

当 BRE=emb_dim 时，BRE 值在编译期未知（不同算子调用的 emb_dim 不同）。需要生成独立的 `_rt_vf` 变体:

**命名约定**:
```
gpu_{op}_{dir}_kernel_vf<T, BRE, TI>        — 模板 BRE/TI（编译期确定）
gpu_{op}_{dir}_sorted_kernel_vf<T, BRE, TI> — 排序变体（模板 BRE/TI）
gpu_{op}_{dir}_sorted_rt_vf<T>              — 排序+运行时 BRE/TI
```

**_rt_vf 函数签名** (BRE/TI + 预计算参数全部作为运行时参数):
```cpp
template <typename DATA_TYPE>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void gpu_{op}_{dir}_sorted_rt_vf(
    GM_ADDR ...,
    int BRE, int TI, int block_read_indices,
    int iter_indices_block, int iter_indices_thread, int iter_emb,
    uint32_t block_index, uint32_t total_block_num) {
  // BRE/TI 等参数在 host dispatcher 中预计算，传入 kernel
  // 配合 P-P21 sorted 使 iter_emb=1 时退化为单标量 accum
}
```

**Host dispatcher 预计算** (在 `{op}_launch_config.h` 中):
```cpp
inline void compute_pooling_params(int emb_dim, int thread_num,
    int& BRE, int& TI, int& bri, int& iib, int& iit, int& ie) {
  BRE = (emb_dim <= 512) ? emb_dim : 512;
  bri = thread_num / BRE;  // block_read_indices
  TI = bri;                // tile indices = index threads per block
  iib = (work_items + 56 * bri - 1) / (56 * bri);  // iter per block
  iit = 1;                                           // iter per thread (simplified)
  ie = (emb_dim + BRE - 1) / BRE;                   // iter_emb
}
```

**何时用 template vs runtime**:
- dim 已知且固定（如 SG hidden_dim=256）→ template BRE/TI + `#pragma unroll`
- dim 运行时变化（如 Pooling 61 clusters dim 1~512）→ runtime `_rt_vf`
- **同时生成两种**: template 用于 benchmark/特化，runtime 用于生产调度

---

### P-P22: 常驻核心分发（Persistent Kernel）

**严重度**: 高 | **来源**: 专家 E8-2 (2026-03-28) + A5 实测 | **平台**: Ascend950PR (56 AIV cores)

**反模式**: `numBlocks = token_num`（token 多时大量 block 排队等 56 核时分复用）
```cpp
numBlocks = token_num;  // e.g. 4096 blocks → 56 核排队 73 轮
for (int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x) {
    // 处理 1 个 token
}
```

**正确模式**: 56 个常驻 block，每个循环处理多个 token
```cpp
numBlocks = 56;  // MAX_AIV_CORES
for (uint32_t token = block_index; token < token_num; token += total_block_num) {
    // 处理 1 个 token（同原始逻辑）
}
```

**效果**: 消除 block 调度开销。medium (512 tokens) 3.2x 加速，xlarge (4096 tokens) 1.86x 加速。

**适用条件**:
- token_num >> MAX_AIV_CORES (56)，且每 token 工作量不大（调度开销占比高）
- Forward 类 kernel（无 atomicAdd 写冲突）
- msprof scalar_ratio > 0.2（间接寻址/调度相关代码占比高 → 减少调度有效）
- **不适用** backward（msprof vec_ratio ≈ 1.0, compute-bound，调度开销不是瓶颈）

**触发条件（generator 必检）**:
- 看到 `numBlocks = token_num` 且 token_num 可能 >> 56 → 建议生成 persistent variant
- msprof scalar_ratio > 0.2 → persistent 可能有效
