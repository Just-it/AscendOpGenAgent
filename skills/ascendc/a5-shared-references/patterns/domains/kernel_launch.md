# Domain: Kernel Launch & Build Practices
> Patterns for launch bounds, grid configuration, loop unrolling, and benchmark methodology.
> Load when: Analyzer detects kernel launch configuration, pragma unroll, or benchmark setup code.

---

## Patterns

### P-P5: LAUNCH_BOUND + LAUNCH_CHECK

**严重度**: 中

`LAUNCH_BOUND(1024)` >= 所有 dispatcher 可能发射的最大线程数。每次 launch 用 `LAUNCH_CHECK` 检查返回值。

---

### P-P6: grid_y 与 thread count 一致性

**严重度**: 中

host 端 grid_y 计算必须匹配 dispatcher 的实际线程数。两边不一致导致错误的工作分配。

---

### P-P7: #pragma unroll 范围

**严重度**: 低

仅对编译时可推断上界的小循环使用。含 `WarpReduceAddSync + ThreadBarrier` 的循环不应 unroll。

---

### P-P8: Host benchmark 最佳实践

**严重度**: 低

Warmup 3 次 → 计时 10 次取均值 → 精度对比 GPU golden ref → 边界测试 (edges=0, dim=1, dim=3)。

---

### P-P9: SIMD vs SIMT 选择

**严重度**: 高 | **更新**: 2026-04-02 Batch 14 SIMT/SIMD crossover 实验

**核心规则**：**优先 SIMD，除非必须 scatter-write (atomicAdd)**

| 场景 | 选择 | 原因 |
|------|------|------|
| scatter-write (atomicAdd 到随机地址) | **SIMT** | SIMD 的 SetAtomicAdd 需对齐且功能受限 |
| indirect-read + 加权求和 (如 SG Forward) | **SIMD** | DataCopy 块传输 + 4 管线并行 |
| 连续对齐读写 | **SIMD** | 管线重叠天然优势 |
| scatter-read + scatter-write 混合 | **SIMT** | 两端都不规则时 SIMD 无法编排 |

**~~旧规则~~（已废弃）**: ~~"间接索引/随机访问 → SIMT"~~

**废弃原因（msprof 实证）**: SG Forward 有间接寻址（每个 token 读不同 expert），但 expert 行本身是连续内存。SIMD DataCopy 批量搬 expert 行到 UB（走 MTE2），比 SIMT 线程标量散读快 2-7x。SIMT 的 VEC pipe 被 GM 读延迟堵死（vec=0.95+, mte2=0.000），SIMD 4 条 pipe 同时工作（vec+scl+mte2+mte3 各 30-90%）。

**关键洞察**: "间接寻址"要区分寻址层和数据层。SG Forward 寻址层间接（哪个 expert），数据层连续（expert 行连续内存）。SIMD 的 DataCopy 处理数据层，寻址由标量计算。indirect-read ≠ 必须 SIMT。

**SIMT 架构限制**: SIMT 模式下 GM 访问只走 VEC pipe 的 load/store 单元，MTE2/MTE3 DMA 引擎不参与。无法实现 4 管线并行，这是硬件限制非代码问题。

---

### P-P19: Kernel 开发必须配套 UT（边界 dim + sorted/原始一致性）

**严重度**: 高 | **来源**: 专家 B6 反馈（accum[12] 溢出未被测试捕获）

每个新 kernel 或 kernel 变体必须有对应的 CPU 参考测试，覆盖：
1. **大 dim 边界**：dim=512, 1024, 4096（验证 BRE=512 路径 + accum fallback）
2. **排序/非排序一致性**：sorted 输出 vs 原始输出逐元素对比
3. **生产级 stress test**：edges > 10K, dim > 256
4. **边界值**：edges=0, edges=1, dim=1

**反模式**: kernel 只在生产数据上"跑通"就认为正确——大 dim 路径从未被测试。

**正确模式**: 先写 CPU 参考实现 + UT → 在 CPU 模式编译运行 → 通过后再部署到 NPU。
