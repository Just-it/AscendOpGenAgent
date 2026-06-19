# 待 A5 硬件验证的候选 Patterns

> 这些 pattern 概念上合理，但缺少 A5 (Ascend950PR) 上的性能数据。
> 转正条件：在 A5 上完成 benchmark 验证，证明有可测量的性能改善。
>
> **验证实验**: 2026-03-26 进行中

---

## U-P1: `__ldg` 缓存提示用于只读全局内存访问

**来源**: HKV 手写版 | **验证状态**: 🔬 实验中

**概念**:
```cpp
// 普通读取
K val = *(ptr + idx);

// 带缓存提示
K val = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(ptr + idx);
```

**API 来源**: AscendC 编译器内置（`__CCE__` 条件编译），HKV 手写版中广泛使用。

**预期效果**: 对反复扫描同一数据（如哈希桶 key/score）时，L2 缓存提示减少 HBM 访问次数。

**存疑点**:
1. 950PR 上 `__ldg` 是否真的影响缓存行为？（某些平台上可能是 no-op）
2. 对随机访问模式（如 Pooling 的 edge_in 间接索引）是否有效？
3. 缓存提示的开销（如果有）是否超过收益？

**验证计划**:
- [ ] 编译测试：`__ldg` 能否在 950PR + CANN 9.0.T501 上编译通过
- [ ] 性能测试：读密集型 kernel，对比有/无 `__ldg` 的吞吐量
- [ ] 结果记录：附 msprof 数据

**验证结果** (2026-03-26, Ascend950PR + CANN 9.0.T501):

✅ **编译通过**: `__ldg` 和 `__stg` 均可用，bisheng 正常编译。

❌ **顺序读取无性能提升**: 4 种 kernel 变体 (plain/ldg-L2/ldg-L2+L1/ldg+stg) 在 4MB~256MB 数据上性能差异 < 0.5%（噪声范围）。

| 数据大小 | plain GM | `__ldg` (L2, no L1) | Speedup |
|---------|----------|-------------------|---------|
| 4 MB | 0.085ms (49.5 GB/s) | 0.084ms (49.7 GB/s) | 1.003x |
| 256 MB | 11.78ms (22.8 GB/s) | 11.79ms (22.8 GB/s) | 0.999x |

**结论**: 对 stride scan 模式，硬件预取已足够高效，`__ldg` 缓存提示无额外收益。可能仅对反复扫描小块数据（如 HKV 桶内查找）有效。**不建议在 pooling/SG 场景使用。**

---

## U-P2: CUDA 共享内存预取 → `__ldg` 缓存替代

**来源**: HKV 迁移对比 | **验证状态**: ⏳ 依赖 U-P1

**概念**: CUDA 的 `__pipeline_memcpy_async` + `__shared__` 双缓冲 → AscendC 替代方案:
1. `__ldg` 带缓存提示（U-P1）
2. 协作组并行读取（已验证的通用 P-P13）

**存疑点**: `__ldg` 缓存提示能否真正替代异步管线 + 共享内存的 compute/memory overlap？

**验证条件**: U-P1 验证通过后，需要在读密集型 kernel 上对比吞吐量。

## P-CAT-1: VECIN→VECOUT bridge for data-movement kernels (VERIFIED on Cat + Split)
- **Trigger**: Pure copy kernel (no computation)
- **Pattern**: VECIN(GM→UB) → Adds(dst, src, 0.0f) → VECOUT(UB→GM)
- **Evidence**: Cat 51/51 PASS 1.20x (2026-04-09), Split 57/57 PASS (2026-04-09)
- **Status**: **READY TO PROMOTE** — verified on 2 independent ops

## P-CAT-2: Overlapping tail write for non-aligned DMA (VERIFIED on Cat)
- **Trigger**: Strided DataCopy with chunk_size % ALIGN != 0
- **Pattern**: Copy aligned portion + re-copy last ALIGN elements from (chunk-ALIGN)
- **Evidence**: Cat V2 fix, 3 previously failing cases now pass (2026-04-09)
- **Status**: Verified on 1 op, generalizable to any strided copy

## P-CAT-3: N-dim op → flat 3D decomposition (VERIFIED on Cat + Split)
- **Trigger**: Any op that processes along arbitrary dim (cat, split, cumsum, etc.)
- **Pattern**: outer=prod(shape[:d]) × target_dim × inner=prod(shape[d+1:])
- **Evidence**: Cat 51/51 PASS (2026-04-09), Split 57/57 PASS (2026-04-09)
- **Status**: **READY TO PROMOTE** — verified on 2 independent ops

## P-SPLIT-2: Padded allocation for sub-ALIGN compact DMA writes (VERIFIED on Split)
- **Trigger**: Kernel writes to compact output where chunk < DataCopy alignment
- **Pattern**: nblk=1 (serial) + padded alloc + narrow view to exact size
- **Evidence**: Split V2 fix, 4 previously failing cases now pass (2026-04-09)
- **Status**: Verified on 1 op. Generalizable to any compact-output kernel.

## P-REG-1: Reg-based 多步融合 — 中间结果留在寄存器避免 UB 搬运 (待 A5 运行时验证)

**来源**: hiascend.com 官方文档 (CANN 9.0 beta2, Reg矢量计算编程)

**Trigger**: 多步 VEC 计算链（3+ 步），中间结果被写回 UB 后立即读回做下一步计算

**Pattern**:
```cpp
// Mem-based (当前): 每步中间结果经过 UB
Cast(work, xd, CAST_NONE, count);      // UB → VEC → UB
PipeBarrier<PIPE_V>();
Mul(work, work, smooth, count);         // UB → VEC → UB
PipeBarrier<PIPE_V>();
Abs(work, work, count);                 // UB → VEC → UB
PipeBarrier<PIPE_V>();

// Reg-based (目标): 中间结果留在寄存器
__simd_vf__ inline void FusedCastMulAbs(...) {
    Reg::RegTensor<float> reg, smoothReg, absReg;
    Reg::LoadAlign(reg, srcAddr, aReg);           // UB → Reg (1次)
    Reg::LoadAlign(smoothReg, smoothAddr, aReg);  // UB → Reg (1次)
    Reg::Mul(reg, reg, smoothReg, mask);          // Reg 内计算
    Reg::Abs(absReg, reg, mask);                  // Reg 内计算
    Reg::StoreAlign(dstAddr, absReg, aReg, mask); // Reg → UB (1次)
}
// 省去 2 次 UB 搬入搬出（~4 cycles/次 × 2 = ~8 cycles 节省/VL）
```

**适用场景**:
- DynamicQuant: cast→mul_smooth→abs→reduce (4步，可融合为 2 次 UB 访问)
- GELU: x→exp→mul→add→div (5步)
- SwiGLU: exp→add→reciprocal→mul (4步)
- LayerNorm: sub_mean→square→reduce (3步)

**约束**:
- 每次只处理 VL 个元素，需要手动循环（Mem-based 可处理完整 LocalTensor）
- `__simd_vf__` 函数内不能调用 `__aicore__` 函数或 SIMT 函数
- 不支持 GM → Register 直接加载，必须先到 UB

**Status**: A5 编译验证通过 (2026-04-12, CANN 9.0.0 + bisheng). 待运行时性能验证 — 需要在实际 op 上对比 Mem-based vs Reg-based 性能。
