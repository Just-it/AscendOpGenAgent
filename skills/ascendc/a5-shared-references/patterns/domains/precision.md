# Domain: Precision & Functional Correctness
> Patterns for dtype handling, bf16/fp16 precision, type conversion, and functional correctness.
> Load when: Analyzer detects bf16/fp16 dtype, type casting, or dtype dispatch logic.

---

## Patterns

### F-P1: bf16 精度处理（scatter-add waiver）

**严重度**: 中 | **模式**: SIMT/SIMD

**反模式**: bf16 测试用 fp32 的 atol/rtol → 大量误报 FAIL

**正确模式**:
```cpp
float atol = (dtype == "bf16") ? 2e-2f : 1e-4f;
bool waiver = (dtype == "bf16");
compare_data(npu, gpu, n, dtype_str, atol, rtol, "fwd", waiver);
```

**注意**: 仅 scatter-add 类算子（Pooling fwd/bwd）的 bf16 不匹配是预期行为。**SG forward 是确定性计算，bf16 不应有任何不匹配**——如果有就是 bug。

---

### F-P2: 多 dtype 支持架构

**严重度**: 低 | **模式**: SIMT/SIMD

模板化 kernel + 按类型分发 dispatcher:
```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...) { ... }

extern "C" __global__ __aicore__ void kernel_fp32(...) {
    Simt::VF_CALL<kernel_vf<float>>(Simt::Dim3{threads}, ...);
}
// 同理 fp16, bf16
```

---

### F-P3: SIMD bf16 混合精度（MicroAPI）

**严重度**: 中 | **模式**: SIMD | **平台**: A5 (bisheng 15.0.5)

高级 `Cast()` 不支持 bf16↔float。使用 MicroAPI 寄存器级 Cast:
```cpp
__VEC_SCOPE__ {
    RegTensor<bfloat16_t> vreg_bf16;
    RegTensor<float> vreg_f32;
    MaskReg preg;
    AscendC::MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vreg_bf16, ub_addr);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTrait>(vreg_f32, vreg_bf16, preg);
    // ... float 计算 ...
}
```

**简化方案**: 精度允许时直接 bf16 累加（`Muls` + `Add` 原生支持 bf16）。

---

### F-P5: Warp 对齐循环边界（n_align_warp）

**严重度**: 高 | **模式**: SIMT | **来源**: HKV 手写版，已验证

**这是协作组编程的强制正确性要求，不是可选优化。**

当循环体内包含 `__shfl` / `__shfl_xor` / `ThreadBarrier`，循环边界必须对齐到组大小:
```cpp
uint64_t n_align = ((n + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE;
for (uint64_t idx = ...; idx < n_align; idx += stride) {
    if (idx < n) { /* 正常处理 */ }
    else { result = ILLEGAL; }  // 越界线程标记无效但仍参与 __shfl
}
```

**根因**: `__shfl` 要求组内所有 lane 同时执行同一指令。部分 lane 退出循环 → 死锁。

---

## Anti-Patterns

### F-AP1: dtype 字符串匹配子串陷阱

**严重度**: **严重**

```cpp
// "bfloat16".find("float16") == 1 → 匹配成功！
if (dtype.find("float16") != npos) { ... }      // BUG
else if (dtype.find("bfloat16") != npos) { ... } // 永远不执行
```

**修复**: `bfloat16` 检查必须在 `float16` 之前。

---

### F-AP2: `__threadfence` 误用为延迟等待

**严重度**: 中 | **来源**: HKV AI 版，已验证

`__threadfence()` 是内存屏障（保证写入对其他线程可见），不是延迟等待。正确方式用协作组 `__shfl` 同步或自旋。
