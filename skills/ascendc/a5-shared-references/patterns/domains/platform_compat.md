# Domain: Platform Compatibility & SIMD Specifics
> Patterns for AscendC platform-specific issues: PipeBarrier alignment, cache hints, TQue bugs.
> Load when: Analyzer detects SIMD DataCopy, PipeBarrier, __ldg usage, or TQue configuration.

---

## Patterns

### F-P4: SIMD PipeBarrier 与 DataCopy 对齐

**严重度**: 高 | **模式**: SIMD

**反模式**: `PipeBarrier<PIPE_MTE2>()` — 细粒度 barrier 可能导致数据竞争

**正确模式**: `PipeBarrier<PIPE_ALL>()` 保证正确性。最优用 TQue depth=2 双缓冲。

**DataCopy 对齐要求**: fp32: %8==0, fp16/bf16: %16==0。不对齐时回退 SIMT 或用 `DataCopyPad`。

---

### P-P18: `__ldg`/`__stg` L2 Cache Hint（更新 2026-04-02）

**严重度**: 高 | **来源**: A5 实测 + HKV 专家代码 + DavidV100 手册 | **平台**: Ascend950PR

**API**: AscendC 提供带模板参数的 `__ldg`/`__stg`，控制 L2 cache 和 L1/dcache 行为:
```cpp
#include <kernel_operator.h>  // LD_L2CacheType, ST_L2CacheType, L1CacheType

// 读: 控制 L2 分配策略 + L1/dcache 缓存
T val = __ldg<LD_L2CacheType::hint, L1CacheType::hint>(ptr);

// 写: 控制 L2 写回策略 + L1 缓存
__stg<ST_L2CacheType::hint, L1CacheType::hint>(ptr, val);
```

**可用 hint 值** (来源: HKV 专家代码 + 手册 HA.FS007):

| 读 (LD_L2CacheType) | 含义 |
|---------------------|------|
| `L2_CACHE_HINT_NORMAL` | 正常缓存（默认，等同不带参数的 `__ldg`） |
| `L2_CACHE_HINT_NOTALLOC_CLEAN` | 读完不占 L2 slot，防止大范围扫描污染 cache |

| 写 (ST_L2CacheType) | 含义 |
|---------------------|------|
| `L2_CACHE_HINT_NORMAL_FV` | 正常写回 L2 |

| L1/dcache (L1CacheType) | 含义 |
|-------------------------|------|
| `CACHEABLE` | 通过 L1/dcache 缓存 |
| `NON_CACHEABLE` | 绕过 L1/dcache |

**按访问模式选择 hint**:

```cpp
// 1. 数据被多个 core/token 重复读（expert 行, embedding table）
//    → L2 保留 + dcache 缓存: 最大化命中率
val = __ldg<L2_CACHE_HINT_NORMAL, L1CacheType::CACHEABLE>(expert_ptr);

// 2. 数据顺序扫描只读一次（edge index, weight 数组）
//    → L2 不分配: 防止污染 cache，留空间给热点数据
val = __ldg<L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(index_ptr);

// 3. 输出写（一次性写，不需要后续读）
//    → L1 不缓存: 不浪费 dcache 空间
__stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(out_ptr, val);

// 4. HKV 桶内随机查找（小块数据反复扫描）
//    → L2 不分配 + L1 缓存: 桶内数据走 dcache 高速路径
val = __ldg<L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(bucket_ptr);
```

**历史**: 之前测不带模板参数的 `__ldg`（OL-18, 2026-03-26）无效果——因为默认是 `L2_CACHE_HINT_NORMAL` + 默认 L1 策略，在大范围顺序扫描下与普通读无差异。带 hint 的版本可以区分热点数据（保留 L2）和冷数据（不分配 L2），这才是正确用法。

**实验结果 (Batch 14-5)**: SIMT persistent SG forward 上测试 `NOTALLOC_CLEAN` (index/weight) + `NORMAL_PERS` (expert)。**无正面效果**——dim=64 慢 24%（指令开销），其余无变化。dcache 已有效缓存。L2 hint 价值在跨 core 共享场景（如 HKV），不在 SIMT persistent 顺序遍历场景。

**A5 实测数据**（56 blocks × 32 threads, stride-scan, aclrtEvent 计时）:

| 数据大小 | 普通读 BW | `__ldg` BW | 差异 |
|---------|----------|-----------|------|
| 4 MB | 49.5 GB/s | 49.7 GB/s | +0.3% |
| 16 MB | 54.3 GB/s | 54.4 GB/s | +0.2% |
| 64 MB | 43.5 GB/s | 43.6 GB/s | +0.2% |
| 256 MB | 22.8 GB/s | 22.8 GB/s | -0.1% |

**决策规则**:
- 数据集 >> L2 缓存 → 不用 `__ldg`（pooling, SG, 大规模归约）
- 数据集 ≤ L2 缓存 且反复访问 → 用 `__ldg`（哈希桶扫描、小矩阵乘法）
- 不确定时 → 不加（0 收益但增加代码复杂度）

---

### P-P27: bf16 标量转换 — Cast(bf16→float) + GetValue

**严重度**: CRITICAL | **来源**: A5 实测 (2026-03-31) | **平台**: Ascend950PR + CANN 9.0.0

**核心发现**: bisheng 不支持 `static_cast<float>(bfloat16_t)` 标量转换。SIMD `Cast()` 向量 intrinsic 正常工作。

**错误模式**:
```cpp
// ❌ 编译失败: "not support bf16 type cast"
bfloat16_t val = gmBuf.GetValue(i);
float fval = static_cast<float>(val);  // FAIL

// ❌ 有损: bf16 exponent=8bit > half exponent=5bit → 值域溢出为 inf
Cast(halfBuf, bf16Buf, RoundMode::CAST_NONE, n);  // bf16→half 有损!
```

**正确模式 (P-P27)**:
```cpp
// ✅ bf16 标量读取: DataCopyPad → Cast(bf16→float) → GetValue(float)
DataCopyPad(bf16Buf, weightGm_[offset], copyParams, padNone);
PipeBarrier<PIPE_ALL>();
Cast(floatBuf, bf16Buf, RoundMode::CAST_NONE, count);  // bf16→float 无损
PipeBarrier<PIPE_V>();
float w = floatBuf.GetValue(i);  // float 标量读取正常

// ✅ SIMT 场景 (无法用 Cast): 位操作 workaround
float simt_to_float(bfloat16_t v) {
  uint16_t bits; __builtin_memcpy(&bits, &v, sizeof(bits));
  uint32_t f32 = (uint32_t)bits << 16;
  float r; __builtin_memcpy(&r, &f32, sizeof(r)); return r;
}
```

**类型转换路径表** (reg_convert.h):

| 源→目标 | SIMD Cast() | 标量 static_cast | 备注 |
|---------|:-----------:|:---------------:|------|
| bf16→float | ✅ `asc_bfloat162float` | ❌ | **用 Cast 后 GetValue** |
| float→bf16 | ✅ `asc_float2bfloat16_rn` | ❌ | Cast 后 SetValue |
| bf16→half | ✅ `asc_bfloat162half_rn` | ❌ | **有损!** exponent 溢出 |
| half→bf16 | ✅ `asc_half2bfloat16_rn` | ❌ | 有损 (mantissa truncation) |
| half→float | ✅ `asc_half2float` | ✅ | 两边都行 |
| float→half | ✅ `asc_float2half_rn` | ✅ | 两边都行 |

**决策规则**:
1. bf16 需要标量值 → **先 Cast(bf16→float) 再 GetValue**，不要 Cast(bf16→half)
2. SIMT kernel 中 bf16 标量 → **simt_to_float 位操作**（无法用 SIMD Cast）
3. half 标量转换 → `static_cast<float>(half)` 直接可用，无需特殊处理

**参考**: CANN `reg_convert.h`, 最小复现 `tests/repro/bf16_cast_repro.cpp`

---

### P-P30: fp16/bf16 Scalar Kernel Argument Passing

**严重度**: 高 | **来源**: E2E 技能测试 (2026-04-01) | **平台**: 全 AscendC

**问题**: `extern "C" __global__ __aicore__` kernel 入口不能直接传 `half`/`bfloat16_t` 标量参数。ABI 不支持，会导致值损坏或未定义行为。

**反模式**:
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, half num, int64_t size) {  // ❌ half 不能过 extern "C" 边界
```

**正确模式** (uint16_t bit-pattern):
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  half num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;  // 从 bit pattern 重建
  // ... 使用 num ...
}

// bf16 同理:
extern "C" __global__ __aicore__ void init_kernel_bf16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  bfloat16_t num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;
  // ...
}
```

**Host 端调用**:
```cpp
half h_val = ...;
uint16_t bits = *reinterpret_cast<uint16_t*>(&h_val);
aclrtlaunch_init_kernel_fp16(..., bits, size);
```

**触发条件**: 任何 kernel 有非 float/int/int64_t 标量参数（尤其是 init/fill kernel 的初始值参数）。

---

### P-P31: NPU Native atomicAdd (无需 fastAtomicAdd Packed Pair)

**严重度**: 中 | **来源**: E2E 技能测试 (2026-04-01) | **平台**: Ascend950PR

**背景**: GPU 的 `fastAtomicAdd` 使用 `half2`/`__nv_bfloat162` 打包对做 32-bit aligned atomic，因为 GPU 不支持原生 16-bit atomicAdd。

**NPU 区别**: Ascend950PR SIMT 模式原生支持 `atomicAdd` 对 `half` 和 `bfloat16_t` 类型。**不需要** packed pair workaround。

**翻译规则**:
```cpp
// GPU (CUDA):
fastAtomicAdd(base, offset, length, value);  // packed half2 aligned atomic

// NPU (AscendC):
atomicAdd(base + offset, value);  // 直接调用，NPU 原生支持 fp16/bf16
```

**注意**:
- `atomicAdd` 是全局内建函数，不需要 `Simt::` 前缀
- 如果项目中有 `fast_atomic_add.h` wrapper，NPU 端可以简化为直接转发
- 性能: NPU native atomicAdd 已是硬件最优路径，无需 packed optimization

---

## Known Bugs

### TQue depth=1 数据竞争

**严重度**: 高 | **来源**: A5 SIMD 实测

TQue<TPosition::VECIN, 1> 在某些 kernel 中出现间歇性数据竞争（结果随机偏差）。
**临时修复**: 使用 depth=2 双缓冲。根因待 CANN 团队确认。

参见 [unverified/candidates.md](../unverified/candidates.md) 中 U-P2 对 TQue 的讨论。
