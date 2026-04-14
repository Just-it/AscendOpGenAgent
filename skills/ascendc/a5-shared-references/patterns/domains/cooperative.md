# Domain: Cooperative Group Programming
> Patterns for warp-level cooperative traversal, shuffle reduction, and value transport.
> Load when: Analyzer detects __shfl, __shfl_xor, GROUP_SIZE, or cooperative parallel loops.

---

## Patterns

### P-P13: 协作组并行遍历 + shuffle 规约

**严重度**: 高 | **来源**: HKV 手写版，已验证（通用化）

将单线程顺序遍历 → N 线程协作并行 + `__shfl_xor` 分治规约。

```cpp
auto rank = threadIdx.x % GROUP_SIZE;
for (uint32_t pos = rank; pos < array_size; pos += GROUP_SIZE) { ... }
for (int32_t offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    auto other = __shfl_xor(val, offset, GROUP_SIZE);
    if (other < val) val = other;  // 或 +=, max, etc.
}
```

适用于任何需要在 warp 内做 min/max/sum 的场景。

---

### P-P16: 协作组 Value 搬运

**严重度**: 中 | **来源**: HKV 手写版，已验证（通用化）

GROUP_SIZE 线程分担大向量拷贝:
```cpp
for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
    dst[pos * dim + j] = src[idx * dim + j];
}
```

dim=128 时: 128 次 store → 16 线程各 8 次。适用于任何大 embedding/feature 向量搬运。
