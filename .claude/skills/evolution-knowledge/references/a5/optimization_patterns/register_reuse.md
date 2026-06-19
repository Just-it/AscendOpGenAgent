# 寄存器复用优化

## 问题

每个 VF 函数最多 32 个 RegTensor 和 8 个 MaskReg。超过限制后，编译器插入 spill/fill 和 sync 指令，性能显著下降。

## 优化策略

### 1. 布尔代数简化

通过数学恒等式减少中间寄存器。

```cpp
// [BAD] 使用 5 个 RegTensor
RegTensor<float> a, b, c, d, result;
Add(c, a, b, mask);      // c = a + b
Mul(d, c, a, mask);       // d = c * a
Add(result, d, b, mask);  // result = d + b

// [OK] 复用寄存器，使用 3 个 RegTensor
RegTensor<float> a, b, tmp;
Add(tmp, a, b, mask);     // tmp = a + b
Mul(tmp, tmp, a, mask);   // tmp = tmp * a（原地复用）
Add(tmp, tmp, b, mask);   // tmp = tmp + b
```

### 2. 指令重排

将消费者指令紧跟生产者，允许寄存器尽早释放。

```cpp
// [BAD] 寄存器生命周期长
LoadAlign(r0, addr0);
LoadAlign(r1, addr1);
LoadAlign(r2, addr2);   // r0, r1, r2 同时存活
Add(r3, r0, r1, mask);
Mul(r4, r3, r2, mask);

// [OK] 寄存器尽早释放
LoadAlign(r0, addr0);
LoadAlign(r1, addr1);
Add(r0, r0, r1, mask);  // r0 原地复用，r1 可释放
LoadAlign(r1, addr2);    // 复用 r1
Mul(r0, r0, r1, mask);  // r0 原地复用
```

### 3. 循环分块

当单次循环需要太多寄存器时，拆分为多次。

```cpp
// [BAD] 单次循环需要 40 个 RegTensor（超过 32 限制）

// [OK] 拆分为两个 VF，每个使用 20 个 RegTensor
__simd_vf__ inline void Phase1VF(...) { /* 前半部分计算 */ }
__simd_vf__ inline void Phase2VF(...) { /* 后半部分计算 */ }
```

## 监控寄存器使用

计算方法: 统计 VF 函数中同时存活的 RegTensor 最大数量。

| 操作 | 新增 RegTensor | 建议 |
|------|---------------|------|
| LoadAlign | +1 | 尽早消费 |
| 二元计算 (Add/Mul) | +1 (结果) | 使用原地复用 |
| StoreAlign | -1 (释放) | 尽早 Store |
| Duplicate | +1 | 提到循环外 |
