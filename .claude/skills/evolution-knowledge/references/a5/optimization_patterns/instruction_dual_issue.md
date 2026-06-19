# 指令双发射优化

## 原理

A5 Vector Core 支持指令双发射：在同一时钟周期内发射两条**无数据依赖**的指令。

## 条件

两条指令双发射需满足：
1. **无数据依赖**: 两条指令的源寄存器和目标寄存器无交集
2. **无资源冲突**: 不同时使用同一功能单元
3. **指令类型兼容**: 某些指令组合支持双发射

## 优化方法

### 1. 循环展开暴露并行性

```cpp
// [BAD] 单次循环，指令间有依赖
for (uint16_t i = 0; i < repTimes; ++i) {
    LoadAlign(r0, src + i * repSize);
    Add(r1, r0, r0, mask);        // 依赖 r0
    StoreAlign(dst + i * repSize, r1, mask);  // 依赖 r1
}

// [OK] 2x 展开，交错无依赖指令
for (uint16_t i = 0; i < repTimes; i += 2) {
    LoadAlign(r0, src + i * repSize);
    LoadAlign(r2, src + (i+1) * repSize);     // 与上一条无依赖 → 双发射
    Add(r1, r0, r0, mask);
    Add(r3, r2, r2, mask);                     // 与上一条无依赖 → 双发射
    StoreAlign(dst + i * repSize, r1, mask);
    StoreAlign(dst + (i+1) * repSize, r3, mask); // 双发射
}
```

### 2. 指令重排

将无依赖的指令交错排列。

```cpp
// [BAD] 串行依赖链
LoadAlign(a, addrA);
Add(a, a, b, mask);       // 依赖 a
StoreAlign(addrA, a, mask); // 依赖 a
LoadAlign(c, addrC);
Add(c, c, d, mask);
StoreAlign(addrC, c, mask);

// [OK] 交错排列
LoadAlign(a, addrA);
LoadAlign(c, addrC);       // 与 a 无依赖 → 双发射
Add(a, a, b, mask);
Add(c, c, d, mask);         // 与 a 无依赖 → 双发射
StoreAlign(addrA, a, mask);
StoreAlign(addrC, c, mask); // 双发射
```

## 约束

- 循环展开会增加 RegTensor 使用量（2x 展开 → 约 2x RegTensor）
- 需在双发射收益和寄存器压力之间平衡
- **建议**: 先确保 RegTensor ≤ 32，再尝试展开
- 展开倍数建议: 2x（安全起点），4x（激进但需检查寄存器）
