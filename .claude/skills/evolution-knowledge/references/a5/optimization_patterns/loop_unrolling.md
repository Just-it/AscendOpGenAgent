# 循环展开优化

## 目标

通过循环展开减少循环控制开销、暴露指令级并行性、提升双发射利用率。

## 展开方式

### 手动展开

```cpp
// 原始
for (uint16_t i = 0; i < repTimes; ++i) {
    mask = UpdateMask<T>(count);
    LoadAlign(srcReg, srcAddr + i * oneRepSize);
    Adds(dstReg, srcReg, scalar, mask);
    StoreAlign(dstAddr + i * oneRepSize, dstReg, mask);
}

// 2x 展开
uint16_t mainLoop = repTimes / 2;
uint16_t remainder = repTimes % 2;
for (uint16_t i = 0; i < mainLoop; ++i) {
    uint16_t idx0 = i * 2;
    uint16_t idx1 = i * 2 + 1;
    mask0 = UpdateMask<T>(count);
    mask1 = UpdateMask<T>(count);
    LoadAlign(srcReg0, srcAddr + idx0 * oneRepSize);
    LoadAlign(srcReg1, srcAddr + idx1 * oneRepSize);
    Adds(dstReg0, srcReg0, scalar, mask0);
    Adds(dstReg1, srcReg1, scalar, mask1);
    StoreAlign(dstAddr + idx0 * oneRepSize, dstReg0, mask0);
    StoreAlign(dstAddr + idx1 * oneRepSize, dstReg1, mask1);
}
// 处理余数
for (uint16_t i = 0; i < remainder; ++i) {
    mask = UpdateMask<T>(count);
    LoadAlign(srcReg0, srcAddr + (mainLoop*2+i) * oneRepSize);
    Adds(dstReg0, srcReg0, scalar, mask);
    StoreAlign(dstAddr + (mainLoop*2+i) * oneRepSize, dstReg0, mask);
}
```

## 展开因子选择

| 展开因子 | RegTensor 增量 | 适用条件 |
|---------|---------------|---------|
| 1x (不展开) | 基线 | RegTensor 已接近 32 |
| 2x | +1~3 RegTensor | 默认推荐 |
| 4x | +3~6 RegTensor | RegTensor 余量充足 (≤20) |
| 8x | +7~12 RegTensor | 极简计算（如纯搬运） |

## 与 Hardware Loop 的交互

- 展开后的循环仍需满足 Hardware Loop 条件
- `uint16_t` 迭代变量、起始 0、步长 1
- 避免在展开后引入 if/else

## 注意事项

1. **尾部处理**: 展开后必须处理 `repTimes % unrollFactor` 的余数
2. **寄存器预算**: 展开前计算总 RegTensor 数量，确保 ≤ 32
3. **代码体积**: 过度展开增加 ICache 压力
4. **编译器行为**: 编译器可能自动展开，手动展开前检查是否有冲突
