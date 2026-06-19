# AddrReg - 地址寄存器

## 定义

AddrReg 用于存储地址偏移量，支持最多 4 层循环的地址计算。

## API 签名

```cpp
// 单层循环: offset = index0 * stride0
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0);

// 双层循环: offset = index0 * stride0 + index1 * stride1
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0,
                                             uint16_t index1, uint32_t stride1);

// 三层循环: offset = index0 * stride0 + index1 * stride1 + index2 * stride2
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0,
                                             uint16_t index1, uint32_t stride1,
                                             uint16_t index2, uint32_t stride2);

// 四层循环: offset = i0*s0 + i1*s1 + i2*s2 + i3*s3
template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0,
                                             uint16_t index1, uint32_t stride1,
                                             uint16_t index2, uint32_t stride2,
                                             uint16_t index3, uint32_t stride3);
```

## 参数说明

- `T`: 数据类型，决定地址计算的步长单位（以 sizeof(T) 为单位）
- `indexN`: 第 N 层循环的迭代索引（uint16_t）
- `strideN`: 第 N 层循环的地址步长（以元素数为单位）

## 使用示例

### 单层循环
```cpp
for (uint16_t i = 0; i < repeatTimes; ++i) {
    AddrReg aReg = CreateAddrReg<float>(i, oneRepeatSize);
    LoadAlign(srcReg, srcAddr, aReg);   // srcAddr + i * oneRepeatSize * sizeof(float)
    StoreAlign(dstAddr, dstReg, aReg, mask);
}
```

### 多层循环
```cpp
for (uint16_t i = 0; i < outerLoop; ++i) {
    for (uint16_t j = 0; j < innerLoop; ++j) {
        AddrReg aReg = CreateAddrReg<half>(i, outerStride, j, innerStride);
        // offset = i * outerStride + j * innerStride (元素单位)
        LoadAlign(srcReg, srcAddr, aReg);
    }
}
```

## 编译器自动优化

编译器能识别以下模式并自动生成高效的 AddrReg 代码：

```cpp
// 编译器可识别的模式
for (uint16_t i = 0; i < extent1; i++) {
    for (uint16_t j = 0; j < extent2; j++) {
        LoadAlign(srcReg, srcAddr + i * const1 + j * const2);
    }
}
// 自动优化为 CreateAddrReg<T>(i, const1, j, const2)
```

**最佳实践**: 在简单场景下让编译器自动匹配，不需要手动使用 AddrReg。手动使用可能限制编译器全局优化。
