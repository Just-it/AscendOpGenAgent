# 低延迟归约指令选择

## 问题

归约操作（ReduceSum/ReduceMax/ReduceMin）在 A5 上有多种实现方式，延迟差异显著。

## 指令选择指南

### 方案对比

| 方案 | 适用数据量 | 延迟 | 说明 |
|------|----------|------|------|
| WholeReduceSum 单次 | ≤ VL 元素 | 高 | 全局归约，开销大 |
| 二分累加 + WholeReduceSum | > VL 元素 | 中 | 先树形归约到 VL，再 WholeReduceSum |
| BlockReduceSum + WholeReduceSum | > VL 元素 | 低 | 块归约 + 全局归约 |

### BlockReduceSum + WholeReduceSum (推荐)

```cpp
// 大数据量归约: 先 BlockReduceSum 到每块的部分和，再 WholeReduceSum
__simd_vf__ inline void ReduceVF(__ubuf__ float* dstAddr, __ubuf__ float* srcAddr,
                                 __ubuf__ float* workAddr,
                                 uint32_t count, uint32_t oneRepSize, uint16_t repTimes)
{
    RegTensor<float> srcReg, accReg;
    MaskReg mask;

    // Phase 1: BlockReduceSum — 累加到 accReg
    mask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(accReg, 0.0f);  // 初始化累加器

    for (uint16_t i = 0; i < repTimes; ++i) {
        MaskReg iterMask = UpdateMask<float>(count);
        LoadAlign(srcReg, srcAddr + i * oneRepSize);
        Add(accReg, accReg, srcReg, iterMask);  // 按 VL 粒度累加
    }

    // Phase 2: WholeReduceSum — 将 VL 长度的 accReg 归约为标量
    StoreAlign(workAddr, accReg, mask);
    // 在 VF 外部使用高阶 API 完成最终归约
}
```

### 二分累加模式

```cpp
// 适用于 repTimes 为 2 的幂时
RegTensor<float> regs[LOG2_REPTIMES];
// 树形两两相加，最终归约到一个 RegTensor
```

## 选择建议

| 数据量 | 推荐方案 |
|--------|---------|
| ≤ 64 float (= VL) | 单次 WholeReduceSum |
| 64~1024 float | 二分累加 + WholeReduceSum |
| > 1024 float | BlockReduceSum + WholeReduceSum |

## ReduceMax/ReduceMin 同理

归约类操作的指令选择思路一致，将 Add 替换为 Max/Min 即可。
