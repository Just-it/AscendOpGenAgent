# Membase → Regbase 核心转换规则

本文档定义 A3 (220x/Membase) 到 A5 (351x/Regbase) 的确定性翻译规则。

## 总体原则

1. **保守正确**: 每个 A3 操作都有最直接的 Regbase 对应写法
2. **不做优化**: 不融合 VF、不复用寄存器、不展开循环
3. **一对一映射**: 每个原始操作对应一个独立的 `__simd_vf__` 函数
4. **保持 tiling 框架**: 只调整参数，不改变分块逻辑

## R-01: 计算函数封装

### A3 (Membase)
```cpp
__aicore__ inline void Process()
{
    LocalTensor<float> src0 = inQueue0.DeQue<float>();
    LocalTensor<float> src1 = inQueue1.DeQue<float>();
    LocalTensor<float> dst = outQueue.AllocTensor<float>();
    Add(dst, src0, src1, count);
    outQueue.EnQue(dst);
    inQueue0.FreeTensor(src0);
    inQueue1.FreeTensor(src1);
}
```

### A5 (Regbase)
```cpp
template <typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr,
                              __ubuf__ T* src1Addr, uint32_t count,
                              uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}

__aicore__ inline void Process()
{
    LocalTensor<float> src0 = inQueue0.DeQue<float>();
    LocalTensor<float> src1 = inQueue1.DeQue<float>();
    LocalTensor<float> dst = outQueue.AllocTensor<float>();

    constexpr uint32_t oneRepSize = 256 / sizeof(float);  // VL / sizeof(T)
    uint32_t cnt = count;
    uint16_t repTimes = (count + oneRepSize - 1) / oneRepSize;

    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* src0Addr = (__ubuf__ float*)src0.GetPhyAddr();
    __ubuf__ float* src1Addr = (__ubuf__ float*)src1.GetPhyAddr();

    AscendC::VF_CALL<AddVF<float>>(dstAddr, src0Addr, src1Addr, cnt, oneRepSize, repTimes);

    outQueue.EnQue(dst);
    inQueue0.FreeTensor(src0);
    inQueue1.FreeTensor(src1);
}
```

## R-02: 中间结果处理

### A3: 中间结果存 UB
```cpp
LocalTensor<float> tmp = tmpBuf.AllocTensor<float>();
Add(tmp, src0, src1, count);
Mul(dst, tmp, src2, count);
tmpBuf.FreeTensor(tmp);
```

### A5: 中间结果留在寄存器
```cpp
__simd_vf__ inline void FusedAddMulVF(__ubuf__ float* dstAddr,
    __ubuf__ float* src0Addr, __ubuf__ float* src1Addr, __ubuf__ float* src2Addr,
    uint32_t count, uint32_t oneRepSize, uint16_t repTimes)
{
    RegTensor<float> s0, s1, s2, tmp, dst;
    MaskReg mask;
    for (uint16_t i = 0; i < repTimes; ++i) {
        mask = UpdateMask<float>(count);
        LoadAlign(s0, src0Addr + i * oneRepSize);
        LoadAlign(s1, src1Addr + i * oneRepSize);
        LoadAlign(s2, src2Addr + i * oneRepSize);
        Add(tmp, s0, s1, mask);       // tmp 留在寄存器
        Mul(dst, tmp, s2, mask);       // 直接使用 tmp，无需写回 UB
        StoreAlign(dstAddr + i * oneRepSize, dst, mask);
    }
}
```

**注意**: Phase 1 保守翻译时，可以先拆成两个独立 VF（AddVF + MulVF），Phase 2 再融合。

## R-03/R-04: 数据加载/存储

### A3: 隐式（API 直接操作 UB 地址）
```cpp
Add(dst, src0, src1, count);  // 编译器自动处理 UB 读写
```

### A5: 显式 Load/Store
```cpp
LoadAlign(srcReg0, src0Addr + offset);   // 显式加载
LoadAlign(srcReg1, src1Addr + offset);
Add(dstReg, srcReg0, srcReg1, mask);     // 寄存器计算
StoreAlign(dstAddr + offset, dstReg, mask);  // 显式存储
```

## R-05: Mask 处理

### A3: scalar mask 参数
```cpp
uint64_t mask = 128;
Add(dst, src0, src1, mask, repeatTimes, {1,1,1,8,8,8});
```

### A5: MaskReg + UpdateMask
```cpp
MaskReg mask = UpdateMask<float>(count);  // 自动递减
// 或
MaskReg mask = CreateMask<float, MaskPattern::ALL>();  // 全部激活
```

## R-06: 地址计算

### A3: 手动 offset 计算
```cpp
for (int i = 0; i < repeatTimes; i++) {
    int offset = i * blockSize;
    Add(dst[offset], src0[offset], src1[offset], blockSize);
}
```

### A5: AddrReg
```cpp
for (uint16_t i = 0; i < repeatTimes; ++i) {
    AddrReg aReg = CreateAddrReg<float>(i, oneRepSize);
    LoadAlign(srcReg, srcAddr, aReg);
    StoreAlign(dstAddr, dstReg, aReg, mask);
}
```

## R-07: 流水同步

### A3
```cpp
pipe_barrier(PIPE_V);
// 或
SetFlag<HardEvent::V_MTE3>(eventId);
WaitFlag<HardEvent::V_MTE3>(eventId);
```

### A5
```cpp
// VF 内部同步
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

// VF 外部（__aicore__ 函数中）仍可使用 SetFlag/WaitFlag
// 但推荐使用 Mutex 进行更细粒度控制
```

## R-08: 尾块处理

### A3: scalar 循环或 mask 参数
```cpp
int tailCount = totalCount % blockSize;
if (tailCount > 0) {
    uint64_t mask = tailCount;
    Add(dst[mainOffset], src0[mainOffset], src1[mainOffset], mask, 1, {1,1,1,0,0,0});
}
```

### A5: UpdateMask 自动处理
```cpp
uint32_t count = totalCount;
for (uint16_t i = 0; i < repeatTimes; ++i) {
    MaskReg mask = UpdateMask<float>(count);  // 最后一次自动处理尾部
    LoadAlign(srcReg, srcAddr + i * oneRepSize);
    Add(dstReg, srcReg, srcReg, mask);
    StoreAlign(dstAddr + i * oneRepSize, dstReg, mask);
}
```

## 翻译流程

1. 识别 A3 代码中的所有计算操作（Add, Mul, Sub, ...）
2. 为每个操作（或紧密相关的操作组）创建一个 `__simd_vf__` 函数
3. 在 VF 函数内：声明 RegTensor/MaskReg → 循环 → Load → Compute → Store
4. 在原 Process() 中：提取 `__ubuf__` 指针 → 计算 repeatTimes → VF_CALL
5. 保持 Queue 的 EnQue/DeQue/AllocTensor/FreeTensor 逻辑不变
