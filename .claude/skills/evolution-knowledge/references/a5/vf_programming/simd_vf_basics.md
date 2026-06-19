# __simd_vf__ 函数编程基础

## 函数声明

```cpp
__simd_vf__ inline void FunctionName(__ubuf__ T* param1, __ubuf__ T* param2, ...)
```

### 约束
- **无返回值** (void)
- 参数仅支持 **Pass-by-Pointer** (`__ubuf__ T*`) 和 **基本数据类型** (PoD)
- **不支持** Pass-by-Reference
- **不支持** 函数指针、函数对象
- **不支持** struct/array 参数
- 函数必须标记为 `inline`

## 调用方式

### VF_CALL (SIMD)
```cpp
// 在 __aicore__ 函数中调用
AscendC::VF_CALL<FunctionName<T>>(arg1, arg2, ...);
```

### asc_vf_call (SIMT)
```cpp
// 在 __aicore__ 函数中调用
asc_vf_call<FunctionName>(dim3{threadCount, 1, 1}, arg1, arg2, ...);
```

## 函数调用链

```
__global__ __aicore__       → 核函数入口
    ↓
__aicore__                  → 设备侧函数
    ↓ VF_CALL / asc_vf_call
__simd_vf__ / __simt_vf__  → VF 函数
    ↓
__simd_callee__ / __simt_callee__  → VF 子函数
    ↓
constexpr                   → 编译期函数
```

**调用规则**:
- `__simd_vf__` 只能调用 `__simd_callee__` 和 `constexpr`
- `__simt_vf__` 只能调用 `__simt_callee__` 和 `constexpr`
- 不能交叉调用（SIMD 不能调用 SIMT 的 callee，反之亦然）
- VF 函数**不能是类成员函数**，推荐使用普通函数或 static 类函数

## 完整示例

### SIMD VF 函数
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

// 调用侧
template <typename T>
__aicore__ inline void Compute()
{
    LocalTensor<T> dst = outQueue.AllocTensor<T>();
    LocalTensor<T> src0 = inQueue0.DeQue<T>();
    LocalTensor<T> src1 = inQueue1.DeQue<T>();

    constexpr uint32_t oneRepSize = 256 / sizeof(T);  // VL / sizeof(T)
    uint32_t count = totalElements;
    uint16_t repTimes = (count + oneRepSize - 1) / oneRepSize;

    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
    __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();

    AscendC::VF_CALL<AddVF<T>>(dstAddr, src0Addr, src1Addr, count, oneRepSize, repTimes);

    outQueue.EnQue(dst);
    inQueue0.FreeTensor(src0);
    inQueue1.FreeTensor(src1);
}
```

### SIMT VF 函数
```cpp
__simt_vf__ __launch_bounds__(2048)
inline void SimtAdd(__gm__ float* dst, __gm__ float* src0,
                    __gm__ float* src1, int count)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < count; idx += gridDim.x * blockDim.x)
    {
        dst[idx] = src0[idx] + src1[idx];
    }
}

// 调用侧
__aicore__ inline void Compute()
{
    asc_vf_call<SimtAdd>(dim3{1024, 1, 1}, gmDst, gmSrc0, gmSrc1, size);
}
```

## LocalTensor 到 __ubuf__ 指针转换

从 A3 的 LocalTensor 编程模型转到 A5 的 Regbase 模型，需要提取 UB 地址：

```cpp
LocalTensor<T> tensor = queue.DeQue<T>();
__ubuf__ T* addr = (__ubuf__ T*)tensor.GetPhyAddr();
// 传给 VF_CALL
```
