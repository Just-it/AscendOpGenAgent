# RegTensor - 矢量数据寄存器

## 定义

```cpp
template <typename T, const RegTrait& regTrait = RegTraitNumOne>
struct RegTensor;
```

RegTensor 是 A5 Regbase 架构的基本存储单元，宽度为 VL (256 Bytes)。

## 模板参数

### T - 数据类型
| 位宽 | 支持类型 |
|------|---------|
| b8 | uint8_t, int8_t, bool, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, fp8_e8m0_t |
| b16 | int16_t, uint16_t, half, bfloat16_t |
| b32 | int32_t, uint32_t, float, complex32 |
| b64 | int64_t, uint64_t, complex64 |

### regTrait - 寄存器特征
- `RegTraitNumOne`: 单个 VL 长度寄存器（默认）
- `RegTraitNumTwo`: 双 VL 长度寄存器（仅 b64/complex32 使用）

## 元素数量

| 数据类型 | sizeof(T) | 每个 RegTensor 的元素数 |
|---------|-----------|----------------------|
| int8_t/uint8_t | 1 | 256 |
| half/bfloat16_t | 2 | 128 |
| float/int32_t | 4 | 64 |
| int64_t/double | 8 | 32 |

## 硬件约束

- **每个 VF 函数最多 32 个 RegTensor**
- 超过后编译器插入 spill/fill 和 sync 指令，显著降低性能
- 应通过布尔代数简化、指令重排来减少 RegTensor 使用量

## 计算操作 (MicroAPI)

### 算术操作
```cpp
// 加法
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Add(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

// 标量加法
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Adds(U& dstReg, U& srcReg, T scalar, MaskReg& mask);

// 乘法
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Mul(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

// 标量乘法
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Muls(U& dstReg, U& srcReg, T scalar, MaskReg& mask);

// 减法
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Sub(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

// 除法
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Div(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

// 绝对值
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Abs(U& dstReg, U& srcReg, MaskReg& mask);

// 最大值
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Max(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

// 最小值
template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Min(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);

// 填充标量值
template <typename T, typename U>
__simd_callee__ inline void Duplicate(U& dstReg, T scalar);

template <typename T, typename U>
__simd_callee__ inline void Duplicate(U& dstReg, T scalar, MaskReg& mask);
```

### 比较操作
```cpp
template <typename T, CMPMODE mode, typename U>
__simd_callee__ inline void Compare(MaskReg& dst, U& srcReg0, U& srcReg1, MaskReg& mask);
// CMPMODE: LT, GT, GE, EQ, NE, LE
```

### 选择操作
```cpp
template <typename T, typename U>
__simd_callee__ inline void Select(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);
// mask bit=1 → srcReg0, mask bit=0 → srcReg1
```

### 类型转换
```cpp
template <typename T, typename U, const CastTrait& trait, typename S, typename V>
__simd_callee__ inline void Cast(S& dstReg, V& srcReg, MaskReg& mask);
// CastTrait: RegLayout + SatMode + MaskMergeMode + RoundMode
```

## 使用示例

```cpp
template <typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr,
                              __ubuf__ T* src1Addr, uint32_t count,
                              uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::MicroAPI::MaskReg mask;
    AscendC::MicroAPI::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
```
