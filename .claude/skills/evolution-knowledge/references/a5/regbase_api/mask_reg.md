# MaskReg - 掩码寄存器

## 定义

MaskReg 用于控制哪些元素参与计算，宽度为 VL/8 = 32 Bytes。

## 硬件约束

- **每个 VF 函数最多 8 个 MaskReg**
- 超过后寄存器溢出，性能下降

## API

### CreateMask - 创建掩码

```cpp
template <typename T, MaskPattern mode = MaskPattern::ALL,
          const RegTrait& regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg CreateMask();
```

### MaskPattern 枚举

| 模式 | 值 | 含义 |
|------|-----|------|
| ALL | 0 | 所有元素为 True |
| VL1 | 1 | 最低 1 个元素 |
| VL2 | 2 | 最低 2 个元素 |
| VL3 | 3 | 最低 3 个元素 |
| VL4 | 4 | 最低 4 个元素 |
| VL8 | 5 | 最低 8 个元素 |
| VL16 | 6 | 最低 16 个元素 |
| VL32 | 7 | 最低 32 个元素 |
| VL64 | 8 | 最低 64 个元素 |
| VL128 | 9 | 最低 128 个元素 |
| M3 | 10 | 3 的倍数位置 |
| M4 | 11 | 4 的倍数位置 |
| H | 12 | 低半部元素 |
| Q | 13 | 低四分之一元素 |
| ALLF | 15 | 所有元素为 False |

### UpdateMask - 递减式掩码

```cpp
template <typename T, const RegTrait& regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg UpdateMask(uint32_t& scalarValue);
```

**行为**:
- 根据 scalarValue 设置掩码：前 min(scalarValue, VL_T) 个元素为 True
- 调用后自动递减：`scalarValue = (scalarValue < VL_T) ? 0 : (scalarValue - VL_T)`
- `VL_T = VL / sizeof(T)` = 每个 RegTensor 能装的 T 类型元素数

**VL_T 速查**:
| 类型 | sizeof(T) | VL_T |
|------|-----------|------|
| int8_t | 1 | 256 |
| half/bfloat16_t | 2 | 128 |
| float | 4 | 64 |
| int64_t | 8 | 32 |

### MaskReg 搬入搬出

```cpp
// 从 UB 加载到 MaskReg
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void LoadAlign(MaskReg& mask, __ubuf__ T* srcAddr);

template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void LoadAlign(MaskReg& mask, __ubuf__ T* srcAddr, AddrReg offset);

// 从 MaskReg 存储到 UB
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, MaskReg& mask);

template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, MaskReg& mask, AddrReg offset);

// 从 RegTensor 生成 MaskReg
template <typename T, int16_t offset, typename U>
__simd_callee__ inline void MaskGenWithRegTensor(MaskReg& dst, U& srcReg);
```

**MaskDist 模式**:
- `DIST_NORM`: 正常模式 (VL/8 Byte 对齐)
- `DIST_US`: 上采样模式 (VL/16 Byte 对齐，每 bit 重复)
- `DIST_DS`: 下采样模式 (每隔一个 bit 丢弃)
- `DIST_PACK`: 打包模式 (Store 专用)

## 典型用法

```cpp
// 处理任意长度数据
uint32_t count = totalElements;
for (uint16_t i = 0; i < repeatTimes; ++i) {
    MaskReg mask = UpdateMask<float>(count);  // 自动处理尾部
    LoadAlign(srcReg, srcAddr, CreateAddrReg<float>(i, oneRepSize));
    Add(dstReg, srcReg, srcReg, mask);
    StoreAlign(dstAddr, dstReg, CreateAddrReg<float>(i, oneRepSize), mask);
}
// 最后一次循环 count < VL_T 时，mask 自动只激活有效元素
```
