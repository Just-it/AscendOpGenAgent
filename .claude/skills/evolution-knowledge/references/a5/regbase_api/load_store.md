# LoadAlign / StoreAlign / UnAlign - 数据搬运 API

## 对齐搬运

### LoadAlign - 对齐加载 (UB → RegTensor)

要求源地址 32 Byte 对齐。

```cpp
// 基本形式
template <typename T, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr);

// 带地址寄存器
template <typename T, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr, AddrReg offset);

// POST_MODE_UPDATE (自动更新地址指针)
template <typename T, PostLiteral postMode, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T*& srcAddr, int32_t postUpdateStride);

// 双寄存器加载
template <typename T, LoadDist dist, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg0, U& dstReg1, __ubuf__ T* srcAddr);

template <typename T, LoadDist dist, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg0, U& dstReg1, __ubuf__ T* srcAddr, AddrReg offset);
```

**LoadDist 模式**:
| 模式 | 含义 |
|------|------|
| `DIST_NORM` | 自动检测（默认） |
| `DIST_NORM_B8/B16/B32` | 正常模式，加载 VL 数据 |
| `DIST_DINTLV_B8/B16/B32` | 去交织模式（双寄存器加载时） |

### StoreAlign - 对齐存储 (RegTensor → UB)

要求目标地址 32 Byte 对齐。

```cpp
// 基本形式（带 mask）
template <typename T, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, MaskReg& mask);

// 带地址寄存器
template <typename T, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, AddrReg offset, MaskReg& mask);

// POST_MODE_UPDATE
template <typename T, PostLiteral postMode, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T*& dstAddr, U& srcReg, int32_t postUpdateStride, MaskReg& mask);

// 双寄存器存储
template <typename T, StoreDist dist, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T, StoreDist dist, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg0, U& srcReg1, AddrReg offset, MaskReg& mask);
```

**StoreDist 模式（单寄存器）**:
| 模式 | 含义 | 对齐要求 |
|------|------|---------|
| `DIST_NORM` | 自动检测（默认） | 32B |
| `DIST_NORM_B8/B16/B32` | 正常模式 | 32B |
| `DIST_FIRST_ELEMENT_B8/B16/B32` | 仅存储首个元素 | sizeof(T) |
| `DIST_PACK_B16/B32/B64` | 打包模式（压缩到 VL/2） | 32B |
| `DIST_PACK4_B32` | 打包到 VL/4 | 32B |

**StoreDist 模式（双寄存器）**:
| 模式 | 含义 |
|------|------|
| `DIST_INTLV_B8/B16/B32` | 交织模式 (32B 对齐) |

## 非对齐搬运

### UnalignReg 缓冲区

```cpp
AscendC::MicroAPI::UnalignRegForLoad uregLoad;    // 最多 4 个/VF
AscendC::MicroAPI::UnalignRegForStore uregStore;   // 最多 4 个/VF
```

缓冲区大小: 32 Bytes。用于处理非 32B 对齐地址的访问。

### LoadUnAlign - 非对齐加载

```cpp
// 预初始化（必须在循环外调用）
template <typename T>
__simd_callee__ inline void LoadUnAlignPre(UnalignRegForLoad& ureg, __ubuf__ T* srcAddr);

template <typename T>
__simd_callee__ inline void LoadUnAlignPre(UnalignRegForLoad& ureg, __ubuf__ T* srcAddr, AddrReg& areg);

// 场景1: POST_MODE_UPDATE
template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void LoadUnAlign(U& dstReg, UnalignRegForLoad& ureg,
                                        __ubuf__ T*& srcAddr, uint32_t postUpdateStride);

// 场景2: 无地址更新
template <typename T, typename U>
__simd_callee__ inline void LoadUnAlign(U& dstReg, UnalignRegForLoad& ureg, __ubuf__ T* srcAddr);

// 场景3: 带 AddrReg
template <typename T, typename U>
__simd_callee__ inline void LoadUnAlign(U& dstReg, UnalignRegForLoad& ureg,
                                        __ubuf__ T*& srcAddr, AddrReg& areg, uint32_t inc);
```

### StoreUnAlign - 非对齐存储

```cpp
// 场景1: POST_MODE_UPDATE
template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void StoreUnAlign(__ubuf__ T*& dstAddr, U& srcReg,
                                         UnalignRegForStore& ureg, uint32_t postUpdateStride);

// Post 处理（循环结束后必须调用）
template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE>
__simd_callee__ inline void StoreUnAlignPost(__ubuf__ T*& dstAddr, UnalignRegForStore& ureg,
                                             int32_t postUpdateStride);

// 场景2: 带 AddrReg
template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void StoreUnAlign(__ubuf__ T*& dstAddr, U& srcReg,
                                         UnalignRegForStore& ureg, AddrReg& areg);

template <typename T>
__simd_callee__ inline void StoreUnAlignPost(__ubuf__ T*& dstAddr, UnalignRegForStore& ureg,
                                             AddrReg& areg);
```

## 关键约束

1. LoadAlign/StoreAlign **必须** 32B 对齐
2. LoadUnAlign 必须先调用 `LoadUnAlignPre` 初始化
3. StoreUnAlign 必须在循环结束后调用 `StoreUnAlignPost`
4. **StoreAlign 必须带 MaskReg 参数**（与 LoadAlign 不同）
5. 非对齐寄存器资源有限：每个 VF 最多 4 个 UnalignRegForLoad + 4 个 UnalignRegForStore
