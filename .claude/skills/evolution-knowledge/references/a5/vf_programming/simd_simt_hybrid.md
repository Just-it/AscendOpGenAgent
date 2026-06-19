# SIMD/SIMT 混合编程

## 概述

A5 支持在同一个 kernel 中混合使用 SIMD 和 SIMT 编程模型。

## 执行模型

- 一个 AIV 核在任意时刻只能执行 SIMD 或 SIMT（不能同时）
- 不同类型的 VF 函数可以快速切换（VF 粒度）
- UB Data Cache 在 VF 切换时保持

## SIMD vs SIMT 对比

| 维度 | SIMD (`__simd_vf__`) | SIMT (`__simt_vf__`) |
|------|---------------------|---------------------|
| 数据源 | UB (LoadAlign → RegTensor) | GM 或 UB (直接访问) |
| 并行模型 | 向量化（VL 宽度） | 线程级（最多 2048 线程） |
| 适用场景 | 规则、连续的逐元素操作 | 不规则、分支、动态索引 |
| 子函数 | `__simd_callee__` | `__simt_callee__` |
| 寄存器分配 | 每 VF 最多 32 RegTensor | 按线程数分配（见下表） |

### SIMT 线程寄存器分配
| 线程数 | 每线程寄存器数 |
|--------|-------------|
| 1~256 | 127 |
| 257~512 | 64 |
| 513~1024 | 32 |
| 1025~2048 | 16 |

## SIMT VF 声明

```cpp
__simt_vf__ __launch_bounds__(thread_num)
inline void SimtFunction(__gm__ float* dst, __gm__ float* src, int count)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < count; idx += gridDim.x * blockDim.x)
    {
        dst[idx] = src[idx] * 2.0f;
    }
}
```

### __launch_bounds__
- 可选参数（默认 1024）
- 指定编译时最大线程数
- 影响寄存器分配策略

### 参数约束
- 支持指针类型: `__ubuf__*`, `__gm__*`
- 支持标量类型: bool, int8_t, uint8_t, int16_t, uint16_t, half, bfloat16_t, int32_t, uint32_t, float, int64_t, uint64_t
- 参数必须是 PoD 类型
- **不支持** struct/array 参数传递

### 内置变量
| 变量 | 含义 |
|------|------|
| `threadIdx.x/y/z` | 线程在 block 内的索引 |
| `blockIdx.x/y/z` | block 在 grid 内的索引 |
| `blockDim.x/y/z` | block 的维度 |
| `gridDim.x/y/z` | grid 的维度 |

## 混合编程示例

```cpp
// SIMT: 不规则的 gather 操作
__simt_vf__ __launch_bounds__(1024)
inline void SimtGather(__ubuf__ float* output, __gm__ float* input,
                       __ubuf__ int32_t* indices, int count)
{
    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        output[idx] = input[indices[idx]];  // 随机访问
    }
}

// SIMD: 规则的逐元素操作
template <typename T>
__simd_vf__ inline void SimdAdd(__ubuf__ T* dst, __ubuf__ T* src,
                                uint32_t count, uint32_t oneRepSize,
                                uint16_t repeatTimes)
{
    RegTensor<T> srcReg, dstReg;
    MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<T>(count);
        LoadAlign(srcReg, src + i * oneRepSize);
        Adds(dstReg, srcReg, 1.0f, mask);
        StoreAlign(dst + i * oneRepSize, dstReg, mask);
    }
}

// 混合调用
__aicore__ inline void Compute()
{
    // 第一步: SIMT 做不规则 gather
    asc_vf_call<SimtGather>(dim3{256, 1, 1}, ubOutput, gmInput, ubIndices, count);

    // 第二步: SIMD 做规则计算
    AscendC::VF_CALL<SimdAdd<float>>(ubOutput, ubOutput, count, oneRepSize, repTimes);
}
```

## 共享资源

- SIMT VF 和 SIMD VF 共享 **ICache**（指令缓存）
- 共享 **Vector ALU 单元**
- UB 有共享空间 + SIMT Data Cache

## 选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 逐元素加减乘除 | SIMD | 规则访问，向量化效率高 |
| Reduce/Scan | SIMD | 顺序依赖，向量化处理 |
| Gather/Scatter | SIMT | 随机访问，需要线程级并行 |
| 条件分支密集 | SIMT | 每个线程独立分支 |
| Attention mask | SIMT | 动态索引 |
| 矩阵逐行操作 | SIMD | 行内连续 |
