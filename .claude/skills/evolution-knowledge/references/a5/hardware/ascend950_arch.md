# Ascend 950 (351x/A5) 硬件架构

## 架构概述

Ascend 950PR/950DT 采用 AIC-C-310 架构（NpuArch=3510），核心变化是 **Vector Core 从 Membase 切换到 Regbase 架构**。

### 与 220x (A3) 的核心差异

| 维度 | 220x (A3/Membase) | 351x (A5/Regbase) |
|------|-------------------|-------------------|
| Vector Core 架构 | Membase（所有操作基于 LocalTensor/UB 内存） | Regbase（直接操作 VF 寄存器，中间结果留在寄存器） |
| Vector Length | N/A（以 128B vec_calc_size 为单位） | VL = 256B |
| UB 总大小 | 192KB (Gen2) / 256KB (Gen1) | 248KB |
| UB Bank 结构 | 16 groups × 3 banks × 4KB | 8 groups × 2 banks × 16KB |
| UB 读写口 | 每 group: 1 读 + 1 写 | 每 group: 2 读 + 2 写（支持 2读0写 或 1读1写） |
| 同步机制 | SetFlag/WaitFlag | Mutex + CrossCore 增强 + LocalMemBar |
| 编程模型 | 纯 SIMD | SIMD + SIMT 混合 |
| 稀疏计算 | 支持 4:2 结构化稀疏 | 不支持（sparsity=0） |
| Subnormal | 硬件支持 | 软仿（需 config 模板参数配置） |

## AI Core 架构

### 核类型
- **CubeCore (AIC)**: 矩阵乘计算，Cube 单元 16×16×16
- **VectorCore (AIV)**: 向量计算，Regbase 架构
- `vector_core_cnt = 2 × cube_core_cnt`（每个 CubeCore 配 2 个 VectorCore）

### 芯片 Tier

| Tier | Cube 核数 | Vector 核数 | 频率 | 代表型号 |
|------|----------|------------|------|---------|
| 8-core | 8 | 16 | 1650 MHz | 910_950z |
| 28-core | 28 | 56 | 1650 MHz | 910_9572 ~ 910_957d |
| 32-core | 32 | 64 | 1650 MHz | 910_9581 ~ 910_958b |
| 36-core | 36 | 72 | 1800 MHz | 910_9591/9592/9599 |

## 存储层次

### Unified Buffer (UB)
- **总大小**: 248KB (253952 Bytes)
- **Bank 结构**: 8 bank groups × 2 banks/group = 16 banks
- **Bank 大小**: 16KB (512 rows × 32B/row)
- **对齐**: 32 Bytes
- **读写口**: 每 group 2 读 + 2 写（比 220x 多一倍）

### L0A/L0B Buffer
- **大小**: 64KB each
- **对齐**: 512 Bytes
- **分形**: L0A 采用 NZ 分形（220x 为 ZZ 分形）

### L0C Buffer
- **大小**: 256KB
- **对齐**: 64 Bytes
- **新增通路**: L0C → UB（通过 Fixpipe），L0C → GM（通过 Fixpipe NZ2DN）

### L1 Buffer
- **大小**: 512KB
- **新增通路**: UB → L1（双向），L1 → L0A/L0B

### SSBuffer (核间共享)
- **总大小**: 3KB
- **非 MIX 模式**: AIC 1KB + AIV0 1KB + AIV1 1KB（独立访问）
- **MIX 模式 (1:2)**: AIC:AIV = 1:2 共享
- **对齐**: 32B / 64B
- **用途**: AIC-AIV 核间少量数据通信

## 数据通路变更

### 删除的通路
| 通路 | 影响 | 替代方案 |
|------|------|---------|
| L1 Buffer → GM | DataCopy 不再支持 | L0C → Fixpipe → GM |
| GM → L0A/L0B | LoadData 不再直接访问 | GM → L1 → L0A/L0B（两步） |

### 新增的通路
| 通路 | 用途 | API |
|------|------|-----|
| UB ↔ L1 Buffer | 直接搬运 | DataCopy |
| L0C → UB | Fixpipe 输出到 UB | Fixpipe |
| ND-DMA | 多维数据搬运 | DataCopy (SetLoopModePara) |
| Fixpipe NZ2DN | L0C → GM 格式转换 | Fixpipe |

## Regbase 架构详述

### 编程模型
```
GM → UB → LoadAlign → RegTensor (VL=256B)
                          ↓
                     寄存器级计算 (Add/Mul/...)
                          ↓
                     StoreAlign → UB → GM
```

中间结果保留在 RegTensor 中，无需写回 UB，减少内存往返。

### 寄存器资源限制（每个 VF 函数）
| 资源 | 最大数量 | 溢出后果 |
|------|---------|---------|
| RegTensor | 32 | 编译器插入 spill/fill + sync 指令，性能下降 |
| MaskReg | 8 | 寄存器溢出，性能下降 |
| UnalignRegForLoad | 4 | 编译错误 |
| UnalignRegForStore | 4 | 编译错误 |

### 函数调用层级
```
__global__ __aicore__  (核函数入口)
    ↓
__aicore__  (设备侧函数)
    ↓
__simd_vf__ / __simt_vf__  (VF 函数，通过 VF_CALL/asc_vf_call 调用)
    ↓
__simd_callee__ / __simt_callee__  (VF 子函数)
    ↓
constexpr  (编译期函数)
```

### SIMD vs SIMT

| 维度 | SIMD | SIMT |
|------|------|------|
| 声明 | `__simd_vf__` | `__simt_vf__` |
| 子函数 | `__simd_callee__` | `__simt_callee__` |
| 数据源 | UB (LoadAlign → RegTensor) | GM 或 UB (直接访问) |
| 适用场景 | 规则、连续的逐元素操作 | 不规则、分支、动态访问 |
| 调用方式 | `VF_CALL<func>(args)` | `asc_vf_call<func>(dim3{...}, args)` |

## 同步机制

### Mutex (新增)
- 核内异步流水指令间的同步
- 类似 CPU 锁机制：锁定指定流水 → 释放完成同步依赖

### LocalMemBar (新增)
```cpp
LocalMemBar<MemType src, MemType dst>()
```
- VF 函数内的流水线同步
- MemType: VEC_STORE, VEC_LOAD, SCALAR_STORE, SCALAR_LOAD, VEC_ALL, SCALAR_ALL

### CrossCore 增强 (改进)
- AIV0 和 AIV1 可单独触发 AIC 等待
- 支持 1:1 和 1:2 模式

## 计算单元变更

### Cube
- 不支持 int4b_t (s4) 类型 → 需先 Cast 为 int8_t
- 不支持 4:2 结构化稀疏 → 使用稠密矩阵计算
- L0A 分形变化: ZZ → NZ

### Vector
- **Membase → Regbase**（核心变更）
- 基础 API 仍可使用但部分场景性能下降（编译器保守转换为 Regbase）
- 推荐使用 Reg 矢量计算 API 获得最佳性能

### 数学运算
- 硬件不支持 Subnormal → 通过 config 模板参数启用软仿
```cpp
constexpr AscendC::LnConfig CONFIG = {
    AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE  // 支持 Subnormal
};
AscendC::Ln<T, CONFIG>(dstLocal, srcLocal, count);
```

## 瓶颈分类（Evo 世界模型用）

| 瓶颈类型 | 症状 | A5 特有优化方向 |
|---------|------|---------------|
| `bandwidth` | 搬运密集、Load/Store 占比高 | VF 融合减少 Load/Store、寄存器复用 |
| `tiling` | UB 利用率低、核间不均衡 | 适配新 UB bank 结构、VL 对齐分块 |
| `algorithm` | 算法效率低 | SIMD/SIMT 混合、低延迟归约指令 |
| `register_pressure` | RegTensor 溢出、spill/fill 指令 | 布尔代数简化、指令重排 |
| `vf_fusion` | 多个 VF 未融合、冗余 Load/Store | 控制流对齐、Hardware Loop 规范 |
| `instruction_sched` | 指令依赖链长、双发射利用率低 | 循环展开、指令重排 |
