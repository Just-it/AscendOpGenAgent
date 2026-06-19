# Host 侧迁移规则

## H-01: UB 分配计算

### A3 (220x)
```
UB: 192KB, 16 groups × 3 banks × 4KB
tileSize = (ubSize / BUFFER_NUM / pipeCount / sizeof(T)) / 32 * 32
```

### A5 (351x)
```
UB: 248KB, 8 groups × 2 banks × 16KB
tileSize = (ubSize / BUFFER_NUM / pipeCount / sizeof(T)) / 32 * 32
```

**变更点**:
- UB 总大小从 192KB 变为 248KB → tileSize 可以更大
- Bank 结构变化 → 需重新评估 bank 冲突
- 建议 tile 粒度对齐到 VL (256B) 以获得最佳 Regbase 性能

### Bank 冲突避免

A5 UB: 8 bank groups × 2 banks × 16KB

**冲突类型**:
- 读写冲突: 读写同一 bank
- 写写冲突: 多个写操作同一 bank group
- 读读冲突: 两个读操作同一 bank，或 2+ 读操作同一 bank group

**优化策略**:
```cpp
// 避免 bank 冲突: 扩大内存申请，错开 bank group
pipe.InitBuffer(inQueueX, 1, tileSize * sizeof(T) + 256);  // 多申请 256B
pipe.InitBuffer(inQueueY, 1, tileSize * sizeof(T));
```

## H-02: Tile 对齐

### A3
- 32B 对齐即可

### A5
- 32B 对齐（基本要求不变）
- **建议**: tile 大小对齐到 VL/sizeof(T) 的整数倍
  - float: 对齐到 64 元素 (256B / 4B)
  - half: 对齐到 128 元素 (256B / 2B)
  - int8: 对齐到 256 元素 (256B / 1B)

### Tiling 参数新增
```cpp
struct TilingData {
    // 原有字段
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t lastTileLength;
    // A5 新增字段
    uint32_t oneRepeatSize;   // VL / sizeof(T)
    uint16_t repeatTimes;     // ceil(tileLength / oneRepeatSize)
};
```

## H-03: 搬运路径

### 删除的路径
| A3 路径 | A5 替代方案 |
|---------|-----------|
| L1 Buffer → GM | L0C → Fixpipe → GM |
| GM → L0A/L0B | GM → L1 → L0A/L0B（两步） |

### 新增的路径
| A5 新路径 | 用途 |
|----------|------|
| UB ↔ L1 Buffer | 直接搬运 |
| L0C → UB | Fixpipe 输出到 UB |
| ND-DMA | 多维数据搬运 |

### 代码迁移示例

```cpp
// A3: L1 → GM (直接)
DataCopy(gmTensor, l1Tensor, count);

// A5: L0C → Fixpipe → GM (间接)
// 需要通过 Mmad 输出到 L0C，再用 Fixpipe 搬到 GM
FixpipeParamsV220 fixpipeParams;
// ... 配置参数 ...
Fixpipe<S, S, CFG_ROW_MAJOR>(gmTensor, l0cTensor, fixpipeParams);
```

## H-04: Tiling 参数约束

### A3 约束
- UB 容量
- 32B 对齐
- 核数

### A5 新增约束
- **寄存器数量**: 每个 VF 最多 32 个 RegTensor
  - 如果一个 tile 内需要多个中间结果，需确保 RegTensor 总数 ≤ 32
- **VL 对齐**: tile 大小建议为 VL/sizeof(T) 的整数倍
- **UB bank 结构**: 8 groups × 2 banks × 16KB

### Tiling 计算公式

```python
# A5 tiling 计算
VL = 256  # bytes
VL_T = VL // sizeof_T  # 每个 RegTensor 的元素数

# 基本 tile 大小（对齐到 VL_T）
max_tile = (ub_size // buffer_num // pipe_count // sizeof_T)
tile_size = (max_tile // VL_T) * VL_T  # 对齐到 VL_T

# repeatTimes
repeat_times = tile_size // VL_T
```

## 编译配置迁移

### CMakeLists.txt
```cmake
# A3
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-c220>
)

# A5
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-c310>
)
```

### OpDef
```cpp
// A3
this->AICore().AddConfig("ascend910b");

// A5
this->AICore().AddConfig("ascend910_95xx");  // 具体型号按实际填写
```
