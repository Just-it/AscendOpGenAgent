---
id: P7
bottlenecks: [mte2_stall, undersize_transfer]
op_families: [index_scatter, matmul]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P7: 32B Alignment + DataCopyPad (数据对齐与填充)

## 核心思想

## 代码骨架

// === 改造前（基线）===
```cpp
AscendC::DataCopy(outputGm[tile_offset], outLocal, current_tile_size);
```

// === 改造后（专家模式）===
```cpp
// 专家实现: 32字节对齐优化
constexpr uint8_t ADDCDIV_LIST_BYTE_PER_BLOCK = 32;
if (uValue * sizeof(T) % ADDCDIV_LIST_BYTE_PER_BLOCK == 0) {
    DataCopy(dstLocal, tensor1Local, uValue);
} else {
    int32_t dataCountInBlock = ADDCDIV_LIST_BYTE_PER_BLOCK / sizeof(T);
    DataCopy(dstLocal, tensor1Local, (uValue + dataCountInBlock - 1) / dataCountInBlock * dataCountInBlock);
}
```

## 关键修改点

1. 预期收益: 提高内存访问效率，减少非对齐访问开销约5-10%; 最大化内存带宽利用率，提升数据拷贝效率

## 常见陷阱

⚠️ 需要复杂的参数计算
⚠️ 可能需要额外的内存开销用于对齐填充
⚠️ 需要额外计算stride参数

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
