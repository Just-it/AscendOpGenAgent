---
id: A7
bottlenecks: []
op_families: [index_scatter, pooling_gather]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# A7: Index & Boundary Safety (索引与边界安全处理)

## 核心思想

## 代码骨架

// === 改造前（基线）===
```cpp
this->Input("indices")
    .DataType({ge::DT_INT64})
```

// === 改造后（专家模式）===
```cpp
this->Input("argmax")
    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
```

## 关键修改点

1. 预期收益: 减少50%索引数据传输带宽，提升内存受限场景性能

## 常见陷阱

⚠️ 索引范围限制在2^31-1以内（足够覆盖实际场景）
⚠️ 每次迭代增加一次比较操作
⚠️ 需要上层框架保证索引合法性

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
