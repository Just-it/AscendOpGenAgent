---
id: D1
bottlenecks: [compute_bound]
op_families: [omni]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# D1: Mixed Precision Architecture (混合精度架构)

## 核心思想

## 代码骨架

// === 改造前（基线）===
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

// === 改造后（专家模式）===
```cpp
this->Input("x")
    .ParamType(DYNAMIC)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
this->Input("scalar")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

## 关键修改点

1. 预期收益: 支持更广泛的业务场景，减少数据类型转换开销，提高端到端性能; 覆盖更广的精度需求，支持2-4倍内存带宽优化，适应不同硬件平台和精度要求; 一个算子支持多种数据类...

## 常见陷阱

⚠️ 代码复杂度增加，需要维护多类型模板实例
⚠️ 模板代码复杂度增加，编译时间可能增加
⚠️ 需要在运行时检查属性，略微增加开销

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
