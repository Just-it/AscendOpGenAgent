---
id: P10
bottlenecks: [mte2_stall, no_overlap, undersize_transfer]
op_families: [omni]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P10: Vectorized Data Copy & Instructions (向量化数据与指令)

## 核心思想

## 代码骨架

// === 改造后（专家模式）===
```cpp
template <typename T>
__aicore__ inline void TransposeBase16M8(LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, 
                                         uint64_t rowNum, uint64_t colNum) {
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];
    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}

if constexpr (is_same<TGrad, float>::value) {
    TransposeBase16M8(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign8);
} else {
    TransposeBase16M16(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign16);
}
```

## 关键修改点

1. 预期收益: 向量化转置可将内存访问模式优化为连续访问，减少bank conflict，提升Vector指令效率4-8倍

## 常见陷阱

⚠️ 需要额外的UB buffer存储转置数据
⚠️ 增加一层函数调用开销(通常可内联)
⚠️ 对于单次运算略显复杂

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|SyncAll" op_kernel/*.cpp op_host/*_tiling.cpp
```
