---
id: A3
bottlenecks: []
op_families: [attention, matmul, quantization]
complexity: L0
conflicts_with: []
synergizes_with: [A1, A5]
has_preconditions: true
has_playbook: true
---

# A3: Rounding Mode Control (舍入模式控制)

## 核心思想

## 代码骨架

// === 改造前（基线）===
```cpp
// 统一使用CAST_NONE
AscendC::Cast(addedLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
AscendC::Cast(outputLocal, addedLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
```

// === 改造后（专家模式）===
```cpp
// 精细化舍入控制
// BF16: 使用RINT保证精度
Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
PipeBarrier<PIPE_V>();
Add(x1_fp32, x1_fp32, x2_fp32, numCol);
PipeBarrier<PIPE_V>();
Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);  // 输出用RINT

// FP16: 输入输出都用NONE
Cast(yLocal, xFp32, RoundMode::CAST_NONE, numCol);
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 中间计算使用截断减少指令开销，最终输出使用四舍五入提高精度

## 常见陷阱

⚠️ 需要针对不同数据类型选择不同策略，增加代码复杂度
⚠️ 增加4条指令和同步开销
⚠️ 需要额外的条件分支，可能略微影响性能

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
