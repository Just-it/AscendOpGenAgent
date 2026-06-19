---
id: A4
bottlenecks: []
op_families: [attention, elementwise, flash_attention]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# A4: SetFlag/WaitFlag Event Sync (事件同步保证精度)

## 核心思想

## 代码骨架

// === 改造后（专家模式）===
```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
PipeBarrier<PIPE_V>();
op(float32Tensor[offset], float32Tensor, scalarVal, dataCount);
PipeBarrier<PIPE_V>();
Cast(outLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 确保计算顺序正确性，防止数据竞争，保证数值精度

## 常见陷阱

⚠️ 引入同步开销，可能降低流水线效率
⚠️ 增加了同步开销，但精度收益显著
⚠️ 同步会引入轻微性能损失，但对于正确性是必需的

## 代码搜索关键词

```bash
grep -n "SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|SyncAll\|SetFlag\|WaitFlag\|PipeBarrier" op_kernel/*.cpp op_host/*_tiling.cpp
```
