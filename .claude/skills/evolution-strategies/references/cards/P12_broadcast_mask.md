---
id: P12
bottlenecks: [ub_memory_pressure]
op_families: [broadcast_mask]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P12: Broadcast & Mask Operations (广播与掩码操作)

## 核心思想

## 代码骨架

// === 改造前（基线）===
```cpp
lrScalar = lrGm.GetValue(0);
AscendC::Muls(lrMulGradLocal, gradLocal, this->lrScalar, this->tileSize);
```

// === 改造后（专家模式）===
```cpp
// 专家实现 - 标量广播为向量
using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>;
using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
using OpLrMulGrad = Bind<Vec::Mul<T>, OpGradCast, OpLrCast>;  // 向量-向量乘法
```

## 关键修改点

1. 预期收益: 统一的向量计算流程，更高的执行效率，更好的精度控制

## 常见陷阱

⚠️ 需要额外的广播操作，但通常可以被内存访问隐藏
⚠️ Host端需要额外的广播模式识别逻辑，增加代码复杂度
⚠️ 需要额外buffer，增加计算步骤

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
