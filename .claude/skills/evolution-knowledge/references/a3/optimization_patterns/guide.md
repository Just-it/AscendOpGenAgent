# 优化模式快速参考

## 模式选择决策树

```
内核是否内存密集？
  ├─ 是 → 已有双缓冲？
  │       ├─ 否 → double_buffering.md (20-80% 提升)
  │       └─ 是 → tiling_strategies.md (10-50%) + pipeline_overlap.md (5-30%)
  └─ 否（计算密集或平衡）
      ├─ 有因果/掩码逻辑？ → causal_block_skip.md (20-50%)
      ├─ 有跨步访问？ → memory_coalescing.md (10-40%)
      └─ 以上都不是 → 需要算法级优化，转 algorithm_insights/
```

## 模式速查表

| 模式 | 文件 | 适用场景 | 典型提升 |
|------|------|---------|---------|
| 双缓冲 | `double_buffering.md` | 所有内存密集型内核 | 20-80% |
| 自适应分块 | `tiling_strategies.md` | 可变形状算子 | 10-50% |
| 因果块跳过 | `causal_block_skip.md` | Attention + causal mask | 20-50% |
| 流水线重叠 | `pipeline_overlap.md` | 已有双缓冲的内核 | 5-30% |
| 内存合并 | `memory_coalescing.md` | 跨步访问模式 | 10-40% |
