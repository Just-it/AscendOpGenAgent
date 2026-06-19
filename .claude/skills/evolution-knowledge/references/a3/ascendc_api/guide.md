# API 避坑快速参考

## Top 5 致命陷阱（编译可过但结果错）

| # | 陷阱 | 症状 | 快速修复 |
|---|------|------|---------|
| 1 | 32 字节对齐违规 | 输出部分正确部分乱码 | FP16 count 必须是 16 的倍数 |
| 2 | 忘记 FreeTensor | 多 tile 后 UB 耗尽死锁 | 每个 DeQue 配一个 FreeTensor |
| 3 | PipeBarrier 缺失 | 输出全 0 或随机值 | VECTOR 写后、MTE3 读前加 PipeBarrier |
| 4 | Cast 舍入模式错误 | FP16 精度偏差大 | FP32→FP16 用 CAST_ROUND |
| 5 | 尾块越界 | 最后一批数据错误 | 最后一个 tile 用 tailLen |

## 完整列表

15 个常见陷阱详见 → [common_pitfalls.md](common_pitfalls.md)

## 何时必读

- **子 agent 写内核代码前**：必读本 guide.md
- 遇到编译通过但精度不对 → 读 common_pitfalls.md 逐项排查
