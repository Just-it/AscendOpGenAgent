# Memory Index

## 算子优化经验
- [历史探索经验积累方案](kernel-opt-framework.md) — 算子分类体系、四层隔离存储模型、复用机制与防依赖策略
- [Pad 算子优化经验](kernel-opt-pad.md) — 多 kernel 分支、维度压缩、constant 模式特化、边界映射模板

- [Repeat 算子优化经验](kernel-opt-repeat.md) — transformation-memory 类算子、逐维度串行处理、constexpr 循环展开、多核分区策略
## 完整代码归档（Layer 4，Agent 默认不可读）
- [归档目录说明](archive/README.md) — 读取约束、归档规则、目录结构
- [Pad](archive/pad/pad_v1_20260522.py) — 1.68x，51/51；[R](archive/pad/pad_v1_20260522_report.md)/[S](archive/pad/pad_v1_20260522_summary.json)
- [Repeat](archive/repeat/repeat_v2_20260526.py) — 0.88x，49/49 pass
