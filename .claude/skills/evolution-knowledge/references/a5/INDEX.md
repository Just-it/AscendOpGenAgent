# A5 (351x/Regbase) 知识库索引

本知识库为 A3→A5 跨架构进化优化提供参考知识，覆盖 Ascend 950PR/950DT (NpuArch=3510, AIC-C-310) 芯片。

## 目录结构

| 目录 | 内容 | 使用阶段 |
|------|------|---------|
| `hardware/` | A5 (351x) 硬件架构、Regbase 模型、UB bank 结构 | Phase 1 翻译 + Phase 2 优化 |
| `regbase_api/` | Regbase API 完整参考：RegTensor、MaskReg、AddrReg、Load/Store | Phase 1 翻译 + Phase 2 优化 |
| `vf_programming/` | VF 函数编程规范、融合规则、SIMD/SIMT 混合编程 | Phase 2 优化（evo） |
| `translation_rules/` | A3→A5 确定性翻译规则（Membase→Regbase 模式映射） | Phase 1 翻译（skill） |
| `optimization_patterns/` | A5 Regbase 优化模式：寄存器复用、双发射、循环展开 | Phase 2 优化（evo） |

## 使用指南

### Phase 1: 确定性翻译（a3-to-a5-translation skill）
1. 先读 `translation_rules/` 获取完整的 A3→A5 转换规则
2. 参考 `regbase_api/` 确认 A5 API 签名
3. 参考 `hardware/ascend950_arch.md` 确认硬件约束

### Phase 2: 探索性优化（evo agent）
1. World Model Init 时读 `hardware/ascend950_arch.md` 获取硬件参数
2. 策略选择时参考 `optimization_patterns/` 获取具体优化方案
3. 子 agent 代码生成时参考 `regbase_api/` 和 `vf_programming/`

## 硬件关键约束速查

| 参数 | 值 | 来源 |
|------|-----|------|
| Vector Length (VL) | 256 Bytes | 架构规格 |
| UB 总大小 | 248KB (253952B) | ini 配置 |
| UB Bank 结构 | 8 groups × 2 banks × 16KB | 架构规格 |
| 每个 VF 最多 RegTensor | 32 | 编译器约束 |
| 每个 VF 最多 MaskReg | 8 | 编译器约束 |
| 每个 VF 最多 UnalignRegLoad | 4 | 编译器约束 |
| 每个 VF 最多 UnalignRegStore | 4 | 编译器约束 |
| MaskReg 宽度 | VL/8 = 32B | 架构规格 |
| SIMT 最大线程数 | 2048 | 架构规格 |
| 基础对齐 | 32B (UB), 512B (L0A/L0B), 64B (L0C) | 架构规格 |
| SSBuffer 大小 | 3KB (非MIX: 1KB/核, MIX: 共享3KB) | 架构规格 |
