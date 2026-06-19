---
name: evolution-knowledge
description: Domain knowledge base for AscendC evolution optimization covering hardware architecture, algorithm insights, API pitfalls, optimization patterns, and proven solutions for both A3 (910B) and A5 (950) architectures
---

# Evolution Knowledge Base

AscendC 进化优化领域知识库，覆盖硬件架构、算法洞察、API 陷阱、优化模式和验证方案。支持 A3 (Ascend 910B) 和 A5 (Ascend 950) 双架构。

## 文件布局

```
references/
  a3/                           # Ascend 910B (Membase/A3) 知识
    guide.md                    # 顶层快速入口
    hardware/
      guide.md                  # 瓶颈诊断启发式、tiling 公式
      ascend910b_arch.md        # 架构详情：UB=196KB, 40核, DMA 模型
    algorithm_insights/
      guide.md                  # 算法优化原则
      attention_family.md       # MHA/GQA/MQA 优化
      reduction_ops.md          # Norm/Softmax: two-pass, Welford
      elementwise_fusion.md     # 算子融合
    ascendc_api/
      guide.md                  # Top 5 致命陷阱（快速诊断）
      common_pitfalls.md        # 15 个常见陷阱详解
    optimization_patterns/
      guide.md                  # 决策树：bandwidth vs algorithm vs register
      double_buffering.md       # 20-80% 增益
      tiling_strategies.md      # 10-50% 增益
      causal_block_skip.md      # 20-50% (attention)
      pipeline_overlap.md       # 5-30% 增益
      memory_coalescing.md      # 10-40% (strided access)
    proven_solutions/
      INDEX.md                  # 已验证方案索引（可追加写入）

  a5/                           # Ascend 950 (Regbase/A5) 知识
    INDEX.md                    # Phase 1 translation vs Phase 2 evolution 指南
    hardware/
      ascend950_arch.md         # 950PR/DT: Regbase, UB 248KB, 8 banks
    regbase_api/
      reg_tensor.md, load_store.md, addr_reg.md, mask_reg.md, local_mem_bar.md
    vf_programming/
      simd_vf_basics.md, hardware_loop.md, vf_fusion_rules.md, simd_simt_hybrid.md
    optimization_patterns/
      register_reuse.md, instruction_dual_issue.md, loop_unrolling.md, low_latency_reduce.md
    translation_rules/
      membase_to_regbase.md, data_path_migration.md, host_side_migration.md, sync_migration.md
```

## 渐进式查询协议

知识库采用**渐进式披露**（progressive disclosure）：先读 guide.md 快速参考，按需深入。

### 按工作流阶段查询

| 阶段 | 必读 | 按需 |
|---|---|---|
| Init (世界模型初始化) | `a3/hardware/guide.md`, `a3/optimization_patterns/guide.md` | `a3/algorithm_insights/{family}.md` |
| Sub-agent 代码生成前 | `a3/ascendc_api/guide.md` (Top 5 致命陷阱) | `a3/optimization_patterns/*.md` (按策略类型) |
| Refine (新颖性评估) | `a3/proven_solutions/INDEX.md` | — |
| A5 翻译 | `a5/translation_rules/*.md` | `a5/hardware/ascend950_arch.md` |
| A5 进化优化 | `a5/optimization_patterns/*.md` | `a5/vf_programming/*.md` |

### 按 Agent 角色查询

| Agent | 必读 | 可选 |
|---|---|---|
| ops-evo | hardware + optimization_patterns guide.md | algorithm_insights (匹配算子族) |
| ops-partial | ascendc_api/guide.md | optimization_patterns (匹配策略类型) |

## 检索优先级链

knowledge_base → evolution-strategies → proven_solutions → WebSearch

## 写入协议

- `a3/proven_solutions/INDEX.md`: 当策略提炼出通用方案时，追加条目
- 其他文件为只读参考

## 架构选择

- 目标芯片 Ascend 910B → 使用 `references/a3/`
- 目标芯片 Ascend 950 → 使用 `references/a3/`（基础知识）+ `references/a5/`（架构扩展）
