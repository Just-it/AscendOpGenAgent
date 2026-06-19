# 世界模型 JSON Schema 定义

本文件定义了世界模型的完整 JSON 数据结构。世界模型以文件形式持久化于磁盘：

```
output/{op_name}_evo_{timestamp}/world_model.json
```

---

## 顶层结构

```json
{
  "kernel_summary": "<字符串> 算子功能、计算模式、已知约束的一段描述（1-3句话）",

  "session": {
    "session_id": "ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933",
    "start_time": "2026-04-30T20:29:33+0800",
    "requested_rounds": 5,
    "actual_rounds_completed": 3,
    "evo_dir": "output/ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933",
    "op_name": "ai_infra_sparse_flash_attention_gqa"
  },
  // session: session 级身份锚定，防止 agent 失忆后误用历史目录
  //   session_id: 唯一标识，格式为 {op_name}_ops-evo_{timestamp}
  //   start_time: ISO8601 时间戳
  //   requested_rounds: 用户请求的最大轮数
  //   actual_rounds_completed: 实际已完成的轮数（由 refine 自动更新）
  //   evo_dir: 本次进化的根目录绝对路径
  //   op_name: 算子名称

  "baseline_performance": {
    "speedup": 1.0,       // 基线加速比（描述模式默认1.0，基线模式填入实际测量值）
    "time_ms": null       // 基线执行时间（ms），可为null
  },

  "decision_tree": {
    "nodes": {
      "<node_id>": { ... }   // 节点字典，key为节点ID
    }
  },

  "open_questions": [
    "<字符串> 当前未解答的性能假设或待验证的优化方向"
  ],

  "stagnation_count": 0,         // 连续无显著提升的轮数计数器（vs 全局 best_score×1.02）
  "stagnation_count_vs_base": 0, // 连续本轮最佳未超越父节点得分的轮数计数器（K-Search 风格）
  "best_score": 1.0,             // 迄今为止所有已评测节点中的最高 speedup 值
  "world_model_active": true,    // false 表示世界模型初始化失败，已回退到 tiered sampling
  "solution_db_path": null,      // solution_db.jsonl 的相对路径（血统追踪，可选）
  "hw_params": null,
  "discovered_strategies": [],
  "baseline_evidence": null,
  // 基线 profiling 的硬映射证据（可选，根级字段）。
  // 由 wm_ops.py attach-baseline-evidence 在跑完 baseline_evaluation 后写入；
  // 结构与节点的 profiling_evidence 完全一致（见节点结构）。
  // 非 null 时，用于：
  //   1) select_nodes 中触发 baseline 对齐惩罚（见 compute_utility w_baseline_mismatch）
  //   2) 注入 ops-partial prompt 的 [Profiling Context] 作为 Baseline 行
  // null 表示 baseline pipeline 不可用或 attach-baseline-evidence 未被调用，
  // 系统回退到不做对齐惩罚、prompt 中跳过 Baseline 行的行为

  "target_speedup": 1.10,
  // 用户在 Step1 提供的目标加速比（multi-shape 模式下作为硬门控阈值，
  // 每个 target shape 必须 ≥ 此值才算 fully_passed）。单 shape 场景同样使用。

  "shape_targets": null
  // [multi-shape only] 顶层 shape 配置概览，单 shape 退化时可为 null。
  // 非 null 时结构：
  // {
  //   "target_count": 2,                      // target_shapes 数量
  //   "generalization_count": 3,              // generalization_shapes 数量
  //   "target_speedup_threshold": 1.10,       // 与顶层 target_speedup 同步
  //   "generalization_floor": 1.0             // 泛化几何平均下限（软门控）
  // }
  // 与节点字段 gating / shape_results / aggregate 配套使用。SELECT/REFINE 通过
  // 此结构判断进化期目标达成（gating==fully_passed）与泛化退化兜底。
}
```

---

## 节点结构（decision_tree.nodes 中的每个节点）

```json
{
  "id": "<字符串> 节点唯一标识符（如 'root', 'n1', 'n2', 'n1_1'）",

  "parent_id": "<字符串> 父节点ID，根节点为 null>",

  "description": "<字符串> 该节点的优化方向描述（1-2句话，说明用什么策略解决什么瓶颈）",

  "strategy_combination": ["P1", "P7"],
  // 适用于此节点的策略ID列表（对应 .claude/skills/evolution-strategies/references/strategy-index.md 中的ID）
  // 根节点和自由探索节点为 []

  "mode": "strategy_guided",
  // 节点探索模式，取值：
  //   "strategy_guided"   — 默认：使用 strategy_combination 中的策略，或从策略库自由选择
  //   "open_exploration"  — 开放探索：禁止读取策略库，LLM 从最优内核代码和 profiling 自主推理新优化方向
  //   "profiling_driven"  — Profiling驱动：基于父节点的 profiling_insight/profiling_evidence 中的具体瓶颈诊断，
  //                         由 LLM 自主设计针对性优化方案。不限于策略库，但必须针对诊断出的具体瓶颈。
  //                         与 open_exploration 的区别：profiling_driven 有明确的瓶颈靶点（来自profiling数据），
  //                         而 open_exploration 是完全自由的第一原理推理。
  //                         strategy_combination 字段保留为空数组 []，优化方向完全由 description 中的
  //                         profiling 瓶颈描述指导。

  "optimization_type": "bandwidth",
  // 性能优化类型标签，取值: "bandwidth" | "tiling" | "algorithm" | "register_opt" | "vf_fusion" | "instruction_sched"
  // - bandwidth: 带宽/搬运优化（P1双缓冲, P7对齐, P10向量化搬运, P11尾块, R5非对齐优化）
  // - tiling: 分块/并行优化（P2自适应分块, P4多核均衡, P5流水线同步, P8 UB分区）
  // - algorithm: 算法级重构（P13高层API, open_exploration, R6低延迟归约, R7 SIMD/SIMT混合）
  // - register_opt: [A5专用] 寄存器优化（R2寄存器复用、减少RegTensor数量避免spill/fill）
  // - vf_fusion: [A5专用] VF函数融合（R1多VF合并、减少冗余Load/Store）
  // - instruction_sched: [A5专用] 指令调度优化（R3双发射、R4 Hardware Loop规范、R8 Mutex同步）
  // 由 Init 和 Refine 自动推导，用于 Select 阶段的多样性保底约束
  // D/A 系列策略不参与类型推导（它们是精度约束，非性能方向）
  // A5 (351x/Regbase) 架构新增 register_opt/vf_fusion/instruction_sched 三个类型，
  // 对应 strategy-index-a5.md 中的 R 系列策略

  "status": "open",
  // 节点状态，取值：
  //   "open"        — 尚未执行，可被选择
  //   "in_progress" — 当前轮次已选中，正在执行
  //   "passed"      — 评测通过（编译成功 + 精度正确）
  //   "failed"      — 评测失败（编译失败 或 精度不通过）
  //   "completed"   — 特殊状态，仅用于根节点（作为起始点使用）

  "score": null,
  // 节点的 speedup 值（评测通过后填入，失败节点保持 null）
  // **multi-shape 模式下含义为 min(target shape speedups)**（保守聚合 —
  // score 反映最差 target shape 的提速程度）。
  // 单 shape 模式下即为该 shape 的 speedup。

  "difficulty": 2,
  // 实现难度估计（1-5整数），失败节点设为5
  // 用于效用函数计算，难度越高优先级越低

  "depth": 1,
  // 节点在决策树中的深度（根节点为0，子节点递增）

  "solution_ref": null,
  // 评测通过后填入该变体的输出路径（如 "round_1/parallel_0"）
  // 失败或未执行节点为 null

  "parent_code_ref": null,
  // 父节点实现的 kernel 路径（由 wm_ops.py 从 solution_db 填充，可选）
  // 格式示例："round_1/parallel_0/{op_name}Custom/op_kernel/{op_name}_custom.cpp"

  "children": [],
  // 子节点ID列表（评测通过且有实质提升时生成子节点，子节点ID追加到此列表）

  "failure_type": null,
  // 失败节点的失败原因分类（仅 status="failed" 时有意义，其他节点为 null）：
  //   "impl_error"         — 实现错误：策略方向本身可行，但子agent代码写错了
  //                         （如：A6 Rsqrt未做NR迭代、Welford未用FP64累积）
  //                         → 该方向仍值得重试，difficulty 不设为5
  //   "strategy_infeasible" — 策略不可行：策略本身与该算子的精度/约束不兼容
  //                         （如：近似误差量级合理但仍超阈值）
  //                         → 封锁该方向，difficulty 设为5

  "failure_reason": null,
  // 一句话说明失败诊断依据（供 Analyze 和子节点 description 引用）
  // 示例："rmse=25.79 >> 1，判断为 Rsqrt 未做 NR 迭代所致（impl_error）"

  "retry_count": 0
  // 该策略方向已重试次数（impl_error 节点的修复子节点继承 parent.retry_count + 1）
  // retry_count >= 2 时即使 impl_error 也不再生成子节点（避免无限重试）

  "profiling_insight": null,
  // 性能瓶颈分析（可选字段，精度通过后由 ops-evo 步骤 4.4.P 填入）
  // 非 null 时结构：{ "bottleneck": "memory_bound", "recommended_strategies": ["P1","P2"],
  //                   "profiling_one_liner": "MTE2:45% | Vec:37% | ..." }
  // null 表示尚未分析或 profiling 数据不存在

  "profiling_evidence": null,
  // 指令级深度空泡分析（可选字段，由 ops-evo 步骤 4.4.P2 在触发条件满足时填入）
  // 数据来源：ascendc-profiling-analysis skill 的 T1/T2/T3 分析结果
  // ⚠️ v3.2 起本字段为兼容 alias，新代码请用 facts + diagnosis + candidate_sources 三件套
  // 非 null 时结构：
  // {
  //   "bottleneck_type": "mte2_stall",          // 瓶颈分类: mte2_stall / mte3_stall / tiling_imbalance / scalar_loading / scalar_compute / compute_bound / near_optimal / no_overlap / partial_overlap / undersize_transfer / icache_miss / bus_contention
  //   "d_class_pct": 62.5,                      // D类空泡占比（跨流水线同步等待）
  //   "c_class_pct": 15.3,                      // C类空泡占比（标量参数加载阻塞）
  //   "imbalance_ratio": 1.12,                  // 跨核负载不均衡比（>1.3 为不均衡）
  //   "primary_bottleneck": "MTE2",             // D类空泡的主等待流水线
  //   "suggested_strategies": ["P1", "P10"],    // 基于空泡分析推荐的策略
  //   "anti_strategies": ["P3"],                // 应避免的策略
  //   "description": "MTE2 数据加载等待是主瓶颈...",
  //   "top_recommendation": "启用双缓冲隐藏MTE2搬运延迟",
  //   "pattern_type": "periodic",               // 空泡模式: periodic / sporadic / cold_start_dominant / tail_dominant (T7)
  //   "overlap_status": "no_overlap",           // 流水线重叠状态: no_overlap / partial_overlap / good_overlap (T8)
  //   "dominant_subtype": "D_MTE2_WAIT",        // 主导空泡子类型 (二级分类)
  //   "dma_efficiency": {                       // DMA 搬运效率 (T9)
  //     "mte2_short_pct": 35.0,                 // MTE2 短搬运占比
  //     "mte3_short_pct": 20.0                  // MTE3 短搬运占比
  //   }
  // }
  // null 表示深度分析未触发、trace 数据不存在或分析失败
  // 与 profiling_insight 的关系：profiling_insight 是 CSV 级快速诊断（每轮必做），
  // profiling_evidence 是 trace 级深度分析（仅在特定条件触发时执行），两者互补

  // ═══ v3.2 新增三字段（三阶段诊断管线） ═══

  "facts": null,
  // Stage 1: 纯事实抽取（由 profiling_evidence.extract_facts() 在 refine 阶段写入）
  // 字段：
  //   _source           : "deep_profiling" / "pipeline_csv"
  //   dominant_pipe     : "mte2" / "mte3" / "vec" / "scalar" / null
  //   mte2_ratio        : 0~1
  //   mte3_ratio        : 0~1
  //   vec_ratio         : 0~1
  //   scalar_ratio      : 0~1
  //   bw_utilization    : 0~1（真假 MTE2 bound 区分依据）
  //   bank_cflt_ratio   : 0~1
  //   l2_hit_rate       : 0~1
  //   icache_miss_rate  : 0~1
  //   imbalance_ratio   : >=1
  //   imbalance_pct     : >=0
  //   pattern_type      : 仅 deep_profiling 才有
  //   overlap_status    : "no_overlap" / "partial_overlap" / "good_overlap" / null
  //   dominant_subtype  : 仅 deep_profiling 才有
  //   dma_efficiency    : {} 或 deep profiling 细节
  // null 表示无 profiling 数据可用

  "diagnosis": null,
  // Stage 2: LLM narrative 诊断（由 refine 阶段主 agent prompt 输出）
  // 必填字段（wm_ops refine 会校验）：
  //   diagnosis_text     : 自由文本，≥ 20 字符
  //   bottleneck_labels  : 字符串列表，每项 ∈ KNOWN_BOTTLENECK_LABELS（18 个，
  //                        见 profiling_evidence.py 或 evolution-knowledge#a3/profiling_reference/INDEX.md）
  //   confidence         : 0~1 float
  // 可选字段：
  //   root_cause_hypothesis : 根因假设描述
  //   evidence_refs         : 引用的 facts 字段名（如 ["bw_utilization", "mte2_ratio"]）
  // 校验失败 → wm_ops.refine 拒收，强制重做

  "candidate_sources": null,
  // Stage 3: bottleneck_to_sources 反查结果（由 wm_ops finalize-ledger 阶段回填到**自身节点**）
  // 写入时机：LLM 离线写完 node.diagnosis 后，agent 调用 `wm_ops finalize-ledger`，
  // 该命令 Step 4 扫描所有 diagnosis != null AND candidate_sources is None 的节点，
  // 调 match_strategies_by_labels(diagnosis.bottleneck_labels) 回填。幂等。
  // 用途：纯审计/可解释性日志，供 agent 检视该节点诊断对应的策略候选池。
  // ⚠️ utility 计算**不依赖**此字段——compute_utility 在 select 阶段从
  //   parent.facts × parent.diagnosis.labels × node.strategy_combination
  //   现算 expected_gain（避免父 diagnosis 改写后字段 stale）。
  // 结构：
  //   candidate_source_keys : [str] 按命中 label 数降序
  //   candidate_ids         : [str] 同上的 strategy ID
  //   by_label              : {label: [ids]}
  //   unknown_labels        : [str] (若有 Stage 2 给了词表外的 labels)
  // null 表示 Stage 3 尚未执行或本节点无 diagnosis


  "gating": null,
  // [multi-shape only] 评测综合判定，五枚举之一（由 evaluate_ops_direct.py 写入
  // evaluation_results.json 后被 REFINE 转录到节点）：
  //   "failed"                    — 编译 / 精度 / 运行错（status=failed）
  //   "target_regression"         — 任一 target shape speedup < 1.0x；
  //                                 status=passed 但 parent_eligible=false，
  //                                 REFINE 自动给子节点 strategy_combination 注入
  //                                 P-ShapeSpec-01
  //   "generalization_regression" — 仅 fully_passed 候选触发的二次泛化验证后写入；
  //                                 进化期主流程不直接产出此 gating
  //   "fully_passed"              — 所有 target shape ≥ target_speedup 且无泛化退化
  //                                 → 候选最终变体，进入 §4.6.5 泛化验证流程
  //   "partial_passed"            — 所有 target ≥ 1.0x 但部分未达 target_speedup
  //                                 → status=passed、parent_eligible=true，
  //                                 仍可作 parent（partial_passed 宽限退路）
  // 单 shape 模式（legacy evaluation_results.json）不写 gating 字段

  "parent_eligible": true,
  // [multi-shape only] 该节点能否被 SELECT 选为下一轮 parent。
  //   true  : 默认；status=passed 且 gating ∈ {partial_passed, fully_passed} 时为 true
  //   false : gating == "target_regression" 时强制 false，SELECT 屏蔽其 open 子节点
  // 旧节点（无 aggregate / gating 字段）SELECT 默认视为 eligible，向后兼容

  "target_shape_regression": false,
  // [multi-shape only] 是否任一 target shape speedup < 1.0x
  // = aggregate.any_target_regression
  // REFINE 据此字段在子节点 strategy_combination 中自动注入 P-ShapeSpec-01

  "shape_results": null,
  // [multi-shape only] per-shape 详细评测数据，直接从 evaluation_results.json
  // 复制。结构：
  // {
  //   "target": [
  //     {"name": "T1", "baseline_time_us": 123.4, "evolved_time_us": 100.0,
  //      "speedup": 1.234, "precision_passed": true, "compilation_success": true,
  //      "pipeline": {...}, "bottleneck": "compute_bound", "cv_pct": 1.2}, ...
  //   ],
  //   "generalization": [...]   // 进化期为空；fully_passed 候选验证后填充
  // }

  "aggregate": null
  // [multi-shape only] 聚合统计字段（直接从 evaluation_results.json 复制）：
  // {
  //   "target_min_speedup": 1.05,           // = 节点 score
  //   "target_geo_mean_speedup": 1.18,
  //   "target_max_speedup": 1.40,
  //   "all_target_meet_target": false,      // 所有 target ≥ target_speedup
  //   "all_target_above_baseline": true,    // 所有 target ≥ 1.0x（= parent_eligible）
  //   "any_target_regression": false,       // = target_shape_regression
  //   "generalization_geo_mean_speedup": null,
  //   "any_generalization_regression": null
  // }
  // shape_divergence = (target_max_speedup - target_min_speedup) / target_max_speedup
  // 由 compute_utility 在线计算；divergence ≥ 0.20 且子节点带 P-ShapeSpec-01 时
  // utility +1.0（鼓励主动分支化）
}
```

---

## 节点 ID 命名规范

- 根节点：`"root"`
- 第一层子节点：`"n1"`, `"n2"`, `"n3"`, ...
- 深层子节点：`"n1_1"`, `"n1_2"`, `"n2_1"`, ...（父节点ID + 下划线 + 序号）

---

## 完整示例（初始化后的世界模型）

```json
{
  "kernel_summary": "FastGELU算子，对输入张量逐元素应用GELU激活函数。计算密集度低，主要为内存带宽瓶颈。支持FP16/BF16，形状可变，存在尾块对齐问题。",

  "baseline_performance": {
    "speedup": 1.0,
    "time_ms": null
  },

  "decision_tree": {
    "nodes": {
      "root": {
        "id": "root",
        "parent_id": null,
        "description": "基线内核，未应用优化策略",
        "strategy_combination": [],
        "status": "completed",
        "score": 1.0,
        "difficulty": 1,
        "depth": 0,
        "solution_ref": null,
        "children": ["n1", "n2", "n3", "n4", "n5", "n6"]
      },
      "n1": {
        "id": "n1",
        "parent_id": "root",
        "description": "双缓冲流水线 + 32字节对齐，通过计算与数据搬运重叠提升吞吐量",
        "strategy_combination": ["P1", "P7"],
        "status": "open",
        "score": null,
        "difficulty": 2,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n2": {
        "id": "n2",
        "parent_id": "root",
        "description": "自适应分块 + 多核负载均衡，针对不同输入形状自动选择最优分块策略",
        "strategy_combination": ["P2", "P4"],
        "status": "open",
        "score": null,
        "difficulty": 3,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n3": {
        "id": "n3",
        "parent_id": "root",
        "description": "向量化数据搬运 + 尾块GatherMask处理，最大化内存带宽利用率",
        "strategy_combination": ["P10", "P11"],
        "status": "open",
        "score": null,
        "difficulty": 3,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n4": {
        "id": "n4",
        "parent_id": "root",
        "description": "混合精度架构 + FP32中间计算，FP16输入使用FP32中间值保证精度",
        "strategy_combination": ["D1", "A1"],
        "status": "open",
        "score": null,
        "difficulty": 2,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n5": {
        "id": "n5",
        "parent_id": "root",
        "description": "双缓冲 + 自适应分块组合，综合提升流水线效率和分块适应性",
        "strategy_combination": ["P1", "P2"],
        "status": "open",
        "score": null,
        "difficulty": 3,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n6": {
        "id": "n6",
        "parent_id": "root",
        "description": "UB内存分区优化 + 流水线同步，精细化UB使用减少等待",
        "strategy_combination": ["P8", "P5"],
        "status": "open",
        "score": null,
        "difficulty": 4,
        "depth": 1,
        "solution_ref": null,
        "children": []
      }
    }
  },

  "open_questions": [
    "该算子是内存带宽瓶颈还是计算密集型？需要通过profiling确认",
    "不同输入形状（大shape vs 小shape）对最优分块策略的影响程度？",
    "尾块对齐问题是否对整体性能有显著影响？"
  ],

  "stagnation_count": 0,
  "stagnation_count_vs_base": 0,
  "best_score": 1.0,
  "world_model_active": true,
  "hw_params": null,
  "discovered_strategies": [],
  "baseline_evidence": null
}
```

---

## 完整示例（两轮进化后的世界模型）

```json
{
  "kernel_summary": "FastGELU算子，内存带宽瓶颈已确认，双缓冲效果显著。尾块对齐影响较小。",

  "baseline_performance": {
    "speedup": 1.0,
    "time_ms": 0.5
  },

  "decision_tree": {
    "nodes": {
      "root": {
        "id": "root", "parent_id": null, "description": "基线内核",
        "strategy_combination": [], "status": "completed", "score": 1.0,
        "difficulty": 1, "depth": 0, "solution_ref": null,
        "children": ["n1", "n2", "n3", "n4", "n5", "n6"]
      },
      "n1": {
        "id": "n1", "parent_id": "root",
        "description": "双缓冲流水线 + 32字节对齐",
        "strategy_combination": ["P1", "P7"],
        "status": "passed", "score": 2.3, "difficulty": 2, "depth": 1,
        "solution_ref": "round_1/parallel_0",
        "children": ["n1_1", "n1_2"]
      },
      "n1_1": {
        "id": "n1_1", "parent_id": "n1",
        "description": "在n1基础上增加向量化数据搬运，进一步提升带宽利用率",
        "strategy_combination": ["P1", "P7", "P10"],
        "status": "open", "score": null, "difficulty": 3, "depth": 2,
        "solution_ref": null, "children": []
      },
      "n1_2": {
        "id": "n1_2", "parent_id": "n1",
        "description": "在n1基础上增加多核负载均衡，分摊计算压力",
        "strategy_combination": ["P1", "P7", "P4"],
        "status": "open", "score": null, "difficulty": 3, "depth": 2,
        "solution_ref": null, "children": []
      },
      "n2": {
        "id": "n2", "parent_id": "root",
        "description": "自适应分块 + 多核负载均衡",
        "strategy_combination": ["P2", "P4"],
        "status": "passed", "score": 1.7, "difficulty": 3, "depth": 1,
        "solution_ref": "round_1/parallel_1",
        "children": []
      },
      "n3": {
        "id": "n3", "parent_id": "root",
        "description": "向量化数据搬运 + 尾块GatherMask处理",
        "strategy_combination": ["P10", "P11"],
        "status": "failed", "score": null, "difficulty": 5, "depth": 1,
        "solution_ref": null, "children": []
      }
    }
  },

  "open_questions": [
    "双缓冲（P1）是关键优化，所有变体都应包含它作为基础",
    "尾块处理（P11）导致编译失败，暂时避免单独使用P11",
    "n1分支值得深度探索：在P1+P7基础上叠加P10或P4的效果？"
  ],

  "stagnation_count": 0,
  "stagnation_count_vs_base": 0,
  "best_score": 2.3,
  "world_model_active": true,
  "hw_params": {
    "chip_model": "910B3",
    "ub_size_bytes": 196608,
    "core_num": 40,
    "peak_bw_gbps": 57.6,
    "peak_vector_tflops_per_core": 0.2304,
    "alignment_bytes": 32,
    "max_tile_fp16_double_buf": 16384
  },
  "discovered_strategies": ["X1"]
}
```

---

## 字段约束

| 字段 | 类型 | 约束 |
|------|------|------|
| `status` | string | 必须是 open/in_progress/passed/failed/completed 之一 |
| `score` | float 或 null | 通过后为正数，失败或未执行为 null |
| `difficulty` | int | 1-5 整数，失败节点强制设为 5 |
| `depth` | int | 非负整数，根节点为 0 |
| `strategy_combination` | string[] | 元素必须是 strategy-index.md 中存在的策略 ID |
| `failure_type` | string 或 null | 仅失败节点有值："impl_error" 或 "strategy_infeasible" |
| `failure_reason` | string 或 null | 一句话诊断说明，失败节点必填 |
| `retry_count` | int | 非负整数，impl_error 修复子节点继承 parent+1；≥2 时不再生成子节点 |
| `stagnation_count` | int | 非负整数，连续无提升轮数（vs best_score×1.02） |
| `stagnation_count_vs_base` | int | 非负整数，连续本轮最佳未超越父节点得分的轮数 |
| `best_score` | float | 始终 ≥ 1.0（至少等于基线） |
| `hw_params` | object 或 null | null 表示硬件查询失败；非 null 时含 chip_model、ub_size_bytes、core_num、peak_bw_gbps 等字段 |
| `mode` | string | 节点字段，必须是 strategy_guided、open_exploration 或 profiling_driven 之一；缺省视为 strategy_guided |
| `optimization_type` | string | 必须是 bandwidth/tiling/algorithm 之一 |
| `discovered_strategies` | string[] | 顶层字段，X 前缀策略 ID 列表；空列表表示本次运行尚未发现新策略 |
| `session` | object 或 null | session 身份锚定，非 null 时含 session_id、start_time、requested_rounds、actual_rounds_completed、evo_dir、op_name |
| `profiling_insight` | object 或 null | 节点可选字段；null 表示数据缺失或分析失败；非 null 时含 bottleneck、recommended_strategies、profiling_one_liner 三个子字段 |
| `profiling_evidence` | object 或 null | 节点可选字段；指令级深度空泡分析结果；null 表示未触发或分析失败；非 null 时含 bottleneck_type、d_class_pct、c_class_pct、imbalance_ratio、suggested_strategies、anti_strategies、pattern_type、overlap_status、dominant_subtype、dma_efficiency 等子字段。⚠️ v3.2 起为兼容 alias，新代码请用 `facts`/`diagnosis`/`candidate_sources` |
| `facts` | object 或 null | v3.2 新增。节点可选字段；Stage 1 纯事实抽取（由 `profiling_evidence.extract_facts()` 在 refine 写入）；非 null 时含 _source / dominant_pipe / mte2_ratio / vec_ratio / scalar_ratio / bw_utilization / bank_cflt_ratio / l2_hit_rate / icache_miss_rate / imbalance_ratio / pattern_type / overlap_status / dma_efficiency |
| `diagnosis` | object 或 null | v3.2 新增。节点可选字段；Stage 2 LLM narrative 诊断（由 refine prompt 输出）；非 null 时必须含 diagnosis_text (≥20 字符) + bottleneck_labels (子集 ⊂ KNOWN_BOTTLENECK_LABELS 18 项词表) + confidence (0~1)；wm_ops refine 自动校验，不合规则拒收强制重做 |
| `candidate_sources` | object 或 null | v3.2 新增。节点可选字段；Stage 3 反查结果（由 **wm_ops finalize-ledger** 回填到自身节点，时机：LLM 写完 diagnosis 后）；非 null 时含 candidate_source_keys (按命中数降序) / candidate_ids / by_label / unknown_labels。⚠️ utility 计算**不依赖**此字段——compute_utility 在 select 时从 parent.facts × parent.diagnosis × node.strategy_combination 现算 expected_gain |
| `baseline_evidence` | object 或 null | 顶层可选字段；根级基线 profiling 证据（由 attach-baseline-evidence 写入）；结构同 `profiling_evidence`；null 表示基线 pipeline 不可用或该子命令未被调用；非 null 时必须 `bottleneck_type ∈ BOTTLENECK_STRATEGY_MAP` 的合法键 |
