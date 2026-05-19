# Precision Tuning Skill — 设计说明

## 概述

修复编译通过但精度测试失败的 AscendC 算子。作为 Lingxi-Code pipeline 中 Scheduler 优化环路的"精度优化"分支。

本 Skill 采用**双 Subagent 架构**，共用同一套基础设施，但提供两种不同的精度审计策略。

## 文件结构

```
.opencode/
├── agents/
│   ├── precision-tuning-discovery.md          # 发现式 Subagent
│   └── precision-tuning.md                    # 构建式 Subagent
└── skills/
    └── precision-tuning/
        ├── SKILL.md                           # Skill 定义 (subagent 模式)
        ├── README.md                          # 本文件：设计文档
        ├── STRUCTURE.md                       # 目录结构说明
        ├── MODIFICATIONS.md                   # precision_gate.py 实现说明（当前状态）
        ├── test_baseline_coverage.py          # 覆盖性测试（3 场景）
        ├── scripts/                           # 共用脚本
        │   ├── precision_forensics.py         # 数值取证 (L0-L4 + L6 + L8 + available_files)
        │   ├── precision_gate.py              # 链式 Gate 验证 + 循环控制
        │   └── precision_knowledge.py         # 知识库管理: load / search / dump
        ├── references/                        # 共用参考资料
        │   ├── precision_knowledge_base.json  # 精度问题知识库
        │   └── decomposition_examples/        # 算子计算分解示例
        │       ├── README.md                  # 分解示例索引
        │       ├── softmax.md                 # Softmax: 单行归约 (5 步)
        │       ├── layer_norm.md              # LayerNorm: 单行归约 3-pass (7 步)
        │       ├── reduce_sum.md              # ReduceSum: 单步归约 + 跨步访存 (2 步)
        │       ├── mse_loss.md                # MSELoss: 跨核两阶段归约 (6 步)
        │       ├── matmul.md                  # MatMul: 分块累加 (4 步)
        │       ├── average_pooling2d.md       # AvgPool2d: 滑窗累加 (3 步)
        │       └── cumsum.md                  # CumSum: 前缀累加 (2 步)
```

## 双 Subagent 架构

### 发现式 Subagent (`precision-tuning-discovery`)

**审计策略**: 直接从数值取证数据出发，运用 AscendC 领域知识推理根因。

**特点**:
- 不强制预读参考示例，依赖 Agent 自身的 AscendC 知识储备
- 快速从 diff 模式锁定嫌疑区域
- 适用场景：Agent 对 AscendC API 规范已有充分了解

**核心能力**:
- 数值取证解读（L0-L8）
- 代码精度分析（发现式）
- 精度修复

### 构建式 Subagent (`precision-tuning`)

**审计策略**: 严格遵循 Phase A→B→C 的构建式流程。

**特点**:
- Phase A: 先建规范，再看代码（强制读取 lowering 示例）
- Phase B: 读取当前实现
- Phase C: 以 REFERENCE_IMPL_SPEC 为基准进行结构化对照
- 适用场景：需要严格参照规范进行审计

**核心能力**:
- 数值取证解读（L0-L8）
- 构建式审计（Phase A→B→C）
- 精度修复

### 共用基础设施

| 组件 | 类型 | 说明 |
|------|------|------|
| `precision_forensics.py` | 脚本 | L0-L8 数值取证 |
| `precision_gate.py` | 脚本 | 链式 Gate 验证 + 循环控制 |
| `precision_knowledge.py` | 脚本 | 知识库管理 |
| `precision_knowledge_base.json` | 数据 | 精度问题模式库 |
| `decomposition_examples/` | 文档 | 算子计算分解示例 |

### 策略对比

| 维度 | 发现式 | 构建式 |
|------|--------|--------|
| **分析起点** | 直接从数值取证数据出发 | 先建立规范基准 |
| **Phase A** | 可选查阅 | 强制读取 lowering 示例 |
| **Reference** | 按需查阅 | 必须产出 `[REFERENCE_IMPL_SPEC]` |
| **优势** | 快速、灵活 | 严谨、系统化 |
| **适用** | 经验丰富的 Agent | 规范要求高的场景 |

## 信息层级 (L0-L8)

| 层级 | 信息类型 | 状态 | 实现位置 |
|------|---------|------|---------|
| L0 | PASS/FAIL | ✅ 已实现 | precision_forensics.py |
| L1 | 统计值 + 误差分布 | ✅ 已实现 | precision_forensics.py |
| L2 | 位置特征 (尾块/维度/边界) | ✅ 已实现 | precision_forensics.py |
| L3 | 数值特征 (幅值/NaN/符号) | ✅ 已实现 | precision_forensics.py |
| L4 | 张量切片 (per-index) | ✅ 已实现 | precision_forensics.py |
| L5 | 中间结果探测 | ❌ 不实现，由 L7 替代 | 见下方 L5 设计决策 |
| L6 | 内存布局分析 | ✅ 已实现 | MemoryLayoutAnalyzer |
| L7 | 代码位置映射 | ✅ Agent 手动完成 | Sub-step 2.3 L7 手动映射 (静态推算) |
| L8 | 算子语义 | ✅ 部分实现 | OperatorTypeDetector + 知识库 CHECKLIST |

## 分工原则

| 操作 | 执行者 | 原因 |
|------|--------|------|
| L0-L4 数值取证 | Python | 确定性计算 |
| L6 内存布局 | Python | tensor 属性读取 |
| L8 算子类型检测 + 属性提取 | Python | 规则推断, 含 dim/reduction_axis |
| 可用文件检测 | Python | 纯文件 IO |
| Pattern hint 分类 + 语义加权 | Python | 规则化, 作为建议 |
| Gate 验证 + 循环控制 | Python | 结构化检查, Agent 不可覆盖 |
| 知识库 IO + RAG 检索 | Python | 结构化评分, 纯文件操作 |
| 算子计算分解 (Sub-step 2.2) | Agent | 需要理解参考实现 + DSL 的语义 |
| AscendC 逐步对照 (Sub-step 2.3) | Agent | 需要理解 C++ 代码结构 |
| 根因诊断 (Sub-step 2.4) | Agent | 需要推理 |
| 修复计划 + 代码修复 | Agent | 创造性工作 |

## 链式 Gate 设计

每个 Gate 不仅检查当前步骤产物, 还验证前序步骤是否完成:

```
Gate-F (forensics) → 无前置依赖, 检查 attempt 号匹配
Gate-A (audit)     → 前置: forensics 存在且 attempt 匹配
                     检查 6 个必填 section: FORENSICS_SUMMARY, COMPUTATION_DECOMPOSITION,
                     KERNEL_STEP_TRACE, ROOT_CAUSE, FIX_PLAN, TARGET_FILES
                     attempt > 0 时额外检查: DIRECTION_ASSESSMENT
Gate-X (fix)       → 前置: audit 存在
Gate-V (validate)  → 前置: 代码文件存在
```

返回码区分:
- 0: 通过
- 1: 产物不完整 (可重试)
- 2: 前置依赖缺失 (必须回退补完前序步骤)

## 循环控制

Gate-V 输出 loop_signal, Agent **必须遵守**:

| 信号 | 条件 | Agent 操作 |
|------|------|-----------|
| PASS | 精度通过 | 成功收尾 + 知识库跃迁 |
| NEARLY_SUCCESS | match_rate ≥ 99% 但 evaluate 返回 FAIL | **立即跳到 Step 5**，不得发起下一轮 |
| CONTINUE | 未通过但 mismatch 有改善 | 归档本轮, 回到 Step 1 |
| STOP | 达到 `MAX_ATTEMPTS` 轮上限 或 连续 `MAX_STAGNANT_ROUNDS` 轮无改善 | 失败报告 |

## 知识库

扁平五字段结构, RAG-ready。包含两类条目:
1. **问题模式** (9 条): 具体精度问题的 feature/reason/fix
2. **算子 CHECKLIST** (5 条): 按算子类型的精度检查清单 (reduction/pooling/loss/matmul/normalization)

成功修复后自动追加跃迁条目 (仅成功时写入, 避免污染)。

## 知识库 RAG 检索（已实现）

### 方案：结构化关键词筛选 + 评分排序

**设计决策**: 不使用向量嵌入 (sentence-transformers + FAISS), 原因:
1. aarch64 环境下 FAISS 编译成本高, sentence-transformers 依赖重
2. 当前规模 (18 条, 预期增长到 100-200 条) 不需要向量检索
3. 知识库已有结构化标签 (`type`, `feature` 中的 `pattern=xxx` / `op_type=xxx`), 精确匹配比语义相似度更可靠
4. Fallback 到全量 load 保底, 不会漏掉任何条目

**实现**: `precision_knowledge.py search` 命令

```bash
# 第一次检索 (Sub-step 2.1 完成后)
python3 precision_knowledge.py search \
    --kb-path <path> --op-type <type> --pattern <hint> --top-k 3

# 第二次精化检索 (Sub-step 2.4 开始前, 增加位置特征)
python3 precision_knowledge.py search \
    --kb-path <path> --op-type <type> --pattern <hint> --position <tail/boundary/scattered> --top-k 3
```

**评分逻辑**:
- pattern 匹配 feature 中的 `pattern=xxx` → 权重 3
- op_type 匹配 feature 中的 `op_type=xxx` 或标题/feature 中的类型关键词 → 权重 2
- type 字段与 pattern→type 亲和性映射匹配 → 权重 1
- position 与 pattern 亲和性映射匹配 (仅第二次检索) → 权重 1

**返回结构**:
- `matched_entries`: top-K 普通条目 (按 score 降序)
- `checklists`: op_type 匹配的 CHECKLIST 条目 (不占 K 配额, 始终附带)
- `fallback_to_full_load`: 无任何命中时自动全量返回

**触发时机**:
- Sub-step 2.1 完成后 (已知 op_type + primary_hint) → 第一次检索
- Sub-step 2.4 开始前 (增加 worst_element 位置特征) → 第二次精化检索

**Fallback**: search 返回 0 条普通条目 + 0 条 CHECKLIST → 自动全量 load

## 计算分解示例 (decomposition_examples/)

Step 2 的 Sub-step 2.2 要求 Agent 将算子的参考实现分解为逐步计算链。
`references/decomposition_examples/` 目录提供了 7 个算子的分解示例, 覆盖 5 种计算模式:

| 计算模式 | 示例算子 | 关键审计点 |
|---------|---------|-----------|
| 单行归约 | softmax, layer_norm | padding 值、count 对齐、归约维完整性 |
| 跨核归约 | mse_loss | workspace 同步、Phase 2 正确性、分母计算 |
| 分块累加 | matmul | 累加器初始化、分块边界、精度累积 |
| 滑窗累加 | average_pooling2d | 有效面积计算、边界/padding 处理 |
| 前缀累加 | cumsum | 跨 tile 累加器传递、顺序正确性 |

Agent 在 Sub-step 2.2 中:
1. 先读取 `decomposition_examples/README.md` 了解格式和模式分类
2. 查找与当前算子最匹配的示例文件
3. 按示例的粒度标准完成计算分解
4. 每步标注精度风险点和知识库关联

> **注意**: 发现式 Subagent 可选查阅分解示例，构建式 Subagent 则强制参考。

## L5 设计决策：不实现，由 L7 Agent 手动映射替代

**背景**: 早期设计希望通过 Python 脚本自动探测 Kernel 内部每个计算步骤的中间输出，定位误差引入步骤。

**技术评估（基于 Softmax 工程结构）**:

| 维度 | 评估结果 |
|------|---------|
| 实现路径 | 需要在 kernel 代码中插入额外 GlobalTensor 输出、修改 host TilingFunc 分配 workspace、修改 op_proto 增加输出端、重新完整编译 |
| 通用性 | 每个算子的中间步骤不同（ReduceMax/ReduceSum/Exp 各自需要独立探针），无法写一个通用脚本 |
| 侵入风险 | 探针修改可能影响 buffer 对齐，甚至改变精度问题的表现（Heisenbug 效应） |
| 工程代价 | 相当于为每个算子维护一个 debug 版本，成本高、不可复用 |

**替代方案**: Sub-step 2.3 的 L7 手动映射已覆盖 L5 的核心价值:
- `worst element index [row, col]` → 静态推算落在哪个 Core、main/tail block
- 对照 tiling 参数（tileLength、rowsPerCore）直接定位到 Compute() 中的 K-Step
- 对单行归约（softmax、layer_norm）、逐元素算子完全够用
- 对跨核归约（mse_loss Phase1→Phase2），L7 同样可通过 workspace 地址偏移推算

**结论**: L5 Python 实现不可行，已由 L7 Agent 手动映射替代。`IntermediateProbe` 类保留为存根，不再标记为 TODO。

## 中间文件完整说明

> 本节基于 `output/cumsum_precision_tuned_20260414_020456/cumsum/precision_tuning/` 的真实运行结果编写，每个文件均经过内容读取与代码对照验证。

### 一、目录总览

```
{output_path}/precision_tuning/
│
│── [Step 1]      forensics_report_{attempt}.json     # 取证报告（Python 脚本生成，{attempt} 为轮次编号）
│── [Gate-F→]     baseline_state.json                # 初始精度基线（Gate-F 自动写入）
│── [Step 2]      precision_audit_{attempt}.md       # 深度审计全文（Agent 写入，{attempt} 为轮次编号）
│── [Step 2.1/4]  knowledge_search_log_{N}.json      # 知识库检索日志（precision_knowledge.py 写入）
│── [Step 4.2]    compilation_log_{N}.json           # 编译失败日志（仅编译出错时出现）
│── [Step 4.4]    validation_result_attempt_{N}.json # 精度验证结果（Agent 写入）
│── [Gate-A→V]    round_summary_{N}.json             # 本轮综合摘要（Gate-A + Gate-V 分两次写入）
│── [Gate-V→]     tuning_directions.json             # 跨轮方向学习表（Gate-V 每轮追加）
│── [Step 5.1]    candidate_kb_entry.json            # 候选知识库条目（Agent 写入，精度通过后）
│── [Step 5.4]    {op_name}_precision_tuning_process.md  # 完整过程记录（Agent 写入）
│
└── history/
    ├── baseline/code_snapshot/                      # [Step 0.1] 不可变基线代码（Agent cp）
    ├── attempt_{N}/
    │   ├── code_snapshot/                           # [Step 0.2/归档] 本轮起始代码（Agent cp）
    │   ├── sections/                                # [Gate-A→] 各 section 独立文件（Gate-A 自动提取）
    │   ├── forensics_report.json                    # [归档步骤] 取证报告副本（Agent cp）
    │   └── precision_audit.md                       # [归档步骤] 审计报告副本（Agent cp）
    ├── current_best/code_snapshot/                  # [归档步骤] 当前最佳代码（Agent 按 match_rate 更新）
    └── success/code_snapshot/                       # [Step 5.3] 最终成功代码（Agent cp，永久不覆盖）
```

---

### 二、顶层文件详解

#### `forensics_report_{attempt}.json` — Step 1 生成

**创建者**：`precision_forensics.py`（Python 脚本，不可跳过）

**创建时机**：每轮 Step 1 运行后写入，文件名包含轮次编号（如 `forensics_report_0.json`、`forensics_report_1.json`），不会被后续轮次覆盖。历史取证报告同时通过 `cp` 归档到 `history/attempt_{N}/forensics_report.json`。

**内容**：L0-L8 全层次数值取证结果，是所有后续分析的原始数据源。

```json
{
  "version": "2.0",
  "op_name": "cumsum",
  "attempt": 0,
  "status": "completed",
  "L0_pass": false,
  "outputs": [{
    "basic_stats": {
      "mismatch_ratio": 0.989227,   // 0~1 比例，非百分比
      "match_rate": 0.010773,       // 0~1 比例，非百分比
      "max_abs_diff": 35.73,
      "mean_abs_diff": 10.91
    },
    "error_distribution": { ... },  // 误差分布、符号分析
    "tail_analysis": { ... },       // 尾块 mismatch 率分析
    "dimension_analysis": { ... },  // 各维度 mismatch 率
    "worst_elements": [ ... ]       // top-3 最大误差元素及其位置
  }],
  "primary_hint": "all_wrong",      // 误差模式分类（Gate 和知识库检索均依赖此字段）
  "primary_confidence": 0.90,
  "L6_memory_layout": { ... },      // 输入/输出内存 shape/stride/对齐情况
  "L8_operator": {                  // 算子语义
    "op_type": "reduction",
    "attributes": { "dim": 2 },
    "reduction_axis": { "axis_length": 64 }
  },
  "history_trend": null             // attempt 0 时为 null；attempt ≥ 1 时由取证脚本填入历史 mismatch 趋势
}
```

> **注意**：`match_rate` 和 `mismatch_ratio` 在 forensics 中的单位是 **0\~1 比例**（而非百分比）。Gate 和 `baseline_state.json` 写入时会乘以 100 转换为百分比。

---

#### `baseline_state.json` — Gate-F 通过后自动写入

**创建者**：`precision_gate.py` 的 `_write_baseline_from_forensics()`

**创建时机**：Gate-F 验证通过 **且 attempt == 0** 时立即写入。此时代码尚未被任何修复操作修改，是真正的初始精度基线。

**幂等性**：文件一旦存在则不覆盖，保证 baseline 始终记录第一次取证时的原始精度。

```json
{
  "match_rate": 1.0773,
  "mismatch_ratio": 0.989227,
  "max_abs_diff": 35.73,
  "mean_abs_diff": 10.91,
  "primary_hint": "all_wrong",
  "source": "forensics_report_{attempt}.json/outputs[0]/basic_stats",
  "note": "Initial precision captured at Gate-F before any code modification"
}
```

> **为何在 Gate-F 而非 Gate-V 写入**：Gate-V 在代码修复、编译、验证之后才运行，此时 `forensics_report_0.json` 是 attempt 0 的取证结果（含代码修改前的原始精度），`history_trend` 仍为 null（attempt 0 无历史），从 forensics 读 baseline 是唯一可靠来源。

---

#### `precision_audit_{attempt}.md` — Step 2 写入

**创建者**：Agent（Step 2 各 Sub-step 的产出合并写入单一文件）

**创建时机**：Step 2 深度分析过程中 Agent 逐步追加，Gate-A 通过时文件已完整。文件名包含轮次编号（如 `precision_audit_0.md`），不被后续轮次覆盖。历史版本同时通过 `cp` 归档到 `history/attempt_N/precision_audit.md`。

**内容**：包含 9 个结构化 section，Gate-A 逐项校验其存在性和完整性：

| Section | 对应 Sub-step | 内容 |
|---------|--------------|------|
| `[FORENSICS_SUMMARY]` | 2.1 | 取证数据逐字段摘录，含 L6/L8/dtype 判断 |
| `[COMPUTATION_DECOMPOSITION]` | 2.2 | 参考实现的逐步计算链，含精度风险点 |
| `[REFERENCE_IMPL_SPEC]` | 2.3 Phase A | TQue/TBuf 规范、关键 API 签名、非对齐处理规范 |
| `[KERNEL_STEP_TRACE]` | 2.3 Phase B/C | Kernel Compute() 逐步对照，含 L7 手动映射 |
| `[KNOWLEDGE_MATCH]` | 2.4 | 知识库检索结果及 CHECKLIST 逐项核查 |
| `[ROOT_CAUSE]` | 2.4 | 根因判断 + 证据链（数值/布局/代码/逻辑） |
| `[FIX_PLAN]` | 2.4 | 修复类型、修改文件、修改点、预期效果 |
| `[TARGET_FILES]` | 2.4 | 需要修改的文件列表 |
| `[DIRECTION_ASSESSMENT]` | 2.4 | attempt > 0 时填写：是否延续上一轮方向 + 换方向理由 |

Gate-A 通过后，`precision_gate.py` 自动将上述每个 section 提取为独立的 `.md` 小文件保存到 `history/attempt_N/sections/`，供后续轮次按需读取。

---

#### `knowledge_search_log_{N}.json` — Sub-step 2.1 / 2.4 写入

**创建者**：`precision_knowledge.py search` 命令（每次检索追加一条记录，覆盖写入同一文件）

**创建时机**：每次 Agent 执行 `python3 precision_knowledge.py search ...` 时追加一条记录。每轮最多两次检索（Sub-step 2.1 基础检索 + Sub-step 2.4 精化检索）。

```json
[
  {
    "attempt": null,
    "call_index": 0,           // 0=第一次检索（2.1），1=第二次检索（2.4）
    "timestamp": "...",
    "query": {
      "op_type": "reduction",
      "pattern": "all_wrong",
      "position": null,
      "top_k": 3
    },
    "matched_count": 3,
    "checklist_count": 1,
    "fallback_to_full_load": false,
    "top_titles": [
      "归约轴切分破坏导致局部归约错误 (Reduction Axis Split Error)",
      "..."
    ]
  }
]
```

> 此文件供事后回溯知识库检索的命中质量，不参与 Gate 验证。

---

#### `compilation_log_{N}.json` — Step 4.2 写入（仅编译失败时存在）

**创建者**：Agent（Step 4.2 编译失败时写入，编译通过则不创建此文件）

**创建时机**：每次 `generate_pybind.py` 编译失败后 Agent 追加写入，最多 3 次重试。

```json
{
  "attempt": 0,
  "entries": [
    {
      "compile_retry": 0,
      "error_category": "undefined_api",
      "error_snippet": "error: 'Vmax' was not declared in this scope",
      "fix_applied": "将 Vmax 改为 Max（AscendC 正确 API 名称）"
    }
  ]
}
```

> cumsum 首轮编译通过，故该文件不存在。`round_summary_0.json` 的 `index.compilation_log` 字段为 `null`，这是正常情况。

---

#### `validation_result_attempt_{N}.json` — Step 4.4 写入

**创建者**：Agent（Step 4.4 从 `evaluate.py` 的 stdout 解析后写入）

**创建时机**：每轮 Step 4.3 精度验证完成后，Agent 立即写入。Gate-V 读取此文件判断精度是否通过。

```json
{
  "attempt": 0,
  "correctness_passed": true,
  "evaluate_stdout": "INFO - Evaluation correctness: [PASS]\nOutput 0: shape=[16, 32, 64], match_rate=100.00% (32768/32768), max_diff=0.00000e+00, ...",
  "match_rate": "100.00",   // 字符串，百分比数值（不带 % 号）
  "max_diff": "0.0"
}
```

> `match_rate` 在此文件中是**百分比字符串**（如 `"100.00"`），与 `forensics_report_{attempt}.json` 中的 0\~1 比例不同。`_write_round_summary()` 读取时用 `float(mr_str)` 直接得到百分比数值。

---

#### `round_summary_{N}.json` — Gate-A + Gate-V 两阶段写入

**创建者**：`precision_gate.py`（两次写入，分别由 `check_audit()` 和 `check_validate()` 触发）

**第一次写入（Gate-A 通过后）**：`_write_audit_index()` 写入 `diagnostics` + `index` 字段，`metrics` 各字段初始化为 `null`。

**第二次写入（Gate-V 后）**：`_write_round_summary()` 读取 Gate-A 已写内容，合并后补充 `metrics` 数值字段，同时补充 `diagnostics.forensics_hint`、`diagnostics.op_type` 和 `index.compilation_log`。

```json
{
  "attempt": 0,
  "metrics": {
    "match_rate": 100.0,             // 来自 validation_result_attempt_0.json
    "mismatch_ratio": 0.0,
    "improvement_ratio": null,       // attempt 0 且 baseline_state.json 不存在时为 null（已修复）
    "absolute_improvement": null,    // 同上
    "stop_reason_code": "precision_passed"  // 永不为 null
  },
  "diagnostics": {
    "forensics_hint": "all_wrong",   // Gate-V 从 forensics_report_{attempt}.json 补充
    "op_type": "reduction",          // 同上
    "fix_type": "FIX_PRECISION_LOGIC",    // Gate-A 从 [FIX_PLAN] section 提取
    "changed_locations": ["op_host.cpp"], // Gate-A 从 [TARGET_FILES] 提取
    "direction_verdict": null             // Gate-A 从 [DIRECTION_ASSESSMENT] 提取，attempt 0 可为 null
  },
  "index": {
    "forensics": "precision_tuning/history/attempt_0/forensics_report.json",
    "audit_full": "precision_tuning/precision_audit_0.md",
    "sections": {
      "forensics_summary": "precision_tuning/history/attempt_0/sections/forensics_summary.md",
      "root_cause": "precision_tuning/history/attempt_0/sections/root_cause.md",
      "fix_plan": "precision_tuning/history/attempt_0/sections/fix_plan.md",
      // ... 其余 section 路径
    },
    "code_snapshot": "precision_tuning/history/attempt_0/code_snapshot/",
    "validation": "precision_tuning/validation_result_attempt_0.json",
    "compilation_log": null,
    "tuning_directions": "precision_tuning/tuning_directions.json",
    "forensics_used": "precision_tuning/forensics_report_0.json"
  }
}
```

**Agent 使用方式**：下一轮 attempt > 0 时，先读 `tuning_directions.json` 获取全局方向概览；若需某轮的具体根因或修复计划，通过 `round_summary_N.index.sections.root_cause` / `.fix_plan` 路径直接定位 section 小文件，而非全量读取 `precision_audit_{attempt}.md`。

---

#### `tuning_directions.json` — Gate-V 每轮追加写入

**创建者**：`precision_gate.py` 的 `_write_tuning_directions()`

**创建时机**：每轮 Gate-V 运行结束时追加当前轮 entry；精度通过时回溯填写所有 entry 的 `contributed` 字段。

**作用**：整个调优过程的**统一方向学习入口**，Agent 每轮优先读此文件获取历史概览，无需逐轮读取 round_summary。

```json
{
  "op_name": "cumsum",
  "final_status": "success",       // "in_progress" | "success" | "nearly_success" | "failed"
  "entries": [{
    "attempt": 0,
    "fix_type": "FIX_PRECISION_LOGIC",
    "forensics_hint": "all_wrong",
    "direction_verdict": null,      // attempt 0 无上一轮，为 null
    "direction_reason": "首轮分析，根因已定位",  // 从 [DIRECTION_ASSESSMENT] 提取
    "improvement_ratio": null,      // attempt 0 且 baseline 缺失时为 null
    "absolute_improvement": null,   // 同上
    "outcome": "passed",            // "passed" | "improved" | "stagnant" | "regressed"
    "evidence": {
      "forensics_ref": "precision_tuning/forensics_report_0.json",
      "audit_ref":     "precision_tuning/precision_audit_0.md",
      "match_rate":    100.0,
      "mismatch_ratio": 0.0
    },
    "contributed": true             // 仅 final_status=success 时存在
  }]
}
```

---

#### `candidate_kb_entry.json` — Step 5.1 写入

**创建者**：Agent（Step 5.1，精度通过后手动生成）

**创建时机**：Gate-V 返回 `PASS` 后，Agent 在 Step 5.1 基于本轮 `[ROOT_CAUSE]` 和 `[FIX_PLAN]` 生成泛化的知识条目。Step 5.2 的 `precision_knowledge.py dump` 命令读取此文件写入全局知识库。

```json
{
  "title": "Cumsum Host Tiling 与 Python 转置逻辑不一致导致输出全零 (Cumsum Host Tiling Mismatch)",
  "feature": "all_wrong 模式，actual_value 全为 0，Kernel 未写入数据，或部分不匹配且数值偏离",
  "reason": "Python 层的 transpose 逻辑与 Host Tiling 函数对 scan 轴的假设不一致...",
  "fix": "统一 Python 层与 Host Tiling 的维度转置约定...",
  "type": "FIX_PRECISION_LOGIC"
}
```

---

#### `{op_name}_precision_tuning_process.md` — Step 5.4 写入

**创建者**：Agent（Step 5.4，精度通过后写入完整过程记录）

**创建时机**：精度调优成功收尾时一次性写入，是整个调优过程的人类可读总结文档。

**内容**：调优概览（算子名、调优状态、最终 match_rate）、问题现象、取证数据关键发现、根因分析（证据链）、修复计划与代码变更对比、编译验证结果、知识库条目、文件归档说明。

> 此文件是**人工复盘和知识沉淀**的主要阅读对象，不参与任何 Gate 验证。

---

### 三、`history/` 子目录详解

#### `history/baseline/code_snapshot/` — Step 0.1 创建

**创建者**：Agent（SKILL.md Step 0.1 的 shell 命令）

**创建时机**：整个调优流程**首次**执行时（`if [ ! -d history/baseline/code_snapshot ]`），将算子原始代码一次性复制保存。**全程不覆盖**。

**内容**：4 个文件，覆盖算子完整实现：

| 文件 | 来源 |
|------|------|
| `op_kernel.cpp` | `{OpName}Custom/op_kernel/{op_name}_custom.cpp` |
| `op_host.cpp` | `{OpName}Custom/op_host/{op_name}_custom.cpp` |
| `op_tiling.h` | `{OpName}Custom/op_host/{op_name}_custom_tiling.h` |
| `op_pybind.cpp` | `{output_path}/{op_name}.cpp` |

**用途**：任何时候可从此恢复到最初基线，是精度回溯和问题复现的最终参照。

---

#### `history/attempt_{N}/code_snapshot/` — 归档步骤创建

**创建者**：Agent（shell 命令）

**创建时机**：
- **CONTINUE 归档时**：在当前轮 Gate-V 后，保存本轮修改后的代码
- **PASS / NEARLY_SUCCESS 时**：Step 5.0 归档时保存本轮修改后的代码

**内容**：保存该轮修复**后**的代码状态，即本轮 attempt_N 的产出。与 `baseline/code_snapshot/` 的区别在于：baseline 记录修复前原始代码，`attempt_N/code_snapshot/` 记录本轮修复完成后的代码。

---

#### `history/attempt_{N}/sections/` — Gate-A 通过后自动提取

**创建者**：`precision_gate.py` 的 `_write_audit_index()`

**创建时机**：每轮 Gate-A 验证通过后立即提取，无需 Agent 手动操作。

**内容**：将 `precision_audit_{attempt}.md` 中的每个 section 提取为独立 `.md` 文件：

| 文件 | 对应 section | 主要内容 |
|------|-------------|---------|
| `forensics_summary.md` | `[FORENSICS_SUMMARY]` | L0-L8 取证摘要 |
| `computation_decomposition.md` | `[COMPUTATION_DECOMPOSITION]` | 算子计算链分解 |
| `reference_impl_spec.md` | `[REFERENCE_IMPL_SPEC]` | AscendC API 规范对照 |
| `kernel_step_trace.md` | `[KERNEL_STEP_TRACE]` | Kernel 步骤逐行追踪 |
| `knowledge_match.md` | `[KNOWLEDGE_MATCH]` | 知识库命中条目 |
| `root_cause.md` | `[ROOT_CAUSE]` | 根因判断 + 证据链 |
| `fix_plan.md` | `[FIX_PLAN]` | 修复计划详情 |
| `target_files.md` | `[TARGET_FILES]` | 修改文件清单 |
| `direction_assessment.md` | `[DIRECTION_ASSESSMENT]` | 方向延续/切换判断 |

**设计意图**：避免下一轮 attempt 读取整个 `precision_audit_{attempt}.md`（几百行），Agent 按需通过 `round_summary_N.index.sections.*` 的路径直接读取对应的单个 section 文件。

> **提取失败处理**：若某 section 未找到，对应 `index.sections.{name}` 置为 `null`，Gate-A 不因此阻断，Agent fallback 读取 `index.audit_full`（完整审计文件）。

---

#### `history/attempt_{N}/forensics_report.json` 和 `history/attempt_{N}/precision_audit.md` — 归档步骤创建

**创建者**：Agent（CONTINUE 时的归档步骤，或 PASS 时的 Step 5.0）

**创建时机**：本轮 Gate-V 返回信号后，Agent 执行 `cp` 命令将顶层的 `forensics_report_{attempt}.json` 和 `precision_audit_{attempt}.md` 归档至 `history/attempt_N/`（归档副本使用不带编号的平铺名，由文件夹提供作用域隔离）。

> **真实运行的说明**（cumsum 案例）：`round_summary_0.json` 的 `index.forensics` 和 `index.audit_full` 均指向 `history/attempt_0/` 下的副本，但该目录中仅存在 `sections/` 和 `code_snapshot/`，这两个文件副本未被创建。说明 cumsum 实际运行时未完整执行归档步骤。`sections/` 由 Gate-A 自动生成，是可靠的；而 `forensics_report.json` 和 `precision_audit.md` 的副本依赖 Agent 手动执行，存在遗漏风险。

---

#### `history/current_best/code_snapshot/` — 归档步骤动态更新

**创建者**：Agent（CONTINUE 归档步骤，通过比较当前 match_rate 与 `match_rate.txt` 决定是否更新）

**创建时机**：每轮 CONTINUE 时检查当前 match_rate 是否优于历史最优，是则覆盖更新；PASS 时在 Step 5.0 无条件更新（match_rate = 100.0）。

**附属文件**：`current_best/match_rate.txt`，记录当前最优 match_rate 数值，供下一轮归档步骤比较。

**用途**：调优失败（Gate-V 返回 STOP）时，可从此处恢复精度最高的代码继续探索，而无需从头开始。

---

#### `history/success/code_snapshot/` — Step 5.3 创建

**创建者**：Agent（Step 5.3，精度 100% 通过后执行）

**创建时机**：精度验证通过（Gate-V 返回 `PASS`）后，Step 5.3 将最终成功代码复制至此。**不覆盖**（与 baseline/code_snapshot/ 同为不可变存档）。

**与 current_best 的区别**：

| 目录 | 更新时机 | 是否覆盖 | 用途 |
|------|---------|---------|------|
| `current_best/` | 每轮 match_rate 改善时 | 覆盖 | 失败时的最优恢复点 |
| `success/` | 仅在精度 100% 通过时 | 不覆盖 | 成功代码的永久存档 |

---

### 四、文件生成时序图

```
Step 0.1  →  history/baseline/code_snapshot/        (Agent, 首次执行，原始代码，全程不覆盖)
Step 1    →  forensics_report_{attempt}.json         (precision_forensics.py)
Gate-F    →  baseline_state.json                    (precision_gate.py, attempt 0 自动写入)
Step 2    →  precision_audit_{attempt}.md            (Agent, 逐 Sub-step 追加)
Step 2.1  →  knowledge_search_log_0.json            (precision_knowledge.py)
Step 2.4  →  knowledge_search_log_0.json            (precision_knowledge.py, 追加第二条)
Gate-A    →  history/attempt_0/sections/*.md        (precision_gate.py, 自动提取)
          →  round_summary_0.json (diagnostics+index)(precision_gate.py)
Step 4.2  →  compilation_log_0.json                 (Agent, 仅编译失败时)
Step 4.4  →  validation_result_attempt_0.json       (Agent)
Gate-V    →  round_summary_0.json (metrics 补充)    (precision_gate.py)
          →  tuning_directions.json                 (precision_gate.py)
            ├─ loop_signal=CONTINUE → 归档步骤:
            │    cp forensics_report_{attempt}.json → history/attempt_0/forensics_report.json
            │    cp precision_audit_{attempt}.md   → history/attempt_0/precision_audit.md
            │    cp 修改后代码 → history/attempt_0/code_snapshot/  ← 本轮修改结果
            │    更新 history/current_best/ (若 match_rate 改善)
            │    → 回到 Step 1 (attempt+1)
            ├─ loop_signal=NEARLY_SUCCESS → Step 5 (nearly_success 路径，不再发起 attempt+1):
            │    Step 5.0: 归档 + 保存 attempt_N/code_snapshot + 更新 current_best
            │    Step 5.1~5.4: 同 PASS 路径
            └─ loop_signal=PASS → Step 5:
                 Step 5.0: 归档 + 保存 attempt_N/code_snapshot + 更新 current_best (match_rate=100.0)
                 Step 5.1: candidate_kb_entry.json  (Agent)
                 Step 5.2: precision_knowledge.py dump → 写入知识库
                 Step 5.3: history/success/code_snapshot/ (Agent)
                 Step 5.4: {op_name}_precision_tuning_process.md (Agent)
```

## TODO 接口清单

| 接口 | 类 | 位置 | Phase | 接口定义 |
|------|-----|------|-------|---------|
| 中间结果探测 | IntermediateProbe | precision_forensics.py | ❌ 不实现 (见 L5 决策) | 接口存根，不调用 |
| 代码位置映射 | CodeMapper | precision_forensics.py | ❌ 不实现，由 Agent 手动完成 | Sub-step 2.3 L7 手动映射 |
| 多测试案例 | MultiCaseForensics | precision_forensics.py | evaluate.py 扩展后 | 详见类 docstring |
