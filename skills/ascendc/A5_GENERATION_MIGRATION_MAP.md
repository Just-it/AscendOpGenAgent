# A5 生成能力迁移映射

这份文档记录 `a5_ops` 中 A5 算子生成能力，与 `AscendOpGenAgent` 中迁移后文件的对应关系。

目的：
1. 记录迁移的要求、步骤和适配规则，供后续维护者理解迁移决策
2. 后续如果 `a5_ops` 里的 A5 生成 agent / skill / references 有内容更新，可以快速判断应该同步到 `AscendOpGenAgent` 的哪个文件

当前迁移范围只包含 **A5 算子生成能力**，不包含环境管理、知识维护、hook 质量门。

---

## 0. 迁移要求与步骤

### 0.1 迁移目标

将 `a5_ops` 中的 A5 AscendC 算子生成能力完整迁移到 `AscendOpGenAgent`，使其成为独立可用的算子生成系统。迁移后的系统必须能在本地 A5 环境中独立运行，不依赖 `a5_ops` 的环境管理和知识维护基础设施。

### 0.2 迁移核心要求

**保持一致性原则**：迁移后的 skill/agent 必须尽可能保留原本的信息，仅做最小改动。只在以下必要场景做修改：

1. **文件名/路径适配**
   - skill 目录加 `a5` 前缀：`ascendc-op-gen` → `ascendc-a5-op-gen`
   - references 路径：`${CLAUDE_SKILL_DIR}/references/` → `../a5-shared-references/`
   - 静态检查脚本：`src/scripts/` → `skills/ascendc/a5-common-scripts/`
   - agent 名称：`migration-executor` → `ascend-a5-kernel-developer`

2. **删除远程部署流程**
   - 删除所有 SSH / Docker / 容器管理逻辑
   - 删除 `.ascendc_env` 配置文件读取和 preflight 检查
   - 删除 `ascendc-preflight` skill 依赖
   - 远程命令（如 `/a5_op msprof`）替换为本地等效命令

3. **删除知识库自更新**
   - 删除 `ascendc-knowledge-maintain` skill 依赖
   - 删除 `knowledge_update.md` 产出要求
   - 删除 hooks 质量门（`pre-commit` / `post-op` 检查）
   - 删除 Step/Phase 中的 "knowledge update" 步骤

4. **本地 A5 环境硬编码**
   - `SOC_VERSION=Ascend950PR_9589`
   - `CANN_PATH=/usr/local/Ascend/cann-9.0.T501/`
   - 构建命令：`python3 utils/build_ascendc.py {output_dir} -v Ascend950PR_9589 --build-type Release`
   - msprof 路径：`/usr/local/Ascend/cann-9.0.T501/tools/profiler/bin/msprof`

5. **增加文件访问边界**
   - 每个 skill/agent 都必须声明允许读取和禁止读取的路径
   - 统一禁止：`archive_tasks/**`、`ascendc-translator/**`、`tilelang-designer/**`、`performance-analyzer/**`、`trace-recorder/**`、任何非 `ascendc-a5-*` 的 skill 目录

### 0.3 迁移步骤（已完成）

按以下顺序执行迁移，每步都对照 `a5_ops` 原文件逐段检查：

**Step 1: Agent 迁移**
- 源：`a5_ops/src/agents/migration-executor.md`
- 目标：`AscendOpGenAgent/agents/ascend-a5-kernel-developer.md`
- 保留：Phase 0（KB 加载）→ Stage 1（构建+精度）→ Stage 2（优化）→ Stage 3（探索）→ Finalize
- 保留：File Structure（5 文件）、Checkpoint Assertions、Fault Tolerance、Anti-Hack Rules、Quality Discipline
- 适配：本地构建命令、文件访问边界、删除 hooks/deploy/knowledge-update

**Step 2: Orchestrator 迁移**
- 源：`a5_ops/src/skills/ascendc-op-gen/SKILL.md`
- 目标：`AscendOpGenAgent/skills/ascendc/ascendc-a5-op-gen/SKILL.md`
- 保留：Step 0 Detect Mode（`--benchmark` / `--opgen` / 正则 / 路径检测）
- 保留：Step 2 Spawn Worker（含完整 prompt template）
- 保留：Step 3 Monitor Progress（60s 轮询）
- 保留：Step 4 Independent Verification（6 项 anti-hack + performance gate）
- 保留：Benchmark Worker Prompt Template（6 步工作流 + KB 加载指令）
- 保留：Op-Gen Worker Prompt Template（7 步工作流 + Rules & KB）
- 适配：删除 preflight/knowledge-maintain 步骤、本地环境变量、`PROGRESS_FILE` 默认 `{OUTPUT_DIR}/PROGRESS.md`

**Step 3: Analyzer 迁移**
- 源：`a5_ops/src/skills/ascendc-op-analyzer/SKILL.md`
- 目标：`AscendOpGenAgent/skills/ascendc/ascendc-a5-op-analyzer/SKILL.md`
- 保留：Step 0（源类型检测）→ Step 1（选择性 KB 加载）→ Step 2P（PyTorch 路径）→ Step 2（CUDA 路径）→ Step 3（pattern 加载）→ Step 4（强制审计：int64/scatter-add/TUA/PyTorch-CUDA diff）→ Step 5（SIMT/SIMD 决策）→ Step 6（test spec）→ Step 7（输出）→ Step 8（人工检查点）
- 适配：references 路径

**Step 4: Generator 迁移**
- 源：`a5_ops/src/skills/ascendc-kernel-generator/SKILL.md`
- 目标：`AscendOpGenAgent/skills/ascendc/ascendc-a5-kernel-generator/SKILL.md`
- 保留：Step 1（完整 context loading + PATTERN_INDEX trigger 规则）
- 保留：Step 2（代码生成：header template、VF 命名、dispatcher template、bf16 scalar 传递 P-P30、mandatory patterns、variant 生成、compile-time specialization）
- 保留：Step 3（编译循环：static check → build → conductor error classification A/B/C → EC-1..EC-9 → compile_history.json）
- 保留：Step 4（输出 + compile_report.md）
- 适配：构建命令、references 路径、删除 post-implementation knowledge update

**Step 5: QA Verifier 迁移**
- 源：`a5_ops/src/skills/ascendc-qa-verifier/SKILL.md`
- 目标：`AscendOpGenAgent/skills/ascendc/ascendc-a5-qa-verifier/SKILL.md`
- 保留：Section 0（容器健康检查）→ Section 0.5（ground truth 选择）→ Section 1（精度 gate + per-dtype 阈值 + scatter-add waiver）→ Section 2（性能 A/B + noise guard）→ Section 3（msprof profiling）→ Section 4（bottleneck diagnosis + structured output）→ Section 5（optimization directive）→ Section 6（gate decision + per-category performance）
- 适配：本地 msprof 命令、references 路径

**Step 6: Researcher 迁移**
- 源：`a5_ops/src/skills/ascendc-researcher/SKILL.md`
- 目标：`AscendOpGenAgent/skills/ascendc/ascendc-a5-researcher/SKILL.md`
- 保留：6-step protocol（Profile → Grounding Chain GC-1~GC-7 → Enumerate → Filter → Hypotheses → Rank）
- 保留：Expert Code Diff 流程、Bounding Rules（max 3 structural、max 90min、early termination）
- 适配：`/a5_op msprof` → 本地 msprof 命令、references 路径、文件访问边界

### 0.4 迁移质量检查清单

每个迁移文件完成后，逐项确认：

- [ ] 与 `a5_ops` 原文件逐段对比，核心工作流无遗漏
- [ ] 代码模板（header/VF/dispatcher）完整保留
- [ ] Anti-Hack Rules 完整保留
- [ ] 所有 `${CLAUDE_SKILL_DIR}/references/` 已替换为 `../a5-shared-references/`
- [ ] 所有远程命令已替换为本地等效命令
- [ ] 无 `.ascendc_env` / SSH / Docker / preflight / knowledge-maintain 残留
- [ ] 文件访问边界已声明（允许 + 禁止列表）
- [ ] 构建命令使用 `python3 utils/build_ascendc.py ... -v Ascend950PR_9589`

---

## 1. 迁移总原则

### 已迁移

- A5 算子生成主 Agent
- 源码分析能力
- AscendC 代码生成能力
- QA / benchmark / profiling 能力
- 性能研究能力
- A5 共享 references
- 静态检查脚本

### 未迁移

- `.ascendc_env` / `workspace/.ascendc_env` 配置流
- `ascendc-preflight`
- `ascendc-knowledge-maintain`
- `ascendc-op-gen/hooks/*`
- SSH / Docker / 凭据管理逻辑
- 自动知识库更新流程
- `knowledge_update.md` 产出要求
- `archive_tasks/**` 历史任务目录
- A3 / TileLang 相关 skill 与 references，例如：
  - `skills/ascendc/ascendc-translator/**`
  - `skills/ascendc/tilelang-designer/**`
  - `skills/ascendc/performance-analyzer/**`
  - `skills/ascendc/trace-recorder/**`
  - 任何非 `ascendc-a5-*` 的 skill 内容

### 当前迁移版环境前提

迁移后的 A5 流程默认运行在 **当前本地 A5 环境**，只确认以下配置：

- `SOC_VERSION=Ascend950PR_9589`
- `CANN_PATH=/usr/local/Ascend/cann-9.0.T501/`

---

## 2. Agent / Skill 对应关系

| a5_ops 源文件 | AscendOpGenAgent 目标文件 | 关系 | 同步说明 |
|---|---|---|---|
| `a5_ops/src/agents/migration-executor.md` | `AscendOpGenAgent/agents/ascend-a5-kernel-developer.md` | 主体迁移 + 本地化改写 | 如果 `migration-executor.md` 的生成阶段、质量门、编排顺序变化，需要先评估是否同步到 A5 agent |
| `a5_ops/src/skills/ascendc-op-gen/SKILL.md` | `AscendOpGenAgent/skills/ascendc/ascendc-a5-op-gen/SKILL.md` | 一对一迁移，但去掉 preflight / `.ascendc_env` / knowledge-maintain | 如果 a5_ops 的 orchestrator 有新的模式识别、worker 编排、独立校验逻辑，要同步 |
| `a5_ops/src/skills/ascendc-op-analyzer/SKILL.md` | `AscendOpGenAgent/skills/ascendc/ascendc-a5-op-analyzer/SKILL.md` | 一对一迁移 | 如果 analyzer 的分类、审计、SIMT/SIMD 决策、test spec 规则更新，要同步 |
| `a5_ops/src/skills/ascendc-kernel-generator/SKILL.md` | `AscendOpGenAgent/skills/ascendc/ascendc-a5-kernel-generator/SKILL.md` | 一对一迁移，但去掉知识维护与外部环境编排 | 如果 generator 的代码模式、编译修复策略、Anti-Hack 规则更新，要同步 |
| `a5_ops/src/skills/ascendc-qa-verifier/SKILL.md` | `AscendOpGenAgent/skills/ascendc/ascendc-a5-qa-verifier/SKILL.md` | 一对一迁移 | 如果 verifier 的 gate、profiling、ratio 计算、瓶颈诊断更新，要同步 |
| `a5_ops/src/skills/ascendc-researcher/SKILL.md` | `AscendOpGenAgent/skills/ascendc/ascendc-a5-researcher/SKILL.md` | 一对一迁移 | 如果 researcher 的 hypothesis 模板、bounded exploration 规则更新，要同步 |
| `a5_ops/src/skills/ascendc-preflight/SKILL.md` | 无 | 明确不迁移 | 不同步 |
| `a5_ops/src/skills/ascendc-knowledge-maintain/SKILL.md` | 无 | 明确不迁移 | 不同步 |

---

## 3. References 对应关系

`a5_ops` 的 references 在迁移后被集中放到一个共享目录里：

- 源：`a5_ops/src/skills/references/**`
- 目标：`AscendOpGenAgent/skills/ascendc/a5-shared-references/**`

### 目录级映射

| a5_ops | AscendOpGenAgent |
|---|---|
| `a5_ops/src/skills/references/ASCENDC_LANGUAGE_REFERENCE.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/ASCENDC_LANGUAGE_REFERENCE.md` |
| `a5_ops/src/skills/references/ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/ASCENDC_SIMD_DEVELOPMENT_REFERENCE.md` |
| `a5_ops/src/skills/references/ASCENDC_SIMT_PATTERNS.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/ASCENDC_SIMT_PATTERNS.md` |
| `a5_ops/src/skills/references/BENCHMARK_METHODOLOGY.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/BENCHMARK_METHODOLOGY.md` |
| `a5_ops/src/skills/references/ERROR_CORRECTIONS.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/ERROR_CORRECTIONS.md` |
| `a5_ops/src/skills/references/KB_INDEX.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/KB_INDEX.md` |
| `a5_ops/src/skills/references/MSPROF_AGENT_GUIDE.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/MSPROF_AGENT_GUIDE.md` |
| `a5_ops/src/skills/references/OPERATIONAL_KNOWLEDGE.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/OPERATIONAL_KNOWLEDGE.md` |
| `a5_ops/src/skills/references/PLATFORM_BUGS.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/PLATFORM_BUGS.md` |
| `a5_ops/src/skills/references/ROOFLINE_MODEL.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/ROOFLINE_MODEL.md` |
| `a5_ops/src/skills/references/SIMT_VS_SIMD_DECISION.md` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/SIMT_VS_SIMD_DECISION.md` |
| `a5_ops/src/skills/references/exploration/**` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/exploration/**` |
| `a5_ops/src/skills/references/hardware/**` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/hardware/**` |
| `a5_ops/src/skills/references/patterns/**` | `AscendOpGenAgent/skills/ascendc/a5-shared-references/patterns/**` |

### 同步规则

- 只要 `a5_ops/src/skills/references/**` 有内容更新，优先检查 `a5-shared-references/**` 是否需要同步
- 如果更新只影响 **环境管理 / knowledge maintenance** 相关流程，不同步
- 如果更新影响 **SIMT/SIMD 决策、错误修复、pattern、profiling、hardware、benchmark 规则**，通常应同步

---

## 4. 脚本对应关系

| a5_ops 源文件 | AscendOpGenAgent 目标文件 | 关系 | 同步说明 |
|---|---|---|---|
| `a5_ops/src/scripts/ascendc_static_check.py` | `AscendOpGenAgent/skills/ascendc/a5-common-scripts/ascendc_static_check.py` | 一对一迁移 | 如果静态检查规则有新增/删减，要同步 |
| `a5_ops/src/scripts/a5_pipeline.sh` | 无 | 不迁移 | 环境/流水线脚本，不同步 |
| `a5_ops/src/scripts/a5_sync_and_build.sh` | 无 | 不迁移 | 环境/同步脚本，不同步 |
| `a5_ops/src/deploy.sh` | 无 | 不迁移 | 部署脚本，不同步 |

---

## 5. 内容同步时的决策规则

后续如果 `a5_ops` 有更新，按下面规则判断是否同步。

### 应同步的更新

1. **分析规则变化**
   - 新增算法分类
   - 新增审计项
   - SIMT/SIMD 决策变化
   - test spec 规则变化

2. **生成规则变化**
   - 新的 kernel 代码模式
   - 新的 bf16 / int64 / launch / TQue / SIMT/SIMD 约束
   - 新的编译修复经验
   - 新的 Anti-Hack 规则

3. **QA 规则变化**
   - 精度 gate 变化
   - ratio 计算方式变化
   - profiling 提取方式变化
   - bottleneck diagnosis 规则变化

4. **research 规则变化**
   - grounding chains 更新
   - structural dimensions 更新
   - hypothesis 模板更新

5. **references 变化**
   - pattern / hardware / operational knowledge / benchmark / msprof / error corrections 更新

### 不应同步的更新

1. **环境管理变化**
   - `.ascendc_env`
   - `ascendc-preflight`
   - A5 IP / SSH / Docker / 凭据流程

2. **知识维护变化**
   - `ascendc-knowledge-maintain`
   - `knowledge_update.md`
   - hooks 检查项

3. **独属于 a5_ops 的流水线变化**
   - deploy / sync / pipeline shell 脚本
   - a5_ops 项目级 CLAUDE 规则里只跟该仓库运行方式相关的部分

4. **A3 / TileLang 侧的内容**
   - `archive_tasks/**`
   - `skills/ascendc/ascendc-translator/**`
   - `skills/ascendc/tilelang-designer/**`
   - `skills/ascendc/performance-analyzer/**`
   - `skills/ascendc/trace-recorder/**`
   - 任何非 A5 迁移 skill 的实现细节

---

## 6. 推荐同步步骤

每次 `a5_ops` 的 A5 生成能力有更新时，建议按这个顺序检查：

1. 先看 `a5_ops/src/skills/ascendc-op-analyzer/SKILL.md`
2. 再看 `a5_ops/src/skills/ascendc-kernel-generator/SKILL.md`
3. 再看 `a5_ops/src/skills/ascendc-qa-verifier/SKILL.md`
4. 再看 `a5_ops/src/skills/ascendc-researcher/SKILL.md`
5. 再看 `a5_ops/src/skills/ascendc-op-gen/SKILL.md`
6. 然后检查 `a5_ops/src/skills/references/**`
7. 最后检查 `a5_ops/src/scripts/ascendc_static_check.py`

如果更新跨多个文件，优先保证：

- references 先同步
- skill 描述再同步
- 最后再调 agent 编排描述

---

## 7. 当前迁移文件清单

### Agent

- `AscendOpGenAgent/agents/ascend-a5-kernel-developer.md`

### Skills

- `AscendOpGenAgent/skills/ascendc/ascendc-a5-op-gen/SKILL.md`
- `AscendOpGenAgent/skills/ascendc/ascendc-a5-op-analyzer/SKILL.md`
- `AscendOpGenAgent/skills/ascendc/ascendc-a5-kernel-generator/SKILL.md`
- `AscendOpGenAgent/skills/ascendc/ascendc-a5-qa-verifier/SKILL.md`
- `AscendOpGenAgent/skills/ascendc/ascendc-a5-researcher/SKILL.md`

### Shared references

- `AscendOpGenAgent/skills/ascendc/a5-shared-references/**`

### Common scripts

- `AscendOpGenAgent/skills/ascendc/a5-common-scripts/ascendc_static_check.py`

---

## 8. 最小同步检查表

每次 `a5_ops` 更新后，至少过一遍下面这张表：

- [ ] `migration-executor.md` 是否改变了阶段编排
- [ ] `ascendc-op-gen/SKILL.md` 是否改变了 orchestrator 规则
- [ ] `ascendc-op-analyzer/SKILL.md` 是否改变了分析 / 审计 / 决策
- [ ] `ascendc-kernel-generator/SKILL.md` 是否改变了生成 / 编译修复规则
- [ ] `ascendc-qa-verifier/SKILL.md` 是否改变了精度 / 性能 / profiling gate
- [ ] `ascendc-researcher/SKILL.md` 是否改变了研究策略
- [ ] `src/skills/references/**` 是否有新增或更新
- [ ] `src/scripts/ascendc_static_check.py` 是否有新增检查项
- [ ] 更新是否属于环境管理 / 知识维护（如果是，跳过同步）
