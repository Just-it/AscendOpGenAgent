---
name: precision-tuning-discovery
description: >
  AscendC 算子精度调优的方法论参考(发现式审计)。与 precision-tuning(构造式)
  互为镜像:直接从数值取证数据出发,运用 agent 自身的 AscendC 领域知识推理根因,
  不强制预读 lowering 示例。适用场景:agent 已能从 diff 模式快速锁定嫌疑区域。
  本 skill 不构成强约束,agent 可全部、部分或完全不采纳。
argument-hint: >
  无强制参数。agent 在 Phase 4 卡住时可读本 SKILL.md 学习发现式分析思路;
  scripts 与 references 复用 @../precision-tuning/(共享同一套取证脚本与知识库)。
---

## 轻约束声明(必读)

**本 skill 仅作为 agent 在 Phase 4 AscendC 精度迭代陷入停滞、且常规 Conductor 修复建议无法收敛时的可选参考资料。** ascend-kernel-developer 应优先按自身 4.1-4.4 主流程判断;本 skill 的方法、分类、流程仅供借鉴,不构成强约束。

**精度判定标准始终以 `utils/verification_ascendc.py` 中的 MERE/MARE 准则为准** —— `MERE < threshold AND MARE < 10 × threshold`,阈值表见 `PRECISION_THRESHOLDS` 字典(fp16=2^-10 / bf16=2^-7 / fp32=2^-13 等),int8/int16 量化输出允许 ±1 LSB(`INT_LSB_TOLERANCE`)。**忽略本文档及 @../precision-tuning/ 其它位置出现的 allclose/rtol/atol 数值**,那是 OpenOps 原版的判定方式。

## 与 precision-tuning(构造式)的区别

| 维度 | precision-tuning(构造式) | precision-tuning-discovery(本 skill) |
|------|--------------------------|---------------------------------------|
| 入口 | 先建参考实现规范,再读 kernel | 先看 diff 模式与 worst 位置,凭直觉锁定嫌疑区域 |
| 必读资料 | 强制读 lowering 示例 + error_correction + tl_asc_routing + non_aligned | 不强制预读,需要时主动查阅 |
| 适用前提 | agent 对 AscendC API 还不够熟 | agent 已能从 pattern hint 快速判断根因 |
| 风险 | 流程长,但稳 | 流程短,但容易凭印象误判 |
| 输出审计模板 | `[REFERENCE_IMPL_SPEC]` 先行 | `[REFERENCE_IMPL_SPEC]` 仍需补,但可放在分析尾段 |

两套 skill **是给 agent 看的不同审计思路**,agent 应根据当时具体情况自行决定采用哪种(或都不采用)。

## 共享资源(全部经由 @../precision-tuning/ 引用)

- 取证脚本:`@../precision-tuning/scripts/precision_forensics.py`
- Gate-F/A/V 脚本:`@../precision-tuning/scripts/precision_gate.py`
- 知识库 search/dump:`@../precision-tuning/scripts/precision_knowledge.py`
- 知识库 JSON:`@../precision-tuning/references/precision_knowledge_base.json`
- 算子分解示例:`@../precision-tuning/references/decomposition_examples/`(8 个算子,不含 dynamic_quant)
- skill 内部文档:`@../precision-tuning/README.md`、`@../precision-tuning/STRUCTURE.md`

**注意路径适配**:`@../precision-tuning/` 中的 scripts/SKILL.md 含大量 OpenOps 项目路径(如 `.opencode/skills/ascendc-evaluation/`、`{OpName}Custom/op_kernel/`),这些**在 OpGen 中对应**:
- `{OpName}Custom/op_kernel/` → `{output_dir}/kernel/`
- OpenOps `generate_pybind.py` / `evaluate.py` → OpGen `skills/ascendc/ascendc-translator/references/evaluate_ascendc.sh`
- OpenOps 整体管线 Step 0-6 → 由 ascend-kernel-developer 的 Phase 4 迭代承担

不要期望取证/Gate 脚本在 OpGen 容器里直接 bash 跑通,**只读它们学习诊断思路**。

## What I do(方法论摘要)

发现式分析的关键动作:

1. **数值取证解读**(读 `forensics_report_{attempt}.json` 或同等数据):
   - L0-L4 直接数值事实:diff 统计、pattern hint、worst 元素位置、误差分布
   - L6 内存布局:tensor shape/stride/对齐情况
   - L8 算子类型 → 路由到对应的 checklist 思路

2. **dtype 精度级别快判**:
   - float32:max_abs_diff > 1e-4 → 逻辑错误;≤ 1e-4 → 精度损失
   - float16:> 1e-2 → 逻辑错误;1e-3~1e-2 → float16 精度损失
   - bfloat16:> 5e-2 → 逻辑错误;5e-3~5e-2 → bfloat16 精度损失
   - (注:这是 OpenOps allclose 风格的快判,**用于定向**;最终 PASS/FAIL 仍以 OpGen 的 MERE/MARE 为准)

3. **凭 AscendC 领域知识识别精度反模式**(无需预读示例):
   - **TBuf 数据竞争**:TBuf 绕过 outQueue 直接写 GM = 数据未真正写出 = 输出全零或残留旧值
   - **Padding 污染**:DataCopy 未按 32-byte 对齐,边界元素被写入垃圾值;尾块未用 DataCopyPad
   - **类型精度损失**:float16 累加溢出/欠精;`-INFINITY` 写法错误(float16 不存在该常量)
   - **归约 work_buf 未初始化**:ReduceMax 前没 `Duplicate(-INF)` / ReduceSum 前没 `Duplicate(0)`,残留上一步数据
   - **跨核同步缺失**:SyncAll 缺失,Core 0 在其他核未完成时读 workspace 拿到脏数据
   - **错误的 API 名称**:用 `Vmax`(不存在,应为 `Max`)、`Subs`(不存在,应为 `Sub` 标量参数)、`AscendC::INFINITY`(不存在)

4. **借助知识库精化(可选)**:
   - 检索 `precision_knowledge_base.json` 看有没有匹配 pattern 的现成知识条目
   - 不命中也没关系,继续凭领域知识推理

5. **修复计划 → 修复 → 重编 → 验证**:由 ascend-kernel-developer Phase 4 的 ascendc-translator 迭代承担,**本 skill 不替代该流程**

## When to use me

ascend-kernel-developer Phase 4 已迭代 ≥ 2 次仍报精度类失败,Conductor 给出的修复建议在最近一轮 outcome 为 stagnant 或 regressed,且 agent 对 AscendC API 有信心(否则选 @../precision-tuning/ 构造式 skill 更稳)。

仍然:agent 可以**完全自行决定**是否查阅本 skill。
