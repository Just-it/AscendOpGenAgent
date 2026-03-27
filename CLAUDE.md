# AscendOpGenAgent

Ascend NPU 算子自动生成框架，支持 Triton Ascend 和 AscendC 两种 DSL。

## 项目结构

- `agents/` — 所有 Agent（Triton + AscendC + 路由）
- `skills/triton/` — Triton 侧 Skill
- `skills/ascendc/` — AscendC 侧 Skill
- `benchmarks/` — 评测数据集（KernelBench、NPUKernelBench）
- `scripts/` — 工具脚本（含 chopper0126 同步脚本）
- `.claude/` `.opencode/` — AI 工具配置（不放 agent/skill 内容）

## Agent 体系

- `op-router` — 统一入口，任务类型 + DSL 路由
- `AKG-triton` — Triton 算子生成/优化编排
- `lingxi-code` — AscendC 算子生成编排
- `benchmark-scheduler` — Benchmark 评测调度（支持双框架）
- `kernelgen-workflow` — Triton 代码生成迭代（subagent）

## 约定

- Agent/Skill 命名统一使用连字符（kebab-case）
- 所有思考和用户交互使用中文，代码和标识符使用英文
- temperature 统一 0.1
- 输出目录：`${pwd}/triton_ascend_output/`（Triton）或 `${pwd}/output/`（AscendC）
