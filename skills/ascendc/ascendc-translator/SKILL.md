---
name: ascendc-translator
description: >
  AscendC kernel 转译与实现专家 Skill。将 TileLang 设计转译为 AscendC kernel，
  并生成 model_new_ascendc.py 调用 AscendC kernel。
argument-hint: >
  输入：output_dir 目录路径（包含 tile_level/ 和 model_new_tilelang.py）。
  输出：kernel/ 下的 AscendC 实现、model_new_ascendc.py。
---

# AscendC Kernel 转译 Skill

你是一名 AscendC kernel 转译与实现专家。你的目标是将 TileLang 设计转译为 AscendC kernel，并生成 `{output_dir}/model_new_ascendc.py` 调用 AscendC kernel，最终通过 AscendC 验证。TileLang 在这里是设计输入，不是 correctness gate。

## 前置条件
本阶段开始前，以下产物必须已经存在：
- `{output_dir}/design/tile_level/` — TileLang tile-level 设计，作为转译输入
- `{output_dir}/model_new_tilelang.py` — TileLang 绑定层/设计表达，可参考但不作为正确性依据

## 关键限制
- 必须将核心计算融合成单个算子实现，不要拆分成多个独立算子。
- `model_new_ascendc.py` 中禁止使用 torch 算子；只允许进行张量创建，张量变换以及调用你实现的自定义算子。
- 在 AscendC 实现中应尽可能避免标量逐元素写法，优先使用块级或向量化操作；只有在确实无法避免时才使用标量逻辑。
- 只允许修改或新增 `{output_dir}/` 目录中的文件，不要改动其他目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录；禁止读取当前工作区之外的任何路径，包括父目录、兄弟目录、用户目录、绝对路径以及系统其他目录。
- 禁止读取 `@references/TileLangAscendProgrammingGuide.md`；该文档是 TileLang 编程指南，仅供 TileLang 阶段使用，与本阶段无关。
- 严格按照算子描述生成kernel，ascend c kernel的功能应该和标杆完全一致，不能出现部分功能使用ascend c，部分使用torch算子的情况
- 即使测试用例中不包含某个功能或者分支对应的case，也要生成对应的ascend c kernel代码

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型，禁止修改
│   ├── model.json        # 测试用例文件（JSON Lines），model.py 按 __file__ 查找
│   ├── <op_name>.json    # 原始测试用例文件（备份保留）
│   ├── <op_name>.json.bak# 原始 .json 备份
│   ├── design/           # TileLang DSL 用于表达 kernel 设计
│   │   ├── block_level/  # TileLang block-level 设计（已由上一阶段完成）
│   │   └── tile_level/   # TileLang tile-level 设计（已由上一阶段完成，作为转译输入）
│   ├── kernel/           # 你的主要实现位置，放置 AscendC kernel
│   ├── model_new_tilelang.py # 上一阶段产物，可参考但不要修改
│   └── model_new_ascendc.py  # 你的 AscendC 优化实现，调用 AscendC kernel
└── <other_tasks>/        # 其他历史任务，可作为参考实现
```

## Skill 参考资料
本 skill 提供以下参考资料（位于 `@references/` 目录）：
- `@references/dsl2Ascendc.md` — TileLang 转 AscendC 指南
- `@references/TileLang-AscendC-API-Mapping.md` — TileLang 与 AscendC API 映射表
- `@references/AscendC_knowledge/` — AscendC 知识库目录
- `@references/AscendCVerification.md` — AscendC 验证指南
- `@references/evaluate_ascendc.sh` — AscendC 评测脚本

除非用户明确指定其他目录，否则默认使用传入的 `output_dir` 作为当前任务目录。
其他任务目录可以作为参考实现。

## 流程
执行以下各步骤前，必须先阅读对应的参考文档，再开始实现、验证与迭代。

1. `TileLang 转译成 AscendC`
   将 `{output_dir}/design/tile_level/` 下的 TileLang 设计转译为对应的 AscendC 实现，在 `{output_dir}/kernel/` 中生成 AscendC kernel 文件。
   参考文档：`@references/dsl2Ascendc.md`
   **实施转译前必须先阅读 `@references/TileLang-AscendC-API-Mapping.md`，逐一确认每个 TileLang API 对应的 AscendC API 映射关系，再根据映射查阅 `@references/AscendC_knowledge/` 下的具体 API 文档。禁止跳过 Mapping 直接编写 AscendC 代码。**
2. `AscendC 验证`
   编写 `{output_dir}/model_new_ascendc.py`，并调用 `@references/evaluate_ascendc.sh {output_dir}` 验证 AscendC；如果结果不正确，继续迭代修改直到通过验证。迭代次数上限为 3 次，若 3 次迭代后仍未通过验证，停止迭代并报告当前状态。不要要求 TileLang 先通过验证后再进入本阶段；若 TileLang 表达与真实执行语义存在偏差，应以设计意图和参考实现为准完成 AscendC 落地。
   参考文档：`@references/AscendCVerification.md`
