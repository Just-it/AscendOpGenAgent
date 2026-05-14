---
name: performance-analyzer
description: >
  性能分析专家 Skill。对已通过正确性验证的算子实现进行性能测试，
  对比参考实现以及可用优化实现（通常为 AscendC）的性能表现。
argument-hint: >
  输入：output_dir 目录路径（包含已通过验证的 model_new_ascendc.py，若存在也可包含 model_new_tilelang.py）。
  输出：性能测试报告，包含各实现的耗时对比。
---

# 性能分析 Skill

你是一名性能分析专家。你的目标是对 `{output_dir}` 目录下的算子实现进行性能测试，对比参考实现以及可用优化实现的性能表现。当前仓库中的 TileLang 实现主要用于设计表达，不默认作为性能测试输入。

## 前置条件
本阶段开始前，以下产物必须已经存在且通过正确性验证：
- `{output_dir}/model.py` — 参考 PyTorch 实现
- `{output_dir}/model_new_ascendc.py` — AscendC 优化实现（必选）
- `{output_dir}/model_new_tilelang.py` — TileLang 设计实现（可选；仅在用户明确要求时纳入测试）

## 关键限制
- 只允许读取 `{output_dir}/` 目录中的文件，禁止修改任何文件。
- 只允许读取当前工作区目录结构内的文件与子目录。
- 性能测试必须在 NPU 设备上执行。

## 任务目录结构
```text
.
├── {output_dir}/         # 当前活跃任务目录
│   ├── model.py          # 参考 PyTorch 模型
│   ├── model.json        # 测试用例文件（JSON Lines），model.py 按 __file__ 查找
│   ├── <op_name>.json    # 原始测试用例文件（备份保留）
│   ├── <op_name>.json.bak# 原始 .json 备份
│   ├── model_new_tilelang.py # TileLang 优化实现（如存在）
│   ├── model_new_ascendc.py  # AscendC 优化实现（如存在）
└── <other_tasks>/        # 其他历史任务
```

## Skill 参考资料
本 skill 提供以下参考资料（位于 `@references/` 目录）：
- `@references/performance.py` — 性能测试脚本，支持对比多种实现

## 流程

1. **准备性能测试**
   验证 `{output_dir}` 目录下必须存在以下实现文件：
   - `model.py` — 参考实现，作为 baseline
   - `model_new_ascendc.py` — AscendC 实现（必选）
   - `model_new_tilelang.py` — TileLang 实现（可选，且仅在用户明确要求时测试）

2. **执行性能测试**
   调用 `@references/performance.py` 脚本进行性能测试。默认对比 `reference` 和 `ascendc`；只有在用户明确要求时才额外纳入 `tilelang`：
   ```bash
   python3 @references/performance.py --output_dir {output_dir} --output {output_dir}
   ```
   默认必须测试：reference（baseline）、ascendc

3. **生成性能报告**
   收集性能数据，生成结构化报告：
   - 每种实现的平均耗时
   - 相对于参考实现的加速比
   - TileLang vs AscendC 的对比（仅在两者都被实际测试时提供）
   - 输出 `performance.json`,用于记录每个case的加速比
   - 默认在 stdout 同时输出含 DType 列的 markdown 报告，供 agent 在对话中直接回显

## 输出格式

将性能测试结果以结构化 markdown 形式输出。**借鉴 `ascendc-operator-performance-eval` 的报告骨架**，含
统一对比表（带 Shape/DType 列）、全量汇总、按数据类型汇总与简短分析四节：

```markdown
## Performance Analysis

- **Operator**: {op}
- **Device**: {device}
- **Warmup**: {warmup}
- **Repeat**: {repeats}
- **Reference**: `model.py` (OK)
- **AscendC**: `model_new_ascendc.py` (OK)

### 性能对比

| Case | Shape | DType | 参考(ms) | AscendC(ms) | 加速比 |
| ---- | ----- | ----- | -------- | ----------- | ------ |
| 0 | [128, 4096] | float16 | 0.086 | 0.046 | 1.870x |
| 1 | [64, 3584]  | bfloat16 | 0.072 | 0.061 | 1.180x |
| ... | ... | ... | ... | ... | ... |

### 全量汇总

| 指标 | 值 |
| ---- | -- |
| 用例数 | N |
| 平均加速比(>1 表示 AscendC 更快) | X.XXXx |
| AscendC 更优(比值>1) | M |
| 参考更优(比值<1) | K |
| Overall speedup | Y.YYYx |

### 按数据类型汇总

| DType | 用例数 | 平均加速比 | AscendC 更优 | 参考更优 |
| ----- | ------ | ---------- | ------------ | -------- |
| float16  | 4 | 1.870x | 4 | 0 |
| bfloat16 | 2 | 1.180x | 2 | 0 |
| float32  | 2 | 0.920x | 1 | 1 |

### 简短分析

- 整体平均加速比 X.XXx,AscendC 实现整体快于/慢于参考实现。
- 不同 dtype 表现差异:float16 平均 X.XXx 最优,float32 平均 Y.YYx 相对劣势。
- shape 规模影响:小规模平均 A.AAx,大规模平均 B.BBx,大 shape 优势更明显。
```

> **注意**：默认只要求 Reference 和 AscendC 通过正确性验证。TileLang 当前主要用于设计表达，除非用户明确要求，否则不纳入性能 gate。

## 对话内回显（MANDATORY）

调用 `performance.py` 后，agent **必须**在当前对话回复中同时完成下列事项，**不得**只输出
"报告已生成"和路径而不展示数据：

1. **粘贴性能对比表**：表头 `Case | Shape | DType | 参考(ms) | AscendC(ms) | 加速比`，所有
   dtype 在同一张表中展示。case 多时可截断并注明"其余见 `performance.json` / markdown 报告"。
2. **粘贴全量汇总**：用例数、平均加速比、AscendC/参考各自更优条数。
3. **粘贴按数据类型汇总**：分 dtype 的统计表。
4. **粘贴简短分析**：≥3 条结论，涵盖整体趋势、不同 dtype 差异、shape 规模差异等。

> 脚本默认在 stdout 末尾打印上述 markdown 形式的报告，可直接复制到对话中；如需关闭可使用 `--no-display`。

**NEVER**：仅回复报告路径；NEVER 用"请自行打开 markdown"替代在对话中展示核心数字与结论。

## 异常处理

| 情况 | 处理方式 |
|------|---------|
| 无 NPU 设备 | 报错，提示需要 NPU 环境 |
| TileLang 实现不存在 | 继续执行 reference + ascendc 测试 |
| 实现文件不存在 | 跳过不存在的实现，仅测试存在的 |
| 性能测试失败 | 记录错误信息，尝试其他实现 |
