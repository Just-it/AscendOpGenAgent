---
name: case-simplifier
description: >
  Triton-Ascend 工作流专用的测试用例精简 Skill。读取 triton-ascend-coder 工作目录中
  与算子配套的 `.json`（JSON Lines 格式，每行一个 `{"inputs": [...]}` 对象），
  对其中的 cases 进行精简，使数量不超过 10，同时保证 dtype/shape/广播覆盖。
  原 `.json` 备份为 `.json.bak`，供 Phase 5 全量终评时恢复使用。
argument-hint: >
  输入：work_dir 目录路径（即 triton-ascend-coder 的工作目录，
  形如 `triton_ascend_output/op_xxx_xxx/`）。
  输出：精简后的 cases 已写回原 `.json`，原内容备份为 `.json.bak`。
---

# Triton 测试用例精简 Skill

你是 triton-ascend-coder 工作流中的测试用例精简专家。目的：在 Phase 3/4 的迭代验证中，
让 verify.py 和 benchmark.py 只跑 ≤10 个有代表性的 case，加速生成与优化迭代。

## 1. 确定目标 `.json` 文件

按以下优先级在 `{work_dir}/` 内定位目标 `.json`：

1. 读取 `{work_dir}/{op_name}.py`（即 Phase 1 产出的 KernelBench 任务文件），
   分析 `get_input_groups()` 中 `os.path.join(os.path.dirname(__file__), "xxx.json")`
   引用的文件名，据此定位目标 `.json`。
2. 如果 `{op_name}.py` 中没有显式引用，则查找 `{work_dir}/` 下唯一的 `.json` 文件。

如果找不到唯一的 `.json` 文件，报错并停止。

## 2. 跳过条件

读取目标 `.json` 后，先统计 case 总数：
- 若 case 数 ≤ 10：**直接跳过精简，不创建 `.bak`**，输出 "无需精简" 并退出。
- 若 case 数 > 10：进入下面的精简流程。

## 3. 关键限制

- 只允许修改目标 `.json` 文件，禁止改 `{work_dir}/{op_name}.py`。
- 只允许在 `{work_dir}/` 内读写文件，不要触碰其他目录。
- 必须先备份：将目标 `.json` 复制为同名的 `.json.bak`（例如 `1_LogicalAnd.json.bak`），
  Phase 5 全量终评时会从此恢复。

## 4. 精简原则（按优先级从高到低）

1. **dtype 覆盖**：原 cases 中出现的每种 tensor dtype（float16/float32/bfloat16/
   int8/int16/int32/int64/uint8/bool/complex64/complex128/float64 等）至少保留一个 case。
2. **混合 dtype 覆盖**：若存在同一 case 内多个输入 dtype 不同（混合 dtype）的场景，
   至少保留 1 个混合 dtype case。
3. **attribute 取值覆盖**：对于 `type: "attr"` 的输入，覆盖出现的不同取值类别
   （bool 的 True/False、整型的正/负/零、枚举值等代表性边界）。
4. **shape 维度覆盖**：覆盖原 cases 中出现的不同 tensor 维度数（0维/1维/2维/3维/4维/...），
   每种维度至少保留一个 case。
5. **shape 极端值覆盖**：保留极端小（如 0-d 标量、最小 shape）和极端大（最大元素数）的 case。
6. **广播模式覆盖**：若原 cases 中存在 shape 不完全一致需广播的 tensor 对，
   至少保留一个 broadcasting case。

## 5. 流程

1. **读取**：JSON Lines 格式逐行解析。
2. **统计**：dtype 集合、是否存在混合 dtype、attr 值集合、shape 维度集合、shape 大小范围、
   是否存在 broadcasting。
3. **备份**：`cp {target}.json {target}.json.bak`。
4. **选取**：按上述原则选取 ≤10 个代表性 case，尽量让单个 case 同时覆盖多个维度差异。
5. **写回**：以 JSON Lines 格式写回目标 `.json`（每行一个 `json.dumps(...)`），
   保持与原始格式一致。
6. **报告**：打印一行摘要，例如
   `精简完成：原 52 cases → 8 cases，覆盖 dtype=[int8,int16,...,complex128]`。
