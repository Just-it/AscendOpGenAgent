# AutoResearch

基于 Claude Code 的算子迭代优化框架。Claude 读代码、写 plan、改 kernel、诊断
失败；Hook 负责阶段转移、plan 校验、eval 调度、KEEP/DISCARD、回滚。Eval 走
本地子进程，没有远端形态。

栈固定为 Triton-Ascend kernel + Ascend NPU + PyTorch ref，因此命令面里没有
"选 backend / 选语言"这类参数。

## Quick Start

候选源文件放 [workspace/](../workspace/)，命名 `<op>_ref.py` /
`<op>_kernel.py`：

```bash
cd <repo>
claude
```

进入 Claude 后：

```
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices 5 --max-rounds 200
```

另开终端 `python .autoresearch/scripts/dashboard.py` 看实时进度。

## 启动模式

`/autoresearch` 入参语义：

- `--`-prefixed flag → 新建任务（scaffold + 首次 baseline 原子完成）
- 已存在的目录路径 → resume 该目录
- `--resume` → resume 最近活跃 task
- 无参数 → 交互式询问

输入来源 × 起步阶段：

| 参数 | 用例 | 起步阶段 |
|------|------|----------|
| `--ref X.py --kernel Y.py` | 已有 ref + 种子 kernel | PLAN（先跑 baseline）|
| `--ref X.py` | 只有 ref | GENERATE_KERNEL |
| `--desc "..."` | 自然语言 | GENERATE_REF → GENERATE_KERNEL |
| `--desc "..." --kernel Y.py` | 描述 + 种子 kernel | GENERATE_REF |

CLI 用户面只有两个维度：

| flag | 取值 | 说明 |
|------|------|------|
| `--devices` | 本地 NPU 下标，逗号分隔：`5` 或 `0,1,2,3` | **必填** |
| `--no-code-checker` | 关闭静态分析 | 可选 |

`arch`（如 `ascend910b3`）从 `--devices` 选中的卡 `npu-smi info` 推出，写
进 `task.yaml` 仅用于 dashboard / report 显示。

## 主循环

单轮：**PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle**。
连续 3 次 FAIL 切到 DIAGNOSE，plan 全部 settle 切到 REPLAN，预算耗尽切到
FINISH。

```
INIT
  ├─ (--desc?)            GENERATE_REF ─→ GENERATE_KERNEL
  ├─ (--ref only?)                       GENERATE_KERNEL
  └─ (--ref + --kernel?)                          ─────→ BASELINE
                                                          │
                                        ┌─ scaffold --run-baseline 原子完成
                                        ▼
                                       PLAN
                                        │ create_plan.py 校验
                                        ▼
   ┌────────────────────────────────── EDIT ◀────────────┐
   │  pipeline.py:                                       │
   │    quick_check → eval_wrapper → keep_or_discard     │
   │   ├─ KEEP    : git commit (editable_files)，更新 best│
   │   ├─ DISCARD : 回滚 editable_files                   │
   │   └─ FAIL    : consecutive_failures++，回滚         │
   │                                                     │
   │   ├─ failures ≥ 3              ─→ DIAGNOSE ─→ PLAN ─┤
   │   ├─ plan 全部 settle           ─→ REPLAN  ─→ PLAN ─┤
   │   └─ eval_rounds == max_rounds ─→ FINISH
   └─────────────────────────────────────────────────────┘
```

DIAGNOSE / REPLAN 不绕回 PLAN——`create_plan.py` 校验通过后 hook 直接写
`phase = EDIT`。每个 plan item 要么在 `history.jsonl` 里有 KEEP / DISCARD /
FAIL 终态，要么在 REPLAN/DIAGNOSE 边界被静默丢弃；pid 单调推进、不复用。

阶段产物：

| 阶段 | Claude 操作 | 产物 |
|------|-------------|------|
| GENERATE_REF | Edit `reference.py` | reference.py |
| GENERATE_KERNEL | Edit `kernel.py` | kernel.py（种子）|
| BASELINE | `baseline.py` | seed_metric → progress.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py` | plan.md（含 (ACTIVE) 标记）|
| EDIT | Edit kernel.py → `pipeline.py` | history.jsonl + 可选 git commit |
| FINISH | (auto) `pipeline.py` → `report.py` | report.md（含内嵌 SVG）|

## Eval 执行

eval 全在本机子进程里跑，一轮一个 subprocess。链路：

```
eval_wrapper.py
 └─ task_config.run_eval(task_dir, config, device_id)
     └─ run_local_eval
         └─ eval_runner.local_eval(task_dir, ...)
              └─ subprocess.Popen([python, eval_kernel.py,
                                   --task-dir T --op-name X
                                   --kernel-file kernel --ref-file reference
                                   --phases verify,profile_gen[,profile_base]
                                   --device-id N ...],
                                  env={DEVICE_ID, ASCEND_RT_VISIBLE_DEVICES, ...})
```

`eval_kernel.py` 是静态脚本（不再 codegen），同一进程里串行跑请求的 phase——
verify 之后做 profile_gen，triton JIT 编译缓存自然 hot 给 profile_gen 复用。
调试时可以直接：

```bash
cd <task_dir>
python <repo>/.autoresearch/scripts/eval_kernel.py \
    --task-dir . --op-name <op> --kernel-file kernel --ref-file reference \
    --device-id 0 \
    --phases verify,profile_gen
```

verify 失败时 ref 时延仍由 profile_base 单独测得（同进程下一段），与 verify
解耦，dashboard 顶栏始终显示 PyTorch baseline。Sticky baseline 写定后
（`baseline_metric` + `baseline_source=ref` 落 progress.json）后续轮 phase 列表
里直接不带 `profile_base`，省一段。

## 精度容差

verify.py（Tier 2 预检）和 `/autoresearch` 每轮 verify 共用同一份比较函数
[`correctness.py`](../.autoresearch/scripts/correctness.py)，对齐
`main` 分支 `skills/triton/kernel-verifier/scripts/verify.py` 的
allclose-style 标准：

- 按 ref dtype 取 `(rtol, atol)`：fp32 → `(2^-13, 1e-5)`，
  fp16 → `(2^-10, 1e-3)`，bf16 → `(2^-7, 1e-2)`，未知 dtype 回落 fp32。
- 元素级判定：`|new - ref| <= atol + rtol·|ref|`，**每个元素都得过**。
- 额外硬性检查：NaN 位置一致、Inf 位置 + 符号一致、bool / int 精确匹配。

## 配置与状态

| 路径 | 用途 |
|------|------|
| `workspace/<op>_ref.py` / `<op>_kernel.py` | 候选 ref / kernel 输入 |
| `task.yaml` | 任务配置（arch / metric / editable_files / devices / eval_timeout / max_rounds）|
| `.ar_state/.phase` | 当前阶段 |
| `.ar_state/plan.md` | 规划 + 结算历史（权威态）|
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit |
| `.ar_state/progress.json` | 运行时状态 |
| `.ar_state/plan_items.xml` | PLAN/DIAGNOSE/REPLAN 写给 `create_plan.py` 的 XML |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 结构化诊断报告（CLAUDE.md 不变量 #10）|
| `.autoresearch/config.yaml` | `hallucinated_scripts`（脚本名容错映射）|
| `.claude/settings.json` | Hook + 权限配置 |
| `.claude/settings.local.json` | API key、model 覆盖（不进 git）|

`.ar_state/` 内除 `plan_items.xml` / `diagnose_v<N>.md`(DIAGNOSE) 外都由
hook 和脚本机控，Claude 不能手写。

## Skills 库

唯一参考源：`skills/triton/latency-optimizer/references/*.md`——
纯 perf 调优 markdown（无 frontmatter），按文件名挑：autotune、
scalar_to_vector、tiling_optimization、dimension-merge、
vector_core_partition、libdevice-usage、load-order、
loop-invariant-hoisting、pass-merge、discrete_memory_access、
constexpr_parameters、checklist、avoid_scalar_lowering。

PLAN 阶段 hook 提示 Claude Glob 这个目录，Read 1-3 个最相关的，把
文件名写进 plan item rationale。

## 内部机制（按需阅读）

外部接口稳定（slash 命令、`task.yaml`、`.ar_state/` 路径）。想动内部时的
入口：

| 想了解 | 看哪里 |
|--------|--------|
| Bash gate（哪条命令在哪个 phase 合法）| [phase_policy.py](../.autoresearch/scripts/phase_machine/phase_policy.py) 头部注释 |
| Hook 接线 | [.claude/settings.json](../.claude/settings.json)；脚本一律 `hook_*.py` |
| phase 转移 | [phase_machine/state_store.py](../.autoresearch/scripts/phase_machine/state_store.py) 阶段常量；`compute_next_phase` / `compute_resume_phase` 在 [phase_policy.py](../.autoresearch/scripts/phase_machine/phase_policy.py) 末尾 |
| 测时 + verify/profile 实现 | [eval_kernel.py](../.autoresearch/scripts/eval_kernel.py)（静态脚本，CLI 接 `--phases`）|
| Eval 执行链 | [task_config/eval_client.py](../.autoresearch/scripts/task_config/eval_client.py) → [eval_runner.py](../.autoresearch/scripts/eval_runner.py) `local_eval` → subprocess 调 [eval_kernel.py](../.autoresearch/scripts/eval_kernel.py) |
| Triton 退化静态检查 | [validate_triton_impl.py](../.autoresearch/scripts/validate_triton_impl.py)（AST-only）。`quick_check.py` 在 EDIT 之前调它；`skills/triton/kernel-verifier/` 单跑场景也有一份独立副本。 |
| DIAGNOSE 契约 | [CLAUDE.md](../CLAUDE.md) 不变量 #9（canonical-form bash）和 #10（DIAGNOSE artifact）|
| 子代理 | [.claude/agents/ar-diagnosis.md](../.claude/agents/ar-diagnosis.md) |

## 依赖

- Python ≥ 3.10
- `pip install pyyaml torch`
- Claude Code CLI 或 VS Code 扩展
- `torch_npu` + `triton` + CANN（Ascend NPU 上必装）

eval 在哪里跑就在哪里 `cd <repo> && claude`——卡的可见性靠 `--devices N`
（subprocess 启动时设 `ASCEND_RT_VISIBLE_DEVICES`）。

---

## 批量跑

`.autoresearch/scripts/batch/` 下的脚本对一批 `(ref.py, kernel.py?)` 任务
全自动跑 `/autoresearch`。批级状态（done/error/pending、指向 task_dir 的
链接）落 `<batch_dir>`，round 级状态（plan / history / 各轮 kernel）由
`/autoresearch` 落 `<repo>/ar_tasks/<op>_<ts>_<uuid>/`。

### happy path

约定：cwd 在仓库根目录；远端长跑用 tmux/screen；`--devices` 必填，整批
串行复用一张卡。

```bash
BATCH_DIR=/tmp/batch_001
DEVICE=0

# 1. 摆 ref/kernel 文件
mkdir -p $BATCH_DIR/refs $BATCH_DIR/kernels
cp workspace/*_ref.py    $BATCH_DIR/refs/
cp workspace/*_kernel.py $BATCH_DIR/kernels/

# 2. discover + Tier 1 verify（语法 / import / 必备 export）
python .autoresearch/scripts/batch/prepare.py $BATCH_DIR --mode ref-kernel

# 3. (可选) Tier 2 预检：本机实跑 ref vs kernel
python .autoresearch/scripts/batch/verify.py $BATCH_DIR --full

# 4. 后台跑
tmux new -d -s ar_batch \
    "python -u .autoresearch/scripts/batch/run.py $BATCH_DIR --devices $DEVICE"

# 5. 监控（另开终端）
python .autoresearch/scripts/batch/monitor.py $BATCH_DIR

# 6. 跑完汇总
python .autoresearch/scripts/batch/summarize.py $BATCH_DIR
```

### 两种模式

| Mode | 用法 | `/autoresearch` 行为 |
|------|------|----------------------|
| `--mode ref-kernel` | 已有验证过的 seed kernel | scaffold `--run-baseline` 跑 seed → BASELINE PASS → 直接 PLAN |
| `--mode ref` | 只有 reference | GENERATE_KERNEL 从 0 生成 → BASELINE → PLAN |

整批必须统一一种模式。

### Batch 目录布局

```
<batch_dir>/                         ← 批级
  manifest.yaml                      # prepare.py 写
  batch_progress.json                # run.py 写：每个 op 的 status / task_dir / metrics
  batch.log                          # run.py 写：claude --print 的全部 stdout
  verify_results.json                # prepare.py / verify.py 写
  refs/<op>_ref.py                   # ⚠️ 文件名必须严格 <op>_ref.py
  kernels/<op>_kernel.py             # ⚠️ 文件名必须严格 <op>_kernel.py（仅 ref-kernel）

<repo>/ar_tasks/<op>_<ts>_<uuid>/    ← round 级（/autoresearch 自己维护）
  kernel.py / reference.py / task.yaml
  .ar_state/{.phase, progress.json, plan.md, history.jsonl, report.md}
```

`batch_progress.json:cases.<op>.task_dir` 把两层穿起来。

### manifest.yaml

`prepare.py` 自动生成，长这样：

```yaml
mode: ref-kernel
ref_dir: refs
kernel_dir: kernels
ops:
- op_a
- op_b
```

加 / 删 ref/kernel 文件后重跑 `prepare.py $BATCH_DIR`（不必再传 `--mode`，
从 manifest 继承），ops 列表整体重扫。

### Verify 两档

| Tier | 触发 | 检查 | 需要硬件 |
|------|------|------|----------|
| 1 | `prepare.py` 默认 | 语法 / import / 必备 export（`Model`, `ModelNew`, `get_inputs`...）| 否 |
| 2 | `verify.py --full` | 实跑 ref vs kernel + allclose-style 元素级判定（per-dtype rtol+atol，与 main 对齐）| NPU + runtime |

退出码 0=全过 / 1=任何 fail/error，结果写 `verify_results.json`。Tier 2
仅 `--mode ref-kernel` 有意义。

### 监控

| 工具 | 数据源 | 用途 |
|------|--------|------|
| `monitor.py` | progress + ar_tasks/ 实时 | 跑批时另开终端看队列 / phase / heartbeat |
| `monitor.py --dashboard` | 同上 | `execvp` 进 `dashboard.py` 看 active task TUI |
| `summarize.py` | 仅 progress JSON | 跑完后离线汇总，复制粘贴友好 |
| `tail -f batch.log` | claude stdout | 看 hook 输出 / Edit / Bash 实时 |

### 断点续跑

总规律：

- `done` 不会重跑
- `error` 默认跳过；`--retry-errored` 捞回
- `pending` 自动续上
- `running`（被杀那瞬间在跑的 op）下次 `run.py` 启动时自动降级为 `error`，
  写 note `stale running, demoted on batch restart`，配 `--retry-errored` 捞回

> ⚠️ 同一 batch 目录不能同时跑两个 `run.py`（`<batch_dir>/.batch.lock`
> 排它）。死进程留下的 stale lock 会在下次启动时被自动判活清理。

### `run.py` 参数

```
batch_dir                       位置参数

--mode {ref-kernel,ref}         整批模式（也接受 manifest.mode；CLI 优先）
--devices N                     必填；透传给每个 op 的 /autoresearch

per-op 透传：
  --max-rounds 30
  --eval-timeout 120

batch 兜底：
  --timeout-min 180             单 op wall-clock 上限

队列筛选：
  --only A,B,C
  --limit N
  --retry-errored

调度 / CLI 透传：
  --cooldown-sec 5
  --claude-bin claude
  --model ""
  --extra-claude-arg ...
```

### Autoresearch 内部机制速记（debug 用）

```
hook_post_edit (Write/Edit kernel.py 后)
  └→ gate: [ -f .autoresearch/.active_task ] || exit 0   ⚠️ 依赖 .active_task
  └→ phase_machine.validate_kernel(task_dir)
       ├→ is_placeholder_file? 是 → reject
       └→ quick_check.check_editable_files → validate_triton_impl.validate
              (AST 退化检测：Type 1 / 2 / 3)
       └→ 全过 → write_phase(BASELINE)
       └→ 任一挂 → emit "[AR] kernel.py invalid"，phase 不动

hook_post_bash (任何 bash 之后)
  └→ 检测 "AR_TASK_DIR=" → set_task_dir 写 .active_task → _fresh_start 设 phase
  └→ 检测 baseline.py / pipeline.py / create_plan.py → 推进对应 phase

hook_guard_bash (bash 之前)
  └→ 不依赖 .active_task（关键差异！）
  └→ 直接读 .ar_state/.phase 决定允/禁；GENERATE_KERNEL 时禁所有 user bash
```

**关键不对称**：guard_bash 不依赖 `.active_task`，但 post_edit 依赖。这就是
"忘 export AR_TASK_DIR" 症状的成因。
