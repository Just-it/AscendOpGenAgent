# ascendc-code-review

**Tagline**: 完整的 AscendC 编码红线 Review + 修复 + 验证工作流

**Triggers**:
- User 说 "code review" 或 "编码红线" 或 "red line" 时
- User 要求对 AscendC 算子进行安全检查时
- 提交代码前的强制检查

---

## 1. Introduction

这是一套完整的 AscendC 算子编码规范审查、修复、验证工作流。它将帮助你：

1. **Review 阶段** - 自动识别 7 大类编码红线问题和 13 条 TopN 问题
2. **Fix 阶段** - 提供标准修复模式，生成详细整改报告
3. **Verify 阶段** - 三层验证（编译、UT、ST），确保 100% 通过率

**核心特性**：
- [通过] **系统化检查** - 基于《编码红线.md》规范，覆盖 20+ 检查点
- [通过] **优先级分级** - P0-P3 严重性分级，优先修复高危问题
- [通过] **自动化验证** - 编译→UT→ST 三层验证，可复现
- [通过] **文档输出** - 生成详细整改报告，便于团队复盘

---

## 2. Usage

### 完整工作流（推荐）

```bash
# 对单个算子执行完整 Review + Fix + Verify
使用 ascendc-code-review 技能对 ai_infra_sinkhorn_grad 算子进行完整检查
```

**工作流程**：
1. 自动调用 `ascendc-review` 进行编码红线检查
2. 根据检查结果调用 `ascendc-fix` 进行修复
3. 调用 `ascendc-verify` 进行编译和测试验证
4. 生成最终验证报告

### 独立使用子 Skills

```bash
# 仅进行 Review（不修复）
使用 ascendc-review 技能检查 ai_infra_sinkhorn_grad 算子

# 仅修复已知问题
使用 ascendc-fix 技能修复 ai_infra_sinkhorn_grad 算子的 SINK-01 和 SINK-03 问题

# 仅验证修复结果
使用 ascendc-verify 技能验证 ai_infra_sinkhorn_grad 算子
```

---

## 3. How It Works

### 主入口逻辑

```
┌─────────────────────────────────────────────┐
│   ascendc-code-review (主入口)              │
│                                             │
│   1. 解析用户请求                           │
│   2. 确定算子范围                           │
│   3. 调度子 Skills                          │
│   4. 汇总最终报告                           │
└─────────────────────────────────────────────┘
              │
              ├──→ Phase 1: ascendc-review
              │    - 读取算子代码
              │    - 应用编码红线检查清单
              │    - 输出问题清单 (JSON)
              │
              ├──→ Phase 2: ascendc-fix
              │    - 读取问题清单
              │    - 应用标准修复模式
              │    - 生成整改报告 (Markdown)
              │
              └──→ Phase 3: ascendc-verify
                   - 编译验证 (build.sh)
                   - UT 验证 (运行单元测试)
                   - ST 验证 (运行集成测试)
                   - 生成验证报告 (Markdown)
```

### 工作流步骤详解

#### Phase 1: Review (检查阶段)

**输入**：
- 算子路径（如 `training/ascendc/src/ops-transformer/mhc/ai_infra_sinkhorn_grad/`）

**执行**：
1. 读取 `op_kernel/*.h` 和 `op_host/*_tiling.cpp` 文件
2. 应用编码红线检查清单：
   - 红线 1.1: 除零保护
   - 红线 1.2: 数组越界
   - 红线 1.3: 溢出保护
   - 红线 1.4: 变量初始化
   - 红线 1.5: 空指针保护
   - TopN 2.1-2.13: 特殊值、输入校验、int64 偏移等
3. 标记每个问题的：
   - 问题 ID (如 SINK-01)
   - 严重性 (P0-P3)
   - 代码位置 (文件名:行号)
   - 违反的规范

**输出**：
- JSON 格式问题清单（供 ascendc-fix 消费）
- Markdown 格式检查报告（供人阅读）

**示例输出**：
```json
{
  "operator_name": "ai_infra_sinkhorn_grad",
  "issues": [
    {
      "id": "SINK-01",
      "severity": "P0",
      "rule": "红线1.5 - 空指针保护",
      "location": "op_kernel/ai_infra_sinkhorn_grad_generalized.h:63-73",
      "description": "Init函数未检查 tPipe 和 tilingData 是否为 null"
    }
  ]
}
```

#### Phase 2: Fix (修复阶段)

**输入**：
- 问题清单 (JSON)
- 算子源代码路径

**执行**：
1. 按优先级排序问题 (P0 → P1 → P2 → P3)
2. 对每个问题：
   - 定位到具体代码行
   - 应用标准修复模式（见下文"标准修复模式"）
   - 添加安全注释
   - 更新文档（如 `docs/*.md`）
3. 生成详细整改报告

**输出**：
- 修复后的源代码文件
- Markdown 格式整改报告（包含修复前后对比）

**标准修复模式**：

| 问题类型 | 修复模式 | 代码示例 |
|---------|---------|---------|
| 除零保护 | 添加 `if (divisor == 0) return;` | `if (blockNum == 0) { return; } result = value / blockNum;` |
| 数组越界 | 循环内添加 `if (index >= size) return;` | `if (offset1 >= hFusionBufLen_) { return; }` |
| 溢出保护 | 改为 int64_t，添加溢出检查 | `if (a > INT64_MAX - b) { return; } sum = a + b;` |
| 变量初始化 | 声明时赋初值 | `bool flag_ = false;` |
| 空指针保护 | `if (ptr == nullptr) return;` | `if (tilingData == nullptr) { return; }` |

#### Phase 3: Verify (验证阶段)

**输入**：
- 修复后的算子代码
- 算子名称列表

**执行**：
1. **编译验证**：
   ```bash
   cd training/ascendc
   bash build.sh -c "ascend910b"
   ```
   - 检查编译输出，确认 0 错误 0 警告
   - 验证生成的 run 包

2. **UT 验证**：
   ```bash
   cd build/test/ut
   ./transformer_op_host_ut --gtest_filter="*OpName*"
   ```
   - 统计通过/失败用例数
   - 分析失败原因
   - 如果失败率 > 0，回到 Fix 阶段

3. **ST 验证**（如果存在）：
   ```bash
   cd training/ascendc
   source <USER_HOME>/pkg/0211/vendors/omni_training_custom_ops/bin/set_env.bash
   pytest --device npu:* src/ops-transformer/mhc/[op_name]/tests/st/
   ```
   - 统计通过/失败用例数
   - 分析失败原因

**输出**：
- Markdown 格式验证报告
- 包含完整的编译日志、UT 日志、ST 日志摘要
- 最终结论：通过 [通过] 或需返工 [注意]

**示例输出**：
```markdown
# MHC算子编码红线整改 - 最终验证报告

## 验证结果概览

| 验证项 | 结果 | 通过率 |
|-------|------|--------|
| 编译验证 | [通过] 通过 | 100% |
| UT验证 | [通过] 通过 | 100% (98/98) |
| ST验证 | [通过] 通过 | 100% (6/6) |

## UT验证详情

### ai_infra_sinkhorn_grad
- Tiling UT: 34/34 [OK]
- 修复了 MIN_N_SIZE=1 后全部通过
```

---

## 4. Examples

### 示例 1: 完整工作流

**用户请求**：
```
对 training/ascendc/src/ops-transformer/mhc/ 路径下的 3 个算子进行编码红线整改并验证
```

**执行流程**：

1. **Review 阶段** - 扫描 3 个算子目录
   ```
   [ascendc-review] 正在检查 ai_infra_manifold_constrained_hyper_connection_pre_grad...
   发现 6 个问题: MHC-01 (P0), MHC-02 (P0), MHC-03 (P1), MHC-04 (P1), MHC-05 (P2), MHC-06 (P2)

   [ascendc-review] 正在检查 ai_infra_sinkhorn_grad...
   发现 4 个问题: SINK-01 (P0), SINK-02 (P1), SINK-03 (P2), SINK-04 (P3)

   [ascendc-review] 正在检查 manifold_constrained_hyper_connection_sinkhorn_enhance...
   发现 9 个问题: SKHE-01 (P0), SKHE-02 (P0), ...
   ```

2. **Fix 阶段** - 按优先级修复
   ```
   [ascendc-fix] 修复 MHC-01: 在 InitStage2() 添加 GetBlockNum() 除零保护
   [ascendc-fix] 修复 MHC-02: 在 gatherOffsetBuf_ 访问前添加边界检查
   ...
   [ascendc-fix] 生成整改报告: review/ai_infra_manifold_constrained_hyper_connection_pre_grad_红线整改.md
   ```

3. **Verify 阶段** - 三层验证
   ```
   [ascendc-verify] 编译验证: [通过] 0 错误 0 警告
   [ascendc-verify] UT 验证: [注意] 3 个失败用例 (MAX_D=4096 过严)

   [用户反馈] 修复 MAX_D 限制问题

   [ascendc-verify] 重新编译验证: [通过] 通过
   [ascendc-verify] UT 验证: [通过] 98/98 全部通过
   [ascendc-verify] ST 验证: [通过] 6/6 全部通过

   [ascendc-verify] 生成最终报告: review/MHC算子编码红线整改_最终验证报告.md
   ```

### 示例 2: 仅 Review（不修复）

**用户请求**：
```
只检查 ai_infra_sinkhorn_grad 算子的编码红线问题，不要修复
```

**执行**：
```
[ascendc-review] 检查模式: Review Only
[ascendc-review] 正在扫描 op_kernel/ai_infra_sinkhorn_grad_generalized.h...
[ascendc-review] 正在扫描 op_host/ai_infra_sinkhorn_grad_tiling.cpp...

发现 4 个问题:
  [P0] SINK-01: Init函数缺少空指针保护 (line 63-73)
  [P1] SINK-02: colNormGrad/rowNormGrad 缺少特殊值处理 (line 324-396)
  [P2] SINK-03: nSize范围校验缺失 (line 285-306)
  [P3] SINK-04: 数组访问缺少边界注释 (line 346-351)

生成报告: review/ai_infra_sinkhorn_grad_review_only.md
```

### 示例 3: 验证失败后的迭代修复

**场景**：UT 验证发现新问题

**用户请求**：
```
刚才 UT 失败了，修复 MIN_N_SIZE 参数，然后重新验证
```

**执行**：
```
[ascendc-fix] 定位问题: ai_infra_sinkhorn_grad_tiling.cpp:314
[ascendc-fix] 当前代码: constexpr int64_t MIN_N_SIZE = 2;
[ascendc-fix] 修复方案: 改为 MIN_N_SIZE = 1  // 允许 n=1 边界场景
[ascendc-fix] 应用修复...

[ascendc-verify] 重新编译...
[ascendc-verify] 运行 UT...
[ascendc-verify] [通过] 34/34 全部通过 (包括 case_normal_small_n)

[ascendc-verify] 更新验证报告: review/MHC算子编码红线整改_最终验证报告.md
```

---

## 5. Dependencies

### 必需工具

1. **CANN 环境**：
   ```bash
   source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
   ```

2. **Python 依赖**：
   ```bash
   pip3 install decorator scipy attrs cloudpickle ml-dtypes psutil tornado pandas expecttest
   ```

3. **编译工具链**：
   - bisheng 编译器
   - CMake 3.16+

### 必需文件

- 算子源代码（`op_kernel/`, `op_host/`, `docs/`）
- 测试代码（`tests/ut/`, `tests/st/`）
- 编译脚本（`build.sh`）

---

## 6. Configuration

可在项目根目录创建 `.ascendc-review.json` 配置文件：

```json
{
  "rules": {
    "division_by_zero": true,
    "array_bounds": true,
    "overflow_protection": true,
    "null_pointer": true,
    "variable_initialization": true,
    "special_values": true,
    "input_validation": true,
    "int64_offsets": true
  },
  "verify": {
    "compile": true,
    "ut": true,
    "st": true,
    "required_pass_rate": 100
  },
  "output": {
    "review_dir": "review",
    "report_format": "markdown"
  }
}
```

---

## 7. Sub-Skills

本 skill 依赖以下子 skills（自动调用）：

- **ascendc-review.md** - 编码红线检查引擎
- **ascendc-fix.md** - 标准修复模式应用
- **ascendc-verify.md** - 编译和测试验证

用户可独立使用这些子 skills，或通过本 skill 统一调度。

---

## 8. Tips & Tricks

### 提高 Review 效率

1. **批量检查**：
   ```
   对 mhc 目录下所有算子执行 ascendc-code-review
   ```

2. **增量检查**（仅检查修改过的文件）：
   ```bash
   git diff --name-only HEAD~1 | grep '\.cpp\|\.h' | xargs ascendc-review
   ```

3. **CI/CD 集成**：
   ```yaml
   # .gitlab-ci.yml
   code_review:
     script:
       - ascendc-code-review --ci-mode --fail-on-p0
   ```

### 验证优化

1. **并行 UT 执行**：
   ```bash
   ./transformer_op_host_ut --gtest_filter="*Sinkhorn*" &
   ./transformer_op_host_ut --gtest_filter="*MHC*" &
   wait
   ```

2. **跳过 ST 验证**（加速开发）：
   ```
   使用 ascendc-verify 技能验证算子，跳过 ST 测试
   ```

### 文档生成

1. **汇总报告**：
   所有单个算子的整改报告会自动汇总到 `review/README.md`

2. **生成 Prompt 和 Skills**（供团队分发）：
   ```
   生成 ascendc-code-review 的 Prompt 和 Skill 文档，放到 review/ 目录
   ```

---

## 9. Known Issues & Limitations

1. **编译环境依赖**：
   - 必须在 910B3 NPU 机器上运行（或有 CANN 模拟器）
   - 无法在纯 x86 环境下验证

2. **UT 测试覆盖率**：
   - 当前仅检查通过率，未统计代码覆盖率
   - 建议配合 `lcov` 工具检查覆盖率

3. **ST 测试环境**：
   - 部分算子没有 ST 测试（会跳过）
   - ST 测试依赖 NPU 硬件，无法离线验证

4. **False Positives**：
   - 某些"除零"场景可能是开发者已验证安全的（需人工判断）
   - 建议在代码中添加注释说明："// Safe: already validated in Tiling"

---

## 10. Next Steps

完成 Review + Fix + Verify 后：

1. **提交代码**：
   ```bash
   git add .
   git commit -m "fix: 修复编码红线问题 (SINK-01, SINK-03等)"
   git push
   ```

2. **团队复盘**：
   - 使用生成的整改报告进行团队分享
   - 将典型问题加入团队编码规范培训

3. **持续集成**：
   - 将 `ascendc-code-review` 集成到 CI/CD 流程
   - 在 pre-commit hook 中强制执行编码红线检查

---

**Version**: 1.0.0
**Last Updated**: 2026-02-11
**Maintainer**: Claude Code User
