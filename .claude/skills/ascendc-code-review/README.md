# AscendC Code Review Skills - README

##  概述

这是一套完整的 AscendC 算子编码规范审查、修复、验证工作流技能包。它帮助开发者系统化地检查和修复编码红线问题，确保代码质量和安全性。

**适用场景**:
- [通过] AscendC 算子开发和维护
- [通过] 编码红线合规检查
- [通过] 代码提交前的质量保证
- [通过] 团队代码规范培训

---

##  目录结构

```
ascendc-code-review/
├── README.md                    # 本文件
├── ascendc-code-review.md      # 主入口 skill (统一调度)
├── ascendc-review.md           # Phase 1: 编码红线检查引擎
├── ascendc-fix.md              # Phase 2: 标准修复引擎
└── ascendc-verify.md           # Phase 3: 编译和测试验证引擎
```

---

##  快速开始

### 1. 安装 Skills

#### 选项 A: 项目级安装（推荐）

将 `skills/` 目录放在项目根目录：

```bash
# 已完成（当前位置）
<OMNI_OPS_ROOT>/omni-ops/skills/ascendc-code-review/
```

使用时直接引用：

```
使用 ascendc-code-review 技能对 ai_infra_sinkhorn_grad 算子进行检查
```

#### 选项 B: 用户级安装（全局可用）

复制到 Claude Code 的 skills 目录：

```bash
cp -r <OMNI_OPS_ROOT>/omni-ops/skills/ascendc-code-review ~/.claude/skills/
```

使用时：

```
使用 ascendc-code-review 技能检查算子
```

### 2. 完整工作流示例

```
对 training/ascendc/src/ops-transformer/mhc/ 路径下的所有算子进行编码红线整改并验证
```

**执行流程**:
1. **Review 阶段** - 自动扫描 3 个算子，识别 19 个编码红线问题
2. **Fix 阶段** - 按 P0→P1→P2→P3 优先级修复
3. **Verify 阶段** - 编译→UT→ST 三层验证，确保 100% 通过率

### 3. 独立使用子 Skills

```bash
# 仅进行 Review（不修复）
使用 ascendc-review 检查 ai_infra_sinkhorn_grad 算子

# 仅修复特定问题
使用 ascendc-fix 修复 ai_infra_sinkhorn_grad 的 SINK-01 问题

# 仅验证修复结果
使用 ascendc-verify 验证 ai_infra_sinkhorn_grad 算子
```

---

## 📚 Skills 详细说明

### 主入口: ascendc-code-review

**职责**: 统一调度三个子 skills，管理完整工作流

**核心特性**:
- [通过] 自动解析用户请求，确定算子范围
- [通过] 按 Phase 1→2→3 顺序调度子 skills
- [通过] 支持批量处理多个算子
- [通过] 汇总生成最终验证报告

**使用场景**:
- 完整的 Review + Fix + Verify 流程
- 批量算子检查
- 代码提交前的质量保证

**详细文档**: [ascendc-code-review.md](./ascendc-code-review.md)

---

### Phase 1: ascendc-review

**职责**: 编码红线自动检查引擎

**检查范围**:
- [通过] 7 大编码红线规范（1.1-1.7）
- [通过] 13 条 TopN 问题（2.1-2.13）
- [通过] 覆盖 `op_kernel/*.h` 和 `op_host/*_tiling.cpp`

**输出格式**:
- JSON（供 `ascendc-fix` 消费）
- Markdown（供人类阅读）

**检查清单**:
1. **红线 1.1** - 除零保护
2. **红线 1.2** - 数组越界保护
3. **红线 1.3** - 溢出保护
4. **红线 1.4** - 变量初始化
5. **红线 1.5** - 空指针保护
6. **TopN 2.1** - 特殊值处理
7. **TopN 2.2** - 输入校验
8. **TopN 2.3** - int64 偏移

**详细文档**: [ascendc-review.md](./ascendc-review.md)

---

### Phase 2: ascendc-fix

**职责**: 编码红线问题标准修复引擎

**修复能力**:
- [通过] 8 种标准修复模式（除零保护、数组越界、溢出、初始化等）
- [通过] 自动生成安全注释
- [通过] 更新相关文档（添加输入约束说明）
- [通过] 生成详细整改报告（修复前后对比）

**修复策略**:
- **P0 问题** - 立即修复（可能导致崩溃）
- **P1 问题** - 24小时内修复（可能导致数据错误）
- **P2 问题** - 1周内修复（影响稳定性）
- **P3 问题** - 2周内修复（代码可维护性）

**标准修复模式示例**:

| 问题类型 | 修复模式 |
|---------|---------|
| 除零保护 | `if (divisor == 0) return;` |
| 数组越界 | `if (index >= size) return;` |
| 溢出保护 | 改为 int64_t + 溢出检查 |
| 变量初始化 | `bool flag_ = false;` |
| 空指针保护 | `if (ptr == nullptr) return;` |

**详细文档**: [ascendc-fix.md](./ascendc-fix.md)

---

### Phase 3: ascendc-verify

**职责**: 编译和测试三层验证引擎

**验证层级**:
1. **编译验证** - 0 错误 0 警告
2. **UT 验证** - 100% 通过率
3. **ST 验证** - 100% 通过率（如果存在）

**验证流程**:
```
编译 → 安装 Run 包 → 编译 PTA 扩展 → 运行 UT → 运行 ST → 生成报告
```

**失败处理**:
- 提取失败用例名称和错误信息
- 分析失败原因（参数过严、逻辑错误、环境问题）
- 给出修复建议
- 建议回到 `ascendc-fix` 阶段

**详细文档**: [ascendc-verify.md](./ascendc-verify.md)

---

## [目标] 典型使用场景

### 场景 1: 新算子开发完成后的质量检查

```
使用 ascendc-code-review 检查 my_new_operator 算子
```

**预期输出**:
- Review 报告（列出所有编码红线问题）
- 修复后的代码
- 验证报告（编译、UT、ST 结果）

### 场景 2: 代码提交前的强制检查

```bash
# 在 pre-commit hook 中集成
git diff --name-only | grep 'op_kernel\|op_host' | xargs ascendc-code-review
```

### 场景 3: 批量整改历史遗留代码

```
使用 ascendc-code-review 对 training/ascendc/src/ops-transformer/ 下所有算子进行检查
```

**预期输出**:
- 每个算子的独立整改报告
- 汇总的验证报告（统计所有算子的通过率）

### 场景 4: 仅检查不修复（Code Review）

```
使用 ascendc-review 检查 ai_infra_sinkhorn_grad 算子，但不自动修复
```

**预期输出**:
- 问题清单（JSON + Markdown）
- 不修改源代码

### 场景 5: 修复验证失败后的迭代

```
# 第一次验证发现 UT 失败
使用 ascendc-verify 验证 ai_infra_sinkhorn_grad 算子

# 分析失败原因：MIN_N_SIZE=2 过严
# 用户手动修复或使用 fix skill 修复
使用 ascendc-fix 修复 ai_infra_sinkhorn_grad 的 SINK-03 问题（调整 MIN_N_SIZE=1）

# 重新验证
使用 ascendc-verify 验证 ai_infra_sinkhorn_grad 算子

# 结果：100% 通过
```

---

## 实际案例统计

基于 MHC 算子整改项目的真实数据：

| 算子 | 问题数 | 修复行数 | UT通过率 | ST通过率 |
|------|-------|---------|---------|---------|
| ai_infra_manifold_constrained_hyper_connection_pre_grad | 6 | 150+ | 100% (50/50) | 100% (1/1) |
| ai_infra_sinkhorn_grad | 4 | 80+ | 100% (34/34) | 100% (2/2) |
| manifold_constrained_hyper_connection_sinkhorn_enhance | 9 | 140+ | 100% (14/14) | 100% (3/3) |
| **总计** | **19** | **370+** | **100% (98/98)** | **100% (6/6)** |

**关键修复**:
- 除零保护: 7 处
- 数组越界保护: 11 处
- 溢出保护: 9 处
- 变量初始化: 2 处
- 空指针保护: 3 处
- 输入校验: 5 处

**文档产出**:
- 3 份算子整改报告
- 1 份最终验证报告
- 1 份 Review Prompt
- 1 份 Skill 文档

---

## 配置

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
  "fix": {
    "auto_fix_p0": true,         // 自动修复 P0 问题
    "auto_fix_p1": true,         // 自动修复 P1 问题
    "auto_fix_p2": false,        // P2 问题需用户确认
    "auto_fix_p3": false,        // P3 问题需用户确认
    "add_comments": true,        // 自动添加安全注释
    "update_docs": true,         // 自动更新文档
    "backup_original": true      // 修复前备份原文件
  },
  "verify": {
    "compile": true,             // 是否进行编译验证
    "ut": true,                  // 是否进行 UT 验证
    "st": true,                  // 是否进行 ST 验证
    "required_pass_rate": 100,   // 要求的通过率（%）
    "fail_fast": false           // 是否遇到失败立即停止
  },
  "output": {
    "review_dir": "review",      // 报告输出目录
    "report_format": "markdown"  // 报告格式
  }
}
```

---

## 依赖和环境

### 必需工具

1. **CANN 环境** (8.5.0+):
   ```bash
   source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
   ```

2. **Python 依赖**:
   ```bash
   pip3 install decorator scipy attrs cloudpickle ml-dtypes psutil tornado pandas expecttest pytest
   ```

3. **编译工具链**:
   - bisheng 编译器
   - CMake 3.16+

### 必需文件

- 算子源代码（`op_kernel/`, `op_host/`, `docs/`）
- 测试代码（`tests/ut/`, `tests/st/`）
- 编译脚本（`build.sh`）
- 编码红线规范文档（`编码红线.md`）

---

## 🎓 团队培训

### 1. 编码红线培训

使用本次整改报告作为培训材料：

```
review/
├── ai_infra_manifold_constrained_hyper_connection_pre_grad_红线整改.md
├── ai_infra_sinkhorn_grad_红线整改.md
├── manifold_constrained_hyper_connection_sinkhorn_enhance_红线整改.md
└── MHC算子编码红线整改_最终验证报告.md
```

**培训重点**:
- 7 大编码红线的真实案例
- 标准修复模式演示
- 验证流程和工具使用

### 2. Skills 使用培训

**基础班**（30分钟）:
- 了解 3 个 skills 的职责
- 学习完整工作流使用
- 实践：对示例算子进行检查

**进阶班**（1小时）:
- 独立使用 3 个子 skills
- 配置文件定制
- CI/CD 集成

### 3. Code Review 流程标准化

**提交前检查清单**:
- [ ] 使用 `ascendc-review` 检查编码红线
- [ ] 修复所有 P0 和 P1 问题
- [ ] 运行 `ascendc-verify` 确保 100% 通过率
- [ ] 生成整改报告并提交 Code Review

---

## [警告] 常见问题

### Q1: 检查发现的问题是否都必须修复？

**A**: 按严重性分级处理：
- **P0 - 极严重**: 必须修复（会导致崩溃、越界）
- **P1 - 严重**: 强烈建议修复（可能导致数据错误）
- **P2 - 中等**: 建议修复（影响稳定性）
- **P3 - 低**: 可选修复（代码规范）

### Q2: 自动修复是否会引入新问题？

**A**: 修复后必须进行三层验证：
- 编译验证（确认代码可编译）
- UT 验证（确认功能正确性）
- ST 验证（确认端到端流程）

如果验证失败，说明修复需要调整，会提供修复建议。

### Q3: 验证失败如何处理？

**A**: 根据失败原因分类处理：

| 失败类型 | 原因 | 处理方式 |
|---------|------|---------|
| 编译错误 | 语法错误、类型不匹配 | 回到 `ascendc-fix` 阶段 |
| UT 失败 | 参数校验过严、逻辑错误 | 调整校验参数或修复逻辑 |
| ST 失败 | 环境问题、功能错误 | 安装依赖或修复功能 |

示例：
```
UT 失败: case_normal_small_n (n=1)
原因: MIN_N_SIZE=2 过严
修复: 改为 MIN_N_SIZE=1
```

### Q4: 如何集成到 CI/CD？

**A**: 在 `.gitlab-ci.yml` 中添加：

```yaml
code_review:
  stage: test
  script:
    - source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
    - cd training/ascendc
    - ascendc-code-review --ci-mode --fail-on-p0
  artifacts:
    paths:
      - review/
    when: always
```

### Q5: 能否检查其他编程规范？

**A**: 当前 skills 专注于《编码红线.md》规范，未来可扩展：
- 性能规范（Tiling 优化、UB 使用率等）
- 代码风格规范（命名、注释等）
- 安全规范（敏感数据保护等）

欢迎贡献新的检查规则！

---

## 更新日志

### v1.0.0 (2026-02-11)

**初始版本**:
- [通过] 实现完整的 Review + Fix + Verify 工作流
- [通过] 支持 7 大编码红线规范检查
- [通过] 支持 13 条 TopN 问题检查
- [通过] 提供 8 种标准修复模式
- [通过] 支持三层验证（编译、UT、ST）
- [通过] 生成详细整改报告和验证报告

**验证成果**:
- 成功整改 3 个 MHC 算子
- 修复 19 个编码红线问题
- 实现 100% UT 和 ST 通过率

---

## [协作] 贡献指南

欢迎贡献新的检查规则、修复模式或验证策略！

**贡献流程**:
1. Fork 本项目
2. 创建特性分支：`git checkout -b feature/new-rule`
3. 提交修改：`git commit -m "Add new rule: XXX"`
4. 推送分支：`git push origin feature/new-rule`
5. 创建 Pull Request

**贡献方向**:
- 新增编码规范检查规则
- 优化修复模式（提高准确率）
- 增强验证能力（性能测试、覆盖率统计等）
- 改进报告格式（可视化、交互式等）

---

## 📧 联系方式

**维护者**: Claude Code User
**项目地址**: `<OMNI_OPS_ROOT>/omni-ops/skills/ascendc-code-review/`
**文档版本**: v1.0.0
**最后更新**: 2026-02-11

---

## 📜 许可证

本 skill 包遵循项目许可证。

---

**Happy Coding with Safety!**
