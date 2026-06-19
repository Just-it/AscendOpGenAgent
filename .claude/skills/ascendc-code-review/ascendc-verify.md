# ascendc-verify

**Tagline**: 编译和测试三层验证引擎

**Triggers**:
- 当 `ascendc-code-review` 主 skill 进入 Phase 3 时自动调用
- User 明确说"验证修复结果"时独立使用

---

## 1. Introduction

`ascendc-verify` 负责对修复后的算子代码进行三层验证：编译、UT、ST，确保修复正确且无引入新问题。

**验证层级**:
1. **编译验证** - 确认代码可编译通过（0 错误 0 警告）
2. **UT 验证** - 确认单元测试 100% 通过
3. **ST 验证** - 确认集成测试 100% 通过（如果存在）

**目标**:
- [成功] 100% 编译成功
- [通过] 100% UT 通过率
- [通过] 100% ST 通过率

---

## 2. Usage

### 完整三层验证

```
使用 ascendc-verify 验证 ai_infra_sinkhorn_grad 算子
```

### 仅编译验证

```
使用 ascendc-verify 仅验证 ai_infra_sinkhorn_grad 的编译（跳过测试）
```

### 批量验证

```
使用 ascendc-verify 验证 training/ascendc/src/ops-transformer/mhc/ 下所有算子
```

---

## 3. How It Works

### 验证流程

```
1. 编译验证 (Layer 1)
   └─> cd training/ascendc
   └─> bash build.sh -c "ascend910b"
   └─> 检查编译输出：0 错误 0 警告
   └─> 验证生成的 run 包

2. 安装自定义算子包
   └─> bash output/CANN-*.run --install
   └─> 验证安装成功

3. 编译 PTA 扩展
   └─> cd torch_ops_extension
   └─> bash build_and_install.sh
   └─> 验证扩展安装成功

4. UT 验证 (Layer 2)
   └─> cd build/test/ut
   └─> ./transformer_op_host_ut --gtest_filter="*OpName*"
   └─> 统计通过/失败用例数
   └─> 如果失败率 > 0，标记失败并分析原因

5. ST 验证 (Layer 3)
   └─> source set_env.bash
   └─> cd src/ops-transformer/mhc/[op_name]/tests/st/
   └─> pytest --device npu:* .
   └─> 统计通过/失败用例数
   └─> 如果失败率 > 0，标记失败并分析原因

6. 生成验证报告
   └─> Markdown 格式，包含所有验证日志和结论
```

### 验证标准

#### Layer 1: 编译验证

**通过条件**:
- [通过] 编译命令返回码 = 0
- [通过] 编译输出无 "error" 关键字
- [通过] 编译输出无 "warning" 关键字
- [通过] 生成 run 包文件存在且大小 > 0

**失败处理**:
- 提取编译错误信息
- 定位到具体文件和行号
- 建议回到 `ascendc-fix` 阶段修复

#### Layer 2: UT 验证

**通过条件**:
- [通过] UT 命令返回码 = 0
- [通过] 所有测试用例 PASSED（无 FAILED）
- [通过] 通过率 = 100%

**失败处理**:
- 提取失败用例名称和错误信息
- 分析失败原因（如参数校验过严、逻辑错误等）
- 给出修复建议（如调整 MIN_N_SIZE、MAX_D 等）
- 建议回到 `ascendc-fix` 阶段修复

**示例失败分析**:
```
失败用例: AiInfraSinkhornGradTiling.case_normal_small_n
错误信息: Expected n=1 to pass, but CheckInputShape returned FAILED
原因分析: MIN_N_SIZE=2 过严，拒绝了 n=1 的边界场景
修复建议: 将 ai_infra_sinkhorn_grad_tiling.cpp:314 行改为 MIN_N_SIZE=1
```

#### Layer 3: ST 验证

**通过条件**:
- [通过] pytest 返回码 = 0
- [通过] 所有测试 PASSED
- [通过] 通过率 = 100%

**失败处理**:
- 提取失败测试名称和 traceback
- 分析是否为环境问题（缺依赖）或代码问题
- 给出修复建议

**示例环境问题**:
```
错误: ModuleNotFoundError: No module named 'pandas'
分析: ST 测试依赖 pandas 库
修复: pip3 install pandas
```

---

## 4. Examples

### 示例 1: 完整验证流程

**用户请求**:
```
使用 ascendc-verify 验证 ai_infra_sinkhorn_grad 算子
```

**执行日志**:

#### Step 1: 编译验证

```bash
[ascendc-verify] 开始编译验证...
[ascendc-verify] 执行命令: cd training/ascendc && bash build.sh -c "ascend910b"

[编译输出]
[INFO] Building operator: ai_infra_sinkhorn_grad
[INFO] Tiling compilation... OK
[INFO] Kernel compilation... OK
[INFO] Linking... OK
[SUCCESS] Build completed successfully

[ascendc-verify] [通过] 编译通过 (0 错误 0 警告)
[ascendc-verify] [通过] 生成 run 包: output/CANN-omni_training_custom_ops--linux.aarch64.run (4.7MB)
```

#### Step 2: 安装验证

```bash
[ascendc-verify] 安装自定义算子包...
[ascendc-verify] 执行命令: bash output/CANN-*.run --install --install-path=<USER_HOME>/pkg/0211/

[安装输出]
[ops_custom] [INFO] upgrade framework
[ops_custom] [INFO] upgrade op proto
[ops_custom] [INFO] upgrade op impl
[ops_custom] [INFO] upgrade op api
SUCCESS

[ascendc-verify] [成功] 安装成功
```

#### Step 3: UT 验证

```bash
[ascendc-verify] 开始 UT 验证...
[ascendc-verify] 执行命令: cd build/test/ut && ./transformer_op_host_ut --gtest_filter="*Sinkhorn*"

[UT 输出]
[==========] Running 34 tests from 1 test suite.
[----------] 34 tests from AiInfraSinkhornGradTiling
[ RUN      ] AiInfraSinkhornGradTiling.case_normal_3d
[       OK ] AiInfraSinkhornGradTiling.case_normal_3d (0 ms)
[ RUN      ] AiInfraSinkhornGradTiling.case_normal_4d
[       OK ] AiInfraSinkhornGradTiling.case_normal_4d (0 ms)
[ RUN      ] AiInfraSinkhornGradTiling.case_normal_small_n
[  FAILED  ] AiInfraSinkhornGradTiling.case_normal_small_n (0 ms)
[ RUN      ] AiInfraSinkhornGradTiling.case_normal_large_n
[       OK ] AiInfraSinkhornGradTiling.case_normal_large_n (0 ms)
...
[  PASSED  ] 33 tests.
[  FAILED  ] 1 test.

[ascendc-verify] [注意] UT 验证失败 (33/34 通过，97.1%)
[ascendc-verify] 失败用例分析:
  - case_normal_small_n: 测试参数 n=1，但 MIN_N_SIZE=2 拒绝此输入
  - 建议修复: ai_infra_sinkhorn_grad_tiling.cpp:314 改为 MIN_N_SIZE=1
```

#### Step 4: 修复后重新验证

```bash
[用户执行修复]
修改 MIN_N_SIZE 从 2 到 1

[ascendc-verify] 重新编译...
[ascendc-verify] [通过] 编译通过

[ascendc-verify] 重新运行 UT...
[UT 输出]
[==========] Running 34 tests from 1 test suite.
[  PASSED  ] 34 tests.

[ascendc-verify] [通过] UT 验证通过 (34/34, 100%)
```

#### Step 5: ST 验证

```bash
[ascendc-verify] 开始 ST 验证...
[ascendc-verify] 执行命令: source set_env.bash && pytest --device npu:* tests/st/

[ST 输出]
============================= test session starts ==============================
collected 2 items

test_sinkhorn_grad.py::TestCustomSinkhornGrad::test_sinkhorn_grad_netshape PASSED [ 50%]
test_sinkhorn_grad.py::TestCustomSinkhornGrad::test_sinkhorn_grad_random PASSED [100%]

============================== 2 passed in 12.62s ==============================

[ascendc-verify] [通过] ST 验证通过 (2/2, 100%)
```

#### Step 6: 生成最终报告

```bash
[ascendc-verify] 生成验证报告: review/ai_infra_sinkhorn_grad_验证报告.md
```

### 示例 2: 批量验证 3 个算子

**用户请求**:
```
使用 ascendc-verify 验证 mhc 目录下所有算子
```

**执行流程**:

```
[ascendc-verify] 批量验证模式
[ascendc-verify] 发现 3 个算子:
  1. ai_infra_manifold_constrained_hyper_connection_pre_grad
  2. ai_infra_sinkhorn_grad
  3. manifold_constrained_hyper_connection_sinkhorn_enhance

[ascendc-verify] 第 1 个算子验证中...
  [通过] 编译通过
  [注意] UT 失败 (17/20, 85%) - 3 个 D=6144/8192 用例失败
  建议: 调整 MAX_D 限制

[ascendc-verify] 第 2 个算子验证中...
  [通过] 编译通过
  [注意] UT 失败 (33/34, 97.1%) - 1 个 n=1 用例失败
  建议: 调整 MIN_N_SIZE

[ascendc-verify] 第 3 个算子验证中...
  [通过] 编译通过
  [通过] UT 通过 (14/14, 100%)
  [通过] ST 通过 (3/3, 100%)

[ascendc-verify] 汇总报告:
  - 编译: 3/3 通过 (100%)
  - UT: 64/68 通过 (94.1%)
  - ST: 3/3 通过 (100%)

[ascendc-verify] 需要修复 2 个算子的 UT 问题
```

---

## 5. Output Format

每次验证后生成详细验证报告（Markdown 格式），包含以下章节：

### 报告模板

```markdown
# [算子名] - 验证报告

**验证日期**: 2026-02-11
**验证人**: Claude Code
**验证状态**: [通过] 全部通过 / [注意] 部分失败

---

## 验证结果概览

| 验证项 | 结果 | 通过率 |
|-------|------|--------|
| **编译验证** | [通过] 通过 | 100% |
| **UT验证** | [通过] 通过 | 100% (X/X) |
| **ST验证** | [通过] 通过 | 100% (X/X) |

---

## 一、编译验证

### 1.1 编译配置

- **平台**: Ascend 910B3
- **CANN版本**: 8.5.0
- **编译参数**: `-c "ascend910b"`

### 1.2 编译结果

#### [成功] 编译成功

```bash
执行命令: cd training/ascendc && bash build.sh -c "ascend910b"

[编译日志摘要]
[INFO] Building operator: [算子名]
[INFO] Tiling compilation... OK
[INFO] Kernel compilation... OK
[SUCCESS] Build completed successfully
```

#### 生成产物

```
输出目录: <OMNI_OPS_ROOT>/omni-ops/training/ascendc/output/
Run包文件: CANN-omni_training_custom_ops--linux.aarch64.run
文件大小: 4.7 MB
安装路径: <USER_HOME>/pkg/0211/
```

---

## 二、UT 验证

### 2.1 测试配置

- **测试框架**: Google Test
- **测试命令**: `./transformer_op_host_ut --gtest_filter="*OpName*"`

### 2.2 测试结果

#### [通过] 全部通过 (X/X)

```
[==========] X tests from [TestSuite]
[  PASSED  ] X tests [OK]
```

**测试覆盖**:
- [通过] 正常场景: [描述]
- [通过] 边界场景: [描述]
- [通过] 错误场景: [描述]

### 2.3 UT 日志摘要

```
[----------] X tests from [TestSuite]
[ RUN      ] [TestCase1]
[       OK ] [TestCase1] (0 ms)
[ RUN      ] [TestCase2]
[       OK ] [TestCase2] (0 ms)
...
[  PASSED  ] X tests.
```

---

## 三、ST 验证

### 3.1 测试配置

- **测试框架**: pytest
- **测试命令**: `pytest --device npu:* tests/st/`

### 3.2 测试结果

#### [通过] 全部通过 (X/X)

```
============================= test session starts ==============================
collected X items

test_xxx.py::test_case1 PASSED [ XX%]
test_xxx.py::test_case2 PASSED [100%]

============================== X passed in XX.XXs ==============================
```

**测试覆盖**:
- [通过] 网络真实 shape 测试
- [通过] NPU vs CPU 精度对比
- [通过] 功能正确性验证

---

## 四、问题修复记录

（如果验证过程中发现问题并修复）

### 修复 1: [问题描述]

**问题**: [详细描述]
**修复**: [修复方案]
**效果**: [修复后结果]

---

## 五、验证结论

### 5.1 验证完成情况

| 验证项 | 要求 | 实际完成 | 状态 |
|-------|------|----------|------|
| 编译通过 | 100% | 100% | [通过] |
| UT通过率 | 100% | 100% (X/X) | [通过] |
| ST通过率 | 100% | 100% (X/X) | [通过] |

### 5.2 质量保证

[通过] **编译**: 0错误 0警告
[通过] **UT**: X/X 通过 (100%)
[通过] **ST**: X/X 通过 (100%)

### 5.3 推荐

**[通过] 通过验收**

---

**报告生成时间**: 2026-02-11 17:00
**验证负责人**: Claude Code
**审核建议**: [通过] **通过**
```

---

## 6. Dependencies

### 必需工具

1. **CANN 环境**:
   ```bash
   source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
   ```

2. **Python 依赖** (ST 测试):
   ```bash
   pip3 install pandas expecttest pytest
   ```

3. **编译工具链**:
   - bisheng 编译器
   - CMake 3.16+

### 必需文件

- 修复后的算子源代码
- 测试代码（`tests/ut/`, `tests/st/`）
- 编译脚本（`build.sh`）

---

## 7. Configuration

可在 `.ascendc-review.json` 中配置验证行为：

```json
{
  "verify": {
    "compile": true,             // 是否进行编译验证
    "ut": true,                  // 是否进行 UT 验证
    "st": true,                  // 是否进行 ST 验证
    "required_pass_rate": 100,   // 要求的通过率（%）
    "fail_fast": false,          // 是否遇到失败立即停止
    "retry_on_failure": 1,       // 失败后重试次数
    "timeout": {
      "compile": 600,            // 编译超时（秒）
      "ut": 300,                 // UT 超时（秒）
      "st": 600                  // ST 超时（秒）
    }
  }
}
```

---

## 8. Known Limitations

1. **环境依赖**:
   - 必须在 910B3 NPU 机器上运行
   - 无法在纯 x86 环境下验证 NPU 算子

2. **测试覆盖率**:
   - 当前仅检查通过率，未统计代码覆盖率
   - 建议配合 `lcov` 工具检查覆盖率

3. **ST 测试可用性**:
   - 部分算子没有 ST 测试（会跳过）
   - ST 测试依赖 NPU 硬件，无法离线验证

4. **并行验证**:
   - 批量验证时按顺序执行，暂不支持并行
   - 可考虑使用 `xargs -P` 实现并行

---

## 9. Tips & Tricks

### 加速编译

```bash
# 只编译特定算子
./build.sh -n "ai_infra_sinkhorn_grad"

# 只编译 host 代码（跳过 kernel）
./build.sh -b host

# 使用增量编译
./build.sh --incremental
```

### UT 调试

```bash
# 运行特定测试用例
./transformer_op_host_ut --gtest_filter="AiInfraSinkhornGradTiling.case_normal_small_n"

# 输出详细日志
./transformer_op_host_ut --gtest_filter="*Sinkhorn*" --gtest_also_run_disabled_tests
```

### ST 调试

```bash
# 运行特定测试
pytest --device npu:* tests/st/test_sinkhorn_grad.py::TestCustomSinkhornGrad::test_sinkhorn_grad_netshape

# 输出详细日志
pytest --device npu:* tests/st/ -v -s
```

### 验证失败快速定位

当 UT 或 ST 失败时：

1. **定位失败用例**:
   ```bash
   # 查看失败用例名称
   grep "FAILED" /tmp/ut_log.txt
   ```

2. **提取错误信息**:
   ```bash
   # 查看失败原因
   grep -A 10 "FAILED" /tmp/ut_log.txt
   ```

3. **分析修复建议**:
   - 参数校验过严 → 调整 MIN/MAX 范围
   - 逻辑错误 → 回到 `ascendc-fix` 阶段
   - 环境问题 → 安装缺失依赖

---

**Version**: 1.0.0
**Last Updated**: 2026-02-11
