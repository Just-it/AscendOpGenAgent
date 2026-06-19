# AscendC 编码红线 Review

## 功能描述

系统性地对 AscendC 算子代码进行编码红线审查和整改，生成详细的整改报告并实施修复。

## 使用场景

- 新开发的 AscendC 算子提交前的安全审查
- 现有算子的编码规范升级
- 编码红线问题的系统性排查和修复
- 代码 Review 和技术债务清理

## 输入参数

- `operator_path`: 算子路径（相对于仓库根目录）
- `operator_name`: 算子名称
- `review_output_dir`: 整改报告输出目录（默认为 `review/`）

## 执行流程

### 阶段1：系统性代码审查

使用 Task 工具的 Explore 子代理进行全面扫描：

```markdown
使用 Explore 子代理，thoroughness 设置为 "very thorough"，审查以下内容：

1. 编码红线扫描（7条必查）：
   - 红线1.1: 除零保护 - 搜索 `/`, `%`, `CeilDiv`, `GetBlockNum()`
   - 红线1.2: 数组越界 - 搜索 `SetValue`, `[index]`, `offset++`
   - 红线1.3: 溢出保护 - 搜索 `uint32_t offset`, `+`, `*`, `sizeof`
   - 红线1.4: 变量初始化 - 检查成员变量和局部变量声明
   - 红线1.5: 空指针保护 - 搜索 `->`, `*ptr`, 指针参数
   - 红线1.6: 资源匹配 - 搜索 `InitBuffer`, `AllocTensor`, `FreeTensor`
   - 红线1.7: 数据竞争 - 搜索 `GetBlockIdx`, 全局变量写入

2. TopN 问题扫描（重点关注2.1-2.3）：
   - TopN2.1: 特殊值处理 - 检查文档是否说明 nan/inf/±0
   - TopN2.2: 输入校验 - 检查 Tiling 中 shape/attr 验证
   - TopN2.3: int64 偏移 - 检查 GM 偏移是否使用 int64_t
```

### 阶段2：生成整改报告

创建详细的整改报告文档，包含：

**报告模板**:

```markdown
# {算子名称} 编码红线整改报告

## 一、整改概述

### 1.1 算子信息
- 算子名称：
- 功能描述：
- 所在路径：
- 审查日期：

### 1.2 问题统计
| 严重性 | 数量 | 修复时限 |
|--------|------|----------|
| P0 - 极严重 | X | 立即修复 |
| P1 - 严重 | X | 24小时内 |
| P2 - 中等 | X | 1周内 |
| P3 - 低 | X | 2周内 |

## 二、问题清单与修复方案

{每个问题的详细分析和修复方案}

## 三、整改成果

### 3.1 编码红线覆盖率
### 3.2 修改统计

## 四、验证建议
```

### 阶段3：实施修复

按照 **P0 → P1 → P2 → P3** 优先级修复：

使用标准修复模式（详见下文"标准修复模式"）。

为每个问题创建 Task 跟踪进度。

### 阶段4：文档更新

如果涉及特殊值处理或输入约束，更新 `docs/npu_{算子名称}.md`。

### 阶段5：验证

生成验证清单：

```markdown
## 整改验证清单

### 编码红线核查（必须100%通过）
- [ ] 1.1 除零保护
- [ ] 1.2 数组越界
- [ ] 1.3 溢出保护
- [ ] 1.4 变量初始化
- [ ] 1.5 空指针保护
- [ ] 1.6 资源匹配
- [ ] 1.7 数据竞争

### 验证步骤
1. [ ] 编译通过：`./build.sh -n "{算子名称}"`
2. [ ] 单元测试通过：`./build.sh -u -n "{算子名称}"`
```

## 标准修复模式

### 1. 除零保护

```cpp
// 模式1: 检查-返回
if (divisor == 0) {
    return ERROR_CODE;
}
result = value / divisor;

// 模式2: 检查-默认值
uint32_t blockNum = GetBlockNum();
if (blockNum == 0) {
    cubeDealnDPeCore_ = defaultValue;
    return;
}
```

### 2. 数组越界保护

```cpp
// 每次访问前检查
for (...) {
    if (index >= bufferSize) {
        return ERROR_OUT_OF_BOUNDS;
    }
    buffer[index++] = value;
}
```

### 3. 溢出保护

```cpp
// 类型改为 int64_t/uint64_t
int64_t offset = base;

// 加法前检查
if (offset > INT64_MAX - increment) {
    return ERROR_OVERFLOW;
}
offset += increment;

// 乘法前检查
if (a > 0 && b > INT64_MAX / a) {
    return ERROR_OVERFLOW;
}
```

### 4. 变量初始化

```cpp
// 类内初始化（推荐）
class MyClass {
    bool flag_ = false;
    int count_ = 0;
    float value_ = 0.0f;
};
```

### 5. 空指针保护

```cpp
// 函数入口集中检查
if (ptr == nullptr) {
    return;
}
if (tilingData == nullptr) {
    return;
}
```

### 6. 输入校验

```cpp
// Tiling 中校验
if (param < MIN || param > MAX) {
    OP_LOGE(..., "Invalid param=%d, expected [%d, %d]", param, MIN, MAX);
    return ge::GRAPH_FAILED;
}
```

### 7. 边界注释

```cpp
// 【编码红线1.2修复】Buffer大小验证注释
// xLocal大小: nnt = n_ * n_ * tAlign_ 元素
// 循环索引验证: i ∈ [0, n_), j ∈ [0, n_)
// 访问模式: xLocal[i * nt + j * tAlign_]
// 最大索引: (n_-1)*nt + (n_-1)*tAlign_ + tAlign_ - 1 = nnt - 1 [PASS]
```

## 关键成功因素

1. **系统性审查** - 使用 Task 工具的 Explore 子代理
2. **优先级驱动** - 严格按照 P0 → P1 → P2 → P3
3. **标准模式** - 使用上述标准修复模式
4. **文档先行** - 先生成报告，再实施修复
5. **进度追踪** - 使用 Task 工具跟踪
6. **验证闭环** - 编译和测试验证

## 调用示例

**单个算子整改**:

```
使用 ascendc-code-review skill 对 training/ascendc/src/ops-transformer/mhc/manifold_constrained_hyper_connection_sinkhorn_enhance 算子进行 Review 和整改
```

**批量整改**:

```
使用 ascendc-code-review skill 对 training/ascendc/src/ops-transformer/mhc/ 路径下的所有算子进行 Review 和整改
```

## 输出产物

1. **整改报告 Markdown 文档** - `review/{算子名称}_红线整改.md`
2. **修复后的源代码** - 原地修改
3. **更新的算子文档** - `docs/npu_{算子名称}.md`（如适用）
4. **验证清单** - 包含在整改报告中

## 七大编码口诀

1. **除法必检查，0是拦路虎**（红线1.1）
2. **下标递增前，边界要问遍**（红线1.2）
3. **内存偏移量，int64保安康**（红线1.3）
4. **变量必有值，不能靠运气**（红线1.4）
5. **指针用之前，非空要验遍**（红线1.5）
6. **外来数据毒，校验是解药**（TopN2.2）
7. **特殊值万千，文档说在前**（TopN2.1）

## 版本信息

- **版本**: v1.0
- **创建日期**: 2026-02-11
- **维护者**: [待填写]
- **适用范围**: 所有 AscendC 算子

## 相关资源

- **编码红线完整规范**: `编码红线.md`
- **AscendC 开发指南**: `.claude/skills/ascendc-dev-knowledge/references/`
- **整改案例库**: `review/` 目录下的各算子整改报告
