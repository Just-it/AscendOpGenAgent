---
name: kernelgen-conductor
description: >
  KernelGen Conductor Skill - 负责分析运行结果、判断是否需要重新生成代码、生成修复建议。
  核心功能：结果查看、错误分类、修复决策、建议生成。
argument-hint: >
  输入：verifier结果、错误日志、生成代码、执行历史。
  输出：是否重新生成、错误分类、修复建议。
---

# KernelGen Conductor Skill

## 功能概述

Conductor是KernelGen Agent的核心组件，负责：
1. **结果查看**：分析Verifier运行结果
2. **错误分类**：识别和分类错误类型
3. **修复决策**：判断是否重新生成代码
4. **建议生成**：为下一轮生成提供修复建议

## 工作流程

```
输入：Verifier结果 + 错误日志 + 代码 + 历史
    ↓
[结果分析]
    ↓
[错误分类] → SyntaxError/LogicError/ShapeError/DeviceError/TimeoutError
    ↓
[决策判断] → 是否重新生成？
    ↓
[建议生成] → 具体修复建议
    ↓
输出：决策 + 建议
```

## 核心功能

### 1. 结果分析 (Result Analysis)

分析Verifier返回的所有信息：

```python
class ResultAnalyzer:
    def analyze(self, verifier_output: dict) -> ResultAnalysis:
        """
        分析Verifier输出
        
        Returns:
            ResultAnalysis:
            - success: 是否成功
            - error_type: 错误类型
            - error_location: 错误位置
            - error_message: 错误信息
            - context: 上下文信息
        """
```

**分析维度**：
- 编译状态（成功/失败）
- 运行时状态（成功/失败/超时）
- 数值验证（通过/失败）
- 性能数据（如适用）

### 2. 错误分类 (Error Classification)

将错误归类到标准类别：

| 错误类别 | 识别特征 | 常见原因 |
|---------|---------|---------|
| **SyntaxError** | 编译失败，语法错误提示 | 语法错误、缩进问题、缺失符号 |
| **ImportError** | ModuleNotFoundError | 依赖缺失、路径错误、版本不兼容 |
| **LogicError** | 数值验证失败 | 算法错误、边界处理、精度问题 |
| **ShapeError** | Shape mismatch | Tensor维度处理错误、广播错误 |
| **DeviceError** | CUDA/Ascend错误 | 设备访问、内存分配、线程配置 |
| **TimeoutError** | 执行超时 | 死循环、低效实现、配置错误 |
| **MemoryError** | 内存不足 | 内存泄漏、过大分配、未释放 |

**分类逻辑**：
```python
class ErrorClassifier:
    def classify(self, error_message: str, error_traceback: str) -> ErrorCategory:
        # 基于关键词和模式匹配
        if "SyntaxError" in error_message or "IndentationError" in error_message:
            return ErrorCategory.SYNTAX_ERROR
        elif "ModuleNotFoundError" in error_message or "ImportError" in error_message:
            return ErrorCategory.IMPORT_ERROR
        elif "shape" in error_message.lower() or "dimension" in error_message.lower():
            return ErrorCategory.SHAPE_ERROR
        # ... 更多分类规则
```

### 3. 修复决策 (Regeneration Decision)

基于多因素决策是否重新生成：

**决策因素**：
- 当前迭代次数 vs 最大迭代次数
- 错误类型（可修复 vs 不可修复）
- 历史修复成功率
- 错误复杂度评估
- 是否为新类型错误

**决策逻辑**：
```python
class RegenerationDecision:
    def should_regenerate(
        self,
        current_iteration: int,
        max_iterations: int,
        error_category: ErrorCategory,
        error_history: List[ErrorRecord],
        code_quality_score: float
    ) -> Decision:
        """
        决策是否重新生成
        
        Returns:
            Decision:
            - action: "regenerate" | "abort" | "finish"
            - confidence: 决策置信度
            - reason: 决策理由
        """
```

**决策规则**：
| 条件 | 决策 | 理由 |
|------|------|------|
| current >= max | abort | 达到最大迭代次数 |
| SyntaxError + 首次 | regenerate | 语法错误通常可修复 |
| LogicError + 连续3次 | abort | 逻辑错误难以自动修复 |
| TimeoutError + 复杂度低 | regenerate | 超时可能是临时问题 |
| DeviceError + 配置错误 | abort | 环境问题需人工介入 |

### 4. 建议生成 (Suggestion Generation)

为下一轮KernelGen生成具体修复建议：

**建议模板**：

#### SyntaxError建议
```
错误分析：
- 类型：SyntaxError
- 位置：第{line}行
- 具体错误：{error_detail}

修复建议：
1. 检查语法：{specific_syntax_check}
2. 参考示例：{relevant_example}
3. 常见模式：{common_pattern}
```

#### LogicError建议
```
错误分析：
- 类型：LogicError
- 症状：数值验证失败
- 差异：{numerical_difference}

修复建议：
1. 检查算法：{algorithm_check}
2. 边界处理：{boundary_handling}
3. 精度控制：{precision_control}
4. 参考实现：{reference_implementation}
```

#### ShapeError建议
```
错误分析：
- 类型：ShapeError
- 预期形状：{expected_shape}
- 实际形状：{actual_shape}

修复建议：
1. 检查维度：{dimension_check}
2. 广播规则：{broadcast_rule}
3. 形状变换：{reshape_operation}
```

## 使用方式

### 在KernelGen中调用

Conductor作为KernelGen的内部组件自动调用：

```python
# KernelGen workflow中
conductor = ConductorSkill(config)

# Verifier失败后
analysis = conductor.analyze_result(
    verifier_result=verifier_output,
    generated_code=code,
    execution_log=log,
    iteration=current_iter
)

decision = conductor.decide_next_action(analysis)

if decision.action == "regenerate":
    # 将建议传递给下一轮KernelGen
    next_generation_input = {
        "previous_code": code,
        "verifier_error": analysis.error_message,
        "conductor_suggestion": decision.suggestion
    }
```

### 独立调用（调试用）

```bash
skill kernelgen/conductor \
  --verifier-result ./verifier_output.json \
  --generated-code ./code.py \
  --execution-log ./run.log \
  --iteration 2 \
  --max-iterations 5
```

## 输出格式

```json
{
  "analysis": {
    "success": false,
    "error_category": "LogicError",
    "error_location": "forward函数第23行",
    "error_message": "数值验证失败，最大差异0.05",
    "context": {
      "expected_output": [...],
      "actual_output": [...],
      "tolerance": 0.01
    }
  },
  "decision": {
    "action": "regenerate",
    "confidence": 0.85,
    "reason": "逻辑错误，有明确修复方向"
  },
  "suggestion": {
    "summary": "reduce操作维度处理错误",
    "details": [
      "检查tl.sum的axis参数",
      "确保输出形状与预期一致",
      "参考elementwise reduction模板"
    ],
    "code_hints": {
      "line_23": "tl.sum(input, axis=1) 应改为 tl.sum(input, axis=0)"
    }
  }
}
```

## 配置选项

```yaml
conductor:
  max_iterations: 5
  
  # 错误分类配置
  error_classification:
    enable_detailed_analysis: true
    context_lines: 3
  
  # 决策配置
  decision:
    abort_on_consecutive_failures: 3
    min_confidence_threshold: 0.6
  
  # 建议生成配置
  suggestion:
    max_suggestions: 5
    include_code_examples: true
    reference_similar_errors: true
```

## 与其他组件的关系

```
KernelGen (Code Generator)
    ↓ 生成代码
Verifier
    ↓ 验证失败
Conductor (本Skill)
    ↓ 分析 + 决策 + 建议
KernelGen (重新生成)
```

## 最佳实践

1. **详细日志**：确保Verifier提供完整的错误日志
2. **上下文保留**：保留历史错误信息用于趋势分析
3. **建议具体化**：建议应具体到代码行和修复方法
4. **避免循环**：检测重复错误模式，避免无限循环

## 示例场景

### 场景1：语法错误修复

```
第1轮生成：
- 代码：包含语法错误（缺少冒号）
- Verifier：编译失败
- Conductor：
  - 分类：SyntaxError
  - 决策：regenerate
  - 建议：第15行if语句后添加冒号

第2轮生成：
- 代码：语法错误已修复
- Verifier：编译通过，数值验证通过
- Conductor：
  - 决策：finish
```

### 场景2：逻辑错误放弃

```
第1-3轮生成：
- 代码：不同实现尝试
- Verifier：数值验证持续失败
- Conductor：
  - 分类：LogicError
  - 趋势：连续3次逻辑错误
  - 决策：abort
  - 理由：逻辑错误难以自动修复，建议人工介入
```
