---
name: kernelgen-verifier-wrapper
description: >
  KernelGen Verifier Wrapper Skill - 验证生成的内核代码。
  负责编译验证、数值正确性验证、性能分析。
argument-hint: >
  输入：生成的代码、任务描述、测试配置。
  输出：验证结果、错误信息、性能数据。
---

# KernelGen Verifier Wrapper Skill

## 功能概述

Verifier Wrapper是KernelGen的代码验证组件，负责：
1. **编译验证**：验证代码能否正确编译
2. **数值验证**：验证计算结果正确性
3. **性能分析**：测量执行性能（可选）
4. **错误收集**：收集详细的错误信息

## 工作流程

```
输入：生成的代码 + 任务描述
    ↓
[代码准备] → 创建测试环境
    ↓
[编译验证] → 编译代码
    ↓（成功）
[运行验证] → 执行代码
    ↓（成功）
[数值对比] → 对比输出结果
    ↓（通过）
[性能分析] → 测量性能（可选）
    ↓
输出：验证结果 + 性能数据
```

## 验证阶段

### 阶段1：编译验证

检查代码是否能正确编译：

```python
class CompileVerifier:
    def verify(self, code: str, dsl: str) -> CompileResult:
        """
        编译验证
        
        Returns:
            CompileResult:
            - success: 是否编译成功
            - error_message: 错误信息
            - compile_time: 编译耗时
        """
```

**不同DSL的编译方式**：
| DSL | 编译方式 |
|-----|---------|
| triton_cuda | Triton JIT编译 |
| triton_ascend | Triton Ascend编译 |
| cuda_c | nvcc编译 |
| cpp | g++编译 |

### 阶段2：运行验证

验证代码能否正确运行：

```python
class RunVerifier:
    def verify(self, compiled_code, test_inputs) -> RunResult:
        """
        运行验证
        
        Returns:
            RunResult:
            - success: 是否运行成功
            - error_message: 错误信息
            - output: 实际输出
            - execution_time: 执行耗时
        """
```

**测试输入生成**：
- 从任务描述中提取输入形状和dtype
- 生成随机测试数据
- 支持多种数据类型（float32、float16等）

### 阶段3：数值验证

对比生成代码与参考实现的输出：

```python
class NumericalVerifier:
    def verify(
        self,
        expected_output: Tensor,
        actual_output: Tensor,
        tolerance: float = 1e-5
    ) -> NumericalResult:
        """
        数值验证
        
        Returns:
            NumericalResult:
            - success: 是否通过验证
            - max_diff: 最大差异
            - mean_diff: 平均差异
            - failed_indices: 失败位置
        """
```

**验证指标**：
- **最大绝对误差** (Max Absolute Error)
- **平均绝对误差** (Mean Absolute Error)
- **相对误差** (Relative Error)
- **通过阈值** (Tolerance)

### 阶段4：性能分析（可选）

测量代码执行性能：

```python
class PerformanceProfiler:
    def profile(
        self,
        code,
        test_inputs,
        warmup_times: int = 5,
        run_times: int = 50
    ) -> ProfileResult:
        """
        性能分析
        
        Returns:
            ProfileResult:
            - gen_time: 生成代码耗时(us)
            - base_time: 基线代码耗时(us)
            - speedup: 加速比
        """
```

## 错误处理

### 编译错误

```json
{
  "stage": "compile",
  "success": false,
  "error_type": "SyntaxError",
  "error_message": "invalid syntax",
  "error_location": "line 15",
  "error_detail": "if x > 0\n         ^\nSyntaxError: expected ':'"
}
```

### 运行时错误

```json
{
  "stage": "run",
  "success": false,
  "error_type": "RuntimeError",
  "error_message": "CUDA out of memory",
  "error_location": "kernel execution",
  "error_detail": "Tried to allocate 2.00 GiB..."
}
```

### 数值错误

```json
{
  "stage": "numerical",
  "success": false,
  "error_type": "AccuracyError",
  "max_diff": 0.05,
  "tolerance": 0.01,
  "failed_indices": [[0, 1], [2, 3]],
  "expected_sample": [1.0, 2.0, 3.0],
  "actual_sample": [1.05, 2.0, 2.95]
}
```

## 使用方式

### 在KernelGen中调用

Verifier作为KernelGen的内部组件自动调用。

### 独立调用

```bash
skill kernelgen/verifier_wrapper \
  --code-file ./generated_code.py \
  --task-file ./task.py \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --enable-profile true
```

## 输出格式

### 成功结果

```json
{
  "success": true,
  "stage": "complete",
  "compile": {
    "success": true,
    "time_ms": 1250
  },
  "run": {
    "success": true,
    "time_ms": 50
  },
  "numerical": {
    "success": true,
    "max_diff": 1e-06,
    "tolerance": 1e-05
  },
  "profile": {
    "gen_time_us": 45.2,
    "base_time_us": 89.5,
    "speedup": 1.98
  }
}
```

### 失败结果

```json
{
  "success": false,
  "stage": "numerical",
  "compile": {
    "success": true,
    "time_ms": 1200
  },
  "run": {
    "success": true,
    "time_ms": 45
  },
  "numerical": {
    "success": false,
    "max_diff": 0.05,
    "tolerance": 0.01,
    "failed_count": 10
  },
  "error": {
    "type": "AccuracyError",
    "message": "数值验证失败，最大差异0.05超过阈值0.01",
    "details": {...}
  }
}
```

## 配置选项

```yaml
verifier:
  # 编译配置
  compile:
    timeout_seconds: 60
    optimization_level: "O2"
  
  # 运行配置
  run:
    timeout_seconds: 30
    max_memory_gb: 4
  
  # 数值验证配置
  numerical:
    tolerance: 1e-5
    relative_tolerance: 1e-3
    check_shape: true
  
  # 性能分析配置
  profile:
    enable: false
    warmup_times: 5
    run_times: 50
    measure_memory: true
```

## 与其他组件的关系

```
Code Generator
    ↓ 生成代码
Verifier Wrapper (本Skill)
    ↓ 验证
    ├─ 成功 → 任务完成
    └─ 失败 → 
        ↓
    Conductor
        ↓ 分析 + 建议
    Code Generator (重新生成)
```

## 最佳实践

1. **渐进验证**：先编译，再运行，最后数值对比
2. **详细日志**：记录每个阶段的详细输出
3. **超时控制**：防止死循环或长时间运行
4. **资源限制**：控制内存和CPU使用
