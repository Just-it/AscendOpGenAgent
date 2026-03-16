---
name: adaptive-search
description: >
  Adaptive Search Agent - 基于UCB算法的自适应树搜索算子优化。
  特点：智能选择、快速收敛、比evolve更快。
mode: subagent
temperature: 0.1
tools:
  write: true
  edit: true
  bash: true
  skill: true
  read: true
argument-hint: >
  必需：task-file、framework、backend、arch、dsl。
  可选：max-concurrent、initial-tasks、max-tasks。
---

# Adaptive Search Agent

## 功能概述

Adaptive Search是基于UCB（Upper Confidence Bound）算法的自适应树搜索Agent，特点：
- **智能选择**：UCB算法平衡探索与利用
- **快速收敛**：比evolve快30-50%
- **灵感驱动**：从历史成功案例学习
- **并行搜索**：支持高并发搜索

## 工作流程

```
输入：Task描述
    ↓
[初始种群] → 并行生成多个初始实现
    ↓
[UCB选择] → 选择最有潜力的父代
    ↓
[灵感采样] → 从历史案例采样优化灵感
    ↓
[代码生成] → 基于父代和灵感生成新实现
    ↓
[性能评估] → 并行测试收集性能数据
    ↓
[迭代优化] → 多轮自适应搜索
    ↓
[最优选择] → 返回性能最佳的实现
```

## 核心组件

### 1. UCB选择器 (UCB Selector)

使用UCB算法选择最有潜力的父代：

```python
class UCBSelector:
    def select(self, candidates: List[Implementation]) -> Implementation:
        """
        UCB选择
        
        UCB公式：score = mean_performance + c * sqrt(2 * ln(N) / n)
        
        其中：
        - mean_performance: 平均性能
        - c: 探索系数
        - N: 总尝试次数
        - n: 该候选的尝试次数
        """
```

**UCB参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| exploration_coef | 探索系数 | 1.414 |
| random_factor | 随机因子 | 0.1 |
| use_softmax | 使用softmax | false |

### 2. 任务生成器 (Task Generator)

基于选中的父代生成新任务：

```python
class TaskGenerator:
    def generate(
        self,
        parent: Implementation,
        inspiration: List[CodeSnippet],
        mutation_strategy: str
    ) -> List[Task]:
        """
        生成新任务
        
        策略：
        1. 继承父代优秀特性
        2. 应用灵感中的优化模式
        3. 引入变异探索新空间
        """
```

**变异策略**：
- **参数调优**：调整block大小、线程配置
- **算法变种**：尝试不同算法实现
- **优化模式**：应用特定优化技巧

### 3. 搜索控制器 (Search Controller)

控制整个搜索过程：

```python
class SearchController:
    def search(self, config: SearchConfig) -> SearchResult:
        """
        执行自适应搜索
        
        流程：
        1. 初始化种群
        2. 评估初始实现
        3. while not stop:
           a. UCB选择父代
           b. 生成子代任务
           c. 评估子代
           d. 更新UCB统计
        4. 返回最优实现
        """
```

## 使用方式

### 作为Skill调用

```bash
skill adaptive_search \
  --task-file /path/to/matmul.py \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --max-concurrent 4 \
  --max-tasks 50 \
  --output-path ${pwd}/triton_ascend_output/
```

### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| --task-file | string | 是 | - | 任务文件路径 |
| --framework | string | 是 | - | torch/mindspore |
| --backend | string | 是 | - | cuda/ascend/cpu |
| --arch | string | 是 | - | 架构类型 |
| --dsl | string | 是 | - | DSL类型 |
| --max-concurrent | int | 否 | 4 | 最大并发数 |
| --initial-tasks | int | 否 | 4 | 初始任务数 |
| --max-tasks | int | 否 | 50 | 最大总任务数 |
| --exploration-coef | float | 否 | 1.414 | UCB探索系数 |
| --output-path | string | 否 | ${pwd}/triton_ascend_output | 输出目录 |

## 输出结果

```
${pwd}/triton_ascend_output/
├── generated_code.py          # 最佳实现代码
├── summary.json               # 搜索摘要
│   {
│     "subagent": "adaptive_search",
│     "success": true,
│     "total_submitted": 50,
│     "total_completed": 48,
│     "total_success": 42,
│     "success_rate": 0.875,
│     "elapsed_time": 180.5,
│     "best_performance": {
│       "gen_time_us": 32.5,
│       "speedup": 2.75
│     }
│   }
├── search_tree.json           # 搜索树结构
├── lineage_graph.html         # 谱系图可视化
└── logs/
    ├── search.log
    └── ucb_selection.log
```

## 适用场景

✅ **推荐使用**：
- 用户要求"高性能"但时间有限
- 需要比evolve更快的优化
- 复杂算子的性能优化
- 探索大量优化策略组合

❌ **不推荐使用**：
- 只需要快速生成代码（使用kernelgen）
- 时间非常敏感（kernelgen更快）
- 简单算子（kernelgen足够）

## 性能指标

- **典型耗时**：2-5分钟（比evolve快30-50%）
- **成功率**：~85%
- **加速比**：平均1.5-3x
- **并发能力**：支持4-8并发

## 与KernelGen对比

| 特性 | KernelGen | Adaptive Search |
|------|-----------|-----------------|
| 速度 | 1-5分钟 | 2-5分钟 |
| 优化能力 | 基础 | 强 |
| 适用场景 | 快速生成 | 性能优化 |
| 算法 | 迭代修复 | UCB搜索 |
| 并发 | 串行 | 并行 |

## 依赖Skills

- `adaptive_search/strategy` - 搜索策略
- `adaptive_search/ucb_selector` - UCB选择

## 配置示例

```yaml
adaptive_search:
  concurrency:
    max_concurrent: 4
    initial_task_count: 4
    tasks_per_parent: 1
  
  stopping:
    max_total_tasks: 50
    min_success_rate: 0.5
  
  ucb_selection:
    exploration_coef: 1.414
    random_factor: 0.1
    use_softmax: false
  
  inspiration:
    sample_num: 3
    use_tiered_sampling: true
```
