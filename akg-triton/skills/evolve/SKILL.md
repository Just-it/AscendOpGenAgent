---
name: evolve
description: >
  Evolve Agent - 基于岛屿模型的进化算法算子优化。
  特点：多轮迭代、多样性强、追求极致性能。
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
  可选：max-rounds、parallel-num、num-islands。
---

# Evolve Agent

## 功能概述

Evolve是基于岛屿模型（Island Model）的进化算法Agent，特点：
- **多轮迭代**：5-20轮演化持续优化
- **岛屿模型**：多岛屿并行演化，定期迁移
- **多样性强**：探索多种优化策略
- **极致性能**：追求最佳性能表现

## 工作流程

```
输入：Task描述
    ↓
[初始化] → 创建多个岛屿，每个岛屿生成初始种群
    ↓
[进化循环] (多轮)
    ├─ [选择] → 选择优秀个体作为父代
    ├─ [变异] → 基于父代生成子代
    ├─ [评估] → 验证并评估性能
    ├─ [选择] → 优胜劣汰
    └─ [迁移] → 岛屿间交换优秀个体（定期）
    ↓
[最优选择] → 从所有岛屿选择最佳实现
```

## 核心组件

### 1. 岛屿管理器 (Island Manager)

管理多个进化岛屿：

```python
class IslandManager:
    def __init__(self, num_islands: int):
        self.islands = [Island() for _ in range(num_islands)]
    
    def evolve_round(self):
        """执行一轮进化"""
        for island in self.islands:
            island.evolve()
    
    def migrate(self):
        """岛屿间迁移优秀个体"""
        # 选择优秀个体迁移到其他岛屿
```

**岛屿参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| num_islands | 岛屿数量 | 2 |
| migration_interval | 迁移间隔（轮数） | 2 |
| elite_size | 精英个体数 | 2 |

### 2. 进化算子 (Evolution Operators)

实现进化算法的核心操作：

```python
class EvolutionOperators:
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        """选择父代（锦标赛选择）"""
        pass
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """交叉操作"""
        pass
    
    def mutate(self, individual: Individual) -> Individual:
        """变异操作"""
        pass
```

**变异策略**：
- **参数变异**：调整block大小、线程数
- **结构变异**：修改算法结构
- **融合变异**：融合多个优秀特性

### 3. 适应度评估器 (Fitness Evaluator)

评估个体（代码实现）的性能：

```python
class FitnessEvaluator:
    def evaluate(self, code: str) -> FitnessScore:
        """
        评估适应度
        
        指标：
        - 正确性：是否通过验证
        - 性能：执行速度
        - 复杂度：代码复杂度
        
        Returns:
            FitnessScore: 综合适应度分数
        """
```

## 使用方式

### 作为Skill调用

```bash
skill evolve \
  --task-file /path/to/matmul.py \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --max-rounds 5 \
  --parallel-num 4 \
  --num-islands 2 \
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
| --max-rounds | int | 否 | 5 | 最大进化轮数 |
| --parallel-num | int | 否 | 4 | 每轮并行数 |
| --num-islands | int | 否 | 2 | 岛屿数量 |
| --migration-interval | int | 否 | 2 | 迁移间隔（轮数） |
| --output-path | string | 否 | ${pwd}/triton_ascend_output | 输出目录 |

## 输出结果

```
${pwd}/triton_ascend_output/
├── generated_code.py          # 最佳实现代码
├── summary.json               # 进化摘要
│   {
│     "subagent": "evolve",
│     "success": true,
│     "total_rounds": 5,
│     "total_tasks": 80,
│     "successful_tasks": 68,
│     "success_rate": 0.85,
│     "best_performance": {
│       "gen_time_us": 28.3,
│       "speedup": 3.16
│     },
│     "evolution_history": [...]
│   }
├── evolution_history.json     # 详细进化历史
├── island_snapshots/          # 每轮岛屿状态
│   ├── round_0/
│   ├── round_1/
│   └── ...
└── logs/
    └── evolution.log
```

## 适用场景

✅ **推荐使用**：
- 用户要求"极致性能"
- 时间充足（15-60分钟）
- 关键算子需要深度优化
- 需要探索多样化实现

❌ **不推荐使用**：
- 时间敏感（使用kernelgen或adaptive_search）
- 只需要快速原型（使用kernelgen）
- 资源受限（evolve资源消耗高）

## 性能指标

- **典型耗时**：15-60分钟
- **成功率**：~80%
- **加速比**：平均2-4x（比baseline）
- **并发能力**：支持多设备并行

## 与Adaptive Search对比

| 特性 | Adaptive Search | Evolve |
|------|-----------------|--------|
| 算法 | UCB树搜索 | 岛屿模型进化 |
| 速度 | 2-5分钟 | 15-60分钟 |
| 探索能力 | 强 | 极强 |
| 多样性 | 中 | 高 |
| 适用场景 | 高性能 | 极致性能 |

## 依赖Skills

- `evolve/evolution` - 进化算法
- `evolve/selection` - 选择策略

## 配置示例

```yaml
evolve:
  evolution:
    max_rounds: 5
    parallel_num: 4
    handwrite_decay_rate: 2.0
  
  island:
    num_islands: 2
    migration_interval: 2
    elite_size: 2
    parent_selection_prob: 0.5
  
  population:
    initial_size: 8
    max_size: 16
  
  mutation:
    rate: 0.3
    strategies:
      - parameter_tuning
      - algorithm_variant
      - optimization_pattern
```

## 进化过程可视化

```
Round 0:  [Island 0: 8个体]  [Island 1: 8个体]
              ↓                    ↓
Round 1:  [Island 0: 8个体]  [Island 1: 8个体]
              ↓ ←——迁移——→ ↓
Round 2:  [Island 0: 8个体]  [Island 1: 8个体]
              ↓                    ↓
           ...                  ...
              ↓                    ↓
Round 5:  [Best: 28.3us]   [Best: 31.2us]
              
Global Best: 28.3us (Speedup: 3.16x)
```
