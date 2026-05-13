# precision-tuning 目录结构

```
precision-tuning/
├── README.md                          # Skill 说明文档（设计概览、双 Subagent 架构、知识库结构）
├── SKILL.md                           # Agent 执行手册（主流程、Sub-steps、Gate 协议）
├── STRUCTURE.md                       # 本文件：目录结构示意图
├── MODIFICATIONS.md                   # precision_gate.py 实现说明（当前状态）
├── test_baseline_coverage.py          # 覆盖性测试：3 场景（首轮成功/两轮成功/最终失败）
│
├── references/                        # 静态参考资料
│   ├── precision_knowledge_base.json  # 精度问题知识库（问题模式 + 算子 CHECKLIST）
│   │
│   └── decomposition_examples/        # 算子计算分解示例（Sub-step 2.2 参考）
│       ├── README.md                  # 示例格式说明与模式分类索引
│       ├── average_pooling2d.md       # 滑窗累加模式
│       ├── cumsum.md                  # 前缀累加模式
│       ├── layer_norm.md              # Normalization 模式
│       ├── matmul.md                  # 分块累加模式
│       ├── mse_loss.md                # Loss 计算模式
│       ├── reduce_sum.md              # Reduction 模式
│       └── softmax.md                 # Softmax（Reduction + Normalization）
│
└── scripts/                           # 精度调优工具脚本（Subagent 共用）
    ├── precision_forensics.py         # 取证脚本：运行算子、采集误差数据
    ├── precision_gate.py              # Gate 脚本：判断精度是否达标、决定 loop_signal
    ├── precision_knowledge.py         # 知识库管理：load / search / dump
    └── __pycache__/                   # Python 编译缓存（自动生成，无需手动维护）
```

## Subagent 架构

本 Skill 被两个 Subagent 共用，分别采用不同的审计策略：

| Subagent | 文件位置 | 审计策略 | 特点 |
|----------|----------|----------|------|
| **发现式** | `.opencode/agents/precision-tuning-discovery.md` | 发现式审计 | 直接运用 AscendC 领域知识推理根因，不强制预读参考示例，依赖 Agent 自身知识储备快速诊断 |
| **构建式** | `.opencode/agents/precision-tuning.md` | 构建式审计 | 严格遵循 Phase A→B→C：先建规范（强制读取 lowering 示例），再看代码，最后结构化对照 |

### 共用组件

两个 Subagent 共享以下基础设施：

| 组件 | 说明 |
|------|------|
| `precision_forensics.py` | L0-L8 数值取证，输出结构化报告 |
| `precision_gate.py` | 链式 Gate 验证（Gate-F/A/X/V）+ 循环控制 |
| `precision_knowledge.py` | 知识库 RAG 检索与管理 |
| `precision_knowledge_base.json` | 精度问题模式库 + 算子 CHECKLIST |
| `decomposition_examples/` | 算子计算分解参考 |

### 策略差异

| 维度 | 发现式 | 构建式 |
|------|--------|--------|
| **Phase A** | 可选查阅参考资料，依 Agent 经验判断 | 强制读取 lowering 示例，产出 `[REFERENCE_IMPL_SPEC]` |
| **分析起点** | 直接从数值取证数据出发 | 先建立规范基准，再对照实际代码 |
| **适用场景** | Agent 对 AscendC API 规范已有充分了解 | 需要严格参照规范进行结构化审计 |
| **Gate-A 要求** | 仍需 `[REFERENCE_IMPL_SPEC]` section | 强制验证 `[REFERENCE_IMPL_SPEC]` 完整性 |

## 文件职责速查

| 文件 | 阶段 | 职责 | 使用方 |
|------|------|------|--------|
| `SKILL.md` | 全流程 | Agent 执行手册，定义 Sub-steps 1~3 及 Gate 协议 | 双 Subagent |
| `README.md` | 参考 | 设计文档，含双 Subagent 架构说明、知识库结构 | 开发者 |
| `precision_forensics.py` | Sub-step 1 | 运行算子取证，输出误差统计与 worst element 数据 | 双 Subagent |
| `precision_gate.py` | Sub-step 3 末尾 | 评估精度结果，输出 `loop_signal=PASS/CONTINUE/FAIL` | 双 Subagent |
| `precision_knowledge.py` | Sub-step 2.1 / 2.4 / 3 | 知识库 RAG 检索、加载、写入 | 双 Subagent |
| `precision_knowledge_base.json` | Sub-step 2.4 | 已知精度问题模式 + 算子 CHECKLIST | 双 Subagent |
| `decomposition_examples/*.md` | Sub-step 2.2 | 按算子类型提供计算分解示例 | 双 Subagent（构建式强制、发现式可选） |
