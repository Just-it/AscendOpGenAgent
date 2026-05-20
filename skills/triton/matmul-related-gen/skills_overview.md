# Triton Matmul 相关 Skill 体系说明

本目录包含 4 个 Skill，它们构成了一套完整的 Triton 矩阵乘法（GEMM）算子自动生成体系。

特别注意，需要环境中事先安装好 `msprof op`（强调不是 `msprof`）。

## 架构概览

```
matmul_skill_for_any_shape  (终端用户入口 Skill)
    │
    │  离线阶段：通过以下 3 个原子 Skill 生成 code_lib.zip
    │
    ├── triton_w_nd2nz      (原子 Skill ①: B 矩阵伪NZ格式优化)
    ├── tiling_diagonal      (原子 Skill ②: 对角 Tiling 核心映射)
    └── k_axis_offset        (原子 Skill ③: K 轴偏移优化)
```

## 1. matmul_skill_for_any_shape — 终端用户入口 Skill

### 定位

这是一个 **可直接生成 matmul 算子的完整 Skill**，面向最终用户。用户无需手动组合各种优化策略，只需传入 shape 参数即可自动从预生成的代码库中筛选最优 Kernel 并完成测试验证。

### 使用前准备：服务器配置

该 Skill 的运行需要远程 NPU 服务器来编译和性能测试 Kernel，因此 **必须先配置服务器连接信息**。

编辑 `script/server_config.json`：

```json
{
  "ip": "你的服务器IP",
  "username": "root",
  "auth_method": "key",
  "password": "",
  "ssh_key_path": "你的密钥文件路径",
  "docker_container": "容器名称",
  "npu_device": "NPU设备ID"
}
```

| 字段 | 说明 |
|------|------|
| `ip` | NPU 服务器 IP 地址 |
| `auth_method` | `"password"` 密码认证 或 `"key"` 密钥认证 |
| `password` | 密码认证时填写 |
| `ssh_key_path` | 密钥认证时填写 .pem 文件路径 |
| `docker_container` | Docker 容器名称 |
| `npu_device` | NPU 设备 ID（0-7） |

配置完成后运行检测：

```bash
python script/check_config.py
```

### 使用方式

**单 shape 测试**（例如 M=128 N=7168 K=4096）：

```
请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 FP16 高性能算子
或 请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 FP16 ND格式算子
或 请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 INT8 高性能算子
或 请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 INT8 ND格式算子
```

**多 shape 批量测试**（使用 `shape_list.md` 中预定义的 shape）：

```
请按照 SKILL.md 生成 FP16 高性能算子，且请使用 shape_list 中的 shape 进行测试
或 请按照 SKILL.md 生成 FP16 ND格式算子，且请使用 shape_list 中的 shape 进行测试
或 请按照 SKILL.md 生成 INT8 高性能算子，且请使用 shape_list 中的 shape 进行测试
或 请按照 SKILL.md 生成 INT8 ND格式算子，且请使用 shape_list 中的 shape 进行测试
```

### 支持的任务类型

| 用户需求 | `--code-subdir` | 精度 | 输入格式 | Kernel 数量 |
|----------|-----------------|------|----------|-------------|
| FP16 高性能 matmul | `code-fp16` | FP16 | 伪NZ | 36 |
| FP16 ND 格式 matmul | `code-fp16-fuse` | FP16 | ND + 融合 | 72 |
| INT8 高性能 matmul | `code-int8` | INT8 | 伪NZ | 36 |
| INT8 ND 格式 matmul | `code-int8-fuse` | INT8 | ND + 融合 | 72 |

### 完整工作流

```
Step 0: CONFIG_CHECK — 自动检测服务器配置是否有效
    ↓
Step 1: CLEAN — 清理服务器端旧测试数据
    ↓
Step 2: UPLOAD + START — 上传 code_lib.zip + 启动远程批量测试
    ↓
Step 3: VERIFY — 确认远程测试进程已启动
    ↓
用户自行等待并查看进度 / 下载结果
```

### 输出

测试完成后，通过 `python script/analyze_results.py` 下载结果到本地 `output/` 目录：
- `best_kernels_report.xlsx` — 每个 shape 的最优 Kernel 性能报告
- `*.py` — 所有最优 Kernel 的源代码

---

## 2. 三个原子 Skill

以下三个 Skill 是构成 matmul_skill_for_any_shape 的底层基础。它们各自独立可用，专注于单一的优化维度。**matmul_skill_for_any_shape 中的 `code_lib.zip` 正是通过组合这三个原子 Skill 离线生成的。**

### 2.1 triton_w_nd2nz — B 矩阵伪NZ格式优化

**文件**: [triton_w_nd2nz/SKILL.md](file:///d:/AscendICT/AscendOpGenAgent/skills/triton/matmul-related-gen/triton_w_nd2nz/SKILL.md)

---

### 2.2 tiling_diagonal — 对角 Tiling 核心映射

**文件**: [tiling_diagonal/skill.md](file:///d:/AscendICT/AscendOpGenAgent/skills/triton/matmul-related-gen/tiling_diagonal/skill.md)

---

### 2.3 k_axis_offset — K 轴偏移优化

**文件**: [k_axis_offset/skill.md](file:///d:/AscendICT/AscendOpGenAgent/skills/triton/matmul-related-gen/k_axis_offset/skill.md)

---

## 3. 整体关系总结

```
┌────────────────────────────────────────────────────────────────┐
│                matmul_skill_for_any_shape                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    code_lib.zip                           │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │  │
│  │  │ code-fp16/   │  │ code-fp16-   │  │ code-int8/     │   │  │
│  │  │ (36 kernels) │  │ fuse/ (72)   │  │ code-int8-fuse/│   │  │
│  │  └─────────────┘  └──────────────┘  └───────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│      离线阶段：通过组合 3 个原子 Skill 生成 code_lib.zip          │
│                              │                                  │
│    ┌─────────────────────────┼─────────────────────────┐        │
│    │                         │                         │        │
│    ▼                         ▼                         ▼        │
│  triton_w_nd2nz         tiling_diagonal          k_axis_offset  │
│  (B矩阵格式优化)         (对角Tiling映射)          (K轴偏移)      │
│                                                         │        │
│    ┌────────────────────────────────────────────────────┘        │
│    │  运行时：根据用户 shape 从 code_lib 中筛选最优 Kernel       │
│    │  上传 → 远程编译 → 性能测试 → 输出最优 Kernel 和报告        │
│    └─────────────────────────────────────────────────────────    │
└────────────────────────────────────────────────────────────────┘
```

### 三个原子 Skill 的职责分工

| Skill | 优化维度 | 影响范围 |
|-------|---------|---------|
| **triton_w_nd2nz** | 内存布局 | B 矩阵数据排布，Host 端预处理 + Kernel 端加载模式 |
| **tiling_diagonal** | Core 调度 | 计算块的分配策略，多核负载均衡 |
| **k_axis_offset** | 内存访问 | K 轴遍历顺序，减少多核内存访问冲突 |

### 运行时流程

1. 用户指定 shape 和精度要求
2. 系统从 `code_lib.zip` 中提取对应子目录的所有 Kernel
3. 上传至 NPU 服务器
4. 远程逐一编译并执行性能测试（使用 `msprof`）
5. 收集性能数据，输出每个 shape 的最优 Kernel 和对应的源代码

生成的代码和测试报告存储在本地 `output/` 文件夹下。
