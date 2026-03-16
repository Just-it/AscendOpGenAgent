---
name: akg-env-setup
description: >
  AKG环境设置Skill - 检查环境、安装依赖、配置参数。
argument-hint: >
  模式：FULL_SETUP（完整设置）或基础模式（仅环境检查）。
---

# AKG环境设置 Skill

## 功能概述

AKG环境设置Skill负责：
1. **环境检查**：检查Python环境、依赖安装情况
2. **硬件检测**：检测可用硬件（GPU/NPU/CPU）
3. **参数确认**：确认framework/backend/arch/dsl参数
4. **依赖安装**：按需安装运行时依赖

## 两种模式

### 模式1：基础模式（Basic Mode）

仅执行环境检查，不确认参数：

```bash
skill akg-env-setup
```

**执行内容**：
- 检查Python环境（conda/venv）
- 检测可用硬件
- 写入环境缓存

### 模式2：完整模式（FULL_SETUP Mode）

执行完整设置，包含参数确认：

```bash
skill akg-env-setup --mode FULL_SETUP
```

**执行内容**：
- 基础模式的所有检查
- 推断并确认framework/backend/arch/dsl
- 安装运行时依赖
- 返回命令模板

## 工作流程

```
[检查缓存] ~/.akg/check_env.md
    ↓ 存在
[读取缓存] 使用缓存的环境配置
    ↓ 不存在
[环境检查]
    ├─ Python环境（conda/venv）
    └─ 硬件检测
    ↓
[参数确认]（FULL_SETUP模式）
    ├─ framework: torch/mindspore
    ├─ backend: cuda/ascend/cpu
    ├─ arch: a100/ascend910b4等
    └─ dsl: triton_cuda/triton_ascend等
    ↓
[依赖安装]（FULL_SETUP模式）
    ↓
[写入缓存] ~/.akg/check_env.md
```

## 环境缓存

缓存文件位置：`~/.akg/check_env.md`

**缓存内容**：
```yaml
---
env_type: conda
conda_env: akg_agents
akg_agents_dir: /home/user/akg/akg_agents
hardware:
  cuda:
    available: true
    devices: [0, 1]
    arch: a100
  ascend:
    available: false
frameworks:
  torch: installed
  mindspore: not_installed
detected_config:
  framework: torch
  backend: cuda
  arch: a100
  dsl: triton_cuda
command_template: |
  conda run -n akg_agents --no-capture-output bash -c \
    "cd /home/user/akg/akg_agents && source env.sh && <CMD>"
---
```

## 参数有效值

### Framework
- `torch` - PyTorch
- `mindspore` - MindSpore

### Backend
- `cuda` - NVIDIA GPU
- `ascend` - Huawei Ascend NPU
- `cpu` - CPU

### Arch
- **CUDA**: `a100`, `v100`, `h20`, `l20`, `rtx3090`
- **Ascend**: `ascend910b1`, `ascend910b2`, `ascend910b3`, `ascend910b4`, `ascend310p3`
- **CPU**: `x86_64`, `aarch64`

### DSL
- `triton_cuda` - Triton for CUDA
- `triton_ascend` - Triton for Ascend
- `cuda_c` - CUDA C
- `cpp` - C++
- `tilelang_cuda` - TileLang for CUDA
- `ascendc` - AscendC
- `pypto` - PyPTO

## 使用方式

### 在op-optimizer中调用

Phase 0自动调用：

```
Phase 0: 环境准备
    ↓
加载 akg-env-setup skill（FULL_SETUP模式）
    ↓
获取命令模板和确认的配置参数
```

### 独立调用

```bash
# 基础模式
skill akg-env-setup

# 完整模式
skill akg-env-setup --mode FULL_SETUP

# 强制重新检查（删除缓存）
rm ~/.akg/check_env.md && skill akg-env-setup
```

## 输出结果

### 成功结果

```json
{
  "success": true,
  "env_type": "conda",
  "conda_env": "akg_agents",
  "akg_agents_dir": "/home/user/akg/akg_agents",
  "hardware": {
    "cuda": {
      "available": true,
      "devices": [0, 1],
      "arch": "a100"
    }
  },
  "config": {
    "framework": "torch",
    "backend": "cuda",
    "arch": "a100",
    "dsl": "triton_cuda"
  },
  "command_template": "conda run -n akg_agents --no-capture-output bash -c \"cd /home/user/akg/akg_agents && source env.sh && <CMD>\""
}
```

### 失败结果

```json
{
  "success": false,
  "error": "未检测到可用的后端",
  "suggestion": "请安装CUDA驱动或配置Ascend环境"
}
```

## 命令模板使用

获取的命令模板用于包裹其他命令：

```bash
# 模板
conda run -n akg_agents --no-capture-output bash -c \
  "cd /home/user/akg/akg_agents && source env.sh && <CMD>"

# 实际使用（替换<CMD>）
conda run -n akg_agents --no-capture-output bash -c \
  "cd /home/user/akg/akg_agents && source env.sh && python script.py"
```

## 环境变量

Skill会自动设置以下环境变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `$HOME_DIR` | 用户home目录 | `/home/user` |
| `$AKG_AGENTS_DIR` | akg_agents目录 | `/home/user/akg/akg_agents` |
| `$ENV_TYPE` | 环境类型 | `conda` |
| `$CONDA_ENV` | conda环境名 | `akg_agents` |

## 最佳实践

1. **首次使用**：必须运行FULL_SETUP模式
2. **日常调用**：使用基础模式，读取缓存
3. **环境变更**：删除缓存文件强制重新检查
4. **参数确认**：每次任务都确认参数，不依赖缓存
