---
name: "matmul-code-generator"
description: "Generate high-performance Triton matmul kernels (FP16/INT8, standard ND or ND2NZ input format). Invoke when user wants to create or modify matmul kernel code with various optimization strategies (tiling, koffset, diagonal, a_fuse/w_fuse/aw_fuse)."
---

# Matmul 代码生成器

特别注意，需要环境中事先安装好 `msprof op`（强调不是 `msprof`）。

生成高性能 Triton 矩阵乘法 Kernel，支持 FP16/INT8 两种精度 × 标准 ND / ND2NZ 两种输入格式，含 tiling、koffset、diagonal、a_fuse/w_fuse/aw_fuse 等优化策略。

## 🔄 Workflow

严格按顺序执行，每步成功后再进入下一步，失败则重试。

```
Step 0: CONFIG_CHECK (自动检测)
  🔍 **自动检测**: 检查 script/server_config.json 是否存在且有效
  
  检测命令:
  ```bash
  python script/check_config.py
  ```
  
  - ✅ 配置已存在且有效 → 跳过配置，直接进入 Step 1
  - ❌ 配置不存在或无效 → 按下方模板配置后重新检测
  
  测试 SSH 连接 (可选):
  ```bash
  python script/check_config.py --test-connection
  ```
  
  配置模板 (仅首次或配置变更时需要):
  
  方式1: 密码认证 (推荐新手使用)
  ```json
  {
    "ip": "你的服务器IP",
    "username": "root",
    "auth_method": "password",
    "password": "你的密码",
    "docker_container": "容器名称",
    "npu_device": "NPU设备ID"
  }
  ```
  
  方式2: 密钥认证 (推荐生产环境)
  ```json
  {
    "ip": "你的服务器IP",
    "username": "root",
    "auth_method": "key",
    "password": "",
    "ssh_key_path": "密钥文件路径",
    "docker_container": "容器名称",
    "npu_device": "NPU设备ID"
  }
  ```

Step 1: CLEAN
  python script/server_check.py clean
  验证: 服务器 /root/MyAICode/batch_verification/ 已删除

Step 2: UPLOAD + START
  # 启动上传（verify_code.py 在启动时初始化 script/.upload_status.json 为 {"status": "false"}，
  # 全部上传+启动完成后自动设为 {"status": "true"}）
  python script/verify_code.py --code-subdir <SUB> --server-config script/server_config.json --shape "<M N K>" --npu-device <ID>
  或
  python script/verify_code.py --code-subdir <SUB> --server-config script/server_config.json --shape-list shape_list.md --npu-device <ID>

  ⚠️ **Repeatedly execute** the following command until the `status` field in `script/.upload_status.json` becomes `"true"`:

  ```
  python script/check_file_exist.py --code-subdir <SUB>
  ```

  Once `status` is confirmed as `"true"`, proceed with subsequent startup steps.

  **⚠️ CRITICAL: Do NOT perform any other operations during this polling/monitoring process. Focus exclusively on running the command above in a loop until verification succeeds. Repeat this polling cycle no more than 100 times; stop if the limit is reached.**

Step 3: VERIFY
  python script/server_check.py status
  验证: 输出包含 "python.*batch_verification" 进程

--- 直接退出，等待用户自行操作 ---
--- 用户自行操作 ---
  查看进度: python script/server_check.py progress
  下载结果: python script/analyze_results.py  →  output/
```

## 📁 项目结构

```
├── SKILL.md                  # 本文件
├── code_lib.zip              # 所有 kernel 代码 (上传到服务器)
├── shape_list.md             # Shape 列表模板
└── script/
    ├── verify_code.py        # 主入口: clean + 上传 + 启动测试
    ├── batch_verification.py # 服务器端批量测试 (自动运行)
    ├── server_check.py       # 查看状态/进度/结果/clean/kill
    ├── analyze_results.py    # 下载最优代码 + 生成 Excel
    ├── check_config.py       # 配置检查
    ├── check_file_exist.py   # 上传状态轮询
    ├── reorder_excel.py      # Excel 排序
    ├── server_config.json    # 服务器连接配置
    └── .upload_status.json   # 上传状态标记
```

> **注意**: `code_lib.zip` 解压后包含 `code-fp16/`、`code-fp16-fuse/`、`code-int8/`、`code-int8-fuse/` 四个子目录，对应四种 kernel 变体。运行时需通过 `--code-subdir` 指定要测试的目录。

## 🔧 配置

`script/server_config.json`:
```json
{
  "ip": "你的服务器IP",
  "username": "root",
  "auth_method": "key",
  "password": "",
  "ssh_key_path": "你的密钥文件路径",
  "docker_container": "你的容器名",
  "npu_device": "5"
}
```

**认证方式说明:**
- `auth_method`: `"password"` 使用密码认证，`"key"` 使用密钥认证
- `username`: SSH 用户名，默认为 `"root"`
- `password`: 当 `auth_method="password"` 时填写密码
- `ssh_key_path`: 当 `auth_method="key"` 时填写密钥文件路径

测试 SSH:
- 密码方式: `ssh root@你的服务器IP`
- 密钥方式: `ssh -i 你的密钥文件路径 root@你的服务器IP`

## 🎯 四个任务 → 参数映射

| 用户需求 | `--code-subdir` | 精度 | 格式 | 文件 |
|----------|-----------------|------|------|------|
| 生成 FP16 高性能 matmul | `code-fp16` | FP16 | 伪NZ | 36 |
| 生成 FP16 ND 格式 matmul | `code-fp16-fuse` | FP16 | ND + 融合 | 72 |
| 生成 INT8 高性能 matmul | `code-int8` | INT8 | 伪NZ | 36 |
| 生成 INT8 ND 格式 matmul | `code-int8-fuse` | INT8 | ND + 融合 | 72 |

## 📋 脚本 I/O 说明

### verify_code.py — 主入口

| 参数 | 必需 | 取值 |
|------|------|------|
| `--code-subdir` | ✅ | `code-fp16` / `code-fp16-fuse` / `code-int8` / `code-int8-fuse` |
| `--server-config` | ✅ | `script/server_config.json` |
| `--shape` | 二选一 | `"M N K"` 如 `"128 4096 7168"` |
| `--shape-list` | 二选一 | `shape_list.md` |
| `--npu-device` | ❌ | 0-7，默认取 config |
| `--pattern` | ❌ | 默认 `*.py`，如 `"gemm*.py"` |

输入: code_lib.zip + batch_verification.py + (shape_list.md)
输出: 服务器后台启动 batch_verification → 生成 results/ 目录

### server_check.py — 状态/进度

```bash
python script/server_check.py clean      # 删除服务器整个 batch_verification 目录
python script/server_check.py status     # 查看测试进程是否在运行
python script/server_check.py progress   # 查看日志末尾 + 结果目录文件数
python script/server_check.py results    # 列出已生成的结果目录
python script/server_check.py summary    # 查看 summary.json
python script/server_check.py durations  # 按性能排序显示所有 kernel
python script/server_check.py kill       # 安全停止当前 NPU 上的测试进程
```

### analyze_results.py — 下载结果

输入: 服务器 `/root/MyAICode/batch_verification/results/summary.json`
输出 (到本地 `output/`):
- `best_kernels_report.xlsx` — 每个 shape 最优 kernel 一行 (Shape / M/N/K / Duration / Kernel File / Kernel Name)
- `*.py` — 所有最优 kernel 源代码

### batch_verification.py — 服务器端 (无需手动调用)

由 verify_code.py 自动启动。扫描 codes/<subdir>/*.py，用 msprof 逐个测试，生成 AAAA.json / summary.json。

## 📝 Shape 列表格式

`shape_list.md` 每行一个 shape，空格或逗号分隔，支持 `#` 注释:

```
128 4096 7168
256,8192,14336
# LLM 推理
1 4096 4096
```

## ❓ FAQ

| 问题 | 解决 |
|------|------|
| SSH 连接失败 | `chmod 600 密钥文件路径` |
| code_lib.zip 不存在 | 确保项目根目录有 `code_lib.zip` |
| 测试太慢 | 减少 shape_list 中的 shape 数量；用 `--pattern` 过滤 kernel |
| 中断测试 | `python script/server_check.py kill` |
