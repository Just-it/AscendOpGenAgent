# Matmul Code Generator

特别注意，需要环境中事先安装好 `msprof op`（强调不是 `msprof`）。

## 使用方法

使用方式有 2 种：

**1. 单shape测试**（单个shape约需20~30分钟，例如 M=128 N=7168 K=4096 性能约为57us）：

直接在终端执行：

```
请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 FP16 高性能算子
或 请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 FP16 ND格式算子
或 请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 INT8 高性能算子
或 请按照 SKILL.md 生成 M=128 N=7168 K=4096 的 INT8 ND格式算子
```

**2. 多shape测试**（使用 `shape_list.md` 中的 shape 列表逐一测试）：

直接在终端执行：

```
请按照 SKILL.md 生成 FP16 高性能算子，且请使用 shape_list 中的 shape 进行测试
或 请按照 SKILL.md 生成 FP16 ND格式算子，且请使用 shape_list 中的 shape 进行测试
或 请按照 SKILL.md 生成 INT8 高性能算子，且请使用 shape_list 中的 shape 进行测试
或 请按照 SKILL.md 生成 INT8 ND格式算子，且请使用 shape_list 中的 shape 进行测试
```

代码和测试报告会存储在 `output` 文件夹下。

## 服务器配置

编辑 `script/server_config.json`：

```json
{
  "ip": "你的服务器IP",
  "username": "root",
  "auth_method": "key",
  "password": "",
  "ssh_key_path": "你的密钥文件路径",
  "docker_container": "容器名称",
  "host_temp_dir": "/tmp/triton_upload",
  "docker_working_dir": "/root/MyAICode",
  "npu_device": "NPU设备ID"
}
```

| 字段 | 说明 |
|------|------|
| `ip` | 服务器 IP 地址 |
| `auth_method` | `"password"` 密码认证 或 `"key"` 密钥认证 |
| `password` | 密码认证时填写 |
| `ssh_key_path` | 密钥认证时填写 .pem 文件路径 |
| `docker_container` | Docker 容器名称 |
| `npu_device` | NPU 设备 ID（0-7） |

配置完成后运行检测：

```bash
python script/check_config.py
```
