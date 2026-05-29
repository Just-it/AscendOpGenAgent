#!/usr/bin/env python3
"""
配置检查脚本 - 检测服务器配置文件是否存在且有效

使用方法：
python script/check_config.py                    # 检查配置文件
python script/check_config.py --test-connection  # 测试 SSH 连接
"""

import json
import sys
import os
import argparse
from pathlib import Path

REQUIRED_FIELDS = ['ip', 'username', 'auth_method', 'docker_container', 'npu_device']

def check_config_exists(config_path):
    """检查配置文件是否存在"""
    return os.path.exists(config_path)

def validate_config(config):
    """验证配置文件内容"""
    errors = []
    
    for field in REQUIRED_FIELDS:
        if field not in config:
            errors.append(f"缺少必填字段: {field}")
    
    if 'auth_method' in config:
        auth_method = config['auth_method']
        if auth_method == 'password':
            if not config.get('password'):
                errors.append("密码认证方式需要填写 'password' 字段")
        elif auth_method == 'key':
            if not config.get('ssh_key_path'):
                errors.append("密钥认证方式需要填写 'ssh_key_path' 字段")
            elif not os.path.exists(config['ssh_key_path']):
                errors.append(f"密钥文件不存在: {config['ssh_key_path']}")
        else:
            errors.append(f"不支持的认证方式: {auth_method} (应为 'password' 或 'key')")
    
    return errors

def test_connection(config):
    """测试 SSH 连接"""
    try:
        import paramiko
    except ImportError:
        print("❌ 错误: 未安装 paramiko 库，请运行: pip install paramiko")
        return False
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        auth_method = config.get('auth_method', 'key')
        username = config.get('username', 'root')
        
        print(f"正在连接 {config['ip']}...")
        
        if auth_method == 'password':
            ssh.connect(
                hostname=config['ip'],
                username=username,
                password=config.get('password'),
                port=22,
                timeout=15
            )
        else:
            ssh.connect(
                hostname=config['ip'],
                username=username,
                key_filename=config['ssh_key_path'],
                port=22,
                timeout=15
            )
        
        print("✅ SSH 连接成功!")
        
        if config.get('docker_container'):
            stdin, stdout, stderr = ssh.exec_command(
                f"docker inspect {config['docker_container']}", 
                timeout=10
            )
            if stdout.read():
                print(f"✅ Docker 容器 '{config['docker_container']}' 存在")
            else:
                print(f"⚠️  警告: Docker 容器 '{config['docker_container']}' 不存在或无法访问")
        
        ssh.close()
        return True
        
    except paramiko.AuthenticationException:
        print("❌ 认证失败: 请检查用户名和密码/密钥")
        return False
    except paramiko.SSHException as e:
        print(f"❌ SSH 连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def print_config_template():
    """打印配置模板"""
    print("\n📋 配置模板 (保存到 script/server_config.json):")
    print("""
{
  "ip": "你的服务器IP",
  "username": "root",
  "auth_method": "password",
  "password": "你的密码",
  "docker_container": "容器名称",
  "npu_device": "NPU设备ID"
}

或使用密钥认证:
{
  "ip": "你的服务器IP",
  "username": "root",
  "auth_method": "key",
  "password": "",
  "ssh_key_path": "密钥文件路径",
  "docker_container": "容器名称",
  "npu_device": "NPU设备ID"
}
""")

def main():
    parser = argparse.ArgumentParser(description='检查服务器配置文件')
    parser.add_argument('--test-connection', action='store_true', 
                        help='测试 SSH 连接')
    parser.add_argument('--config', default='script/server_config.json',
                        help='配置文件路径 (默认: script/server_config.json)')
    args = parser.parse_args()
    
    config_path = args.config
    
    print("=" * 50)
    print("🔍 配置检查")
    print("=" * 50)
    
    if not check_config_exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print_config_template()
        sys.exit(1)
    
    print(f"✅ 配置文件存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件格式错误: {e}")
        sys.exit(1)
    
    errors = validate_config(config)
    if errors:
        print("\n❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        print_config_template()
        sys.exit(1)
    
    print("✅ 配置验证通过")
    print(f"  - 服务器: {config['ip']}")
    print(f"  - 用户: {config['username']}")
    print(f"  - 认证方式: {config['auth_method']}")
    print(f"  - Docker 容器: {config['docker_container']}")
    print(f"  - NPU 设备: {config['npu_device']}")
    
    if args.test_connection:
        print("\n" + "=" * 50)
        print("🔌 测试连接")
        print("=" * 50)
        if not test_connection(config):
            sys.exit(1)
    
    print("\n✅ 所有检查通过，可以开始使用!")

if __name__ == '__main__':
    main()
