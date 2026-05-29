#!/usr/bin/env python3
import paramiko
import json
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_FILE = os.path.join(PROJECT_ROOT, 'script', '.upload_status.json')

WORK_DIR = '/root/MyAICode/batch_verification'
CODE_REMOTE_DIR = WORK_DIR + '/codes'
SCRIPT_PATH = WORK_DIR + '/batch_verification.py'

CODE_SUBDIRS = ['code-fp16', 'code-fp16-fuse', 'code-int8', 'code-int8-fuse']


def create_ssh_client(config):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=config['ip'],
        username='root',
        key_filename=config['ssh_key_path'],
        port=22,
        timeout=15
    )
    return ssh


def run_command(ssh, container, command):
    full_cmd = f'docker exec {container} bash -c "{command}"'
    stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=30)
    return stdout.read().decode('utf-8'), stderr.read().decode('utf-8')


def write_status(value):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"status": value}, f)


def main():
    parser = argparse.ArgumentParser(description='Check if code files and batch script exist on server')
    parser.add_argument('--code-subdir', required=True, choices=CODE_SUBDIRS,
                        help='Code subdirectory to check')
    parser.add_argument('--server-config', default=os.path.join(PROJECT_ROOT, 'script', 'server_config.json'),
                        help='Path to server_config.json')
    args = parser.parse_args()

    code_subdir = args.code_subdir
    code_subdir_path = f'{CODE_REMOTE_DIR}/{code_subdir}'

    config_path = args.server_config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f'check_file_exist: config not found -> status=false')
        write_status("false")
        sys.exit(1)

    container = config['docker_container']
    ssh = create_ssh_client(config)

    try:
        py_count_output, _ = run_command(ssh, container,
            f'ls {code_subdir_path}/*.py 2>/dev/null | wc -l')
        py_count = int(py_count_output.strip()) if py_count_output.strip().isdigit() else 0

        script_output, _ = run_command(ssh, container,
            f'test -f {SCRIPT_PATH} && echo EXISTS')
        script_exists = 'EXISTS' in script_output

        if py_count >= 1 and script_exists:
            print(f'check_file_exist: {code_subdir}({py_count} .py files) + batch_verification.py -> status=true')
            write_status("true")
        else:
            print(f'check_file_exist: {code_subdir}={py_count} .py scripts={"YES" if script_exists else "NO"} -> status=false')
            write_status("false")

    except Exception as e:
        print(f'check_file_exist: error -> status=false ({e})')
        write_status("false")
    finally:
        ssh.close()


if __name__ == '__main__':
    main()
