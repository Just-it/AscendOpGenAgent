import os
import sys
import time
import argparse
import json
import paramiko
import posixpath
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_TIMEOUT = 120
UPLOAD_STATUS_FILE = os.path.join(PROJECT_ROOT, 'script', '.upload_status.json')


def create_ssh_client(host, port, user, password=None, key_path=None):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    if password:
        client.connect(host, port=port, username=user, password=password, timeout=30)
    elif key_path and os.path.exists(key_path):
        client.connect(host, port=port, username=user, key_filename=key_path, timeout=30)
    else:
        raise ValueError("Either password or key_path must be provided")
    
    return client


def run_ssh_command(client, command, max_retries=3, timeout=300):
    for attempt in range(max_retries):
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            stdout_content = stdout.read().decode('utf-8', errors='replace')
            stderr_content = stderr.read().decode('utf-8', errors='replace')
            returncode = stdout.channel.recv_exit_status()
            return returncode, stdout_content, stderr_content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return -1, '', str(e)


def poll_verify(client, container, check_cmd, label, interval=5):
    """Poll until check_cmd succeeds or UPLOAD_TIMEOUT expires"""
    deadline = time.time() + UPLOAD_TIMEOUT
    while time.time() < deadline:
        rc, out, _ = run_ssh_command(client,
            f'docker exec {container} bash -c "{check_cmd}"', timeout=15)
        if rc == 0 and out.strip():
            print(f'  [{label}] verified OK after {int(time.time() - (deadline - UPLOAD_TIMEOUT))}s', flush=True)
            return True
        time.sleep(interval)
    print(f'  [{label}] verify FAILED after {UPLOAD_TIMEOUT}s', flush=True)
    return False


def upload_file_to_docker(client, container, local_path, remote_tmp_name, docker_dest, label):
    """Upload a file via SFTP, then poll-verify until it appears or timeout"""
    remote_tmp = posixpath.join('/tmp', remote_tmp_name)

    deadline = time.time() + UPLOAD_TIMEOUT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        print(f'  [{label}] upload attempt {attempt} ...', flush=True)

        sftp = client.open_sftp()
        sftp.put(local_path, remote_tmp)
        sftp.close()

        rc, _, err = run_ssh_command(client,
            f'docker cp {remote_tmp} {container}:{docker_dest}', timeout=60)
        run_ssh_command(client, f'rm -f {remote_tmp}', timeout=10)

        if rc != 0:
            print(f'  [{label}] docker cp failed: {err[:200]}', flush=True)
            continue

        print(f'  [{label}] docker cp done, polling for {docker_dest} ...', flush=True)
        if poll_verify(client, container,
                       f'test -e {docker_dest} && echo OK', label):
            return True

        print(f'  [{label}] retrying upload ...', flush=True)

    print(f'  [{label}] FATAL after {UPLOAD_TIMEOUT}s', flush=True)
    return False


def upload_code_lib(client, container, zip_path, remote_dir):
    """Upload code_lib.zip, extract, poll-verify .py files exist"""
    if not os.path.isfile(zip_path):
        print(f'code_lib.zip not found: {zip_path}', flush=True)
        return False

    zip_name = os.path.basename(zip_path)
    deadline = time.time() + UPLOAD_TIMEOUT
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        print(f'  [code_lib.zip] upload attempt {attempt} ...', flush=True)

        remote_tmp = posixpath.join('/tmp', zip_name)
        sftp = client.open_sftp()
        sftp.put(zip_path, remote_tmp)
        sftp.close()

        unzip_cmd = (
            f'docker exec {container} mkdir -p {remote_dir} && '
            f'docker cp {remote_tmp} {container}:{remote_dir}/ && '
            f'docker exec {container} bash -c "cd {remote_dir} && unzip -o {zip_name} && rm -f {zip_name}"'
        )
        rc, out, err = run_ssh_command(client, unzip_cmd, timeout=120)
        run_ssh_command(client, f'rm -f {remote_tmp}', timeout=10)

        if rc != 0:
            print(f'  [code_lib.zip] unzip error (rc={rc}): {err[:500]}', flush=True)
            time.sleep(5)
            continue
        
        print(f'  [code_lib.zip] unzip output: {out[:200]}', flush=True)

        print(f'  [code_lib.zip] extract done, polling for .py files ...', flush=True)
        if poll_verify(client, container,
                       f'ls {remote_dir}/code-fp16/*.py 2>/dev/null | wc -l',
                       'code_lib.zip'):
            return True

        print(f'  [code_lib.zip] retrying ...', flush=True)

    print(f'  [code_lib.zip] FATAL after {UPLOAD_TIMEOUT}s', flush=True)
    return False


def main():
    os.makedirs(os.path.dirname(UPLOAD_STATUS_FILE), exist_ok=True)
    with open(UPLOAD_STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"status": "false"}, f)

    parser = argparse.ArgumentParser(description='Generate and test Triton matmul kernels')
    parser.add_argument('--code-subdir', required=True,
                        choices=['code-fp16', 'code-fp16-fuse', 'code-int8', 'code-int8-fuse'])
    parser.add_argument('--server-config', required=True)
    parser.add_argument('--npu-device', default=None)
    parser.add_argument('--pattern', default='*.py')
    parser.add_argument('--shape', default=None)
    parser.add_argument('--shape-list', default=None)
    args = parser.parse_args()

    code_lib_path = os.path.join(PROJECT_ROOT, 'code_lib.zip')
    if not os.path.isfile(code_lib_path):
        print(f'Error: code_lib.zip not found at {code_lib_path}')
        sys.exit(1)

    with open(args.server_config, 'r', encoding='utf-8') as f:
        server_config = json.load(f)

    host = server_config['ip']
    username = server_config.get('username', 'root')
    auth_method = server_config.get('auth_method', 'key')
    key_path = server_config.get('ssh_key_path')
    password = server_config.get('password')
    container = server_config['docker_container']
    npu_device = args.npu_device or server_config.get('npu_device', '0')

    if not all([host, container]):
        print('Error: Missing server config fields')
        sys.exit(1)

    work_dir = '/root/MyAICode/batch_verification'
    code_remote_dir = posixpath.join(work_dir, 'codes')
    output_dir = posixpath.join(work_dir, 'results')
    script_path = posixpath.join(work_dir, 'batch_verification.py')
    log_file = posixpath.join(work_dir, 'batch_verification.log')

    shape_list_remote = None
    if args.shape_list:
        if not os.path.isfile(args.shape_list):
            print(f'Error: Shape list file not found: {args.shape_list}')
            sys.exit(1)
        shape_list_remote = posixpath.join(work_dir, 'shape_list.txt')

    print(f'Connecting to server: {host}')
    print(f'Auth method: {auth_method}')
    print(f'Username: {username}')
    print(f'Docker container: {container}')
    print(f'Code subdirectory: {args.code_subdir}')
    print(f'NPU device: {npu_device}')
    if args.shape:
        print(f'Shape: {args.shape}')
    if args.shape_list:
        print(f'Shape list: {args.shape_list}')
    print()

    if auth_method == 'password':
        client = create_ssh_client(host, 22, username, password=password)
    else:
        client = create_ssh_client(host, 22, username, key_path=key_path)

    try:
        print('=== Step 1: Clean server ===')
        run_ssh_command(client,
            f'docker exec {container} bash -c "rm -rf {work_dir}"', timeout=30)
        run_ssh_command(client,
            f'docker exec {container} mkdir -p {work_dir} {code_remote_dir} {output_dir}',
            timeout=15)
        print('Server cleanup OK')
        print()

        print('=== Step 2a: Upload code_lib.zip ===')
        if not upload_code_lib(client, container, code_lib_path, code_remote_dir):
            print('FATAL: Failed to upload code_lib.zip')
            sys.exit(1)
        print()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_batch = os.path.join(script_dir, 'batch_verification.py')

        print('=== Step 2b: Upload batch_verification.py ===')
        if not upload_file_to_docker(client, container, local_batch,
                                      'batch_verification_tmp.py', script_path,
                                      'batch_verification.py'):
            print('FATAL: Failed to upload batch_verification.py')
            sys.exit(1)
        run_ssh_command(client,
            f'docker exec {container} chmod +x {script_path}', timeout=10)
        print()

        if args.shape_list:
            print('=== Step 2c: Upload shape_list ===')
            if not upload_file_to_docker(client, container, args.shape_list,
                                          'shape_list_tmp.txt', shape_list_remote,
                                          'shape_list'):
                print('FATAL: Failed to upload shape_list')
                sys.exit(1)
            print()

        print('=== Step 3: Start batch verification ===')
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        shape_arg = f"--shape '{args.shape}'" if args.shape else ''
        shape_list_arg = f'--shape-list {shape_list_remote}' if args.shape_list else ''

        exec_cmd = (
            f'docker exec {container} bash -c '
            f'"nohup python3 {script_path} '
            f'--code-dir {code_remote_dir} '
            f'--code-subdir {args.code_subdir} '
            f'--work-dir {work_dir} '
            f'--output-dir {output_dir} '
            f'--npu-device {npu_device} '
            f'--pattern {args.pattern} '
            f'{shape_arg} {shape_list_arg} '
            f'> {log_file} 2>&1 &"'
        )
        run_ssh_command(client, exec_cmd, timeout=30)

        print(f'Batch verification started at {start_time}')
        print(f'Log: {log_file}')
        print(f'Results: {output_dir}')
        print()
        print('Monitor: python script/server_check.py status')
        print('Monitor: python script/server_check.py progress')

        with open(UPLOAD_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"status": "true"}, f)

    finally:
        client.close()


if __name__ == '__main__':
    main()
