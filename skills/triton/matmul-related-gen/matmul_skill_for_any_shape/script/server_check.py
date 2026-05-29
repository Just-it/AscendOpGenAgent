#!/usr/bin/env python3
"""
通用服务器检查脚本 - 用于检查测试进度、状态和结果

使用方法：
python script/server_check.py status    # 检查测试状态
python script/server_check.py progress  # 检查测试进度
python script/server_check.py results   # 检查测试结果
python script/server_check.py summary   # 查看汇总结果
python script/server_check.py clean     # 清理服务器
python script/server_check.py kill      # 清除指定NPU上的进程
"""

import paramiko
import json
import sys

def create_ssh_client(config):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    auth_method = config.get('auth_method', 'key')
    username = config.get('username', 'root')
    
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
    
    return ssh

def run_command(ssh, container, command):
    full_cmd = f'docker exec {container} bash -c "{command}"'
    stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=30)
    return stdout.read().decode('utf-8'), stderr.read().decode('utf-8')

def check_status(config):
    """检查服务器上的测试状态"""
    ssh = create_ssh_client(config)
    try:
        output, error = run_command(ssh, config['docker_container'], 
            "ps aux | grep -E 'python.*batch_verification|msprof' | grep -v grep || echo 'No running verification processes'")
        print("=== 运行状态 ===")
        print(output)
        if error:
            print(f"Error: {error}")
    finally:
        ssh.close()

def check_progress(config):
    """检查测试进度"""
    ssh = create_ssh_client(config)
    try:
        # 检查日志
        output, error = run_command(ssh, config['docker_container'], 
            'if [ -f /root/MyAICode/batch_verification/batch_verification.log ]; then tail -30 /root/MyAICode/batch_verification/batch_verification.log; else echo "Log file not found"; fi')
        print("=== 测试进度 ===")
        print(output)
        
        # 检查结果目录
        output, error = run_command(ssh, config['docker_container'], 
            'ls -la /root/MyAICode/batch_verification/results/ 2>/dev/null | wc -l')
        print(f"\n结果目录文件数: {output.strip()}")
        
    finally:
        ssh.close()

def check_results(config):
    """检查测试结果"""
    ssh = create_ssh_client(config)
    try:
        output, error = run_command(ssh, config['docker_container'], 
            'ls -la /root/MyAICode/batch_verification/results/ 2>/dev/null | head -20')
        print("=== 结果目录 ===")
        print(output)
    finally:
        ssh.close()

def check_summary(config):
    """查看汇总结果"""
    ssh = create_ssh_client(config)
    try:
        output, error = run_command(ssh, config['docker_container'], 
            'cat /root/MyAICode/batch_verification/results/summary.json 2>/dev/null || echo "summary.json not found"')
        print("=== 汇总结果 ===")
        print(output)
    finally:
        ssh.close()

def check_all_durations(config):
    """查看所有kernel的task_duration（从summary.json读取）"""
    import json as json_module
    ssh = create_ssh_client(config)
    try:
        output, error = run_command(ssh, config['docker_container'],
            'cat /root/MyAICode/batch_verification/results/summary.json 2>/dev/null || echo "SUMMARY_NOT_FOUND"')

        if not output or "SUMMARY_NOT_FOUND" in output:
            print("summary.json not found")
            return

        try:
            summary = json_module.loads(output)
        except json_module.JSONDecodeError as e:
            print(f"Error parsing summary.json: {e}")
            return

        durations = []
        for item in summary.get('results', []):
            durations.append({
                'kernel': item.get('kernel', 'unknown'),
                'file': item.get('file', 'unknown'),
                'duration': item.get('duration', 999999),
                'passed': item.get('passed', False),
                'shape': item.get('shape', {})
            })

        if durations:
            print("=== 所有Kernel性能 ===")
            print(f"{'Kernel':<60} {'Shape':<25} {'Duration(us)':<15} {'Passed'}")
            print("=" * 115)
            for item in sorted(durations, key=lambda x: x['duration']):
                status = "✓" if item['passed'] else "✗"
                shape = item.get('shape', {})
                shape_str = f"M={shape.get('M','?')} N={shape.get('N','?')} K={shape.get('K','?')}"
                print(f"{item['kernel']:<60} {shape_str:<25} {item['duration']:<15.2f} {status}")
            print("=" * 115)
            print(f"Total: {len(durations)} kernels")
        else:
            print("No results found in summary.json")

    finally:
        ssh.close()

def check_detail(config, kernel_name):
    """查看单个kernel的详细结果"""
    ssh = create_ssh_client(config)
    try:
        output, error = run_command(ssh, config['docker_container'], 
            f'cat /root/MyAICode/batch_verification/results/{kernel_name}/AAAA.json 2>/dev/null || echo "AAAA.json not found"')
        print("=== 详细结果 ===")
        print(output)
    finally:
        ssh.close()

def clean_server(config):
    """清理服务器（删除整个batch_verification工作目录）"""
    ssh = create_ssh_client(config)
    try:
        output, error = run_command(ssh, config['docker_container'], 
            'rm -rf /root/MyAICode/batch_verification && echo "Cleanup completed: /root/MyAICode/batch_verification removed"')
        print("=== 清理结果 ===")
        print(output)
        if error:
            print(f"Error: {error}")
    finally:
        ssh.close()

def kill_npu_processes(config):
    """清除指定NPU上的相关进程"""
    npu_device = config.get('npu_device', 'unknown')
    ssh = create_ssh_client(config)
    try:
        print(f"=== 正在清除 NPU {npu_device} 上的进程 ===")
        
        # 步骤1: 找到指定NPU的batch_verification主进程PID
        find_main_cmd = f"ps aux | grep 'python.*batch_verification.*--npu-device {npu_device}' | grep -v grep | awk '{{print $2}}'"
        main_pids_output, _ = run_command(ssh, config['docker_container'], find_main_cmd)
        main_pids = [pid.strip() for pid in main_pids_output.split('\n') if pid.strip()]
        
        if not main_pids:
            print(f"未找到 NPU {npu_device} 上的运行进程")
            return
        
        print(f"找到主进程 PID: {', '.join(main_pids)}")
        
        # 步骤2: 显示将要清除的进程
        print("\n将要清除的进程:")
        for pid in main_pids:
            show_cmd = f"ps aux | grep -E 'PID|{pid}' | head -5"
            output, _ = run_command(ssh, config['docker_container'], show_cmd)
            print(output)
        
        # 步骤3: 找到这些主进程的所有子进程（msprof, msopprof等）
        all_pids_to_kill = main_pids.copy()
        for main_pid in main_pids:
            find_children_cmd = f"pgrep -P {main_pid}"
            children_output, _ = run_command(ssh, config['docker_container'], find_children_cmd)
            child_pids = [pid.strip() for pid in children_output.split('\n') if pid.strip()]
            all_pids_to_kill.extend(child_pids)
            
            # 递归查找孙子进程
            for child_pid in child_pids:
                find_grandchildren_cmd = f"pgrep -P {child_pid}"
                grandchildren_output, _ = run_command(ssh, config['docker_container'], find_grandchildren_cmd)
                grandchild_pids = [pid.strip() for pid in grandchildren_output.split('\n') if pid.strip()]
                all_pids_to_kill.extend(grandchild_pids)
        
        if len(all_pids_to_kill) > len(main_pids):
            print(f"\n包含子进程 PID: {', '.join(all_pids_to_kill[len(main_pids):])}")
        
        # 步骤4: 执行kill
        pids_str = ' '.join(all_pids_to_kill)
        kill_cmd = f"kill -9 {pids_str} 2>/dev/null || true"
        run_command(ssh, config['docker_container'], kill_cmd)
        
        # 步骤5: 检查是否清除成功
        import time
        time.sleep(1)
        
        check_cmd = f"ps aux | grep 'python.*batch_verification.*--npu-device {npu_device}' | grep -v grep || echo '已清除'"
        output, _ = run_command(ssh, config['docker_container'], check_cmd)
        print("\n当前进程状态:")
        print(output)
        
        if '已清除' in output:
            print(f"\n✅ NPU {npu_device} 进程清除完成！")
        else:
            print(f"\n⚠️  部分进程可能未清除，请手动检查")
            
    finally:
        ssh.close()

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    try:
        with open('script/server_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: script/server_config.json not found")
        return
    
    action = sys.argv[1].lower()
    
    if action == 'status':
        check_status(config)
    elif action == 'progress':
        check_progress(config)
    elif action == 'results':
        check_results(config)
    elif action == 'summary':
        check_summary(config)
    elif action == 'clean':
        clean_server(config)
    elif action == 'kill':
        kill_npu_processes(config)
    elif action == 'detail':
        if len(sys.argv) < 3:
            print("Usage: python script/server_check.py detail <kernel_name>")
            return
        kernel_name = sys.argv[2]
        check_detail(config, kernel_name)
    elif action == 'durations':
        check_all_durations(config)
    else:
        print(f"Unknown action: {action}")
        print(__doc__)

if __name__ == '__main__':
    main()
