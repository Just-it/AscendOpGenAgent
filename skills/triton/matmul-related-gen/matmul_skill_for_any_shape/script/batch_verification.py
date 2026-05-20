#!/usr/bin/env python3
"""
Batch verification script - runs on server
Scans multiple kernel code files and executes msprof op for each one
"""

import os
import subprocess
import json
import re
import ast
import argparse
import glob
import difflib
from datetime import datetime


def extract_kernel_name(code_content, filename):
    """Extract kernel name from code using AST parsing"""
    kernel_names = []
    try:
        tree = ast.parse(code_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_triton_jit = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == 'jit':
                            if isinstance(decorator.value, ast.Name) and decorator.value.id == 'triton':
                                has_triton_jit = True
                    elif isinstance(decorator, ast.Name) and decorator.id == 'jit':
                        has_triton_jit = True
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr == 'jit':
                                if isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == 'triton':
                                    has_triton_jit = True
                if has_triton_jit:
                    kernel_names.append(node.name)
    except SyntaxError as e:
        print(f'AST parse error: {e}')
    
    if not kernel_names:
        return None
    if len(kernel_names) == 1:
        return kernel_names[0]
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    best_name = None
    best_ratio = -1
    for name in kernel_names:
        ratio = difflib.SequenceMatcher(None, base_filename, name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = name
    return best_name


def replace_shape_in_code(code_content, M, N, K):
    """Replace M, N, K values in the main() function"""
    patterns = [
        (r'(\n    M = )\d+', rf'\g<1>{M}'),
        (r'(\n    N = )\d+', rf'\g<1>{N}'),
        (r'(\n    K = )\d+', rf'\g<1>{K}'),
    ]
    
    modified_code = code_content
    for pattern, replacement in patterns:
        modified_code = re.sub(pattern, replacement, modified_code)
    
    return modified_code


def run_single_kernel(code_file, work_dir, output_dir, npu_device, shape=None):
    """Run verification for a single kernel code file"""
    filename = os.path.basename(code_file)
    kernel_name = os.path.splitext(filename)[0]
    
    if shape:
        M, N, K = shape
        result_dirname = f"{kernel_name}_M{M}_N{N}_K{K}"
    else:
        result_dirname = kernel_name
    
    result_dir = os.path.join(output_dir, result_dirname)
    os.makedirs(result_dir, exist_ok=True)
    
    stdout_file = os.path.join(result_dir, 'stdout.txt')
    stderr_file = os.path.join(result_dir, 'stderr.txt')
    aaaa_file = os.path.join(result_dir, 'AAAA.json')
    
    with open(code_file, 'r') as f:
        original_code = f.read()
    
    extracted_kernel = extract_kernel_name(original_code, code_file)
    if extracted_kernel:
        kernel_name = extracted_kernel
    
    code_modified = original_code
    
    if shape:
        M, N, K = shape
        code_modified = replace_shape_in_code(code_modified, M, N, K)
    
    device_setting = f"import torch\ntorch.npu.set_device({npu_device})\n"
    code_with_device = code_modified.replace('import torch\n', device_setting, 1)
    
    temp_code_file = os.path.join(work_dir, f'temp_{filename}')
    with open(temp_code_file, 'w') as f:
        f.write(code_with_device)
    
    start_time = datetime.now().strftime('%H:%M:%S')
    
    with open(aaaa_file, 'w') as f:
        aaaa_data = {'status': 'in_progress', 'passed': None, 'start_time': start_time}
        if shape:
            aaaa_data['shape'] = {'M': shape[0], 'N': shape[1], 'K': shape[2]}
        json.dump(aaaa_data, f)
    
    env = os.environ.copy()
    env['ASCEND_VISIBLE_DEVICES'] = npu_device
    
    cmd = f'PS1=dummy && source /root/.bashrc && export ASCEND_VISIBLE_DEVICES={npu_device} && msprof op --output={result_dir} --kernel-name={kernel_name} python {temp_code_file}'
    
    shape_str = f" (M={shape[0]}, N={shape[1]}, K={shape[2]})" if shape else ""
    print(f'[{start_time}] Running: {filename}{shape_str} (kernel: {kernel_name})')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash', env=env)
    
    with open(stdout_file, 'w') as f:
        f.write(result.stdout)
    with open(stderr_file, 'w') as f:
        f.write(result.stderr)
    
    stdout_content = result.stdout
    stderr_content = result.stderr
    
    passed = 'Test passed!' in stdout_content
    
    task_duration_match = re.search(r'Task Duration\(us\):\s*([0-9.]+)', stdout_content)
    task_duration = float(task_duration_match.group(1)) if task_duration_match else 999999
    
    end_time = datetime.now().strftime('%H:%M:%S')
    
    data = {
        'status': 'completed',
        'passed': passed,
        'start_time': start_time,
        'end_time': end_time,
        'task_duration': task_duration,
        'kernel_name': kernel_name,
        'code_file': filename,
        'stdout': stdout_content,
        'stderr': stderr_content
    }
    
    if shape:
        data['shape'] = {'M': shape[0], 'N': shape[1], 'K': shape[2]}
    
    with open(aaaa_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    status = 'PASSED' if passed else 'FAILED'
    print(f'[{end_time}] {status}: {filename}{shape_str} (duration: {task_duration} us)')
    
    os.remove(temp_code_file)
    
    return passed, task_duration, kernel_name


def parse_shape(shape_str):
    """Parse shape string like '128 4096 7168' (or '128,4096,7168') to (M, N, K)"""
    if not shape_str:
        return None
    # Try splitting by space first, then comma for backward compatibility
    parts = shape_str.split()
    if len(parts) != 3:
        parts = shape_str.split(',')
    if len(parts) != 3:
        raise ValueError(f"Invalid shape format: {shape_str}. Expected 'M N K' or 'M,N,K'")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def parse_shape_list_file(file_path):
    """Parse shape list from a file, returns list of (M, N, K) tuples"""
    shapes = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('```'):
                continue
            try:
                shape = parse_shape(line)
                shapes.append(shape)
            except:
                continue
    return shapes


def main():
    parser = argparse.ArgumentParser(description='Batch verification script for multiple kernel codes')
    parser.add_argument('--code-dir', required=True, help='Directory containing all kernel code subdirectories')
    parser.add_argument('--code-subdir', required=True, help='Which subdirectory to test (e.g., code-fp16, code-fp16-fuse)')
    parser.add_argument('--work-dir', default='/root/MyAICode/batch_verification', help='Working directory')
    parser.add_argument('--output-dir', default='/root/MyAICode/batch_verification/results', help='Output directory')
    parser.add_argument('--npu-device', default='0', help='NPU device ID')
    parser.add_argument('--pattern', default='*.py', help='File pattern to match (default: *.py)')
    parser.add_argument('--shape', default=None, help='Matrix shape as M N K (e.g., "128 4096 7168")')
    parser.add_argument('--shape-list', default=None, help='File containing list of shapes (one per line)')
    args = parser.parse_args()

    CODE_DIR = os.path.join(args.code_dir, args.code_subdir)
    WORK_DIR = args.work_dir
    OUTPUT_DIR = args.output_dir
    NPU_DEVICE = args.npu_device
    PATTERN = args.pattern
    
    # Parse shapes: either from --shape or from --shape-list
    SHAPES = []
    if args.shape:
        SHAPES = [parse_shape(args.shape)]
    elif args.shape_list:
        SHAPES = parse_shape_list_file(args.shape_list)
    
    # If no shapes specified, use None (use shape from code)
    if not SHAPES:
        SHAPES = [None]

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    code_files = sorted(glob.glob(os.path.join(CODE_DIR, PATTERN)))
    
    if not code_files:
        print(f'No code files found in {CODE_DIR} with pattern {PATTERN}')
        return
    
    print(f'Found {len(code_files)} code files in {CODE_DIR}')
    print(f'NPU Device: {NPU_DEVICE}')
    print(f'Number of shapes to test: {len(SHAPES)}')
    for i, shape in enumerate(SHAPES, 1):
        if shape:
            print(f'  Shape {i}: M={shape[0]}, N={shape[1]}, K={shape[2]}')
        else:
            print(f'  Shape {i}: (using shape from code)')
    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 60)
    
    all_results = []
    total_passed = 0
    total_failed = 0
    
    for shape_idx, shape in enumerate(SHAPES, 1):
        print(f'\n{"=" * 60}')
        if shape:
            print(f'TESTING SHAPE {shape_idx}/{len(SHAPES)}: M={shape[0]}, N={shape[1]}, K={shape[2]}')
        else:
            print(f'TESTING SHAPE {shape_idx}/{len(SHAPES)}: (using shape from code)')
        print('=' * 60)
        
        shape_results = []
        shape_passed = 0
        shape_failed = 0
        
        for code_idx, code_file in enumerate(code_files, 1):
            print(f'\n[{shape_idx}/{len(SHAPES)}][{code_idx}/{len(code_files)}] Processing: {os.path.basename(code_file)}')
            passed, duration, kernel_name = run_single_kernel(code_file, WORK_DIR, OUTPUT_DIR, NPU_DEVICE, shape)
            
            result_entry = {
                'file': os.path.basename(code_file),
                'kernel': kernel_name,
                'passed': passed,
                'duration': duration
            }
            if shape:
                result_entry['shape'] = {'M': shape[0], 'N': shape[1], 'K': shape[2]}
            shape_results.append(result_entry)
            all_results.append(result_entry)
            
            if passed:
                shape_passed += 1
                total_passed += 1
            else:
                shape_failed += 1
                total_failed += 1
        
        # Save per-shape summary
        if shape:
            shape_summary_file = os.path.join(OUTPUT_DIR, f'summary_M{shape[0]}_N{shape[1]}_K{shape[2]}.json')
        else:
            shape_summary_file = os.path.join(OUTPUT_DIR, 'summary_default.json')
        
        shape_summary = {
            'total': len(code_files),
            'passed': shape_passed,
            'failed': shape_failed,
            'npu_device': NPU_DEVICE,
            'timestamp': datetime.now().isoformat(),
            'results': shape_results
        }
        
        if shape:
            shape_summary['shape'] = {'M': shape[0], 'N': shape[1], 'K': shape[2]}
        
        with open(shape_summary_file, 'w') as f:
            json.dump(shape_summary, f, indent=2)
        
        print(f'\nShape {shape_idx} summary:')
        print(f'  Total:  {len(code_files)}')
        print(f'  Passed: {shape_passed}')
        print(f'  Failed: {shape_failed}')
        print(f'  Summary saved to: {shape_summary_file}')
    
    # Save overall summary
    summary_file = os.path.join(OUTPUT_DIR, 'summary.json')
    summary = {
        'total_shapes': len(SHAPES),
        'shapes': [{'M': s[0], 'N': s[1], 'K': s[2]} if s else None for s in SHAPES],
        'total_tests': len(SHAPES) * len(code_files),
        'total_passed': total_passed,
        'total_failed': total_failed,
        'npu_device': NPU_DEVICE,
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('\n' + '=' * 60)
    print('BATCH VERIFICATION SUMMARY (ALL SHAPES)')
    print('=' * 60)
    print(f'Total shapes: {len(SHAPES)}')
    print(f'Total tests: {len(SHAPES) * len(code_files)}')
    print(f'Total passed: {total_passed}')
    print(f'Total failed: {total_failed}')
    print(f'Overall summary saved to: {summary_file}')


if __name__ == '__main__':
    main()
