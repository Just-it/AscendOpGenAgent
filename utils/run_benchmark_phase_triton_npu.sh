#!/bin/bash
# 批量调度 triton-ascend-coder（TritonNPUKernelBench GPU Kernel 模式）
# 支持多 NPU 并行 + 算子索引分块执行 + 断点续跑
#
# 目录结构：{benchmark_dir}/{repo_name}/{op_name}/{op_name}.py
#
# 核心特性：
# 1. 自动为每个 repo 的 kernel 分配稳定索引（0,1,2...），持久化到注册表
# 2. START_INDEX/END_INDEX 基于 repo-local 索引分块执行
# 3. 断点续跑：已成功的算子自动跳过（验证 output_dir + summary.json）
#    失败的算子重新执行，未执行的算子正常执行
# 4. 执行完成后自动记录 output_dir 到进度文件

set -euo pipefail

# ── 默认值 ──
BENCHMARK_DIR=""
REPO_NAME=""
KERNELS=""
NPU_ID=0
NPU_LIST=""
OUTPUT_DIR=""
EXTRACTED_KERNELS_DIR="/home/z00841464/benchmark/data/output/extracted_kernels"
START_INDEX=0
END_INDEX=""
INDEX_REGISTRY=""
FORCE_REGISTRY=false
NO_SKIP=false

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --repo-name)     REPO_NAME="$2"; shift 2 ;;
        --kernels)       KERNELS="$2"; shift 2 ;;
        --npu)           NPU_ID="$2"; shift 2 ;;
        --npu-list)      NPU_LIST="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --extracted-kernels-dir) EXTRACTED_KERNELS_DIR="$2"; shift 2 ;;
        --start-index)   START_INDEX="$2"; shift 2 ;;
        --end-index)     END_INDEX="$2"; shift 2 ;;
        --index-registry) INDEX_REGISTRY="$2"; shift 2 ;;
        --force-registry) FORCE_REGISTRY=true; shift ;;
        --no-skip)       NO_SKIP=true; shift ;;
        -h|--help)
            echo "用法：bash utils/run_benchmark_triton_npu.sh --benchmark-dir <path> --repo-name <str> [--kernels <kernel_list>] [--npu <id> | --npu-list <list>] --output <path> [选项]"
            echo ""
            echo "参数:"
            echo "  --benchmark-dir        TritonNPUKernelBench 根目录路径 (必填)"
            echo "  --repo-name            Repo 名称，如 vllm, sglang, fbgemm (必填)"
            echo "  --kernels              指定算子名称列表，逗号分隔 (可选，不指定则扫描所有算子)"
            echo "  --extracted-kernels-dir 提取的 kernels 目录，包含 pt 文件 (默认：${EXTRACTED_KERNELS_DIR})"
            echo "  --npu                  单 NPU 设备 ID (默认 0，与 --npu-list 互斥)"
            echo "  --npu-list             多 NPU 列表，逗号分隔 (与 --npu 互斥，优先级更高)"
            echo "  --output               输出目录 (必填)"
            echo "  --start-index          算子起始索引(0基，默认0)，分块执行时使用"
            echo "  --end-index            算子结束索引(默认全部)，左闭右开"
            echo "  --index-registry       索引注册表路径 (默认: {benchmark_dir}/.op_index_registry.json)"
            echo "  --force-registry       强制重建索引注册表"
            echo "  --no-skip              强制重新执行已成功的算子（忽略断点续跑）"
            echo ""
            echo "示例:"
            echo "  # 全量执行 vllm 下所有算子"
            echo "  bash utils/run_benchmark_triton_npu.sh --benchmark-dir /path/to/TritonNPUKernelBench --repo-name vllm --npu 0 --output /path/to/output"
            echo ""
            echo "  # 分块执行：索引 0~4（共5个）"
            echo "  bash utils/run_benchmark_triton_npu.sh --benchmark-dir /path/to/TritonNPUKernelBench --repo-name vllm --npu 0 --output /path/to/output --start-index 0 --end-index 5"
            echo ""
            echo "  # 断点续跑：索引 0~9，已成功的自动跳过，失败的重新执行"
            echo "  bash utils/run_benchmark_triton_npu.sh --benchmark-dir /path/to/TritonNPUKernelBench --repo-name vllm --npu 0 --output /path/to/output --start-index 0 --end-index 10"
            echo ""
            echo "  # 多NPU + 分块"
            echo "  bash utils/run_benchmark_triton_npu.sh --benchmark-dir /path/to/TritonNPUKernelBench --repo-name vllm --npu-list \"0,1,2\" --output /path/to/output --start-index 0 --end-index 20"
            exit 0
            ;;
        *) echo "未知参数：$1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "错误：必须指定 --benchmark-dir"
    exit 1
fi

if [[ -z "$REPO_NAME" ]]; then
    echo "错误：必须指定 --repo-name"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误：必须指定 --output"
    exit 1
fi

REPO_DIR="${BENCHMARK_DIR}/${REPO_NAME}"
if [[ ! -d "$REPO_DIR" ]]; then
    echo "错误：目录不存在：${REPO_DIR}"
    exit 1
fi

# ── 确定脚本和 Agent 根目录 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 将关键路径转为绝对路径
BENCHMARK_DIR="$(cd "$BENCHMARK_DIR" && pwd)"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
if [[ -n "$EXTRACTED_KERNELS_DIR" && -d "$EXTRACTED_KERNELS_DIR" ]]; then
    EXTRACTED_KERNELS_DIR="$(cd "$EXTRACTED_KERNELS_DIR" && pwd)"
fi

# 切换到 Agent 根目录，确保 triton_ascend_output/ 位置固定
cd "$AGENT_ROOT"

# ── 索引注册表 ──
INDEX_REGISTRY="${INDEX_REGISTRY:-${BENCHMARK_DIR}/.op_index_registry.json}"

build_index_registry() {
    python3 -c "
import json, os, sys

benchmark_dir = '$BENCHMARK_DIR'
registry = {}

try:
    for repo in sorted(os.listdir(benchmark_dir)):
        repo_path = os.path.join(benchmark_dir, repo)
        if not os.path.isdir(repo_path):
            continue

        repo_idx = 0
        for kernel in sorted(os.listdir(repo_path)):
            kernel_path = os.path.join(repo_path, kernel)
            if not os.path.isdir(kernel_path):
                continue
            if '_gpu_perf.csv' in kernel:
                continue
            # 检查是否有 .py 描述文件
            py_file = os.path.join(kernel_path, f'{kernel}.py')
            if not os.path.exists(py_file):
                continue
            registry[f'{repo}/{kernel}'] = repo_idx
            repo_idx += 1

    with open('$INDEX_REGISTRY', 'w') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f'Built index registry: {len(registry)} kernels', file=sys.stderr)
except Exception as e:
    print(f'Error building registry: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1
}

# 如果不存在或强制重建，则构建注册表
if [[ ! -f "$INDEX_REGISTRY" ]] || [[ "$FORCE_REGISTRY" == true ]]; then
    echo "[INFO] 构建索引注册表: $INDEX_REGISTRY"
    build_index_registry
fi

# 校验注册表存在
if [[ ! -f "$INDEX_REGISTRY" ]]; then
    echo "错误：无法创建或找到索引注册表：$INDEX_REGISTRY"
    exit 1
fi

# ── 确定执行模式 ──
USE_PARALLEL=false
if [[ -n "$NPU_LIST" ]]; then
    USE_PARALLEL=true
    IFS=',' read -ra NPU_ARRAY <<< "$NPU_LIST"
    NPU_COUNT=${#NPU_ARRAY[@]}
    if [[ $NPU_COUNT -eq 0 ]]; then
        echo "错误：NPU 列表为空"
        exit 1
    fi
else
    NPU_ARRAY=("$NPU_ID")
    NPU_COUNT=1
fi

# ── 构建算子列表（带稳定索引）──
OP_LIST_RAW=$(python3 -c "
import json, sys

registry = json.load(open('$INDEX_REGISTRY'))
repo = '$REPO_NAME'
specified = '$KERNELS'
start = int('$START_INDEX')
end_str = '$END_INDEX'

# 收集当前 repo 的所有算子
kernels = []
for entry, idx in registry.items():
    if entry.startswith(repo + '/'):
        name = entry.split('/', 1)[1]
        kernels.append((idx, name))

# 按索引排序
kernels.sort(key=lambda x: x[0])

# 如果指定了 --kernels，过滤并检查缺失
if specified:
    specified_set = set(k.strip() for k in specified.split(','))
    filtered = [(idx, name) for idx, name in kernels if name in specified_set]
    missing = specified_set - set(name for _, name in filtered)
    for m in missing:
        print(f'WARNING:MISSING:{m}', file=sys.stderr)
    kernels = filtered

# 应用 START_INDEX/END_INDEX（repo-local 索引）
end = int(end_str) if end_str else len(kernels)
if start < 0:
    print(f'ERROR: START_INDEX {start} 不能为负数', file=sys.stderr)
    sys.exit(1)
if end > len(kernels):
    end = len(kernels)
if start >= end:
    print(f'ERROR: START_INDEX {start} >= END_INDEX {end}，无可执行算子', file=sys.stderr)
    sys.exit(1)

kernels = [(idx, name) for idx, name in kernels if start <= idx < end]

# 输出格式: index|name
for idx, name in kernels:
    print(f'{idx}|{name}')
" 2>&1)

# 解析 OP_LIST_RAW 到数组
OP_NAMES=()
OP_INDICES=()
while IFS='|' read -r idx name; do
    [[ -z "$idx" ]] && continue
    if [[ "$idx" == WARNING:* ]] || [[ "$idx" == ERROR:* ]]; then
        continue
    fi
    if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        continue
    fi
    OP_INDICES+=("$idx")
    OP_NAMES+=("$name")
done <<< "$OP_LIST_RAW"

if [[ ${#OP_NAMES[@]} -eq 0 ]]; then
    echo "错误：未找到任何算子（请检查 --kernels、--start-index、--end-index 参数）"
    exit 1
fi

TOTAL_OP_COUNT=$(python3 -c "
import json
registry = json.load(open('$INDEX_REGISTRY'))
repo = '$REPO_NAME'
count = sum(1 for entry in registry if entry.startswith(repo + '/'))
print(count)
")

if [[ -z "$END_INDEX" ]]; then
    END_INDEX=$TOTAL_OP_COUNT
fi

echo "========================================"
echo "Repo: ${REPO_NAME}"
echo "该 Repo 总算子数量：${TOTAL_OP_COUNT}"
echo "执行索引区间：[${START_INDEX}, ${END_INDEX})"
echo "本次实际执行：${#OP_NAMES[@]} 个算子"
if [[ "$NO_SKIP" == true ]]; then
    echo "模式：--no-skip（强制重新执行所有算子）"
fi
echo "========================================"

echo "找到 ${#OP_NAMES[@]} 个待执行算子："
for i in "${!OP_NAMES[@]}"; do
    echo "  [${OP_INDICES[$i]}] ${OP_NAMES[$i]}"
done

# ── 创建输出目录 ──
mkdir -p "$OUTPUT_DIR"

# ── 进度文件 ──
PROGRESS_FILE="${OUTPUT_DIR}/.batch_progress.json"
touch "$PROGRESS_FILE"

# ── 文件锁 ──
LOCK_FILE="${OUTPUT_DIR}/.lock"
touch "$LOCK_FILE"

# ── 结果记录 ──
REPORT_FILE="${OUTPUT_DIR}/batch_report.md"
echo "# 批量执行报告" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- benchmark: ${BENCHMARK_DIR}" >> "$REPORT_FILE"
echo "- repo: ${REPO_NAME}" >> "$REPORT_FILE"
echo "- 该 Repo 总算子数: ${TOTAL_OP_COUNT}" >> "$REPORT_FILE"
echo "- 执行索引区间: [${START_INDEX}, ${END_INDEX})" >> "$REPORT_FILE"
if [[ "$NO_SKIP" == true ]]; then
    echo "- 模式: --no-skip（强制重新执行）" >> "$REPORT_FILE"
fi
if [[ "$USE_PARALLEL" == true ]]; then
    echo "- npu-list: ${NPU_LIST}" >> "$REPORT_FILE"
    echo "- 执行模式：多 NPU 并行" >> "$REPORT_FILE"
else
    echo "- npu: ${NPU_ID}" >> "$REPORT_FILE"
    echo "- 执行模式：单 NPU 串行" >> "$REPORT_FILE"
fi
echo "- 开始时间：$(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 索引 | 算子名称 | 状态 | 耗时 (s) | 备注 |" >> "$REPORT_FILE"
echo "|------|----------|------|----------|------|" >> "$REPORT_FILE"

# ── 辅助函数：追加报告行（带文件锁）──
append_report() {
    local line="$1"
    {
        flock -x 200
        echo "$line" >> "$REPORT_FILE"
    } 200>"$LOCK_FILE"
}

# ── 辅助函数：更新进度文件（带文件锁）──
update_progress() {
    local repo=$1
    local name=$2
    local idx=$3
    local status=$4
    local output_dir=${5:-""}

    {
        flock -x 200
        python3 -c "
import json, os

repo = '$repo'
name = '$name'
idx = $idx
status = '$status'
output_dir = '$output_dir'
progress_file = '$PROGRESS_FILE'

data = {}
if os.path.exists(progress_file) and os.path.getsize(progress_file) > 0:
    try:
        with open(progress_file) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {}

if repo not in data:
    data[repo] = {}

entry = {
    'status': status,
    'index': idx,
    'timestamp': '$(date -Iseconds)'
}
if output_dir:
    entry['output_dir'] = output_dir

data[repo][name] = entry

with open(progress_file, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
" 2>/dev/null
    } 200>"$LOCK_FILE"
}

# ── 辅助函数：查找算子对应的最新 output_dir ──
find_op_output_dir() {
    local name=$1
    python3 -c "
import os, glob, sys

name = '$name'
agent_root = '$AGENT_ROOT'
pattern = os.path.join(agent_root, 'triton_ascend_output', f'op_*{name}_*')
matches = glob.glob(pattern)

if not matches:
    sys.exit(1)

# 按目录修改时间排序，取最新的
matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(matches[0])
" 2>/dev/null
}

# ── 辅助函数：检查算子是否已成功完成 ──
is_op_completed() {
    local repo=$1
    local name=$2
    local idx=$3

    if [[ "$NO_SKIP" == true ]]; then
        return 1
    fi

    python3 -c "
import json, os, sys, glob

repo = '$repo'
name = '$name'
idx = $idx
agent_root = '$AGENT_ROOT'
progress_file = '$PROGRESS_FILE'

try:
    with open(progress_file) as f:
        data = json.load(f)
    repo_data = data.get(repo, {})
    op_data = repo_data.get(name, {})

    # 状态必须是 success
    if op_data.get('status') != 'success':
        sys.exit(1)

    # 如果有记录的 output_dir，优先检查
    recorded_dir = op_data.get('output_dir', '')
    if recorded_dir and os.path.exists(recorded_dir):
        summary_path = os.path.join(recorded_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as sf:
                summary = json.load(sf)
            if summary.get('success', False):
                sys.exit(0)

    # fallback：扫描 triton_ascend_output 下匹配目录
    pattern = os.path.join(agent_root, 'triton_ascend_output', f'op_*{name}_*')
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        summary_path = os.path.join(matches[0], 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as sf:
                summary = json.load(sf)
            if summary.get('success', False):
                sys.exit(0)
except Exception:
    pass
sys.exit(1)
" 2>/dev/null
}

# ── 辅助函数：执行后记录 output_dir ──
record_op_output_dir() {
    local repo=$1
    local name=$2
    local idx=$3
    local status=$4

    local output_dir
    output_dir=$(find_op_output_dir "$name")
    if [[ -n "$output_dir" ]]; then
        update_progress "$repo" "$name" "$idx" "$status" "$output_dir"
    else
        update_progress "$repo" "$name" "$idx" "$status"
    fi
}

TOTAL=${#OP_NAMES[@]}
SUCCESS=0
FAIL=0
SKIP=0

# ── 执行模式选择 ──
if [[ "$USE_PARALLEL" == true ]]; then
    # ========== 多 NPU 并行模式 ==========
    echo ""
    echo "================================================================"
    echo "多 NPU 并行模式：${NPU_COUNT} 个 NPU，本次 ${TOTAL} 个算子"
    echo "NPU 列表：${NPU_LIST}"
    echo "================================================================"
    echo ""

    # 任务分配：轮询分配算子到各 NPU 队列
    declare -A npu_tasks
    declare -A npu_indices
    npu_index=0
    for i in "${!OP_NAMES[@]}"; do
        name="${OP_NAMES[$i]}"
        idx="${OP_INDICES[$i]}"
        npu=${NPU_ARRAY[$((npu_index % NPU_COUNT))]}
        npu_tasks[$npu]+="${name}"
        npu_indices[$npu]+="${idx}"
        npu_tasks[$npu]+=$'\x01'
        npu_indices[$npu]+=$'\x01'
        npu_index=$((npu_index + 1))
    done

    # 为每个 NPU 启动 worker 进程
    for npu in "${NPU_ARRAY[@]}"; do
        if [[ -n "${npu_tasks[$npu]:-}" ]]; then
            (
                IFS=$'\x01' read -ra TASK_NAMES <<< "${npu_tasks[$npu]}"
                IFS=$'\x01' read -ra TASK_INDICES <<< "${npu_indices[$npu]}"

                for i in "${!TASK_NAMES[@]}"; do
                    name="${TASK_NAMES[$i]}"
                    idx="${TASK_INDICES[$i]}"

                    [[ -z "$name" ]] && continue

                    kernel_dir="${REPO_DIR}/${name}"
                    file="${kernel_dir}/${name}.py"

                    # 断点续跑检查
                    if is_op_completed "$REPO_NAME" "$name" "$idx"; then
                        echo "[NPU $npu] ⏭️  算子 [${idx}] ${name}: 已存在成功记录，跳过"
                        append_report "| ${idx} | ${name} | ⏭️ 跳过 | - | 断点续跑 |"
                        continue
                    fi

                    START_TIME=$(date +%s)

                    PT_FILE="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${name}/${name}.pt"
                    TEST_FILE="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${name}/original_test.py"
                    GPU_PERF_FILE="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${REPO_NAME}_gpu_perf.csv"

                    PROMPT="""
                    生成 triton-ascend 算子:
                    1. 算子描述在${file}, op_index=${idx}
                    2. arch 是 ascend910_9382
                    3. 设置环境变量ASCEND_RT_VISIBLE_DEVICES=${npu}
                    4. 包含输入输出的 pt 文件在${PT_FILE}
                    5. pt文件数据到kernel输入以及预期输出的处理逻辑在${TEST_FILE}
                    6. 性能基线文件在${GPU_PERF_FILE}
                    """
                    
                    if claude -p "$PROMPT" \
                        --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
                        >> "${OUTPUT_DIR}/npu_${npu}.log" 2>&1; then

                        END_TIME=$(date +%s)
                        ELAPSED=$((END_TIME - START_TIME))

                        echo "[NPU $npu] ✅ 算子 [${idx}] ${name}: 完成 (${ELAPSED}s)"
                        append_report "| ${idx} | ${name} | ✅ 成功 | ${ELAPSED} | - |"
                        record_op_output_dir "$REPO_NAME" "$name" "$idx" "success"
                    else
                        END_TIME=$(date +%s)
                        ELAPSED=$((END_TIME - START_TIME))

                        echo "[NPU $npu] ❌ 算子 [${idx}] ${name}: 失败 (${ELAPSED}s)"
                        append_report "| ${idx} | ${name} | ❌ 失败 | ${ELAPSED} | - |"
                        record_op_output_dir "$REPO_NAME" "$name" "$idx" "failed"
                    fi
                done
            ) &
        fi
    done

    wait

else
    # ========== 单 NPU 串行模式 ==========
    echo ""
    echo "================================================================"
    echo "单 NPU 串行模式：NPU ${NPU_ID}，本次 ${TOTAL} 个算子"
    echo "================================================================"
    echo ""

    CURRENT=0
    for i in "${!OP_NAMES[@]}"; do
        name="${OP_NAMES[$i]}"
        idx="${OP_INDICES[$i]}"

        kernel_dir="${REPO_DIR}/${name}"
        file="${kernel_dir}/${name}.py"

        CURRENT=$((CURRENT + 1))

        echo ""
        echo "================================================================"
        echo "[${CURRENT}/${TOTAL}] 算子 [${idx}] ${name}"
        echo "================================================================"

        # 断点续跑检查
        if is_op_completed "$REPO_NAME" "$name" "$idx"; then
            echo "⏭️  算子 [${idx}] ${name}: 已存在成功记录，跳过"
            echo "| ${idx} | ${name} | ⏭️ 跳过 | - | 断点续跑 |" >> "$REPORT_FILE"
            SKIP=$((SKIP + 1))
            continue
        fi

        START_TIME=$(date +%s)

        PT_FILE="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${name}/${name}.pt"
        GPU_PERF_FILE="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${REPO_NAME}_gpu_perf.csv"

        PROMPT="""
            生成 triton-ascend 算子:
            1. 算子描述在${file}, op_index=${idx}
            2. arch 是 ascend910_9382
            3. 设置环境变量ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
            4. 包含输入输出的 pt 文件在${PT_FILE}
            5. pt文件数据到kernel输入以及预期输出的处理逻辑在${TEST_FILE}
            6. 性能基线文件在${GPU_PERF_FILE}
            """

        if claude -p "$PROMPT" \
            --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)'; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "| ${idx} | ${name} | ✅ 成功 | ${ELAPSED} | - |" >> "$REPORT_FILE"
            record_op_output_dir "$REPO_NAME" "$name" "$idx" "success"
            SUCCESS=$((SUCCESS + 1))
            echo "[${CURRENT}/${TOTAL}] ✅ 算子 [${idx}] ${name} 完成 (${ELAPSED}s)"
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "| ${idx} | ${name} | ❌ 失败 | ${ELAPSED} | - |" >> "$REPORT_FILE"
            record_op_output_dir "$REPO_NAME" "$name" "$idx" "failed"
            FAIL=$((FAIL + 1))
            echo "[${CURRENT}/${TOTAL}] ❌ 算子 [${idx}] ${name} 失败 (${ELAPSED}s)"
        fi
    done
fi

# ── 写入汇总 ──
echo "" >> "$REPORT_FILE"
echo "## 汇总" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

SUCCESS=$(grep -c "✅ 成功" "$REPORT_FILE" 2>/dev/null || echo 0)
FAIL=$(grep -c "❌ 失败" "$REPORT_FILE" 2>/dev/null || echo 0)
SKIP=$(grep -c "⏭️ 跳过" "$REPORT_FILE" 2>/dev/null || echo 0)

echo "- 本次计划执行：${TOTAL}" >> "$REPORT_FILE"
echo "- 成功：${SUCCESS}" >> "$REPORT_FILE"
echo "- 失败：${FAIL}" >> "$REPORT_FILE"
echo "- 跳过（已存在成功记录）：${SKIP}" >> "$REPORT_FILE"
echo "- 结束时间：$(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"

if [[ "$USE_PARALLEL" == true ]]; then
    echo "- 执行模式：多 NPU 并行" >> "$REPORT_FILE"
    echo "- NPU 日志：npu_0.log, npu_1.log, ... (在输出目录中)" >> "$REPORT_FILE"
fi

echo ""
echo "================================================================"
echo "批量执行完成：计划 ${TOTAL} 个算子"
echo "  - 成功：${SUCCESS}"
echo "  - 失败：${FAIL}"
echo "  - 跳过：${SKIP}"
echo "报告：${REPORT_FILE}"
echo "进度：${PROGRESS_FILE}"
echo "索引注册表：${INDEX_REGISTRY}"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "NPU 日志目录：${OUTPUT_DIR}/"
fi
echo "================================================================"
