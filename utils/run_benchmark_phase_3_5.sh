#!/bin/bash
# 批量续跑 Phase 3-5：扫描 triton_ascend_output/ 下已完成 Phase 0-2 的工作目录

set -euo pipefail

# ── 默认值 ──
TRITON_OUTPUT_DIR=""
EXTRACTED_KERNELS_DIR=""
REPO_NAME=""
OP_NAMES=""
RANGE=""
IDS=""
NPU_ID=0
NPU_LIST=""
OUTPUT_DIR=""
NO_SKIP=false

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --triton-output-dir)      TRITON_OUTPUT_DIR="$2"; shift 2 ;;
        --extracted-kernels-dir)  EXTRACTED_KERNELS_DIR="$2"; shift 2 ;;
        --repo-name)              REPO_NAME="$2"; shift 2 ;;
        --op-names)               OP_NAMES="$2"; shift 2 ;;
        --range)                  RANGE="$2"; shift 2 ;;
        --ids)                    IDS="$2"; shift 2 ;;
        --npu)                    NPU_ID="$2"; shift 2 ;;
        --npu-list)               NPU_LIST="$2"; shift 2 ;;
        --output)                 OUTPUT_DIR="$2"; shift 2 ;;
        --no-skip)                NO_SKIP=true; shift ;;
        -h|--help)
            cat << 'EOF'
用法: bash utils/run_benchmark_phase_3_5.sh --triton-output-dir <path> --extracted-kernels-dir <path> --repo-name <name> [选项]

参数:
  --triton-output-dir      triton_ascend_output 目录路径 (必填)
  --extracted-kernels-dir  提取的 kernels 目录 (必填)
  --repo-name              仓库名称，如 vllm / sglang / fbgemm (必填)
  --range                  算子范围，如 1-30 (与 --ids 二选一)
  --ids                    指定算子编号列表，逗号分隔，如 3,7,15 (与 --range 二选一)
  --op-names               额外按算子名称过滤，逗号分隔
  --npu                    单 NPU 设备 ID (默认 0)
  --npu-list               多 NPU 列表，逗号分隔
  --output                 输出报告目录 (默认: <triton-output-dir>/../phase3_5_reports)
  --no-skip                即使已有 summary.json 也重新执行

示例:
  bash utils/run_benchmark_phase_3_5.sh \
    --triton-output-dir ./triton_ascend_output \
    --extracted-kernels-dir ./extracted_kernels \
    --repo-name vllm \
    --range 1-30 \
    --npu 0
EOF
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ -z "$TRITON_OUTPUT_DIR" || -z "$EXTRACTED_KERNELS_DIR" || -z "$REPO_NAME" ]]; then
    echo "错误: 必须指定 --triton-output-dir、--extracted-kernels-dir 和 --repo-name"
    exit 1
fi

if [[ -z "$RANGE" && -z "$IDS" && -z "$OP_NAMES" ]]; then
    echo "错误: 必须指定 --range、--ids 或 --op-names 之一"
    exit 1
fi

TRITON_OUTPUT_DIR=$(cd "$TRITON_OUTPUT_DIR" && pwd)
EXTRACTED_KERNELS_DIR=$(cd "$EXTRACTED_KERNELS_DIR" && pwd)

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(dirname "$TRITON_OUTPUT_DIR")/phase3_5_reports"
fi
mkdir -p "$OUTPUT_DIR"

# ── 确定执行模式 ──
USE_PARALLEL=false
if [[ -n "$NPU_LIST" ]]; then
    USE_PARALLEL=true
    IFS=',' read -ra NPU_ARRAY <<< "$NPU_LIST"
    NPU_COUNT=${#NPU_ARRAY[@]}
else
    NPU_ARRAY=("$NPU_ID")
    NPU_COUNT=1
fi

# ── 扫描工作目录 ──
echo "[INFO] 扫描 ${TRITON_OUTPUT_DIR}/ 下的工作目录..."

NO_SKIP_PY="False"
[[ "$NO_SKIP" == true ]] && NO_SKIP_PY="True"

SCAN_RESULT=$(python3 -c "
import os, re, json, sys

base = '${TRITON_OUTPUT_DIR}'
op_filter = '${OP_NAMES}'
range_str = '${RANGE}'
ids_str = '${IDS}'
no_skip = ${NO_SKIP_PY}

filters = [f.strip().lower() for f in op_filter.split(',') if f.strip()] if op_filter else []

# 解析 range
range_start = range_end = None
if range_str:
    try:
        range_start, range_end = map(int, range_str.split('-'))
    except ValueError:
        print('ERROR: --range 格式错误，应为 start-end', file=sys.stderr)
        sys.exit(1)

# 解析 ids
allowed_ids = set()
if ids_str:
    try:
        allowed_ids = set(int(x.strip()) for x in ids_str.split(',') if x.strip())
    except ValueError:
        print('ERROR: --ids 格式错误，应为逗号分隔的整数', file=sys.stderr)
        sys.exit(1)

results = []
for entry in os.listdir(base):
    path = os.path.join(base, entry)
    if not os.path.isdir(path):
        continue
    if not os.path.exists(os.path.join(path, 'sketch.txt')):
        continue

    # 从目录名解析 op_index 和 op_name
    # 格式: op_{op_index}_{op_name_parts}_{YYYYMMDD}_{HHMM}_{rid}
    parts = entry.split('_')
    if len(parts) < 5 or not parts[0].startswith('op'):
        continue

    try:
        op_index = int(parts[1])
    except ValueError:
        continue

    # op_name = 从 index 之后到日期(8位数字)之前的所有部分
    op_name_parts = []
    for p in parts[2:]:
        if len(p) == 8 and p.isdigit():
            break
        if p:
            op_name_parts.append(p)
    op_name = '_'.join(op_name_parts) if op_name_parts else parts[2]

    # range 过滤
    if range_str is not None and range_str != '':
        if not (range_start <= op_index <= range_end):
            continue

    # ids 过滤
    if allowed_ids:
        if op_index not in allowed_ids:
            continue

    # 名称过滤
    if filters and not any(f in op_name.lower() for f in filters):
        continue

    # summary.json 检查
    has_summary = os.path.exists(os.path.join(path, 'summary.json'))
    if has_summary and not no_skip:
        try:
            with open(os.path.join(path, 'summary.json')) as f:
                if json.load(f).get('success', False):
                    continue
        except:
            pass

    results.append((op_index, op_name, path, has_summary))

# 按 op_index 排序
results.sort(key=lambda x: x[0])

for op_index, op_name, path, has_summary in results:
    print(f'{op_index}|{op_name}|{path}|{str(has_summary).lower()}')
" 2>&1)

# 解析结果
OP_INDEXES=()
OP_NAMES_LIST=()
OP_PATHS=()
OP_HAS_SUMMARY=()

while IFS='|' read -r op_index op_name dir_path has_summary; do
    [[ -z "$op_index" ]] && continue
    if [[ "$op_index" == ERROR:* ]]; then
        echo "$op_index"; exit 1
    fi
    OP_INDEXES+=("$op_index")
    OP_NAMES_LIST+=("$op_name")
    OP_PATHS+=("$dir_path")
    OP_HAS_SUMMARY+=("$has_summary")
done <<< "$SCAN_RESULT"

TOTAL=${#OP_NAMES_LIST[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "[INFO] 未发现待续跑的算子"
    exit 0
fi

echo "========================================"
echo "发现 ${TOTAL} 个待续跑算子:"
for i in "${!OP_NAMES_LIST[@]}"; do
    status="待续跑"
    [[ "${OP_HAS_SUMMARY[$i]}" == "true" ]] && status="有summary（失败）"
    echo "  [${OP_INDEXES[$i]}] ${OP_NAMES_LIST[$i]} -> ${OP_PATHS[$i]} (${status})"
done
echo "========================================"

# ── 创建报告 ──
REPORT_FILE="${OUTPUT_DIR}/batch_report_p35.md"
{
    echo "# Phase 3-5 批量续跑报告"
    echo "- repo-name: ${REPO_NAME}"
    echo "- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "| 算子ID | 算子名称 | 状态 | 耗时(s) |"
    echo "|--------|----------|------|---------|"
} > "$REPORT_FILE"

LOCK_FILE="${OUTPUT_DIR}/.lock"
touch "$LOCK_FILE"

append_report() {
    {
        flock -x 200
        echo "$1" >> "$REPORT_FILE"
    } 200>"$LOCK_FILE"
}


run_single() {
    local idx="$1"
    local npu="$2"

    local op_index="${OP_INDEXES[$idx]}"
    local name="${OP_NAMES_LIST[$idx]}"
    local work_dir="${OP_PATHS[$idx]}"

    # 文件路径直接按固定规则拼接
    local desc_file="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${name}/${name}.py"
    local pt_file="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${name}/${name}.pt"
    local test_file="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${name}/original_test.py"
    local gpu_perf_file="${EXTRACTED_KERNELS_DIR}/${REPO_NAME}/${REPO_NAME}_gpu_perf.csv"

    # 动态计算当前项目的 Claude 思维轨迹目录
    local CWD=$(pwd)
    local CLAUDE_PROJECT_NAME=$(echo "$CWD" | sed 's|/|-|g')
    local CLAUDE_PROJECT_DIR="$HOME/.claude/projects/$CLAUDE_PROJECT_NAME"

    local SESSION_ID=$(uuidgen)
    local START_TIME=$(date +%s)

    local PROMPT="""
你是 triton-ascend-coder, 请生成 triton-ascend 算子:
1. 算子描述在 ${desc_file}
2. arch 是 ascend910_9382
3. op_index 是 ${op_index}, op_name 是 ${name}
4. 包含输入输出的 pt 文件在 ${pt_file}
5. pt文件数据到kernel输入以及预期输出的处理逻辑在 ${test_file}
6. 性能基线文件在 ${gpu_perf_file}
当前已完成 Phase 0-2，工作目录在 ${work_dir}，请继续生成 Phase 3-5。
"""

    echo "[NPU ${npu}] 开始续跑 ${name}..."

    if ASCEND_RT_VISIBLE_DEVICES=${npu} claude -p "$PROMPT" \
        --session-id "$SESSION_ID" \
        --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
        >> "${OUTPUT_DIR}/phase_3_5_npu_${npu}.log" 2>&1; then

        local END_TIME=$(date +%s)
        local ELAPSED=$((END_TIME - START_TIME))
        echo "[NPU ${npu}] 算子 ${name}: 完成 (${ELAPSED}s)"
        append_report "| ${op_index} | ${name} | 成功 | ${ELAPSED} |"
        local STATUS="success"
    else
        local END_TIME=$(date +%s)
        local ELAPSED=$((END_TIME - START_TIME))
        echo "[NPU ${npu}] 算子 ${name}: 失败 (${ELAPSED}s)"
        append_report "| ${op_index} | ${name} | 失败 | ${ELAPSED} |"
        local STATUS="fail"
    fi

    # 重命名思维轨迹文件（UUID 精确匹配）
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local JSONL_FILE="$CLAUDE_PROJECT_DIR/${SESSION_ID}.jsonl"
    if [[ -f "$JSONL_FILE" ]]; then
        mv "$JSONL_FILE" "${CLAUDE_PROJECT_DIR}/${op_index}_${name}_phase_3_5_${STATUS}_${TIMESTAMP}.jsonl"
        if [[ -d "$CLAUDE_PROJECT_DIR/${SESSION_ID}" ]]; then
            mv "$CLAUDE_PROJECT_DIR/${SESSION_ID}" "${CLAUDE_PROJECT_DIR}/${op_index}_${name}_phase_3_5_${STATUS}_${TIMESTAMP}"
        fi
    fi
}

# ── 执行 ──
if [[ "$USE_PARALLEL" == true ]]; then
    echo "多 NPU 并行: ${NPU_COUNT} 个 NPU，${TOTAL} 个算子"

    declare -A npu_tasks
    for i in "${!OP_NAMES_LIST[@]}"; do
        npu=${NPU_ARRAY[$((i % NPU_COUNT))]}
        npu_tasks[$npu]+="${i} "
    done

    for npu in "${NPU_ARRAY[@]}"; do
        [[ -n "${npu_tasks[$npu]:-}" ]] || continue
        (
            for idx in ${npu_tasks[$npu]}; do
                run_single "$idx" "$npu"
            done
        ) &
    done
    wait
else
    echo "单 NPU 串行: NPU ${NPU_ID}，${TOTAL} 个算子"
    for i in "${!OP_NAMES_LIST[@]}"; do
        echo ""
        echo "[$(($i + 1))/${TOTAL}] ${OP_NAMES_LIST[$i]}"
        run_single "$i" "$NPU_ID"
    done
fi

# ── 汇总 ──
SUCCESS=$(grep -c "| 成功 |" "$REPORT_FILE" 2>/dev/null || echo 0)
FAIL=$(grep -c "| 失败 |" "$REPORT_FILE" 2>/dev/null || echo 0)

{
    echo ""
    echo "## 汇总"
    echo "- 计划: ${TOTAL}"
    echo "- 成功: ${SUCCESS}"
    echo "- 失败: ${FAIL}"
    echo "- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$REPORT_FILE"

echo ""
echo "Phase 3-5 完成: 成功 ${SUCCESS}/${TOTAL}, 失败 ${FAIL}/${TOTAL}"
echo "报告: ${REPORT_FILE}"
