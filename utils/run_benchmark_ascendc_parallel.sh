#!/bin/bash
# 批量调度 ascendc-coder，支持多 NPU 并行 + debug 修复
#
# 修复点（相比原脚本）：
# 1. 每个 worker 显式 cd 到 PROJECT_ROOT，确保 .claude/CLAUDE.md 被加载
# 2. Prompt 明确 agent 身份和工作流，不依赖"当前agent"这种模糊表述
# 3. 失败后自动触发 ascendc-debugger 重试
# 4. 每个算子独立状态文件，避免子进程间竞争
# 5. 增加 --timeout 防止 debug 过程被提前中断
#
# 用法与原脚本完全一致

set -euo pipefail

# ── 项目根目录（关键！确保 .claude/CLAUDE.md 可被加载）──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 环境变量 ──
export ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-Ascend910B3}"

# ── 默认值 ──
BENCHMARK_DIR=""
LEVEL=""
RANGE=""
IDS=""
NPU_ID=0
NPU_LIST=""
OUTPUT_DIR=""
TIMEOUT=7200  # 单个算子超时（秒），含 debugger 弹性重试

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
        --level)         LEVEL="$2"; shift 2 ;;
        --range)         RANGE="$2"; shift 2 ;;
        --ids)           IDS="$2"; shift 2 ;;
        --npu)           NPU_ID="$2"; shift 2 ;;
        --npu-list)      NPU_LIST="$2"; shift 2 ;;
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash utils/run_benchmark_ascendc_parallel.sh --benchmark-dir <path> --level <N> [--range <start-end> | --ids <id_list>] [--npu <id> | --npu-list <list>] --output <path> [--timeout <seconds>]"
            echo ""
            echo "参数:"
            echo "  --benchmark-dir  KernelBench 根目录路径 (必填)"
            echo "  --level          Level 编号，如 1, 2, 3 (必填)"
            echo "  --range          算子范围，如 41-53 (与 --ids 二选一)"
            echo "  --ids            指定算子编号列表，逗号分隔，如 3,7,15"
            echo "  --npu            单 NPU 设备 ID，如 0 (默认 0)"
            echo "  --npu-list       多 NPU 列表，逗号分隔，如 0,1,2,3,4,5"
            echo "  --output         输出目录 (必填)"
            echo "  --timeout        单个算子超时时间，默认 3600 秒"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ -z "$BENCHMARK_DIR" || -z "$LEVEL" || (-z "$RANGE" && -z "$IDS") || -z "$OUTPUT_DIR" ]]; then
    echo "错误: 缺少必填参数，使用 --help 查看用法"
    exit 1
fi

LEVEL_DIR="${BENCHMARK_DIR}/level${LEVEL}"
if [[ ! -d "$LEVEL_DIR" ]]; then
    echo "错误: 目录不存在: ${LEVEL_DIR}"
    exit 1
fi

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

# ── 构建算子 ID 列表 ──
OP_IDS=()
if [[ -n "$RANGE" ]]; then
    START=$(echo "$RANGE" | cut -d'-' -f1)
    END=$(echo "$RANGE" | cut -d'-' -f2)
    for i in $(seq "$START" "$END"); do
        OP_IDS+=("$i")
    done
elif [[ -n "$IDS" ]]; then
    IFS=',' read -ra OP_IDS <<< "$IDS"
fi

# ── 扫描算子文件 ──
declare -A OP_FILES
for id in "${OP_IDS[@]}"; do
    matched=$(find "$LEVEL_DIR" -maxdepth 1 -name "${id}_*.py" -type f 2>/dev/null | head -1)
    if [[ -n "$matched" ]]; then
        OP_FILES[$id]="$matched"
    else
        echo "警告: 未找到算子 ${id} 的文件，跳过"
    fi
done

if [[ ${#OP_FILES[@]} -eq 0 ]]; then
    echo "错误: 未找到任何算子文件"
    exit 1
fi

# ── 创建输出目录和状态目录 ──
mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/.status"
touch "${OUTPUT_DIR}/.lock"

# ── 结果记录 ──
REPORT_FILE="${OUTPUT_DIR}/batch_report.md"
cat > "$REPORT_FILE" << 'EOF'
# 批量执行报告

EOF
echo "- benchmark: ${BENCHMARK_DIR}" >> "$REPORT_FILE"
echo "- level: ${LEVEL}" >> "$REPORT_FILE"
echo "- project_root: ${PROJECT_ROOT}" >> "$REPORT_FILE"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "- npu-list: ${NPU_LIST}" >> "$REPORT_FILE"
    echo "- 执行模式: 多 NPU 并行" >> "$REPORT_FILE"
else
    echo "- npu: ${NPU_ID}" >> "$REPORT_FILE"
    echo "- 执行模式: 单 NPU 串行" >> "$REPORT_FILE"
fi
echo "- timeout: ${TIMEOUT}s" >> "$REPORT_FILE"
echo "- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 算子ID | 文件 | 状态 | 耗时(s) | 备注 |" >> "$REPORT_FILE"
echo "|--------|------|------|---------|------|" >> "$REPORT_FILE"

TOTAL=${#OP_FILES[@]}

# ================================================================
# 核心函数：执行单个算子（由 ascend-kernel-developer 内部循环处理 debug）
# ================================================================
run_single_op() {
    local npu="$1"
    local id="$2"
    local file="$3"
    local filename=$(basename "$file")
    local op_name="${filename%.*}"
    local TARGET_OP_DIR="${OUTPUT_DIR}/${op_name}"
    local STATUS_FILE="${OUTPUT_DIR}/.status/${op_name}.status"
    local LOG_FILE="${OUTPUT_DIR}/${op_name}.log"

    mkdir -p "$TARGET_OP_DIR"
    echo "pending" > "$STATUS_FILE"

    local START_TIME END_TIME ELAPSED
    START_TIME=$(date +%s)

    echo "[NPU ${npu}] 算子 ${id}: ${filename} 开始生成..."
    echo "generating" > "$STATUS_FILE"

    # Prompt 必须包含 workdir，否则 ascendc-debugger 在 Phase 4 找不到脚本路径
    local PROMPT
    PROMPT=$(cat << EOF
生成ascendC算子，npu=${npu}，算子描述文件为 ${file}，输出到 ${TARGET_OP_DIR}/，工作目录为 ${PROJECT_ROOT}
EOF
)

    if timeout "$TIMEOUT" bash -c "
        cd '$PROJECT_ROOT'
        export ASCEND_RT_VISIBLE_DEVICES=${npu}
        claude -p '$PROMPT' \
            --agent ascend-kernel-developer \
            --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
            > '$LOG_FILE' 2>&1
    "; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "success" > "$STATUS_FILE"
        {
            flock -x 200
            echo "| ${id} | ${filename} | 成功 | ${ELAPSED} | - |" >> "$REPORT_FILE"
        } 200>"${OUTPUT_DIR}/.lock"
        echo "[NPU ${npu}] 算子 ${id}: ${filename} 完成 (${ELAPSED}s)"
        return 0
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "failed" > "$STATUS_FILE"
        {
            flock -x 200
            echo "| ${id} | ${filename} | 失败 | ${ELAPSED} | timeout 或 agent 返回非零 |" >> "$REPORT_FILE"
        } 200>"${OUTPUT_DIR}/.lock"
        echo "[NPU ${npu}] 算子 ${id}: ${filename} 失败 (${ELAPSED}s)"
        return 1
    fi
}

# ================================================================
# 执行模式选择
# ================================================================
if [[ "$USE_PARALLEL" == true ]]; then
    echo ""
    echo "================================================================"
    echo "多 NPU 并行模式: ${NPU_COUNT} 个 NPU，${TOTAL} 个算子"
    echo "NPU 列表: ${NPU_LIST}"
    echo "PROJECT_ROOT: ${PROJECT_ROOT}"
    echo "================================================================"
    echo ""

    # 任务分配：轮询分配
    declare -A npu_tasks
    npu_index=0
    for id in "${OP_IDS[@]}"; do
        if [[ -v OP_FILES[$id] ]]; then
            npu=${NPU_ARRAY[$((npu_index % NPU_COUNT))]}
            npu_tasks[$npu]+="${id} "
            npu_index=$((npu_index + 1))
        fi
    done

    # 为每个 NPU 启动 worker 进程
    for npu in "${NPU_ARRAY[@]}"; do
        if [[ -n "${npu_tasks[$npu]:-}" ]]; then
            (
                cd "$PROJECT_ROOT"  # 关键：确保 .claude/CLAUDE.md 可被加载
                for id in ${npu_tasks[$npu]}; do
                    run_single_op "$npu" "$id" "${OP_FILES[$id]}"
                done
            ) &
        fi
    done

    wait

else
    echo ""
    echo "================================================================"
    echo "单 NPU 串行模式: NPU ${NPU_ID}，${TOTAL} 个算子"
    echo "PROJECT_ROOT: ${PROJECT_ROOT}"
    echo "================================================================"
    echo ""

    cd "$PROJECT_ROOT"  # 关键

    CURRENT=0
    for id in $(echo "${!OP_FILES[@]}" | tr ' ' '\n' | sort -n); do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[${CURRENT}/${TOTAL}] 算子 ${id}: ${OP_FILES[$id]}"
        run_single_op "$NPU_ID" "$id" "${OP_FILES[$id]}"
    done
fi

# ── 写入汇总 ──
echo "" >> "$REPORT_FILE"
echo "## 汇总" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

SUCCESS=$(grep -cE "成功|success" "${OUTPUT_DIR}/.status/"*.status 2>/dev/null || echo 0)
FAIL=$(grep -c "failed" "${OUTPUT_DIR}/.status/"*.status 2>/dev/null || echo 0)
DEBUG_SUCCESS=$(grep -c "success_after_debug" "${OUTPUT_DIR}/.status/"*.status 2>/dev/null || echo 0)

echo "- 总数: ${TOTAL}" >> "$REPORT_FILE"
echo "- 首次通过: ${SUCCESS}" >> "$REPORT_FILE"
echo "- Debug修复后通过: ${DEBUG_SUCCESS}" >> "$REPORT_FILE"
echo "- 失败: ${FAIL}" >> "$REPORT_FILE"
echo "- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"

if [[ "$USE_PARALLEL" == true ]]; then
    echo "- 执行模式: 多 NPU 并行" >> "$REPORT_FILE"
    echo "- NPU 日志: npu_*.log" >> "$REPORT_FILE"
fi

echo ""
echo "================================================================"
echo "批量执行完成: 成功 $((SUCCESS + DEBUG_SUCCESS))/${TOTAL}, 失败 ${FAIL}/${TOTAL}"
echo "  首次通过: ${SUCCESS}"
echo "  Debug修复: ${DEBUG_SUCCESS}"
echo "  失败: ${FAIL}"
echo "  基础报告: ${REPORT_FILE}"
echo "  状态目录: ${OUTPUT_DIR}/.status/"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "  NPU 日志: ${OUTPUT_DIR}/npu_*.log"
fi
echo "================================================================"
