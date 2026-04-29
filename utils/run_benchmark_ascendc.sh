#!/bin/bash
# 批量调度 ascendc-coder，支持多 NPU 并行
#
# 支持两种模式：
# 1. 单 NPU 模式（--npu）：串行执行，向后兼容
# 2. 多 NPU 并行模式（--npu-list）：NPU 间并行，NPU 内串行
#
# 用法:
#   # 单 NPU 模式
#   bash utils/run_benchmark_ascendc.sh --benchmark-dir /path/to/KernelBench --level 1 --range 41-53 --npu 0 --output /path/to/output
#
#   # 多 NPU 并行模式
#   bash utils/run_benchmark_ascendc.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu-list "0,1,2,3,4,5" --output /path/to/output

# ── 环境变量 ──
# export ANTHROPIC_AUTH_TOKEN=sk-  # 替换为您的实际令牌
# export ANTHROPIC_BASE_URL=https://yunwu.ai
# export API_TIMEOUT_MS=300000  # 设置为 300 秒超时

set -euo pipefail

# ── 默认值 ──
BENCHMARK_DIR=""
LEVEL=""
RANGE=""
IDS=""
NPU_ID=0
NPU_LIST=""
OUTPUT_DIR=""

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
        -h|--help)
            echo "用法: bash utils/run_benchmark_ascendc.sh --benchmark-dir <path> --level <N> [--range <start-end> | --ids <id_list>] [--npu <id> | --npu-list <list>] --output <path>"
            echo ""
            echo "参数:"
            echo "  --benchmark-dir  KernelBench 根目录路径 (必填)"
            echo "  --level          Level 编号，如 1, 2, 3 (必填)"
            echo "  --range          算子范围，如 41-53 (与 --ids 二选一)"
            echo "  --ids            指定算子编号列表，逗号分隔，如 3,7,15 (与 --range 二选一)"
            echo "  --npu            单 NPU 设备 ID，如 0 (默认 0，与 --npu-list 互斥)"
            echo "  --npu-list       多 NPU 列表，逗号分隔，如 0,1,2,3,4,5 (与 --npu 互斥，优先级更高)"
            echo "  --output         输出目录 (必填)"
            echo ""
            echo "示例:"
            echo "  # 单 NPU 串行模式"
            echo "  bash utils/run_benchmark_ascendc.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu 0 --output /path/to/output"
            echo ""
            echo "  # 多 NPU 并行模式"
            echo "  bash utils/run_benchmark_ascendc.sh --benchmark-dir /path/to/KernelBench --level 1 --range 1-30 --npu-list \"0,1,2,3,4,5\" --output /path/to/output"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 参数校验 ──
if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "错误: 必须指定 --benchmark-dir"
    exit 1
fi

if [[ -z "$LEVEL" ]]; then
    echo "错误: 必须指定 --level"
    exit 1
fi

if [[ -z "$RANGE" && -z "$IDS" ]]; then
    echo "错误: 必须指定 --range 或 --ids"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 必须指定 --output"
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
    # 解析 NPU 列表
    IFS=',' read -ra NPU_ARRAY <<< "$NPU_LIST"
    NPU_COUNT=${#NPU_ARRAY[@]}
    if [[ $NPU_COUNT -eq 0 ]]; then
        echo "错误: NPU 列表为空"
        exit 1
    fi
else
    # 单 NPU 模式
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
    # 匹配 {id}_{name}.py 格式
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

# ── 创建输出目录 ──
mkdir -p "$OUTPUT_DIR"

# ── 创建文件锁 ──
touch "${OUTPUT_DIR}/.lock"

# ── 结果记录 ──
REPORT_FILE="${OUTPUT_DIR}/batch_report.md"
echo "# 批量执行报告" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- benchmark: ${BENCHMARK_DIR}" >> "$REPORT_FILE"
echo "- level: ${LEVEL}" >> "$REPORT_FILE"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "- npu-list: ${NPU_LIST}" >> "$REPORT_FILE"
    echo "- 执行模式: 多 NPU 并行（NPU 间并行，NPU 内串行）" >> "$REPORT_FILE"
else
    echo "- npu: ${NPU_ID}" >> "$REPORT_FILE"
    echo "- 执行模式: 单 NPU 串行" >> "$REPORT_FILE"
fi
echo "- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| 算子ID | 文件 | 状态 | 耗时(s) |" >> "$REPORT_FILE"
echo "|--------|------|------|---------|" >> "$REPORT_FILE"

TOTAL=${#OP_FILES[@]}
SUCCESS=0
FAIL=0

# ── 执行模式选择 ──
if [[ "$USE_PARALLEL" == true ]]; then
    # ========== 多 NPU 并行模式 ==========
    echo ""
    echo "================================================================"
    echo "多 NPU 并行模式: ${NPU_COUNT} 个 NPU，${TOTAL} 个算子"
    echo "NPU 列表: ${NPU_LIST}"
    echo "================================================================"
    echo ""

    # 任务分配：轮询分配算子到各 NPU 队列
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
        # 检查该 NPU 是否有任务
        if [[ -n "${npu_tasks[$npu]:-}" ]]; then
            (
                # ========== Worker 进程开始 ==========
                for id in ${npu_tasks[$npu]}; do
                    file="${OP_FILES[$id]}"
                    filename=$(basename "$file")
                    op_name="${filename%.*}"
                    TARGET_OP_DIR="${OUTPUT_DIR}/${op_name}"

                    mkdir -p "$TARGET_OP_DIR"

                    START_TIME=$(date +%s)

                    # 动态计算当前项目的 Claude 思维轨迹目录
                    CWD=$(pwd)
                    CLAUDE_PROJECT_NAME=$(echo "$CWD" | sed 's|/|-|g')
                    CLAUDE_PROJECT_DIR="/root/.claude/projects/$CLAUDE_PROJECT_NAME"

                    PROMPT="使用当前agent生成ascendC算子，npu=${npu}，算子描述文件为 ${file}，输出到 ${TARGET_OP_DIR}/"

                    if claude -p "$PROMPT" \
                        --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)' \
                        >> "${OUTPUT_DIR}/npu_${npu}.log" 2>&1; then

                        END_TIME=$(date +%s)
                        ELAPSED=$((END_TIME - START_TIME))

                        # 立即输出到主终端
                        echo "[NPU $npu] ✅ 算子 ${id}: ${filename} 完成 (${ELAPSED}s)" >&2

                        # 加锁写入报告
                        {
                            flock -x 200
                            echo "| ${id} | ${filename} | ✅ 成功 | ${ELAPSED} |" >> "$REPORT_FILE"
                        } 200>"${OUTPUT_DIR}/.lock"

                        STATUS="success"
                    else
                        END_TIME=$(date +%s)
                        ELAPSED=$((END_TIME - START_TIME))

                        # 立即输出到主终端
                        echo "[NPU $npu] ❌ 算子 ${id}: ${filename} 失败 (${ELAPSED}s)" >&2

                        # 加锁写入报告
                        {
                            flock -x 200
                            echo "| ${id} | ${filename} | ❌ 失败 | ${ELAPSED} |" >> "$REPORT_FILE"
                        } 200>"${OUTPUT_DIR}/.lock"

                        STATUS="fail"
                    fi

                    # 串行重命名思维轨迹文件（带时间戳防止同名覆盖）
                    {
                        flock -x 201
                        LATEST_JSONL=""
                        if [[ -d "$CLAUDE_PROJECT_DIR" ]]; then
                            LATEST_JSONL=$(ls -t "$CLAUDE_PROJECT_DIR"/*.jsonl 2>/dev/null | head -1 || true)
                        fi
                        if [[ -n "$LATEST_JSONL" && -f "$LATEST_JSONL" ]]; then
                            BASENAME=$(basename "$LATEST_JSONL" .jsonl)
                            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
                            mv "$LATEST_JSONL" "${CLAUDE_PROJECT_DIR}/${op_name}_${STATUS}_${TIMESTAMP}.jsonl"
                            if [[ -d "${CLAUDE_PROJECT_DIR}/${BASENAME}" ]]; then
                                mv "${CLAUDE_PROJECT_DIR}/${BASENAME}" "${CLAUDE_PROJECT_DIR}/${op_name}_${STATUS}_${TIMESTAMP}"
                            fi
                        fi
                    } 201>"${OUTPUT_DIR}/.trace_lock"
                done
                # ========== Worker 进程结束 ==========
            ) &
        fi
    done

    # 等待所有 worker 完成
    wait

else
    # ========== 单 NPU 串行模式（原逻辑）==========
    echo ""
    echo "================================================================"
    echo "单 NPU 串行模式: NPU ${NPU_ID}，${TOTAL} 个算子"
    echo "================================================================"
    echo ""

    CURRENT=0
    for id in $(echo "${!OP_FILES[@]}" | tr ' ' '\n' | sort -n); do
        file="${OP_FILES[$id]}"
        filename=$(basename "$file")

        # 提取不带后缀的文件名，例如 "31_ELU.py" 变成 "31_ELU"
        op_name="${filename%.*}"

        # 构建当前算子的专属输出子目录
        TARGET_OP_DIR="${OUTPUT_DIR}/${op_name}"

        # 创建该子目录
        mkdir -p "$TARGET_OP_DIR"

        CURRENT=$((CURRENT + 1))

        echo ""
        echo "================================================================"
        echo "[${CURRENT}/${TOTAL}] 算子 ${id}: ${filename} (输出至: ${op_name}/)"
        echo "================================================================"

        START_TIME=$(date +%s)

        # 动态计算当前项目的 Claude 思维轨迹目录
        CWD=$(pwd)
        CLAUDE_PROJECT_NAME=$(echo "$CWD" | sed 's|/|-|g')
        CLAUDE_PROJECT_DIR="/root/.claude/projects/$CLAUDE_PROJECT_NAME"

        PROMPT="使用当前agent生成ascendC算子，npu=${NPU_ID}，算子描述文件为 ${file}，输出到 ${TARGET_OP_DIR}/"

        if claude -p "$PROMPT" \
            --allowedTools 'Bash(*)' 'Read(*)' 'Write(*)' 'Edit(*)' 'Glob(*)' 'Grep(*)' 'Skill(*)'; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "| ${id} | ${filename} | ✅ 成功 | ${ELAPSED} |" >> "$REPORT_FILE"
            SUCCESS=$((SUCCESS + 1))
            echo "[${CURRENT}/${TOTAL}] ✅ 算子 ${id} 完成 (${ELAPSED}s)"
            STATUS="success"
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "| ${id} | ${filename} | ❌ 失败 | ${ELAPSED} |" >> "$REPORT_FILE"
            FAIL=$((FAIL + 1))
            echo "[${CURRENT}/${TOTAL}] ❌ 算子 ${id} 失败 (${ELAPSED}s)"
            STATUS="fail"
        fi

        # 重命名思维轨迹文件（带时间戳防止同名覆盖）
        LATEST_JSONL=""
        if [[ -d "$CLAUDE_PROJECT_DIR" ]]; then
            LATEST_JSONL=$(ls -t "$CLAUDE_PROJECT_DIR"/*.jsonl 2>/dev/null | head -1 || true)
        fi
        if [[ -n "$LATEST_JSONL" && -f "$LATEST_JSONL" ]]; then
            BASENAME=$(basename "$LATEST_JSONL" .jsonl)
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            mv "$LATEST_JSONL" "${CLAUDE_PROJECT_DIR}/${op_name}_${STATUS}_${TIMESTAMP}.jsonl"
            if [[ -d "${CLAUDE_PROJECT_DIR}/${BASENAME}" ]]; then
                mv "${CLAUDE_PROJECT_DIR}/${BASENAME}" "${CLAUDE_PROJECT_DIR}/${op_name}_${STATUS}_${TIMESTAMP}"
            fi
        fi
    done
fi

# ── 写入汇总 ──
echo "" >> "$REPORT_FILE"
echo "## 汇总" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# 统计成功和失败数
SUCCESS=$(grep -c "✅ 成功" "$REPORT_FILE" 2>/dev/null || echo 0)
FAIL=$(grep -c "❌ 失败" "$REPORT_FILE" 2>/dev/null || echo 0)

echo "- 总数: ${TOTAL}" >> "$REPORT_FILE"
echo "- 成功: ${SUCCESS}" >> "$REPORT_FILE"
echo "- 失败: ${FAIL}" >> "$REPORT_FILE"
echo "- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"

if [[ "$USE_PARALLEL" == true ]]; then
    echo "- 执行模式: 多 NPU 并行" >> "$REPORT_FILE"
    echo "- NPU 日志: npu_0.log, npu_1.log, ... (在输出目录中)" >> "$REPORT_FILE"
fi


if [[ "$USE_PARALLEL" == true ]]; then
    echo "NPU 日志目录: ${OUTPUT_DIR}/"
fi
echo "================================================================"

# ==================================================================
# 🚀 新增：调用外部 Python 汇总脚本
# ==================================================================
echo ""
echo "================================================================"
echo "正在调用 generate_report_dynamic.py 生成详细报告..."
echo "================================================================"
# 请根据你的文件实际路径调整下面这一行
python3 utils/generate_report_dynamic.py -i "$OUTPUT_DIR" -o "$OUTPUT_DIR/final_batch_report.md"


# 最后的结束语
echo ""
echo "================================================================"
echo "批量执行完成: 成功 ${SUCCESS}/${TOTAL}, 失败 ${FAIL}/${TOTAL}"
echo "基础报告: ${REPORT_FILE}"
echo "详细报告: ${OUTPUT_DIR}/final_batch_report.md (如果生成成功)"
if [[ "$USE_PARALLEL" == true ]]; then
    echo "NPU 日志目录: ${OUTPUT_DIR}/"
fi
echo "================================================================"

