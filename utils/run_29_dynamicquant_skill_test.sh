#!/bin/bash
# 29_DynamicQuant 单算子验证 —— 用于检查 skills/ascendc/precision-tuning(/discovery)
# 是否在 Phase 4 精度迭代陷入停滞时被实际触发。
#
# 与 utils/run_benchmark_ascendc.sh 的差异:
# - 锁死 --level 1 --ids 29
# - 自动屏蔽专项辅助 KB(ascendc_dynamic_quant_kb.md / dequant_kernel_patterns.md),
#   防止 ascendc-translator 一次过、precision-tuning 永远没机会被翻开
# - 同时屏蔽源树和 .claude/skills/ 副本,因为 agent 实际读 .claude/skills/
# - 退出/中断时 trap 恢复 KB 原文件,本地工程零残留
#
# 用法:
#   bash utils/run_29_dynamicquant_skill_test.sh [--npu N] [--output DIR]
# 默认:
#   --npu 0
#   --output runs/29_DynamicQuant_$(date +%Y%m%d_%H%M)/

set -euo pipefail

NPU_ID=0
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --npu)    NPU_ID="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash utils/run_29_dynamicquant_skill_test.sh [--npu N] [--output DIR]"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/runs/29_DynamicQuant_$(date +%Y%m%d_%H%M)"
fi

# ── 验证前置 ──
[[ -f .claude/CLAUDE.md ]] || { echo "ERR: .claude/CLAUDE.md 不存在,先运行 README 3.2 装载步骤"; exit 1; }
[[ -d .claude/skills/precision-tuning ]] || { echo "ERR: .claude/skills/precision-tuning 缺失"; exit 1; }
[[ -d .claude/skills/precision-tuning-discovery ]] || { echo "ERR: .claude/skills/precision-tuning-discovery 缺失"; exit 1; }
[[ -f benchmarks/NPUKernelBench/level1/29_DynamicQuant.py ]] || { echo "ERR: 29_DynamicQuant.py 不存在"; exit 1; }

# ── 需要屏蔽的专项辅助 KB(源树 + .claude 副本各两份) ──
KB_FILES=(
    "skills/ascendc/ascendc-translator/references/ascendc_dynamic_quant_kb.md"
    "skills/ascendc/ascendc-translator/references/dequant_kernel_patterns.md"
    ".claude/skills/ascendc-translator/references/ascendc_dynamic_quant_kb.md"
    ".claude/skills/ascendc-translator/references/dequant_kernel_patterns.md"
)

SHIELDED=()

restore_kbs() {
    local f
    for f in "${SHIELDED[@]}"; do
        if [[ -f "${f}.shielded" ]]; then
            mv "${f}.shielded" "$f"
            echo "[RESTORE] $f"
        fi
    done
}
trap restore_kbs EXIT INT TERM

shield_kbs() {
    local f
    for f in "${KB_FILES[@]}"; do
        if [[ -f "$f" ]]; then
            mv "$f" "${f}.shielded"
            SHIELDED+=("$f")
            echo "[SHIELD] $f -> ${f}.shielded"
        else
            echo "[SKIP] $f (不存在,已跳过)"
        fi
    done
}

echo "================================================================"
echo "29_DynamicQuant skill-trigger validation"
echo "  NPU         : $NPU_ID"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Repo root   : $REPO_ROOT"
echo "================================================================"

shield_kbs

mkdir -p "$OUTPUT_DIR"

bash utils/run_benchmark_ascendc.sh \
    --benchmark-dir "$REPO_ROOT/benchmarks/NPUKernelBench" \
    --level 1 \
    --ids 29 \
    --npu "$NPU_ID" \
    --output "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "完成。验收要点(满足任一即可证明 skill 被触发):"
echo "  1) trace.md 引用 precision-tuning(/discovery)"
echo "  2) Phase 4 trace 出现 forensics / Gate-F/A/V / precision_audit / precision_knowledge_base 字样"
echo "  3) $OUTPUT_DIR/29_DynamicQuant/ 下有 precision_* 系列产物"
echo "================================================================"
echo "建议立刻 grep 验证:"
echo "  grep -rE 'precision-tuning|forensics|Gate-[FAV]|precision_audit|precision_knowledge' $OUTPUT_DIR/29_DynamicQuant/trace.md $OUTPUT_DIR/29_DynamicQuant/precision_tuning/ 2>/dev/null"
