#!/bin/bash
# 从 chopper0126 仓库同步 AscendC 侧内容到合并仓库
# 用法: ./scripts/sync-from-chopper.sh /path/to/chopper0126/AscendOpGenAgent
#
# 功能:
#   1. 复制 agent (lingxi_code.md → lingxi-code.md)
#   2. 复制 skills (下划线目录名 → 连字符目录名)
#   3. 更新 SKILL.md 中的 name 字段为连字符格式

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "用法: $0 /path/to/chopper0126/AscendOpGenAgent"
  exit 1
fi

SRC="$1"
DST="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$SRC/.opencode/agents" ]; then
  echo "错误: 找不到 $SRC/.opencode/agents 目录"
  exit 1
fi

echo "源仓库: $SRC"
echo "目标仓库: $DST"
echo ""

# Agent: 重命名 + 复制
echo "同步 agent..."
cp "$SRC/.opencode/agents/lingxi_code.md" "$DST/agents/lingxi-code.md"
# 更新 skills 引用为连字符格式
sed -i '' \
  -e 's/op_desc_generation/op-desc-generation/g' \
  -e 's/reference_generation/reference-generation/g' \
  -e 's/functional_conversion/functional-conversion/g' \
  -e 's/ascend_call_generation/ascend-call-generation/g' \
  -e 's/dsl_baseline_generation/dsl-baseline-generation/g' \
  -e 's/dsl_lowering/dsl-lowering/g' \
  -e 's/dsl_optimization/dsl-optimization/g' \
  -e 's/ascendc_evalution/ascendc-evaluation/g' \
  "$DST/agents/lingxi-code.md"

# Skills: 目录名下划线→连字符 + 复制到 ascendc 子目录
declare -A SKILL_MAP=(
  [op_desc_generation]=op-desc-generation
  [reference_generation]=reference-generation
  [functional_conversion]=functional-conversion
  [ascend_call_generation]=ascend-call-generation
  [dsl_baseline_generation]=dsl-baseline-generation
  [dsl_lowering]=dsl-lowering
  [dsl_optimization]=dsl-optimization
  [ascendc_evalution]=ascendc-evaluation
)

echo "同步 skills..."
mkdir -p "$DST/skills/ascendc"

for src_name in "${!SKILL_MAP[@]}"; do
  dst_name=${SKILL_MAP[$src_name]}
  echo "  $src_name → ascendc/$dst_name"
  rsync -av --delete "$SRC/.opencode/skills/$src_name/" "$DST/skills/ascendc/$dst_name/"
done

# ascend-benchmark-evaluator 名字不变
echo "  ascend-benchmark-evaluator → ascendc/ascend-benchmark-evaluator"
rsync -av --delete "$SRC/.opencode/skills/ascend-benchmark-evaluator/" "$DST/skills/ascendc/ascend-benchmark-evaluator/"

# 更新 SKILL.md 中的 name 字段（下划线→连字符）
echo ""
echo "更新 SKILL.md name 字段..."
for dst_name in "${SKILL_MAP[@]}"; do
  skill_file="$DST/skills/ascendc/$dst_name/SKILL.md"
  if [ -f "$skill_file" ]; then
    # 将 name 字段中的下划线替换为连字符
    sed -i '' "s/^name: .*$/name: $dst_name/" "$skill_file"
  fi
done

echo ""
echo "同步完成。请检查以下内容："
echo "  1. agents/lingxi-code.md 的 skills 引用"
echo "  2. skills/ascendc/ 下各 SKILL.md 的 name 字段"
echo "  3. 是否有 .opencode/ 绝对路径引用需要更新"
