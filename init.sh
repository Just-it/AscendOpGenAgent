#!/bin/bash
# -----------------------------------------------------------------------------
# AscendOpGenAgent - 开发环境初始化脚本
# 通过软链接将 skills/hooks/agents 接入 .claude/ 目录，避免手动复制。
# -----------------------------------------------------------------------------
set -e

# --- 颜色 ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="$SCRIPT_DIR/skills/ascendc"
AGENT_SRC="$SCRIPT_DIR/agents"
HOOK_SRC="$SCRIPT_DIR/hooks"
CLAUDE_DIR="$SCRIPT_DIR/.claude"

echo ""
echo -e "  ${BOLD}AscendOpGenAgent${NC}"
echo ""

# --- Step 1: Skills 软链接 ---
step "[1/4] Setting up skills symlinks..."
mkdir -p "$CLAUDE_DIR/skills"

for d in "$SKILL_SRC"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  target="$CLAUDE_DIR/skills/$name"
  # 删除已有实体目录
  if [ -d "$target" ] && [ ! -L "$target" ]; then
    rm -rf "$target"
  fi
  # 删除已有链接（重新指向）
  [ -L "$target" ] && rm -f "$target"
  ln -sfn "$(realpath "$d")" "$target"
  ok "skills/$name"
done

# --- Step 2: CLAUDE.md 软链接 ---
step "[2/4] Setting up CLAUDE.md symlink..."
AGENT_MD="$AGENT_SRC/ascend-kernel-developer.md"
CLAUDE_MD="$CLAUDE_DIR/CLAUDE.md"

if [ -f "$AGENT_MD" ]; then
  if [ -d "$CLAUDE_MD" ] || { [ -f "$CLAUDE_MD" ] && [ ! -L "$CLAUDE_MD" ]; }; then
    rm -rf "$CLAUDE_MD"
  fi
  [ -L "$CLAUDE_MD" ] && rm -f "$CLAUDE_MD"
  ln -sfn "$(realpath "$AGENT_MD")" "$CLAUDE_MD"
  ok "CLAUDE.md -> agents/ascend-kernel-developer.md"
else
  warn "agents/ascend-kernel-developer.md not found, skipping"
fi

# --- Step 3: Hooks 软链接 ---
step "[3/4] Setting up hooks symlinks..."
mkdir -p "$CLAUDE_DIR/hooks"

for f in "$HOOK_SRC"/*; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  target="$CLAUDE_DIR/hooks/$name"
  # 删除已有实体文件
  if [ -f "$target" ] && [ ! -L "$target" ]; then
    rm -f "$target"
  fi
  [ -L "$target" ] && rm -f "$target"
  ln -sfn "$(realpath "$f")" "$target"
  ok "hooks/$name"
done

# --- Step 4: 验证 ---
step "[4/4] Verifying..."
health_ok=true

# 检查 skills
skill_count=$(ls -d "$CLAUDE_DIR/skills"/*/ 2>/dev/null | wc -l)
if [ "$skill_count" -gt 0 ]; then
  ok "Skills: $skill_count symlinks"
else
  err "Skills: empty"
  health_ok=false
fi

# 检查 CLAUDE.md
if [ -L "$CLAUDE_MD" ] && [ -e "$CLAUDE_MD" ]; then
  ok "CLAUDE.md: linked"
else
  err "CLAUDE.md: broken or missing"
  health_ok=false
fi

# 检查 hooks
hook_count=$(ls "$CLAUDE_DIR/hooks"/ 2>/dev/null | wc -l)
if [ "$hook_count" -gt 0 ]; then
  ok "Hooks: $hook_count symlinks"
else
  err "Hooks: empty"
  health_ok=false
fi

echo ""
if [ "$health_ok" = true ]; then
  echo -e "  ${GREEN}${BOLD}✓ All done!${NC}"
else
  echo -e "  ${RED}${BOLD}✗ Some checks failed, see above${NC}"
fi
echo ""
