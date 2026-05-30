#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# One-click environment dependency installer for CANNBot development.
#
# Checks / installs:
#   1. Python 3.8+          — check, install via deadsnakes PPA if missing
#   2. Ascend CANN 8.0+     — check only (no install)
#   3. PyTorch 2.0+         — pip install
#   4. torch_npu 2.9.0      — pip install
#   5. tilelang-ascend      — git clone + bash install_ascend.sh + source set_env.sh
#   6. Node.js + Tool       — download Node.js, install claude-code or opencode (arg: $1)
# ----------------------------------------------------------------------------------------------------------

set -e

# --- Color & output helpers ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "\n${BOLD}${CYAN}[$1]${NC} ${BOLD}$2${NC}"; }

# --- Configurable paths ---
TILELANG_INSTALL_DIR="${TILELANG_INSTALL_DIR:-$HOME/tilelang-ascend}"
NODE_INSTALL_DIR="${NODE_INSTALL_DIR:-$HOME/node-v24.14.0-linux-x64}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
TOOL="${1:-}"

# --- Usage ---
usage() {
  echo "Usage: $0 <claude-code|opencode>"
  echo ""
  echo "  One-click environment dependency installer for CANNBot development."
  echo ""
  echo "  Steps:"
  echo "    1. Python 3.8+          — check, install via deadsnakes PPA if missing"
  echo "    2. Ascend CANN 8.0+     — check only (no install)"
  echo "    3. PyTorch 2.0+         — pip install"
  echo "    4. torch_npu 2.9.0      — pip install"
  echo "    5. tilelang-ascend      — git clone + compile, skips if already installed"
  echo "    6. Node.js + Tool       — download Node.js + install specified tool"
  echo ""
  echo "  Arguments:"
  echo "    claude-code   Install @anthropic-ai/claude-code@2.1.153"
  echo "    opencode      Install opencode-ai"
  echo ""
  echo "  Options:"
  echo "    -h, --help    Show this help message"
  echo ""
  echo "  Environment variables:"
  echo "    TILELANG_INSTALL_DIR   tilelang source dir (default: \$HOME/tilelang-ascend)"
  echo "    NODE_INSTALL_DIR       Node.js install dir (default: \$HOME/node-v24.14.0-linux-x64)"
  echo "    PYTHON_CMD             Python command (default: python3)"
  echo "    FORCE_TILELANG_REINSTALL=1   Force recompile tilelang even if installed"
  echo ""
  echo "  Examples:"
  echo "    $0 claude-code"
  echo "    $0 opencode"
  echo "    FORCE_TILELANG_REINSTALL=1 $0 claude-code"
  echo "    TILELANG_INSTALL_DIR=/opt/tilelang $0 opencode"
  exit 0
}

case "$TOOL" in
  -h|--help) usage ;;
esac

# --- Banner ---
echo ""
echo -e "${CYAN}${BOLD}  CANNBot — Environment Dependency Installer${NC}"
echo -e "${DIM}  -------------------------------------------${NC}"
echo ""

# ====================================================================
# 1. Python 3.8+
# ====================================================================
step "1/6" "Checking Python (>= 3.8)..."

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+' || echo "0.0")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

need_install_python=false
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
  need_install_python=true
fi

if $need_install_python; then
  warn "Python $PYTHON_VERSION found, but Python >= 3.8 is required."
  info "Attempting to install Python 3.10 via deadsnakes PPA..."

  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
    PYTHON_CMD="python3.10"
    ok "Python 3.10 installed"
  elif command -v yum &>/dev/null; then
    sudo yum install -y python38 python38-devel
    PYTHON_CMD="python3.8"
    ok "Python 3.8 installed"
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y python3.10 python3.10-devel
    PYTHON_CMD="python3.10"
    ok "Python 3.10 installed"
  else
    err "Cannot detect package manager. Please install Python >= 3.8 manually."
    exit 1
  fi
else
  ok "Python $PYTHON_VERSION found"
fi

# ====================================================================
# 2. Ascend CANN 8.0+  (check only)
# ====================================================================
step "2/6" "Checking Ascend CANN (>= 8.0)..."

CANN_FOUND=false
CANN_VERSION=""
CANN_PATH=""

# Extract CANN version from a directory that contains compiler/version.info or opp/version.info.
# The version file uses format: Version=X.Y.Z
_detect_cann_version() {
  local dir="$1"
  local ver_file=""
  if [ -f "$dir/compiler/version.info" ]; then
    ver_file="$dir/compiler/version.info"
  elif [ -f "$dir/opp/version.info" ]; then
    ver_file="$dir/opp/version.info"
  fi
  if [ -n "$ver_file" ]; then
    grep -oP 'Version=\K[0-9.]+' "$ver_file" 2>/dev/null || echo ""
  fi
}

# Priority 1: env vars
for env_var in ASCEND_HOME ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH; do
  candidate="${!env_var}"
  if [ -n "$candidate" ] && [ -d "$candidate" ]; then
    # ASCEND_OPP_PATH points to .../cann-8.5.1/opp, go up one level
    if [[ "$candidate" == */opp ]]; then
      candidate="$(dirname "$candidate")"
    fi
    ver=$(_detect_cann_version "$candidate")
    if [ -n "$ver" ]; then
      CANN_VERSION="$ver"
      CANN_PATH="$candidate"
      CANN_FOUND=true
      break
    fi
  fi
done

# Priority 2: /usr/local/Ascend/cann-X.Y.Z/  pattern
if ! $CANN_FOUND; then
  for dir in /usr/local/Ascend/cann-*/; do
    [ -d "$dir" ] || continue
    ver=$(_detect_cann_version "$dir")
    if [ -n "$ver" ]; then
      CANN_VERSION="$ver"
      CANN_PATH="$dir"
      CANN_FOUND=true
      break
    fi
  done
fi

# Priority 3: /usr/local/Ascend/ascend-toolkit/
if ! $CANN_FOUND && [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
  for dir in /usr/local/Ascend/ascend-toolkit/*/; do
    [ -d "$dir" ] || continue
    ver=$(_detect_cann_version "$dir")
    if [ -n "$ver" ]; then
      CANN_VERSION="$ver"
      CANN_PATH="$dir"
      CANN_FOUND=true
      break
    fi
  done
fi

# Priority 4: /usr/local/Ascend/ (bare layout with compiler/version.info)
if ! $CANN_FOUND && [ -f "/usr/local/Ascend/compiler/version.info" ]; then
  ver=$(_detect_cann_version "/usr/local/Ascend")
  if [ -n "$ver" ]; then
    CANN_VERSION="$ver"
    CANN_PATH="/usr/local/Ascend"
    CANN_FOUND=true
  fi
fi

if $CANN_FOUND; then
  CANN_MAJOR=$(echo "$CANN_VERSION" | cut -d. -f1)
  if [ "$CANN_MAJOR" -ge 8 ]; then
    ok "Ascend CANN $CANN_VERSION found at $CANN_PATH"
  else
    warn "Ascend CANN $CANN_VERSION found at $CANN_PATH, but < 8.0 — please upgrade"
    CANN_FOUND=false
  fi
fi

if ! $CANN_FOUND; then
  err "Ascend CANN >= 8.0 NOT found!"
  err "Please install Ascend CANN 8.0+ manually before continuing."
  err "Download: https://www.hiascend.com/software/cann"
  exit 1
fi

# ====================================================================
# 3. PyTorch 2.0+
# ====================================================================
step "3/6" "Installing PyTorch (>= 2.0)..."

TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "0.0")
TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1)

if [ "$TORCH_MAJOR" -ge 2 ]; then
  ok "PyTorch $TORCH_VERSION already installed"
else
  info "Installing PyTorch 2.0+..."
  $PYTHON_CMD -m pip install --upgrade pip -q
  $PYTHON_CMD -m pip install torch>=2.0 -q
  TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "0.0")
  ok "PyTorch $TORCH_VERSION installed"
fi

# ====================================================================
# 4. torch_npu 2.9.0
# ====================================================================
step "4/6" "Installing torch_npu 2.9.0..."

TORCH_NPU_VERSION=$($PYTHON_CMD -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null || echo "0.0")

if [ "$TORCH_NPU_VERSION" = "2.9.0" ]; then
  ok "torch_npu $TORCH_NPU_VERSION already installed"
else
  if [ -n "$TORCH_NPU_VERSION" ] && [ "$TORCH_NPU_VERSION" != "0.0" ]; then
    info "torch_npu $TORCH_NPU_VERSION found, upgrading to 2.9.0..."
  fi
  $PYTHON_CMD -m pip install torch-npu==2.9.0 -q
  ok "torch_npu 2.9.0 installed"
fi

# ====================================================================
# 5. tilelang-ascend  (source install)
# ====================================================================
step "5/6" "Installing tilelang-ascend (from source)..."

TILELANG_ALREADY_INSTALLED=false
if [ "${FORCE_TILELANG_REINSTALL:-}" != "1" ] && [ -d "$TILELANG_INSTALL_DIR" ] && [ -f "$TILELANG_INSTALL_DIR/set_env.sh" ]; then
  # Source set_env.sh first so Python can find tilelang (set_env.sh adds it to PYTHONPATH)
  if bash -c "source $TILELANG_INSTALL_DIR/set_env.sh && $PYTHON_CMD -c 'import tilelang'" 2>/dev/null; then
    TILELANG_VERSION=$(bash -c "source $TILELANG_INSTALL_DIR/set_env.sh && $PYTHON_CMD -c \"import tilelang; print(getattr(tilelang, '__version__', 'unknown'))\"" 2>/dev/null || echo "unknown")
    ok "tilelang-ascend $TILELANG_VERSION already installed, skipping"
    TILELANG_ALREADY_INSTALLED=true
  fi
fi

if ! $TILELANG_ALREADY_INSTALLED; then
  if [ -d "$TILELANG_INSTALL_DIR" ]; then
    info "Directory $TILELANG_INSTALL_DIR already exists, updating..."
    cd "$TILELANG_INSTALL_DIR"
    git pull --recurse-submodules -q || true
  else
    info "Cloning tilelang-ascend (recursive)..."
    git clone --recursive https://github.com/tile-ai/tilelang-ascend.git "$TILELANG_INSTALL_DIR"
    cd "$TILELANG_INSTALL_DIR"
  fi

  info "Compiling and installing tilelang-ascend..."
  bash install_ascend.sh
  ok "tilelang-ascend installed"

  info "Setting up environment variables..."
  # shellcheck disable=SC1091
  source set_env.sh
  ok "Environment variables configured"
fi

info "Writing source set_env.sh to ~/.bashrc..."
SET_ENV_LINE="source $TILELANG_INSTALL_DIR/set_env.sh"
if grep -Fxq "$SET_ENV_LINE" "$HOME/.bashrc" 2>/dev/null; then
  ok "Already present in ~/.bashrc"
else
  echo "" >> "$HOME/.bashrc"
  echo "# tilelang-ascend environment" >> "$HOME/.bashrc"
  echo "$SET_ENV_LINE" >> "$HOME/.bashrc"
  ok "Appended to ~/.bashrc"
fi

# ====================================================================
# 6. Node.js + Claude Code / Opencode
# ====================================================================
step "6/6" "Installing Node.js and tool..."

if [ -z "$TOOL" ]; then
  warn "No tool specified. Usage: $0 <claude-code|opencode>"
  warn "Skipping Node.js and tool installation."
  TOOL_SKIPPED=true
else
  case "$TOOL" in
    claude-code|opencode)
      ;;
    *)
      err "Invalid tool: $TOOL. Must be 'claude-code' or 'opencode'."
      exit 1
      ;;
  esac

  TOOL_SKIPPED=false

  # Download and install Node.js
  if [ ! -d "$NODE_INSTALL_DIR" ]; then
    info "Downloading Node.js v24.14.0..."
    wget --no-check-certificate https://mirrors.huaweicloud.com/nodejs/v24.14.0/node-v24.14.0-linux-x64.tar.xz -O /tmp/node-v24.14.0-linux-x64.tar.xz
    info "Extracting Node.js..."
    tar -xJf /tmp/node-v24.14.0-linux-x64.tar.xz -C "$HOME"
    rm -f /tmp/node-v24.14.0-linux-x64.tar.xz
    ok "Node.js extracted to $NODE_INSTALL_DIR"
  else
    ok "Node.js already exists at $NODE_INSTALL_DIR"
  fi

  # Configure environment
  export NODE_TLS_REJECT_UNAUTHORIZED=0
  export PATH="$NODE_INSTALL_DIR/bin:$PATH"
  ok "Node.js added to PATH"

  # Configure npm
  npm config set strict-ssl false
  npm config set registry https://registry.npmmirror.com
  npm cache clean -f
  ok "npm configured"

  # Install tool (skip if already installed)
  TOOL_ALREADY_INSTALLED=false
  if [ "$TOOL" = "claude-code" ]; then
    if TOOL_VERSION=$(claude -v 2>&1); then
      ok "Claude Code already installed: $TOOL_VERSION"
      TOOL_ALREADY_INSTALLED=true
    fi
  else
    if TOOL_VERSION=$(opencode -v 2>&1); then
      ok "Opencode already installed: $TOOL_VERSION"
      TOOL_ALREADY_INSTALLED=true
    fi
  fi

  if ! $TOOL_ALREADY_INSTALLED; then
    if [ "$TOOL" = "claude-code" ]; then
      info "Installing Claude Code..."
      npm install -g @anthropic-ai/claude-code@2.1.153 --verbose
      ok "Claude Code installed"
      TOOL_VERSION=$(claude -v 2>&1 || echo "unknown")
    else
      info "Installing Opencode..."
      npm i -g opencode-ai
      ok "Opencode installed"
      TOOL_VERSION=$(opencode -v 2>&1 || echo "unknown")
    fi
    ok "$TOOL version: $TOOL_VERSION"
  fi

  # Write Node.js env to ~/.bashrc
  NODE_PATH_LINE="export PATH=\"$NODE_INSTALL_DIR/bin:\$PATH\""
  if grep -Fxq "$NODE_PATH_LINE" "$HOME/.bashrc" 2>/dev/null; then
    ok "Node.js PATH already in ~/.bashrc"
  else
    echo "" >> "$HOME/.bashrc"
    echo "# Node.js environment" >> "$HOME/.bashrc"
    echo "$NODE_PATH_LINE" >> "$HOME/.bashrc"
    ok "Node.js PATH appended to ~/.bashrc"
  fi
fi

# ====================================================================
# Summary
# ====================================================================
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ All dependencies installed successfully!${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Summary:${NC}"
echo -e "  ${DIM}Python:       ${NC}$($PYTHON_CMD --version 2>&1)"
echo -e "  ${DIM}CANN:        ${NC}CANN $CANN_VERSION"
echo -e "  ${DIM}PyTorch:      ${NC}$TORCH_VERSION"
echo -e "  ${DIM}torch_npu:    ${NC}$($PYTHON_CMD -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)"
echo -e "  ${DIM}tilelang:     ${NC}$TILELANG_INSTALL_DIR"
echo -e "  ${DIM}Node.js:      ${NC}$NODE_INSTALL_DIR"
if [ "$TOOL_SKIPPED" = false ]; then
  echo -e "  ${DIM}$TOOL:        ${NC}$TOOL_VERSION"
fi
echo ""
echo -e "  ${YELLOW}Note:${NC} Env vars written to ~/.bashrc. Run the following to activate in current shell:"
echo -e "  ${BOLD}        source ~/.bashrc${NC}"
if [ "$TOOL_SKIPPED" = false ]; then
  echo -e "  ${DIM}        Then:${NC} ${BOLD}$TOOL -v${NC}"
fi
echo ""
