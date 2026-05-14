#!/usr/bin/env bash
# Build concat_dv2 AscendC kernel and install as .so
# Usage: bash build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Source CANN environment
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
fi

# Activate conda if available
if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${CONDA_DEFAULT_ENV}" 2>/dev/null || true
fi

# Clean build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"
cmake "${SCRIPT_DIR}" \
    -DSOC_VERSION="${SOC_VERSION:-Ascend910B2}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DCMAKE_BUILD_TYPE=Debug

make -j$(nproc)

echo "Build complete: ${BUILD_DIR}"
echo "Output library:"
find "${BUILD_DIR}" -name "*.so" -type f 2>/dev/null || echo "  (check build/ for output)"
