#!/bin/bash
# build_standalone_kernel.sh — Build libkernel_standalone.so from libkernels.a
#
# This creates a standalone shared library that only depends on ACL libraries,
# not on libtorch_npu.so. Used by cv_profiling_runner.py for msprof simulator.
#
# Usage: bash build_standalone_kernel.sh /path/to/kernel/
#
# Prerequisites:
#   - kernel/build/lib/libkernels.a must exist (from build_ascendc.py)
#   - ASCEND_HOME_PATH must be set

set -e

KERNEL_DIR="$1"
if [ -z "$KERNEL_DIR" ]; then
    echo "Usage: $0 <kernel_dir>"
    exit 1
fi

KERNEL_DIR=$(realpath "$KERNEL_DIR")
BUILD_DIR="${KERNEL_DIR}/build"
LIB_DIR="${BUILD_DIR}/lib"
STATIC_LIB="${LIB_DIR}/libkernels.a"
OUTPUT_SO="${BUILD_DIR}/libkernel_standalone.so"

if [ ! -f "$STATIC_LIB" ]; then
    echo "ERROR: libkernels.a not found at $STATIC_LIB"
    exit 1
fi

ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
if [ ! -d "$ASCEND_HOME" ]; then
    # Try common paths
    for p in /home/CANN/cann-8.5.0 /usr/local/Ascend/cann-8.5.0 /usr/local/Ascend/ascend-toolkit/latest; do
        if [ -d "$p" ]; then
            ASCEND_HOME="$p"
            break
        fi
    done
fi

ACL_LIB_DIR="${ASCEND_HOME}/lib64"
if [ ! -d "$ACL_LIB_DIR" ]; then
    echo "ERROR: ACL lib dir not found at $ACL_LIB_DIR"
    exit 1
fi

echo "[build_standalone] Static lib: $STATIC_LIB"
echo "[build_standalone] ACL lib dir: $ACL_LIB_DIR"
echo "[build_standalone] Output: $OUTPUT_SO"

# Extract extern "C" function names from pybind11.cpp
PYBIND_CPP="${KERNEL_DIR}/pybind11.cpp"
if [ -f "$PYBIND_CPP" ]; then
    EXTERN_FNS=$(grep -oP 'extern\s+"C"\s+void\s+\K\w+' "$PYBIND_CPP" || true)
    echo "[build_standalone] Found extern C functions: $EXTERN_FNS"
fi

# Build standalone .so using --whole-archive to export all symbols from .a
g++ -shared \
    -Wl,--whole-archive "$STATIC_LIB" -Wl,--no-whole-archive \
    -L"$ACL_LIB_DIR" \
    -lascendcl \
    -lruntime \
    -o "$OUTPUT_SO"

if [ $? -eq 0 ] && [ -f "$OUTPUT_SO" ]; then
    echo "[build_standalone] Success: $OUTPUT_SO ($(du -h "$OUTPUT_SO" | cut -f1))"
    # Verify the extern C functions are exported
    if [ -n "$EXTERN_FNS" ]; then
        for fn in $EXTERN_FNS; do
            if nm -D "$OUTPUT_SO" 2>/dev/null | grep -q "$fn"; then
                echo "[build_standalone]   [OK] $fn exported"
            else
                echo "[build_standalone]   [FAIL] $fn NOT found in exports"
            fi
        done
    fi
else
    echo "ERROR: Failed to build $OUTPUT_SO"
    exit 1
fi
