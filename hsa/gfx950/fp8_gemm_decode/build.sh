#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Build script for FP8 GEMM Decode ASM kernels
# Run on a Linux system with ROCm installed
#
# Usage:
#   ./build.sh           # Build for both gfx950 and gfx942
#   ./build.sh gfx950    # Build only for MI355X
#   ./build.sh gfx942    # Build only for MI300X

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find clang from ROCm
if [ -d "/opt/rocm/llvm/bin" ]; then
    CLANG="/opt/rocm/llvm/bin/clang"
elif command -v clang &> /dev/null; then
    CLANG="clang"
else
    echo "Error: clang not found. Please install ROCm or add clang to PATH."
    exit 1
fi

echo "Using clang: $CLANG"
$CLANG --version | head -n 1

# Function to build for a specific architecture
build_kernel() {
    local ARCH=$1
    local SOURCE="fp8_gemm_decode_128x128.s"
    local OUTPUT="fp8_gemm_decode_128x128_${ARCH}.co"
    
    echo "Building $SOURCE for $ARCH..."
    
    $CLANG \
        -target amdgcn-amd-amdhsa \
        -mcpu=$ARCH \
        -x assembler \
        -c "$SOURCE" \
        -o "$OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "Successfully built: $OUTPUT"
        ls -la "$OUTPUT"
    else
        echo "Failed to build for $ARCH"
        return 1
    fi
}

# Parse arguments
TARGET=${1:-all}

case $TARGET in
    gfx950)
        build_kernel gfx950
        ;;
    gfx942)
        build_kernel gfx942
        ;;
    all)
        echo "Building for all supported architectures..."
        build_kernel gfx950
        build_kernel gfx942
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [gfx950|gfx942|all]"
        exit 1
        ;;
esac

echo ""
echo "Build complete. Output files:"
ls -la *.co 2>/dev/null || echo "No .co files generated"

