#!/bin/bash
# Build LiteRT for Linux x86_64 using official CMake presets
# Reference: https://ai.google.dev/edge/litert/build/cmake_litert
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PLUGIN_DIR/prebuilt/linux"
LITERT_DIR="$PLUGIN_DIR/third_party/LiteRT/litert"

JOBS=$(nproc 2>/dev/null || echo 4)
[[ "$1" == "--jobs" || "$1" == "-j" ]] && JOBS="$2"

echo "=== LiteRT Linux Build (CMake) ==="
echo "  Jobs: $JOBS"

cd "$LITERT_DIR"

# Use official CMake preset
cmake --preset default
cmake --build cmake_build -j"$JOBS"

# Copy output
mkdir -p "$OUTPUT_DIR"
cp -v cmake_build/c/libLiteRtRuntimeCApi.so "$OUTPUT_DIR/libLiteRt.so"
strip --strip-unneeded "$OUTPUT_DIR/libLiteRt.so" 2>/dev/null || true

echo "=== Done ==="
ls -lh "$OUTPUT_DIR/"
