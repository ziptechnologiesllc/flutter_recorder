#!/bin/bash
# Build LiteRT for macOS using official CMake presets
# Reference: https://ai.google.dev/edge/litert/build/cmake_litert
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PLUGIN_DIR/prebuilt/macos"
LITERT_DIR="$PLUGIN_DIR/third_party/LiteRT/litert"

JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
[[ "$1" == "--jobs" || "$1" == "-j" ]] && JOBS="$2"

echo "=== LiteRT macOS Build (CMake) ==="
echo "  Jobs: $JOBS"

cd "$LITERT_DIR"

# Use official CMake preset
cmake --preset default \
    -DCMAKE_SHARED_LINKER_FLAGS_RELEASE="-Wl,-dead_strip"
cmake --build cmake_build -j"$JOBS"

# Copy output
mkdir -p "$OUTPUT_DIR"
cp -v cmake_build/c/libLiteRtRuntimeCApi.dylib "$OUTPUT_DIR/libLiteRt.dylib"

# Fix install name for runtime linking
install_name_tool -id "@rpath/libLiteRt.dylib" "$OUTPUT_DIR/libLiteRt.dylib"
codesign --force --sign - "$OUTPUT_DIR/libLiteRt.dylib"

echo "=== Done ==="
ls -lh "$OUTPUT_DIR/"
