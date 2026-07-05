#!/bin/bash

# Build flutter_recorder as a static library using CMake.
# Invoked by CocoaPods script_phase during Xcode builds.
# Mirrors flutter_soloud v4's approach.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Please install cmake (e.g., 'brew install cmake')"
    exit 1
fi

echo "  Using cmake: $(which cmake)"

if [ -z "$ARCHS" ]; then
    ARCHS=$(uname -m)
    echo "ARCHS not set, defaulting to: ${ARCHS}"
fi

if [ -z "$SDKROOT" ]; then
    SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
    echo "SDKROOT not set, defaulting to: ${SDKROOT}"
fi

CMAKE_ARCHS=$(echo "$ARCHS" | tr ' ' ';')
BUILD_DIR="${SCRIPT_DIR}/cmake_build/macosx"

echo "=== flutter_recorder: CMake build for macOS ==="
echo "  ARCHS: ${ARCHS}"
echo "  BUILD_DIR: ${BUILD_DIR}"

cmake -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="${CMAKE_ARCHS}" \
    -DCMAKE_OSX_SYSROOT="${SDKROOT}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="14.0" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build "${BUILD_DIR}" -j$(sysctl -n hw.ncpu)

echo "=== flutter_recorder: CMake build complete ==="
echo "  Library: ${BUILD_DIR}/libflutter_recorder_plugin.a"
