#!/bin/bash
# Build LiteRT for Android arm64-v8a using official CMake presets
# Reference: https://ai.google.dev/edge/litert/build/cmake_litert
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PLUGIN_DIR/prebuilt/android/arm64-v8a"
LITERT_DIR="$PLUGIN_DIR/third_party/LiteRT/litert"
HOST_FLATC_DIR="$PLUGIN_DIR/third_party/LiteRT/host_flatc_build"

JOBS=$(nproc 2>/dev/null || echo 4)
[[ "$1" == "--jobs" || "$1" == "-j" ]] && JOBS="$2"

echo "=== LiteRT Android Build (CMake) ==="
echo "  Jobs: $JOBS"

# Check for Android NDK
if [[ -z "$ANDROID_NDK_HOME" ]]; then
    # Try common locations
    for NDK_PATH in "$HOME/Android/Sdk/ndk"/* "/opt/android-ndk"* "$ANDROID_HOME/ndk"/*; do
        if [[ -d "$NDK_PATH" ]]; then
            export ANDROID_NDK_HOME="$NDK_PATH"
            break
        fi
    done
fi

if [[ -z "$ANDROID_NDK_HOME" || ! -d "$ANDROID_NDK_HOME" ]]; then
    echo "ERROR: Android NDK not found. Set ANDROID_NDK_HOME environment variable."
    echo "  Example: export ANDROID_NDK_HOME=~/Android/Sdk/ndk/27.0.12077973"
    exit 1
fi

echo "  NDK: $ANDROID_NDK_HOME"

cd "$LITERT_DIR"

# Build host flatc first if not present
if [[ ! -f "$HOST_FLATC_DIR/_deps/flatbuffers-build/flatc" ]]; then
    echo "=== Building host flatc ==="
    mkdir -p "$HOST_FLATC_DIR"
    cat > "$HOST_FLATC_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(HostFlatc LANGUAGES C CXX)
include(FetchContent)
FetchContent_Declare(flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v25.2.10)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flatbuffers)
EOF
    cmake -S "$HOST_FLATC_DIR" -B "$HOST_FLATC_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOST_FLATC_DIR" --target flatc -j"$JOBS"
fi

# Configure with Android preset and host tools
echo "=== Configuring for Android arm64 ==="
cmake --preset android-arm64 \
    -DTFLITE_HOST_TOOLS_DIR="$HOST_FLATC_DIR/_deps/flatbuffers-build"

# Build
echo "=== Building LiteRT ==="
cmake --build cmake_build_android_arm64 -j"$JOBS"

# Copy output
mkdir -p "$OUTPUT_DIR"
cp -v cmake_build_android_arm64/c/libLiteRtRuntimeCApi.so "$OUTPUT_DIR/libLiteRt.so"

echo "=== Done ==="
ls -lh "$OUTPUT_DIR/"
