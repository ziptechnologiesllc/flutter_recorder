#!/bin/bash
# Build TensorFlow Lite C library for desktop platforms (macOS, Linux, Windows)
# Run this script once to generate the pre-built libraries
# For macOS: builds universal binary (x86_64 + arm64)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
TFLITE_VERSION="v2.14.0"
BUILD_DIR="$PLUGIN_DIR/build_tflite"
OUTPUT_DIR="$PLUGIN_DIR/prebuilt"

echo "=== Building TensorFlow Lite C API for Desktop ==="
echo "Plugin dir: $PLUGIN_DIR"
echo "TFLite version: $TFLITE_VERSION"

# Detect platform
PLATFORM="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
fi
echo "Platform: $PLATFORM"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR/$PLATFORM"

# Clone TensorFlow if not present
TENSORFLOW_DIR="$BUILD_DIR/tensorflow"
if [ ! -d "$TENSORFLOW_DIR" ]; then
    echo "Cloning TensorFlow..."
    git clone --depth 1 --branch "$TFLITE_VERSION" https://github.com/tensorflow/tensorflow.git "$TENSORFLOW_DIR"
else
    echo "TensorFlow already cloned"
fi

build_for_arch() {
    local ARCH=$1
    echo ""
    echo "=== Building for $ARCH ==="

    CMAKE_BUILD_DIR="$BUILD_DIR/cmake_build_${PLATFORM}_${ARCH}"
    mkdir -p "$CMAKE_BUILD_DIR"
    cd "$CMAKE_BUILD_DIR"

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DTFLITE_C_BUILD_SHARED_LIBS=ON
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -Wno-dev
        # Disable problematic optimizations for macOS cross-compile
        -DTFLITE_ENABLE_RUY=OFF
        -DTFLITE_ENABLE_XNNPACK=OFF
    )

    if [[ "$PLATFORM" == "macos" ]]; then
        CMAKE_ARGS+=(-DCMAKE_OSX_ARCHITECTURES="$ARCH")
    fi

    echo "Configuring CMake for $ARCH..."
    # CMake 4.x requires policy version - export for all subprojects
    export CMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake "$TENSORFLOW_DIR/tensorflow/lite/c" "${CMAKE_ARGS[@]}" \
        -DCMAKE_POLICY_DEFAULT_CMP0135=NEW \
        -DCMAKE_POLICY_DEFAULT_CMP0169=OLD

    echo "Building for $ARCH..."
    cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
}

# Copy headers (same for all platforms)
copy_headers() {
    echo "Copying headers..."

    # Main C API headers
    mkdir -p "$OUTPUT_DIR/include/tensorflow/lite/c"
    cp -v "$TENSORFLOW_DIR/tensorflow/lite/c/c_api.h" "$OUTPUT_DIR/include/tensorflow/lite/c/"
    cp -v "$TENSORFLOW_DIR/tensorflow/lite/c/c_api_types.h" "$OUTPUT_DIR/include/tensorflow/lite/c/"
    cp -v "$TENSORFLOW_DIR/tensorflow/lite/c/common.h" "$OUTPUT_DIR/include/tensorflow/lite/c/"

    # Core C API headers
    if [ -d "$TENSORFLOW_DIR/tensorflow/lite/core/c" ]; then
        mkdir -p "$OUTPUT_DIR/include/tensorflow/lite/core/c"
        cp -v "$TENSORFLOW_DIR/tensorflow/lite/core/c/"*.h "$OUTPUT_DIR/include/tensorflow/lite/core/c/" 2>/dev/null || true
    fi

    # Additional required headers at tensorflow/lite level
    mkdir -p "$OUTPUT_DIR/include/tensorflow/lite"
    cp -v "$TENSORFLOW_DIR/tensorflow/lite/builtin_ops.h" "$OUTPUT_DIR/include/tensorflow/lite/" 2>/dev/null || true
    cp -v "$TENSORFLOW_DIR/tensorflow/lite/builtin_op_data.h" "$OUTPUT_DIR/include/tensorflow/lite/" 2>/dev/null || true

    # Core headers
    if [ -d "$TENSORFLOW_DIR/tensorflow/lite/core" ]; then
        mkdir -p "$OUTPUT_DIR/include/tensorflow/lite/core"
        cp -v "$TENSORFLOW_DIR/tensorflow/lite/core/builtin_ops.h" "$OUTPUT_DIR/include/tensorflow/lite/core/" 2>/dev/null || true
    fi
}

if [[ "$PLATFORM" == "macos" ]]; then
    # Detect native architecture
    NATIVE_ARCH=$(uname -m)
    echo "Native architecture: $NATIVE_ARCH"

    # Build for native architecture first
    build_for_arch "$NATIVE_ARCH"
    NATIVE_LIB="$BUILD_DIR/cmake_build_macos_${NATIVE_ARCH}/libtensorflowlite_c.dylib"

    # Try to build for the other architecture (cross-compile)
    if [[ "$NATIVE_ARCH" == "arm64" ]]; then
        OTHER_ARCH="x86_64"
    else
        OTHER_ARCH="arm64"
    fi

    echo ""
    echo "=== Attempting cross-compile for $OTHER_ARCH ==="
    if build_for_arch "$OTHER_ARCH" 2>/dev/null; then
        OTHER_LIB="$BUILD_DIR/cmake_build_macos_${OTHER_ARCH}/libtensorflowlite_c.dylib"

        # Create universal binary
        echo ""
        echo "=== Creating Universal Binary ==="
        UNIVERSAL_LIB="$OUTPUT_DIR/$PLATFORM/libtensorflowlite_c.dylib"
        lipo -create "$NATIVE_LIB" "$OTHER_LIB" -output "$UNIVERSAL_LIB"
    else
        echo ""
        echo "=== Cross-compile failed, using native-only binary ==="
        cp "$NATIVE_LIB" "$OUTPUT_DIR/$PLATFORM/libtensorflowlite_c.dylib"
        UNIVERSAL_LIB="$OUTPUT_DIR/$PLATFORM/libtensorflowlite_c.dylib"
    fi

    # Verify
    echo "Binary architectures:"
    lipo -info "$UNIVERSAL_LIB"

    # Fix install name
    install_name_tool -id "@rpath/libtensorflowlite_c.dylib" "$UNIVERSAL_LIB"

    copy_headers

elif [[ "$PLATFORM" == "linux" ]]; then
    build_for_arch "native"
    cp -v "$BUILD_DIR/cmake_build_linux_native/libtensorflowlite_c.so" "$OUTPUT_DIR/$PLATFORM/"
    copy_headers

elif [[ "$PLATFORM" == "windows" ]]; then
    build_for_arch "native"
    cp -v "$BUILD_DIR/cmake_build_windows_native/Release/tensorflowlite_c.dll" "$OUTPUT_DIR/$PLATFORM/" 2>/dev/null || \
    cp -v "$BUILD_DIR/cmake_build_windows_native/tensorflowlite_c.dll" "$OUTPUT_DIR/$PLATFORM/"
    copy_headers
fi

echo ""
echo "=== Build Complete ==="
echo "Output: $OUTPUT_DIR/$PLATFORM"
ls -la "$OUTPUT_DIR/$PLATFORM"
