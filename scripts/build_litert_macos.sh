#!/bin/bash
# Build LiteRT for macOS (Universal Binary: x86_64 + arm64)
# This script builds the LiteRT C API library needed for neural echo cancellation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PLUGIN_DIR/build_litert"
OUTPUT_DIR="$PLUGIN_DIR/prebuilt/macos"
LITERT_SOURCE_DIR="$PLUGIN_DIR/prebuilt/include"

echo "=== Building LiteRT for macOS (Universal Binary) ==="
echo "Plugin dir: $PLUGIN_DIR"
echo "Build dir: $BUILD_DIR"

# Check for required tools
command -v cmake >/dev/null 2>&1 || { echo "cmake required but not found"; exit 1; }
command -v ninja >/dev/null 2>&1 || { echo "ninja required but not found"; exit 1; }

mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Clone TensorFlow Lite if not present (needed as dependency)
TENSORFLOW_DIR="$BUILD_DIR/tensorflow"
TFLITE_VERSION="v2.18.0"

if [ ! -d "$TENSORFLOW_DIR" ]; then
    echo "Cloning TensorFlow $TFLITE_VERSION..."
    git clone --depth 1 --branch "$TFLITE_VERSION" https://github.com/tensorflow/tensorflow.git "$TENSORFLOW_DIR"
else
    echo "TensorFlow already present"
fi

build_for_arch() {
    local ARCH=$1
    echo ""
    echo "=== Building LiteRT for $ARCH ==="

    local CMAKE_BUILD_DIR="$BUILD_DIR/cmake_build_${ARCH}"
    mkdir -p "$CMAKE_BUILD_DIR"

    # First build TensorFlow Lite
    local TFLITE_BUILD_DIR="$CMAKE_BUILD_DIR/tflite"
    mkdir -p "$TFLITE_BUILD_DIR"

    echo "Building TensorFlow Lite for $ARCH..."
    cmake -S "$TENSORFLOW_DIR/tensorflow/lite" \
        -B "$TFLITE_BUILD_DIR" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
        -DTFLITE_ENABLE_XNNPACK=ON \
        -DTFLITE_ENABLE_RUY=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DABSL_PROPAGATE_CXX_STD=ON \
        -Wno-dev

    cmake --build "$TFLITE_BUILD_DIR" -j$(sysctl -n hw.ncpu) --target tensorflow-lite

    # Now build LiteRT with the TFLite dependency
    echo "Building LiteRT for $ARCH..."
    cmake -S "$LITERT_SOURCE_DIR/litert" \
        -B "$CMAKE_BUILD_DIR/litert" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DTFLITE_SOURCE_DIR="$TENSORFLOW_DIR/tensorflow/lite" \
        -DTFLITE_BUILD_DIR="$TFLITE_BUILD_DIR" \
        -DTENSORFLOW_SOURCE_DIR="$TENSORFLOW_DIR" \
        -Wno-dev || {
            echo "LiteRT CMake configuration failed for $ARCH"
            echo "Attempting fallback: building combined static library..."
            build_combined_static "$ARCH" "$TFLITE_BUILD_DIR"
            return
        }

    cmake --build "$CMAKE_BUILD_DIR/litert" -j$(sysctl -n hw.ncpu)

    # Find the output library
    local LIB_PATH=$(find "$CMAKE_BUILD_DIR/litert" -name "libLiteRt*.dylib" -o -name "libLiteRtRuntimeCApi.dylib" 2>/dev/null | head -1)
    if [ -z "$LIB_PATH" ]; then
        echo "Warning: Dynamic library not found, attempting static build..."
        build_combined_static "$ARCH" "$TFLITE_BUILD_DIR"
    else
        cp "$LIB_PATH" "$BUILD_DIR/libLiteRt_${ARCH}.dylib"
    fi
}

build_combined_static() {
    local ARCH=$1
    local TFLITE_BUILD_DIR=$2

    echo "Building combined static library for $ARCH..."

    # Create a minimal shared library that wraps the TFLite C API
    # This is a fallback when the full LiteRT build fails
    local WRAPPER_DIR="$BUILD_DIR/wrapper_${ARCH}"
    mkdir -p "$WRAPPER_DIR"

    # The TFLite C library should be sufficient for basic inference
    local TFLITE_C_LIB="$TFLITE_BUILD_DIR/libtensorflowlite_c.dylib"
    if [ -f "$TFLITE_C_LIB" ]; then
        cp "$TFLITE_C_LIB" "$BUILD_DIR/libLiteRt_${ARCH}.dylib"
        echo "Using TFLite C API as fallback for $ARCH"
    else
        # Build TFLite C API if not present
        cmake --build "$TFLITE_BUILD_DIR" -j$(sysctl -n hw.ncpu) --target tensorflowlite_c
        TFLITE_C_LIB="$TFLITE_BUILD_DIR/libtensorflowlite_c.dylib"
        if [ -f "$TFLITE_C_LIB" ]; then
            cp "$TFLITE_C_LIB" "$BUILD_DIR/libLiteRt_${ARCH}.dylib"
        else
            echo "ERROR: Could not build library for $ARCH"
            return 1
        fi
    fi
}

# Build for both architectures
ARCHS=("x86_64" "arm64")
BUILT_LIBS=()

for arch in "${ARCHS[@]}"; do
    if build_for_arch "$arch"; then
        if [ -f "$BUILD_DIR/libLiteRt_${arch}.dylib" ]; then
            BUILT_LIBS+=("$BUILD_DIR/libLiteRt_${arch}.dylib")
        fi
    fi
done

# Create universal binary
if [ ${#BUILT_LIBS[@]} -eq 2 ]; then
    echo ""
    echo "=== Creating Universal Binary ==="
    lipo -create "${BUILT_LIBS[@]}" -output "$OUTPUT_DIR/libLiteRt.dylib"

    # Fix install name
    install_name_tool -id "@rpath/libLiteRt.dylib" "$OUTPUT_DIR/libLiteRt.dylib"

    echo "Universal binary created:"
    lipo -info "$OUTPUT_DIR/libLiteRt.dylib"
elif [ ${#BUILT_LIBS[@]} -eq 1 ]; then
    echo ""
    echo "=== Single architecture build ==="
    cp "${BUILT_LIBS[0]}" "$OUTPUT_DIR/libLiteRt.dylib"
    install_name_tool -id "@rpath/libLiteRt.dylib" "$OUTPUT_DIR/libLiteRt.dylib"
    echo "Single-arch binary created:"
    lipo -info "$OUTPUT_DIR/libLiteRt.dylib"
else
    echo "ERROR: No libraries were built successfully"
    exit 1
fi

echo ""
echo "=== Build Complete ==="
ls -la "$OUTPUT_DIR/libLiteRt.dylib"
