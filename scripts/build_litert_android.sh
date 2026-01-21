#!/bin/bash
# Build LiteRT C API for Android
# Produces libLiteRt.so for arm64-v8a and armeabi-v7a

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PLUGIN_DIR/build_litert"
OUTPUT_DIR="$PLUGIN_DIR/prebuilt/android"

# LiteRT version - use main branch or specify a release tag
LITERT_BRANCH="main"

echo "=== Building LiteRT C API for Android ==="
echo "Plugin dir: $PLUGIN_DIR"
echo "Output dir: $OUTPUT_DIR"

# Check for Bazel
if ! command -v bazel &> /dev/null; then
    echo "ERROR: Bazel is not installed."
    echo "Install Bazel from: https://bazel.build/install"
    exit 1
fi

echo "Bazel version: $(bazel --version)"

# Check for Android NDK - LiteRT requires NDK 25b
# Supported versions: 19, 20, 21, 25
NDK_BASE="$HOME/Android/Sdk/ndk"
if [ -z "$ANDROID_NDK_HOME" ]; then
    # Look for NDK 25 specifically (required by LiteRT)
    if [ -d "$NDK_BASE" ]; then
        # Try to find NDK 25.x
        ANDROID_NDK_HOME=$(ls -d "$NDK_BASE"/25.* 2>/dev/null | sort -V | tail -1)
        if [ -z "$ANDROID_NDK_HOME" ]; then
            echo "ERROR: NDK 25 not found. LiteRT requires NDK 25b."
            echo "Available NDK versions:"
            ls "$NDK_BASE" 2>/dev/null || echo "  (none)"
            echo ""
            echo "Install NDK 25.2.9519653 via Android Studio SDK Manager or:"
            echo "  sdkmanager 'ndk;25.2.9519653'"
            exit 1
        fi
    fi
fi

if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "ERROR: Android NDK not found."
    echo "LiteRT requires NDK 25b. Install via:"
    echo "  sdkmanager 'ndk;25.2.9519653'"
    exit 1
fi

echo "Using Android NDK: $ANDROID_NDK_HOME"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR/arm64-v8a"
mkdir -p "$OUTPUT_DIR/armeabi-v7a"

# Clone LiteRT if not present
LITERT_DIR="$BUILD_DIR/LiteRT"
if [ ! -d "$LITERT_DIR" ]; then
    echo "Cloning LiteRT..."
    git clone --depth 1 --branch "$LITERT_BRANCH" https://github.com/google-ai-edge/LiteRT.git "$LITERT_DIR"
else
    echo "LiteRT already cloned at $LITERT_DIR"
    echo "To update, delete $LITERT_DIR and re-run this script"
fi

cd "$LITERT_DIR"

# Check for Android SDK
if [ -z "$ANDROID_SDK_HOME" ]; then
    if [ -d "$HOME/Android/Sdk" ]; then
        export ANDROID_SDK_HOME="$HOME/Android/Sdk"
    elif [ -d "/usr/local/lib/android/sdk" ]; then
        export ANDROID_SDK_HOME="/usr/local/lib/android/sdk"
    fi
fi

if [ -z "$ANDROID_SDK_HOME" ] || [ ! -d "$ANDROID_SDK_HOME" ]; then
    echo "ERROR: Android SDK not found."
    echo "Set ANDROID_SDK_HOME environment variable."
    exit 1
fi

echo "Using Android SDK: $ANDROID_SDK_HOME"

# Find latest API level
ANDROID_API_LEVEL=$(ls "$ANDROID_SDK_HOME/platforms" | grep "android-" | sed 's/android-//' | sort -n | tail -1)
echo "Using Android API Level: $ANDROID_API_LEVEL"

# Find latest build tools version
ANDROID_BUILD_TOOLS_VERSION=$(ls "$ANDROID_SDK_HOME/build-tools" | sort -V | tail -1)
echo "Using Build Tools: $ANDROID_BUILD_TOOLS_VERSION"

# Run configure script to set up Android toolchain
echo ""
echo "=== Running configure for Android ==="
export ANDROID_NDK_HOME
export ANDROID_SDK_HOME
export ANDROID_API_LEVEL
export ANDROID_BUILD_TOOLS_VERSION
export TF_SET_ANDROID_WORKSPACE=1

# Run configure with Android settings (non-interactive)
python3 configure.py << CONFIGURE_EOF
$ANDROID_NDK_HOME
$ANDROID_SDK_HOME
$ANDROID_API_LEVEL
$ANDROID_BUILD_TOOLS_VERSION
CONFIGURE_EOF

echo "Configure completed"

build_for_arch() {
    local ARCH=$1
    local CONFIG=$2
    local JNI_ARCH=$3  # e.g., arm64-v8a for JNI paths

    echo ""
    echo "=== Building LiteRT for $ARCH ==="

    # Build the C API shared library and GPU accelerator
    # The GPU accelerator is fetched from @litert_gpu external repo
    bazel build \
        --config=$CONFIG \
        //litert/c:litert_runtime_c_api_so \
        @litert_gpu//:jni/${JNI_ARCH}/libLiteRtGpuAccelerator.so

    # Find and copy the LiteRT runtime
    local SO_FILE=$(find bazel-bin/litert/c -name "libLiteRt.so" -type f 2>/dev/null | head -1)

    if [ -z "$SO_FILE" ]; then
        echo "ERROR: Could not find libLiteRt.so for $ARCH"
        echo "Searching in bazel-bin..."
        find bazel-bin -name "*.so" -type f 2>/dev/null | head -20
        return 1
    fi

    echo "Found LiteRT: $SO_FILE"
    rm -f "$OUTPUT_DIR/$ARCH/libLiteRt.so"
    cp -v "$SO_FILE" "$OUTPUT_DIR/$ARCH/libLiteRt.so"

    # Find and copy the GPU accelerator from external repo
    # The GPU library is prebuilt in the @litert_gpu external repo
    local GPU_SO="bazel-LiteRT/external/litert_gpu/jni/${JNI_ARCH}/libLiteRtGpuAccelerator.so"
    if [ -f "$GPU_SO" ]; then
        echo "Found GPU accelerator: $GPU_SO"
        rm -f "$OUTPUT_DIR/$ARCH/libLiteRtGpuAccelerator.so"
        cp -v "$GPU_SO" "$OUTPUT_DIR/$ARCH/libLiteRtGpuAccelerator.so"
    else
        # Try finding it in bazel cache
        GPU_SO=$(find bazel-LiteRT/external/litert_gpu -name "libLiteRtGpuAccelerator.so" -path "*${JNI_ARCH}*" -type f 2>/dev/null | head -1)
        if [ -n "$GPU_SO" ]; then
            echo "Found GPU accelerator: $GPU_SO"
            rm -f "$OUTPUT_DIR/$ARCH/libLiteRtGpuAccelerator.so"
            cp -v "$GPU_SO" "$OUTPUT_DIR/$ARCH/libLiteRtGpuAccelerator.so"
        else
            echo "WARNING: GPU accelerator not found at expected path"
            echo "  Expected: bazel-LiteRT/external/litert_gpu/jni/${JNI_ARCH}/libLiteRtGpuAccelerator.so"
        fi
    fi

    echo "Built libraries for $ARCH"
}

# Build for arm64-v8a (most common)
build_for_arch "arm64-v8a" "android_arm64" "arm64-v8a"

# Optionally build for armeabi-v7a (32-bit, less common now)
if [ "$BUILD_32BIT" = "1" ]; then
    build_for_arch "armeabi-v7a" "android_arm" "armeabi-v7a"
fi

# Copy headers if not already present
HEADERS_DIR="$PLUGIN_DIR/prebuilt/include/litert"
if [ ! -d "$HEADERS_DIR/c" ]; then
    echo ""
    echo "=== Copying LiteRT headers ==="
    mkdir -p "$HEADERS_DIR"
    cp -rv "$LITERT_DIR/litert/c" "$HEADERS_DIR/"
    cp -rv "$LITERT_DIR/litert/cc" "$HEADERS_DIR/" 2>/dev/null || true
    echo "Headers copied to $HEADERS_DIR"
else
    echo "Headers already present at $HEADERS_DIR"
fi

echo ""
echo "=== Build Complete ==="
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/arm64-v8a/"

echo ""
echo "To build 32-bit ARM as well, run: BUILD_32BIT=1 $0"