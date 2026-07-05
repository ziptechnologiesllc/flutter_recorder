#!/bin/bash
# Build LiteRT for all supported platforms
# Reference: https://ai.google.dev/edge/litert/build/cmake_litert
#
# Platform matrix:
#   - Linux x64, arm64
#   - macOS x64, arm64 (universal binary)
#   - Windows x64, arm64
#   - Android arm64-v8a, armeabi-v7a
#   - iOS (arm64, simulator)
#
# Usage:
#   ./build_litert.sh [platform] [options]
#
# Platforms:
#   linux-x64, linux-arm64, linux (native)
#   macos-x64, macos-arm64, macos-universal, macos (native)
#   windows-x64, windows-arm64, windows (native)
#   android-arm64, android-arm32, android (both)
#   ios
#   all (build everything for current host)
#
# Options:
#   --jobs N, -j N    Parallel jobs (default: nproc)
#   --clean           Clean build directories first
#   --debug           Build debug version

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
LITERT_DIR="$PLUGIN_DIR/third_party/LiteRT/litert"
PREBUILT_DIR="$PLUGIN_DIR/prebuilt"
HOST_FLATC_DIR="$PLUGIN_DIR/third_party/LiteRT/host_flatc_build"

# Detect host
case "$(uname -s)" in
  Linux*)  HOST_OS="linux" ;;
  Darwin*) HOST_OS="macos" ;;
  MINGW*|MSYS*|CYGWIN*) HOST_OS="windows" ;;
  *) echo "Unknown host OS"; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64|amd64) HOST_ARCH="x64" ;;
  aarch64|arm64) HOST_ARCH="arm64" ;;
  *) echo "Unknown host architecture"; exit 1 ;;
esac

# Defaults
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CLEAN=0
BUILD_TYPE="Release"
TARGET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs|-j) JOBS="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --debug) BUILD_TYPE="Debug"; shift ;;
    --help|-h)
      head -30 "$0" | tail -28
      exit 0
      ;;
    *) TARGET="$1"; shift ;;
  esac
done

echo "=== LiteRT Multi-Platform Build ==="
echo "  Host: $HOST_OS-$HOST_ARCH"
echo "  Target: ${TARGET:-native}"
echo "  Jobs: $JOBS"
echo "  Build type: $BUILD_TYPE"
echo ""

# Check LiteRT source
if [[ ! -f "$LITERT_DIR/CMakeLists.txt" ]]; then
  echo "ERROR: LiteRT source not found at $LITERT_DIR"
  echo "Initialize submodule: git submodule update --init third_party/LiteRT"
  exit 1
fi

# Build host flatc for cross-compilation
build_host_flatc() {
  if [[ -f "$HOST_FLATC_DIR/_deps/flatbuffers-build/flatc" ]]; then
    echo "Host flatc already built"
    return
  fi

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
}

# Linux native build
build_linux() {
  local arch="${1:-$HOST_ARCH}"
  local build_dir="cmake_build_linux_$arch"
  local output_dir="$PREBUILT_DIR/linux/$arch"

  echo "=== Building LiteRT for Linux $arch ==="

  cd "$LITERT_DIR"
  [[ $CLEAN -eq 1 ]] && rm -rf "$build_dir"

  local cmake_args=(
    -S . -B "$build_dir"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DLITERT_AUTO_BUILD_TFLITE=ON
    -DLITERT_ENABLE_GPU=OFF
    -DLITERT_ENABLE_NPU=OFF
    -DLITERT_DISABLE_KLEIDIAI=ON
    -DLITERT_BUILD_TOOLS=OFF
    -DLITERT_BUILD_TESTS=OFF
  )

  # Cross-compile for arm64 on x64 host
  if [[ "$arch" == "arm64" && "$HOST_ARCH" == "x64" ]]; then
    cmake_args+=(-DCMAKE_SYSTEM_PROCESSOR=aarch64)
    cmake_args+=(-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc)
    cmake_args+=(-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++)
  fi

  cmake "${cmake_args[@]}"
  cmake --build "$build_dir" -j"$JOBS" --target litert_runtime_c_api_shared_lib

  mkdir -p "$output_dir"
  cp -v "$build_dir/c/libLiteRtRuntimeCApi.so" "$output_dir/libLiteRt.so"
  strip --strip-unneeded "$output_dir/libLiteRt.so" 2>/dev/null || true

  echo "Built: $output_dir/libLiteRt.so"
}

# macOS build (universal binary)
build_macos() {
  local arch="${1:-universal}"
  local output_dir="$PREBUILT_DIR/macos"

  echo "=== Building LiteRT for macOS $arch ==="

  cd "$LITERT_DIR"

  build_macos_arch() {
    local a="$1"
    local build_dir="cmake_build_macos_$a"
    [[ $CLEAN -eq 1 ]] && rm -rf "$build_dir"

    cmake -S . -B "$build_dir" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_OSX_ARCHITECTURES="$a" \
      -DLITERT_AUTO_BUILD_TFLITE=ON \
      -DLITERT_ENABLE_GPU=OFF \
      -DLITERT_ENABLE_NPU=OFF \
      -DLITERT_BUILD_TOOLS=OFF \
      -DLITERT_BUILD_TESTS=OFF \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-dead_strip"

    cmake --build "$build_dir" -j"$JOBS" --target litert_runtime_c_api_shared_lib
  }

  mkdir -p "$output_dir"

  if [[ "$arch" == "universal" ]]; then
    build_macos_arch arm64
    build_macos_arch x86_64

    lipo -create \
      "cmake_build_macos_arm64/c/libLiteRtRuntimeCApi.dylib" \
      "cmake_build_macos_x86_64/c/libLiteRtRuntimeCApi.dylib" \
      -output "$output_dir/libLiteRt.dylib"
  else
    build_macos_arch "$arch"
    cp -v "cmake_build_macos_$arch/c/libLiteRtRuntimeCApi.dylib" "$output_dir/libLiteRt.dylib"
  fi

  # Fix install name and sign
  install_name_tool -id "@rpath/libLiteRt.dylib" "$output_dir/libLiteRt.dylib"
  codesign --force --sign - "$output_dir/libLiteRt.dylib"

  echo "Built: $output_dir/libLiteRt.dylib"
  lipo -info "$output_dir/libLiteRt.dylib"
}

# Download GPU accelerator library from LiteRT releases
download_gpu_accelerator() {
  local abi="$1"
  local output_dir="$2"

  # GPU accelerator is distributed via litert_gpu external repo
  # For now, try to find it in the Bazel external cache or download from Maven
  local GPU_AAR_URL="https://repo1.maven.org/maven2/com/google/ai/edge/litert/litert-gpu/1.0.1/litert-gpu-1.0.1.aar"
  local GPU_AAR="$PLUGIN_DIR/build_litert/litert-gpu.aar"

  if [[ ! -f "$output_dir/libLiteRtGpuAccelerator.so" ]]; then
    echo "  Downloading GPU accelerator..."
    mkdir -p "$(dirname "$GPU_AAR")"

    if curl -fsSL -o "$GPU_AAR" "$GPU_AAR_URL" 2>/dev/null; then
      # Extract the .so file from AAR (it's a zip)
      unzip -q -o "$GPU_AAR" "jni/$abi/libLiteRtGpuAccelerator.so" -d "$PLUGIN_DIR/build_litert/" 2>/dev/null || true
      if [[ -f "$PLUGIN_DIR/build_litert/jni/$abi/libLiteRtGpuAccelerator.so" ]]; then
        cp -v "$PLUGIN_DIR/build_litert/jni/$abi/libLiteRtGpuAccelerator.so" "$output_dir/"
        echo "  GPU accelerator downloaded for $abi"
      else
        echo "  WARNING: GPU accelerator not available for $abi"
      fi
    else
      echo "  WARNING: Could not download GPU accelerator"
    fi
  else
    echo "  GPU accelerator already present"
  fi
}

# Android build
build_android() {
  local abi="${1:-arm64-v8a}"
  local output_dir="$PREBUILT_DIR/android/$abi"
  local build_dir="cmake_build_android_$abi"

  echo "=== Building LiteRT for Android $abi ==="

  # Check NDK
  if [[ -z "$ANDROID_NDK_HOME" ]]; then
    for p in "$HOME/Android/Sdk/ndk"/* "$ANDROID_HOME/ndk"/*; do
      [[ -d "$p" ]] && export ANDROID_NDK_HOME="$p" && break
    done
  fi

  if [[ -z "$ANDROID_NDK_HOME" || ! -d "$ANDROID_NDK_HOME" ]]; then
    echo "ERROR: Android NDK not found. Set ANDROID_NDK_HOME."
    return 1
  fi

  echo "  NDK: $ANDROID_NDK_HOME"

  # Need host flatc for cross-compilation
  build_host_flatc

  cd "$LITERT_DIR"
  [[ $CLEAN -eq 1 ]] && rm -rf "$build_dir"

  local platform=26
  [[ "$abi" == "armeabi-v7a" ]] && platform=21

  # Enable GPU on arm64-v8a (OpenGL ES), NPU requires external Qualcomm libs
  local enable_gpu="ON"
  local enable_npu="OFF"
  [[ "$abi" == "armeabi-v7a" ]] && enable_gpu="OFF"

  cmake -S . -B "$build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$abi" \
    -DANDROID_PLATFORM="$platform" \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DLITERT_AUTO_BUILD_TFLITE=ON \
    -DLITERT_ENABLE_GPU="$enable_gpu" \
    -DLITERT_ENABLE_NPU="$enable_npu" \
    -DLITERT_BUILD_TOOLS=OFF \
    -DLITERT_BUILD_TESTS=OFF \
    -DTFLITE_HOST_TOOLS_DIR="$HOST_FLATC_DIR/_deps/flatbuffers-build"

  cmake --build "$build_dir" -j"$JOBS" --target litert_runtime_c_api_shared_lib

  mkdir -p "$output_dir"
  cp -v "$build_dir/c/libLiteRtRuntimeCApi.so" "$output_dir/libLiteRt.so"

  # Download GPU accelerator dispatch library (separate from runtime)
  if [[ "$enable_gpu" == "ON" ]]; then
    download_gpu_accelerator "$abi" "$output_dir"
  fi

  echo "Built: $output_dir/libLiteRt.so"
  ls -la "$output_dir/"
}

# Windows build (requires running on Windows or cross-compile setup)
build_windows() {
  local arch="${1:-x64}"
  local output_dir="$PREBUILT_DIR/windows/$arch"
  local build_dir="cmake_build_windows_$arch"

  echo "=== Building LiteRT for Windows $arch ==="

  if [[ "$HOST_OS" != "windows" ]]; then
    echo "WARNING: Cross-compiling for Windows from $HOST_OS is not directly supported."
    echo "Build natively on Windows or use a cross-compilation toolchain."
    return 1
  fi

  cd "$LITERT_DIR"
  [[ $CLEAN -eq 1 ]] && rm -rf "$build_dir"

  local cmake_arch="x64"
  [[ "$arch" == "arm64" ]] && cmake_arch="ARM64"

  cmake -S . -B "$build_dir" -A "$cmake_arch" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DLITERT_AUTO_BUILD_TFLITE=ON \
    -DLITERT_ENABLE_GPU=OFF \
    -DLITERT_ENABLE_NPU=OFF \
    -DLITERT_BUILD_TOOLS=OFF \
    -DLITERT_BUILD_TESTS=OFF

  cmake --build "$build_dir" --config "$BUILD_TYPE" --target litert_runtime_c_api_shared_lib -j"$JOBS"

  mkdir -p "$output_dir"
  cp -v "$build_dir/c/$BUILD_TYPE/LiteRtRuntimeCApi.dll" "$output_dir/LiteRt.dll"
  cp -v "$build_dir/c/$BUILD_TYPE/LiteRtRuntimeCApi.lib" "$output_dir/LiteRt.lib"

  echo "Built: $output_dir/LiteRt.dll"
}

# iOS build (framework)
build_ios() {
  echo "=== Building LiteRT for iOS ==="
  echo "iOS uses TensorFlowLiteC CocoaPod - no separate build needed."
  echo "The podspec at ios/flutter_recorder.podspec handles the dependency."
  echo ""
  echo "To use LiteRT instead of TensorFlowLiteC, you would need to:"
  echo "  1. Build LiteRT xcframework from source"
  echo "  2. Update the podspec to use vendored_frameworks"
  echo ""
  echo "For now, iOS continues to use TensorFlowLiteC from CocoaPods."
}

# Dispatch to the right builder
case "$TARGET" in
  linux-x64)      build_linux x64 ;;
  linux-arm64)    build_linux arm64 ;;
  linux|"")
    if [[ "$HOST_OS" == "linux" ]]; then
      build_linux "$HOST_ARCH"
    fi
    ;;

  macos-x64)      build_macos x86_64 ;;
  macos-arm64)    build_macos arm64 ;;
  macos-universal) build_macos universal ;;
  macos)
    if [[ "$HOST_OS" == "macos" ]]; then
      build_macos universal
    fi
    ;;

  windows-x64)    build_windows x64 ;;
  windows-arm64)  build_windows arm64 ;;
  windows)
    if [[ "$HOST_OS" == "windows" ]]; then
      build_windows "$HOST_ARCH"
    fi
    ;;

  android-arm64)  build_android arm64-v8a ;;
  android-arm32)  build_android armeabi-v7a ;;
  android)
    build_android arm64-v8a
    build_android armeabi-v7a
    ;;

  ios)            build_ios ;;

  all)
    echo "Building all platforms supported on this host..."
    case "$HOST_OS" in
      linux)
        build_linux x64
        # build_linux arm64  # Requires cross-compiler
        build_android arm64-v8a
        build_android armeabi-v7a
        ;;
      macos)
        build_macos universal
        build_android arm64-v8a
        build_android armeabi-v7a
        build_ios
        ;;
      windows)
        build_windows x64
        # build_windows arm64  # Requires ARM64 toolchain
        ;;
    esac
    ;;

  *)
    echo "Unknown target: $TARGET"
    echo "Run with --help for usage"
    exit 1
    ;;
esac

echo ""
echo "=== Build Complete ==="
echo "Prebuilt libraries are in: $PREBUILT_DIR/"
ls -la "$PREBUILT_DIR"/*/ 2>/dev/null || true
