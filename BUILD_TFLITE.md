# TensorFlow Lite Build Guide

This document describes how TensorFlow Lite is integrated on each platform for the flutter_recorder plugin's neural post-filter (AEC).

## Platform Overview

| Platform | TFLite Integration | Source | Build Required |
|----------|-------------------|--------|----------------|
| Android  | Google Play Services TFLite (Prefab) | Runtime from device | No |
| iOS      | TensorFlowLiteC CocoaPod | Pre-built via CocoaPods | No |
| macOS    | Pre-built universal dylib | Built from source | Yes (CI) |
| Windows  | Pre-built DLL | Built from source | Yes (CI) |
| Linux    | Pre-built shared library | Built from source | Yes (CI) |

## CI/CD Build Matrix

For release builds, TFLite must be built on each target platform:

| Runner | Architecture | Output |
|--------|--------------|--------|
| macos-13 (Intel) | x86_64 | `libtensorflowlite_c.dylib` |
| macos-14 (Apple Silicon) | arm64 | `libtensorflowlite_c.dylib` |
| ubuntu-latest | x86_64 | `libtensorflowlite_c.so` |
| windows-latest | x86_64 | `tensorflowlite_c.dll` |

After building on each platform, use `lipo` to create universal binaries for macOS.

## Android

**Integration**: Google Play Services TFLite with Prefab native support.

**Files modified**:
- `android/build.gradle`: Adds `play-services-tflite-java` dependency
- `src/CMakeLists.txt`: Uses `find_package(tensorflowlite_jni_gms_client)`

**How it works**:
- TFLite runtime is provided by Google Play Services on the device
- No bundled library needed - runtime downloaded automatically
- Prefab provides CMake integration for native code

**Build flags**:
- `USE_TFLITE` - Enables TFLite code paths
- `TFLITE_IN_GMSCORE` - Uses Play Services runtime
- `TFLITE_WITH_STABLE_ABI` - Uses stable C API

## iOS

**Integration**: TensorFlowLiteC CocoaPod (official pre-built framework).

**Files modified**:
- `ios/flutter_recorder.podspec`: Adds `TensorFlowLiteC` dependency

**How it works**:
- CocoaPods downloads the pre-built TensorFlowLiteC framework
- Framework includes both arm64 (device) and x86_64 (simulator)
- Podspec sets `USE_TFLITE=1` preprocessor definition

## macOS

**Integration**: Pre-built universal dylib (x86_64 + arm64).

**Files modified**:
- `macos/flutter_recorder.podspec`: Vendors `libtensorflowlite_c.dylib`

**Build requirements**:
- CMake 3.16+
- Xcode Command Line Tools
- ~5GB disk space for build

**Build steps**:
```bash
cd plugins/flutter_recorder
./scripts/build_tflite_desktop.sh
```

**Output**:
- `prebuilt/macos/libtensorflowlite_c.dylib` (universal binary)
- `prebuilt/include/tensorflow/lite/c/*.h` (headers)

**How it works**:
- Script clones TensorFlow v2.14.0
- Builds TFLite C API for both x86_64 and arm64
- Creates universal binary with `lipo`
- Podspec vendors the dylib and sets `USE_TFLITE=1`

## Windows

**Integration**: Pre-built DLL.

**Files modified**:
- `src/CMakeLists.txt`: Finds and links prebuilt DLL

**Build requirements**:
- CMake 3.16+
- Visual Studio 2019+ with C++ workload
- ~5GB disk space for build

**Build steps** (in Git Bash or WSL):
```bash
cd plugins/flutter_recorder
./scripts/build_tflite_desktop.sh
```

**Output**:
- `prebuilt/windows/tensorflowlite_c.dll`
- `prebuilt/windows/tensorflowlite_c.lib`
- `prebuilt/include/tensorflow/lite/c/*.h`

**How it works**:
- CMakeLists.txt checks for prebuilt DLL
- Links against import library
- Copies DLL to output directory at build time

## Linux

**Integration**: Pre-built shared library.

**Files modified**:
- `src/CMakeLists.txt`: Finds and links prebuilt .so

**Build requirements**:
- CMake 3.16+
- GCC or Clang
- ~5GB disk space for build

**Build steps**:
```bash
cd plugins/flutter_recorder
./scripts/build_tflite_desktop.sh
```

**Output**:
- `prebuilt/linux/libtensorflowlite_c.so`
- `prebuilt/include/tensorflow/lite/c/*.h`

**How it works**:
- CMakeLists.txt checks for prebuilt .so
- Links directly against shared library
- App must ship with the .so or have it in library path

## Fallback Behavior

If TFLite is not available on a platform:
- `USE_TFLITE` is not defined
- Neural post-filter compiles as stub
- `loadModel()` returns false
- `process()` passes audio through unchanged
- AEC still works via NLMS filter (linear AEC only)

## Updating TFLite Version

1. Edit `scripts/build_tflite_desktop.sh`:
   ```bash
   TFLITE_VERSION="v2.16.0"  # or desired version
   ```

2. Delete build cache:
   ```bash
   rm -rf build_tflite
   ```

3. Rebuild:
   ```bash
   ./scripts/build_tflite_desktop.sh
   ```

4. Update iOS podspec if needed (check compatible versions):
   ```ruby
   s.dependency 'TensorFlowLiteC', '~> 2.16'
   ```

## Creating Universal Binary (macOS)

Cross-compilation often fails due to CPU feature detection issues. Build natively on each architecture instead:

```bash
# On Intel Mac:
./scripts/build_tflite_desktop.sh
mv prebuilt/macos/libtensorflowlite_c.dylib prebuilt/macos/libtensorflowlite_c_x86_64.dylib

# On Apple Silicon Mac:
./scripts/build_tflite_desktop.sh
mv prebuilt/macos/libtensorflowlite_c.dylib prebuilt/macos/libtensorflowlite_c_arm64.dylib

# Create universal binary (on either machine):
lipo -create \
  prebuilt/macos/libtensorflowlite_c_x86_64.dylib \
  prebuilt/macos/libtensorflowlite_c_arm64.dylib \
  -output prebuilt/macos/libtensorflowlite_c.dylib

# Verify:
lipo -info prebuilt/macos/libtensorflowlite_c.dylib
# Should show: Architectures in the fat file: x86_64 arm64
```

## GitHub Actions Workflow Example

```yaml
name: Build TFLite

on:
  workflow_dispatch:

jobs:
  build-macos-intel:
    runs-on: macos-13
    steps:
      - uses: actions/checkout@v4
      - name: Build TFLite
        run: |
          cd plugins/flutter_recorder
          ./scripts/build_tflite_desktop.sh
      - uses: actions/upload-artifact@v4
        with:
          name: tflite-macos-x86_64
          path: plugins/flutter_recorder/prebuilt/macos/libtensorflowlite_c.dylib

  build-macos-arm:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - name: Build TFLite
        run: |
          cd plugins/flutter_recorder
          ./scripts/build_tflite_desktop.sh
      - uses: actions/upload-artifact@v4
        with:
          name: tflite-macos-arm64
          path: plugins/flutter_recorder/prebuilt/macos/libtensorflowlite_c.dylib

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build TFLite
        run: |
          cd plugins/flutter_recorder
          ./scripts/build_tflite_desktop.sh
      - uses: actions/upload-artifact@v4
        with:
          name: tflite-linux-x86_64
          path: plugins/flutter_recorder/prebuilt/linux/libtensorflowlite_c.so

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build TFLite
        shell: bash
        run: |
          cd plugins/flutter_recorder
          ./scripts/build_tflite_desktop.sh
      - uses: actions/upload-artifact@v4
        with:
          name: tflite-windows-x86_64
          path: plugins/flutter_recorder/prebuilt/windows/tensorflowlite_c.dll

  create-universal:
    needs: [build-macos-intel, build-macos-arm]
    runs-on: macos-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: tflite-macos-x86_64
          path: x86_64
      - uses: actions/download-artifact@v4
        with:
          name: tflite-macos-arm64
          path: arm64
      - name: Create universal binary
        run: |
          lipo -create x86_64/libtensorflowlite_c.dylib arm64/libtensorflowlite_c.dylib \
            -output libtensorflowlite_c.dylib
          install_name_tool -id "@rpath/libtensorflowlite_c.dylib" libtensorflowlite_c.dylib
          lipo -info libtensorflowlite_c.dylib
      - uses: actions/upload-artifact@v4
        with:
          name: tflite-macos-universal
          path: libtensorflowlite_c.dylib
```

## Troubleshooting

### CMake version too new
If you get errors about `cmake_minimum_required`, the script includes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to handle this.

### AVX512 errors on macOS
Apple's clang doesn't support AVX512. The script disables ruy and AVX512 XNNPACK features:
```cmake
-DTFLITE_ENABLE_RUY=OFF
-DXNNPACK_ENABLE_AVX512F=OFF
```

### Build takes too long
The first build downloads ~500MB of dependencies and compiles a lot of code. Subsequent builds are faster if the `build_tflite` directory is preserved.

### macOS: Library not loaded
Ensure the dylib has the correct install name:
```bash
otool -L prebuilt/macos/libtensorflowlite_c.dylib
install_name_tool -id "@rpath/libtensorflowlite_c.dylib" prebuilt/macos/libtensorflowlite_c.dylib
```

### Windows: DLL not found
The CMakeLists.txt copies the DLL to the output directory. If it's still not found, copy manually to the same directory as the executable.

### Cross-compilation fails
Native compilation is more reliable than cross-compilation for TFLite due to CPU feature detection. Use CI/CD runners for each target architecture.
