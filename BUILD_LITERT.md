# Building LiteRT for macOS

This document describes how to build the LiteRT library for macOS in a reproducible way.

## Prerequisites

- Xcode Command Line Tools
- Bazel (install via Homebrew: `brew install bazelisk`)
- Git

## Build Steps

### 1. Clone LiteRT Repository

```bash
cd /path/to/your/workspace
git clone https://github.com/google-ai-edge/LiteRT.git
cd LiteRT
```

### 2. Configure Build

```bash
./configure
# Accept defaults for most options
# Choose "No" for GPU/NPU support (we only need CPU)
```

### 3. Build Universal Binary (arm64 + x86_64)

```bash
# Build for arm64 (Apple Silicon)
bazel build -c opt --config=macos_arm64 //litert/c:litert_c_api_shared

# Build for x86_64 (Intel)
bazel build -c opt --config=macos //litert/c:litert_c_api_shared

# Create universal binary
lipo -create \
  bazel-out/darwin_arm64-opt/bin/_solib_darwin_arm64/_U_S_Slitert_Sc_Clitert_Uruntime_Uc_Uapi_Ushared_Ulib___Ulitert_Sc/libLiteRt.dylib \
  bazel-out/darwin_x86_64-opt/bin/_solib_darwin_x86_64/_U_S_Slitert_Sc_Clitert_Uruntime_Uc_Uapi_Ushared_Ulib___Ulitert_Sc/libLiteRt.dylib \
  -output libLiteRt_universal.dylib

# Verify universal binary
lipo -info libLiteRt_universal.dylib
# Should show: Architectures in the fat file: libLiteRt_universal.dylib are: x86_64 arm64
```

### 4. Install to Plugin

```bash
# Copy to plugin prebuilt directory
cp libLiteRt_universal.dylib /path/to/flowstate/plugins/flutter_recorder/prebuilt/macos/libLiteRt.dylib

# Copy headers (needed for compilation)
cp -r litert/c /path/to/flowstate/plugins/flutter_recorder/prebuilt/include/litert/
cp -r litert/build_common /path/to/flowstate/plugins/flutter_recorder/prebuilt/include/litert/

# Generate build_config.h
cp bazel-out/darwin_x86_64-opt/bin/litert/build_common/build_config.h \
   /path/to/flowstate/plugins/flutter_recorder/prebuilt/include/litert/build_common/
```

### 5. Verify Installation

```bash
cd /path/to/flowstate
flutter clean
flutter pub get
cd macos && pod install
cd ..
flutter build macos --debug
```

## Current Build (x86_64 only)

The current prebuilt binary was built with:

```bash
bazel build -c opt --config=macos //litert/c:litert_c_api_shared
```

And copied from:
```
bazel-out/darwin_x86_64-opt/bin/_solib_darwin_x86_64/_U_S_Slitert_Sc_Clitert_Uruntime_Uc_Uapi_Ushared_Ulib___Ulitert_Sc/libLiteRt.dylib
```

To build a universal binary supporting both Apple Silicon and Intel, follow the steps above.

## Troubleshooting

### Library Not Found at Runtime

If you see `dyld: Library not loaded: @rpath/libLiteRt.dylib`, the podspec's `prepare_command` should automatically fix the install name. If not, manually run:

```bash
install_name_tool -id "@rpath/libLiteRt.dylib" prebuilt/macos/libLiteRt.dylib
codesign --force --sign - prebuilt/macos/libLiteRt.dylib
```

### Build for Wrong Architecture

Check the current architecture:
```bash
lipo -info prebuilt/macos/libLiteRt.dylib
```

Exclude the architecture you don't have:
- Intel Mac: Keep `'EXCLUDED_ARCHS[sdk=macosx*]' => 'arm64'` in podspec
- Apple Silicon: Change to `'EXCLUDED_ARCHS[sdk=macosx*]' => 'x86_64'` in podspec

## References

- [LiteRT C++ Documentation](https://ai.google.dev/edge/litert/next/cpp)
- [LiteRT GitHub Repository](https://github.com/google-ai-edge/LiteRT)
- [Building with Bazel](https://github.com/google-ai-edge/LiteRT/blob/main/docs/build/build.md)
