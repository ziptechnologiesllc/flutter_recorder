#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_recorder.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_recorder'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter FFI plugin project.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'FlutterMacOS'

  # LiteRT for macOS: use prebuilt binary (built from google-ai-edge/LiteRT)
  # Located at: prebuilt/macos/libLiteRt.dylib
  s.vendored_libraries = '../prebuilt/macos/libLiteRt.dylib'

  # Ensure the dylib is copied to the app bundle at runtime
  s.prepare_command = <<-CMD
    # Verify dylib exists
    if [ ! -f "../prebuilt/macos/libLiteRt.dylib" ]; then
      echo "Error: libLiteRt.dylib not found. Please build LiteRT first."
      echo "See: https://ai.google.dev/edge/litert/next/cpp"
      exit 1
    fi

    # Fix install name if needed
    install_name_tool -id "@rpath/libLiteRt.dylib" "../prebuilt/macos/libLiteRt.dylib" 2>/dev/null || true

    # Re-sign after modification
    codesign --force --sign - "../prebuilt/macos/libLiteRt.dylib" 2>/dev/null || true
  CMD

  s.platform = :osx, '14.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
    # Enhanced optimization flags
    'OTHER_CFLAGS' => '-O3 -ffast-math -flto -funroll-loops -pthread -Wno-strict-prototypes',
    'OTHER_CPLUSPLUSFLAGS' => '-O3 -ffast-math -flto -funroll-loops -pthread -Wno-strict-prototypes',
    'GCC_OPTIMIZATION_LEVEL' => '3',
    # Add audio and threading optimization flags - enable USE_TFLITE for neural post-filter
    'GCC_PREPROCESSOR_DEFINITIONS' => 'MA_NO_RUNTIME_LINKING=1 NDEBUG=1 _REENTRANT=1 USE_TFLITE=1',
    'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../src $(PODS_TARGET_SRCROOT)/../prebuilt/include',
    'LIBRARY_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../prebuilt/macos',
    'OTHER_LDFLAGS' => '-lLiteRt -Wl,-rpath,@executable_path/../Frameworks -Wl,-rpath,@loader_path',
    # Universal binary support
    'EXCLUDED_ARCHS[sdk=macosx*]' => ''
  }

  # Also set on user target
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=macosx*]' => ''
  }

  # Embed the dylib as a resource so it gets copied to the framework
  s.resource = '../prebuilt/macos/libLiteRt.dylib'

  # Add a script phase to copy the dylib to the Frameworks directory
  s.script_phase = {
    :name => 'Embed LiteRT Library',
    :execution_position => :after_compile,
    :script => <<-SCRIPT
      echo "Embedding libLiteRt.dylib into framework"
      DYLIB_PATH="${PODS_TARGET_SRCROOT}/../prebuilt/macos/libLiteRt.dylib"
      FRAMEWORK_DIR="${BUILT_PRODUCTS_DIR}/${PRODUCT_NAME}.framework/Versions/A"

      if [ -f "$DYLIB_PATH" ]; then
        mkdir -p "$FRAMEWORK_DIR"
        cp -f "$DYLIB_PATH" "$FRAMEWORK_DIR/"
        echo "Copied libLiteRt.dylib to $FRAMEWORK_DIR"

        # Update install name to use @loader_path (relative to framework binary)
        # This ensures that when flutter_recorder is loaded, it looks for libLiteRt.dylib in the same dir
        install_name_tool -id "@loader_path/libLiteRt.dylib" "$FRAMEWORK_DIR/libLiteRt.dylib"

        # Re-sign the dylib
        codesign --force --sign - "$FRAMEWORK_DIR/libLiteRt.dylib"
      else
        echo "WARNING: libLiteRt.dylib not found at $DYLIB_PATH"
      fi
SCRIPT
  }

  s.swift_version = '5.0'
  s.framework  = ['CoreAudio', 'AudioToolbox', 'AVFoundation']
end
