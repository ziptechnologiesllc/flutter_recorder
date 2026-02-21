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
  s.dependency 'Flutter'

  # LiteRT for iOS: use prebuilt binary (built from google-ai-edge/LiteRT via Bazel)
  # Static library avoids code signing issues with dynamic frameworks on iOS
  # Falls back to TensorFlowLiteC CocoaPod if prebuilt not available
  prebuilt_static = File.exist?(File.join(__dir__, '..', 'prebuilt', 'ios', 'libLiteRt.a'))
  prebuilt_dylib = File.exist?(File.join(__dir__, '..', 'prebuilt', 'ios', 'libLiteRt.dylib'))

  if prebuilt_static
    s.vendored_libraries = '../prebuilt/ios/libLiteRt.a'
    s.pod_target_xcconfig = {
      'DEFINES_MODULE' => 'YES',
      'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
      "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
      'OTHER_CFLAGS' => '-O3 -ffast-math -funroll-loops -pthread',
      'OTHER_CPLUSPLUSFLAGS' => '-O3 -ffast-math -funroll-loops -pthread',
      'GCC_OPTIMIZATION_LEVEL' => '3',
      'GCC_PREPROCESSOR_DEFINITIONS' => 'MA_NO_RUNTIME_LINKING=1 NDEBUG=1 _REENTRANT=1 USE_TFLITE=1',
      'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../src $(PODS_TARGET_SRCROOT)/../prebuilt/include',
      'LIBRARY_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../prebuilt/ios',
      'OTHER_LDFLAGS' => '-lLiteRt'
    }
    # Force-load the static archive in the final app binary.
    # With use_frameworks! :linkage => :static, pod_target_xcconfig linker flags
    # don't propagate to the app target. user_target_xcconfig ensures the archive
    # is force-loaded during the final link step.
    s.user_target_xcconfig = {
      'OTHER_LDFLAGS' => '-force_load $(SRCROOT)/../plugins/flutter_recorder/prebuilt/ios/libLiteRt.a'
    }
  elsif prebuilt_dylib
    s.vendored_libraries = '../prebuilt/ios/libLiteRt.dylib'
    s.pod_target_xcconfig = {
      'DEFINES_MODULE' => 'YES',
      'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
      "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
      'OTHER_CFLAGS' => '-O3 -ffast-math -funroll-loops -pthread',
      'OTHER_CPLUSPLUSFLAGS' => '-O3 -ffast-math -funroll-loops -pthread',
      'GCC_OPTIMIZATION_LEVEL' => '3',
      'GCC_PREPROCESSOR_DEFINITIONS' => 'MA_NO_RUNTIME_LINKING=1 NDEBUG=1 _REENTRANT=1 USE_TFLITE=1',
      'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../src $(PODS_TARGET_SRCROOT)/../prebuilt/include',
      'LIBRARY_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../prebuilt/ios',
      'OTHER_LDFLAGS' => '-lLiteRt -Wl,-rpath,@executable_path/Frameworks'
    }
  else
    # Fallback: use TensorFlowLiteC CocoaPod (legacy TFLite C API)
    # Note: This provides tensorflow/lite/c/ headers, NOT litert/c/ headers
    # Neural post-filter may not compile without litert/c/ headers
    s.dependency 'TensorFlowLiteC'
    s.pod_target_xcconfig = {
      'DEFINES_MODULE' => 'YES',
      'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
      "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
      'OTHER_CFLAGS' => '-O3 -ffast-math -funroll-loops -pthread',
      'OTHER_CPLUSPLUSFLAGS' => '-O3 -ffast-math -funroll-loops -pthread',
      'GCC_OPTIMIZATION_LEVEL' => '3',
      'GCC_PREPROCESSOR_DEFINITIONS' => 'MA_NO_RUNTIME_LINKING=1 NDEBUG=1 _REENTRANT=1 USE_TFLITE=1',
      'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../src'
    }
  end

  s.platform = :ios, '13.0'
  s.swift_version = '5.0'
  s.ios.framework  = ['CoreAudio', 'AudioToolbox', 'AVFoundation']
end
