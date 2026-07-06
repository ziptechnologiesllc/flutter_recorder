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

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*.{h,mm,c}'
  s.dependency 'FlutterMacOS'

  # LiteRT disabled on macOS for now — no prebuilt Intel/ARM binary yet

  # NOTE: ma_ios_notification_handler=fr_ma_ios_notification_handler renames
  # miniaudio's Obj-C notification handler class to avoid colliding with
  # flutter_soloud's embedded miniaudio copy. src/miniaudio.h is kept stock —
  # the rename lives in the build config only (also in macos/CMakeLists.txt
  # and ios/flutter_recorder.podspec).
  s.platform = :osx, '14.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
    'OTHER_CFLAGS' => '-O3 -ffast-math -flto -funroll-loops -pthread -Wno-strict-prototypes -fvisibility=hidden',
    'OTHER_CPLUSPLUSFLAGS' => '-O3 -ffast-math -flto -funroll-loops -pthread -Wno-strict-prototypes -fvisibility=hidden -fvisibility-inlines-hidden',
    'GCC_OPTIMIZATION_LEVEL' => '3',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) MA_NO_RUNTIME_LINKING=1 NDEBUG=1 _REENTRANT=1 ma_ios_notification_handler=fr_ma_ios_notification_handler AEC_DEBUG_LOGGING=1',
    'HEADER_SEARCH_PATHS' => '$(inherited) $(PODS_TARGET_SRCROOT)/../src',
    'EXCLUDED_ARCHS[sdk=macosx*]' => ''
  }

  # Preserve FFI symbols using CMake-built static lib with hidden visibility.
  # Both plugins embed miniaudio and shared analyzer/fft code — hidden visibility
  # on the CMake lib prevents duplicate symbol conflicts with flutter_soloud.
  plugin_root = '${PODS_ROOT}/../Flutter/ephemeral/.symlinks/plugins/flutter_recorder/macos'

  s.script_phase = {
    :name => 'Build flutter_recorder with CMake',
    :script => 'bash "${PODS_TARGET_SRCROOT}/build_cmake.sh"',
    :execution_position => :before_compile,
    :output_files => ['$(PODS_TARGET_SRCROOT)/cmake_build/macosx/libflutter_recorder_plugin.a'],
  }

  s.user_target_xcconfig = {
    'OTHER_LDFLAGS' => "$(inherited) -force_load #{plugin_root}/cmake_build/macosx/libflutter_recorder_plugin.a -lc++",
    'LIBRARY_SEARCH_PATHS' => "$(inherited) \"#{plugin_root}/cmake_build/macosx\"",
    'STRIP_STYLE' => 'debugging',
    'DEBUG_INFORMATION_FORMAT' => 'dwarf-with-dsym',
    'EXCLUDED_ARCHS[sdk=macosx*]' => ''
  }

  s.swift_version = '5.0'
  s.framework  = ['CoreAudio', 'AudioToolbox', 'AVFoundation']
end
