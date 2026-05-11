cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME "flutter_recorder")
project(${PROJECT_NAME} LANGUAGES CXX C OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PLUGIN_NAME "${PROJECT_NAME}_plugin")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

# All source files (mirrors the unity build in Classes/flutter_recorder.mm)
set(PLUGIN_SOURCES
  "${SRC_DIR}/flutter_recorder.cpp"
  "${SRC_DIR}/analyzer.cpp"
  "${SRC_DIR}/audio_engine/audio_engine.cpp"
  "${SRC_DIR}/audio_engine/inference.cpp"
  "${SRC_DIR}/audio_engine/local_clock.cpp"
  "${SRC_DIR}/capture.cpp"
  "${SRC_DIR}/fft/soloud_fft.cpp"
  "${SRC_DIR}/filters/aec/adaptive_echo_cancellation.cpp"
  "${SRC_DIR}/filters/aec/aec_test.cpp"
  "${SRC_DIR}/filters/aec/calibration.cpp"
  "${SRC_DIR}/filters/aec/delay_estimator.cpp"
  "${SRC_DIR}/filters/aec/neural_post_filter.cpp"
  "${SRC_DIR}/filters/aec/reference_buffer.cpp"
  "${SRC_DIR}/filters/aec/vss_nlms_filter.cpp"
  "${SRC_DIR}/filters/autogain.cpp"
  "${SRC_DIR}/filters/echo_cancellation.cpp"
  "${SRC_DIR}/filters/filters.cpp"
  "${SRC_DIR}/native_ring_buffer.cpp"
  "${SRC_DIR}/native_scheduler.cpp"
  "${SRC_DIR}/soloud_slave_bridge.cpp"
)

add_library(${PLUGIN_NAME} STATIC ${PLUGIN_SOURCES})

# CRITICAL: Hidden visibility prevents miniaudio symbols from conflicting
# with flutter_soloud's copy when both are force-loaded into the app binary.
set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  C_VISIBILITY_PRESET hidden)

target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

# Rename shared symbols to avoid conflicts with flutter_soloud's copies
target_compile_definitions(${PLUGIN_NAME} PRIVATE
  "fftimpl=flutter_recorder_fftimpl"
  "waveData=flutter_recorder_waveData"
  "Analyzer=FlutterRecorderAnalyzer"
)

# LiteRT disabled on macOS (no prebuilt Intel/ARM binary yet)
# Neural post-filter code compiles but is guarded by #ifdef USE_TFLITE
target_compile_definitions(${PLUGIN_NAME} PRIVATE
  MA_NO_RUNTIME_LINKING=1
  NDEBUG=1
  _REENTRANT=1
)

target_include_directories(${PLUGIN_NAME} PRIVATE
  "${SRC_DIR}"
  "${SRC_DIR}/fft"
  "${SRC_DIR}/filters"
  "${SRC_DIR}/filters/aec"
)

# Optimization flags matching the podspec
target_compile_options(${PLUGIN_NAME} PRIVATE
  -O3 -ffast-math -funroll-loops -pthread
  -Wno-strict-prototypes -Wno-error
)

# Several source files include ObjC headers on Apple (miniaudio.h, Foundation.h)
# Compile them as ObjC++ to avoid "unknown type name NSString" errors.
set_source_files_properties(
  "${SRC_DIR}/capture.cpp"
  "${SRC_DIR}/flutter_recorder.cpp"
  "${SRC_DIR}/filters/aec/adaptive_echo_cancellation.cpp"
  PROPERTIES LANGUAGE OBJCXX
)

# Apple framework dependencies
find_library(COREAUDIO_FRAMEWORK CoreAudio)
find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox)
find_library(AVFOUNDATION_FRAMEWORK AVFoundation)
if(COREAUDIO_FRAMEWORK)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${COREAUDIO_FRAMEWORK})
endif()
if(AUDIOTOOLBOX_FRAMEWORK)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${AUDIOTOOLBOX_FRAMEWORK})
endif()
if(AVFOUNDATION_FRAMEWORK)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${AVFOUNDATION_FRAMEWORK})
endif()
