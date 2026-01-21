# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(flutter_recorder_library VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fft)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow_headers)

add_library(flutter_recorder SHARED
  "flutter_recorder.cpp"
  "analyzer.cpp"
  "capture.cpp"
  "soloud_slave_bridge.cpp"
  "fft/soloud_fft.cpp"
  "filters/filters.cpp"
  "filters/autogain.cpp"
  "filters/echo_cancellation.cpp"
  "filters/aec/reference_buffer.cpp"
  "filters/aec/adaptive_echo_cancellation.cpp"
  "filters/aec/calibration.cpp"
  "filters/aec/neural_post_filter.cpp"
  "filters/aec/aec_test.cpp"
  "filters/aec/vss_nlms_filter.cpp"
  "filters/aec/delay_estimator.cpp"
)

set_target_properties(flutter_recorder PROPERTIES
  PUBLIC_HEADER flutter_recorder.h
  OUTPUT_NAME "flutter_recorder"
)

target_compile_definitions(flutter_recorder PUBLIC DART_SHARED_LIB)

if (WIN32)
  target_compile_definitions(flutter_recorder PRIVATE _IS_WIN_)
endif()

if (ANDROID)
  # Support Android 15 16k page size.
  target_link_options(flutter_recorder PRIVATE "-Wl,-z,max-page-size=16384,-O2")

  # LiteRT prebuilt library location
  set(LITERT_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/android/${ANDROID_ABI}")

  if (EXISTS "${LITERT_PREBUILT_DIR}/libLiteRt.so")
    message(STATUS "[flutter_recorder] Using prebuilt LiteRT for Android ${ANDROID_ABI}")

    # Include LiteRT headers and OpenCL headers (for GPU support)
    target_include_directories(flutter_recorder SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")

    # Link LiteRT runtime and Android libs
    target_link_libraries(flutter_recorder log android EGL GLESv3 "${LITERT_PREBUILT_DIR}/libLiteRt.so")
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE)

    # Copy libLiteRt.so to output for packaging
    add_custom_command(TARGET flutter_recorder POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LITERT_PREBUILT_DIR}/libLiteRt.so"
        $<TARGET_FILE_DIR:flutter_recorder>)

    # GPU accelerator (loaded dynamically at runtime)
    if (EXISTS "${LITERT_PREBUILT_DIR}/libLiteRtGpuAccelerator.so")
      message(STATUS "[flutter_recorder] GPU accelerator available")
      target_compile_definitions(flutter_recorder PRIVATE USE_LITERT_GPU)
      add_custom_command(TARGET flutter_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${LITERT_PREBUILT_DIR}/libLiteRtGpuAccelerator.so"
          $<TARGET_FILE_DIR:flutter_recorder>)
    endif()

    # NPU accelerator - Qualcomm QNN HTP (loaded dynamically at runtime)
    if (EXISTS "${LITERT_PREBUILT_DIR}/libQnnHtp.so")
      message(STATUS "[flutter_recorder] Qualcomm NPU accelerator available")
      target_compile_definitions(flutter_recorder PRIVATE USE_LITERT_NPU)
      add_custom_command(TARGET flutter_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${LITERT_PREBUILT_DIR}/libQnnHtp.so"
          $<TARGET_FILE_DIR:flutter_recorder>)
    endif()

  else()
    message(STATUS "[flutter_recorder] LiteRT prebuilt not found at ${LITERT_PREBUILT_DIR}")
    message(STATUS "[flutter_recorder] Run scripts/build_litert_android.sh to build LiteRT")
    message(STATUS "[flutter_recorder] Neural post-filter will be disabled")
  endif()
endif()

# iOS: TFLite linked via CocoaPods (TensorFlowLiteC) - podspec handles linking
# macOS: TFLite linked via CocoaPods (vendored dylib) - podspec handles linking

if (WIN32)
  # Windows: use prebuilt TFLite DLL
  set(TFLITE_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/windows")
  if (EXISTS "${TFLITE_PREBUILT_DIR}/tensorflowlite_c.dll")
    target_include_directories(flutter_recorder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder "${TFLITE_PREBUILT_DIR}/tensorflowlite_c.lib")
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE)
    # Copy DLL to output directory
    add_custom_command(TARGET flutter_recorder POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TFLITE_PREBUILT_DIR}/tensorflowlite_c.dll"
        $<TARGET_FILE_DIR:flutter_recorder>)
    message(STATUS "[flutter_recorder] Using prebuilt TFLite for Windows")
  else()
    message(STATUS "[flutter_recorder] TFLite prebuilt not found at ${TFLITE_PREBUILT_DIR}, neural post-filter disabled")
    message(STATUS "[flutter_recorder] Run scripts/build_tflite_desktop.sh to build TFLite")
  endif()
endif()

if (UNIX AND NOT APPLE AND NOT ANDROID)
  # Linux: use prebuilt TFLite shared library
  set(TFLITE_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/linux")
  if (EXISTS "${TFLITE_PREBUILT_DIR}/libtensorflowlite_c.so")
    target_include_directories(flutter_recorder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder "${TFLITE_PREBUILT_DIR}/libtensorflowlite_c.so")
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE)
    message(STATUS "[flutter_recorder] Using prebuilt TFLite for Linux")
  else()
    message(STATUS "[flutter_recorder] TFLite prebuilt not found at ${TFLITE_PREBUILT_DIR}, neural post-filter disabled")
    message(STATUS "[flutter_recorder] Run scripts/build_tflite_desktop.sh to build TFLite")
  endif()
endif()
