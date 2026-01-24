# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(flutter_recorder_library VERSION 0.0.1 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fft)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow_headers)

# LiteRT source directory (git submodule)
set(LITERT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/LiteRT/litert")

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

  # LiteRT paths
  set(LITERT_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/android/${ANDROID_ABI}")
  set(LITERT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_build_android")
  set(LITERT_OUTPUT_LIB "${LITERT_BUILD_DIR}/c/libLiteRtRuntimeCApi.so")
  set(HOST_FLATC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/LiteRT/host_flatc_build")

  if (EXISTS "${LITERT_PREBUILT_DIR}/libLiteRt.so")
    # Use prebuilt LiteRT
    message(STATUS "[flutter_recorder] Using prebuilt LiteRT for Android ${ANDROID_ABI}")
    set(LITERT_LIB "${LITERT_PREBUILT_DIR}/libLiteRt.so")
    set(LITERT_FOUND TRUE)

  elseif (EXISTS "${LITERT_SOURCE_DIR}/CMakeLists.txt")
    # Build LiteRT from source for Android
    message(STATUS "[flutter_recorder] Building LiteRT from source for Android ${ANDROID_ABI}")

    # Build host flatc first if needed
    if (NOT EXISTS "${HOST_FLATC_DIR}/_deps/flatbuffers-build/flatc")
      message(STATUS "[flutter_recorder] Building host flatc compiler...")
      file(MAKE_DIRECTORY "${HOST_FLATC_DIR}")
      file(WRITE "${HOST_FLATC_DIR}/CMakeLists.txt" "cmake_minimum_required(VERSION 3.16)\nproject(HostFlatc LANGUAGES C CXX)\ninclude(FetchContent)\nFetchContent_Declare(flatbuffers GIT_REPOSITORY https://github.com/google/flatbuffers.git GIT_TAG v25.2.10)\nset(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\nFetchContent_MakeAvailable(flatbuffers)\n")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -S "${HOST_FLATC_DIR}" -B "${HOST_FLATC_DIR}" -DCMAKE_BUILD_TYPE=Release
        WORKING_DIRECTORY "${HOST_FLATC_DIR}"
        RESULT_VARIABLE FLATC_CFG_RESULT
      )
      if (NOT FLATC_CFG_RESULT EQUAL 0)
        message(WARNING "[flutter_recorder] Failed to configure host flatc")
      else()
        execute_process(
          COMMAND ${CMAKE_COMMAND} --build "${HOST_FLATC_DIR}" --target flatc
          WORKING_DIRECTORY "${HOST_FLATC_DIR}"
          RESULT_VARIABLE FLATC_BUILD_RESULT
        )
      endif()
    endif()

    ExternalProject_Add(litert_external_android
      SOURCE_DIR "${LITERT_SOURCE_DIR}"
      BINARY_DIR "${LITERT_BUILD_DIR}"
      CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_PLATFORM=${ANDROID_PLATFORM}
        -DCMAKE_SYSTEM_NAME=Android
        -DCMAKE_BUILD_TYPE=Release
        -DLITERT_AUTO_BUILD_TFLITE=ON
        -DLITERT_ENABLE_GPU=ON
        -DLITERT_ENABLE_NPU=ON
        -DLITERT_BUILD_TOOLS=OFF
        -DLITERT_BUILD_TESTS=OFF
        -DTFLITE_HOST_TOOLS_DIR=${HOST_FLATC_DIR}/_deps/flatbuffers-build
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target litert_runtime_c_api_shared_lib -j
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS "${LITERT_OUTPUT_LIB}"
    )

    add_library(litert_imported_android SHARED IMPORTED)
    set_target_properties(litert_imported_android PROPERTIES
      IMPORTED_LOCATION "${LITERT_OUTPUT_LIB}"
      IMPORTED_NO_SONAME TRUE
    )
    add_dependencies(litert_imported_android litert_external_android)

    set(LITERT_LIB litert_imported_android)
    set(LITERT_FOUND TRUE)
    set(LITERT_BUILT_FROM_SOURCE TRUE)

    # Cache built library and GPU accelerator if present
    set(LITERT_GPU_LIB "${LITERT_BUILD_DIR}/libLiteRtGpuAccelerator.so")
    ExternalProject_Add_Step(litert_external_android copy_to_prebuilt
      DEPENDEES build
      COMMAND ${CMAKE_COMMAND} -E make_directory "${LITERT_PREBUILT_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LITERT_OUTPUT_LIB}" "${LITERT_PREBUILT_DIR}/libLiteRt.so"
      COMMAND ${CMAKE_COMMAND} -E true  # Placeholder for conditional copy
      COMMENT "Caching built LiteRT library for Android ${ANDROID_ABI}"
    )

  else()
    message(STATUS "[flutter_recorder] LiteRT source not found at ${LITERT_SOURCE_DIR}")
    message(STATUS "[flutter_recorder] Initialize submodule: git submodule update --init third_party/LiteRT")
    message(STATUS "[flutter_recorder] Neural post-filter will be disabled")
    set(LITERT_FOUND FALSE)
  endif()

  # Link LiteRT if found
  if (LITERT_FOUND)
    target_include_directories(flutter_recorder SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder log android EGL GLESv3 ${LITERT_LIB})
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE)

    if (LITERT_BUILT_FROM_SOURCE)
      add_dependencies(flutter_recorder litert_external_android)
    endif()

    # Copy libLiteRt.so to output for packaging
    if (NOT LITERT_BUILT_FROM_SOURCE)
      add_custom_command(TARGET flutter_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${LITERT_PREBUILT_DIR}/libLiteRt.so"
          $<TARGET_FILE_DIR:flutter_recorder>)
    else()
      add_custom_command(TARGET flutter_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${LITERT_OUTPUT_LIB}"
          "$<TARGET_FILE_DIR:flutter_recorder>/libLiteRt.so")
    endif()

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
  endif()
endif()

# iOS: TFLite linked via CocoaPods (TensorFlowLiteC) - podspec handles linking
# macOS: TFLite linked via CocoaPods (vendored dylib) - podspec handles linking

if (WIN32)
  # Windows: use prebuilt LiteRT or build from source
  # Determine architecture suffix
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
    set(WIN_ARCH "arm64")
  else()
    set(WIN_ARCH "x64")
  endif()

  set(LITERT_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/windows/${WIN_ARCH}")
  set(LITERT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_build_windows")
  set(LITERT_OUTPUT_LIB "${LITERT_BUILD_DIR}/c/LiteRtRuntimeCApi.dll")
  set(LITERT_OUTPUT_IMPLIB "${LITERT_BUILD_DIR}/c/LiteRtRuntimeCApi.lib")

  if (EXISTS "${LITERT_PREBUILT_DIR}/LiteRt.dll")
    # Use prebuilt LiteRT
    message(STATUS "[flutter_recorder] Using prebuilt LiteRT for Windows ${WIN_ARCH}")
    target_include_directories(flutter_recorder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder "${LITERT_PREBUILT_DIR}/LiteRt.lib")
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE USE_LITERT)
    add_custom_command(TARGET flutter_recorder POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LITERT_PREBUILT_DIR}/LiteRt.dll"
        $<TARGET_FILE_DIR:flutter_recorder>)
    set(LITERT_FOUND TRUE)

  elseif (EXISTS "${LITERT_SOURCE_DIR}/CMakeLists.txt")
    # Build LiteRT from source for Windows
    message(STATUS "[flutter_recorder] Building LiteRT from source for Windows ${WIN_ARCH}")

    ExternalProject_Add(litert_external_windows
      SOURCE_DIR "${LITERT_SOURCE_DIR}"
      BINARY_DIR "${LITERT_BUILD_DIR}"
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DLITERT_AUTO_BUILD_TFLITE=ON
        -DLITERT_ENABLE_GPU=OFF
        -DLITERT_ENABLE_NPU=OFF
        -DLITERT_BUILD_TOOLS=OFF
        -DLITERT_BUILD_TESTS=OFF
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target litert_runtime_c_api_shared_lib
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS "${LITERT_OUTPUT_LIB}" "${LITERT_OUTPUT_IMPLIB}"
    )

    add_library(litert_imported_windows SHARED IMPORTED)
    set_target_properties(litert_imported_windows PROPERTIES
      IMPORTED_LOCATION "${LITERT_OUTPUT_LIB}"
      IMPORTED_IMPLIB "${LITERT_OUTPUT_IMPLIB}"
    )
    add_dependencies(litert_imported_windows litert_external_windows)

    target_include_directories(flutter_recorder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder litert_imported_windows)
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE USE_LITERT)
    add_dependencies(flutter_recorder litert_external_windows)

    add_custom_command(TARGET flutter_recorder POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LITERT_OUTPUT_LIB}"
        "$<TARGET_FILE_DIR:flutter_recorder>/LiteRt.dll")

    # Cache built library
    ExternalProject_Add_Step(litert_external_windows copy_to_prebuilt
      DEPENDEES build
      COMMAND ${CMAKE_COMMAND} -E make_directory "${LITERT_PREBUILT_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LITERT_OUTPUT_LIB}" "${LITERT_PREBUILT_DIR}/LiteRt.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LITERT_OUTPUT_IMPLIB}" "${LITERT_PREBUILT_DIR}/LiteRt.lib"
      COMMENT "Caching built LiteRT library for Windows ${WIN_ARCH}"
    )
    set(LITERT_FOUND TRUE)

  elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/windows/tensorflowlite_c.dll")
    # Fallback to legacy TFLite (old path without arch subfolder)
    message(STATUS "[flutter_recorder] Using legacy TFLite for Windows")
    target_include_directories(flutter_recorder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/windows/tensorflowlite_c.lib")
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE)
    add_custom_command(TARGET flutter_recorder POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/windows/tensorflowlite_c.dll"
        $<TARGET_FILE_DIR:flutter_recorder>)
    set(LITERT_FOUND TRUE)

  else()
    message(STATUS "[flutter_recorder] LiteRT not found for Windows ${WIN_ARCH}")
    message(STATUS "[flutter_recorder] Initialize submodule: git submodule update --init third_party/LiteRT")
    message(STATUS "[flutter_recorder] Neural post-filter will be disabled")
    set(LITERT_FOUND FALSE)
  endif()
endif()

if (UNIX AND NOT APPLE AND NOT ANDROID)
  # Linux: use prebuilt LiteRT, build from source, or fallback to TFLite
  # Determine architecture suffix
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(LINUX_ARCH "arm64")
  else()
    set(LINUX_ARCH "x64")
  endif()

  set(LITERT_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/linux/${LINUX_ARCH}")
  set(LITERT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_build")
  set(LITERT_OUTPUT_LIB "${LITERT_BUILD_DIR}/c/libLiteRtRuntimeCApi.so")

  # Also check legacy path without arch subfolder
  if (NOT EXISTS "${LITERT_PREBUILT_DIR}/libLiteRt.so" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/linux/libLiteRt.so")
    set(LITERT_PREBUILT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/linux")
  endif()

  if (EXISTS "${LITERT_PREBUILT_DIR}/libLiteRt.so")
    # Use prebuilt LiteRT
    message(STATUS "[flutter_recorder] Using prebuilt LiteRT for Linux ${LINUX_ARCH}")
    set(LITERT_LIB "${LITERT_PREBUILT_DIR}/libLiteRt.so")
    set(LITERT_FOUND TRUE)

  elseif (EXISTS "${LITERT_SOURCE_DIR}/CMakeLists.txt")
    # Build LiteRT from source using ExternalProject
    message(STATUS "[flutter_recorder] Building LiteRT from source (this may take several minutes on first build)")

    ExternalProject_Add(litert_external
      SOURCE_DIR "${LITERT_SOURCE_DIR}"
      BINARY_DIR "${LITERT_BUILD_DIR}"
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DLITERT_AUTO_BUILD_TFLITE=ON
        -DLITERT_ENABLE_GPU=OFF
        -DLITERT_ENABLE_NPU=OFF
        -DLITERT_DISABLE_KLEIDIAI=ON
        -DLITERT_BUILD_TOOLS=OFF
        -DLITERT_BUILD_TESTS=OFF
        -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--gc-sections
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target litert_runtime_c_api_shared_lib -j
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS "${LITERT_OUTPUT_LIB}"
    )

    # Create imported target for the built library
    add_library(litert_imported SHARED IMPORTED)
    set_target_properties(litert_imported PROPERTIES
      IMPORTED_LOCATION "${LITERT_OUTPUT_LIB}"
      IMPORTED_NO_SONAME TRUE
    )
    add_dependencies(litert_imported litert_external)

    set(LITERT_LIB litert_imported)
    set(LITERT_FOUND TRUE)
    set(LITERT_BUILT_FROM_SOURCE TRUE)

    # Copy the built library to prebuilt dir for future builds
    set(LITERT_CACHE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/linux/${LINUX_ARCH}")
    ExternalProject_Add_Step(litert_external copy_to_prebuilt
      DEPENDEES build
      COMMAND ${CMAKE_COMMAND} -E make_directory "${LITERT_CACHE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LITERT_OUTPUT_LIB}" "${LITERT_CACHE_DIR}/libLiteRt.so"
      COMMENT "Caching built LiteRT library for Linux ${LINUX_ARCH}"
    )

  elseif (EXISTS "${LITERT_PREBUILT_DIR}/libtensorflowlite_c.so")
    # Fallback to legacy TFLite
    message(STATUS "[flutter_recorder] Using prebuilt TFLite for Linux (legacy)")
    target_include_directories(flutter_recorder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder "${LITERT_PREBUILT_DIR}/libtensorflowlite_c.so")
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE)
    set(LITERT_FOUND FALSE)

  else()
    message(STATUS "[flutter_recorder] LiteRT source not found at ${LITERT_SOURCE_DIR}")
    message(STATUS "[flutter_recorder] Initialize submodule: git submodule update --init third_party/LiteRT")
    message(STATUS "[flutter_recorder] Neural post-filter will be disabled")
    set(LITERT_FOUND FALSE)
  endif()

  # Link LiteRT if found
  if (LITERT_FOUND)
    target_include_directories(flutter_recorder SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/include")
    target_link_libraries(flutter_recorder ${LITERT_LIB})
    target_compile_definitions(flutter_recorder PRIVATE USE_TFLITE USE_LITERT)

    if (LITERT_BUILT_FROM_SOURCE)
      add_dependencies(flutter_recorder litert_external)
    endif()

    # Copy libLiteRt.so to output for packaging
    if (NOT LITERT_BUILT_FROM_SOURCE)
      add_custom_command(TARGET flutter_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${LITERT_PREBUILT_DIR}/libLiteRt.so"
          $<TARGET_FILE_DIR:flutter_recorder>)
    else()
      add_custom_command(TARGET flutter_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${LITERT_OUTPUT_LIB}"
          "$<TARGET_FILE_DIR:flutter_recorder>/libLiteRt.so")
    endif()

    # GPU accelerator (WebGPU/Vulkan, loaded dynamically at runtime)
    if (EXISTS "${LITERT_PREBUILT_DIR}/libLiteRtGpuAccelerator.so")
      message(STATUS "[flutter_recorder] GPU accelerator available (WebGPU/Vulkan)")
      target_compile_definitions(flutter_recorder PRIVATE USE_LITERT_GPU)
      add_custom_command(TARGET flutter_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${LITERT_PREBUILT_DIR}/libLiteRtGpuAccelerator.so"
          $<TARGET_FILE_DIR:flutter_recorder>)
    endif()
  endif()
endif()
