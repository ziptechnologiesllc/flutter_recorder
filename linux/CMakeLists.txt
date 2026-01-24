# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

# Project-level configuration.
set(PROJECT_NAME "flutter_recorder")
project(${PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if (UNIX AND NOT WIN32 AND NOT APPLE)
	if (CMAKE_SIZEOF_VOID_P MATCHES "8")
		set (LIB_POSTFIX "64" CACHE STRING "suffix for 32/64 dir placement")
		mark_as_advanced (LIB_POSTFIX)
	endif ()
endif ()
if (MSVC)
	add_definitions (-D_CRT_SECURE_NO_WARNINGS)
endif()
if (NOT DEFINED LIB_POSTFIX)
  set (LIB_POSTFIX "")
endif ()

# list( APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wno-error -fPIC")
# message("*************************************************1 ${CMAKE_CXX_FLAGS} *************************************************")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -Wall -Wno-error -fPIC -msse2 -msse -mavx -mavx2 -mfma")
# message("*************************************************2 ${CMAKE_CXX_FLAGS} *************************************************")

# Invoke the build for native code shared with the other target platforms.
# This can be changed to accommodate different builds.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/shared")

# Determine architecture for prebuilt path
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
  set(LINUX_ARCH "arm64")
else()
  set(LINUX_ARCH "x64")
endif()

# List of absolute paths to libraries that should be bundled with the plugin.
set(LITERT_PREBUILT_ARCH "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/linux/${LINUX_ARCH}/libLiteRt.so")
set(LITERT_PREBUILT_LEGACY "${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/linux/libLiteRt.so")

# Find LiteRT library (prefer arch-specific, fallback to legacy path)
if (EXISTS "${LITERT_PREBUILT_ARCH}")
  set(LITERT_PREBUILT "${LITERT_PREBUILT_ARCH}")
elseif (EXISTS "${LITERT_PREBUILT_LEGACY}")
  set(LITERT_PREBUILT "${LITERT_PREBUILT_LEGACY}")
else()
  set(LITERT_PREBUILT "")
endif()

# Bundle flutter_recorder and LiteRT
if (LITERT_PREBUILT)
  set(flutter_recorder_bundled_libraries
    $<TARGET_FILE:flutter_recorder>
    "${LITERT_PREBUILT}"
    PARENT_SCOPE
  )
else()
  set(flutter_recorder_bundled_libraries
    $<TARGET_FILE:flutter_recorder>
    PARENT_SCOPE
  )
endif()
