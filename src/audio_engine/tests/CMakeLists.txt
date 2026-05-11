# Standalone test target for the audio_engine primitives.
#
# Usage (from repo root):
#   cmake -S plugins/flutter_recorder/src/audio_engine/tests \
#         -B build/audio_engine_tests
#   cmake --build build/audio_engine_tests
#   ctest --test-dir build/audio_engine_tests --output-on-failure
#
# These tests do not link against the plugin's shared library; they only
# include the header-only primitives, so they're cheap to build and safe to
# run in CI without any audio hardware.

cmake_minimum_required(VERSION 3.10)
project(flowstate_audio_engine_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# Optional sanitizers — enable with -DENABLE_SANITIZERS=ON for catching
# data races and undefined behavior the tests might otherwise miss.
option(ENABLE_SANITIZERS "Build tests with TSan + UBSan" OFF)
if(ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=thread,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread,undefined)
endif()

add_executable(spsc_queue_test spsc_queue_test.cpp)
target_link_libraries(spsc_queue_test PRIVATE Threads::Threads)
target_compile_options(spsc_queue_test PRIVATE -Wall -Wextra -Wpedantic -O2)

add_executable(seqlock_test seqlock_test.cpp)
target_link_libraries(seqlock_test PRIVATE Threads::Threads)
target_compile_options(seqlock_test PRIVATE -Wall -Wextra -Wpedantic -O2)

# Engine simulation tests — drive AudioEngine::process() with synthetic
# buffers and assert on snapshot + drained events. Links audio_engine.cpp
# directly (not via the plugin shared library).
add_executable(audio_engine_test
  audio_engine_test.cpp
  ../audio_engine.cpp
  ../local_clock.cpp
  ../inference.cpp
)
target_include_directories(audio_engine_test PRIVATE ..)
target_link_libraries(audio_engine_test PRIVATE Threads::Threads)
target_compile_options(audio_engine_test PRIVATE -Wall -Wextra -Wpedantic -O2)

add_executable(local_clock_test
  local_clock_test.cpp
  ../local_clock.cpp
)
target_include_directories(local_clock_test PRIVATE ..)
target_link_libraries(local_clock_test PRIVATE Threads::Threads)
target_compile_options(local_clock_test PRIVATE -Wall -Wextra -Wpedantic -O2)

add_executable(inference_test
  inference_test.cpp
  ../inference.cpp
)
target_include_directories(inference_test PRIVATE ..)
target_compile_options(inference_test PRIVATE -Wall -Wextra -Wpedantic -O2)

enable_testing()
add_test(NAME spsc_queue   COMMAND spsc_queue_test)
add_test(NAME seqlock      COMMAND seqlock_test)
add_test(NAME audio_engine COMMAND audio_engine_test)
add_test(NAME local_clock  COMMAND local_clock_test)
add_test(NAME inference    COMMAND inference_test)
