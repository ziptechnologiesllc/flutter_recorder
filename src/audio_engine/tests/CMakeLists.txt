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

enable_testing()
add_test(NAME spsc_queue COMMAND spsc_queue_test)
add_test(NAME seqlock    COMMAND seqlock_test)
