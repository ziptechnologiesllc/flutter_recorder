// Tests for SPSCQueue.
//
// Run via the audio_engine/tests CMakeLists; or compile standalone:
//   c++ -std=c++17 -O2 -pthread spsc_queue_test.cpp -o spsc_queue_test

#include "../spsc_queue.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <type_traits>

using flowstate::audio_engine::SPSCQueue;

namespace {

struct TestItem {
  std::uint64_t seq;
  std::uint64_t payload;
};
static_assert(std::is_trivially_copyable<TestItem>::value,
              "TestItem must be trivially copyable");

void check(bool ok, const char* label) {
  if (!ok) {
    std::fprintf(stderr, "[FAIL] %s\n", label);
    std::exit(1);
  }
}

void test_basic_fill_drain() {
  SPSCQueue<int, 8> q;
  check(q.empty(), "starts empty");
  // Capacity-1 = 7 usable slots.
  for (int i = 0; i < 7; ++i) {
    check(q.push(i), "push within capacity");
  }
  check(!q.push(99), "push fails when full");
  int out = -1;
  for (int i = 0; i < 7; ++i) {
    check(q.pop(&out), "pop succeeds");
    check(out == i, "FIFO ordering");
  }
  check(!q.pop(&out), "pop fails when empty");
  check(q.empty(), "back to empty");
  std::puts("[PASS] basic fill/drain");
}

void test_wrap_around() {
  SPSCQueue<int, 4> q;  // 3 usable slots
  // Many fill/drain cycles to exercise wrap-around.
  for (int round = 0; round < 100000; ++round) {
    for (int i = 0; i < 3; ++i) {
      check(q.push(round * 1000 + i), "push during wrap round");
    }
    int out = -1;
    for (int i = 0; i < 3; ++i) {
      check(q.pop(&out), "pop during wrap round");
      check(out == round * 1000 + i, "FIFO across wrap");
    }
  }
  std::puts("[PASS] wrap-around (100k rounds)");
}

void test_multi_threaded_stress() {
  constexpr std::size_t kCap = 1024;
  constexpr std::uint64_t kOps = 10ULL * 1000ULL * 1000ULL;  // 10M
  SPSCQueue<TestItem, kCap> q;

  std::atomic<bool> done{false};
  std::atomic<std::uint64_t> produced{0};
  std::atomic<std::uint64_t> consumed{0};

  const auto t0 = std::chrono::steady_clock::now();

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kOps; ++i) {
      TestItem item{i, i * 1234567ULL ^ 0xdeadbeefULL};
      while (!q.push(item)) {
        // Spin; producer makes progress when consumer drains.
      }
      produced.fetch_add(1, std::memory_order_relaxed);
    }
    done.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    TestItem item;
    std::uint64_t expected = 0;
    while (true) {
      if (q.pop(&item)) {
        check(item.seq == expected, "consumer FIFO ordering");
        check(item.payload == (expected * 1234567ULL ^ 0xdeadbeefULL),
              "payload integrity");
        ++expected;
        consumed.fetch_add(1, std::memory_order_relaxed);
      } else if (done.load(std::memory_order_acquire) && q.empty()) {
        check(expected == kOps, "consumed all items");
        return;
      }
    }
  });

  producer.join();
  consumer.join();

  const auto t1 = std::chrono::steady_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  std::printf(
      "[PASS] multi-threaded stress: %llu ops in %lld ms (%.1f Mops/s)\n",
      static_cast<unsigned long long>(consumed.load()),
      static_cast<long long>(ms),
      static_cast<double>(kOps) / 1000.0 / static_cast<double>(ms));
}

}  // namespace

int main() {
  test_basic_fill_drain();
  test_wrap_around();
  test_multi_threaded_stress();
  std::puts("All SPSC queue tests passed.");
  return 0;
}
