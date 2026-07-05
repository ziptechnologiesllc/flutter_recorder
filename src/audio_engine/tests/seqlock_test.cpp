// Tests for Seqlock.
//
// The most important test: under heavy writer churn, no reader ever sees a
// "torn" struct where fields disagree with an internal invariant.

#include "../seqlock.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <type_traits>

using flowstate::audio_engine::Seqlock;

namespace {

// Payload chosen to be wider than any atomic primitive on common platforms.
// Invariant maintained by writer: a == b * 2, c == b * 3, d == b * 4.
struct WidePayload {
  std::uint64_t a;
  std::uint64_t b;
  std::uint64_t c;
  std::uint64_t d;
};
static_assert(std::is_trivially_copyable<WidePayload>::value, "");

void check(bool ok, const char* label) {
  if (!ok) {
    std::fprintf(stderr, "[FAIL] %s\n", label);
    std::exit(1);
  }
}

bool invariantHolds(const WidePayload& v) {
  return v.a == v.b * 2 && v.c == v.b * 3 && v.d == v.b * 4;
}

void test_basic_store_load() {
  Seqlock<WidePayload> sl;
  sl.store({10, 5, 15, 20});
  const WidePayload r = sl.load();
  check(r.a == 10 && r.b == 5 && r.c == 15 && r.d == 20,
        "basic store/load round trip");
  std::puts("[PASS] basic store/load");
}

void test_no_torn_reads_single_reader() {
  Seqlock<WidePayload> sl;
  // Prime with a valid invariant value.
  sl.store({2, 1, 3, 4});

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> reads{0};
  std::atomic<bool> tornDetected{false};

  std::thread writer([&] {
    std::uint64_t b = 1;
    while (!stop.load(std::memory_order_acquire)) {
      sl.store({b * 2, b, b * 3, b * 4});
      ++b;
    }
  });

  std::thread reader([&] {
    while (!stop.load(std::memory_order_acquire)) {
      const WidePayload v = sl.load();
      if (!invariantHolds(v)) {
        tornDetected.store(true, std::memory_order_release);
        return;
      }
      reads.fetch_add(1, std::memory_order_relaxed);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  stop.store(true, std::memory_order_release);
  writer.join();
  reader.join();

  check(!tornDetected.load(), "no torn reads under single reader");
  std::printf("[PASS] no torn reads (single reader): %llu reads in 1500 ms\n",
              static_cast<unsigned long long>(reads.load()));
}

void test_no_torn_reads_multi_reader() {
  Seqlock<WidePayload> sl;
  sl.store({2, 1, 3, 4});

  std::atomic<bool> stop{false};
  std::atomic<bool> tornDetected{false};
  std::atomic<std::uint64_t> totalReads{0};

  std::thread writer([&] {
    std::uint64_t b = 1;
    while (!stop.load(std::memory_order_acquire)) {
      sl.store({b * 2, b, b * 3, b * 4});
      ++b;
    }
  });

  auto readerFn = [&] {
    while (!stop.load(std::memory_order_acquire)) {
      const WidePayload v = sl.load();
      if (!invariantHolds(v)) {
        tornDetected.store(true, std::memory_order_release);
        return;
      }
      totalReads.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::thread r1(readerFn);
  std::thread r2(readerFn);
  std::thread r3(readerFn);
  std::thread r4(readerFn);

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  stop.store(true, std::memory_order_release);
  writer.join();
  r1.join();
  r2.join();
  r3.join();
  r4.join();

  check(!tornDetected.load(), "no torn reads under multi-reader");
  std::printf(
      "[PASS] no torn reads (4 readers): %llu total reads in 1500 ms\n",
      static_cast<unsigned long long>(totalReads.load()));
}

void test_sequence_monotonic() {
  Seqlock<WidePayload> sl;
  const std::uint64_t s0 = sl.sequenceForTesting();
  sl.store({2, 1, 3, 4});
  const std::uint64_t s1 = sl.sequenceForTesting();
  sl.store({4, 2, 6, 8});
  const std::uint64_t s2 = sl.sequenceForTesting();
  check(s1 > s0 && s2 > s1, "sequence monotonic");
  check((s1 & 1u) == 0u && (s2 & 1u) == 0u, "sequence even after store");
  std::puts("[PASS] sequence monotonic and even after store");
}

}  // namespace

int main() {
  test_basic_store_load();
  test_sequence_monotonic();
  test_no_torn_reads_single_reader();
  test_no_torn_reads_multi_reader();
  std::puts("All Seqlock tests passed.");
  return 0;
}
