// Sequence-lock for publishing a trivially-copyable struct from a single
// writer to one or more readers, lock-free.
//
// Writer is wait-free: each store completes in bounded time and never blocks.
// Readers are lock-free but not wait-free: a reader retries if the writer
// races. In our deployment the writer runs at most once per audio buffer
// (~5.3 ms) and readers run at 120 Hz, so retries are vanishingly rare.
//
// Protocol:
//   - mSeq counts versions. Even => quiescent. Odd => write in progress.
//   - Writer: bump to odd, fence, copy, fence, bump to even.
//   - Reader: read seq, retry if odd; copy; re-read seq; retry if changed.
//
// The two fences ensure the data writes/reads can't be reordered past the
// sequence transitions.

#ifndef FLOWSTATE_AUDIO_ENGINE_SEQLOCK_H_
#define FLOWSTATE_AUDIO_ENGINE_SEQLOCK_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace flowstate {
namespace audio_engine {

template <typename T>
class Seqlock {
  static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable for lock-free safety");

 public:
  Seqlock() = default;
  Seqlock(const Seqlock&) = delete;
  Seqlock& operator=(const Seqlock&) = delete;
  Seqlock(Seqlock&&) = delete;
  Seqlock& operator=(Seqlock&&) = delete;

  // Writer (audio thread). Wait-free.
  void store(const T& value) noexcept {
    const std::uint64_t seq = mSeq.load(std::memory_order_relaxed);
    // Mark "write in progress" (odd) and ensure any subsequent data write is
    // ordered after this.
    mSeq.store(seq + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);

    // Bit-copy. Using memcpy avoids any chance of implicit operator= calls
    // on types with deleted/non-trivial assignment.
    std::memcpy(&mData, &value, sizeof(T));

    // Mark "write complete" (even).
    std::atomic_thread_fence(std::memory_order_release);
    mSeq.store(seq + 2, std::memory_order_release);
  }

  // Reader (any thread). Lock-free; retries on writer race.
  T load() const noexcept {
    T value;
    for (;;) {
      const std::uint64_t s1 = mSeq.load(std::memory_order_acquire);
      if ((s1 & 1u) != 0u) {
        // Writer in progress; spin.
        continue;
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      std::memcpy(&value, &mData, sizeof(T));
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint64_t s2 = mSeq.load(std::memory_order_acquire);
      if (s1 == s2) {
        return value;
      }
      // Writer raced; retry.
    }
  }

  // For tests/diagnostics. Returns a monotonically-increasing version.
  std::uint64_t sequenceForTesting() const noexcept {
    return mSeq.load(std::memory_order_acquire);
  }

 private:
  static constexpr std::size_t kCacheLine = 64;

  alignas(kCacheLine) std::atomic<std::uint64_t> mSeq{0};
  alignas(kCacheLine) T mData{};
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_SEQLOCK_H_
