// Single-producer / single-consumer lock-free queue.
//
// One thread pushes, one thread pops. Push and pop are wait-free; both bound
// their work to a fixed number of instructions. Capacity must be a power of
// two so head/tail can wrap with a bitmask. T must be trivially copyable.
//
// One slot is reserved to disambiguate full from empty, so usable capacity is
// Capacity - 1.
//
// Memory ordering:
//   producer: release-store on mHead after writing the slot
//   consumer: acquire-load on mHead before reading the slot
//   producer: acquire-load on mTail to detect full
//   consumer: release-store on mTail after consuming the slot
//
// Head and tail live on separate cache lines to avoid false sharing between
// producer and consumer CPUs.

#ifndef FLOWSTATE_AUDIO_ENGINE_SPSC_QUEUE_H_
#define FLOWSTATE_AUDIO_ENGINE_SPSC_QUEUE_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

namespace flowstate {
namespace audio_engine {

template <typename T, std::size_t Capacity>
class SPSCQueue {
  static_assert(Capacity >= 2, "Capacity must be >= 2");
  static_assert((Capacity & (Capacity - 1)) == 0,
                "Capacity must be a power of two");
  static_assert(std::is_trivially_copyable<T>::value,
                "T must be trivially copyable for lock-free safety");

 public:
  using value_type = T;
  static constexpr std::size_t capacity = Capacity;

  SPSCQueue() = default;
  SPSCQueue(const SPSCQueue&) = delete;
  SPSCQueue& operator=(const SPSCQueue&) = delete;
  SPSCQueue(SPSCQueue&&) = delete;
  SPSCQueue& operator=(SPSCQueue&&) = delete;

  // Producer-only. Returns false if the queue is full.
  bool push(const T& item) noexcept {
    const std::size_t head = mHead.load(std::memory_order_relaxed);
    const std::size_t next = (head + 1) & kMask;
    if (next == mTail.load(std::memory_order_acquire)) {
      return false;  // full
    }
    mBuf[head] = item;
    mHead.store(next, std::memory_order_release);
    return true;
  }

  // Consumer-only. Returns false if the queue is empty.
  bool pop(T* out) noexcept {
    const std::size_t tail = mTail.load(std::memory_order_relaxed);
    if (tail == mHead.load(std::memory_order_acquire)) {
      return false;  // empty
    }
    *out = mBuf[tail];
    mTail.store((tail + 1) & kMask, std::memory_order_release);
    return true;
  }

  // Either side may call. Snapshot is best-effort under contention.
  bool empty() const noexcept {
    return mHead.load(std::memory_order_acquire) ==
           mTail.load(std::memory_order_acquire);
  }

  // Diagnostic, not authoritative under contention.
  std::size_t approxSize() const noexcept {
    const std::size_t h = mHead.load(std::memory_order_acquire);
    const std::size_t t = mTail.load(std::memory_order_acquire);
    return (h - t) & kMask;
  }

 private:
  static constexpr std::size_t kMask = Capacity - 1;
  static constexpr std::size_t kCacheLine = 64;

  alignas(kCacheLine) std::atomic<std::size_t> mHead{0};
  alignas(kCacheLine) std::atomic<std::size_t> mTail{0};
  alignas(kCacheLine) std::array<T, Capacity> mBuf{};
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_SPSC_QUEUE_H_
