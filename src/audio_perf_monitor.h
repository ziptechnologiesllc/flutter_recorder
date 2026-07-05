#ifndef AUDIO_PERF_MONITOR_H
#define AUDIO_PERF_MONITOR_H

#include <atomic>
#include <chrono>
#include <cstdint>

/**
 * Lock-free audio thread performance monitor.
 * Tracks callback timing to detect buffer underruns and thread contention.
 *
 * All methods are audio-thread safe (no allocations, no locks, no blocking).
 */
class AudioPerfMonitor {
public:
    static AudioPerfMonitor& instance() {
        static AudioPerfMonitor inst;
        return inst;
    }

    // Call at START of audio callback
    void callbackStart() {
        auto now = std::chrono::steady_clock::now();

        // Track callback interval (time between callbacks)
        if (mLastCallbackStart.time_since_epoch().count() > 0) {
            auto interval = std::chrono::duration_cast<std::chrono::microseconds>(
                now - mLastCallbackStart).count();
            mLastIntervalUs.store(interval, std::memory_order_relaxed);

            // Update max interval (simple atomic max)
            int64_t current = mMaxIntervalUs.load(std::memory_order_relaxed);
            while (interval > current &&
                   !mMaxIntervalUs.compare_exchange_weak(current, interval,
                       std::memory_order_relaxed, std::memory_order_relaxed)) {}
        }

        mLastCallbackStart = now;
        mCallbackStartTime = now;
        mCallbackCount.fetch_add(1, std::memory_order_relaxed);
    }

    // Call at END of audio callback
    void callbackEnd() {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            now - mCallbackStartTime).count();

        mLastDurationUs.store(duration, std::memory_order_relaxed);

        // Update max duration
        int64_t current = mMaxDurationUs.load(std::memory_order_relaxed);
        while (duration > current &&
               !mMaxDurationUs.compare_exchange_weak(current, duration,
                   std::memory_order_relaxed, std::memory_order_relaxed)) {}

        // Track if we exceeded budget (callback took longer than buffer period)
        int64_t budget = mBudgetUs.load(std::memory_order_relaxed);
        if (budget > 0 && duration > budget) {
            mOverrunCount.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Set expected callback period (buffer size / sample rate * 1000000)
    void setBudget(int64_t budgetUs) {
        mBudgetUs.store(budgetUs, std::memory_order_relaxed);
    }

    // Reset all counters (call from main thread periodically)
    void reset() {
        mMaxDurationUs.store(0, std::memory_order_relaxed);
        mMaxIntervalUs.store(0, std::memory_order_relaxed);
        mOverrunCount.store(0, std::memory_order_relaxed);
        mCallbackCount.store(0, std::memory_order_relaxed);
    }

    // Getters (safe to call from any thread)
    int64_t getLastDurationUs() const {
        return mLastDurationUs.load(std::memory_order_relaxed);
    }
    int64_t getMaxDurationUs() const {
        return mMaxDurationUs.load(std::memory_order_relaxed);
    }
    int64_t getLastIntervalUs() const {
        return mLastIntervalUs.load(std::memory_order_relaxed);
    }
    int64_t getMaxIntervalUs() const {
        return mMaxIntervalUs.load(std::memory_order_relaxed);
    }
    int64_t getBudgetUs() const {
        return mBudgetUs.load(std::memory_order_relaxed);
    }
    uint64_t getOverrunCount() const {
        return mOverrunCount.load(std::memory_order_relaxed);
    }
    uint64_t getCallbackCount() const {
        return mCallbackCount.load(std::memory_order_relaxed);
    }

    // Calculate load percentage (last duration / budget * 100)
    float getLoadPercent() const {
        int64_t budget = mBudgetUs.load(std::memory_order_relaxed);
        if (budget <= 0) return 0.0f;
        int64_t duration = mLastDurationUs.load(std::memory_order_relaxed);
        return (float)duration / (float)budget * 100.0f;
    }

    // Calculate peak load percentage
    float getPeakLoadPercent() const {
        int64_t budget = mBudgetUs.load(std::memory_order_relaxed);
        if (budget <= 0) return 0.0f;
        int64_t maxDuration = mMaxDurationUs.load(std::memory_order_relaxed);
        return (float)maxDuration / (float)budget * 100.0f;
    }

private:
    AudioPerfMonitor() = default;

    std::chrono::steady_clock::time_point mLastCallbackStart{};
    std::chrono::steady_clock::time_point mCallbackStartTime{};

    std::atomic<int64_t> mLastDurationUs{0};
    std::atomic<int64_t> mMaxDurationUs{0};
    std::atomic<int64_t> mLastIntervalUs{0};
    std::atomic<int64_t> mMaxIntervalUs{0};
    std::atomic<int64_t> mBudgetUs{0};
    std::atomic<uint64_t> mOverrunCount{0};
    std::atomic<uint64_t> mCallbackCount{0};
};

#endif // AUDIO_PERF_MONITOR_H
