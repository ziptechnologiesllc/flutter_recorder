#ifndef AEC_REFERENCE_BUFFER_H
#define AEC_REFERENCE_BUFFER_H

#include <atomic>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <chrono>

/**
 * Lock-free Single-Producer Single-Consumer (SPSC) ring buffer for AEC reference signal.
 *
 * SoLoud audio thread writes playback samples (producer).
 * flutter_recorder audio thread reads reference samples (consumer).
 *
 * This buffer stores the audio being played through speakers so the AEC filter
 * can subtract the estimated echo from the microphone input.
 *
 * TIMESTAMP SYNCHRONIZATION:
 * The key challenge is aligning two unsynchronized audio threads.
 * We track monotonic timestamps for writes, allowing the reader to find
 * samples corresponding to a specific point in time.
 */
class AECReferenceBuffer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;

    /**
     * @param sizeInFrames Buffer size in frames (samples per channel)
     * @param channels Number of audio channels (1=mono, 2=stereo)
     * @param sampleRate Sample rate in Hz (needed for timestamp calculations)
     */
    AECReferenceBuffer(size_t sizeInFrames, unsigned int channels, unsigned int sampleRate = 48000)
        : mChannels(channels),
          mSampleRate(sampleRate),
          mSizeInFrames(sizeInFrames),
          mBuffer(sizeInFrames * channels, 0.0f),
          mWritePos(0),
          mReadPos(0),
          mFramesWritten(0),
          mLastWriteTimestampUs(0),
          mLastWriteFrameCount(0),
          mReferenceTimePoint(Clock::now()) {}

    /**
     * Write audio frames to the buffer (called from SoLoud audio thread).
     * Also records the timestamp of this write for synchronization.
     *
     * @param data Interleaved float samples
     * @param frameCount Number of frames to write
     */
    void write(const float* data, size_t frameCount) {
        if (data == nullptr || frameCount == 0) return;

        size_t samplesToWrite = frameCount * mChannels;
        size_t writePos = mWritePos.load(std::memory_order_relaxed);

        for (size_t i = 0; i < samplesToWrite; ++i) {
            mBuffer[(writePos + i) % mBuffer.size()] = data[i];
        }

        // Update write position atomically
        mWritePos.store((writePos + samplesToWrite) % mBuffer.size(),
                        std::memory_order_release);

        // Track total frames written (for sync purposes)
        size_t totalFrames = mFramesWritten.fetch_add(frameCount, std::memory_order_relaxed) + frameCount;

        // Record timestamp for this write
        // We store the timestamp as microseconds relative to a fixed reference point
        mLastWriteTimestampUs.store(toMicroseconds(Clock::now()), std::memory_order_relaxed);
        mLastWriteFrameCount.store(totalFrames, std::memory_order_release);
    }

    /**
     * Read a single sample from the buffer with delay offset.
     * Used by NLMS filter to get the reference signal aligned with mic input.
     *
     * @param delaySamples Number of samples back from current write position
     * @return The sample value, or 0.0f if not enough data
     */
    float readDelayed(size_t delaySamples) const {
        size_t writePos = mWritePos.load(std::memory_order_acquire);
        size_t bufferSize = mBuffer.size();

        // Calculate read position with delay
        size_t readPos = (writePos + bufferSize - delaySamples) % bufferSize;

        return mBuffer[readPos];
    }

    /**
     * Read multiple frames from the buffer for batch processing.
     *
     * @param dest Destination buffer for interleaved samples
     * @param frameCount Number of frames to read
     * @param delaySamples Delay offset in samples (not frames)
     * @return Number of frames actually read
     */
    size_t readFrames(float* dest, size_t frameCount, size_t delaySamples) const {
        if (dest == nullptr || frameCount == 0) return 0;

        size_t writePos = mWritePos.load(std::memory_order_acquire);
        size_t bufferSize = mBuffer.size();
        size_t samplesToRead = frameCount * mChannels;

        // Calculate read start position with delay
        // Mic block covers time [T, T+N). Echo in mic sample at time T came from ref at time T-D.
        // writePos ≈ T+N (end of current audio). So ref from T-D is at writePos - N - D.
        size_t readPos = (writePos + bufferSize - delaySamples - samplesToRead) % bufferSize;

        for (size_t i = 0; i < samplesToRead; ++i) {
            dest[i] = mBuffer[(readPos + i) % bufferSize];
        }

        return frameCount;
    }

    /**
     * Read frames aligned to a specific timestamp (for AEC synchronization).
     *
     * This method uses the timestamp of the last write to calculate which samples
     * in the buffer correspond to a given point in time. This allows proper
     * alignment between the mic input and reference signal even when the two
     * audio threads run asynchronously.
     *
     * @param dest Destination buffer for interleaved samples
     * @param frameCount Number of frames to read
     * @param targetTime The timestamp of the mic samples we want to align with
     * @param calibratedDelayMs The calibrated acoustic delay in milliseconds
     * @return Number of frames actually read, or 0 if not enough data
     */
    size_t readFramesAtTimestamp(float* dest, size_t frameCount,
                                  TimePoint targetTime, float calibratedDelayMs) const {
        if (dest == nullptr || frameCount == 0) return 0;

        // Get the last write timestamp and frame count
        size_t lastFrameCount = mLastWriteFrameCount.load(std::memory_order_acquire);
        int64_t lastWriteUs = mLastWriteTimestampUs.load(std::memory_order_relaxed);
        TimePoint lastWriteTime = fromMicroseconds(lastWriteUs);

        // Debug: check if buffer has been written to
        static int debugCounter = 0;
        if (++debugCounter % 1000 == 1) {
            printf("[RefBuf DEBUG] lastWriteUs=%lld lastFrameCount=%zu\n",
                   lastWriteUs, lastFrameCount);
            fflush(stdout);
        }

        // If no data has been written yet, return 0
        if (lastWriteUs == 0) {
            return 0;
        }

        // Calculate how far back in time we need to go
        // targetTime = when mic received the sound
        // We need reference samples from: targetTime - calibratedDelay (when sound was mixed)
        // The calibratedDelay includes: output buffer + acoustic path + input buffer
        auto delayDuration = std::chrono::milliseconds(static_cast<int64_t>(calibratedDelayMs));
        TimePoint referenceTime = targetTime - delayDuration;

        // Calculate time difference from last write
        auto timeSinceLastWrite = std::chrono::duration_cast<std::chrono::microseconds>(
            lastWriteTime - referenceTime);

        // Convert time difference to frames
        // If referenceTime is before lastWriteTime, we need to go back in the buffer
        int64_t frameOffset = (timeSinceLastWrite.count() * static_cast<int64_t>(mSampleRate)) / 1000000;

        // Check if the data is within our buffer range
        size_t bufferSize = mBuffer.size();
        size_t bufferSizeFrames = bufferSize / mChannels;

        if (debugCounter % 1000 == 2) {
            printf("[RefBuf DEBUG] timeSinceLastWrite=%lldus frameOffset=%lld lastFrameCount=%zu bufferFrames=%zu\n",
                   timeSinceLastWrite.count(), frameOffset, lastFrameCount, bufferSizeFrames);
            fflush(stdout);
        }

        if (frameOffset < 0) {
            // Reference time is in the future - data not yet available
            // Fill with zeros
            std::memset(dest, 0, frameCount * mChannels * sizeof(float));
            return 0;
        }

        // Check against BOTH buffer capacity and actual written data
        size_t requiredFrames = static_cast<size_t>(frameOffset) + frameCount;
        if (requiredFrames > bufferSizeFrames) {
            // Data is too old, no longer in buffer (wrapped around)
            std::memset(dest, 0, frameCount * mChannels * sizeof(float));
            return 0;
        }

        if (requiredFrames > lastFrameCount) {
            // Not enough data has been written yet
            if (debugCounter % 1000 == 3) {
                printf("[RefBuf DEBUG] Not enough data: need %zu frames but only %zu written\n",
                       requiredFrames, lastFrameCount);
                fflush(stdout);
            }
            std::memset(dest, 0, frameCount * mChannels * sizeof(float));
            return 0;
        }

        // Calculate read position
        size_t currentWritePos = mWritePos.load(std::memory_order_acquire);
        size_t offsetSamples = static_cast<size_t>(frameOffset) * mChannels;
        size_t samplesToRead = frameCount * mChannels;

        size_t readPos = (currentWritePos + bufferSize - offsetSamples - samplesToRead) % bufferSize;

        // Debug: calculate energy before and after read position
        float readEnergy = 0.0f;
        for (size_t i = 0; i < samplesToRead; ++i) {
            dest[i] = mBuffer[(readPos + i) % bufferSize];
            readEnergy += dest[i] * dest[i];
        }

        // Print debug info periodically
        if (debugCounter % 1000 == 5) {
            float avgReadEnergy = samplesToRead > 0 ? readEnergy / samplesToRead : 0;
            printf("[RefBuf READ] writePos=%zu readPos=%zu offset=%zu samples=%zu avgEnergy=%.6f\n",
                   currentWritePos, readPos, offsetSamples, samplesToRead, avgReadEnergy);
            fflush(stdout);
        }

        return frameCount;
    }

    /**
     * Get the current timestamp (for mic callback to use).
     */
    static TimePoint now() {
        return Clock::now();
    }

    /**
     * Get the number of available frames in the buffer.
     * Note: This is approximate due to concurrent access.
     */
    size_t available() const {
        return mFramesWritten.load(std::memory_order_relaxed);
    }

    /**
     * Reset the buffer state.
     */
    void reset() {
        std::fill(mBuffer.begin(), mBuffer.end(), 0.0f);
        mWritePos.store(0, std::memory_order_release);
        mReadPos.store(0, std::memory_order_release);
        mFramesWritten.store(0, std::memory_order_release);
    }

    /**
     * Get the buffer size in frames.
     */
    size_t sizeInFrames() const { return mSizeInFrames; }

    /**
     * Get the number of channels.
     */
    unsigned int channels() const { return mChannels; }

    /**
     * Read recent samples for calibration analysis.
     * Reads the most recent N samples from the buffer (mono, channel 0 only).
     *
     * @param dest Destination buffer for mono samples
     * @param sampleCount Number of samples to read
     * @return Number of samples actually read
     */
    size_t readForCalibration(float* dest, size_t sampleCount) const {
        if (dest == nullptr || sampleCount == 0) return 0;

        size_t writePos = mWritePos.load(std::memory_order_acquire);
        size_t bufferSize = mBuffer.size();

        // Calculate how many samples are available
        size_t available = std::min(sampleCount * mChannels, bufferSize);
        size_t samplesToRead = (sampleCount < available / mChannels) ? sampleCount : available / mChannels;

        // Read position: go back from write position
        size_t startPos = (writePos + bufferSize - samplesToRead * mChannels) % bufferSize;

        // Copy samples (extract channel 0 only for mono output)
        for (size_t i = 0; i < samplesToRead; ++i) {
            size_t srcIdx = (startPos + i * mChannels) % bufferSize;
            dest[i] = mBuffer[srcIdx];
        }

        return samplesToRead;
    }

private:
    unsigned int mChannels;
    unsigned int mSampleRate;
    size_t mSizeInFrames;
    std::vector<float> mBuffer;

    // Atomic positions for lock-free operation
    std::atomic<size_t> mWritePos;
    std::atomic<size_t> mReadPos;
    std::atomic<size_t> mFramesWritten;

    // Timestamp tracking for synchronization
    // We store microseconds since a reference point instead of TimePoint directly
    // because std::atomic<TimePoint> isn't reliably lock-free
    std::atomic<int64_t> mLastWriteTimestampUs{0};
    std::atomic<size_t> mLastWriteFrameCount;
    TimePoint mReferenceTimePoint;  // Fixed reference for converting to/from microseconds

    // Helper to convert TimePoint to microseconds relative to reference
    int64_t toMicroseconds(TimePoint tp) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            tp - mReferenceTimePoint).count();
    }

    // Helper to convert microseconds back to TimePoint
    TimePoint fromMicroseconds(int64_t us) const {
        return mReferenceTimePoint + std::chrono::microseconds(us);
    }
};

// Global reference buffer pointer (set during initialization)
extern AECReferenceBuffer* g_aecReferenceBuffer;

#endif // AEC_REFERENCE_BUFFER_H
