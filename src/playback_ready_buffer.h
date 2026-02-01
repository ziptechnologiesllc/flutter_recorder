#ifndef PLAYBACK_READY_BUFFER_H
#define PLAYBACK_READY_BUFFER_H

#include <atomic>
#include <cstring>
#include <vector>

/**
 * PlaybackReadyBuffer: A pre-deinterleaved audio buffer for instant SoLoud playback.
 *
 * This buffer stores audio in SoLoud's native planar format (all samples for channel 0,
 * then all samples for channel 1, etc.) during recording. This eliminates the O(n)
 * deinterleaving step that normally occurs when loading audio into SoLoud, enabling
 * truly instant playback when recording stops.
 *
 * ARCHITECTURE:
 * ============
 * Audio flows through two parallel paths:
 *
 *   Audio Callback (interleaved input)
 *       │
 *       ├──> Ring Buffer (interleaved) ──> WAV file (async worker)
 *       │
 *       └──> PlaybackReadyBuffer (planar) ──> Instant SoLoud playback
 *
 * THREAD SAFETY:
 * =============
 * - write() is called from audio thread - lock-free, O(frames) per call
 * - startRecording() is called from audio thread - atomic state transition
 * - finalizeAndGetPlanar() is called from audio thread - atomic, returns ready data
 *
 * MEMORY MODEL:
 * ============
 * - Pre-allocated buffers sized for max recording duration (same as ring buffer)
 * - No allocations during audio streaming
 * - Double buffering not needed - single owner (audio thread) during recording
 */
class PlaybackReadyBuffer {
public:
    PlaybackReadyBuffer() = default;

    /**
     * Configure the buffer for a given capacity and format.
     * Must be called before use. Safe to call when not recording.
     *
     * @param maxFrames Maximum frames to store (typically 10 min @ sample rate)
     * @param channels Number of audio channels (1=mono, 2=stereo)
     * @param sampleRate Sample rate in Hz (for duration calculations)
     */
    void configure(size_t maxFrames, unsigned int channels, unsigned int sampleRate) {
        mMaxFrames = maxFrames;
        mChannels = channels;
        mSampleRate = sampleRate;

        // Pre-allocate planar buffers
        // Each channel gets its own contiguous buffer
        mChannelBuffers.resize(channels);
        for (unsigned int ch = 0; ch < channels; ++ch) {
            mChannelBuffers[ch].resize(maxFrames, 0.0f);
        }

        mFrameCount.store(0, std::memory_order_relaxed);
        mRecordingActive.store(false, std::memory_order_relaxed);
    }

    /**
     * Start recording. Resets frame counter and marks buffer as active.
     * Called from audio thread when StartRecording event fires.
     */
    void startRecording() {
        mFrameCount.store(0, std::memory_order_relaxed);
        mRecordingActive.store(true, std::memory_order_release);
    }

    /**
     * Write interleaved audio, deinterleaving on-the-fly to planar format.
     * Called from audio thread in every audio callback.
     *
     * @param interleaved Input audio in interleaved format [L0 R0 L1 R1 ...]
     * @param frames Number of frames (not samples) to write
     *
     * Performance: O(frames * channels) per call, ~256 samples @ 128 frame buffer
     */
    void write(const float* interleaved, size_t frames) {
        if (!mRecordingActive.load(std::memory_order_acquire)) {
            return;
        }

        size_t writePos = mFrameCount.load(std::memory_order_relaxed);

        // Check capacity
        if (writePos + frames > mMaxFrames) {
            // Recording exceeds capacity - stop accepting data
            // (Ring buffer handles the same case similarly)
            return;
        }

        // Deinterleave on-the-fly: [L0 R0 L1 R1 ...] -> [L0 L1 ...] [R0 R1 ...]
        for (size_t f = 0; f < frames; ++f) {
            for (unsigned int ch = 0; ch < mChannels; ++ch) {
                mChannelBuffers[ch][writePos + f] = interleaved[f * mChannels + ch];
            }
        }

        mFrameCount.store(writePos + frames, std::memory_order_release);
    }

    /**
     * Finalize recording and return planar data ready for SoLoud.
     * Called from audio thread when StopRecording event fires.
     *
     * This method:
     * 1. Marks recording as inactive
     * 2. Applies crossfade to prevent clicks
     * 3. Returns pointers to planar data
     *
     * @param outFrameCount Output: number of frames recorded
     * @return Pointer to first channel's data (channel pointers are contiguous)
     *
     * IMPORTANT: The returned pointer is valid until the next startRecording() call.
     */
    float* finalizeAndGetPlanar(size_t* outFrameCount) {
        mRecordingActive.store(false, std::memory_order_release);

        size_t frameCount = mFrameCount.load(std::memory_order_acquire);
        if (frameCount == 0) {
            if (outFrameCount) *outFrameCount = 0;
            return nullptr;
        }

        // Apply crossfade to prevent clicks (same as ring buffer: 2.5ms)
        applyCrossfade(frameCount);

        if (outFrameCount) *outFrameCount = frameCount;

        // Return pointer to channel 0 data
        // Caller can access other channels via getChannelData()
        return mChannelBuffers[0].data();
    }

    /**
     * Get pointer to a specific channel's data.
     * Valid after finalizeAndGetPlanar() has been called.
     */
    float* getChannelData(unsigned int channel) {
        if (channel >= mChannels) return nullptr;
        return mChannelBuffers[channel].data();
    }

    /**
     * Get the current frame count.
     * Useful for determining recording duration.
     */
    size_t getFrameCount() const {
        return mFrameCount.load(std::memory_order_acquire);
    }

    /**
     * Check if recording is currently active.
     */
    bool isRecording() const {
        return mRecordingActive.load(std::memory_order_acquire);
    }

    /**
     * Get the number of channels this buffer is configured for.
     */
    unsigned int channels() const { return mChannels; }

    /**
     * Get the sample rate this buffer is configured for.
     */
    unsigned int sampleRate() const { return mSampleRate; }

    /**
     * Reset the buffer state without reallocating.
     */
    void reset() {
        mRecordingActive.store(false, std::memory_order_release);
        mFrameCount.store(0, std::memory_order_release);
    }

private:
    /**
     * Apply crossfade to prevent clicks at loop boundaries.
     * Modifies the planar data in-place.
     */
    void applyCrossfade(size_t frameCount) {
        const float crossfadeDurationMs = 2.5f;
        size_t crossfadeFrames = static_cast<size_t>(mSampleRate * crossfadeDurationMs / 1000.0f);
        crossfadeFrames = std::min(crossfadeFrames, frameCount / 2);

        if (crossfadeFrames == 0) return;

        // Apply fade-in at start and fade-out at end for each channel
        for (unsigned int ch = 0; ch < mChannels; ++ch) {
            float* data = mChannelBuffers[ch].data();

            // Fade-in at start
            for (size_t f = 0; f < crossfadeFrames; ++f) {
                float gain = static_cast<float>(f) / static_cast<float>(crossfadeFrames);
                data[f] *= gain;
            }

            // Fade-out at end
            size_t fadeOutStart = frameCount - crossfadeFrames;
            for (size_t f = fadeOutStart; f < frameCount; ++f) {
                float gain = static_cast<float>(frameCount - f) / static_cast<float>(crossfadeFrames);
                data[f] *= gain;
            }
        }
    }

    // Planar storage: one vector per channel
    std::vector<std::vector<float>> mChannelBuffers;

    // Configuration
    size_t mMaxFrames = 0;
    unsigned int mChannels = 2;
    unsigned int mSampleRate = 48000;

    // Recording state (atomic for thread safety)
    std::atomic<size_t> mFrameCount{0};
    std::atomic<bool> mRecordingActive{false};
};

// Global playback ready buffer instance (mirrors g_nativeRingBuffer pattern)
extern PlaybackReadyBuffer* g_playbackReadyBuffer;

#endif // PLAYBACK_READY_BUFFER_H
