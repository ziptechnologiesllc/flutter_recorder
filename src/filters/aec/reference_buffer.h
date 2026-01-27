#ifndef AEC_REFERENCE_BUFFER_H
#define AEC_REFERENCE_BUFFER_H

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern void aecLog(const char *fmt, ...);

/**
 * Lock-free Single-Producer Single-Consumer (SPSC) ring buffer for AEC
 * reference signal.
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
  AECReferenceBuffer(size_t sizeInFrames, unsigned int channels,
                     unsigned int sampleRate = 48000)
      : mChannels(channels), mSampleRate(sampleRate),
        mSizeInFrames(sizeInFrames), mBuffer(sizeInFrames * channels, 0.0f),
        mWritePos(0), mReadPos(0), mFramesWritten(0), mLastWriteTimestampUs(0),
        mLastWriteFrameCount(0), mReferenceTimePoint(Clock::now()) {}

  /**
   * Reconfigure the buffer dimensions without reallocating.
   * This allows pre-allocating a max-size buffer and then configuring
   * it to match the actual device settings.
   *
   * @param sizeInFrames New size in frames
   * @param channels New channel count
   * @param sampleRate New sample rate
   * @return true if successful, false if requested size exceeds allocation
   */
  bool configure(size_t sizeInFrames, unsigned int channels,
                 unsigned int sampleRate) {
    size_t requiredSamples = sizeInFrames * channels;
    if (requiredSamples > mBuffer.size()) {
      // Need to reallocate - resize the buffer
      mBuffer.resize(requiredSamples, 0.0f);
    }

    mChannels = channels;
    mSampleRate = sampleRate;
    mSizeInFrames = sizeInFrames;

    // Reset positions
    mWritePos.store(0, std::memory_order_release);
    mReadPos.store(0, std::memory_order_release);
    mFramesWritten.store(0, std::memory_order_release);

    aecLog("[AEC RefBuf] Configured: %zu frames @ %uHz, %u channels\n",
           sizeInFrames, sampleRate, channels);
    return true;
  }

  /**
   * Enable or disable the reference buffer.
   * When disabled, writes are skipped to save CPU when AEC is not in use.
   */
  void setEnabled(bool enabled) {
    mEnabled.store(enabled, std::memory_order_release);
  }

  bool isEnabled() const {
    return mEnabled.load(std::memory_order_acquire);
  }

  /**
   * Write audio frames to the buffer (called from SoLoud audio thread).
   * Also records the timestamp of this write for synchronization.
   *
   * @param data Interleaved float samples
   * @param frameCount Number of frames to write
   */
  void write(const float *data, size_t frameCount) {
    // Skip writes if AEC is disabled
    if (!mEnabled.load(std::memory_order_relaxed))
      return;

    if (data == nullptr || frameCount == 0)
      return;

    size_t samplesToWrite = frameCount * mChannels;
    size_t writePos = mWritePos.load(std::memory_order_relaxed);

    // Copy data to ring buffer
    for (size_t i = 0; i < samplesToWrite; ++i) {
      mBuffer[(writePos + i) % mBuffer.size()] = data[i];
    }

    // During calibration: PROGRESSIVELY copy data as it arrives
    // This prevents the circular buffer from overwriting data before we save it
    if (mCalibrationActive.load(std::memory_order_relaxed)) {
      // ... (calibration specific logging removed to reduce noise if needed, or
      // kept) keeping logic minimal here
    }

    // PROGRESSIVE COPY: Append incoming data to calibration buffer
    // immediately This ensures we capture data before the circular buffer can
    // overwrite it
    if (!mCalibrationDataValid &&
        mCalibrationCapturedFrames < mCalibrationExpectedFrames) {
      size_t framesToAppend = std::min(
          frameCount, mCalibrationExpectedFrames - mCalibrationCapturedFrames);
      size_t oldSize = mCalibrationData.size();
      mCalibrationData.resize(oldSize + framesToAppend);

      // Extract channel 0 (mono) from interleaved data
      for (size_t i = 0; i < framesToAppend; ++i) {
        mCalibrationData[oldSize + i] = data[i * mChannels]; // Channel 0
      }

      mCalibrationCapturedFrames += frameCount;

      // Check if we've captured enough
      if (mCalibrationCapturedFrames >= mCalibrationExpectedFrames) {
        mCalibrationDataValid = true;

        // Calculate RMS of captured data
        float copyEnergy = 0.0f;
        for (size_t i = 0; i < mCalibrationData.size(); ++i) {
          copyEnergy += mCalibrationData[i] * mCalibrationData[i];
        }
        float copyRms = mCalibrationData.size() > 0
                            ? std::sqrt(copyEnergy / mCalibrationData.size())
                            : 0.0f;
        aecLog("[AEC RefBuf] PROGRESSIVE CAPTURE COMPLETE: %zu frames, "
               "RMS=%.4f\n",
               mCalibrationData.size(), copyRms);
      }
    }
    // Note: Don't increment mCalibrationCapturedFrames in else branch
    // - it's only meaningful during active calibration
    // - incrementing every write wastes cycles when AEC is disabled

    // Update write position atomically
    mWritePos.store((writePos + samplesToWrite) % mBuffer.size(),
                    std::memory_order_release);

    // Track total frames written (for sync purposes)
    size_t totalFrames =
        mFramesWritten.fetch_add(frameCount, std::memory_order_relaxed) +
        frameCount;

    // Record timestamp for this write
    // We store the timestamp as microseconds relative to a fixed reference
    // point
    mLastWriteTimestampUs.store(toMicroseconds(Clock::now()),
                                std::memory_order_relaxed);
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
  size_t readFrames(float *dest, size_t frameCount, size_t delaySamples) const {
    if (dest == nullptr || frameCount == 0)
      return 0;

    size_t writePos = mWritePos.load(std::memory_order_acquire);
    size_t bufferSize = mBuffer.size();
    size_t samplesToRead = frameCount * mChannels;

    // Calculate read start position with delay
    // Mic block covers time [T, T+N). Echo in mic sample at time T came from
    // ref at time T-D. writePos ≈ T+N (end of current audio). So ref from T-D
    // is at writePos - N - D.
    size_t readPos =
        (writePos + bufferSize - delaySamples - samplesToRead) % bufferSize;

    // Debug: log buffer positions periodically
    static int readDebugCount = 0;
    if (++readDebugCount <= 5 || readDebugCount % 500 == 0) {
      // Calculate energy of what we're about to read
      float energy = 0.0f;
      for (size_t i = 0; i < std::min(samplesToRead, (size_t)100); ++i) {
        float v = mBuffer[(readPos + i) % bufferSize];
        energy += v * v;
      }
      aecLog("[AEC RefBuf R] wPos=%zu rPos=%zu delay=%zu energy=%.5f\n",
             writePos, readPos, delaySamples, energy);
    }

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
  size_t readFramesAtTimestamp(float *dest, size_t frameCount,
                               TimePoint targetTime,
                               float calibratedDelayMs) const {
    if (dest == nullptr || frameCount == 0)
      return 0;

    // Get the last write timestamp and frame count
    size_t lastFrameCount =
        mLastWriteFrameCount.load(std::memory_order_acquire);
    int64_t lastWriteUs = mLastWriteTimestampUs.load(std::memory_order_relaxed);

    if (lastWriteUs == 0) {
      return 0;
    }

    TimePoint lastWriteTime = fromMicroseconds(lastWriteUs);

    // Calculate time elapsed since the last write operation
    auto timeSinceLastWrite = targetTime - lastWriteTime;
    double secondsSinceLastWrite =
        std::chrono::duration<double>(timeSinceLastWrite).count();

    // SMOOTH SYNCHRONIZATION:
    // Instead of anchoring to the jumpy mWritePos, we use the timestamp to
    // calculate exactly which sample corresponds to targetTime. Current
    // playback head (in frames) = lastFrameCount + (secondsSinceLastWrite *
    // mSampleRate)
    double currentPlaybackHead =
        static_cast<double>(lastFrameCount) +
        (secondsSinceLastWrite * static_cast<double>(mSampleRate));

    // Required reference head = currentPlaybackHead - delaySamples
    // Convert ms to samples: delay_samples = (delay_ms / 1000) * sample_rate
    double delaySamples = (static_cast<double>(calibratedDelayMs) / 1000.0) *
                          static_cast<double>(mSampleRate);
    double requiredRefHead = currentPlaybackHead - delaySamples;

    // We want to read frameCount frames ending at requiredRefHead (exclusive)
    // So the start frame is requiredRefHead - frameCount
    double startFrame = requiredRefHead - static_cast<double>(frameCount);

    // Convert absolute start frame to circular buffer index
    // Note: we use floor to keep it sample-aligned, but the double math
    // maintains the smooth continuous timeline.
    int64_t absoluteStartFrame = static_cast<int64_t>(std::floor(startFrame));

    // Safety check: if we are requesting frames that haven't been written or
    // are too old
    if (absoluteStartFrame < 0 ||
        (absoluteStartFrame + frameCount) >
            static_cast<int64_t>(currentPlaybackHead)) {
      // Not enough data or asking for future samples
      // For now, return 0 or fill with silence
      std::memset(dest, 0, frameCount * mChannels * sizeof(float));
      return 0;
    }

    size_t bufferSize = mBuffer.size();
    size_t bufferSizeFrames = bufferSize / mChannels;

    // Check if data is still in buffer (not overwritten)
    size_t totalWritten = mFramesWritten.load(std::memory_order_relaxed);
    if (totalWritten > bufferSizeFrames &&
        absoluteStartFrame <
            static_cast<int64_t>(totalWritten - bufferSizeFrames)) {
      // Data has been overwritten (buffer overflow for this delay)
      std::memset(dest, 0, frameCount * mChannels * sizeof(float));
      return 0;
    }

    // Map absolute frame to buffer index
    size_t readPosSamples =
        (static_cast<size_t>(absoluteStartFrame) % bufferSizeFrames) *
        mChannels;
    size_t samplesToRead = frameCount * mChannels;

    for (size_t i = 0; i < samplesToRead; ++i) {
      dest[i] = mBuffer[(readPosSamples + i) % bufferSize];
    }

    return frameCount;
  }

  /**
   * Get the current timestamp (for mic callback to use).
   */
  static TimePoint now() { return Clock::now(); }

  /**
   * Read frames at an absolute position in the output stream.
   *
   * This is the NEW SYNC METHOD that uses sample counters instead of timestamps.
   * The capture side calculates: refPosition = captureFrameCount - calibratedOffset
   *
   * @param dest Destination buffer for INTERLEAVED samples (frameCount * channels)
   * @param frameCount Number of frames to read
   * @param absoluteOutputFrame The absolute output frame position to read from
   * @return Number of frames actually read, or 0 if data unavailable
   */
  size_t readFramesAtPosition(float *dest, size_t frameCount,
                              size_t absoluteOutputFrame) const {
    if (dest == nullptr || frameCount == 0)
      return 0;

    size_t totalWritten = mFramesWritten.load(std::memory_order_acquire);
    size_t bufferSizeFrames = mSizeInFrames;
    size_t samplesToRead = frameCount * mChannels;

    // Check if the requested position is valid:
    // - Not in the future (absoluteOutputFrame + frameCount <= totalWritten)
    // - Not overwritten (absoluteOutputFrame >= totalWritten - bufferSizeFrames)

    // Check for future samples
    if (absoluteOutputFrame + frameCount > totalWritten) {
      static int futureWarnCount = 0;
      if (++futureWarnCount % 1000 == 1) {
        aecLog("[AEC RefBuf] readFramesAtPosition: requesting future data! "
               "requested=%zu+%zu, totalWritten=%zu\n",
               absoluteOutputFrame, frameCount, totalWritten);
      }
      // Fill with zeros - data not yet available
      std::memset(dest, 0, samplesToRead * sizeof(float));
      return 0;
    }

    // Check for overwritten samples (circular buffer overflow)
    if (totalWritten > bufferSizeFrames &&
        absoluteOutputFrame < totalWritten - bufferSizeFrames) {
      static int overwriteWarnCount = 0;
      if (++overwriteWarnCount % 1000 == 1) {
        aecLog("[AEC RefBuf] readFramesAtPosition: data overwritten! "
               "requested=%zu, oldest available=%zu\n",
               absoluteOutputFrame, totalWritten - bufferSizeFrames);
      }
      // Fill with zeros - data was overwritten
      std::memset(dest, 0, samplesToRead * sizeof(float));
      return 0;
    }

    // Map absolute frame position to circular buffer index
    // The buffer stores samples interleaved, so we need to handle channels
    size_t bufferIndex = (absoluteOutputFrame % bufferSizeFrames) * mChannels;
    size_t bufferSize = mBuffer.size();

    // Debug logging (sparse)
    static int posReadDebugCount = 0;
    if (++posReadDebugCount % 500 == 0) {
      aecLog("[AEC RefBuf PosR] absFrame=%zu bufIdx=%zu totalWritten=%zu\n",
             absoluteOutputFrame, bufferIndex, totalWritten);
    }

    // Read interleaved samples (matches legacy readFrames behavior)
    // The caller expects stereo interleaved data for stereo audio
    for (size_t i = 0; i < samplesToRead; ++i) {
      dest[i] = mBuffer[(bufferIndex + i) % bufferSize];
    }

    return frameCount;
  }

  /**
   * Get the total number of frames written since start.
   * Used for synchronization - capture side can use this to establish
   * the relationship between output and capture frame counters.
   */
  size_t getFramesWritten() const {
    return mFramesWritten.load(std::memory_order_acquire);
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
   * Start calibration capture - save current write position.
   * Call this BEFORE playing calibration signal.
   * @param expectedFrames Expected number of frames to capture (auto-stops when
   * reached)
   */
  void startCalibrationCapture(size_t expectedFrames = 72000) {
    // Clear any previous calibration data and reserve space
    mCalibrationData.clear();
    mCalibrationData.reserve(expectedFrames); // Pre-allocate for efficiency
    mCalibrationDataValid = false;
    mCalibrationExpectedFrames = expectedFrames;
    mCalibrationCapturedFrames = 0;

    mCalibrationStartPos.store(mWritePos.load(std::memory_order_acquire),
                               std::memory_order_release);
    mCalibrationActive.store(true, std::memory_order_release);
    aecLog("[AEC RefBuf] Calibration started at pos=%zu, expecting %zu frames "
           "(progressive capture)\n",
           mCalibrationStartPos.load(std::memory_order_relaxed),
           expectedFrames);
  }

  /**
   * Stop calibration capture.
   * With progressive capture, data is already saved - just disable capture
   * mode.
   */
  void stopCalibrationCapture() {
    // Disable calibration (stops progressive capture in write())
    mCalibrationActive.store(false, std::memory_order_release);

    // Mark as valid even if we didn't get all expected frames
    if (!mCalibrationDataValid && mCalibrationData.size() > 0) {
      mCalibrationDataValid = true;
    }

    // Calculate RMS of what we captured
    float energy = 0.0f;
    for (size_t i = 0; i < mCalibrationData.size(); ++i) {
      energy += mCalibrationData[i] * mCalibrationData[i];
    }
    float rms = mCalibrationData.size() > 0
                    ? std::sqrt(energy / mCalibrationData.size())
                    : 0.0f;

    aecLog("[AEC RefBuf] Calibration stopped. Progressive capture: %zu frames, "
           "RMS=%.4f\n",
           mCalibrationData.size(), rms);
  }

  /**
   * Read samples captured during calibration.
   * Returns the preserved data that was copied when stopCalibrationCapture()
   * was called. This data is safe from being overwritten by ongoing silence.
   *
   * @param dest Destination buffer for mono samples
   * @param sampleCount Number of mono samples to read
   * @return Number of samples actually read
   */
  size_t readForCalibration(float *dest, size_t sampleCount) const {
    if (dest == nullptr || sampleCount == 0)
      return 0;

    if (!mCalibrationDataValid || mCalibrationData.empty()) {
      aecLog("[AEC RefBuf] readForCalibration: No preserved data available!\n");
      return 0;
    }

    // Read from preserved calibration data (not from live buffer)
    size_t framesToRead = std::min(sampleCount, mCalibrationData.size());

    float energy = 0.0f;
    float maxVal = 0.0f;
    for (size_t i = 0; i < framesToRead; ++i) {
      float val = mCalibrationData[i];
      dest[i] = val;
      energy += val * val;
      if (std::abs(val) > maxVal)
        maxVal = std::abs(val);
    }

    float rms = framesToRead > 0 ? std::sqrt(energy / framesToRead) : 0.0f;
    aecLog("[AEC RefBuf] readForCalibration: %zu samples from preserved data, "
           "RMS=%.4f Peak=%.4f\n",
           framesToRead, rms, maxVal);

    return framesToRead;
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

  // Enabled flag - when false, writes are skipped to save CPU
  std::atomic<bool> mEnabled{false};

  // Calibration capture tracking
  std::atomic<size_t> mCalibrationStartPos{0};
  std::atomic<bool> mCalibrationActive{false};
  size_t mCalibrationExpectedFrames{72000}; // Expected frames to capture
  size_t mCalibrationCapturedFrames{0};     // Frames captured so far

  // Preserved calibration data (auto-copied when expected frames reached)
  std::vector<float> mCalibrationData;
  bool mCalibrationDataValid{false};

  // Timestamp tracking for synchronization
  // We store microseconds since a reference point instead of TimePoint directly
  // because std::atomic<TimePoint> isn't reliably lock-free
  std::atomic<int64_t> mLastWriteTimestampUs{0};
  std::atomic<size_t> mLastWriteFrameCount;
  TimePoint mReferenceTimePoint; // Fixed reference for converting to/from
                                 // microseconds

  // Helper to convert TimePoint to microseconds relative to reference
  int64_t toMicroseconds(TimePoint tp) const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               tp - mReferenceTimePoint)
        .count();
  }

  // Helper to convert microseconds back to TimePoint
  TimePoint fromMicroseconds(int64_t us) const {
    return mReferenceTimePoint + std::chrono::microseconds(us);
  }
};

// Global reference buffer pointer (set during initialization)
extern AECReferenceBuffer *g_aecReferenceBuffer;

#endif // AEC_REFERENCE_BUFFER_H
