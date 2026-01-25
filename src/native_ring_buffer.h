#ifndef NATIVE_RING_BUFFER_H
#define NATIVE_RING_BUFFER_H

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// Platform-specific headers for memory detection
#ifdef __ANDROID__
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

/**
 * Lock-free Single-Producer Single-Consumer (SPSC) ring buffer for continuous
 * audio capture with latency compensation (pre-roll).
 *
 * Audio callback thread writes captured samples (producer).
 * Main thread reads pre-roll samples for latency compensation (consumer).
 *
 * This buffer enables capturing audio from BEFORE the record button was pressed,
 * compensating for input latency (touchscreen ~50ms, Bluetooth ~35ms).
 *
 * Based on the proven AECReferenceBuffer pattern.
 */
class NativeRingBuffer {
public:
  /**
   * @param capacityFrames Buffer size in frames (samples per channel)
   * @param channels Number of audio channels (1=mono, 2=stereo)
   * @param sampleRate Sample rate in Hz
   */
  NativeRingBuffer(size_t capacityFrames, unsigned int channels,
                   unsigned int sampleRate);

  /**
   * Reconfigure the buffer dimensions.
   * Safe to call from main thread when audio is stopped.
   */
  bool configure(size_t capacityFrames, unsigned int channels,
                 unsigned int sampleRate);

  /**
   * Write audio frames to the buffer (called from audio callback thread).
   * Lock-free, safe for real-time audio.
   *
   * @param data Interleaved float samples
   * @param frameCount Number of frames to write
   */
  void write(const float *data, size_t frameCount);

  /**
   * Read frames with latency compensation (pre-roll).
   * Reads audio from the past to compensate for control latency.
   *
   * @param dest Destination buffer for interleaved samples
   * @param frameCount Number of frames to read
   * @param rewindFrames How many frames back in time to start reading
   * @return Number of frames actually read (may be less if not enough data)
   */
  size_t readPreRoll(float *dest, size_t frameCount, size_t rewindFrames);

  /**
   * Read a range of frames by total frame position.
   * Used for "forking" audio from the ring buffer for recording.
   *
   * @param dest Destination buffer for interleaved samples (must be large enough)
   * @param startTotalFrame Starting total frame position (from getTotalFramesWritten)
   * @param endTotalFrame Ending total frame position (exclusive)
   * @return Number of frames actually read (may be less if data overwritten)
   */
  size_t readRange(float *dest, size_t startTotalFrame, size_t endTotalFrame);

  // ==================== RECORDING MODE ====================
  // When recording is active, the buffer tracks start position.
  // On stop, the recorded section can be extracted.

  /**
   * Start recording - marks current position (with latency compensation).
   * Buffer keeps rolling but we track the recording start.
   *
   * @param latencyCompFrames Frames to rewind for latency compensation
   */
  void startRecording(size_t latencyCompFrames = 0);

  /**
   * Stop recording and extract the recorded audio.
   * Returns allocated buffer that caller must free, or nullptr on error.
   *
   * @param outFrameCount Output: number of frames in returned buffer
   * @return Pointer to interleaved float samples (caller owns), or nullptr
   */
  float* stopRecording(size_t* outFrameCount);

  /**
   * Check if recording is active.
   */
  bool isRecording() const { return mRecordingActive.load(std::memory_order_acquire); }

  /**
   * Get recording start frame (total frames at start, with latency comp applied).
   */
  size_t getRecordingStartFrame() const { return mRecordingStartTotalFrame; }

  /**
   * Get available system RAM in bytes.
   * Used to intelligently size recording buffer.
   */
  static size_t getAvailableRAM();

  /**
   * Get maximum recording frames based on available RAM.
   * Uses 10% of available RAM as upper limit (~9 min on 2GB device).
   */
  size_t getMaxRecordingFrames() const;

  /**
   * Pre-allocate buffer for recording (call before startRecording).
   * Allocates up to 10% of available RAM to avoid audio thread allocations.
   */
  void preAllocateForRecording();

  /**
   * Get the number of frames available in the buffer.
   * Returns the total valid frames (up to capacity after wrap).
   */
  size_t available() const;

  /**
   * Get total frames written since buffer creation/reset.
   * Used for synchronization with other components.
   */
  size_t getTotalFramesWritten() const;

  /**
   * Get current audio level in dB (RMS).
   * Calculated continuously in write() for efficiency.
   */
  float getAudioLevelDb() const;

  /**
   * Reset the buffer state.
   */
  void reset();

  /**
   * Get buffer capacity in frames.
   */
  size_t capacityInFrames() const { return mCapacityFrames; }

  /**
   * Get channel count.
   */
  unsigned int channels() const { return mChannels; }

  /**
   * Get sample rate.
   */
  unsigned int sampleRate() const { return mSampleRate; }

private:
  unsigned int mChannels;
  unsigned int mSampleRate;
  size_t mCapacityFrames;
  std::vector<float> mBuffer;

  // Atomic positions for lock-free operation
  std::atomic<size_t> mWritePos{0};
  std::atomic<size_t> mTotalFramesWritten{0};

  // Audio level tracking (updated in write, read from main thread)
  std::atomic<float> mCurrentLevelDb{-100.0f};

  // Moving average for level smoothing
  static constexpr int kLevelAverageSize = 4;
  float mLevelHistory[kLevelAverageSize] = {0};
  int mLevelHistoryIndex = 0;

  // Recording state - when active, buffer stops wrapping and extends linearly
  std::atomic<bool> mRecordingActive{false};
  size_t mRecordingStartTotalFrame{0};
  size_t mRecordingStartWritePos{0};  // Buffer position when recording started
  size_t mMaxRecordingFrames{0};      // Pre-calculated max based on RAM
  bool mPreAllocated{false};          // Whether buffer is pre-allocated for recording
};

// Global ring buffer pointer (set during initialization)
extern NativeRingBuffer *g_nativeRingBuffer;

#endif // NATIVE_RING_BUFFER_H
