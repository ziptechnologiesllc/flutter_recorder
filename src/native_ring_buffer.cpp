#include "native_ring_buffer.h"
#include <cstdio>

// Global ring buffer instance
NativeRingBuffer *g_nativeRingBuffer = nullptr;

// ==================== PLATFORM-SPECIFIC RAM DETECTION ====================

size_t NativeRingBuffer::getAvailableRAM() {
#ifdef __ANDROID__
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    // Available = free + buffers + cached (approximation)
    // On Android, freeram is often very low due to aggressive caching
    size_t available = (size_t)info.freeram * info.mem_unit;
    printf("[RingBuffer] Android available RAM: %zu MB\n", available / (1024 * 1024));
    return available;
  }
  return 256 * 1024 * 1024; // Fallback: 256MB

#elif defined(__APPLE__)
  // macOS/iOS: Use mach APIs
  mach_port_t host = mach_host_self();
  vm_size_t pageSize;
  host_page_size(host, &pageSize);

  vm_statistics64_data_t vmStats;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

  if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vmStats, &count) == KERN_SUCCESS) {
    size_t freePages = vmStats.free_count + vmStats.inactive_count;
    size_t available = freePages * pageSize;
    printf("[RingBuffer] macOS/iOS available RAM: %zu MB\n", available / (1024 * 1024));
    return available;
  }
  return 512 * 1024 * 1024; // Fallback: 512MB

#elif defined(__linux__)
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    size_t available = (size_t)info.freeram * info.mem_unit;
    printf("[RingBuffer] Linux available RAM: %zu MB\n", available / (1024 * 1024));
    return available;
  }
  return 512 * 1024 * 1024; // Fallback: 512MB

#elif defined(_WIN32)
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  if (GlobalMemoryStatusEx(&memInfo)) {
    size_t available = (size_t)memInfo.ullAvailPhys;
    printf("[RingBuffer] Windows available RAM: %zu MB\n", available / (1024 * 1024));
    return available;
  }
  return 512 * 1024 * 1024; // Fallback: 512MB

#else
  return 256 * 1024 * 1024; // Conservative fallback: 256MB
#endif
}

size_t NativeRingBuffer::getMaxRecordingFrames() const {
  if (mMaxRecordingFrames > 0) {
    return mMaxRecordingFrames;
  }

  // Calculate max frames based on 10% of available RAM
  // On 2GB device: ~205MB = ~9 minutes stereo @ 48kHz
  size_t availableRAM = getAvailableRAM();
  size_t maxRAMForRecording = availableRAM / 10; // 10% limit

  // Each frame = channels * sizeof(float) bytes
  size_t bytesPerFrame = mChannels * sizeof(float);
  size_t maxFrames = maxRAMForRecording / bytesPerFrame;

  // Cap at reasonable maximum (10 minutes @ 48kHz = 28.8M frames)
  const size_t absoluteMax = 10 * 60 * 48000; // 10 minutes
  if (maxFrames > absoluteMax) {
    maxFrames = absoluteMax;
  }

  printf("[RingBuffer] Max recording: %zu frames (%.1f seconds @ %uHz), using %zu MB\n",
         maxFrames, (float)maxFrames / mSampleRate, mSampleRate,
         (maxFrames * bytesPerFrame) / (1024 * 1024));

  return maxFrames;
}

void NativeRingBuffer::preAllocateForRecording() {
  if (mPreAllocated) {
    return; // Already pre-allocated
  }

  // Calculate and cache max recording frames
  mMaxRecordingFrames = getMaxRecordingFrames();

  // Pre-allocate both input and output buffers
  size_t maxSamples = mMaxRecordingFrames * mChannels;

  printf("[RingBuffer] Pre-allocating %zu samples (%zu MB) for recording\n",
         maxSamples, (maxSamples * sizeof(float)) / (1024 * 1024));

  try {
    mBuffer.resize(maxSamples, 0.0f);       // Input buffer (recording)
    mOutputBuffer.resize(maxSamples, 0.0f); // Output buffer (for stopRecording)
    mPreAllocated = true;
    printf("[RingBuffer] Pre-allocation successful (buffer size: %zu)\n", mBuffer.size());
  } catch (const std::bad_alloc& e) {
    printf("[RingBuffer] Pre-allocation failed: %s - will grow dynamically\n", e.what());
    mPreAllocated = false;
  }
}

NativeRingBuffer::NativeRingBuffer(size_t capacityFrames, unsigned int channels,
                                   unsigned int sampleRate)
    : mChannels(channels), mSampleRate(sampleRate),
      mCapacityFrames(capacityFrames) {
  mBuffer.resize(capacityFrames * channels, 0.0f);
  mWritePos.store(0, std::memory_order_relaxed);
  mTotalFramesWritten.store(0, std::memory_order_relaxed);
  mCurrentLevelDb.store(-100.0f, std::memory_order_relaxed);
}

bool NativeRingBuffer::configure(size_t capacityFrames, unsigned int channels,
                                 unsigned int sampleRate) {
  mCapacityFrames = capacityFrames;
  mChannels = channels;
  mSampleRate = sampleRate;
  mBuffer.resize(capacityFrames * channels, 0.0f);
  // Reset pre-allocation flag so next preAllocateForRecording() will re-expand buffer
  // Without this, buffer stays at ring capacity (5s) instead of recording capacity (10min)
  mPreAllocated = false;
  mMaxRecordingFrames = 0;
  reset();
  return true;
}

void NativeRingBuffer::write(const float *data, size_t frameCount, unsigned int actualChannels) {
  if (data == nullptr || frameCount == 0 || mCapacityFrames == 0) {
    return;
  }

  // Use actualChannels if provided (non-zero), otherwise use configured mChannels
  unsigned int channelsToUse = (actualChannels > 0) ? actualChannels : mChannels;

  // Auto-reconfigure channels if actual differs from configured
  // This happens on first few callbacks before recording starts
  if (actualChannels > 0 && actualChannels != mChannels) {
    mChannels = actualChannels;
  }

  const size_t samplesToWrite = frameCount * channelsToUse;

  // Calculate RMS level for level meter
  float sumSquares = 0.0f;
  for (size_t i = 0; i < samplesToWrite; i++) {
    sumSquares += data[i] * data[i];
  }
  float rms = std::sqrt(sumSquares / samplesToWrite);

  // Convert to dB with smoothing
  float levelDb = (rms > 1e-10f) ? 20.0f * std::log10(rms) : -100.0f;

  // Simple exponential smoothing
  float prevLevel = mCurrentLevelDb.load(std::memory_order_relaxed);
  float smoothedLevel = 0.8f * prevLevel + 0.2f * levelDb;
  mCurrentLevelDb.store(smoothedLevel, std::memory_order_release);

  // Get current write position
  size_t writePos = mWritePos.load(std::memory_order_acquire);

  // Recording mode: write linearly (no wrap) up to pre-allocated size
  if (mRecordingActive.load(std::memory_order_acquire)) {
    size_t samplePos = writePos * mChannels;
    size_t requiredSize = samplePos + samplesToWrite;

    // Debug logging disabled for production - causes audio glitches
    // Uncomment for debugging interleaving issues:
    // static int recordWriteDebugCount = 0;
    // if (recordWriteDebugCount < 5) {
    //   printf("[RingBuffer RECORDING] writePos=%zu, frameCount=%zu\n", writePos, frameCount);
    //   recordWriteDebugCount++;
    // }

    // Check if we'd exceed pre-allocated buffer (cap recording, don't resize from audio thread!)
    if (requiredSize > mBuffer.size()) {
      // Recording exceeds max capacity - track dropped frames for debugging
      // This typically indicates preAllocateForRecording() wasn't called or buffer was reconfigured
      mDroppedFrames.fetch_add(frameCount, std::memory_order_relaxed);
      return;
    }

    // Simple linear write (no wrap)
    std::memcpy(mBuffer.data() + samplePos, data, samplesToWrite * sizeof(float));

    // Update write position (linear, no modulo)
    mWritePos.store(writePos + frameCount, std::memory_order_release);
  } else {
    // Normal ring buffer mode: wrap around using mCapacityFrames (NOT mBuffer.size()!)
    // CRITICAL: After pre-allocation, mBuffer.size() can be huge (10 min recording buffer)
    // but ring mode should only use the first mCapacityFrames worth of data
    const size_t ringCapacitySamples = mCapacityFrames * mChannels;
    size_t samplePos = writePos * mChannels;

    if (samplePos + samplesToWrite <= ringCapacitySamples) {
      // No wrap - single copy
      std::memcpy(mBuffer.data() + samplePos, data, samplesToWrite * sizeof(float));
    } else {
      // Wrap around - two copies
      size_t firstPart = ringCapacitySamples - samplePos;
      size_t secondPart = samplesToWrite - firstPart;

      std::memcpy(mBuffer.data() + samplePos, data, firstPart * sizeof(float));
      std::memcpy(mBuffer.data(), data + firstPart, secondPart * sizeof(float));
    }

    // Update write position (circular)
    size_t newWritePos = (writePos + frameCount) % mCapacityFrames;
    mWritePos.store(newWritePos, std::memory_order_release);
  }

  // Update total frames written
  mTotalFramesWritten.fetch_add(frameCount, std::memory_order_release);
}

size_t NativeRingBuffer::readPreRoll(float *dest, size_t frameCount,
                                     size_t rewindFrames) {
  if (dest == nullptr || frameCount == 0) {
    return 0;
  }

  const size_t totalWritten =
      mTotalFramesWritten.load(std::memory_order_acquire);
  const size_t writePos = mWritePos.load(std::memory_order_acquire);

  // Calculate how much data is available
  size_t availableFrames = std::min(totalWritten, mCapacityFrames);

  // Clamp rewind to available data
  size_t actualRewind = std::min(rewindFrames, availableFrames);

  // Wait, we want to read from the PAST, so we need data from before writePos
  // The rewind is how far back we start reading
  // We want to read: [writePos - rewind - frameCount, writePos - rewind)

  // Ensure we don't read more than requested or available
  size_t framesToRead = std::min(frameCount, availableFrames);
  if (actualRewind + framesToRead > availableFrames) {
    framesToRead = availableFrames - actualRewind;
  }

  if (framesToRead == 0) {
    return 0;
  }

  // CRITICAL: Use ring capacity, not full buffer size (which can be huge after pre-allocation)
  const size_t ringCapacitySamples = mCapacityFrames * mChannels;
  const size_t samplesToRead = framesToRead * mChannels;

  // Calculate read start position
  // We read from (writePos - rewind - framesToRead) to (writePos - rewind)
  size_t readStartFrame =
      (writePos + mCapacityFrames - actualRewind - framesToRead) %
      mCapacityFrames;
  size_t readStartSample = readStartFrame * mChannels;

  // Read from circular buffer (ring region only)
  if (readStartSample + samplesToRead <= ringCapacitySamples) {
    // No wrap - single copy
    std::memcpy(dest, mBuffer.data() + readStartSample,
                samplesToRead * sizeof(float));
  } else {
    // Wrap around - two copies
    size_t firstPart = ringCapacitySamples - readStartSample;
    size_t secondPart = samplesToRead - firstPart;

    std::memcpy(dest, mBuffer.data() + readStartSample,
                firstPart * sizeof(float));
    std::memcpy(dest + firstPart, mBuffer.data(), secondPart * sizeof(float));
  }

  return framesToRead;
}

size_t NativeRingBuffer::readRange(float *dest, size_t startTotalFrame,
                                    size_t endTotalFrame) {
  if (dest == nullptr || endTotalFrame <= startTotalFrame) {
    return 0;
  }

  const size_t totalWritten =
      mTotalFramesWritten.load(std::memory_order_acquire);
  // CRITICAL: Use ring capacity, not full buffer size
  const size_t ringCapacitySamples = mCapacityFrames * mChannels;

  // Check if the requested range is still available in the buffer
  // Data is overwritten after mCapacityFrames
  size_t oldestAvailable =
      (totalWritten > mCapacityFrames) ? (totalWritten - mCapacityFrames) : 0;

  if (startTotalFrame < oldestAvailable) {
    // Some data has been overwritten - adjust start
    printf("[RingBuffer] WARNING: Start frame %zu overwritten, oldest=%zu\n",
           startTotalFrame, oldestAvailable);
    startTotalFrame = oldestAvailable;
  }

  if (endTotalFrame > totalWritten) {
    // Can't read beyond what's written
    printf("[RingBuffer] WARNING: End frame %zu > written %zu, clamping\n",
           endTotalFrame, totalWritten);
    endTotalFrame = totalWritten;
  }

  if (endTotalFrame <= startTotalFrame) {
    return 0;
  }

  const size_t framesToRead = endTotalFrame - startTotalFrame;
  const size_t samplesToRead = framesToRead * mChannels;

  // Calculate buffer position for start frame
  // Position in circular buffer = totalFrame % capacityFrames
  size_t readStartFrame = startTotalFrame % mCapacityFrames;
  size_t readStartSample = readStartFrame * mChannels;

  // Read from circular buffer (ring region only)
  if (readStartSample + samplesToRead <= ringCapacitySamples) {
    // No wrap - single copy
    std::memcpy(dest, mBuffer.data() + readStartSample,
                samplesToRead * sizeof(float));
  } else {
    // Wrap around - two copies
    size_t firstPart = ringCapacitySamples - readStartSample;
    size_t secondPart = samplesToRead - firstPart;

    std::memcpy(dest, mBuffer.data() + readStartSample,
                firstPart * sizeof(float));
    std::memcpy(dest + firstPart, mBuffer.data(), secondPart * sizeof(float));
  }

  return framesToRead;
}

size_t NativeRingBuffer::available() const {
  size_t total = mTotalFramesWritten.load(std::memory_order_acquire);
  return std::min(total, mCapacityFrames);
}

size_t NativeRingBuffer::getTotalFramesWritten() const {
  return mTotalFramesWritten.load(std::memory_order_acquire);
}

float NativeRingBuffer::getAudioLevelDb() const {
  return mCurrentLevelDb.load(std::memory_order_acquire);
}

void NativeRingBuffer::reset() {
  mWritePos.store(0, std::memory_order_release);
  mTotalFramesWritten.store(0, std::memory_order_release);
  mCurrentLevelDb.store(-100.0f, std::memory_order_release);
  mRecordingActive.store(false, std::memory_order_release);
  mRecordingStartTotalFrame = 0;

  // Clear the buffer
  std::fill(mBuffer.begin(), mBuffer.end(), 0.0f);
}

// AUDIO THREAD SAFE: No printf/fprintf calls
void NativeRingBuffer::startRecording(size_t latencyCompFrames) {
  size_t writePos = mWritePos.load(std::memory_order_acquire);
  size_t currentTotal = mTotalFramesWritten.load(std::memory_order_acquire);

  // Calculate where recording effectively starts (with latency compensation)
  if (latencyCompFrames > 0 && writePos >= latencyCompFrames) {
    mRecordingStartWritePos = writePos - latencyCompFrames;
  } else if (latencyCompFrames > 0 && currentTotal >= latencyCompFrames) {
    // Handle wrap-around case in ring buffer
    mRecordingStartWritePos = (writePos + mCapacityFrames - latencyCompFrames) % mCapacityFrames;
  } else {
    mRecordingStartWritePos = writePos;
  }

  mRecordingStartTotalFrame = currentTotal;
  // Reset dropped frame counter for this recording session
  mDroppedFrames.store(0, std::memory_order_release);
  mRecordingActive.store(true, std::memory_order_release);
}

// AUDIO THREAD SAFE: No allocations, no printf - just memcpy to pre-allocated buffer
float* NativeRingBuffer::stopRecording(size_t* outFrameCount, size_t expectedFrameCount) {
  if (!mRecordingActive.load(std::memory_order_acquire)) {
    if (outFrameCount) *outFrameCount = 0;
    return nullptr;
  }

  mRecordingActive.store(false, std::memory_order_release);

  size_t currentWritePos = mWritePos.load(std::memory_order_acquire);
  size_t startWritePos = mRecordingStartWritePos;

  if (currentWritePos <= startWritePos) {
    if (outFrameCount) *outFrameCount = 0;
    return nullptr;
  }

  size_t availableFrames = currentWritePos - startWritePos;

  // Use expected frame count if provided (for sample-accurate loop multiples)
  // Only use it if we have enough data available
  size_t frameCount;
  if (expectedFrameCount > 0 && expectedFrameCount <= availableFrames) {
    frameCount = expectedFrameCount;
  } else {
    frameCount = availableFrames;
  }
  size_t sampleCount = frameCount * mChannels;

  // Use pre-allocated output buffer (no allocation on audio thread!)
  // If buffer is too small (shouldn't happen), return error
  if (sampleCount > mOutputBuffer.size()) {
    if (outFrameCount) *outFrameCount = 0;
    return nullptr;
  }

  // Copy to pre-allocated output buffer
  size_t startSamplePos = startWritePos * mChannels;
  std::memcpy(mOutputBuffer.data(), mBuffer.data() + startSamplePos, sampleCount * sizeof(float));

  // =========================================================================
  // CROSSFADE: Apply fade-in at start and fade-out at end to prevent clicks
  // =========================================================================
  const float crossfadeDurationMs = 2.5f;
  size_t crossfadeFrames = (size_t)(mSampleRate * crossfadeDurationMs / 1000.0f);
  crossfadeFrames = std::min(crossfadeFrames, frameCount / 2);

  if (crossfadeFrames > 0) {
    float* output = mOutputBuffer.data();

    // Apply fade-in at the start (all channels)
    for (size_t frame = 0; frame < crossfadeFrames; ++frame) {
      float gain = (float)frame / (float)crossfadeFrames;
      for (unsigned int ch = 0; ch < mChannels; ++ch) {
        output[frame * mChannels + ch] *= gain;
      }
    }

    // Apply fade-out at the end (all channels)
    size_t fadeOutStart = frameCount - crossfadeFrames;
    for (size_t frame = fadeOutStart; frame < frameCount; ++frame) {
      float gain = (float)(frameCount - frame) / (float)crossfadeFrames;
      for (unsigned int ch = 0; ch < mChannels; ++ch) {
        output[frame * mChannels + ch] *= gain;
      }
    }
  }

  // Reset write position to beginning (rewind the tape)
  mWritePos.store(0, std::memory_order_release);
  mTotalFramesWritten.store(0, std::memory_order_release);

  if (outFrameCount) *outFrameCount = frameCount;
  return mOutputBuffer.data();  // Caller must NOT free - buffer owned by ring buffer
}
