#include "capture.h"
#include "circular_buffer.h"
#include "native_ring_buffer.h"
#include "soloud_slave_bridge.h"
#include "filters/aec/reference_buffer.h"
#include "native_scheduler.h"

#include "fft/soloud_fft.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <memory.h>
#include <memory>
#include <mutex>
#include <time.h>
#include <vector>

// External logging function defined in calibration.cpp
extern void aecLog(const char *fmt, ...);

#ifdef _IS_ANDROID_
#include <android/log.h>
#include <jni.h>
#include <dlfcn.h>  // For dlsym to dynamically load AAudio buffer APIs

#define LOG_TAG "FlutterRecorder"
#endif

// 128 frames for ultra-low latency monitoring (~2.67ms @ 48kHz)
#define BUFFER_SIZE 128                      // Buffer length in frames
#define STREAM_BUFFER_SIZE (BUFFER_SIZE * 2) // Buffer length in frames
#define MOVING_AVERAGE_SIZE 4                // Moving average window size
#define VISUALIZATION_BUFFER_SIZE                                              \
  8192 // Larger buffer for waveform visualization

// =============================================================================
// DEBUG LOGGING CONTROL
// WARNING: Enabling callback debug logging causes significant latency (600ms+)
// on mobile devices. Only enable for specific debugging sessions.
// =============================================================================
#define DEBUG_CALLBACK_CHANNELS 0    // Log channel count on first N callbacks
#define DEBUG_CALLBACK_SLAVE 0       // Log slave mode mixing
#define DEBUG_CALLBACK_FILTERS 0     // Log filter processing
#define DEBUG_CALLBACK_CALIBRATION 0 // Log calibration capture (less impact)
#define DEBUG_CALLBACK_AEC_REF 0     // Log AEC reference buffer writes

// Double-buffer for lock-free visualization data
// Audio thread writes to one buffer while UI thread reads from the other
static float capturedBufferA[VISUALIZATION_BUFFER_SIZE * 2];
static float capturedBufferB[VISUALIZATION_BUFFER_SIZE * 2];
static std::atomic<int> capturedBufferWriteIndex{0};  // 0 = writing to A, 1 = writing to B
// Legacy pointer for compatibility (points to readable buffer)
float *capturedBuffer = capturedBufferA;
std::atomic<bool> is_silent{true};    // Initial state
bool delayed_silence_started = false; // Whether the silence is delayed
std::atomic<float> energy_db{-100.0f}; // Current energy

/// the buffer used for capturing audio.
std::unique_ptr<CircularBuffer<float>> circularBuffer;

/// the buffer used for streaming.
std::unique_ptr<std::vector<unsigned char>> streamBuffer;

#ifdef _IS_WIN_
#define CLOCK_REALTIME 0
// struct timespec { long long tv_sec; long tv_nsec; };    //header part
// Windows is not POSIX compliant. Implement this.
int clock_gettime(int, struct timespec *spec) // C-file part
{
  __int64 wintime;
  GetSystemTimeAsFileTime((FILETIME *)&wintime);
  wintime -= 116444736000000000i64;            // 1jan1601 to 1jan1970
  spec->tv_sec = wintime / 10000000i64;        // seconds
  spec->tv_nsec = wintime % 10000000i64 * 100; // nano-seconds
  return 0;
}
#endif

void getTime(struct timespec *time) {
  if (clock_gettime(CLOCK_REALTIME, time) == -1) {
    perror("clock getTime");
    exit(EXIT_FAILURE);
  }
}

/// returns the elapsed time in seconds
double getElapsed(struct timespec since) {
  struct timespec now;
  if (clock_gettime(CLOCK_REALTIME, &now) == -1) {
    perror("clock getTime");
    exit(EXIT_FAILURE);
  }
  return ((double)(now.tv_sec - since.tv_sec) +
          (double)(now.tv_nsec - since.tv_nsec) / 1.0e9L);
}

// Function to convert energy to decibels
float energy_to_db(float energy) {
  return 10.0f * log10f(energy + 1e-10f); // Add a small value to avoid log(0)
}

void calculateEnergy(float *captured, ma_uint32 frameCount, int channels) {
  static float moving_average[MOVING_AVERAGE_SIZE] = {
      0};                       // Moving average window
  static int average_index = 0; // Circular buffer index
  float sum = 0.0f;

  // Calculate the average energy of the current buffer
  // Must iterate over all samples: frameCount * channels for stereo
  size_t sampleCount = (size_t)frameCount * channels;
  for (size_t i = 0; i < sampleCount; i++) {
    sum += captured[i] * captured[i];
  }
  float average_energy = sum / sampleCount;

  // Update the moving average window
  moving_average[average_index] = average_energy;
  average_index =
      (average_index + 1) % MOVING_AVERAGE_SIZE; // Circular buffer cycle

  // Calculate the moving average
  float moving_average_sum = 0.0f;
  for (int i = 0; i < MOVING_AVERAGE_SIZE; i++) {
    moving_average_sum += moving_average[i];
  }
  float smoothed_energy = moving_average_sum / MOVING_AVERAGE_SIZE;

  // Convert energy to decibels
  energy_db = energy_to_db(smoothed_energy);
}

void detectSilence(Capture *userData) {
  static struct timespec startSilence; // Start time of silence
  // Check if the signal is below the silence threshold
  if (energy_db < userData->silenceThresholdDb) {
    if (!is_silent.load() && !delayed_silence_started) {
      getTime(&startSilence);
      // Transition: Sound -> Silence
      is_silent = true;
    } else {
      double elapsed = getElapsed(startSilence);
      if (elapsed >= userData->silenceDuration && is_silent.load() &&
          !delayed_silence_started) {
        printf("Silence started after %f s. Level in dB: %.2f\n", elapsed,
               energy_db.load());
        /// empty capturedBuffer
        if (circularBuffer && circularBuffer.get()->size() > BUFFER_SIZE)
          circularBuffer.get()->pop(circularBuffer.get()->size());
        delayed_silence_started = true;
        if (nativeSilenceChangedCallback != nullptr) {
          float energy_value = energy_db.load();
          nativeSilenceChangedCallback(&delayed_silence_started, &energy_value);
        }
      }
    }
  } else {
    if (is_silent.load()) {
      double elapsed = getElapsed(startSilence);
      if (elapsed >= userData->silenceDuration && delayed_silence_started) {
        // Transition: Silence -> Sound
        printf("Sound started after %f s. Level in dB: %.2f   %f %f %f\n",
               elapsed, energy_db.load(), userData->silenceThresholdDb,
               userData->silenceDuration,
               userData->secondsOfAudioToWriteBefore);
        is_silent = false;
        delayed_silence_started = false;
        // Write all the circularBuffer data which contains the audio occurred
        // before the silence ended.
        if (userData->isRecording &&
            userData->secondsOfAudioToWriteBefore > 0 && circularBuffer) {
          ma_uint32 frameCount = (unsigned int)(circularBuffer.get()->size());
          auto data = circularBuffer.get()->pop(frameCount);
          // printf("WRITE secondsOfAudioToWriteBefore buffer size: %u  frames:
          // %u  frame got: %u\n",
          //    circularBuffer.get()->size(), frameCount, data.size());
          // The framCount in wav.write is one for all the channels.
          // Use actual device channels to avoid division by zero (deviceConfig may be 0 in auto mode)
          int actualChannels = userData->getCaptureChannels();
          if (actualChannels < 1) actualChannels = 1;
          userData->wav.write(data.data(), data.size() / actualChannels);
        }
        if (nativeSilenceChangedCallback != nullptr) {
          float energy_value = energy_db.load();
          nativeSilenceChangedCallback(&delayed_silence_started, &energy_value);
        }
      }

      /// Reset the clock if sound happens during the deley after a silence,
      if (elapsed < userData->silenceDuration && is_silent.load()) {
        getTime(&startSilence);
        is_silent = false;
        delayed_silence_started = false;
      }
    }
  }
}

// A "frame" is one sample for each channel. For example, in a stereo stream (2
// channels),
// one frame is 2 samples: one for the left, one for the right.
void data_callback(ma_device *pDevice, void *pOutput, const void *pInput,
                   ma_uint32 frameCount) {
  Capture *userData = (Capture *)pDevice->pUserData;
  if (!userData) return;

  // CRITICAL: Use ACTUAL device channels, not CONFIGURED channels!
  int playbackChannels = pDevice->playback.channels;
  int captureChannels = pDevice->capture.channels;

  // Handle Format Conversion: Convert pInput to Float32 if necessary
  float *captured = nullptr;
  if (pDevice->capture.format == ma_format_f32) {
    captured = (float *)pInput;
  } else if (pInput != nullptr) {
    // Convert Integer -> F32 using the pre-allocated conversion buffer
    size_t samplesCount = (size_t)frameCount * captureChannels;
    if (userData->mConversionBuffer.size() < samplesCount) {
        userData->mConversionBuffer.resize(samplesCount);
    }
    
    if (pDevice->capture.format == ma_format_s16) {
      ma_pcm_s16_to_f32(userData->mConversionBuffer.data(), (const ma_int16 *)pInput, samplesCount, ma_dither_mode_none);
#ifdef _IS_ANDROID_
      static int captureConvDebugCount = 0;
      if (++captureConvDebugCount <= 10) {
        // Check max value in converted buffer
        float maxVal = 0.0f;
        for (size_t i = 0; i < samplesCount && i < 100; i++) {
          if (fabsf(userData->mConversionBuffer[i]) > maxVal) maxVal = fabsf(userData->mConversionBuffer[i]);
        }
        // Check max value in original s16 buffer
        int16_t maxS16 = 0;
        const ma_int16* s16Input = (const ma_int16*)pInput;
        for (size_t i = 0; i < samplesCount && i < 100; i++) {
          if (abs(s16Input[i]) > maxS16) maxS16 = abs(s16Input[i]);
        }
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
            "[Capture Conv #%d] samples=%zu, maxS16=%d, maxF32=%.6f",
            captureConvDebugCount, samplesCount, maxS16, maxVal);
      }
#endif
    } else if (pDevice->capture.format == ma_format_s24) {
      ma_pcm_s24_to_f32(userData->mConversionBuffer.data(), pInput, samplesCount, ma_dither_mode_none);
    } else if (pDevice->capture.format == ma_format_s32) {
      ma_pcm_s32_to_f32(userData->mConversionBuffer.data(), (const ma_int32 *)pInput, samplesCount, ma_dither_mode_none);
    } else {
      // Unsupported format for internal processing - fallback to input pointer
      captured = (float *)pInput; 
    }
    
    if (captured == nullptr) {
      captured = userData->mConversionBuffer.data();
    }
  } else {
    captured = (float *)pInput;
  }

#if DEBUG_CALLBACK_CHANNELS
  static int channelDebugCount = 0;
  if (++channelDebugCount <= 5) {
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                        "[Capture CB] channels: capture=%d, playback=%d, frames=%u",
                        captureChannels, playbackChannels, frameCount);
#endif
  }
#endif

  // =========================================================================
  // NATIVE RING BUFFER: Continuous capture for latency compensation (pre-roll)
  // Write immediately after format conversion, before any processing
  // IMPORTANT: Pass actual capture channels to auto-reconfigure if mismatched
  // =========================================================================
  if (g_nativeRingBuffer != nullptr && captured != nullptr) {
    g_nativeRingBuffer->write(captured, frameCount, captureChannels);
  }

  // =========================================================================
  // SLAVE MODE: SoLoud output driven by this callback (for AEC clock sync)
  // =========================================================================
  // In slave mode, we call SoLoud's mix function directly instead of SoLoud
  // running its own audio device. This ensures perfect clock synchronization
  // between capture and playback, fixing AEC drift issues on Linux.
  if (soloud_isSlaveMode() && g_soloudSlaveMixCallback != nullptr &&
      pOutput != nullptr) {
    // Ensure playback buffer is large enough for f32 processing
    size_t playbackSamples = (size_t)frameCount * playbackChannels;
    if (userData->mPlaybackBuffer.size() < playbackSamples) {
      userData->mPlaybackBuffer.resize(playbackSamples);
    }
    float *playbackFloat = userData->mPlaybackBuffer.data();

    // Zero the buffer first in case SoLoud doesn't write all samples
    memset(playbackFloat, 0, playbackSamples * sizeof(float));

    // Get SoLoud's mixed output into our f32 buffer (not pOutput directly)
    // This allows all processing (AEC, monitoring) to work in f32
    g_soloudSlaveMixCallback(playbackFloat, frameCount, playbackChannels);

#if DEBUG_CALLBACK_SLAVE
    static int soloudMixDebugCount = 0;
    if (++soloudMixDebugCount <= 5) {
#ifdef _IS_ANDROID_
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[Capture Slave] frames=%u ch=%d", frameCount, playbackChannels);
#endif
    }
#endif

    // Mark slave audio as ready after first successful callback
    // This signals that the audio pipeline is flowing and calibration can start
    soloud_setSlaveAudioReady();

    // Write to AEC reference buffer IN THE SAME CALLBACK - guarantees sync!
    // This is the whole point of slave mode: one callback, one clock.
    // Handle channel conversion: pOutput has actual device channels, but AEC buffer
    // may have different channel count (usually mono for AEC purposes).
    // Skip entirely if AEC is disabled to save CPU
    if (g_aecReferenceBuffer != nullptr && g_aecReferenceBuffer->isEnabled()) {
      unsigned int bufferCh = g_aecReferenceBuffer->channels();
      float *outputFloat = playbackFloat; // Use our f32 buffer for AEC

      if ((unsigned int)playbackChannels == bufferCh) {
        // Channels match - direct write
        g_aecReferenceBuffer->write(outputFloat, frameCount);
      } else if (playbackChannels == 2 && bufferCh == 1) {
        // Stereo playback → Mono AEC buffer: average L+R
        static thread_local std::vector<float> monoAec;
        if (monoAec.size() < frameCount) monoAec.resize(frameCount);
        for (ma_uint32 i = 0; i < frameCount; ++i) {
          monoAec[i] = (outputFloat[i * 2] + outputFloat[i * 2 + 1]) * 0.5f;
        }
        g_aecReferenceBuffer->write(monoAec.data(), frameCount);
      } else if (playbackChannels == 1 && bufferCh == 2) {
        // Mono playback → Stereo AEC buffer: duplicate to both channels
        static thread_local std::vector<float> stereoAec;
        if (stereoAec.size() < frameCount * 2) stereoAec.resize(frameCount * 2);
        for (ma_uint32 i = 0; i < frameCount; ++i) {
          stereoAec[i * 2] = stereoAec[i * 2 + 1] = outputFloat[i];
        }
        g_aecReferenceBuffer->write(stereoAec.data(), frameCount);
      } else {
        // Unsupported - write directly and hope for the best
        g_aecReferenceBuffer->write(outputFloat, frameCount);
      }

    }

    // If monitoring is enabled, ADD the captured input to the SoLoud output
    // (This allows hearing yourself while SoLoud plays)
    if (userData->monitoringEnabled && captured != nullptr && pOutput != nullptr) {
      float *outputFloat = playbackFloat; // Use our f32 buffer for monitoring mix
      float *inputFloat = captured;

      // Simple mix: add monitoring signal on top of SoLoud output
      // Scale down monitoring to prevent clipping when both are active
      float monitorGain = 0.8f;

      if (captureChannels == playbackChannels) {
        // Same channel count - direct mix
        for (ma_uint32 i = 0; i < frameCount * playbackChannels; i++) {
          outputFloat[i] += inputFloat[i] * monitorGain;
        }
      } else if (captureChannels == 1 && playbackChannels == 2) {
        // Mono capture to stereo output
        for (ma_uint32 i = 0; i < frameCount; i++) {
          outputFloat[i * 2] += inputFloat[i] * monitorGain;
          outputFloat[i * 2 + 1] += inputFloat[i] * monitorGain;
        }
      } else if (captureChannels == 2 && playbackChannels == 1) {
        // Stereo capture to mono output
        for (ma_uint32 i = 0; i < frameCount; i++) {
          outputFloat[i] += (inputFloat[i * 2] + inputFloat[i * 2 + 1]) * 0.5f *
                            monitorGain;
        }
      }
    }

    // Final step: Convert f32 playback buffer to device format and copy to pOutput
    // This happens AFTER all processing (AEC, monitoring) is done in f32
#ifdef _IS_ANDROID_
    static int convDebugCount = 0;
    if (++convDebugCount <= 10) {
      float maxVal = 0.0f;
      for (size_t i = 0; i < playbackSamples; i++) {
        if (fabsf(playbackFloat[i]) > maxVal) maxVal = fabsf(playbackFloat[i]);
      }
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[Playback Conv #%d] samples=%zu, maxF32=%.6f, format=%d",
          convDebugCount, playbackSamples, maxVal, pDevice->playback.format);
    }
#endif
    if (pDevice->playback.format == ma_format_s16) {
      // Convert f32 -> s16
      ma_pcm_f32_to_s16(pOutput, playbackFloat, playbackSamples, ma_dither_mode_none);
    } else if (pDevice->playback.format == ma_format_s32) {
      // Convert f32 -> s32 (some HALs use 32-bit integer)
      ma_pcm_f32_to_s32(pOutput, playbackFloat, playbackSamples, ma_dither_mode_none);
    } else {
      // f32 output - direct copy
      memcpy(pOutput, playbackFloat, playbackSamples * sizeof(float));
    }
  }
  // =========================================================================
  // NON-SLAVE MODE: Original monitoring passthrough (if not in slave mode)
  // =========================================================================
  else if (userData->monitoringEnabled && pOutput != nullptr &&
           captured != nullptr) {
    float *inputFloat = captured;
    float *outputFloat = (float *)pOutput;
    // Use actual device capture channels for consistency
    // (captureChannels is already defined at top of function)

    if (captureChannels == 2) {
      // Stereo input - apply monitoring mode
      switch (userData->monitoringMode) {
      case 0: // Stereo - normal passthrough at 100%
      {
        int channelsToCopy = std::min(captureChannels, playbackChannels);
        for (ma_uint32 i = 0; i < frameCount * channelsToCopy; i++) {
            outputFloat[i] = inputFloat[i];
        }
      } break;
      case 1: // LM - Left channel at 100% to both outputs
        for (ma_uint32 i = 0; i < frameCount; i++) {
          float leftSample = inputFloat[i * 2];
          outputFloat[i * 2] = leftSample;     // Left output
          outputFloat[i * 2 + 1] = leftSample; // Right output
        }
        break;
      case 2: // RM - Right channel at 100% to both outputs
        for (ma_uint32 i = 0; i < frameCount; i++) {
          float rightSample = inputFloat[i * 2 + 1];
          outputFloat[i * 2] = rightSample;     // Left output
          outputFloat[i * 2 + 1] = rightSample; // Right output
        }
        break;
      case 3: // M - Mono mix at 50% per channel to both outputs
        for (ma_uint32 i = 0; i < frameCount; i++) {
          float monoSample =
              inputFloat[i * 2] * 0.5f + inputFloat[i * 2 + 1] * 0.5f;
          outputFloat[i * 2] = monoSample;     // Left output
          outputFloat[i * 2 + 1] = monoSample; // Right output
        }
        break;
      }
    } else {
      // Mono input or channel count mismatch - just copy first matching channels
      int channelsToCopy = std::min(captureChannels, playbackChannels);
      for (ma_uint32 i = 0; i < frameCount * channelsToCopy; i++) {
          outputFloat[i] = inputFloat[i];
      }
    }
  }

  // TRANSFORM CAPTURED BUFFER in-place to match monitoring mode
  // This ensures recordings, visualizations, and filters all use the same
  // transformed audio
  // Note: captureChannels is already defined at top of function using actual device value
  if (captureChannels == 2 && userData->monitoringMode != 0) {
    switch (userData->monitoringMode) {
    case 1: // LM - Left to both channels
      for (ma_uint32 i = 0; i < frameCount; i++) {
        captured[i * 2 + 1] = captured[i * 2]; // Copy left to right
      }
      break;
    case 2: // RM - Right to both channels
      for (ma_uint32 i = 0; i < frameCount; i++) {
        captured[i * 2] = captured[i * 2 + 1]; // Copy right to left
      }
      break;
    case 3: // M - Mono mix to both channels
      for (ma_uint32 i = 0; i < frameCount; i++) {
        float monoSample = captured[i * 2] * 0.5f + captured[i * 2 + 1] * 0.5f;
        captured[i * 2] = monoSample;
        captured[i * 2 + 1] = monoSample;
      }
      break;
    }
  }

  // Apply filters (SKIP during calibration to capture raw impulse response and
  // save CPU)
  // Note: mFilters may be null if callback runs before init() completes
  // Use lock-free hasFilters() check to avoid mutex contention in hot path
  if (userData->mFilters != nullptr && userData->mFilters->hasFilters() &&
      !userData->mCalibrationActive) {
#if DEBUG_CALLBACK_FILTERS
    static int filterDebugCounter = 0;
    if (++filterDebugCounter <= 5 || filterDebugCounter % 500 == 0) {
#ifdef _IS_ANDROID_
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[Capture CB #%d] hasFilters=true calibActive=%d",
          filterDebugCounter, userData->mCalibrationActive);
#endif
      aecLog("[Capture CB #%d] hasFilters=true calibActive=%d\n",
             filterDebugCounter, userData->mCalibrationActive);
    }
#endif
    // Set the capture frame count for AEC position-based sync BEFORE processing
    // This is the frame count at the START of this block (before we increment)
    size_t captureFrameCount =
        userData->mTotalFramesCaptured.load(std::memory_order_acquire);
    userData->mFilters->setAecCaptureFrameCount(captureFrameCount);

    // Thread-safe filter processing (protects against concurrent addFilter/removeFilter)
    userData->mFilters->processAllFilters(captured, frameCount,
                                          captureChannels,
                                          userData->deviceConfig.capture.format);
  }
#if DEBUG_CALLBACK_FILTERS
  else {
#ifdef _IS_ANDROID_
    static int nullFilterDebugCounter = 0;
    if (++nullFilterDebugCounter <= 5) {
      __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
          "[Capture CB] mFilters is NULL!");
    }
#endif
  }
#endif

  // Do something with the captured audio data...
  // LOCK-FREE: Write to the current write buffer, then swap atomically
  {
    // Get the buffer we should write to
    float *writeBuffer = (capturedBufferWriteIndex.load(std::memory_order_relaxed) == 0)
                         ? capturedBufferA : capturedBufferB;

    // SAFE COPY: Ensure we don't overflow the fixed-size visualization buffer
    size_t maxFloats = VISUALIZATION_BUFFER_SIZE * 2;
    size_t floatsToCopy = (size_t)frameCount * captureChannels;
    if (floatsToCopy > maxFloats) {
        floatsToCopy = maxFloats;
    }
    if (captured != nullptr) {
      memcpy(writeBuffer, captured, sizeof(float) * floatsToCopy);

      // Atomically swap: make the buffer we just wrote to the "read" buffer
      // and switch to writing to the other buffer next time
      int oldIndex = capturedBufferWriteIndex.load(std::memory_order_relaxed);
      capturedBuffer = writeBuffer;  // Update legacy pointer for readers
      capturedBufferWriteIndex.store(1 - oldIndex, std::memory_order_release);
    }
  }

  // Calibration capture: accumulate samples if calibration is active
  if (userData->mCalibrationActive) {
    std::lock_guard<std::mutex> lock(userData->mCalibrationMutex);
    size_t samplesToCapture = frameCount;
    size_t spaceLeft =
        userData->mCalibrationBuffer.size() - userData->mCalibrationWritePos;
    if (samplesToCapture > spaceLeft) {
      samplesToCapture = spaceLeft;
    }
    if (samplesToCapture > 0) {
      // Copy to calibration buffer (Mono Downmix)
      // Use captureChannels (actual device value) for correct buffer interpretation
#if DEBUG_CALLBACK_CALIBRATION
      float debugBatchSum = 0.0f;
#endif
      for (size_t i = 0; i < samplesToCapture; ++i) {
        float sample;
        if (captureChannels >= 2) {
          // Downmix stereo to mono: (L + R) * 0.5
          sample = (captured[i * captureChannels] + captured[i * captureChannels + 1]) * 0.5f;
        } else {
          sample = captured[i * captureChannels]; // Already mono
        }
        userData->mCalibrationBuffer[userData->mCalibrationWritePos + i] = sample;
#if DEBUG_CALLBACK_CALIBRATION
        debugBatchSum += fabsf(sample);
#endif
      }

#if DEBUG_CALLBACK_CALIBRATION
      static int logCounter = 0;
      if (++logCounter % 100 == 0) {
        printf("[Calibration Capture] Added %zu samples. Avg energy in batch: "
               "%.6f\n",
               samplesToCapture, debugBatchSum / (samplesToCapture + 0.0001f));
      }
#endif

      userData->mCalibrationWritePos += samplesToCapture;
    }
  }

  // Calculate energy for FFT visualization
  // NOTE: captured is ALWAYS f32 after conversion (lines 203-230), regardless of device format
  if (captured != nullptr)
    calculateEnergy(captured, frameCount, captureChannels);

  // Stream the audio data?
  if (userData->isStreamingData && nativeStreamDataCallback != nullptr) {
    const unsigned char *data = (const unsigned char *)captured;
    // Calculate total size in bytes considering frame size
    // IMPORTANT: captured is ALWAYS f32 after format conversion (lines 190-216),
    // so we must use sizeof(float), NOT bytesPerSample (which is the native format)
    int frameSize = sizeof(float) * captureChannels;
    int dataSize = frameCount * frameSize;

    // Add new data to the stream buffer
    streamBuffer->insert(streamBuffer->end(), data, data + dataSize);

    // Calculate target buffer size in bytes
    int targetBufferSize = STREAM_BUFFER_SIZE * frameSize;

    // If we've reached the target buffer size, send the data
    if (streamBuffer->size() >= targetBufferSize) {
      // Create a copy of the data to send
      auto *dataCopy = new unsigned char[targetBufferSize];
      memcpy(dataCopy, streamBuffer->data(), targetBufferSize);

      // Send copy to Dart - it will be responsible for freeing the memory
      nativeStreamDataCallback(dataCopy, targetBufferSize);

      // Remove sent data and keep remaining data
      if (streamBuffer->size() > targetBufferSize) {
        std::vector<unsigned char> remaining(
            streamBuffer->begin() + targetBufferSize, streamBuffer->end());
        *streamBuffer = std::move(remaining);
      } else {
        streamBuffer->clear();
      }
    }
  }

  // Detect silence - captured is always f32 after conversion
  if (userData->isDetectingSilence && captured != nullptr) {
    detectSilence(userData);

    // Copy current buffer to circularBuffer
    if (delayed_silence_started && userData->isRecording &&
        userData->secondsOfAudioToWriteBefore > 0) {
      std::vector<float> values(captured, captured + frameCount * captureChannels);
      circularBuffer.get()->push(values);
    }

    if (!delayed_silence_started && userData->isRecording &&
        !userData->isRecordingPaused) {
#ifdef _IS_ANDROID_
      static int wavWriteDebugCount = 0;
      if (++wavWriteDebugCount <= 5) {
        float maxVal = 0.0f;
        for (ma_uint32 i = 0; i < frameCount * captureChannels && i < 100; i++) {
          if (fabsf(captured[i]) > maxVal) maxVal = fabsf(captured[i]);
        }
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
            "[WAV WRITE #%d] frames=%u, channels=%d, samples=%u, maxF32=%.6f",
            wavWriteDebugCount, frameCount, captureChannels, frameCount * captureChannels, maxVal);
      }
#endif
      userData->wav.write(captured, frameCount);
    }
  } else {
    if (userData->isRecording && !userData->isRecordingPaused) {
#ifdef _IS_ANDROID_
      static int wavWriteDebugCount2 = 0;
      if (++wavWriteDebugCount2 <= 5) {
        float maxVal = 0.0f;
        for (ma_uint32 i = 0; i < frameCount * captureChannels && i < 100; i++) {
          if (fabsf(captured[i]) > maxVal) maxVal = fabsf(captured[i]);
        }
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
            "[WAV WRITE2 #%d] frames=%u, channels=%d, samples=%u, maxF32=%.6f",
            wavWriteDebugCount2, frameCount, captureChannels, frameCount * captureChannels, maxVal);
      }
#endif
      userData->wav.write(captured, frameCount);
    }
  }

  // =========================================================================
  // NATIVE SCHEDULER: Process scheduled events at buffer boundaries
  // =========================================================================
  // Calculate buffer start frame (before incrementing the counter)
  int64_t bufferStartFrame = static_cast<int64_t>(
      userData->mTotalFramesCaptured.load(std::memory_order_acquire));
  NativeScheduler::instance().processEvents(bufferStartFrame, frameCount, userData);

  // Increment total frame counter for AEC synchronization
  // This must be done AFTER all processing to mark this block as complete
  userData->mTotalFramesCaptured.fetch_add(frameCount, std::memory_order_release);
}

// /////////////////////////////
// Capture class Implementation
// /////////////////////////////
float waveData[256];
Capture::Capture()
    : isDetectingSilence(false), silenceThresholdDb(-40.0f),
      silenceDuration(2.0f), secondsOfAudioToWriteBefore(0.0f),
      isRecording(false), isRecordingPaused(false), isStreamingData(false),
      monitoringEnabled(false), monitoringMode(0), mInited(false),
      mCalibrationWritePos(0), mCalibrationActive(false),
      mContextInited(false) {
  memset(waveData, 0, sizeof(float) * 256);
}

Capture::~Capture() {
  dispose();

#ifdef _IS_WIN_
  // On Windows, the context was kept alive across init/dispose cycles.
  // Clean it up now in the destructor.
  if (mContextInited) {
    printf("[Capture::~Capture] Windows: Cleaning up context in destructor\n");
    fflush(stdout);
    ma_context_uninit(&context);
    mContextInited = false;
  }
#endif
}

std::vector<CaptureDevice> Capture::listCaptureDevices() {
#ifdef _IS_ANDROID_
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "***************** LIST DEVICES START\n");
#else
  printf("***************** LIST DEVICES START\n");
#endif
  std::vector<CaptureDevice> ret;
  if ((result = ma_context_init(NULL, 0, NULL, &context)) != MA_SUCCESS) {
    printf("Failed to initialize context %d\n", result);
    return ret;
  }

  if ((result = ma_context_get_devices(&context, &pPlaybackInfos,
                                       &playbackCount, &pCaptureInfos,
                                       &captureCount)) != MA_SUCCESS) {
    printf("Failed to get devices %d\n", result);
    return ret;
  }

  // Loop over each device info and do something with it. Here we just print
  // the name with their index. You may want
  // to give the user the opportunity to choose which device they'd prefer.
  for (ma_uint32 i = 0; i < captureCount; i++) {
#ifdef _IS_ANDROID_
    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG, "************ Device: %s %d - %s",
        pCaptureInfos[i].isDefault ? " X" : "-", i, pCaptureInfos[i].name);
#else
    printf("************ Device: %s %d - %s\n",
           pCaptureInfos[i].isDefault ? " X" : "-", i, pCaptureInfos[i].name);
#endif
    CaptureDevice cd;
    cd.name = strdup(pCaptureInfos[i].name);
    cd.isDefault = pCaptureInfos[i].isDefault;
    cd.id = i;
    ret.push_back(cd);
  }
#ifdef _IS_ANDROID_
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "***************** LIST DEVICES END\n");
#else
  printf("***************** LIST DEVICES END\n");
#endif
  return ret;
}

CaptureErrors Capture::init(Filters *filters, int deviceID, PCMFormat pcmFormat,
                            unsigned int sampleRate, unsigned int channels,
                            bool captureOnly) {
  printf("[Capture::init] Starting init: deviceID=%d, sampleRate=%u, "
         "channels=%u, captureOnly=%d, mInited=%d\n",
         deviceID, sampleRate, channels, captureOnly, mInited);
  fflush(stdout);

  // Guard against double initialization
  if (mInited) {
    printf("[Capture::init] Already initialized, calling dispose first\n");
    fflush(stdout);
    dispose();
  }

  // Only initialize context if not already initialized (avoid WASAPI deadlock
  // on repeated init/uninit)
  if (!mContextInited) {
#ifdef _IS_WIN_
    printf("[Capture::init] Windows: Initializing WASAPI context...\n");
    fflush(stdout);

    // Windows: Initialize context with WASAPI backend priority for low latency
    ma_context_config contextConfig = ma_context_config_init();
    ma_backend backends[] = {ma_backend_wasapi};

    result = ma_context_init(backends, 1, &contextConfig, &context);
    if (result != MA_SUCCESS) {
      // Fallback to auto backend selection if WASAPI fails
      printf("WASAPI context init failed, falling back to auto backend\n");
      fflush(stdout);
      result = ma_context_init(NULL, 0, &contextConfig, &context);
      if (result != MA_SUCCESS) {
        printf("Failed to initialize audio context.\n");
        fflush(stdout);
        return captureInitFailed;
      }
    } else {
      printf("Initialized with WASAPI backend for low-latency audio\n");
      fflush(stdout);
    }
#else
    // Other platforms: Use auto backend selection
    result = ma_context_init(NULL, 0, NULL, &context);
    if (result != MA_SUCCESS) {
      printf("Failed to initialize audio context.\n");
      return captureInitFailed;
    }
#endif
    mContextInited = true;
    printf("[Capture::init] Context initialized, mContextInited=true\n");
    fflush(stdout);
  } else {
    printf("[Capture::init] Reusing existing context (mContextInited=true)\n");
    fflush(stdout);
  }

  // Choose device mode based on captureOnly parameter:
  // - captureOnly=true: Use capture-only mode when SoLoud has its own playback device
  //   This prevents two playback devices from competing (which causes grainy audio)
  // - captureOnly=false: Use duplex mode for slave mode where recorder drives SoLoud output
  if (captureOnly) {
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] Using CAPTURE-ONLY mode (SoLoud has own playback)");
#else
    printf("[Capture::init] Using CAPTURE-ONLY mode (SoLoud has own playback)\n");
    fflush(stdout);
#endif
    deviceConfig = ma_device_config_init(ma_device_type_capture);
  } else {
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] Using DUPLEX mode (slave mode for AEC)");
#else
    printf("[Capture::init] Using DUPLEX mode (slave mode for AEC)\n");
    fflush(stdout);
#endif
    deviceConfig = ma_device_config_init(ma_device_type_duplex);
  }

  // Request low-latency mode - critical for real-time audio!
  // Without this, Android AAudio defaults to high-latency mode (~300ms)
  deviceConfig.performanceProfile = ma_performance_profile_low_latency;

#ifdef _IS_ANDROID_
  // Android AAudio FAST PATH configuration
  // HAL fast capture uses 240-frame bursts (5ms @ 48kHz) - request 2 bursts (480 frames = 10ms)
  // This matches the hardware burst size while keeping latency minimal
  deviceConfig.periodSizeInFrames = 480;  // 2 x 240-frame burst = 10ms @ 48kHz

  // AAudio-specific settings for FAST PATH (Mode 12):
  // VOICE_RECOGNITION preset is designed for low-latency speech recognition
  // - No AEC/NS processing (unlike VOICE_COMMUNICATION)
  // - No effects (unlike DEFAULT which may apply system effects)
  // - Should enable MMAP fast path
  deviceConfig.aaudio.inputPreset = ma_aaudio_input_preset_voice_recognition;  // Low-latency speech path
  deviceConfig.aaudio.usage = ma_aaudio_usage_game;                            // GAME bypasses Dolby/effects!
  deviceConfig.aaudio.contentType = ma_aaudio_content_type_music;

  // Request EXCLUSIVE mode - required for true MMAP fast path
  deviceConfig.capture.shareMode = ma_share_mode_exclusive;
  deviceConfig.playback.shareMode = ma_share_mode_exclusive;

  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
      "[Capture::init] AAudio: inputPreset=VOICE_RECOGNITION, usage=GAME, sharingMode=EXCLUSIVE (for Mode 12)");
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
      "[Capture::init] AAudio: periodSize=480 (2 bursts, 10ms @ 48kHz)");
#else
  // Non-Android: Set consistent period size for capture and playback
  deviceConfig.periodSizeInFrames = BUFFER_SIZE;
#endif

#ifdef _IS_WIN_
  // WASAPI-specific low-latency configuration
  // noAutoConvertSRC: Use miniaudio's internal resampler instead of Windows
  // Audio Client's This enables low-latency shared mode even when app sample
  // rate != device sample rate
  deviceConfig.wasapi.noAutoConvertSRC = MA_TRUE;
  // deviceConfig.wasapi.shareMode = ma_share_mode_shared; // Removed in newer
  // miniaudio versions (shared is default)
  printf("WASAPI low-latency config: noAutoConvertSRC=TRUE, buffer=%d frames "
         "(%.2fms @ %dHz)\n",
         BUFFER_SIZE, (BUFFER_SIZE * 1000.0f) / sampleRate, sampleRate);
#endif

  if (deviceID != -1) {
    auto devices = listCaptureDevices();
    if (devices.size() == 0 || deviceID >= devices.size())
      return captureInitFailed;
    deviceConfig.capture.pDeviceID = &pCaptureInfos[deviceID].id;
  }

  ma_format format;
  switch (pcmFormat) {
  case PCMFormat::pcm_u8:
    format = ma_format_u8;
    bytesPerSample = 1;
    break;
  case PCMFormat::pcm_s16:
    format = ma_format_s16;
    bytesPerSample = 2;
    break;
  case PCMFormat::pcm_s24:
    format = ma_format_s24;
    bytesPerSample = 3;
    break;
  case PCMFormat::pcm_s32:
    format = ma_format_s32;
    bytesPerSample = 4;
    break;
  case PCMFormat::pcm_f32:
    format = ma_format_f32;
    bytesPerSample = 4;
    break;
  case PCMFormat::pcm_unknown:
    // Let the system choose optimal format (per AAudio best practices)
    format = ma_format_unknown;
    bytesPerSample = 0;  // Will be set after device init based on actual format
    break;
  default:
    return captureInitFailed;
  }

#ifdef _IS_ANDROID_
  // Android with format=unknown: Configure for AAudio Fast Capture (low-latency)
  // Key requirements for MediaTek Helio G88 Fast Capture path:
  // - PCM_16_BIT format (NOT float32 - HAL only supports s16 on fast path)
  // - 48kHz sample rate (exact match required)
  // - VOICE_RECOGNITION input preset (enables AUDIO_INPUT_FLAG_FAST)
  // - Stereo capture (works per dumpsys evidence)
  if (pcmFormat == PCMFormat::pcm_unknown) {
    // Force PCM_16_BIT (s16) for CAPTURE only - required for Fast Capture path on MediaTek
    // Evidence from dumpsys of working app (com.zuidsoft.looper):
    // - Capture: HAL format 0x1 (16-bit), Processing format 0x1 (16-bit), HAL frame count 240
    // - Playback: HAL format 0x3 (32-bit), Processing format 0x5 (float) - DIFFERENT from capture!
    // The capture and playback formats do NOT need to match.
    // DUPLEX MODE: Both capture and playback must use same format for miniaudio
    // The Oboe example uses separate streams (different formats), but we use duplex
    // So we force s16 for both - the callback converts s16↔f32 for internal processing
    deviceConfig.capture.format = ma_format_s16;       // PCM_16_BIT - required for Fast Capture path
    deviceConfig.capture.channels = 2;                 // STEREO capture
    deviceConfig.sampleRate = 48000;                   // EXPLICIT 48kHz
    deviceConfig.playback.format = ma_format_s16;      // PCM_16_BIT - must match capture for duplex!
    deviceConfig.playback.channels = 2;                // Stereo playback
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] AAudio FAST MODE (DUPLEX): capture=S16, playback=S16, channels=STEREO, rate=48000, captureOnly=%d",
        captureOnly);
  } else {
    // Specific format requested - use provided values
    // Note: This may not achieve low-latency mode if values don't match device native config
    deviceConfig.capture.format = format;
    deviceConfig.capture.channels = channels;
    deviceConfig.sampleRate = sampleRate;
    deviceConfig.playback.format = format;
    deviceConfig.playback.channels = channels;  // Match capture for consistency
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] AAudio explicit mode: format=%d, channels=%d, rate=%d",
        format, channels, sampleRate);
  }
#else
  // Non-Android: use provided values
  deviceConfig.capture.format = format;
  deviceConfig.capture.channels = channels;
  deviceConfig.sampleRate = sampleRate;
  deviceConfig.playback.format = format;
  deviceConfig.playback.channels = channels;
#endif
  deviceConfig.dataCallback = data_callback;
  deviceConfig.pUserData = this;

  printf("[Capture::init] Calling ma_device_init...\n");
  fflush(stdout);

  result = ma_device_init(&context, &deviceConfig, &device);
  if (result != MA_SUCCESS) {
    printf("Failed to initialize capture device. Error: %d\n", result);
    fflush(stdout);
    ma_context_uninit(&context);
    return captureInitFailed;
  }

  printf("[Capture::init] Device initialized successfully\n");

  // If format was unknown, now set bytesPerSample based on actual device format
  if (bytesPerSample == 0) {
    switch (device.capture.format) {
      case ma_format_u8:  bytesPerSample = 1; break;
      case ma_format_s16: bytesPerSample = 2; break;
      case ma_format_s24: bytesPerSample = 3; break;
      case ma_format_s32: bytesPerSample = 4; break;
      case ma_format_f32: bytesPerSample = 4; break;
      default: bytesPerSample = 2; break;  // Safe fallback
    }
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] System chose format=%d, bytesPerSample=%d",
        device.capture.format, bytesPerSample);
#else
    printf("[Capture::init] System chose format=%d, bytesPerSample=%d\n",
           device.capture.format, bytesPerSample);
#endif
  }

  // Pre-allocate conversion buffers if needed (use ACTUAL device period, not config)
  ma_uint32 actualPeriod = device.capture.internalPeriodSizeInFrames;
  if (actualPeriod == 0) actualPeriod = 512;  // Fallback if not reported

  // Capture conversion buffer: s16/s24/s32 -> f32
  if (device.capture.format != ma_format_f32) {
      mConversionBuffer.resize(actualPeriod * device.capture.channels);
  }

  // Playback conversion buffer: f32 -> s16 (for low-latency fast path)
  // Always allocate since slave mode needs f32 buffer for SoLoud output before conversion
  mPlaybackBuffer.resize(actualPeriod * device.playback.channels);

  printf("[Capture::init] ACTUAL device params: sampleRate=%u, capture.channels=%u, playback.channels=%u, capture.format=%d, playback.format=%d\n",
         device.sampleRate, device.capture.channels, device.playback.channels, device.capture.format, device.playback.format);
  printf("[Capture::init] REQUESTED: sampleRate=%u, capture.channels=%u, playback.channels=%u\n",
         sampleRate, channels, channels);
  fflush(stdout);

#ifdef _IS_ANDROID_
  // CRITICAL: Set buffer size to 2x burst for low-latency (per Oboe/AAudio best practices)
  // Default AAudio buffer is much higher than optimal
  // https://developer.android.com/games/sdk/oboe/low-latency-audio#double-buffering
  // Note: We use miniaudio's dynamically loaded function pointers since these APIs require API 26+
  typedef int32_t (*PFN_AAudioStream_getFramesPerBurst)(void* stream);
  typedef int32_t (*PFN_AAudioStream_setBufferSizeInFrames)(void* stream, int32_t numFrames);
  typedef int32_t (*PFN_AAudioStream_getBufferSizeInFrames)(void* stream);

  // Use miniaudio's AAudio library handle for dlsym (not RTLD_DEFAULT which won't find dynamically loaded libs)
  void* aaudioLib = context.aaudio.hAAudio;
  auto getFramesPerBurst = (PFN_AAudioStream_getFramesPerBurst)context.aaudio.AAudioStream_getFramesPerBurst;
  auto setBufferSizeInFrames = aaudioLib ? (PFN_AAudioStream_setBufferSizeInFrames)dlsym(aaudioLib, "AAudioStream_setBufferSizeInFrames") : nullptr;
  auto getBufferSizeInFrames = aaudioLib ? (PFN_AAudioStream_getBufferSizeInFrames)dlsym(aaudioLib, "AAudioStream_getBufferSizeInFrames") : nullptr;

  if (getFramesPerBurst && setBufferSizeInFrames && getBufferSizeInFrames) {
    if (device.aaudio.pStreamCapture != nullptr) {
      int32_t burstSize = getFramesPerBurst(device.aaudio.pStreamCapture);
      int32_t optimalBufferSize = burstSize * 2;  // Double buffering
      int32_t setResult = setBufferSizeInFrames(device.aaudio.pStreamCapture, optimalBufferSize);
      int32_t actualBufferSize = getBufferSizeInFrames(device.aaudio.pStreamCapture);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[LOW-LATENCY] Capture: burstSize=%d, requested=%d, actual=%d, result=%d",
          burstSize, optimalBufferSize, actualBufferSize, setResult);
    }
    if (device.aaudio.pStreamPlayback != nullptr) {
      int32_t burstSize = getFramesPerBurst(device.aaudio.pStreamPlayback);
      int32_t optimalBufferSize = burstSize * 2;  // Double buffering
      int32_t setResult = setBufferSizeInFrames(device.aaudio.pStreamPlayback, optimalBufferSize);
      int32_t actualBufferSize = getBufferSizeInFrames(device.aaudio.pStreamPlayback);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[LOW-LATENCY] Playback: burstSize=%d, requested=%d, actual=%d, result=%d",
          burstSize, optimalBufferSize, actualBufferSize, setResult);
    }
  } else {
    __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
        "[LOW-LATENCY] AAudio buffer APIs not available (API < 26)");
  }

  // CRITICAL: Verify AAudio performance mode and sharing mode
  // perfMode: 12 = LOW_LATENCY, 10 = NONE
  // sharingMode: 0 = EXCLUSIVE, 1 = SHARED
  typedef int32_t (*PFN_AAudioStream_getPerformanceMode)(void* stream);
  typedef int32_t (*PFN_AAudioStream_getSharingMode)(void* stream);
  typedef int32_t (*PFN_AAudioStream_getInputPreset)(void* stream);

  auto getPerformanceMode = aaudioLib ? (PFN_AAudioStream_getPerformanceMode)dlsym(aaudioLib, "AAudioStream_getPerformanceMode") : nullptr;
  auto getSharingMode = aaudioLib ? (PFN_AAudioStream_getSharingMode)dlsym(aaudioLib, "AAudioStream_getSharingMode") : nullptr;
  auto getInputPreset = aaudioLib ? (PFN_AAudioStream_getInputPreset)dlsym(aaudioLib, "AAudioStream_getInputPreset") : nullptr;

  if (getPerformanceMode) {
    if (device.aaudio.pStreamCapture != nullptr) {
      int32_t perfMode = getPerformanceMode(device.aaudio.pStreamCapture);
      int32_t shareMode = getSharingMode ? getSharingMode(device.aaudio.pStreamCapture) : -1;
      int32_t burstSize = getFramesPerBurst ? getFramesPerBurst(device.aaudio.pStreamCapture) : -1;
      int32_t inputPreset = getInputPreset ? getInputPreset(device.aaudio.pStreamCapture) : -1;
      int32_t bufSize = getBufferSizeInFrames ? getBufferSizeInFrames(device.aaudio.pStreamCapture) : -1;

      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[LOW-LATENCY] Capture: perfMode=%d (12=LOW_LATENCY), sharingMode=%d (0=EXCL, 1=SHARED)", perfMode, shareMode);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[LOW-LATENCY] Capture: framesPerBurst=%d, bufferSize=%d, inputPreset=%d (5=VOICE_RECOG, 6=UNPROCESSED)",
          burstSize, bufSize, inputPreset);

      if (perfMode != 12) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
            "[LOW-LATENCY] WARNING: Capture NOT in low-latency mode! perfMode=%d (expected 12)", perfMode);
      }
      if (shareMode != 0) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
            "[LOW-LATENCY] WARNING: Capture NOT in exclusive mode! sharingMode=%d (requested 0)", shareMode);
      }
    }
    if (device.aaudio.pStreamPlayback != nullptr) {
      int32_t perfMode = getPerformanceMode(device.aaudio.pStreamPlayback);
      int32_t shareMode = getSharingMode ? getSharingMode(device.aaudio.pStreamPlayback) : -1;
      int32_t burstSize = getFramesPerBurst ? getFramesPerBurst(device.aaudio.pStreamPlayback) : -1;
      int32_t bufSize = getBufferSizeInFrames ? getBufferSizeInFrames(device.aaudio.pStreamPlayback) : -1;

      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[LOW-LATENCY] Playback: perfMode=%d (12=LOW_LATENCY), sharingMode=%d (0=EXCL, 1=SHARED)", perfMode, shareMode);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[LOW-LATENCY] Playback: framesPerBurst=%d, bufferSize=%d", burstSize, bufSize);

      if (perfMode != 12) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
            "[LOW-LATENCY] WARNING: Playback NOT in low-latency mode! perfMode=%d (expected 12)", perfMode);
      }
    }
  } else {
    __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
        "[LOW-LATENCY] AAudioStream_getPerformanceMode not available (API < 28)");
  }

  // Log ACTUAL vs REQUESTED configuration for debugging
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
      "[LOW-LATENCY] ACTUAL config: capture.ch=%u playback.ch=%u rate=%u capture.fmt=%d playback.fmt=%d",
      device.capture.channels, device.playback.channels, device.sampleRate,
      device.capture.format, device.playback.format);
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
      "[LOW-LATENCY] REQUESTED config: capture.ch=%u playback.ch=%u rate=%u (0=native)",
      deviceConfig.capture.channels, deviceConfig.playback.channels, deviceConfig.sampleRate);

  // Log actual AAudio latency for debugging
  ma_uint32 capturePeriod = device.capture.internalPeriodSizeInFrames;
  ma_uint32 playbackPeriod = device.playback.internalPeriodSizeInFrames;
  float captureLatencyMs = (float)capturePeriod * 1000.0f / device.sampleRate;
  float playbackLatencyMs = (float)playbackPeriod * 1000.0f / device.sampleRate;
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
      "[LATENCY DIAG] capture: period=%u frames (%.2fms), playback: period=%u frames (%.2fms) @ %uHz",
      capturePeriod, captureLatencyMs, playbackPeriod, playbackLatencyMs, device.sampleRate);
#endif

  // Warn if actual sample rate differs from requested
  if (device.sampleRate != sampleRate) {
    printf("[Capture::init] WARNING: Actual sample rate (%u) differs from requested (%u)!\n",
           device.sampleRate, sampleRate);
    fflush(stdout);
  }

  mInited = true;
  mFilters = filters;
  return captureNoError;
}

void Capture::dispose() {
  printf("[Capture::dispose] Starting dispose, mInited=%d\n", mInited);
  fflush(stdout);

  if (!mInited) {
    printf("[Capture::dispose] Not initialized, skipping dispose\n");
    fflush(stdout);
    return;
  }

  mInited = false;
  wav.close();
  if (!circularBuffer)
    circularBuffer.reset();
  if (!streamBuffer)
    streamBuffer.reset();
  isRecording = false;

  printf("[Capture::dispose] Calling ma_device_uninit...\n");
  fflush(stdout);

  ma_device_uninit(&device);

#ifdef _IS_WIN_
  // On Windows/WASAPI, do NOT call ma_context_uninit() here!
  // Repeatedly calling ma_context_init/uninit causes deadlocks.
  // The context is kept alive and reused across init/dispose cycles.
  // It will be cleaned up in the destructor.
  printf("[Capture::dispose] Windows: context kept alive for reuse\n");
  fflush(stdout);
#else
  // On other platforms, uninit context normally
  printf("[Capture::dispose] Calling ma_context_uninit...\n");
  fflush(stdout);
  ma_context_uninit(&context);
  mContextInited = false;
#endif

  printf("[Capture::dispose] Dispose complete\n");
  fflush(stdout);
}

bool Capture::isInited() { return mInited; }

bool Capture::isDeviceStarted() {
  ma_device_state result = ma_device_get_state(&device);
  return result == ma_device_state_started;
}

CaptureErrors Capture::start() {
  if (!mInited)
    return captureNotInited;

  result = ma_device_start(&device);
  if (result != MA_SUCCESS) {
    ma_device_uninit(&device);
    printf("Failed to start device.\n");
    return failedToStartDevice;
  }
  return captureNoError;
}

void Capture::stop() { ma_device_stop(&device); }

void Capture::startStreamingData() {
  if (!streamBuffer)
    streamBuffer.reset();
  streamBuffer = std::make_unique<std::vector<unsigned char>>();
  streamBuffer->reserve(STREAM_BUFFER_SIZE * 6);
  isStreamingData = true;
}

void Capture::stopStreamingData() {
  isStreamingData = false;
  if (!streamBuffer)
    streamBuffer.reset();
}

void Capture::setSilenceDetection(bool enable) {
  this->isDetectingSilence = enable;
}

void Capture::setSilenceThresholdDb(float silenceThresholdDb) {
  this->silenceThresholdDb = silenceThresholdDb;
}

void Capture::setSilenceDuration(float silenceDuration) {
  this->silenceDuration = silenceDuration;
}

void Capture::setSecondsOfAudioToWriteBefore(
    float secondsOfAudioToWriteBefore) {
  this->secondsOfAudioToWriteBefore = secondsOfAudioToWriteBefore;
  // Use ACTUAL device values (deviceConfig may have channels=0 in Android auto mode)
  int channels = device.capture.channels;
  if (channels < 1) channels = 1;  // Safety fallback
  ma_uint32 sampleRate = device.sampleRate > 0 ? device.sampleRate : 48000;
  ma_uint32 frameCount =
      (ma_uint32)(secondsOfAudioToWriteBefore * channels * sampleRate);
  frameCount = (frameCount >> 1) << 1;
  if (!circularBuffer)
    circularBuffer.reset();
  circularBuffer = std::make_unique<CircularBuffer<float>>(frameCount);
}

CaptureErrors Capture::startRecording(const char *path) {
  if (!mInited)
    return captureNotInited;

  // IMPORTANT: Create a config with ACTUAL device values, not configured ones
  // In Android auto mode, deviceConfig.capture.channels may be 0
  ma_device_config actualConfig = deviceConfig;
  // Use f32 format for WAV - the captured buffer is ALWAYS converted to f32
  // regardless of device format (s16/s32 are converted in data_callback lines 203-246)
  actualConfig.capture.format = ma_format_f32;
  actualConfig.capture.channels = device.capture.channels;
  actualConfig.sampleRate = device.sampleRate;

#ifdef _IS_ANDROID_
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
      "[WAV INIT] path=%s, format=f32(%d), channels=%d, sampleRate=%d",
      path, ma_format_f32, actualConfig.capture.channels, actualConfig.sampleRate);
#endif
  printf("[WAV INIT] path=%s, format=f32(%d), channels=%d, sampleRate=%d\n",
      path, ma_format_f32, actualConfig.capture.channels, actualConfig.sampleRate);

  CaptureErrors result = wav.init(path, actualConfig);
  if (result != captureNoError)
    return result;
  setSecondsOfAudioToWriteBefore(secondsOfAudioToWriteBefore);
  isRecording = true;
  isRecordingPaused = false;
  return captureNoError;
}

void Capture::setPauseRecording(bool pause) {
  if (!mInited || !isRecording)
    return;
  isRecordingPaused = pause;
}

void Capture::stopRecording() {
  if (!mInited || !isRecording)
    return;
  wav.close();
  circularBuffer.reset();
  isRecording = false;
}

void Capture::writePrerollToWav(const float* samples, size_t numSamples) {
  if (!mInited || !isRecording || samples == nullptr || numSamples == 0)
    return;

  // Get actual channel count for frame calculation
  unsigned int channels = getCaptureChannels();
  if (channels < 1) channels = 1;

  size_t frameCount = numSamples / channels;
  wav.write((void*)samples, frameCount);
  printf("[Capture] Wrote %zu preroll frames (%zu samples) to WAV\n", frameCount, numSamples);
}

/// @brief Shrinks the captured audio buffer to 256 floats.
/// @param inputBuffer The captured audio buffer.
/// @param outputBuffer The output buffer.
/// @param channels The number of channels.
void shrink_buffer(float *inputBuffer, float *outputBuffer, int channels) {
  for (int i = 0; i < 256; ++i) {
    if (channels == 1) {
      outputBuffer[i] = inputBuffer[i * channels];
    } else {
      outputBuffer[i] =
          (inputBuffer[i * channels] + inputBuffer[i * channels + 1]) * 0.5f;
    }
  }
}

float *Capture::getWave(bool *isTheSameAsBefore) {
  float currentWave[256];

  // LOCK-FREE: Read from the stable buffer (capturedBuffer points to the
  // buffer that was last fully written by the audio callback)
  {
    // IMPORTANT: Use ACTUAL device channels, not configured (which may be 0 in auto mode)
    int channels = device.capture.channels;
    if (channels < 1) channels = 1;  // Safety fallback
    shrink_buffer(capturedBuffer, currentWave, channels);
  }

  if (memcmp(waveData, currentWave, sizeof(waveData)) != 0) {
    *isTheSameAsBefore = false;
  } else {
    *isTheSameAsBefore = true;
  }
  memcpy(waveData, currentWave, sizeof(waveData));
  return waveData;
}

float Capture::getVolumeDb() { return energy_db; }

void Capture::startCalibrationCapture(size_t maxSamples) {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  mCalibrationBuffer.resize(maxSamples, 0.0f);
  mCalibrationWritePos = 0;
  mCalibrationActive = true;
}

void Capture::stopCalibrationCapture() {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  mCalibrationActive = false;
}

size_t Capture::readCalibrationSamples(float *dest, size_t maxSamples) {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  size_t samplesToRead = std::min(maxSamples, mCalibrationWritePos);
  if (samplesToRead > 0 && dest != nullptr) {
    memcpy(dest, mCalibrationBuffer.data(), samplesToRead * sizeof(float));
  }
  return samplesToRead;
}

bool Capture::isCalibrationCaptureActive() const { return mCalibrationActive; }
