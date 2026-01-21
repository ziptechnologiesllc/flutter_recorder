#include "capture.h"
#include "circular_buffer.h"
#include "soloud_slave_bridge.h"
#include "filters/aec/reference_buffer.h"

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

#define LOG_TAG "FlutterRecorder"
#endif

// 128 frames for ultra-low latency monitoring (~2.67ms @ 48kHz)
#define BUFFER_SIZE 128                      // Buffer length in frames
#define STREAM_BUFFER_SIZE (BUFFER_SIZE * 2) // Buffer length in frames
#define MOVING_AVERAGE_SIZE 4                // Moving average window size
#define VISUALIZATION_BUFFER_SIZE                                              \
  8192 // Larger buffer for waveform visualization
float capturedBuffer[VISUALIZATION_BUFFER_SIZE *
                     2];              // Captured audio buffer for visualization
std::mutex capturedBufferMutex;       // Mutex for protecting capturedBuffer
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

void calculateEnergy(float *captured, ma_uint32 frameCount) {
  static float moving_average[MOVING_AVERAGE_SIZE] = {
      0};                       // Moving average window
  static int average_index = 0; // Circular buffer index
  float sum = 0.0f;

  // Calculate the average energy of the current buffer
  for (int i = 0; i < frameCount; i++) {
    sum += captured[i] * captured[i];
  }
  float average_energy = sum / frameCount;

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
          userData->wav.write(data.data(),
                              data.size() /
                                  userData->deviceConfig.capture.channels);
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

  // Debug log first few callbacks to help diagnose channel issues
  static int channelDebugCount = 0;
  if (channelDebugCount < 5) {
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                        "[Capture CB] ACTUAL channels: capture=%d, playback=%d (configured: capture=%d, playback=%d)",
                        captureChannels, playbackChannels,
                        userData->deviceConfig.capture.channels,
                        userData->deviceConfig.playback.channels);
#else
    printf("[Capture CB] ACTUAL channels: capture=%d, playback=%d (configured: capture=%d, playback=%d)\n",
           captureChannels, playbackChannels,
           userData->deviceConfig.capture.channels,
           userData->deviceConfig.playback.channels);
    fflush(stdout);
#endif
    channelDebugCount++;
  }

  // Debug: Track callback timing to detect sample rate issues
  static size_t totalCallbackFrames = 0;
  static auto startTime = std::chrono::steady_clock::now();
  static int debugCallbackCount = 0;
  totalCallbackFrames += frameCount;
  debugCallbackCount++;

  // Log every 5 seconds worth of frames (at expected 48kHz)
  if (debugCallbackCount % 1875 == 0) { // ~5s at 128 frames/callback @ 48kHz
    auto now = std::chrono::steady_clock::now();
    double elapsedSec = std::chrono::duration<double>(now - startTime).count();
    double actualRate = totalCallbackFrames / elapsedSec;
    aecLog("[Capture CB TIMING] frames=%zu elapsed=%.2fs rate=%.0fHz (expected 48000)\n",
           totalCallbackFrames, elapsedSec, actualRate);
  }

  // =========================================================================
  // SLAVE MODE: SoLoud output driven by this callback (for AEC clock sync)
  // =========================================================================
  // In slave mode, we call SoLoud's mix function directly instead of SoLoud
  // running its own audio device. This ensures perfect clock synchronization
  // between capture and playback, fixing AEC drift issues on Linux.
  if (soloud_isSlaveMode() && g_soloudSlaveMixCallback != nullptr &&
      pOutput != nullptr) {
    // Get SoLoud's mixed output directly into pOutput
    g_soloudSlaveMixCallback((float *)pOutput, frameCount, playbackChannels);

    // Debug: Check if SoLoud produced any audio
    static int soloudMixDebugCount = 0;
    if (soloudMixDebugCount++ < 10) {
      float maxSample = 0.0f;
      const float *out = (const float *)pOutput;
      for (ma_uint32 i = 0; i < frameCount * playbackChannels; i++) {
        if (fabsf(out[i]) > maxSample) maxSample = fabsf(out[i]);
      }
#ifdef _IS_ANDROID_
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[Capture Slave Mix] frames=%u ch=%d maxSample=%.4f",
                          frameCount, playbackChannels, maxSample);
#endif
    }

    // Mark slave audio as ready after first successful callback
    // This signals that the audio pipeline is flowing and calibration can start
    soloud_setSlaveAudioReady();

    // Write to AEC reference buffer IN THE SAME CALLBACK - guarantees sync!
    // This is the whole point of slave mode: one callback, one clock.
    // Handle channel conversion: pOutput has actual device channels, but AEC buffer
    // may have different channel count (usually mono for AEC purposes).
    static int aecRefBufDebugCount = 0;
    if (aecRefBufDebugCount++ < 5) {
#ifdef _IS_ANDROID_
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[Capture Slave] g_aecReferenceBuffer=%p, playbackCh=%d",
                          (void*)g_aecReferenceBuffer, playbackChannels);
#endif
    }
    if (g_aecReferenceBuffer != nullptr) {
      unsigned int bufferCh = g_aecReferenceBuffer->channels();
      float *outputFloat = (float *)pOutput; // Slave mixer output is always float

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

      // Debug logging (sparse to avoid spam)
      static int slaveLogCount = 0;
      if (slaveLogCount++ < 10 || slaveLogCount % 1000 == 0) {
        aecLog("[Capture Slave] Wrote %u frames to AEC ref buffer (playback=%d, bufferCh=%u), total=%zu\n",
               frameCount, playbackChannels, bufferCh, g_aecReferenceBuffer->getFramesWritten());
      }
    }

    // If monitoring is enabled, ADD the captured input to the SoLoud output
    // (This allows hearing yourself while SoLoud plays)
    if (userData->monitoringEnabled && captured != nullptr && pOutput != nullptr) {
      float *outputFloat = (float *)pOutput;
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
  if (userData->mFilters != nullptr) {
    static int filterDebugCounter = 0;
    filterDebugCounter++;
    size_t filterCount = userData->mFilters->getFilterCount();  // Thread-safe access
    if (filterDebugCounter <= 5 || filterDebugCounter % 500 == 0) {
#ifdef _IS_ANDROID_
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
          "[Capture CB #%d] filters=%zu calibActive=%d",
          filterDebugCounter, filterCount, userData->mCalibrationActive);
#endif
      aecLog("[Capture CB #%d] filters=%zu calibActive=%d\n",
             filterDebugCounter, filterCount, userData->mCalibrationActive);
    }
    if (filterCount > 0 && !userData->mCalibrationActive) {
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
  } else {
#ifdef _IS_ANDROID_
    static int nullFilterDebugCounter = 0;
    if (++nullFilterDebugCounter <= 5) {
      __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
          "[Capture CB] mFilters is NULL!");
    }
#endif
  }

  // Do something with the captured audio data...
  // Protect the write to capturedBuffer
  {
    std::lock_guard<std::mutex> lock(capturedBufferMutex);
    // SAFE COPY: Ensure we don't overflow the fixed-size visualization buffer
    size_t maxFloats = VISUALIZATION_BUFFER_SIZE * 2;
    size_t floatsToCopy = (size_t)frameCount * captureChannels;
    if (floatsToCopy > maxFloats) {
        floatsToCopy = maxFloats;
    }
    memcpy(capturedBuffer, captured, sizeof(float) * floatsToCopy);
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
      float debugBatchSum = 0.0f;

      for (size_t i = 0; i < samplesToCapture; ++i) {
        float sample;
        if (captureChannels >= 2) {
          // Downmix stereo to mono: (L + R) * 0.5
          sample = (captured[i * captureChannels] + captured[i * captureChannels + 1]) * 0.5f;
        } else {
          sample = captured[i * captureChannels]; // Already mono
        }

        userData->mCalibrationBuffer[userData->mCalibrationWritePos + i] =
            sample;
        debugBatchSum += fabsf(sample);
      }

      // Periodic debug logging (once per ~100 calls to avoid spam, assuming
      // frameCount ~512)
      static int logCounter = 0;
      if (++logCounter % 100 == 0) {
        printf("[Calibration Capture] Added %zu samples. Avg energy in batch: "
               "%.6f\n",
               samplesToCapture, debugBatchSum / (samplesToCapture + 0.0001f));
      }

      userData->mCalibrationWritePos += samplesToCapture;
    }
  }

  if (userData->deviceConfig.capture.format == ma_format_f32)
    calculateEnergy(captured, frameCount);

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

  // Detect silence only when using float32
  if (userData->isDetectingSilence &&
      userData->deviceConfig.capture.format == ma_format_f32) {
    detectSilence(userData);

    // Copy current buffer to circularBuffer
    if (delayed_silence_started && userData->isRecording &&
        userData->secondsOfAudioToWriteBefore > 0) {
      std::vector<float> values(captured, captured + frameCount);
      circularBuffer.get()->push(values);
    }

    if (!delayed_silence_started && userData->isRecording &&
        !userData->isRecordingPaused) {
      userData->wav.write(captured, frameCount);
    }
  } else {
    if (userData->isRecording && !userData->isRecordingPaused) {
      userData->wav.write(captured, frameCount);
    }
  }

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
      mContextInited(false), mCalibrationWritePos(0),
      mCalibrationActive(false) {
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
    printf("[Capture::init] Using CAPTURE-ONLY mode (SoLoud has own playback)\n");
    fflush(stdout);
    deviceConfig = ma_device_config_init(ma_device_type_capture);
  } else {
    printf("[Capture::init] Using DUPLEX mode (slave mode for AEC)\n");
    fflush(stdout);
    deviceConfig = ma_device_config_init(ma_device_type_duplex);
  }

  // Request low-latency mode - critical for real-time audio!
  // Without this, Android AAudio defaults to high-latency mode (~300ms)
  deviceConfig.performanceProfile = ma_performance_profile_low_latency;

#ifdef _IS_ANDROID_
  // Android AAudio-specific configuration following AAudio best practices:
  // https://developer.android.com/ndk/guides/audio/aaudio/aaudio
  //
  // KEY INSIGHT: "It is better to let AAudio select these because some
  // combinations of settings are not supported on some devices."
  //
  // For LOW LATENCY, we:
  // 1. Request EXCLUSIVE sharing mode (lowest latency, but may fail)
  // 2. Do NOT specify periodSizeInFrames (let AAudio choose optimal)
  // 3. Do NOT specify format/channels/sampleRate if pcmFormat=unknown (let AAudio choose)
  // 4. Query actual values after stream opens

  deviceConfig.aaudio.usage = ma_aaudio_usage_game;  // Routes through low-latency audio path
  deviceConfig.aaudio.contentType = ma_aaudio_content_type_music;
  deviceConfig.aaudio.inputPreset = ma_aaudio_input_preset_voice_performance;  // Real-time optimized

  // CRITICAL: Request EXCLUSIVE sharing mode for lowest latency
  // From AAudio docs: "Exclusive streams provide the lowest possible latency"
  // miniaudio maps ma_share_mode_exclusive to AAUDIO_SHARING_MODE_EXCLUSIVE
  deviceConfig.aaudio.noAutoStartAfterReroute = MA_TRUE;

  // Allow variable callback size for low-latency (AAudio may give us smaller buffers)
  deviceConfig.noFixedSizedCallback = MA_TRUE;

  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
      "[Capture::init] AAudio config: usage=GAME, content=MUSIC, input=VOICE_PERFORMANCE, noFixedCallback=true");
#else
  // Non-Android platforms: set period size for consistent behavior
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
  // Android with format=unknown: Let AAudio choose EVERYTHING optimal
  // This follows AAudio best practices for achieving low-latency
  if (pcmFormat == PCMFormat::pcm_unknown) {
    deviceConfig.capture.format = ma_format_unknown;  // Let system choose
    deviceConfig.capture.channels = 0;  // Let system choose (0 = default)
    deviceConfig.sampleRate = 0;  // Let system choose native rate
    deviceConfig.playback.format = ma_format_unknown;
    // IMPORTANT: For duplex mode, we MUST request playback channels explicitly
    // Setting to 0 causes AAudio to skip playback entirely, breaking slave mode
    deviceConfig.playback.channels = captureOnly ? 0 : 2;
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] AAudio FULL AUTO mode: format=unknown, capture=0, playback=%d, captureOnly=%d",
        deviceConfig.playback.channels, captureOnly);
  } else {
    // Specific format requested - use provided values
    deviceConfig.capture.format = format;
    deviceConfig.capture.channels = channels;
    deviceConfig.sampleRate = sampleRate;
    deviceConfig.playback.format = format;
    // Android: Force 2 channels for playback output for Fast Path compatibility
    deviceConfig.playback.channels = 2;
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

  // Pre-allocate conversion buffer if needed (use ACTUAL device period, not config)
  if (device.capture.format != ma_format_f32) {
      // Use actual internal period size since we may not have set deviceConfig.periodSizeInFrames (Android auto mode)
      ma_uint32 actualPeriod = device.capture.internalPeriodSizeInFrames;
      if (actualPeriod == 0) actualPeriod = 512;  // Fallback if not reported
      mConversionBuffer.resize(actualPeriod * device.capture.channels);
  }

  printf("[Capture::init] ACTUAL device params: sampleRate=%u, capture.channels=%u, playback.channels=%u, capture.format=%d, playback.format=%d\n",
         device.sampleRate, device.capture.channels, device.playback.channels, device.capture.format, device.playback.format);
  printf("[Capture::init] REQUESTED: sampleRate=%u, capture.channels=%u, playback.channels=%u\n",
         sampleRate, channels, channels);
  fflush(stdout);

#ifdef _IS_ANDROID_
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
  ma_uint32 frameCount =
      (ma_uint32)(secondsOfAudioToWriteBefore * deviceConfig.capture.channels *
                  deviceConfig.sampleRate);
  frameCount = (frameCount >> 1) << 1;
  if (!circularBuffer)
    circularBuffer.reset();
  circularBuffer = std::make_unique<CircularBuffer<float>>(frameCount);
}

CaptureErrors Capture::startRecording(const char *path) {
  if (!mInited)
    return captureNotInited;
  CaptureErrors result = wav.init(path, deviceConfig);
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

  // Protect the read from capturedBuffer
  {
    std::lock_guard<std::mutex> lock(capturedBufferMutex);
    shrink_buffer(capturedBuffer, currentWave, deviceConfig.capture.channels);
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
