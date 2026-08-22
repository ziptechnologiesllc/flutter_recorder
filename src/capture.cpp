#include "capture.h"
#include "audio_engine/audio_engine.h"
#include "auto_record.h"
#include "circular_buffer.h"
#include "native_ring_buffer.h"
#include "soloud_slave_bridge.h"
#include "filters/aec/reference_buffer.h"
#include "native_scheduler.h"
#include "coreaudio_duplex.h" // single-unit iOS/macOS RemoteIO duplex (one clock)

#include "fft/soloud_fft.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <memory.h>
#include <memory>
#include <mutex>
#include <thread>
#include <time.h>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if defined(_IS_WIN_) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h> // FTZ/DAZ control for the audio thread
#endif

// External logging function defined in calibration.cpp
extern void aecLog(const char *fmt, ...);

#ifdef _IS_ANDROID_
#include <android/log.h>
#include <jni.h>
#include <dlfcn.h> // For dlsym to dynamically load AAudio buffer APIs

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
static std::atomic<int> capturedBufferWriteIndex{
    0}; // 0 = writing to A, 1 = writing to B
// Legacy pointer for compatibility (points to readable buffer)
float *capturedBuffer = capturedBufferA;
std::atomic<bool> is_silent{true};     // Initial state
bool delayed_silence_started = false;  // Whether the silence is delayed
std::atomic<float> energy_db{-100.0f}; // Current energy

/// the buffer used for capturing audio.
std::unique_ptr<CircularBuffer<float>> circularBuffer;

/// the buffer used for streaming.
std::unique_ptr<std::vector<unsigned char>> streamBuffer;

namespace {
// ----------------------------------------------------------------------------
// Off-audio-thread stream dispatch. The Dart FFI contract (fresh heap
// pointer per chunk, freed by Dart via flutter_recorder_nativeFree — see
// recorder_io.dart _streamDataCallback) is unchanged; what moves is WHICH
// thread performs the `new[]` + memcpy + NativeCallable dispatch. Doing that
// directly in data_callback (the original design) put a heap allocation on
// the render thread on every ~256-frame chunk during ANY active recording —
// the same class of RT-unsafety already root-caused twice tonight elsewhere
// (RT-thread fprintf/syslog, RT-thread file I/O). Allocator contention grows
// with session length as the heap fragments, which matches the reported
// symptom exactly: fine at first, progressively worse overruns, eventual
// crash. Fix: a wait-free SPSC byte ring (audio thread writes, drops on
// overflow rather than ever blocking) drained by a dedicated worker thread
// that does the allocation/copy/dispatch — mirrors SpectralGovernor's proven
// pattern from the same session.
class StreamDispatchWorker {
public:
  static StreamDispatchWorker &instance() {
    static StreamDispatchWorker w;
    return w;
  }

  void start() {
    bool expected = false;
    if (!mRunning.compare_exchange_strong(expected, true))
      return;
    mWriteIdx.store(0, std::memory_order_relaxed);
    mReadIdx.store(0, std::memory_order_relaxed);
    mDroppedBytes.store(0, std::memory_order_relaxed);
    mWorker = std::thread([this] { workerLoop(); });
  }

  void stop() {
    if (!mRunning.exchange(false))
      return;
    if (mWorker.joinable())
      mWorker.join();
  }

  // Audio-thread call: wait-free. Drops (and counts) bytes if the ring is
  // full rather than ever blocking the render thread.
  void push(const unsigned char *data, size_t n) {
    if (!mRunning.load(std::memory_order_relaxed) || !data || n == 0)
      return;
    const uint64_t w = mWriteIdx.load(std::memory_order_relaxed);
    const uint64_t r = mReadIdx.load(std::memory_order_acquire);
    if (w - r + n > kCapacity) {
      mDroppedBytes.fetch_add(n, std::memory_order_relaxed);
      return;
    }
    for (size_t i = 0; i < n; ++i)
      mRing[(w + i) % kCapacity] = data[i];
    mWriteIdx.store(w + n, std::memory_order_release);
  }

private:
  StreamDispatchWorker() : mRing(kCapacity) {}

  static constexpr size_t kCapacity = 1 << 20; // 1 MiB — many seconds of PCM
  static constexpr int kPollMs = 2; // low latency: Dart consumers stream live

  std::vector<unsigned char> mRing;
  std::atomic<uint64_t> mWriteIdx{0};
  std::atomic<uint64_t> mReadIdx{0};
  std::atomic<uint64_t> mDroppedBytes{0};
  std::atomic<bool> mRunning{false};
  std::thread mWorker;

  void workerLoop() {
    while (mRunning.load(std::memory_order_relaxed)) {
      const size_t targetBufferSize =
          static_cast<size_t>(STREAM_BUFFER_SIZE) * mChunkFrameSize;
      if (targetBufferSize == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
        continue;
      }
      const uint64_t w = mWriteIdx.load(std::memory_order_acquire);
      const uint64_t r = mReadIdx.load(std::memory_order_relaxed);
      if (w - r < targetBufferSize) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
        continue;
      }
      auto *dataCopy = new unsigned char[targetBufferSize];
      for (size_t i = 0; i < targetBufferSize; ++i)
        dataCopy[i] = mRing[(r + i) % kCapacity];
      mReadIdx.store(r + targetBufferSize, std::memory_order_release);
      if (nativeStreamDataCallback)
        nativeStreamDataCallback(dataCopy, static_cast<int>(targetBufferSize));
      else
        delete[] dataCopy; // no Dart listener registered — avoid a leak
    }
  }

public:
  // Bytes-per-frame for the CURRENT stream (channels * sizeof(float)); set
  // once per startStreamingData() call from the audio-thread-known channel
  // count. Plain int, written only while the worker isn't consuming it
  // (before start()/after stop()) — no synchronization needed.
  int mChunkFrameSize = 0;
};
} // namespace

static CaptureErrors setAndroidInputPreset(ma_device_config *config,
                                           int androidInputPreset) {
  switch (androidInputPreset) {
  case 0:
    return captureNoError;
  case 1:
#ifdef _IS_ANDROID_
    config->aaudio.inputPreset = ma_aaudio_input_preset_generic;
    config->opensl.recordingPreset = ma_opensl_recording_preset_generic;
#endif
    return captureNoError;
  case 2:
#ifdef _IS_ANDROID_
    config->aaudio.inputPreset = ma_aaudio_input_preset_camcorder;
    config->opensl.recordingPreset = ma_opensl_recording_preset_camcorder;
#endif
    return captureNoError;
  case 3:
#ifdef _IS_ANDROID_
    config->aaudio.inputPreset = ma_aaudio_input_preset_voice_recognition;
    config->opensl.recordingPreset =
        ma_opensl_recording_preset_voice_recognition;
#endif
    return captureNoError;
  case 4:
#ifdef _IS_ANDROID_
    config->aaudio.inputPreset = ma_aaudio_input_preset_voice_communication;
    config->opensl.recordingPreset =
        ma_opensl_recording_preset_voice_communication;
#endif
    return captureNoError;
  case 5:
#ifdef _IS_ANDROID_
    config->aaudio.inputPreset = ma_aaudio_input_preset_unprocessed;
    config->opensl.recordingPreset =
        ma_opensl_recording_preset_voice_unprocessed;
#endif
    return captureNoError;
  default:
    return captureInitFailed;
  }
}

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
          // Use actual device channels to avoid division by zero
          // (deviceConfig.capture.channels may be 0 in Android auto mode).
          int actualChannels = userData->getCaptureChannels();
          if (actualChannels < 1)
            actualChannels = 1;
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
        if (nativeSilenceChangedCallback != nullptr) {
          float energy_value = energy_db.load();
          nativeSilenceChangedCallback(&delayed_silence_started, &energy_value);
        }
      }
    }
  }
}

// A "frame" is one sample for each channel. For example, in a stereo stream (2
// channels),
// one frame is 2 samples: one for the left, one for the right.
// ── Audio-callback profiling (pops/clicks hunt) ───────────────────────────
//
// Every audio buffer must be filled within the device's buffer period
// (frameCount / sampleRate — ~5.3 ms at 256@48k). If data_callback's
// wall-time exceeds that, the device's *next* buffer is already late →
// underrun → audible pop. These atomics are written only from the audio
// thread (single producer) and read lock-free from Dart via FFI.
//
// steady_clock::now() is a vDSO read on macOS/Linux — no syscall, safe to
// call on the audio thread.
std::atomic<int64_t> g_cbLastMicros{0};    // most recent callback duration
std::atomic<int64_t> g_cbMaxMicros{0};     // worst duration since last reset
std::atomic<int64_t> g_cbBudgetMicros{0};  // buffer period, for reference
std::atomic<int64_t> g_cbTotalCount{0};    // callbacks measured
std::atomic<int64_t> g_cbOverrunCount{0};  // callbacks that blew the budget
std::atomic<int64_t> g_cbNearMissCount{0}; // callbacks over 80% of budget

// Per-section breakdown of the callback: where does the budget actually go?
// Sections (in callback flow order):
//   0 aec   — input format conversion + filter chain (LSAEC lives here)
//   1 ring  — native ring buffer write
//   2 sched — auto-record detector + scheduler events + audio engine tick
//   3 mix   — SoLoud slave mix render
//   4 post  — metronome, AEC reference write, monitoring, output conversion
constexpr int kCbSections = 5;
std::atomic<int64_t> g_cbSectionLastMicros[kCbSections] = {};
std::atomic<int64_t> g_cbSectionMaxMicros[kCbSections] = {};

void captureGetCallbackStats(int64_t *out) {
  if (out == nullptr)
    return;
  out[0] = g_cbLastMicros.load(std::memory_order_relaxed);
  out[1] = g_cbMaxMicros.load(std::memory_order_relaxed);
  out[2] = g_cbBudgetMicros.load(std::memory_order_relaxed);
  out[3] = g_cbOverrunCount.load(std::memory_order_relaxed);
  out[4] = g_cbNearMissCount.load(std::memory_order_relaxed);
  out[5] = g_cbTotalCount.load(std::memory_order_relaxed);
  for (int s = 0; s < kCbSections; ++s) {
    out[6 + s * 2] = g_cbSectionLastMicros[s].load(std::memory_order_relaxed);
    out[7 + s * 2] = g_cbSectionMaxMicros[s].load(std::memory_order_relaxed);
  }
}

void captureResetCallbackStats() {
  g_cbMaxMicros.store(0, std::memory_order_relaxed);
  g_cbOverrunCount.store(0, std::memory_order_relaxed);
  g_cbNearMissCount.store(0, std::memory_order_relaxed);
  g_cbTotalCount.store(0, std::memory_order_relaxed);
  for (int s = 0; s < kCbSections; ++s) {
    g_cbSectionMaxMicros[s].store(0, std::memory_order_relaxed);
  }
}

namespace {
// RAII timer — measures the whole callback regardless of which `return`
// path it takes. Constructed at data_callback entry, records on scope exit.
struct CallbackTimer {
  const std::chrono::steady_clock::time_point start;
  const int64_t budgetMicros;
  // Section marks (flow order); initialized to `start` so a section a return
  // path never reaches reports 0. Set by data_callback as it passes each
  // boundary; the dtor turns consecutive marks into per-section durations.
  std::chrono::steady_clock::time_point markAec;
  std::chrono::steady_clock::time_point markRing;
  std::chrono::steady_clock::time_point markSched;
  std::chrono::steady_clock::time_point markMix;
  CallbackTimer(ma_uint32 frameCount, ma_uint32 sampleRate)
      : start(std::chrono::steady_clock::now()),
        budgetMicros(sampleRate > 0
                         ? static_cast<int64_t>(frameCount) * 1000000 /
                               static_cast<int64_t>(sampleRate)
                         : 0),
        markAec(start), markRing(start), markSched(start), markMix(start) {}
  static void recordSection(int section, int64_t micros) {
    g_cbSectionLastMicros[section].store(micros, std::memory_order_relaxed);
    int64_t prev = g_cbSectionMaxMicros[section].load(std::memory_order_relaxed);
    while (micros > prev &&
           !g_cbSectionMaxMicros[section].compare_exchange_weak(
               prev, micros, std::memory_order_relaxed)) {
    }
  }
  ~CallbackTimer() {
    const auto end = std::chrono::steady_clock::now();
    const int64_t micros =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    const auto us = [](std::chrono::steady_clock::time_point a,
                       std::chrono::steady_clock::time_point b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
          .count();
    };
    // Clamp marks monotone: a mark a return path never reached collapses to
    // the previous one (0-length section) and the remainder lands in "post".
    const auto m0 = std::max(markAec, start);
    const auto m1 = std::max(markRing, m0);
    const auto m2 = std::max(markSched, m1);
    const auto m3 = std::max(markMix, m2);
    recordSection(0, us(start, m0));
    recordSection(1, us(m0, m1));
    recordSection(2, us(m1, m2));
    recordSection(3, us(m2, m3));
    recordSection(4, us(m3, end));
    g_cbLastMicros.store(micros, std::memory_order_relaxed);
    g_cbBudgetMicros.store(budgetMicros, std::memory_order_relaxed);
    g_cbTotalCount.fetch_add(1, std::memory_order_relaxed);
    // Monotonic max ratchet.
    int64_t prevMax = g_cbMaxMicros.load(std::memory_order_relaxed);
    while (micros > prevMax &&
           !g_cbMaxMicros.compare_exchange_weak(prevMax, micros,
                                                std::memory_order_relaxed)) {
    }
    if (budgetMicros > 0) {
      if (micros > budgetMicros) {
        g_cbOverrunCount.fetch_add(1, std::memory_order_relaxed);
      } else if (micros * 5 > budgetMicros * 4) { // > 80% of budget
        g_cbNearMissCount.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
};
} // namespace

// ============================================================================
// AEC DIAGNOSTIC CAPTURE (AEC_DIAG_CAPTURE)
// Records raw SoLoud output (reference) + raw mic, mono, frame-aligned in the
// SAME callback, to memory; flushes once to the app Documents dir when full.
// Lets us pull the actual signals and MEASURE the true loop period (reference
// autocorrelation) and acoustic-delay drift (ref<->mic xcorr) offline — no
// assumptions. Writes are RT-safe (memcpy into a preallocated buffer); the
// one-time file write at the end accepts a single glitch (diagnostic only).
// ============================================================================
#ifdef AEC_DIAG_CAPTURE
#ifdef __APPLE__
#import <Foundation/Foundation.h>
#endif
#include <cstdio>
namespace {
std::vector<float> g_diagRef; // mono reference (SoLoud output)
std::vector<float> g_diagMic; // mono mic
size_t g_diagCap = 0;         // capacity (frames)
size_t g_diagPos = 0;         // frames captured so far
int g_diagState = 0;          // 0 idle, 1 capturing, 2 flushed
int64_t g_diagStartFrame = 0; // engine frame at capture start
int64_t g_diagP = 0;          // loop period at capture start
int64_t g_diagLoopStart = 0;  // loop start frame at capture start
unsigned int g_diagSR = 48000;

std::string diagDocsDir() {
#ifdef __APPLE__
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                                       NSUserDomainMask, YES);
  if ([paths count] > 0)
    return std::string([[paths objectAtIndex:0] UTF8String]) + "/";
#endif
  return "/tmp/";
}

void diagFlush() {
  std::string dir = diagDocsDir();
  FILE *fr = fopen((dir + "diag_ref.raw").c_str(), "wb");
  FILE *fm = fopen((dir + "diag_mic.raw").c_str(), "wb");
  if (fr) {
    fwrite(g_diagRef.data(), sizeof(float), g_diagPos, fr);
    fclose(fr);
  }
  if (fm) {
    fwrite(g_diagMic.data(), sizeof(float), g_diagPos, fm);
    fclose(fm);
  }
  FILE *fmeta = fopen((dir + "diag_meta.txt").c_str(), "w");
  if (fmeta) {
    fprintf(fmeta,
            "frames=%zu sampleRate=%u P=%lld loopStart=%lld startFrame=%lld\n",
            g_diagPos, g_diagSR, (long long)g_diagP, (long long)g_diagLoopStart,
            (long long)g_diagStartFrame);
    fclose(fmeta);
  }
  aecLog("[AEC DIAG] flushed %zu frames to %s (P=%lld start=%lld)\n", g_diagPos,
         dir.c_str(), (long long)g_diagP, (long long)g_diagStartFrame);
}
} // namespace
#endif // AEC_DIAG_CAPTURE

void data_callback(ma_device *pDevice, void *pOutput, const void *pInput,
                   ma_uint32 frameCount) {
  Capture *userData = (Capture *)pDevice->pUserData;
  if (!userData)
    return;

#if defined(_IS_WIN_) && (defined(_M_X64) || defined(_M_IX86))
  // Flush denormals to zero on this WASAPI thread: the AEC's decaying IIR
  // smoothers reach the denormal range during silence, where SSE denormal
  // handling costs ~100x per op — enough to blow the 2.67ms period budget.
  // (Clang on Apple/Android builds runs with FTZ via -ffast-math defaults.)
  static thread_local bool sFtzApplied = false;
  if (!sFtzApplied) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    sFtzApplied = true;
  }
#endif

  // Profile the full callback (all return paths) — see notes above.
  CallbackTimer cbTimer_(frameCount,
                         static_cast<ma_uint32>(pDevice->sampleRate));

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
      ma_pcm_s16_to_f32(userData->mConversionBuffer.data(),
                        (const ma_int16 *)pInput, samplesCount,
                        ma_dither_mode_none);
#ifdef _IS_ANDROID_
      // First 10 callbacks + a ~25s heartbeat: the early callbacks read the
      // duplex ring's silence pre-seek, so only an ONGOING zero peak means
      // the mic path is dead (e.g. Android's record-client silencing, which
      // keeps the stream running and zeroes frames). The xruns counters
      // surface duplex-ring drops (time-compressed audio) in field logs.
      static int captureConvDebugCount = 0;
      ++captureConvDebugCount;
      if (captureConvDebugCount <= 10 || captureConvDebugCount % 2500 == 0) {
        // Peak over the WHOLE buffer (the first-100-samples scan could miss
        // a transient at the tail).
        int16_t maxS16 = 0;
        const ma_int16 *s16Input = (const ma_int16 *)pInput;
        for (size_t i = 0; i < samplesCount; i++) {
          if (abs(s16Input[i]) > maxS16)
            maxS16 = abs(s16Input[i]);
        }
        extern volatile ma_uint64 g_ma_duplex_capture_overruns;
        extern volatile ma_uint64 g_ma_duplex_playback_underruns;
        __android_log_print(
            ANDROID_LOG_INFO, LOG_TAG,
            "[Capture Conv #%d] samples=%zu, maxS16=%d, xruns=o%llu/u%llu",
            captureConvDebugCount, samplesCount, maxS16,
            (unsigned long long)g_ma_duplex_capture_overruns,
            (unsigned long long)g_ma_duplex_playback_underruns);
      }
#endif
    } else if (pDevice->capture.format == ma_format_s24) {
      ma_pcm_s24_to_f32(userData->mConversionBuffer.data(), pInput,
                        samplesCount, ma_dither_mode_none);
    } else if (pDevice->capture.format == ma_format_s32) {
      ma_pcm_s32_to_f32(userData->mConversionBuffer.data(),
                        (const ma_int32 *)pInput, samplesCount,
                        ma_dither_mode_none);
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
    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG,
        "[Capture CB] channels: capture=%d, playback=%d, frames=%u",
        captureChannels, playbackChannels, frameCount);
#endif
  }
#endif

  // ==========================================================================
  // FILTERS FIRST (AEC included): cancel at the front of the chain so EVERY
  // downstream consumer — the ring buffer (recordings!), auto-record onset
  // detection, visualization, energy metering, streaming — sees the cleaned
  // signal. Historically the ring was written pre-AEC, so overdub recordings
  // carried the full speaker bleed while the on-screen waveform (post-AEC)
  // shrank: "the display converges but the recording still has the track."
  // The AEC reads its reference from PREVIOUS callbacks' writes, so running
  // before this block's slave mix only shifts the constant reference offset
  // by one buffer — absorbed by the loop-synchronous template as a fixed
  // rotation.
  // (SKIP during calibration to capture raw impulse response and save CPU)
  // Note: mFilters may be null if callback runs before init() completes
  // Use lock-free hasFilters() check to avoid mutex contention in hot path
  if (captured != nullptr && userData->mFilters != nullptr &&
      userData->mFilters->hasFilters() && !userData->mCalibrationActive) {
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
    // Set the capture frame count for AEC position-based sync BEFORE
    // processing. This is the frame count at the START of this block (before
    // we increment).
    size_t captureFrameCount =
        userData->mTotalFramesCaptured.load(std::memory_order_acquire);
    userData->mFilters->setAecCaptureFrameCount(captureFrameCount);

    // SHARED-duplex xrun watch: every duplex-ring overrun/underrun (counted
    // by the CLOUDLOOP PATCH in miniaudio.h) shifts the mic-to-reference
    // alignment by up to a burst, which the loop-synchronous template can't
    // see — it would keep subtracting a rotated echo. Re-arm the convergence
    // seed on each new event (internally gated; no-op while a seed is in
    // flight, no-op when LSAEC inactive). Realtime-safe: one atomic-ish read
    // and a flag set.
    {
      extern volatile ma_uint64 g_ma_duplex_capture_overruns;
      extern volatile ma_uint64 g_ma_duplex_playback_underruns;
      extern volatile ma_uint64 g_ma_duplex_recenter_sheds;
      uint64_t xruns = g_ma_duplex_capture_overruns +
                       g_ma_duplex_playback_underruns +
                       g_ma_duplex_recenter_sheds;
      if (xruns != userData->mLastSeenDuplexXruns) {
        // Only mark the xrun consumed when the notify actually landed —
        // on lock contention (filters being swapped) retry next callback.
        if (userData->mFilters->notifyAecReferenceChanged()) {
          userData->mLastSeenDuplexXruns = xruns;
        }
      }
    }

    // Thread-safe filter processing (protects against concurrent
    // addFilter/removeFilter)
    userData->mFilters->processAllFilters(
        captured, frameCount, captureChannels,
        userData->deviceConfig.capture.format);
  }
  cbTimer_.markAec = std::chrono::steady_clock::now();
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

  // =========================================================================
  // NATIVE RING BUFFER: Continuous capture for latency compensation (pre-roll)
  // Written AFTER the filter chain: recordings are cut from this buffer and
  // must contain the AEC-cleaned signal, not the raw speaker bleed.
  // IMPORTANT: Pass actual capture channels to auto-reconfigure if mismatched
  // =========================================================================
  if (g_nativeRingBuffer != nullptr && captured != nullptr) {
    g_nativeRingBuffer->write(captured, frameCount, captureChannels);
  }
  cbTimer_.markRing = std::chrono::steady_clock::now();

  // =========================================================================
  // BOOKKEEPING (must run BEFORE SoLoud mix so the audio engine schedules
  // any per-beat metronome clicks for THIS buffer; otherwise the click voice
  // is at least one buffer late). bufferStartFrame is the global frame of
  // the first sample in this buffer, computed before mTotalFramesCaptured
  // is incremented at the end of the callback.
  // =========================================================================
  int64_t bufferStartFrame = static_cast<int64_t>(
      userData->mTotalFramesCaptured.load(std::memory_order_acquire));

  // Hands-free first-loop capture: when armed, the first onset in this buffer
  // becomes the downbeat and starts recording (rewinding the ring buffer — which
  // was just written above — back to the onset so the lead-in silence is
  // trimmed). No-op unless armed/recording. Runs before processEvents so the
  // start's bookkeeping lines up with this buffer; any stop it schedules is in
  // the future and fires on a later buffer.
  AutoRecorder::instance().process(captured, frameCount, captureChannels,
                                   bufferStartFrame, userData);

  NativeScheduler::instance().processEvents(bufferStartFrame, frameCount,
                                            userData);
  flowstate::audio_engine::AudioEngine::instance().process(
      bufferStartFrame, frameCount, static_cast<uint32_t>(pDevice->sampleRate),
      static_cast<uint16_t>(captureChannels));
  cbTimer_.markSched = std::chrono::steady_clock::now();

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
    cbTimer_.markMix = std::chrono::steady_clock::now();

    // =======================================================================
    // METRONOME (Phase 3c v2): mix sample-accurate clicks ON TOP of SoLoud's
    // output, BEFORE the AEC reference buffer write below. Order matters:
    //   - Doing this before AEC capture means AEC "sees" the click in the
    //     reference signal and will cancel it from the microphone input —
    //     so the recorded clip never contains the metronome.
    //   - Doing this after SoLoud mix means the click sits in the same float
    //     buffer as voices, which then goes through the same monitoring +
    //     format-conversion path.
    flowstate::audio_engine::AudioEngine::instance().mixMetronomeIntoOutput(
        playbackFloat, bufferStartFrame, frameCount,
        static_cast<uint16_t>(playbackChannels),
        static_cast<uint32_t>(pDevice->sampleRate));

#if DEBUG_CALLBACK_SLAVE
    static int soloudMixDebugCount = 0;
    if (++soloudMixDebugCount <= 5) {
#ifdef _IS_ANDROID_
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[Capture Slave] frames=%u ch=%d", frameCount,
                          playbackChannels);
#endif
    }
#endif

    // Mark slave audio as ready after first successful callback
    // This signals that the audio pipeline is flowing and calibration can start
    soloud_setSlaveAudioReady();

    // Write to AEC reference buffer IN THE SAME CALLBACK - guarantees sync!
    // This is the whole point of slave mode: one callback, one clock.
    // Handle channel conversion: pOutput has actual device channels, but AEC
    // buffer may have different channel count (usually mono for AEC purposes).
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
        if (monoAec.size() < frameCount)
          monoAec.resize(frameCount);
        for (ma_uint32 i = 0; i < frameCount; ++i) {
          monoAec[i] = (outputFloat[i * 2] + outputFloat[i * 2 + 1]) * 0.5f;
        }
        g_aecReferenceBuffer->write(monoAec.data(), frameCount);
      } else if (playbackChannels == 1 && bufferCh == 2) {
        // Mono playback → Stereo AEC buffer: duplicate to both channels
        static thread_local std::vector<float> stereoAec;
        if (stereoAec.size() < frameCount * 2)
          stereoAec.resize(frameCount * 2);
        for (ma_uint32 i = 0; i < frameCount; ++i) {
          stereoAec[i * 2] = stereoAec[i * 2 + 1] = outputFloat[i];
        }
        g_aecReferenceBuffer->write(stereoAec.data(), frameCount);
      } else {
        // Unsupported - write directly and hope for the best
        g_aecReferenceBuffer->write(outputFloat, frameCount);
      }
    }

#ifdef AEC_DIAG_CAPTURE
    // Diagnostic: record raw mono reference (SoLoud out) + raw mono mic,
    // aligned in THIS callback. Arm once playback carries real signal;
    // capture 30 s.
    {
      // Arm on first real playback signal.
      if (g_diagState == 0) {
        float e = 0.0f;
        for (ma_uint32 i = 0; i < frameCount; ++i)
          e += playbackFloat[i * playbackChannels] *
               playbackFloat[i * playbackChannels];
        if (frameCount > 0 && (e / frameCount) > 1e-6f) {
          g_diagSR = static_cast<unsigned int>(pDevice->sampleRate);
          g_diagCap = static_cast<size_t>(g_diagSR) * 30; // 30 s
          g_diagRef.assign(g_diagCap, 0.0f);
          g_diagMic.assign(g_diagCap, 0.0f);
          g_diagPos = 0;
          g_diagStartFrame = bufferStartFrame;
          g_diagP = NativeScheduler::instance().getBaseLoopFrames();
          g_diagLoopStart = NativeScheduler::instance().getBaseLoopStartFrame();
          g_diagState = 1;
          aecLog("[AEC DIAG] armed: SR=%u cap=%zu P=%lld start=%lld\n", g_diagSR,
                 g_diagCap, (long long)g_diagP, (long long)g_diagStartFrame);
        }
      }
      // Capture (RT-safe: memcpy-style writes into preallocated buffers).
      if (g_diagState == 1) {
        for (ma_uint32 i = 0; i < frameCount && g_diagPos < g_diagCap; ++i) {
          float r = 0.0f;
          for (int ch = 0; ch < playbackChannels; ++ch)
            r += playbackFloat[i * playbackChannels + ch];
          r /= (float)playbackChannels;
          g_diagRef[g_diagPos] = r;
          g_diagMic[g_diagPos] =
              (captured != nullptr) ? captured[i * captureChannels] : 0.0f;
          ++g_diagPos;
        }
        // Refresh P/loopStart in case the loop committed after arming on noise.
        if (g_diagP <= 0) {
          g_diagP = NativeScheduler::instance().getBaseLoopFrames();
          g_diagLoopStart = NativeScheduler::instance().getBaseLoopStartFrame();
        }
        if (g_diagPos >= g_diagCap) {
          diagFlush(); // one-time write (accepts a single glitch)
          g_diagState = 2;
        }
      }
    }
#endif // AEC_DIAG_CAPTURE

    // If monitoring is enabled, ADD the captured input to the SoLoud output
    // (This allows hearing yourself while SoLoud plays)
    if (userData->monitoringEnabled && captured != nullptr &&
        pOutput != nullptr) {
      float *outputFloat = playbackFloat; // Use our f32 buffer for monitoring
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

    // Final step: Convert f32 playback buffer to device format and copy to
    // pOutput. This happens AFTER all processing (AEC, monitoring) is done
    // in f32.
#ifdef _IS_ANDROID_
    static int convDebugCount = 0;
    if (++convDebugCount <= 10) {
      float maxVal = 0.0f;
      for (size_t i = 0; i < playbackSamples; i++) {
        if (fabsf(playbackFloat[i]) > maxVal)
          maxVal = fabsf(playbackFloat[i]);
      }
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[Playback Conv #%d] samples=%zu, maxF32=%.6f, "
                          "format=%d",
                          convDebugCount, playbackSamples, maxVal,
                          pDevice->playback.format);
    }
#endif
    if (pDevice->playback.format == ma_format_s16) {
      // Convert f32 -> s16
      ma_pcm_f32_to_s16(pOutput, playbackFloat, playbackSamples,
                        ma_dither_mode_none);
    } else if (pDevice->playback.format == ma_format_s32) {
      // Convert f32 -> s32 (some HALs use 32-bit integer)
      ma_pcm_f32_to_s32(pOutput, playbackFloat, playbackSamples,
                        ma_dither_mode_none);
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
      // Mono input or channel count mismatch - just copy first matching
      // channels
      int channelsToCopy = std::min(captureChannels, playbackChannels);
      for (ma_uint32 i = 0; i < frameCount * channelsToCopy; i++) {
        outputFloat[i] = inputFloat[i];
      }
    }
  }

  // TRANSFORM CAPTURED BUFFER in-place to match monitoring mode
  // This ensures recordings, visualizations, and filters all use the same
  // transformed audio
  // Note: captureChannels is already defined at top of function using actual
  // device value
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

  // NOTE: the filter chain (AEC included) now runs at the TOP of the callback,
  // right after format conversion — see "FILTERS FIRST" above. Recordings cut
  // from the ring buffer, auto-record onset detection, visualization, energy
  // and streaming all see the same cleaned signal.

  // Do something with the captured audio data...
  // LOCK-FREE: Write to the current write buffer, then swap atomically
  {
    // Get the buffer we should write to
    float *writeBuffer =
        (capturedBufferWriteIndex.load(std::memory_order_relaxed) == 0)
            ? capturedBufferA
            : capturedBufferB;

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
      capturedBuffer = writeBuffer; // Update legacy pointer for readers
      capturedBufferWriteIndex.store(1 - oldIndex, std::memory_order_release);
    }
  }

  // Calibration capture: accumulate samples if calibration is active.
  // LOCK-FREE: taking mCalibrationMutex here blocked the RT thread whenever
  // the API thread held it (observed as 40ms+ callback overruns during
  // calibration). The acquire on mCalibrationActive pairs with the release in
  // startCalibrationCapture, so the buffer is fully sized before we can see
  // active==true; only this thread ever advances mCalibrationWritePos.
  if (userData->mCalibrationActive.load(std::memory_order_acquire)) {
    const size_t writePos =
        userData->mCalibrationWritePos.load(std::memory_order_relaxed);
    size_t samplesToCapture = frameCount;
    size_t spaceLeft = userData->mCalibrationBuffer.size() > writePos
                           ? userData->mCalibrationBuffer.size() - writePos
                           : 0;
    if (samplesToCapture > spaceLeft) {
      samplesToCapture = spaceLeft;
    }
    if (samplesToCapture > 0) {
      // Copy to calibration buffer (Mono Downmix)
      // Use captureChannels (actual device value) for correct buffer
      // interpretation
#if DEBUG_CALLBACK_CALIBRATION
      float debugBatchSum = 0.0f;
#endif
      for (size_t i = 0; i < samplesToCapture; ++i) {
        float sample;
        if (captureChannels >= 2) {
          // Downmix stereo to mono: (L + R) * 0.5
          sample = (captured[i * captureChannels] +
                    captured[i * captureChannels + 1]) *
                   0.5f;
        } else {
          sample = captured[i * captureChannels]; // Already mono
        }
        userData->mCalibrationBuffer[writePos + i] = sample;
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

      userData->mCalibrationWritePos.store(writePos + samplesToCapture,
                                           std::memory_order_release);
    }
  }

  // Calculate energy for FFT visualization
  // NOTE: captured is ALWAYS f32 after the conversion above, regardless of
  // device format
  if (captured != nullptr)
    calculateEnergy(captured, frameCount, captureChannels);

  // Stream the audio data? Wait-free push into the ring; the dedicated
  // StreamDispatchWorker thread owns the heap allocation + Dart dispatch
  // (see class comment) — the render thread never allocates here.
  if (userData->isStreamingData && nativeStreamDataCallback != nullptr) {
    const unsigned char *data = (const unsigned char *)captured;
    // IMPORTANT: captured is ALWAYS f32 after format conversion above,
    // so we must use sizeof(float), NOT bytesPerSample (which is the native
    // format)
    const int frameSize = sizeof(float) * captureChannels;
    const int dataSize = frameCount * frameSize;
    StreamDispatchWorker::instance().push(data, static_cast<size_t>(dataSize));
  }

  // Detect silence - captured is always f32 after conversion
  if (userData->isDetectingSilence && captured != nullptr) {
    detectSilence(userData);

    // Copy current buffer to circularBuffer
    if (delayed_silence_started && userData->isRecording &&
        userData->secondsOfAudioToWriteBefore > 0) {
      std::vector<float> values(captured,
                                captured + frameCount * captureChannels);
      circularBuffer.get()->push(values);
    }

    if (!delayed_silence_started && userData->isRecording &&
        !userData->isRecordingPaused) {
#ifdef _IS_ANDROID_
      static int wavWriteDebugCount = 0;
      if (++wavWriteDebugCount <= 5) {
        float maxVal = 0.0f;
        for (ma_uint32 i = 0; i < frameCount * captureChannels && i < 100;
             i++) {
          if (fabsf(captured[i]) > maxVal)
            maxVal = fabsf(captured[i]);
        }
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                            "[WAV WRITE #%d] frames=%u, channels=%d, "
                            "samples=%u, maxF32=%.6f",
                            wavWriteDebugCount, frameCount, captureChannels,
                            frameCount * captureChannels, maxVal);
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
        for (ma_uint32 i = 0; i < frameCount * captureChannels && i < 100;
             i++) {
          if (fabsf(captured[i]) > maxVal)
            maxVal = fabsf(captured[i]);
        }
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                            "[WAV WRITE2 #%d] frames=%u, channels=%d, "
                            "samples=%u, maxF32=%.6f",
                            wavWriteDebugCount2, frameCount, captureChannels,
                            frameCount * captureChannels, maxVal);
      }
#endif
      userData->wav.write(captured, frameCount);
    }
  }

  // Increment total frame counter for AEC synchronization. The schedulers
  // and audio engine already ran at the top of this callback against the
  // pre-increment value; we update here so the NEXT callback sees this
  // buffer's contribution to the global frame count.
  userData->mTotalFramesCaptured.fetch_add(frameCount,
                                           std::memory_order_release);
}

#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_OSX)
// Single-unit RemoteIO path (one clock). The render function drives the SAME
// data_callback via a minimal ma_device shim filled with only the fields the
// callback reads (it never calls ma_* on it), so the whole pipeline is
// unchanged. g_caActive routes start/stop/dispose to the custom device.
static ma_device g_caDevice; // shim for the data_callback contract
static bool g_caActive = false;
static void caRenderAdapter(void *userData, const float *mic, float *speaker,
                            unsigned int frames) {
  g_caDevice.pUserData = userData; // Capture*
  data_callback(&g_caDevice, speaker, static_cast<const void *>(mic), frames);
}
#endif

// /////////////////////////////
// Capture class Implementation
// /////////////////////////////
float waveData[256];
Capture::Capture()
    : isDetectingSilence(false), silenceThresholdDb(-40.0f),
      silenceDuration(2.0f), secondsOfAudioToWriteBefore(0.0f),
      isRecording(false), isRecordingPaused(false), isStreamingData(false),
      monitoringEnabled(false), monitoringMode(0), mCalibrationWritePos(0),
      mCalibrationActive(false), mInited(false), mContextInited(false) {
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

CaptureErrors Capture::ensureContext() {
  if (mContextInited)
    return captureNoError;

#ifdef _IS_WIN_
  printf("[Capture::ensureContext] Windows: Initializing WASAPI context...\n");
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
  } else {
    printf("Initialized with WASAPI backend for low-latency audio\n");
    fflush(stdout);
  }
#elif defined(MA_HAS_COREAUDIO)
  // iOS/macOS: do NOT let miniaudio's CoreAudio context manage, activate OR
  // deactivate the AVAudioSession. Our single-unit duplex device
  // (coreaudio_duplex.mm configureSession) is the ONLY session owner; two
  // owners fighting over it breaks playback (capture survives, the speaker
  // goes silent). All THREE flags are required — a missing
  // noAudioSessionDeactivate means ma_context_uninit calls setActive:false
  // app-wide. Upstream flutter_recorder fix 4a0b378.
  ma_context_config contextConfig = ma_context_config_init();
  contextConfig.coreaudio.sessionCategory = ma_ios_session_category_none;
  contextConfig.coreaudio.noAudioSessionActivate = true;
  contextConfig.coreaudio.noAudioSessionDeactivate = true;
  result = ma_context_init(NULL, 0, &contextConfig, &context);
#else
  // Other platforms: Use auto backend selection
  result = ma_context_init(NULL, 0, NULL, &context);
#endif

  if (result != MA_SUCCESS) {
    printf("Failed to initialize audio context %d\n", result);
    fflush(stdout);
    return captureInitFailed;
  }
  mContextInited = true;
  return captureNoError;
}

std::vector<CaptureDevice> Capture::listCaptureDevices() {
#ifdef _IS_ANDROID_
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "***************** LIST DEVICES START\n");
#else
  printf("***************** LIST DEVICES START\n");
#endif
  std::vector<CaptureDevice> ret;
  // Reuse the live explicit context when it exists (upstream v1.1.5 guard,
  // adapted to our up-front context lifecycle) and otherwise create it with
  // the SAME platform config used by init(). A NULL-config enumeration
  // context on Apple would set the AVAudioSession category AND activate it
  // out from under the live duplex unit.
  if (ensureContext() != captureNoError) {
    printf("Failed to initialize context\n");
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
                            int androidInputPreset, bool captureOnly) {
  printf("[Capture::init] Starting init: deviceID=%d, sampleRate=%u, "
         "channels=%u, androidInputPreset=%d, captureOnly=%d, mInited=%d\n",
         deviceID, sampleRate, channels, androidInputPreset, captureOnly,
         mInited);
  fflush(stdout);

  // Guard against double initialization
  if (mInited) {
    printf("[Capture::init] Already initialized, calling dispose first\n");
    fflush(stdout);
    dispose();
  }

  // Create the explicit context up-front if it doesn't exist yet (Windows
  // keeps it alive across init/dispose cycles to avoid the WASAPI re-init
  // deadlock; listCaptureDevices may also have created it already).
  if (ensureContext() != captureNoError)
    return captureInitFailed;

  // Choose device mode based on captureOnly parameter:
  // - captureOnly=true: Use capture-only mode when SoLoud has its own playback
  //   device. This prevents two playback devices from competing (which causes
  //   grainy audio)
  // - captureOnly=false: Use duplex mode for slave mode where recorder drives
  //   SoLoud output
  if (captureOnly) {
#ifdef _IS_ANDROID_
    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] Using CAPTURE-ONLY mode (SoLoud has own playback)");
#else
    printf(
        "[Capture::init] Using CAPTURE-ONLY mode (SoLoud has own playback)\n");
    fflush(stdout);
#endif
    deviceConfig = ma_device_config_init(ma_device_type_capture);
  } else {
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                        "[Capture::init] Using DUPLEX mode (slave mode for "
                        "AEC)");
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
  // HAL fast capture uses 240-frame bursts (5ms @ 48kHz) - request 2 bursts
  // (480 frames = 10ms). This matches the hardware burst size while keeping
  // latency minimal
  deviceConfig.periodSizeInFrames = 480; // 2 x 240-frame burst = 10ms @ 48kHz

  // CRITICAL for the duplex mic path: without this flag miniaudio never
  // forwards periodSizeInFrames to AAudio (setFramesPerDataCallback is
  // gated on aaudio.allowSetBufferCapacity, default FALSE), so its internal
  // period falls back to the 4096-frame buffer capacity. That sized the
  // duplex ring at 4096*5 (~427ms) with a 171ms silence pre-seek — measured
  // as a 372ms mic->AEC-reference echo path on the TB330FU, which blew the
  // click-calibration's match window. With the flag, the internal period is
  // the real 480 frames: ring = the 100ms floor (see the CLOUDLOOP PATCH in
  // ma_duplex_rb_init, which also removes the ring-smaller-than-one-burst
  // hazard this flag used to carry), pre-seek 20ms, echo path ~40-70ms.
  deviceConfig.aaudio.allowSetBufferCapacity = MA_TRUE;

  // AAudio-specific settings for FAST PATH (Mode 12):
  // VOICE_RECOGNITION preset is designed for low-latency speech recognition
  // - No AEC/NS processing (unlike VOICE_COMMUNICATION)
  // - No effects (unlike DEFAULT which may apply system effects)
  // - Should enable MMAP fast path
  deviceConfig.aaudio.inputPreset =
      ma_aaudio_input_preset_voice_recognition;         // Low-latency speech
  deviceConfig.aaudio.usage = ma_aaudio_usage_game;     // GAME bypasses
                                                        // Dolby/effects!
  deviceConfig.aaudio.contentType = ma_aaudio_content_type_music;

  // Request EXCLUSIVE mode - required for true MMAP fast path
  deviceConfig.capture.shareMode = ma_share_mode_exclusive;
  deviceConfig.playback.shareMode = ma_share_mode_exclusive;

  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "[Capture::init] AAudio defaults: inputPreset=VOICE_RECOGNITION "
                      "(Dart preset override applies later), usage=GAME, "
                      "sharingMode=EXCLUSIVE requested");
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "[Capture::init] AAudio: periodSize=480 (2 bursts, 10ms "
                      "@ 48kHz)");
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

#if defined(__APPLE__) && TARGET_OS_IPHONE
  // iOS: always use the default (session-routed) input. Explicit miniaudio
  // device IDs are meaningless here — input selection goes through
  // AVAudioSession routing, and enumeration returns nothing until a
  // record-capable category is active, which only the CADuplex session
  // owner sets (at start, after this init).
  (void)deviceID;
#else
  if (deviceID != -1) {
    auto devices = listCaptureDevices();
    if (devices.size() == 0 || deviceID >= devices.size())
      return captureInitFailed;
    deviceConfig.capture.pDeviceID = &pCaptureInfos[deviceID].id;
  }
#endif

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
    bytesPerSample = 0; // Will be set after device init based on actual format
    break;
  default:
    return captureInitFailed;
  }

#ifdef _IS_ANDROID_
  // Android with format=unknown: Configure for AAudio Fast Capture
  // (low-latency). Key requirements for MediaTek Helio G88 Fast Capture path:
  // - PCM_16_BIT format (NOT float32 - HAL only supports s16 on fast path)
  // - 48kHz sample rate (exact match required)
  // - VOICE_RECOGNITION input preset (enables AUDIO_INPUT_FLAG_FAST)
  // - Stereo capture (works per dumpsys evidence)
  if (pcmFormat == PCMFormat::pcm_unknown) {
    // Force PCM_16_BIT (s16) for CAPTURE only - required for Fast Capture path
    // on MediaTek. Evidence from dumpsys of working app
    // (com.zuidsoft.looper):
    // - Capture: HAL format 0x1 (16-bit), Processing format 0x1 (16-bit), HAL
    //   frame count 240
    // - Playback: HAL format 0x3 (32-bit), Processing format 0x5 (float) -
    //   DIFFERENT from capture!
    // The capture and playback formats do NOT need to match.
    // DUPLEX MODE: Both capture and playback must use same format for
    // miniaudio. The Oboe example uses separate streams (different formats),
    // but we use duplex so we force s16 for both - the callback converts
    // s16<->f32 for internal processing
    deviceConfig.capture.format =
        ma_format_s16;                 // PCM_16_BIT - required for Fast Capture
    deviceConfig.capture.channels = 2; // STEREO capture
    deviceConfig.sampleRate = 48000;   // EXPLICIT 48kHz
    deviceConfig.playback.format =
        ma_format_s16; // PCM_16_BIT - must match capture for duplex!
    deviceConfig.playback.channels = 2; // Stereo playback
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                        "[Capture::init] AAudio FAST MODE (DUPLEX): "
                        "capture=S16, playback=S16, channels=STEREO, "
                        "rate=48000, captureOnly=%d",
                        captureOnly);
  } else {
    // Specific format requested - use provided values
    // Note: This may not achieve low-latency mode if values don't match
    // device native config
    deviceConfig.capture.format = format;
    deviceConfig.capture.channels = channels;
    deviceConfig.sampleRate = sampleRate;
    deviceConfig.playback.format = format;
    deviceConfig.playback.channels = channels; // Match capture for consistency
    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG,
        "[Capture::init] AAudio explicit mode: format=%d, channels=%d, "
        "rate=%d",
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

  // Apply the Dart-requested Android input preset (upstream v1.1.5). Runs
  // AFTER our Android fast-path defaults above so a nonzero preset from Dart
  // overrides them, while 0 ("unset") leaves our voice_recognition default
  // intact.
  CaptureErrors presetResult =
      setAndroidInputPreset(&deviceConfig, androidInputPreset);
  if (presetResult != captureNoError) {
    return presetResult;
  }

#if defined(_IS_ANDROID_) && defined(MA_HAS_OPENSL)
  // Upstream v1.1.5 workaround: on some Android/OEM AAudio voice-communication
  // paths, requestStart() succeeds but the STARTING -> STARTED transition is
  // reported late. OpenSL supports the same recording preset without that
  // AAudio state race. Only taken in capture-only mode: our duplex/slave path
  // is AAudio-only by design. Swaps the up-front context for an OpenSL-backend
  // one (mContextInited bookkeeping kept coherent).
  if (captureOnly && deviceID == -1 &&
      deviceConfig.opensl.recordingPreset ==
          ma_opensl_recording_preset_voice_communication) {
    __android_log_print(
        ANDROID_LOG_WARN, LOG_TAG,
        "Using OpenSL backend for voiceCommunication input preset.");
    if (mContextInited) {
      ma_context_uninit(&context);
      mContextInited = false;
    }
    ma_backend backend = ma_backend_opensl;
    ma_context_config contextConfig = ma_context_config_init();
    if (ma_context_init(&backend, 1, &contextConfig, &context) != MA_SUCCESS) {
      printf("Failed to initialize OpenSL capture context.\n");
      return captureInitFailed;
    }
    mContextInited = true;
  }
#endif

  printf("[Capture::init] Calling ma_device_init...\n");
  fflush(stdout);

  mDuplexDenied = false; // Reset on each init
  mSharedDuplex = false;

#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_OSX)
  // SINGLE-UNIT CoreAudio duplex (one clock → eliminates the AEC reference
  // drift). Fill the device shim with the negotiated format; the actual audio
  // unit is created + started in Capture::start(), with a miniaudio fallback
  // if it fails. captureOnly always uses miniaudio (no playback side to share
  // a clock with).
  g_caActive = false;
  if (!captureOnly) {
    memset(&device, 0, sizeof(device));
    device.pUserData = this;
    device.sampleRate = sampleRate;      // 48000
    device.capture.channels = channels;  // stereo
    device.capture.format = ma_format_f32;
    device.playback.channels = channels;
    device.playback.format = ma_format_f32;
    // CRITICAL: caRenderAdapter hands data_callback &g_caDevice, NOT &device.
    // The shim must carry the same negotiated fields — with g_caDevice zeroed,
    // data_callback sees 0 channels / unknown formats: the slave mix memsets
    // to silence, the output-format switch never writes pOutput, and the
    // visualization loop iterates zero samples (the original "file records
    // but playback/FFT dead" bug).
    memcpy(&g_caDevice, &device, sizeof(device));
    g_caActive = true;
    result = MA_SUCCESS;
    printf("[Capture::init] CADuplex: single-unit RemoteIO duplex selected "
           "(%u Hz, %u ch, f32)\n",
           sampleRate, channels);
    fflush(stdout);
  }
  if (!g_caActive)
#endif
  {
#ifdef _IS_WIN_
    // WASAPI: try exclusive mode first for the duplex (slave/AEC) device — it
    // bypasses the Windows audio engine mixer for the lowest attainable
    // latency. Exclusive access can be denied (endpoint busy, driver refuses
    // the format), so fall back to low-latency shared mode (the previous
    // behavior). While exclusive playback is held, other apps' audio is muted
    // — same trade-off DAWs make in WASAPI-exclusive/ASIO mode.
    if (!captureOnly) {
      deviceConfig.capture.shareMode = ma_share_mode_exclusive;
      deviceConfig.playback.shareMode = ma_share_mode_exclusive;
      result = ma_device_init(&context, &deviceConfig, &device);
      if (result == MA_SUCCESS) {
        printf("[Capture::init] WASAPI EXCLUSIVE mode acquired "
               "(capture period=%u frames, playback period=%u frames)\n",
               device.capture.internalPeriodSizeInFrames,
               device.playback.internalPeriodSizeInFrames);
        fflush(stdout);
      } else {
        printf("[Capture::init] WASAPI exclusive mode unavailable (error %d), "
               "falling back to low-latency shared mode\n", result);
        fflush(stdout);
        deviceConfig.capture.shareMode = ma_share_mode_shared;
        deviceConfig.playback.shareMode = ma_share_mode_shared;
        result = ma_device_init(&context, &deviceConfig, &device);
      }
    } else {
      result = ma_device_init(&context, &deviceConfig, &device);
    }
#else
    result = ma_device_init(&context, &deviceConfig, &device);
#endif
    if (result != MA_SUCCESS) {
      printf("Failed to initialize capture device. Error: %d\n", result);
      fflush(stdout);
#ifdef _IS_WIN_
      // Keep the context alive: Windows reuses it across init cycles (see
      // ensureContext/dispose); uniniting here would re-enter the WASAPI
      // init/uninit deadlock on retry.
#else
      ma_context_uninit(&context);
      mContextInited = false;
#endif
      return captureInitFailed;
    }
  }

  printf("[Capture::init] Device initialized successfully\n");

#ifdef _IS_ANDROID_
  // Duplex health check. EXCLUSIVE only exists on the MMAP path, and many
  // devices (e.g. MediaTek tablets shipping global Dolby DAX) never enable
  // MMAP at all — every app gets SHARED, forever. SHARED duplex is
  // topologically identical to exclusive (two AAudio streams glued by
  // miniaudio's duplex ring), so sharing mode itself is NOT the problem.
  // The real pathology is BURST SIZE: a fat legacy capture burst (e.g.
  // 4096 frames/85ms against a 10ms playback drain) oscillates the duplex
  // ring between starved and full — half-speed audio and static.
  //
  // So: keep SHARED duplex when the HAL grants fast-path-sized capture
  // bursts (the TB330FU grants 240-frame/5ms FAST bursts in shared mode);
  // fall back to capture-only (Dart drops to standard SoLoud) only when
  // bursts are genuinely fat.
  if (!captureOnly && device.aaudio.pStreamCapture != nullptr) {
    void *aaudioLib = context.aaudio.hAAudio;
    auto getSharingMode =
        aaudioLib
            ? (int32_t(*)(void *))dlsym(aaudioLib, "AAudioStream_getSharingMode")
            : nullptr;
    auto getFramesPerBurst =
        aaudioLib
            ? (int32_t(*)(void *))dlsym(aaudioLib,
                                        "AAudioStream_getFramesPerBurst")
            : nullptr;
    if (getSharingMode) {
      int32_t captureShareMode = getSharingMode(device.aaudio.pStreamCapture);
      int32_t captureBurst =
          getFramesPerBurst ? getFramesPerBurst(device.aaudio.pStreamCapture)
                            : -1;
      int32_t playbackBurst =
          (getFramesPerBurst && device.aaudio.pStreamPlayback != nullptr)
              ? getFramesPerBurst(device.aaudio.pStreamPlayback)
              : -1;
      // 960 frames = 20ms @ 48kHz: generous fast-path ceiling. The broken
      // case this guards against is ~4096 frames; unknown (-1) is treated
      // as fat, matching the old conservative behavior.
      const int32_t kMaxAcceptableCaptureBurst = 960;
      bool burstsOk = captureBurst > 0 &&
                      captureBurst <= kMaxAcceptableCaptureBurst;

      if (captureShareMode != 0 && burstsOk) { // 0 = EXCLUSIVE, 1 = SHARED
        mSharedDuplex = true;
        __android_log_print(
            ANDROID_LOG_INFO, LOG_TAG,
            "[Capture::init] SHARED DUPLEX accepted: capture burst=%d frames "
            "(%.1fms), playback burst=%d frames — fast path granted without "
            "exclusive. Slave mode + AEC stay on; duplex-ring xruns are "
            "counted for AEC re-seed.",
            captureBurst, captureBurst * 1000.0f / 48000.0f, playbackBurst);
      } else if (captureShareMode != 0) {
        __android_log_print(
            ANDROID_LOG_WARN, LOG_TAG,
            "[Capture::init] DUPLEX DENIED: capture sharingMode=%d with fat "
            "burst=%d frames (limit %d). Re-initializing as CAPTURE-ONLY.",
            captureShareMode, captureBurst, kMaxAcceptableCaptureBurst);

        // Deinit the broken duplex device
        ma_device_uninit(&device);

        // Re-init as capture-only with same fast-path settings
        deviceConfig = ma_device_config_init(ma_device_type_capture);
        deviceConfig.performanceProfile = ma_performance_profile_low_latency;
        deviceConfig.periodSizeInFrames = 480;
        deviceConfig.aaudio.inputPreset =
            ma_aaudio_input_preset_voice_recognition;
        deviceConfig.aaudio.usage = ma_aaudio_usage_game;
        deviceConfig.aaudio.contentType = ma_aaudio_content_type_music;
        deviceConfig.capture.shareMode = ma_share_mode_exclusive;
        deviceConfig.capture.format = ma_format_s16;
        deviceConfig.capture.channels = 2;
        deviceConfig.sampleRate = 48000;
        deviceConfig.dataCallback = data_callback;
        deviceConfig.pUserData = this;

        // Re-apply any Dart-requested preset on top of the defaults.
        setAndroidInputPreset(&deviceConfig, androidInputPreset);

        result = ma_device_init(&context, &deviceConfig, &device);
        if (result != MA_SUCCESS) {
          __android_log_print(
              ANDROID_LOG_ERROR, LOG_TAG,
              "[Capture::init] Capture-only re-init also failed: %d", result);
          ma_context_uninit(&context);
          mContextInited = false;
          return captureInitFailed;
        }

        mDuplexDenied = true;
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                            "[Capture::init] Capture-only re-init SUCCESS. "
                            "Dart should use standard SoLoud.");
      }
    }
  }
#endif

  // If format was unknown, now set bytesPerSample based on actual device
  // format
  if (bytesPerSample == 0) {
    switch (device.capture.format) {
    case ma_format_u8:
      bytesPerSample = 1;
      break;
    case ma_format_s16:
      bytesPerSample = 2;
      break;
    case ma_format_s24:
      bytesPerSample = 3;
      break;
    case ma_format_s32:
      bytesPerSample = 4;
      break;
    case ma_format_f32:
      bytesPerSample = 4;
      break;
    default:
      bytesPerSample = 2;
      break; // Safe fallback
    }
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                        "[Capture::init] System chose format=%d, "
                        "bytesPerSample=%d",
                        device.capture.format, bytesPerSample);
#else
    printf("[Capture::init] System chose format=%d, bytesPerSample=%d\n",
           device.capture.format, bytesPerSample);
#endif
  }

  // Pre-allocate conversion buffers if needed (use ACTUAL device period, not
  // config)
  ma_uint32 actualPeriod = device.capture.internalPeriodSizeInFrames;
  if (actualPeriod == 0)
    actualPeriod = 512; // Fallback if not reported

  // Capture conversion buffer: s16/s24/s32 -> f32
  if (device.capture.format != ma_format_f32) {
    mConversionBuffer.resize(actualPeriod * device.capture.channels);
  }

  // Playback conversion buffer: f32 -> s16 (for low-latency fast path)
  // Always allocate since slave mode needs f32 buffer for SoLoud output
  // before conversion
  mPlaybackBuffer.resize(actualPeriod * device.playback.channels);

  printf("[Capture::init] ACTUAL device params: sampleRate=%u, "
         "capture.channels=%u, playback.channels=%u, capture.format=%d, "
         "playback.format=%d\n",
         device.sampleRate, device.capture.channels, device.playback.channels,
         device.capture.format, device.playback.format);
  printf("[Capture::init] REQUESTED: sampleRate=%u, capture.channels=%u, "
         "playback.channels=%u\n",
         sampleRate, channels, channels);
  fflush(stdout);

#ifdef _IS_ANDROID_
  // CRITICAL: Set buffer size to 2x burst for low-latency (per Oboe/AAudio
  // best practices). Default AAudio buffer is much higher than optimal
  // https://developer.android.com/games/sdk/oboe/low-latency-audio#double-buffering
  // Note: We use miniaudio's dynamically loaded function pointers since these
  // APIs require API 26+
  typedef int32_t (*PFN_AAudioStream_getFramesPerBurst)(void *stream);
  typedef int32_t (*PFN_AAudioStream_setBufferSizeInFrames)(void *stream,
                                                            int32_t numFrames);
  typedef int32_t (*PFN_AAudioStream_getBufferSizeInFrames)(void *stream);

  // Use miniaudio's AAudio library handle for dlsym (not RTLD_DEFAULT which
  // won't find dynamically loaded libs)
  void *aaudioLib = context.aaudio.hAAudio;
  auto getFramesPerBurst = (PFN_AAudioStream_getFramesPerBurst)
                               context.aaudio.AAudioStream_getFramesPerBurst;
  auto setBufferSizeInFrames =
      aaudioLib ? (PFN_AAudioStream_setBufferSizeInFrames)dlsym(
                      aaudioLib, "AAudioStream_setBufferSizeInFrames")
                : nullptr;
  auto getBufferSizeInFrames =
      aaudioLib ? (PFN_AAudioStream_getBufferSizeInFrames)dlsym(
                      aaudioLib, "AAudioStream_getBufferSizeInFrames")
                : nullptr;

  if (getFramesPerBurst && setBufferSizeInFrames && getBufferSizeInFrames) {
    if (device.aaudio.pStreamCapture != nullptr) {
      int32_t burstSize = getFramesPerBurst(device.aaudio.pStreamCapture);
      int32_t optimalBufferSize = burstSize * 2; // Double buffering
      int32_t setResult = setBufferSizeInFrames(device.aaudio.pStreamCapture,
                                                optimalBufferSize);
      int32_t actualBufferSize =
          getBufferSizeInFrames(device.aaudio.pStreamCapture);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[LOW-LATENCY] Capture: burstSize=%d, requested=%d, "
                          "actual=%d, result=%d",
                          burstSize, optimalBufferSize, actualBufferSize,
                          setResult);
    }
    if (device.aaudio.pStreamPlayback != nullptr) {
      int32_t burstSize = getFramesPerBurst(device.aaudio.pStreamPlayback);
      int32_t optimalBufferSize = burstSize * 2; // Double buffering
      int32_t setResult = setBufferSizeInFrames(device.aaudio.pStreamPlayback,
                                                optimalBufferSize);
      int32_t actualBufferSize =
          getBufferSizeInFrames(device.aaudio.pStreamPlayback);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[LOW-LATENCY] Playback: burstSize=%d, requested=%d, "
                          "actual=%d, result=%d",
                          burstSize, optimalBufferSize, actualBufferSize,
                          setResult);
    }
  } else {
    __android_log_print(
        ANDROID_LOG_WARN, LOG_TAG,
        "[LOW-LATENCY] AAudio buffer APIs not available (API < 26)");
  }

  // CRITICAL: Verify AAudio performance mode and sharing mode
  // perfMode: 12 = LOW_LATENCY, 10 = NONE
  // sharingMode: 0 = EXCLUSIVE, 1 = SHARED
  typedef int32_t (*PFN_AAudioStream_getPerformanceMode)(void *stream);
  typedef int32_t (*PFN_AAudioStream_getSharingMode)(void *stream);
  typedef int32_t (*PFN_AAudioStream_getInputPreset)(void *stream);

  auto getPerformanceMode =
      aaudioLib ? (PFN_AAudioStream_getPerformanceMode)dlsym(
                      aaudioLib, "AAudioStream_getPerformanceMode")
                : nullptr;
  auto getSharingMode = aaudioLib
                            ? (PFN_AAudioStream_getSharingMode)dlsym(
                                  aaudioLib, "AAudioStream_getSharingMode")
                            : nullptr;
  auto getInputPreset = aaudioLib
                            ? (PFN_AAudioStream_getInputPreset)dlsym(
                                  aaudioLib, "AAudioStream_getInputPreset")
                            : nullptr;

  if (getPerformanceMode) {
    if (device.aaudio.pStreamCapture != nullptr) {
      int32_t perfMode = getPerformanceMode(device.aaudio.pStreamCapture);
      int32_t shareMode =
          getSharingMode ? getSharingMode(device.aaudio.pStreamCapture) : -1;
      int32_t burstSize = getFramesPerBurst
                              ? getFramesPerBurst(device.aaudio.pStreamCapture)
                              : -1;
      int32_t inputPreset =
          getInputPreset ? getInputPreset(device.aaudio.pStreamCapture) : -1;
      int32_t bufSize =
          getBufferSizeInFrames
              ? getBufferSizeInFrames(device.aaudio.pStreamCapture)
              : -1;

      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[LOW-LATENCY] Capture: perfMode=%d "
                          "(12=LOW_LATENCY), sharingMode=%d (0=EXCL, "
                          "1=SHARED)",
                          perfMode, shareMode);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[LOW-LATENCY] Capture: framesPerBurst=%d, "
                          "bufferSize=%d, inputPreset=%d (5=VOICE_RECOG, "
                          "6=UNPROCESSED)",
                          burstSize, bufSize, inputPreset);

      if (perfMode != 12) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                            "[LOW-LATENCY] WARNING: Capture NOT in "
                            "low-latency mode! perfMode=%d (expected 12)",
                            perfMode);
      }
      if (shareMode != 0) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                            "[LOW-LATENCY] WARNING: Capture NOT in exclusive "
                            "mode! sharingMode=%d (requested 0)",
                            shareMode);
      }
    }
    if (device.aaudio.pStreamPlayback != nullptr) {
      int32_t perfMode = getPerformanceMode(device.aaudio.pStreamPlayback);
      int32_t shareMode =
          getSharingMode ? getSharingMode(device.aaudio.pStreamPlayback) : -1;
      int32_t burstSize =
          getFramesPerBurst ? getFramesPerBurst(device.aaudio.pStreamPlayback)
                            : -1;
      int32_t bufSize =
          getBufferSizeInFrames
              ? getBufferSizeInFrames(device.aaudio.pStreamPlayback)
              : -1;

      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[LOW-LATENCY] Playback: perfMode=%d "
                          "(12=LOW_LATENCY), sharingMode=%d (0=EXCL, "
                          "1=SHARED)",
                          perfMode, shareMode);
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "[LOW-LATENCY] Playback: framesPerBurst=%d, "
                          "bufferSize=%d",
                          burstSize, bufSize);

      if (perfMode != 12) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                            "[LOW-LATENCY] WARNING: Playback NOT in "
                            "low-latency mode! perfMode=%d (expected 12)",
                            perfMode);
      }
    }
  } else {
    __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                        "[LOW-LATENCY] AAudioStream_getPerformanceMode not "
                        "available (API < 28)");
  }

  // Log ACTUAL vs REQUESTED configuration for debugging
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "[LOW-LATENCY] ACTUAL config: capture.ch=%u "
                      "playback.ch=%u rate=%u capture.fmt=%d playback.fmt=%d",
                      device.capture.channels, device.playback.channels,
                      device.sampleRate, device.capture.format,
                      device.playback.format);
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "[LOW-LATENCY] REQUESTED config: capture.ch=%u "
                      "playback.ch=%u rate=%u (0=native)",
                      deviceConfig.capture.channels,
                      deviceConfig.playback.channels, deviceConfig.sampleRate);

  // Log actual AAudio latency for debugging
  ma_uint32 capturePeriod = device.capture.internalPeriodSizeInFrames;
  ma_uint32 playbackPeriod = device.playback.internalPeriodSizeInFrames;
  float captureLatencyMs = (float)capturePeriod * 1000.0f / device.sampleRate;
  float playbackLatencyMs =
      (float)playbackPeriod * 1000.0f / device.sampleRate;
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "[LATENCY DIAG] capture: period=%u frames (%.2fms), "
                      "playback: period=%u frames (%.2fms) @ %uHz",
                      capturePeriod, captureLatencyMs, playbackPeriod,
                      playbackLatencyMs, device.sampleRate);
#endif

  // Warn if actual sample rate differs from requested
  if (device.sampleRate != sampleRate) {
    printf("[Capture::init] WARNING: Actual sample rate (%u) differs from "
           "requested (%u)!\n",
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
  if (circularBuffer)
    circularBuffer.reset();
  if (streamBuffer)
    streamBuffer.reset();
  StreamDispatchWorker::instance().stop(); // no-op if streaming wasn't active
  isRecording = false;

  printf("[Capture::dispose] Calling ma_device_uninit...\n");
  fflush(stdout);

#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_OSX)
  if (g_caActive) {
    caDuplexStop();
    g_caActive = false;
  } else
#endif
  {
    ma_device_uninit(&device);
  }

#ifdef _IS_WIN_
  // On Windows/WASAPI, do NOT call ma_context_uninit() here!
  // Repeatedly calling ma_context_init/uninit causes deadlocks.
  // The context is kept alive and reused across init/dispose cycles.
  // It will be cleaned up in the destructor.
  printf("[Capture::dispose] Windows: context kept alive for reuse\n");
  fflush(stdout);
#else
  // On other platforms, uninit context normally (guarded, upstream v1.1.5
  // semantics adapted to mContextInited). The three-flag CoreAudio config in
  // ensureContext() makes this a session-neutral teardown on Apple.
  if (mContextInited) {
    printf("[Capture::dispose] Calling ma_context_uninit...\n");
    fflush(stdout);
    ma_context_uninit(&context);
    mContextInited = false;
  }
#endif

  printf("[Capture::dispose] Dispose complete\n");
  fflush(stdout);
}

bool Capture::isInited() { return mInited; }

bool Capture::isDeviceStarted() {
#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_OSX)
  if (g_caActive)
    return caDuplexIsRunning();
#endif
  ma_device_state result = ma_device_get_state(&device);
  return result == ma_device_state_started;
}

CaptureErrors Capture::start() {
  if (!mInited)
    return captureNotInited;

#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_OSX)
  if (g_caActive) {
    // Create + start the single-clock duplex unit; it drives data_callback via
    // the adapter (mic+speaker sample-locked off one clock).
    if (caDuplexStart(device.sampleRate, device.capture.channels, this,
                      caRenderAdapter)) {
      return captureNoError;
    }
    // Fallback: custom device failed (e.g. macOS aggregate unavailable) —
    // bring up the miniaudio duplex instead so we never regress below the
    // baseline.
    printf("[Capture::start] CADuplex failed; falling back to miniaudio\n");
    fflush(stdout);
    g_caActive = false;
    result = ma_device_init(&context, &deviceConfig, &device);
    if (result != MA_SUCCESS) {
      printf("[Capture::start] miniaudio fallback init failed: %d\n", result);
      return failedToStartDevice;
    }
  }
#endif

  result = ma_device_start(&device);
  if (result != MA_SUCCESS) {
#ifdef _IS_ANDROID_
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                        "Failed to start device: %d (%s).", result,
                        ma_result_description(result));
#else
    printf("Failed to start device: %d (%s).\n", result,
           ma_result_description(result));
#endif
    return failedToStartDevice;
  }
  return captureNoError;
}

void Capture::stop() {
#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_OSX)
  if (g_caActive) {
    caDuplexStop();
    return;
  }
#endif
  ma_device_stop(&device);
}

void Capture::startStreamingData() {
  // Channel count for this stream: matches the f32 conversion in
  // data_callback (captureChannels), read from the live device config so the
  // worker's chunk-size math agrees with what push() actually sends.
  StreamDispatchWorker::instance().mChunkFrameSize =
      static_cast<int>(sizeof(float)) *
      static_cast<int>(deviceConfig.capture.channels > 0
                           ? deviceConfig.capture.channels
                           : 1);
  StreamDispatchWorker::instance().start();
  isStreamingData = true;
}

void Capture::stopStreamingData() {
  isStreamingData = false;
  StreamDispatchWorker::instance().stop();
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
  // Use ACTUAL device values (deviceConfig may have channels=0 in Android
  // auto mode)
  int channels = device.capture.channels;
  if (channels < 1)
    channels = 1; // Safety fallback
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
  // regardless of device format (s16/s32 are converted in data_callback)
  actualConfig.capture.format = ma_format_f32;
  actualConfig.capture.channels = device.capture.channels;
  actualConfig.sampleRate = device.sampleRate;

#ifdef _IS_ANDROID_
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                      "[WAV INIT] path=%s, format=f32(%d), channels=%d, "
                      "sampleRate=%d",
                      path, ma_format_f32, actualConfig.capture.channels,
                      actualConfig.sampleRate);
#endif
  printf("[WAV INIT] path=%s, format=f32(%d), channels=%d, sampleRate=%d\n",
         path, ma_format_f32, actualConfig.capture.channels,
         actualConfig.sampleRate);

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

void Capture::writePrerollToWav(const float *samples, size_t numSamples) {
  if (!mInited || !isRecording || samples == nullptr || numSamples == 0)
    return;

  // Get actual channel count for frame calculation
  unsigned int channels = getCaptureChannels();
  if (channels < 1)
    channels = 1;

  size_t frameCount = numSamples / channels;
  wav.write((void *)samples, frameCount);
  printf("[Capture] Wrote %zu preroll frames (%zu samples) to WAV\n",
         frameCount, numSamples);
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
    // IMPORTANT: Use ACTUAL device channels, not configured (which may be 0
    // in auto mode)
    int channels = device.capture.channels;
    if (channels < 1)
      channels = 1; // Safety fallback
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
  // mCalibrationMutex serializes API-thread callers only; the audio callback
  // is lock-free (see capture.h). The resize below must never race a
  // callback write, so if a capture is somehow still active, deactivate and
  // let any in-flight callback block drain before touching the buffer.
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  if (mCalibrationActive.load(std::memory_order_acquire)) {
    mCalibrationActive.store(false, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  mCalibrationBuffer.resize(maxSamples, 0.0f);
  mCalibrationWritePos.store(0, std::memory_order_relaxed);
  // Release pairs with the callback's acquire: buffer is fully sized before
  // the callback can observe active==true.
  mCalibrationActive.store(true, std::memory_order_release);
}

void Capture::stopCalibrationCapture() {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  mCalibrationActive.store(false, std::memory_order_release);
}

size_t Capture::readCalibrationSamples(float *dest, size_t maxSamples) {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  size_t samplesToRead =
      std::min(maxSamples, mCalibrationWritePos.load(std::memory_order_acquire));
  if (samplesToRead > 0 && dest != nullptr) {
    memcpy(dest, mCalibrationBuffer.data(), samplesToRead * sizeof(float));
  }
  return samplesToRead;
}

bool Capture::isCalibrationCaptureActive() const {
  return mCalibrationActive.load(std::memory_order_acquire);
}
