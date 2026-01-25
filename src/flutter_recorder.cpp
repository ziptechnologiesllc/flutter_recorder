#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "analyzer.h"
#include "capture.h"
#include "filters/aec/aec_test.h"
#include "filters/aec/calibration.h"
#include "filters/aec/reference_buffer.h"
#include "filters/filters.h"
#include "flutter_recorder.h"
#include "native_ring_buffer.h"
#include "native_scheduler.h"
#include "soloud_slave_bridge.h"

#include <cmath>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <atomic>

#include "enums.h"

// External logging function defined in calibration.cpp
extern void aecLog(const char *fmt, ...);

// Define global callback pointers
dartSilenceChangedCallback_t dartSilenceChangedCallback = nullptr;
dartSilenceChangedCallback_t nativeSilenceChangedCallback = nullptr;
dartStreamDataCallback_t dartStreamDataCallback = nullptr;
dartStreamDataCallback_t nativeStreamDataCallback = nullptr;
AecStatsCallback dartAecStatsCallback = nullptr;
AecStatsCallback nativeAecStatsCallback = nullptr;
dartRecordingStoppedCallback_t dartRecordingStoppedCallback = nullptr;
dartRecordingStartedCallback_t dartRecordingStartedCallback = nullptr;

//////////////////////////////////////////////////////////////
/// NATIVE AUDIO SINK - Direct native-to-native streaming
/// This allows streaming audio directly to SoLoud without
/// crossing to Dart (avoiding UI thread contention).
//////////////////////////////////////////////////////////////

// Native sink callback type - matches flutter_soloud's expected signature
typedef void (*NativeAudioSinkCallback)(const unsigned char* data, unsigned int dataLen, void* userData);

// Storage for the native sink
static NativeAudioSinkCallback g_nativeAudioSink = nullptr;
static void* g_nativeAudioSinkUserData = nullptr;
static std::atomic<bool> g_nativeAudioSinkEnabled{false};

// Looper bridge - direct native-to-SoLoud playback (raw PCM, no WAV)
static LooperLoadAndPlayRawFunc g_looperBridge = nullptr;

// Worker thread for looper bridge (avoids blocking audio thread)
static std::thread g_looperWorkerThread;
static std::atomic<bool> g_looperWorkReady{false};
static std::atomic<bool> g_looperWorkerRunning{false};

// Forward declarations for recorded audio storage (defined later in file)
extern float* g_lastRecordedAudio;
extern size_t g_lastRecordedFrameCount;
extern unsigned int g_lastRecordedChannels;
extern unsigned int g_lastRecordedSampleRate;

// Worker thread function - waits for work, calls looper bridge
static void looperWorkerThreadFunc() {
  fprintf(stderr, "[Looper Worker] Thread started\n");

  while (g_looperWorkerRunning.load(std::memory_order_acquire)) {
    // Spin-wait for work (low latency, no syscalls)
    if (g_looperWorkReady.load(std::memory_order_acquire)) {
      g_looperWorkReady.store(false, std::memory_order_release);

      // Do the actual work - this can block, we're not on audio thread
      if (g_looperBridge != nullptr &&
          g_lastRecordedAudio != nullptr &&
          g_lastRecordedFrameCount > 0) {

        unsigned int numSamples = g_lastRecordedFrameCount * g_lastRecordedChannels;
        fprintf(stderr, "[Looper Worker] Starting playback: %u samples (%zu frames, %u ch @ %u Hz)\n",
                numSamples, g_lastRecordedFrameCount, g_lastRecordedChannels, g_lastRecordedSampleRate);

        unsigned int handle = 0;
        // Pass raw float samples directly - no WAV container!
        // copy=false: SoLoud uses our buffer directly (zero copy)
        // takeOwnership=true: SoLoud will free the buffer when sound is disposed
        unsigned int soundHash = g_looperBridge(
            g_lastRecordedAudio,
            numSamples,
            (float)g_lastRecordedSampleRate,
            g_lastRecordedChannels,
            false,  // copy - NO COPY, use buffer directly
            true,   // takeOwnership - SoLoud frees when done
            &handle
        );

        if (soundHash != 0) {
          // SoLoud now owns the buffer - clear our pointer
          g_lastRecordedAudio = nullptr;
          g_lastRecordedFrameCount = 0;
          fprintf(stderr, "[Looper Worker] Playback started: hash=%u handle=%u (buffer ownership transferred)\n",
                  soundHash, handle);
        } else {
          fprintf(stderr, "[Looper Worker] Failed to start playback\n");
        }
      }
    }

    // Brief sleep to avoid burning CPU when idle
    // 500us is still very low latency for playback start
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }

  fprintf(stderr, "[Looper Worker] Thread exiting\n");
}

// Start the looper worker thread
static void startLooperWorkerThread() {
  if (g_looperWorkerRunning.load(std::memory_order_acquire)) {
    return; // Already running
  }

  g_looperWorkerRunning.store(true, std::memory_order_release);
  g_looperWorkerThread = std::thread(looperWorkerThreadFunc);
  fprintf(stderr, "[Looper Worker] Thread launched\n");
}

// Stop the looper worker thread
static void stopLooperWorkerThread() {
  if (!g_looperWorkerRunning.load(std::memory_order_acquire)) {
    return; // Not running
  }

  g_looperWorkerRunning.store(false, std::memory_order_release);
  if (g_looperWorkerThread.joinable()) {
    g_looperWorkerThread.join();
  }
  fprintf(stderr, "[Looper Worker] Thread stopped\n");
}

// Set AEC stats callback for Dart FFI
FFI_PLUGIN_EXPORT void
flutter_recorder_set_aec_stats_callback(AecStatsCallback callback) {
  dartAecStatsCallback = callback;
}

// Set recording stopped callback for Dart FFI
// Called when recording auto-stops at loop boundary
FFI_PLUGIN_EXPORT void
flutter_recorder_setRecordingStoppedCallback(dartRecordingStoppedCallback_t callback) {
  dartRecordingStoppedCallback = callback;
  fprintf(stderr, "[Recorder] Recording stopped callback %s\n", callback ? "set" : "cleared");
}

// Set recording started callback for Dart FFI
// Called when recording starts (native scheduler fires StartRecording)
FFI_PLUGIN_EXPORT void
flutter_recorder_setRecordingStartedCallback(dartRecordingStartedCallback_t callback) {
  dartRecordingStartedCallback = callback;
  fprintf(stderr, "[Recorder] Recording started callback %s\n", callback ? "set" : "cleared");
}

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

Capture capture;
std::unique_ptr<Analyzer> analyzerCapture = std::make_unique<Analyzer>(256);
std::unique_ptr<Filters> mFilters = std::make_unique<Filters>(0);

//////////////////////////////////////////////////////////////
/// WEB WORKER

#ifdef __EMSCRIPTEN__
/// Create the web worker and store a global "RecorderModule.workerUri" in JS.
FFI_PLUGIN_EXPORT void flutter_recorder_createWorkerInWasm() {
  EM_ASM({
    if (!RecorderModule.wasmWorker) {
      // Create a new Worker from the URI
      var workerUri = "assets/packages/flutter_recorder/web/worker.dart.js";
      RecorderModule.wasmWorker = new Worker(workerUri);
      console.log("EM_ASM creating web worker! " + workerUri + "  " +
                  RecorderModule.wasmWorker);
    } else {
      console.log("EM_ASM web worker already created!");
    }
  });
}

/// Post a new silence event message with the web worker.
FFI_PLUGIN_EXPORT void
flutter_recorder_sendSilenceEventToWorker(const char *message, bool isSilent,
                                          float energyDb) {
  EM_ASM(
      {
        if (RecorderModule.wasmWorker) {
          // Send the message
          RecorderModule.wasmWorker.postMessage({
            message : UTF8ToString($0),
            isSilent : $1,
            energyDb : $2,
          });
        } else {
          console.error('Worker not found.');
        }
      },
      message, isSilent, energyDb);
}

/// Post a stream of audio data with the web worker.
FFI_PLUGIN_EXPORT void flutter_recorder_sendStreamToWorker(
    const char *message, const unsigned char *audioData, int audioDataLength) {
  EM_ASM(
      {
        if (RecorderModule.wasmWorker) {
          // Convert audioData to Uint8Array for JavaScript compatibility
          const audioDataArray =
              new Uint8Array(RecorderModule.HEAPU8.subarray($1, $1 + $2));
          // Send the message and data
          RecorderModule.wasmWorker.postMessage({
            message : UTF8ToString($0),
            data : audioDataArray,
          });
        } else {
          console.error('Worker not found.');
        }
      },
      message, audioData, audioDataLength);
}
#endif

void silenceChangedCallback(bool *isSilent, float *energyDb) {
#ifdef __EMSCRIPTEN__
  // Calling JavaScript from C/C++
  // https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#interacting-with-code-call-javascript-from-native
  // emscripten_run_script("voiceEndedCallbackJS('1234')");
  flutter_recorder_sendSilenceEventToWorker("silenceChangedCallback", *isSilent,
                                            *energyDb);
#endif
  if (dartSilenceChangedCallback != nullptr)
    dartSilenceChangedCallback(isSilent, energyDb);
}

void streamDataCallback(const unsigned char *samples, const int numSamples) {
#ifdef __EMSCRIPTEN__
  flutter_recorder_sendStreamToWorker("streamDataCallback", samples,
                                      numSamples);
#endif

  // NATIVE-TO-NATIVE PATH: If native sink is enabled, call it directly
  // This handles the audio streaming to SoLoud without crossing to Dart
  if (g_nativeAudioSinkEnabled.load(std::memory_order_acquire) && g_nativeAudioSink != nullptr) {
    g_nativeAudioSink(samples, numSamples, g_nativeAudioSinkUserData);
    // NOTE: We still call Dart callback below for:
    // 1. Ring buffer updates (latency compensation)
    // 2. LaunchScheduler updates (precision timing)
    // Dart will skip heavy work (SoLoud streaming) based on _nativeAudioSinkActive flag
  }

  // Dart callback - needed for ring buffer, scheduler, and visualization
  // When native sink is active, Dart skips SoLoud streaming but still does timing
  if (dartStreamDataCallback != nullptr)
    dartStreamDataCallback(samples, numSamples);
}

/// Set a Dart functions to call when an event occurs.
FFI_PLUGIN_EXPORT void flutter_recorder_setDartEventCallback(
    dartSilenceChangedCallback_t silence_changed_callback,
    dartStreamDataCallback_t stream_data_callback) {
  dartSilenceChangedCallback = silence_changed_callback;
  nativeSilenceChangedCallback = silenceChangedCallback;

  dartStreamDataCallback = stream_data_callback;
  nativeStreamDataCallback = streamDataCallback;
}

//////////////////////////////////////////////////////////////
/// NATIVE AUDIO SINK API
//////////////////////////////////////////////////////////////

/// Set the native audio sink callback for direct native-to-native streaming.
/// This allows audio data to flow directly from recorder to player without
/// crossing to Dart's main thread (avoiding UI thread contention).
///
/// [callback] - function pointer from flutter_soloud's configureNativeAudioSink
/// [userData] - user data pointer from flutter_soloud's configureNativeAudioSink
FFI_PLUGIN_EXPORT void flutter_recorder_setNativeAudioSink(
    NativeAudioSinkCallback callback,
    void* userData) {
  g_nativeAudioSink = callback;
  g_nativeAudioSinkUserData = userData;
  g_nativeAudioSinkEnabled.store(callback != nullptr, std::memory_order_release);
  fprintf(stderr, "[Recorder] Native audio sink %s\n",
          callback ? "enabled" : "disabled");
}

/// Check if native audio sink is active
FFI_PLUGIN_EXPORT bool flutter_recorder_isNativeAudioSinkActive() {
  return g_nativeAudioSinkEnabled.load(std::memory_order_acquire);
}

/// Disable the native audio sink (data will go through Dart again)
FFI_PLUGIN_EXPORT void flutter_recorder_disableNativeAudioSink() {
  g_nativeAudioSinkEnabled.store(false, std::memory_order_release);
  g_nativeAudioSink = nullptr;
  g_nativeAudioSinkUserData = nullptr;
  fprintf(stderr, "[Recorder] Native audio sink disabled\n");
}

/// Set the looper bridge function pointer for direct native-to-SoLoud playback
FFI_PLUGIN_EXPORT void flutter_recorder_setLooperBridge(LooperLoadAndPlayRawFunc func) {
  g_looperBridge = func;
  fprintf(stderr, "[Recorder] Looper bridge %s\n", func ? "set" : "cleared");

  // Start worker thread when bridge is set
  if (func != nullptr) {
    startLooperWorkerThread();
  }
}

/// Clear the looper bridge
FFI_PLUGIN_EXPORT void flutter_recorder_clearLooperBridge() {
  g_looperBridge = nullptr;
  stopLooperWorkerThread();
  fprintf(stderr, "[Recorder] Looper bridge cleared\n");
}

/// Inject preroll audio from ring buffer directly into SoLoud stream.
/// This reads from the native ring buffer and sends to the native audio sink
/// callback, keeping the entire path native (no Dart involvement).
///
/// [frameCount] - number of frames to inject from the ring buffer
FFI_PLUGIN_EXPORT void flutter_recorder_injectPreroll(size_t frameCount) {
  if (!g_nativeRingBuffer) {
    fprintf(stderr, "[Recorder] injectPreroll: no ring buffer\n");
    return;
  }
  if (!g_nativeAudioSinkEnabled.load(std::memory_order_acquire) || !g_nativeAudioSink) {
    fprintf(stderr, "[Recorder] injectPreroll: native audio sink not active\n");
    return;
  }

  const unsigned int channels = g_nativeRingBuffer->channels();
  const size_t numSamples = frameCount * channels;

  // Allocate temp buffer for preroll data
  std::vector<float> prerollData(numSamples);

  // Read preroll from ring buffer (most recent frameCount frames)
  // Pass rewindFrames=0 to get frames ending at current write position
  size_t framesRead = g_nativeRingBuffer->readPreRoll(prerollData.data(), frameCount, 0);

  if (framesRead > 0) {
    // Inject directly into SoLoud stream via native callback
    // Cast float* to unsigned char* - SoLoud stream expects F32LE bytes
    const size_t bytesToSend = framesRead * channels * sizeof(float);
    g_nativeAudioSink(reinterpret_cast<const unsigned char*>(prerollData.data()),
                      static_cast<unsigned int>(bytesToSend),
                      g_nativeAudioSinkUserData);
    fprintf(stderr, "[Recorder] Injected %zu frames (%zu bytes) preroll via native path\n",
            framesRead, bytesToSend);
  } else {
    fprintf(stderr, "[Recorder] injectPreroll: no data read from ring buffer\n");
  }
}

FFI_PLUGIN_EXPORT void flutter_recorder_nativeFree(void *pointer) {
  free(pointer);
}

// ///////////////////////////////
// Capture bindings functions
// ///////////////////////////////
FFI_PLUGIN_EXPORT void flutter_recorder_listCaptureDevices(char **devicesName,
                                                           int **deviceId,
                                                           int **isDefault,
                                                           int *n_devices) {
  std::vector<CaptureDevice> d = capture.listCaptureDevices();

  int numDevices = 0;
  for (int i = 0; i < (int)d.size(); i++) {
    bool hasSpecialChar = false;
    /// check if the device name has some strange chars (happens on Linux)
    for (int n = 0; n < 5; n++) {
      if (d[i].name[n] < 0x20 && d[i].name[n] >= 0)
        hasSpecialChar = true;
    }
    if (strlen(d[i].name) <= 5 || hasSpecialChar)
      continue;

    devicesName[i] = strdup(d[i].name);
    isDefault[i] = (int *)malloc(sizeof(int *));
    *isDefault[i] = d[i].isDefault;
    deviceId[i] = (int *)malloc(sizeof(int *));
    *deviceId[i] = d[i].id;

    numDevices++;
  }
  *n_devices = numDevices;
}

FFI_PLUGIN_EXPORT void
flutter_recorder_freeListCaptureDevices(char **devicesName, int **deviceId,
                                        int **isDefault, int n_devices) {
  for (int i = 0; i < n_devices; i++) {
    free(devicesName[i]);
    free(deviceId[i]);
    free(isDefault[i]);
  }
}

FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_init(int deviceID, int pcmFormat, unsigned int sampleRate,
                      unsigned int channels, int captureOnly) {
  if (!mFilters || mFilters.get()->mSamplerate != sampleRate ||
      mFilters.get()->mChannels != channels) {
    mFilters.reset();
    mFilters = std::make_unique<Filters>(sampleRate, channels);
  }
  CaptureErrors res = capture.init(mFilters.get(), deviceID,
                                   (PCMFormat)pcmFormat, sampleRate, channels,
                                   captureOnly != 0);

  return res;
}

FFI_PLUGIN_EXPORT void flutter_recorder_deinit() {
  if (capture.isRecording)
    capture.stopRecording();
  capture.dispose();
}

FFI_PLUGIN_EXPORT int flutter_recorder_isInited() {
  return capture.isInited() ? 1 : 0;
}

FFI_PLUGIN_EXPORT int flutter_recorder_isDeviceStarted() {
  return capture.isDeviceStarted();
}

FFI_PLUGIN_EXPORT int flutter_recorder_isCaptureStarted() {
  return capture.isDeviceStarted() ? 1 : 0;
}

FFI_PLUGIN_EXPORT enum CaptureErrors flutter_recorder_start() {
  if (!capture.isInited())
    return captureNotInited;
  return capture.start();
}

FFI_PLUGIN_EXPORT void flutter_recorder_stop() {
  if (capture.isRecording)
    capture.stopRecording();
  capture.stop();
}

FFI_PLUGIN_EXPORT void flutter_recorder_startStreamingData() {
  if (!capture.isInited())
    return;
  capture.startStreamingData();
}

FFI_PLUGIN_EXPORT void flutter_recorder_stopStreamingData() {
  if (!capture.isInited())
    return;
  capture.stopStreamingData();
}

FFI_PLUGIN_EXPORT void flutter_recorder_setSilenceDetection(bool enable) {
  capture.setSilenceDetection(enable);
}

FFI_PLUGIN_EXPORT void
flutter_recorder_setSilenceThresholdDb(float silenceThresholdDb) {
  if (!capture.isInited())
    return;
  capture.setSilenceThresholdDb(silenceThresholdDb);
}

FFI_PLUGIN_EXPORT void
flutter_recorder_setSilenceDuration(float silenceDuration) {
  if (!capture.isInited())
    return;
  capture.setSilenceDuration(silenceDuration);
}

FFI_PLUGIN_EXPORT void flutter_recorder_setSecondsOfAudioToWriteBefore(
    float secondsOfAudioToWriteBefore) {
  if (!capture.isInited())
    return;
  capture.setSecondsOfAudioToWriteBefore(secondsOfAudioToWriteBefore);
}

// Track if recording is scheduled or active (for idempotency)
// Not static - accessed by native_scheduler.cpp via extern
std::atomic<bool> g_recordingScheduledOrActive{false};
static char g_pendingRecordingPath[512] = {0};

FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_startRecording(const char *path) {
  if (!capture.isInited())
    return captureNotInited;

  // IDEMPOTENT: If already recording or scheduled, return success
  if (g_recordingScheduledOrActive.load(std::memory_order_acquire)) {
    printf("[Recorder] startRecording called but already active/scheduled - idempotent return\n");
    return captureNoError;
  }

  // Check if we have a base loop (loop mode) - use native scheduler
  int64_t baseLoopFrames = NativeScheduler::instance().getBaseLoopFrames();

  if (baseLoopFrames > 0) {
    // LOOP MODE: Schedule recording start for next loop boundary
    printf("[Recorder] Loop mode detected (baseLoop=%lld frames) - scheduling quantized start\n",
           (long long)baseLoopFrames);

    // Store path for when scheduler fires
    strncpy(g_pendingRecordingPath, path, sizeof(g_pendingRecordingPath) - 1);
    g_pendingRecordingPath[sizeof(g_pendingRecordingPath) - 1] = '\0';

    // Schedule via native scheduler
    uint32_t eventId = NativeScheduler::instance().scheduleQuantizedStart(path);
    if (eventId > 0) {
      g_recordingScheduledOrActive.store(true, std::memory_order_release);
      printf("[Recorder] Scheduled recording start, event ID: %u\n", eventId);
      return captureNoError;
    } else {
      printf("[Recorder] Failed to schedule recording start\n");
      return captureInitFailed;
    }
  } else {
    // FREE MODE: Start recording immediately via ring buffer
    printf("[Recorder] Free mode - starting ring buffer recording immediately\n");

    // Store path for when we stop and write the WAV
    strncpy(g_pendingRecordingPath, path, sizeof(g_pendingRecordingPath) - 1);
    g_pendingRecordingPath[sizeof(g_pendingRecordingPath) - 1] = '\0';

    // Start ring buffer recording with latency compensation
    if (g_nativeRingBuffer) {
      int64_t latencyFrames = NativeScheduler::instance().getLatencyCompensationFrames();
      g_nativeRingBuffer->startRecording(latencyFrames);
      g_recordingScheduledOrActive.store(true, std::memory_order_release);
      printf("[Recorder] Ring buffer recording started, latencyComp=%lld frames\n",
             (long long)latencyFrames);
      return captureNoError;
    } else {
      printf("[Recorder] ERROR: No ring buffer available\n");
      return captureInitFailed;
    }
  }
}

FFI_PLUGIN_EXPORT void flutter_recorder_setPauseRecording(bool pause) {
  if (!capture.isInited())
    return;
  capture.setPauseRecording(pause);
}

FFI_PLUGIN_EXPORT void flutter_recorder_stopRecording() {
  if (!capture.isInited())
    return;

  // Check if ring buffer is recording
  if (g_nativeRingBuffer && g_nativeRingBuffer->isRecording()) {
    // MANUAL STOP: Always stop immediately (at current frame)
    // Cancel any pending stop events (e.g., auto-scheduled loop stop)
    NativeScheduler::instance().cancelAllEvents();

    // Determine WAV path - prefer scheduler's active path, fallback to pending path
    const char* wavPath = NativeScheduler::instance().isRecordingActive()
        ? nullptr  // executeEvent will use mActiveRecordingPath
        : g_pendingRecordingPath;

    int64_t currentFrame = NativeScheduler::instance().getGlobalFrame();
    printf("[Recorder] MANUAL STOP: Stopping immediately at frame %lld\n", (long long)currentFrame);

    // Schedule immediate stop via scheduler (thread-safe, runs in audio callback)
    uint32_t eventId = NativeScheduler::instance().scheduleEvent(
        SchedulerAction::StopRecording, currentFrame, wavPath);
    printf("[Recorder] Scheduled immediate stop via scheduler, event ID: %u\n", eventId);
    return;
  }

  // No active recording
  printf("[Recorder] stopRecording called but not recording\n");
  g_recordingScheduledOrActive.store(false, std::memory_order_release);
  g_pendingRecordingPath[0] = '\0';
}

FFI_PLUGIN_EXPORT void flutter_recorder_getVolumeDb(float *volumeDb) {
  if (!capture.isInited()) {
    *volumeDb = 0;
    return;
  }
  *volumeDb = capture.getVolumeDb();
}

FFI_PLUGIN_EXPORT void flutter_recorder_setFftSmoothing(float smooth) {
  if (!capture.isInited())
    return;
  analyzerCapture.get()->setSmoothing(smooth);
}

/// Return a 256 float array containing FFT data.
FFI_PLUGIN_EXPORT void flutter_recorder_getFft(float **fft,
                                               bool *isTheSameAsBefore) {
  if (!capture.isInited())
    return;
  float *wave = capture.getWave(isTheSameAsBefore);
  *fft = analyzerCapture.get()->calcFFT(wave);
}

/// Return a 256 float array containing wave data.
FFI_PLUGIN_EXPORT void flutter_recorder_getWave(float **wave,
                                                bool *isTheSameAsBefore) {
  if (!capture.isInited())
    return;
  *wave = capture.getWave(isTheSameAsBefore);
}

// Getters for actual device parameters
FFI_PLUGIN_EXPORT unsigned int flutter_recorder_getSampleRate() {
  if (!capture.isInited()) return 0;
  return capture.getSampleRate();
}

FFI_PLUGIN_EXPORT unsigned int flutter_recorder_getCaptureChannels() {
  if (!capture.isInited()) return 0;
  return capture.getCaptureChannels();
}

FFI_PLUGIN_EXPORT unsigned int flutter_recorder_getPlaybackChannels() {
  if (!capture.isInited()) return 0;
  return capture.getPlaybackChannels();
}

FFI_PLUGIN_EXPORT int flutter_recorder_getCaptureFormat() {
  if (!capture.isInited()) return 0;
  return capture.getCaptureFormat();
}

FFI_PLUGIN_EXPORT int flutter_recorder_getPlaybackFormat() {
  if (!capture.isInited()) return 0;
  return capture.getPlaybackFormat();
}

float capturedTexture[512];
FFI_PLUGIN_EXPORT void flutter_recorder_getTexture(float **samples,
                                                   bool *isTheSameAsBefore) {
  if (!capture.isInited())
    return;
  if (analyzerCapture.get() == nullptr || !capture.isInited()) {
    *samples = capturedTexture;
    memset(*samples, 0, sizeof(float) * 512);
    *isTheSameAsBefore = true;
    return;
  }

  float *wave = capture.getWave(isTheSameAsBefore);
  float *fft = analyzerCapture.get()->calcFFT(wave);

  memcpy(capturedTexture, fft, sizeof(float) * 256);
  memcpy(capturedTexture + 256, wave, sizeof(float) * 256);
  *samples = capturedTexture;
}

float capturedTexture2D[256][512];
FFI_PLUGIN_EXPORT void flutter_recorder_getTexture2D(float **samples,
                                                     bool *isTheSameAsBefore) {
  if (!capture.isInited())
    return;
  if (analyzerCapture.get() == nullptr) {
    *samples = *capturedTexture2D;
    memset(*samples, 0, sizeof(float) * 512 * 256);
    *isTheSameAsBefore = true;
    return;
  }

  float *wave = capture.getWave(isTheSameAsBefore);
  float *fft = analyzerCapture.get()->calcFFT(wave);
  if (*isTheSameAsBefore) {
    *samples = *capturedTexture2D;
    return;
  }

  /// shift up 1 row
  memmove(capturedTexture2D[1], capturedTexture2D[0],
          sizeof(float) * 512 * 255);
  /// store the new 1st row
  memcpy(capturedTexture2D[0], fft, sizeof(float) * 256);
  memcpy(capturedTexture2D[0] + 256, wave, sizeof(float) * 256);

  *samples = *capturedTexture2D;
  *isTheSameAsBefore = false;
}

FFI_PLUGIN_EXPORT float flutter_recorder_getTextureValue(int row, int column) {
  if (!capture.isInited())
    return .0f;
  return capturedTexture2D[row][column];
}

/////////////////////////
/// FILTERS
/////////////////////////
FFI_PLUGIN_EXPORT int
flutter_recorder_isFilterActive(enum RecorderFilterType filterType) {
  return mFilters.get()->isFilterActive(filterType);
}

FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_addFilter(enum RecorderFilterType filterType) {
  aecLog("[FFI] flutter_recorder_addFilter called with type %d\n",
         static_cast<int>(filterType));
  CaptureErrors result = mFilters.get()->addFilter(filterType);
  aecLog("[FFI] flutter_recorder_addFilter result: %d\n",
         static_cast<int>(result));
  return result;
}

FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_removeFilter(enum RecorderFilterType filterType) {
  return mFilters.get()->removeFilter(filterType);
}

FFI_PLUGIN_EXPORT void
flutter_recorder_getFilterParamNames(enum RecorderFilterType filterType,
                                     char **names, int *paramsCount) {
  std::vector<std::string> pNames =
      mFilters.get()->getFilterParamNames(filterType);
  *paramsCount = static_cast<int>(pNames.size());
  *names = (char *)malloc(sizeof(char *) * *paramsCount);
  for (int i = 0; i < *paramsCount; i++) {
    names[i] = strdup(pNames[i].c_str());
    printf("C  i: %d  names[i]: %s  names[i]: %p\n", i, names[i], names[i]);
  }
}

FFI_PLUGIN_EXPORT void
flutter_recorder_setFilterParams(enum RecorderFilterType filterType,
                                 int attributeId, float value) {
  mFilters.get()->setFilterParams(filterType, attributeId, value);
}

FFI_PLUGIN_EXPORT float
flutter_recorder_getFilterParams(enum RecorderFilterType filterType,
                                 int attributeId) {
  return mFilters.get()->getFilterParams(filterType, attributeId);
}

/////////////////////////
/// FILTER DEBUG STATS
/////////////////////////

FFI_PLUGIN_EXPORT uint64_t flutter_recorder_getFilterMissCount() {
  return mFilters ? mFilters->getFilterMissCount() : 0;
}

FFI_PLUGIN_EXPORT uint64_t flutter_recorder_getFilterProcessCount() {
  return mFilters ? mFilters->getFilterProcessCount() : 0;
}

FFI_PLUGIN_EXPORT void flutter_recorder_resetFilterStats() {
  if (mFilters) {
    mFilters->resetFilterStats();
  }
}

/////////////////////////
/// MONITORING
/////////////////////////

FFI_PLUGIN_EXPORT void flutter_recorder_setMonitoring(bool enabled) {
  capture.monitoringEnabled = enabled;
}

FFI_PLUGIN_EXPORT void flutter_recorder_setMonitoringMode(int mode) {
  capture.monitoringMode = mode;
}

/////////////////////////
/// SLAVE MODE
/////////////////////////

// Check if slave audio is ready (first callback has run successfully)
// This is used to wait for the audio pipeline to stabilize before calibration
FFI_PLUGIN_EXPORT int flutter_recorder_isSlaveAudioReady() {
  return soloud_isSlaveAudioReady() ? 1 : 0;
}

/////////////////////////
/// AEC (Adaptive Echo Cancellation)
/////////////////////////

// Callback function that SoLoud calls to send its output audio
// Handles channel mismatch: converts SoLoud output to match reference buffer channels
static void aecOutputCallback(const float *data, size_t frameCount,
                              unsigned int channels) {
  if (g_aecReferenceBuffer == nullptr) {
    static int warnCount = 0;
    if (++warnCount <= 10) {
      aecLog("[AEC Callback] WARNING: Buffer not initialized!\n");
    }
    return;
  }

  unsigned int bufferCh = g_aecReferenceBuffer->channels();

  // Log periodically with channel info
  static int callbackCount = 0;
  if (++callbackCount % 100 == 0) {
    aecLog("[AEC Callback #%d] frames=%zu inputCh=%u bufferCh=%u\n",
           callbackCount, frameCount, channels, bufferCh);
  }

  if (channels == bufferCh) {
    // Channels match - direct write
    g_aecReferenceBuffer->write(data, frameCount);
  } else if (channels == 2 && bufferCh == 1) {
    // Stereo → Mono: average L+R
    static std::vector<float> mono;
    if (mono.size() < frameCount) mono.resize(frameCount);
    for (size_t i = 0; i < frameCount; ++i) {
      mono[i] = (data[i * 2] + data[i * 2 + 1]) * 0.5f;
    }
    g_aecReferenceBuffer->write(mono.data(), frameCount);
  } else if (channels == 1 && bufferCh == 2) {
    // Mono → Stereo: duplicate to both channels
    static std::vector<float> stereo;
    if (stereo.size() < frameCount * 2) stereo.resize(frameCount * 2);
    for (size_t i = 0; i < frameCount; ++i) {
      stereo[i * 2] = stereo[i * 2 + 1] = data[i];
    }
    g_aecReferenceBuffer->write(stereo.data(), frameCount);
  } else {
    // Unsupported channel conversion
    static int unsupportedWarnCount = 0;
    if (++unsupportedWarnCount <= 10) {
      aecLog("[AEC Callback] Unsupported channel conversion: %u → %u\n",
             channels, bufferCh);
    }
  }
}

// Configure the AEC reference buffer with the given sample rate and channels
// Note: Buffer is pre-allocated at static init time, this just reconfigures dimensions
FFI_PLUGIN_EXPORT void *
flutter_recorder_aec_createReferenceBuffer(unsigned int sampleRate,
                                           unsigned int channels) {
  // Buffer size: 2 seconds of audio to support calibration
  // Calibration signal is 1.5 seconds white noise plus delay margin
  // During normal AEC operation, only the most recent ~100ms is used
  size_t bufferSizeFrames = sampleRate * 2; // 2 seconds

  // The buffer is pre-allocated at static init time (reference_buffer.cpp)
  // Just reconfigure its dimensions to match the actual device
  if (g_aecReferenceBuffer != nullptr) {
    g_aecReferenceBuffer->configure(bufferSizeFrames, channels, sampleRate);
    aecLog("[AEC] Reference buffer configured: %zu frames @ %uHz, %u channels\n",
           bufferSizeFrames, sampleRate, channels);
  } else {
    aecLog("[AEC] ERROR: Pre-allocated reference buffer is null!\n");
  }

  return static_cast<void *>(g_aecReferenceBuffer);
}

// Reset the AEC reference buffer (don't destroy - it's statically allocated)
FFI_PLUGIN_EXPORT void flutter_recorder_aec_destroyReferenceBuffer() {
  // Don't delete - the buffer is statically allocated
  // Just reset its state
  if (g_aecReferenceBuffer != nullptr) {
    g_aecReferenceBuffer->reset();
    aecLog("[AEC] Reference buffer reset (static allocation preserved)\n");
  }
}

// Get the output callback function pointer (to be set in SoLoud)
FFI_PLUGIN_EXPORT void *flutter_recorder_aec_getOutputCallback() {
  return reinterpret_cast<void *>(&aecOutputCallback);
}

// Reset the AEC reference buffer
FFI_PLUGIN_EXPORT void flutter_recorder_aec_resetBuffer() {
  if (g_aecReferenceBuffer != nullptr) {
    g_aecReferenceBuffer->reset();
  }
}

// Enable/disable AEC reference buffer writes (saves CPU when AEC is not needed)
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setEnabled(bool enabled) {
  if (g_aecReferenceBuffer != nullptr) {
    g_aecReferenceBuffer->setEnabled(enabled);
  }
}

FFI_PLUGIN_EXPORT bool flutter_recorder_aec_isEnabled() {
  if (g_aecReferenceBuffer != nullptr) {
    return g_aecReferenceBuffer->isEnabled();
  }
  return false;
}

// AEC Mode Control
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setMode(int mode) {
  if (mFilters) {
    mFilters->setAecMode(static_cast<AecMode>(mode));
  }
}

FFI_PLUGIN_EXPORT int flutter_recorder_aec_getMode() {
  if (mFilters) {
    return static_cast<int>(mFilters->getAecMode());
  }
  return 3; // Default to Hybrid
}

/////////////////////////
/// Neural Model Control
/////////////////////////

FFI_PLUGIN_EXPORT int flutter_recorder_neural_loadModel(int modelType,
                                                        const char *assetBasePath) {
  if (!mFilters) {
    return 0; // Failure - filters not initialized
  }

  bool success = mFilters->loadNeuralModel(
      static_cast<NeuralModelType>(modelType), std::string(assetBasePath));

  return success ? 1 : 0;
}

FFI_PLUGIN_EXPORT int flutter_recorder_neural_getLoadedModel() {
  if (!mFilters) {
    return 0; // NONE
  }

  return static_cast<int>(mFilters->getLoadedNeuralModel());
}

FFI_PLUGIN_EXPORT void flutter_recorder_neural_setEnabled(int enabled) {
  if (mFilters) {
    mFilters->setNeuralEnabled(enabled != 0);
  }
}

FFI_PLUGIN_EXPORT int flutter_recorder_neural_isEnabled() {
  if (!mFilters) {
    return 0; // Disabled
  }

  return mFilters->isNeuralEnabled() ? 1 : 0;
}

/////////////////////////
/// AEC CALIBRATION

/////////////////////////
/// AEC CALIBRATION
/////////////////////////

// Generate calibration WAV data
// signalType: 0 = Chirp (log sweep), 1 = Click (impulse train)
// Returns pointer to WAV data that caller must free with
// flutter_recorder_nativeFree
FFI_PLUGIN_EXPORT uint8_t *flutter_recorder_aec_generateCalibrationSignal(
    unsigned int sampleRate, unsigned int channels, size_t *outSize,
    int signalType) {
  CalibrationSignalType type = (signalType == 1)
      ? CalibrationSignalType::Click
      : CalibrationSignalType::Chirp;
  return AECCalibration::generateCalibrationWav(sampleRate, channels, outSize, type);
}

// Start capturing samples for calibration analysis
FFI_PLUGIN_EXPORT void
flutter_recorder_aec_startCalibrationCapture(size_t maxSamples) {
  // Start mic calibration capture
  capture.startCalibrationCapture(maxSamples);
  // Also start reference buffer calibration capture with same frame count
  // Note: maxSamples from Dart may include channel multiplier, but ref buffer
  // captures mono so we use the same value which covers the full duration
  if (g_aecReferenceBuffer != nullptr) {
    g_aecReferenceBuffer->startCalibrationCapture(maxSamples);
  }
}

// Stop capturing samples for calibration
FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopCalibrationCapture() {
  capture.stopCalibrationCapture();
  if (g_aecReferenceBuffer != nullptr) {
    g_aecReferenceBuffer->stopCalibrationCapture();
  }
}

// Capture signals from both buffers for cross-correlation analysis
FFI_PLUGIN_EXPORT void flutter_recorder_aec_captureForAnalysis() {
  if (g_aecReferenceBuffer == nullptr) {
    aecLog("[AEC Calibration] Error: Reference buffer not initialized\n");
    return;
  }

  // Capture click-based calibration signal (~2.4s)
  // Duration: (CLICK_COUNT - 1) * CLICK_SPACING_MS + TAIL_MS
  const size_t calibrationDurationMs =
      (AECCalibration::CLICK_COUNT - 1) * AECCalibration::CLICK_SPACING_MS +
      AECCalibration::TAIL_MS + 100; // +100ms margin
  const size_t samplesToCapture = 48000 * calibrationDurationMs / 1000;

  std::vector<float> refData(samplesToCapture);
  std::vector<float> micData(samplesToCapture);

  // Read from AEC reference buffer (playback signal)
  size_t refRead = g_aecReferenceBuffer->readForCalibration(refData.data(),
                                                            samplesToCapture);

  // Read from mic capture buffer
  size_t micRead =
      capture.readCalibrationSamples(micData.data(), samplesToCapture);

  aecLog("[AEC Calibration] Captured ref=%zu mic=%zu samples\n", refRead,
         micRead);

  // Pass to calibration analyzer
  AECCalibration::captureSignals(refData.data(), refRead, micData.data(),
                                 micRead);
}

// Run cross-correlation analysis and return results
FFI_PLUGIN_EXPORT int flutter_recorder_aec_runCalibrationAnalysis(
    unsigned int sampleRate, float *outDelayMs, float *outEchoGain,
    float *outCorrelation) {
  CalibrationResult result = AECCalibration::analyze(sampleRate);

  if (outDelayMs)
    *outDelayMs = result.delayMs; // Preserve full precision
  if (outEchoGain)
    *outEchoGain = result.echoGain;
  if (outCorrelation)
    *outCorrelation = result.correlation;

  printf(
      "[AEC Calibration] Result: delay=%.1fms gain=%.3f corr=%.3f success=%d\n",
      result.delayMs, result.echoGain, result.correlation, result.success);

  return result.success ? 1 : 0;
}

// Reset calibration state
FFI_PLUGIN_EXPORT void flutter_recorder_aec_resetCalibration() {
  AECCalibration::reset();
  capture.stopCalibrationCapture();
}

// Static storage for last calibration impulse response
static std::vector<float> g_lastImpulseResponse;

// Run calibration analysis and store impulse response
FFI_PLUGIN_EXPORT int flutter_recorder_aec_runCalibrationWithImpulse(
    unsigned int sampleRate, float *outDelayMs, float *outEchoGain,
    float *outCorrelation, int *outImpulseLength,
    int64_t *outCalibratedOffset) {
  CalibrationResult result = AECCalibration::analyze(sampleRate);

  if (outDelayMs)
    *outDelayMs = result.delayMs; // Preserve full precision
  if (outEchoGain)
    *outEchoGain = result.echoGain;
  if (outCorrelation)
    *outCorrelation = result.correlation;
  if (outImpulseLength)
    *outImpulseLength = static_cast<int>(result.impulseResponse.size());
  if (outCalibratedOffset)
    *outCalibratedOffset = result.calibratedOffset;

  // Store impulse response for later retrieval
  g_lastImpulseResponse = std::move(result.impulseResponse);

  aecLog("[AEC Calibration] Result: delay=%.1fms gain=%.3f corr=%.3f "
         "impulseLen=%zu offset=%lld success=%d\n",
         result.delayMs, result.echoGain, result.correlation,
         g_lastImpulseResponse.size(), (long long)result.calibratedOffset,
         result.success);

  return result.success ? 1 : 0;
}

// Get stored impulse response
FFI_PLUGIN_EXPORT int flutter_recorder_aec_getImpulseResponse(float *dest,
                                                              int maxLength) {
  if (!dest || maxLength <= 0)
    return 0;

  int copyLen =
      std::min(maxLength, static_cast<int>(g_lastImpulseResponse.size()));
  std::memcpy(dest, g_lastImpulseResponse.data(), copyLen * sizeof(float));

  return copyLen;
}

// Apply impulse response to AEC filter
FFI_PLUGIN_EXPORT void flutter_recorder_aec_applyImpulseResponse() {
  if (g_lastImpulseResponse.empty()) {
    aecLog("[AEC] No impulse response to apply\n");
    return;
  }

  mFilters->setAecImpulseResponse(
      g_lastImpulseResponse.data(),
      static_cast<int>(g_lastImpulseResponse.size()));

  aecLog("[AEC] Applied impulse response: %zu coefficients\n",
         g_lastImpulseResponse.size());
}

// Get captured reference signal for visualization
// Prioritizes aligned buffers (from aligned calibration) over static buffers
FFI_PLUGIN_EXPORT int
flutter_recorder_aec_getCalibrationRefSignal(float *dest, int maxLength) {
  if (!dest || maxLength <= 0) return 0;

  // First try aligned buffers from mFilters (used by aligned calibration)
  if (mFilters) {
    const std::vector<float>& aligned = mFilters->getAecAlignedRef();
    if (!aligned.empty()) {
      int copyLen = std::min(maxLength, static_cast<int>(aligned.size()));
      std::memcpy(dest, aligned.data(), copyLen * sizeof(float));
      return copyLen;
    }
  }

  // Fall back to static buffers (old capture method)
  return AECCalibration::getRefSignal(dest, maxLength);
}

// Get captured mic signal for visualization
// Prioritizes aligned buffers (from aligned calibration) over static buffers
FFI_PLUGIN_EXPORT int
flutter_recorder_aec_getCalibrationMicSignal(float *dest, int maxLength) {
  if (!dest || maxLength <= 0) return 0;

  // First try aligned buffers from mFilters (used by aligned calibration)
  if (mFilters) {
    const std::vector<float>& aligned = mFilters->getAecAlignedMic();
    if (!aligned.empty()) {
      int copyLen = std::min(maxLength, static_cast<int>(aligned.size()));
      std::memcpy(dest, aligned.data(), copyLen * sizeof(float));
      return copyLen;
    }
  }

  // Fall back to static buffers (old capture method)
  return AECCalibration::getMicSignal(dest, maxLength);
}

// Set AEC delay from calibration result
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setDelay(float delayMs) {
  if (mFilters) {
    // DelayMs is parameter index 1 in AdaptiveEchoCancellation::Params
    mFilters->setFilterParams(adaptiveEchoCancellation, 1, delayMs);
    aecLog("[AEC] Set delay to %.1f ms\n", delayMs);
  }
}

// Apply full calibration result: delay + impulse response
FFI_PLUGIN_EXPORT void flutter_recorder_aec_applyCalibration(float delayMs) {
  if (!mFilters) {
    aecLog("[AEC] Filters not initialized\n");
    return;
  }

  // Set the calibrated delay
  mFilters->setFilterParams(adaptiveEchoCancellation, 1, delayMs);

  // Apply impulse response if available
  if (!g_lastImpulseResponse.empty()) {
    mFilters->setAecImpulseResponse(
        g_lastImpulseResponse.data(),
        static_cast<int>(g_lastImpulseResponse.size()));
    aecLog("[AEC] Applied calibration: delay=%.1fms, impulse=%zu coeffs\n",
           delayMs, g_lastImpulseResponse.size());
  } else {
    aecLog("[AEC] Applied calibration: delay=%.1fms (no impulse response)\n",
           delayMs);
  }
}

//////////////////////////
/// AEC Testing Functions
//////////////////////////

FFI_PLUGIN_EXPORT void
flutter_recorder_aec_startTestCapture(size_t maxSamples) {
  AECTest::startTestCapture(maxSamples);
}

FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopTestCapture() {
  AECTest::stopTestCapture();
}

FFI_PLUGIN_EXPORT int flutter_recorder_aec_runTest(
    unsigned int sampleRate, float *outCancellationDb,
    float *outCorrelationBefore, float *outCorrelationAfter, int *outPassed,
    float *outMicEnergyDb, float *outCancelledEnergyDb) {

  AecTestResult result = AECTest::analyze(sampleRate);

  if (outCancellationDb)
    *outCancellationDb = result.cancellationDb;
  if (outCorrelationBefore)
    *outCorrelationBefore = result.correlationBefore;
  if (outCorrelationAfter)
    *outCorrelationAfter = result.correlationAfter;
  if (outPassed)
    *outPassed = result.passed ? 1 : 0;
  if (outMicEnergyDb)
    *outMicEnergyDb = result.micEnergyDb;
  if (outCancelledEnergyDb)
    *outCancelledEnergyDb = result.cancelledEnergyDb;

  return result.passed ? 1 : 0;
}

FFI_PLUGIN_EXPORT int flutter_recorder_aec_getTestMicSignal(float *dest,
                                                            int maxLength) {
  return AECTest::getMicSignal(dest, maxLength);
}

FFI_PLUGIN_EXPORT int
flutter_recorder_aec_getTestCancelledSignal(float *dest, int maxLength) {
  return AECTest::getCancelledSignal(dest, maxLength);
}

FFI_PLUGIN_EXPORT void flutter_recorder_aec_resetTest() { AECTest::reset(); }

// VSS-NLMS parameter control for experimentation
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setVssMuMax(float mu) {
  if (mFilters) {
    mFilters->setAecVssMuMax(mu);
  }
}

FFI_PLUGIN_EXPORT void flutter_recorder_aec_setVssLeakage(float lambda) {
  if (mFilters) {
    mFilters->setAecVssLeakage(lambda);
  }
}

FFI_PLUGIN_EXPORT void flutter_recorder_aec_setVssAlpha(float alpha) {
  if (mFilters) {
    mFilters->setAecVssAlpha(alpha);
  }
}

FFI_PLUGIN_EXPORT float flutter_recorder_aec_getVssMuMax() {
  if (mFilters) {
    return mFilters->getAecVssMuMax();
  }
  return 0.5f; // Default
}

FFI_PLUGIN_EXPORT float flutter_recorder_aec_getVssLeakage() {
  if (mFilters) {
    return mFilters->getAecVssLeakage();
  }
  return 1.0f; // Default (no leakage)
}

FFI_PLUGIN_EXPORT float flutter_recorder_aec_getVssAlpha() {
  if (mFilters) {
    return mFilters->getAecVssAlpha();
  }
  return 0.95f; // Default
}

// Filter length control
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setFilterLength(int length) {
  if (mFilters) {
    mFilters->setAecFilterLength(length);
  }
}

FFI_PLUGIN_EXPORT int flutter_recorder_aec_getFilterLength() {
  if (mFilters) {
    return mFilters->getAecFilterLength();
  }
  return 8192; // Default
}

// Position-based sync for sample-accurate AEC
FFI_PLUGIN_EXPORT size_t flutter_recorder_aec_getOutputFrameCount() {
  if (g_aecReferenceBuffer) {
    return g_aecReferenceBuffer->getFramesWritten();
  }
  return 0;
}

FFI_PLUGIN_EXPORT size_t flutter_recorder_aec_getCaptureFrameCount() {
  return capture.getTotalFramesCaptured();
}

FFI_PLUGIN_EXPORT void flutter_recorder_aec_recordCalibrationFrameCounters() {
  size_t outputFrames =
      g_aecReferenceBuffer ? g_aecReferenceBuffer->getFramesWritten() : 0;
  size_t captureFrames = capture.getTotalFramesCaptured();
  AECCalibration::recordFrameCountersAtStart(outputFrames, captureFrames);
}

FFI_PLUGIN_EXPORT void
flutter_recorder_aec_setCalibratedOffset(int64_t offset) {
  if (mFilters) {
    mFilters->setAecCalibratedOffset(offset);
  }
}

FFI_PLUGIN_EXPORT int64_t flutter_recorder_aec_getCalibratedOffset() {
  if (mFilters) {
    return mFilters->getAecCalibratedOffset();
  }
  return 0;
}

// ==================== ALIGNED CALIBRATION CAPTURE ====================
// These functions capture frame-aligned ref/mic from the AEC processAudio
// callback for accurate delay estimation

FFI_PLUGIN_EXPORT void
flutter_recorder_aec_startAlignedCalibrationCapture(size_t maxSamples) {
  if (mFilters) {
    mFilters->startAecCalibrationCapture(maxSamples);
  } else {
    aecLog("[AEC] Cannot start aligned capture: filters not initialized\n");
  }
}

FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopAlignedCalibrationCapture() {
  if (mFilters) {
    mFilters->stopAecCalibrationCapture();
  }
}

FFI_PLUGIN_EXPORT int flutter_recorder_aec_runAlignedCalibrationAnalysis(
    unsigned int sampleRate, int *outDelaySamples, float *outDelayMs,
    float *outGain, float *outCorrelation, int64_t *outCalibratedOffset) {

  if (!mFilters) {
    aecLog("[AEC] Cannot run aligned analysis: filters not initialized\n");
    return 0;
  }

  const std::vector<float> &alignedRef = mFilters->getAecAlignedRef();
  const std::vector<float> &alignedMic = mFilters->getAecAlignedMic();

  if (alignedRef.empty() || alignedMic.empty()) {
    aecLog("[AEC] Cannot run aligned analysis: no captured data\n");
    return 0;
  }

  CalibrationResult result =
      AECCalibration::analyzeAligned(alignedRef, alignedMic, sampleRate);

  *outDelaySamples = result.delaySamples;
  *outDelayMs = result.delayMs;
  *outGain = result.echoGain;
  *outCorrelation = result.correlation;
  *outCalibratedOffset = result.calibratedOffset;

  // Store impulse response for later application
  static std::vector<float> sLastImpulseResponse;
  sLastImpulseResponse = result.impulseResponse;

  aecLog("[AEC Aligned Calibration] Result: delay=%.1fms gain=%.3f corr=%.3f "
         "success=%d offset=%lld\n",
         result.delayMs, result.echoGain, result.correlation,
         result.success ? 1 : 0, (long long)result.calibratedOffset);

  return result.success ? 1 : 0;
}

FFI_PLUGIN_EXPORT int flutter_recorder_aec_runAlignedCalibrationWithImpulse(
    unsigned int sampleRate, int *outDelaySamples, float *outDelayMs,
    float *outGain, float *outCorrelation, int *outImpulseLength,
    int64_t *outCalibratedOffset, int signalType) {

  if (!mFilters) {
    aecLog("[AEC] Cannot run aligned analysis: filters not initialized\n");
    return 0;
  }

  const std::vector<float> &alignedRef = mFilters->getAecAlignedRef();
  const std::vector<float> &alignedMic = mFilters->getAecAlignedMic();

  if (alignedRef.empty() || alignedMic.empty()) {
    aecLog("[AEC] Cannot run aligned analysis: no captured data\n");
    return 0;
  }

  CalibrationSignalType type = (signalType == 1)
      ? CalibrationSignalType::Click
      : CalibrationSignalType::Chirp;
  CalibrationResult result =
      AECCalibration::analyzeAligned(alignedRef, alignedMic, sampleRate, type);

  *outDelaySamples = result.delaySamples;
  *outDelayMs = result.delayMs;
  *outGain = result.echoGain;
  *outCorrelation = result.correlation;
  *outImpulseLength = static_cast<int>(result.impulseResponse.size());
  *outCalibratedOffset = result.calibratedOffset;

  // Store and apply impulse response
  if (result.success && !result.impulseResponse.empty()) {
    mFilters->setAecImpulseResponse(
        result.impulseResponse.data(),
        static_cast<int>(result.impulseResponse.size()));
    // Set acoustic delay for slave mode (pure room delay without thread timing)
    mFilters->setAecAcousticDelaySamples(result.delaySamples);
  }

  aecLog("[AEC Aligned Calibration] Result: delay=%.1fms gain=%.3f corr=%.3f "
         "impulseLen=%zu offset=%lld success=%d\n",
         result.delayMs, result.echoGain, result.correlation,
         result.impulseResponse.size(), (long long)result.calibratedOffset,
         result.success ? 1 : 0);

  return result.success ? 1 : 0;
}

/////////////////////////
/// NATIVE SCHEDULER
/////////////////////////

// Reset the native scheduler state
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_reset() {
  NativeScheduler::instance().reset();
}

// Set base loop parameters for quantization
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_setBaseLoop(int64_t loopFrames,
                                                               int64_t loopStartFrame) {
  NativeScheduler::instance().setBaseLoop(loopFrames, loopStartFrame);
}

// Clear base loop (free recording mode)
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_clearBaseLoop() {
  NativeScheduler::instance().clearBaseLoop();
}

// Schedule quantized recording start
// Returns event ID (0 if failed)
FFI_PLUGIN_EXPORT uint32_t flutter_recorder_scheduler_scheduleStart(const char* path) {
  return NativeScheduler::instance().scheduleQuantizedStart(path);
}

// Schedule quantized recording stop
// Returns event ID (0 if failed)
FFI_PLUGIN_EXPORT uint32_t flutter_recorder_scheduler_scheduleStop(int64_t startFrame) {
  return NativeScheduler::instance().scheduleQuantizedStop(startFrame);
}

// Schedule event at specific frame
FFI_PLUGIN_EXPORT uint32_t flutter_recorder_scheduler_scheduleEvent(
    int action, int64_t targetFrame, const char* path) {
  return NativeScheduler::instance().scheduleEvent(
      static_cast<SchedulerAction>(action), targetFrame, path);
}

// Cancel a scheduled event
FFI_PLUGIN_EXPORT int flutter_recorder_scheduler_cancelEvent(uint32_t eventId) {
  return NativeScheduler::instance().cancelEvent(eventId) ? 1 : 0;
}

// Cancel all pending events
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_cancelAll() {
  NativeScheduler::instance().cancelAllEvents();
}

// Poll for fired event notification
// Returns 1 if notification available, 0 if queue empty
FFI_PLUGIN_EXPORT int flutter_recorder_scheduler_pollNotification(
    uint32_t* outEventId, int* outAction, int64_t* outFiredFrame, int32_t* outLatency) {
  EventNotification notif;
  if (NativeScheduler::instance().pollNotification(&notif)) {
    if (outEventId) *outEventId = notif.eventId;
    if (outAction) *outAction = static_cast<int>(notif.action);
    if (outFiredFrame) *outFiredFrame = notif.firedAtFrame;
    if (outLatency) *outLatency = notif.latencyFrames;
    return 1;
  }
  return 0;
}

// Check if there are pending notifications
FFI_PLUGIN_EXPORT int flutter_recorder_scheduler_hasNotifications() {
  return NativeScheduler::instance().hasNotifications() ? 1 : 0;
}

// Get current global frame position
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getGlobalFrame() {
  return NativeScheduler::instance().getGlobalFrame();
}

// Get base loop length in frames
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getBaseLoopFrames() {
  return NativeScheduler::instance().getBaseLoopFrames();
}

// Get next loop boundary frame
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getNextLoopBoundary() {
  return NativeScheduler::instance().getNextLoopBoundary();
}

// Set latency compensation in frames (applied at recording start)
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_setLatencyCompensation(int64_t frames) {
  NativeScheduler::instance().setLatencyCompensationFrames(frames);
}

// Get latency compensation in frames
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getLatencyCompensation() {
  return NativeScheduler::instance().getLatencyCompensationFrames();
}

// Set auto-stop enabled (when true, STOP is scheduled upfront with START)
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_setAutoStop(bool enabled) {
  NativeScheduler::instance().setAutoStopEnabled(enabled);
  fprintf(stderr, "[Recorder] Scheduler auto-stop %s\n", enabled ? "enabled" : "disabled");
}

// Get auto-stop enabled state
FFI_PLUGIN_EXPORT bool flutter_recorder_scheduler_isAutoStopEnabled() {
  return NativeScheduler::instance().isAutoStopEnabled();
}

/////////////////////////
/// NATIVE RING BUFFER
/// Latency compensation via continuous capture with pre-roll
/////////////////////////

// Create/configure the native ring buffer for latency compensation
// capacitySeconds: How many seconds of audio to keep (typically 5)
// sampleRate: Sample rate in Hz
// channels: Number of channels (1=mono, 2=stereo)
FFI_PLUGIN_EXPORT void flutter_recorder_createRingBuffer(
    size_t capacitySeconds, unsigned int sampleRate, unsigned int channels) {
  size_t capacityFrames = capacitySeconds * sampleRate;

  if (g_nativeRingBuffer != nullptr) {
    // Reconfigure existing buffer
    g_nativeRingBuffer->configure(capacityFrames, channels, sampleRate);
    fprintf(stderr, "[Recorder] Native ring buffer reconfigured: %zu sec @ %uHz, %u ch\n",
            capacitySeconds, sampleRate, channels);
  } else {
    // Create new buffer
    g_nativeRingBuffer = new NativeRingBuffer(capacityFrames, channels, sampleRate);
    fprintf(stderr, "[Recorder] Native ring buffer created: %zu sec @ %uHz, %u ch\n",
            capacitySeconds, sampleRate, channels);
  }

  // Pre-allocate for recording (10% of available RAM)
  // This happens at init time so no allocations during audio streaming
  g_nativeRingBuffer->preAllocateForRecording();
}

// Destroy/reset the native ring buffer
FFI_PLUGIN_EXPORT void flutter_recorder_destroyRingBuffer() {
  if (g_nativeRingBuffer != nullptr) {
    delete g_nativeRingBuffer;
    g_nativeRingBuffer = nullptr;
    fprintf(stderr, "[Recorder] Native ring buffer destroyed\n");
  }
}

// Read pre-roll samples for latency compensation
// dest: Destination buffer (must be pre-allocated for frameCount * channels floats)
// frameCount: Number of frames to read
// rewindFrames: How many frames back in time to start reading
// Returns: Number of frames actually read (may be less if not enough data)
FFI_PLUGIN_EXPORT size_t flutter_recorder_readPreRoll(
    float* dest, size_t frameCount, size_t rewindFrames) {
  if (g_nativeRingBuffer == nullptr || dest == nullptr) {
    return 0;
  }
  return g_nativeRingBuffer->readPreRoll(dest, frameCount, rewindFrames);
}

// Get current audio level in dB (RMS)
// Calculated continuously in the audio callback
FFI_PLUGIN_EXPORT float flutter_recorder_getAudioLevelDb() {
  if (g_nativeRingBuffer == nullptr) {
    return -100.0f;
  }
  return g_nativeRingBuffer->getAudioLevelDb();
}

// Get total frames written to the ring buffer (for sync)
FFI_PLUGIN_EXPORT size_t flutter_recorder_getRingBufferFramesWritten() {
  if (g_nativeRingBuffer == nullptr) {
    return 0;
  }
  return g_nativeRingBuffer->getTotalFramesWritten();
}

// Get available frames in the ring buffer
FFI_PLUGIN_EXPORT size_t flutter_recorder_getRingBufferAvailable() {
  if (g_nativeRingBuffer == nullptr) {
    return 0;
  }
  return g_nativeRingBuffer->available();
}

// Reset the ring buffer (clear all data)
FFI_PLUGIN_EXPORT void flutter_recorder_resetRingBuffer() {
  if (g_nativeRingBuffer != nullptr) {
    g_nativeRingBuffer->reset();
  }
}

// Storage for last recorded audio (accessed by worker thread via extern)
float* g_lastRecordedAudio = nullptr;
size_t g_lastRecordedFrameCount = 0;
unsigned int g_lastRecordedChannels = 2;
unsigned int g_lastRecordedSampleRate = 48000;

// WAV in native memory (built on demand)
static uint8_t* g_recordedWavData = nullptr;
static size_t g_recordedWavSize = 0;

// Build WAV data in native memory from recorded float samples
static void buildRecordedWav() {
  if (g_recordedWavData != nullptr) {
    delete[] g_recordedWavData;
    g_recordedWavData = nullptr;
    g_recordedWavSize = 0;
  }

  if (g_lastRecordedAudio == nullptr || g_lastRecordedFrameCount == 0) {
    return;
  }

  const size_t numSamples = g_lastRecordedFrameCount * g_lastRecordedChannels;
  const size_t audioBytes = numSamples * sizeof(float);
  const size_t wavSize = 44 + audioBytes;  // WAV header is 44 bytes

  g_recordedWavData = new uint8_t[wavSize];
  g_recordedWavSize = wavSize;

  // Build WAV header (44 bytes, F32LE format)
  uint8_t* p = g_recordedWavData;

  // RIFF header
  memcpy(p, "RIFF", 4); p += 4;
  uint32_t chunkSize = (uint32_t)(wavSize - 8);
  memcpy(p, &chunkSize, 4); p += 4;
  memcpy(p, "WAVE", 4); p += 4;

  // fmt subchunk
  memcpy(p, "fmt ", 4); p += 4;
  uint32_t subchunk1Size = 16;
  memcpy(p, &subchunk1Size, 4); p += 4;
  uint16_t audioFormat = 3;  // IEEE float
  memcpy(p, &audioFormat, 2); p += 2;
  uint16_t numChannels = (uint16_t)g_lastRecordedChannels;
  memcpy(p, &numChannels, 2); p += 2;
  uint32_t sampleRate = g_lastRecordedSampleRate;
  memcpy(p, &sampleRate, 4); p += 4;
  uint32_t byteRate = sampleRate * numChannels * sizeof(float);
  memcpy(p, &byteRate, 4); p += 4;
  uint16_t blockAlign = (uint16_t)(numChannels * sizeof(float));
  memcpy(p, &blockAlign, 2); p += 2;
  uint16_t bitsPerSample = 32;
  memcpy(p, &bitsPerSample, 2); p += 2;

  // data subchunk
  memcpy(p, "data", 4); p += 4;
  uint32_t subchunk2Size = (uint32_t)audioBytes;
  memcpy(p, &subchunk2Size, 4); p += 4;

  // Audio data (float samples)
  memcpy(p, g_lastRecordedAudio, audioBytes);

  fprintf(stderr, "[Recorder] Built WAV in memory: %zu bytes (%zu frames @ %uHz, %u ch)\n",
          wavSize, g_lastRecordedFrameCount, g_lastRecordedSampleRate, g_lastRecordedChannels);
}

// Get recorded audio as WAV - returns pointer and size
// Builds WAV on first call after recording stops
// Pointer valid until next recording or freeRecordedAudio
FFI_PLUGIN_EXPORT const uint8_t* flutter_recorder_getRecordedWav(size_t* outSize) {
  if (g_recordedWavData == nullptr && g_lastRecordedAudio != nullptr) {
    buildRecordedWav();
  }
  if (outSize) {
    *outSize = g_recordedWavSize;
  }
  return g_recordedWavData;
}

// Get just the WAV size (for pre-allocating view)
FFI_PLUGIN_EXPORT size_t flutter_recorder_getRecordedWavSize() {
  if (g_recordedWavData == nullptr && g_lastRecordedAudio != nullptr) {
    buildRecordedWav();
  }
  return g_recordedWavSize;
}

// Free the recorded audio and WAV buffers
FFI_PLUGIN_EXPORT void flutter_recorder_freeRecordedAudio() {
  if (g_lastRecordedAudio != nullptr) {
    delete[] g_lastRecordedAudio;
    g_lastRecordedAudio = nullptr;
    g_lastRecordedFrameCount = 0;
  }
  if (g_recordedWavData != nullptr) {
    delete[] g_recordedWavData;
    g_recordedWavData = nullptr;
    g_recordedWavSize = 0;
  }
  fprintf(stderr, "[Recorder] Freed recorded audio buffers\n");
}

// Internal: Store recorded audio from native ring buffer stopRecording
// Called by native_scheduler.cpp when recording stops
void storeRecordedAudio(float* data, size_t frameCount) {
  // Free previous recording if any
  if (g_lastRecordedAudio != nullptr) {
    delete[] g_lastRecordedAudio;
  }
  if (g_recordedWavData != nullptr) {
    delete[] g_recordedWavData;
    g_recordedWavData = nullptr;
    g_recordedWavSize = 0;
  }

  g_lastRecordedAudio = data;
  g_lastRecordedFrameCount = frameCount;

  // Cache device config for WAV building
  if (g_nativeRingBuffer != nullptr) {
    g_lastRecordedChannels = g_nativeRingBuffer->channels();
    g_lastRecordedSampleRate = g_nativeRingBuffer->sampleRate();
  }

  size_t totalSamples = frameCount * g_lastRecordedChannels;
  fprintf(stderr, "[Recorder] Stored recorded audio: %zu frames, %zu totalSamples @ %uHz, %u ch\n",
          frameCount, totalSamples, g_lastRecordedSampleRate, g_lastRecordedChannels);

  // Debug: Print first few samples to verify stereo interleaving
  if (data != nullptr && frameCount > 10) {
    fprintf(stderr, "[Recorder] First 8 samples (L R L R...): ");
    for (int i = 0; i < 8 && i < (int)totalSamples; i++) {
      fprintf(stderr, "%.4f ", data[i]);
    }
    fprintf(stderr, "\n");
  }

  // LOOPER BRIDGE: Signal worker thread to build WAV and start playback
  // IMPORTANT: Audio thread does NO allocations, NO locks, NO syscalls
  // Just set atomic flag - worker thread does the heavy lifting
  if (g_looperBridge != nullptr && frameCount > 0) {
    fprintf(stderr, "[Recorder] Signaling looper worker thread\n");
    g_looperWorkReady.store(true, std::memory_order_release);
  }
}
