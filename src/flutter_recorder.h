#ifndef FLUTTER_RECORDER_H
#define FLUTTER_RECORDER_H

#include "common.h"
#include "enums.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FFI_PLUGIN_EXPORT void flutter_recorder_createWorkerInWasm();

FFI_PLUGIN_EXPORT void flutter_recorder_setDartEventCallback(
    dartSilenceChangedCallback_t silence_changed_callback,
    dartStreamDataCallback_t stream_data_callback);

// Set callback for when recording stops (auto-stop at loop boundary)
FFI_PLUGIN_EXPORT void flutter_recorder_setRecordingStoppedCallback(
    dartRecordingStoppedCallback_t callback);

// Set callback for when recording starts (native scheduler fires StartRecording)
FFI_PLUGIN_EXPORT void flutter_recorder_setRecordingStartedCallback(
    dartRecordingStartedCallback_t callback);

// Set callback for when looper playback starts (from worker thread)
FFI_PLUGIN_EXPORT void flutter_recorder_setLooperPlaybackStartedCallback(
    dartLooperPlaybackStartedCallback_t callback);

FFI_PLUGIN_EXPORT void flutter_recorder_nativeFree(void *pointer);

FFI_PLUGIN_EXPORT void flutter_recorder_listCaptureDevices(char **devicesName,
                                                           int **deviceId,
                                                           int **isDefault,
                                                           int *n_devices);

FFI_PLUGIN_EXPORT void
flutter_recorder_freeListCaptureDevices(char **devicesName, int **deviceId,
                                        int **isDefault, int n_devices);

FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_init(int deviceID, int pcmFormat, unsigned int sampleRate,
                      unsigned int channels, int androidInputPreset,
                      int captureOnly);

FFI_PLUGIN_EXPORT void flutter_recorder_deinit();

FFI_PLUGIN_EXPORT int flutter_recorder_isInited();

FFI_PLUGIN_EXPORT int flutter_recorder_isDeviceStarted();

FFI_PLUGIN_EXPORT enum CaptureErrors flutter_recorder_start();

FFI_PLUGIN_EXPORT void flutter_recorder_stop();

FFI_PLUGIN_EXPORT void flutter_recorder_startStreamingData();

FFI_PLUGIN_EXPORT void flutter_recorder_stopStreamingData();

FFI_PLUGIN_EXPORT void flutter_recorder_setSilenceDetection(bool enable);

FFI_PLUGIN_EXPORT void
flutter_recorder_setSilenceThresholdDb(float silenceThresholdDb);

FFI_PLUGIN_EXPORT void
flutter_recorder_setSilenceDuration(float silenceDuration);

FFI_PLUGIN_EXPORT void flutter_recorder_setSecondsOfAudioToWriteBefore(
    float secondsOfAudioToWriteBefore);

FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_startRecording(const char *path);

FFI_PLUGIN_EXPORT void flutter_recorder_setPauseRecording(bool pause);

FFI_PLUGIN_EXPORT void flutter_recorder_stopRecording();

FFI_PLUGIN_EXPORT void flutter_recorder_getVolumeDb(float *volumeDb);

FFI_PLUGIN_EXPORT void flutter_recorder_getFft(float **fft,
                                               bool *isTheSameAsBefore);

FFI_PLUGIN_EXPORT void flutter_recorder_getWave(float **wave,
                                                bool *isTheSameAsBefore);

FFI_PLUGIN_EXPORT void flutter_recorder_getTexture(float **samples,
                                                   bool *isTheSameAsBefore);

FFI_PLUGIN_EXPORT void flutter_recorder_getTexture2D(float **samples,
                                                     bool *isTheSameAsBefore);

FFI_PLUGIN_EXPORT float flutter_recorder_getTextureValue(int row, int column);

FFI_PLUGIN_EXPORT void flutter_recorder_setFftSmoothing(float smooth);

// Getters for actual device parameters (populated after init)
FFI_PLUGIN_EXPORT unsigned int flutter_recorder_getSampleRate();
FFI_PLUGIN_EXPORT unsigned int flutter_recorder_getCaptureChannels();
FFI_PLUGIN_EXPORT unsigned int flutter_recorder_getPlaybackChannels();
FFI_PLUGIN_EXPORT int flutter_recorder_getCaptureFormat();
FFI_PLUGIN_EXPORT int flutter_recorder_getPlaybackFormat();

/////////////////////////
/// MONITORING
/////////////////////////
FFI_PLUGIN_EXPORT void flutter_recorder_setMonitoring(bool enabled);
FFI_PLUGIN_EXPORT void flutter_recorder_setMonitoringMode(int mode);

/////////////////////////
/// SLAVE MODE
/////////////////////////
// Check if slave audio is ready (first callback has run successfully)
// Used to wait for the audio pipeline to stabilize before calibration
FFI_PLUGIN_EXPORT int flutter_recorder_isSlaveAudioReady();

/////////////////////////
/// NATIVE AUDIO SINK
/////////////////////////
// Direct native-to-native streaming (bypasses Dart main thread)
// Callback type matches flutter_soloud's expected signature
typedef void (*NativeAudioSinkCallback)(const unsigned char* data,
                                        unsigned int dataLen, void* userData);

// Set native audio sink for direct recorder-to-player streaming
FFI_PLUGIN_EXPORT void flutter_recorder_setNativeAudioSink(
    NativeAudioSinkCallback callback, void* userData);

// Check if native audio sink is active
FFI_PLUGIN_EXPORT bool flutter_recorder_isNativeAudioSinkActive();

// Disable native audio sink
FFI_PLUGIN_EXPORT void flutter_recorder_disableNativeAudioSink();

// Inject preroll audio from ring buffer into SoLoud stream via native path
FFI_PLUGIN_EXPORT void flutter_recorder_injectPreroll(size_t frameCount);

/////////////////////////
/// LOOPER BRIDGE (native-to-SoLoud direct playback)
/////////////////////////
// Function pointer type matching looper_loadAndPlayRaw from flutter_soloud
// Takes raw PCM float samples directly - no WAV container overhead
typedef unsigned int (*LooperLoadAndPlayRawFunc)(float* samples,
                                                  unsigned int numSamples,
                                                  float sampleRate,
                                                  unsigned int channels,
                                                  bool copy,
                                                  bool takeOwnership,
                                                  unsigned int* outHandle);

// Set the looper function pointer (called from Dart during init)
FFI_PLUGIN_EXPORT void flutter_recorder_setLooperBridge(LooperLoadAndPlayRawFunc func);

// Clear the looper bridge
FFI_PLUGIN_EXPORT void flutter_recorder_clearLooperBridge();

/////////////////////////
/// FILTERS
/////////////////////////
FFI_PLUGIN_EXPORT int
flutter_recorder_isFilterActive(enum RecorderFilterType filterType);
FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_addFilter(enum RecorderFilterType filterType);
FFI_PLUGIN_EXPORT enum CaptureErrors
flutter_recorder_removeFilter(enum RecorderFilterType filterType);
FFI_PLUGIN_EXPORT void
flutter_recorder_getFilterParamNames(enum RecorderFilterType filterType,
                                     char **names, int *paramsCount);
FFI_PLUGIN_EXPORT void
flutter_recorder_setFilterParams(enum RecorderFilterType filterType,
                                 int attributeId, float value);
FFI_PLUGIN_EXPORT float
flutter_recorder_getFilterParams(enum RecorderFilterType filterType,
                                 int attributeId);

// Filter lock stats (for debug overlay)
FFI_PLUGIN_EXPORT uint64_t flutter_recorder_getFilterMissCount();
FFI_PLUGIN_EXPORT uint64_t flutter_recorder_getFilterProcessCount();
FFI_PLUGIN_EXPORT void flutter_recorder_resetFilterStats();

/////////////////////////
/// AEC (Acoustic Echo Cancellation)
/////////////////////////
FFI_PLUGIN_EXPORT void *
flutter_recorder_aec_createReferenceBuffer(unsigned int sampleRate,
                                           unsigned int channels);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_destroyReferenceBuffer();
FFI_PLUGIN_EXPORT void *flutter_recorder_aec_getOutputCallback();
FFI_PLUGIN_EXPORT void flutter_recorder_aec_resetBuffer();

// AEC Enable/Disable (controls reference buffer writes)
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setEnabled(bool enabled);
FFI_PLUGIN_EXPORT bool flutter_recorder_aec_isEnabled();

// AEC Mode Control (A/B Testing)
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setMode(int mode);
FFI_PLUGIN_EXPORT int flutter_recorder_aec_getMode();

/////////////////////////
/// Neural Model Control
/////////////////////////
// Load neural model by type
// modelType: 0=NONE, 1=AEC_MASK_V3
// assetBasePath: Platform-specific path to assets directory
// Returns: 1 if successful, 0 if failed
FFI_PLUGIN_EXPORT int flutter_recorder_neural_loadModel(int modelType,
                                                        const char *assetBasePath);

// Get currently loaded neural model type
// Returns: 0=NONE, 1=AEC_MASK_V3
FFI_PLUGIN_EXPORT int flutter_recorder_neural_getLoadedModel();

// Enable/disable neural post-filter
FFI_PLUGIN_EXPORT void flutter_recorder_neural_setEnabled(int enabled);

// Check if neural post-filter is enabled
FFI_PLUGIN_EXPORT int flutter_recorder_neural_isEnabled();

/////////////////////////
/// AEC Calibration
/////////////////////////

// Calibration signal types
// 0 = Chirp (log sweep), 1 = Click (impulse train)
FFI_PLUGIN_EXPORT uint8_t *flutter_recorder_aec_generateCalibrationSignal(
    unsigned int sampleRate, unsigned int channels, size_t *outSize,
    int signalType);
FFI_PLUGIN_EXPORT void
flutter_recorder_aec_startCalibrationCapture(size_t maxSamples);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopCalibrationCapture();
FFI_PLUGIN_EXPORT void flutter_recorder_aec_captureForAnalysis();
FFI_PLUGIN_EXPORT int flutter_recorder_aec_runCalibrationAnalysis(
    unsigned int sampleRate, float *outDelayMs, float *outEchoGain,
    float *outCorrelation);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_resetCalibration();

// Run calibration analysis with impulse response computation
// AecStats defined in enums.h

typedef void (*AecStatsCallback)(AecStats stats);

FFI_PLUGIN_EXPORT void
flutter_recorder_set_aec_stats_callback(AecStatsCallback callback);

FFI_PLUGIN_EXPORT int flutter_recorder_aec_runCalibrationWithImpulse(
    unsigned int sampleRate, float *outDelayMs, float *outEchoGain,
    float *outCorrelation, int *outImpulseLength, int64_t *outCalibratedOffset);

// Get stored impulse response from last calibration
FFI_PLUGIN_EXPORT int flutter_recorder_aec_getImpulseResponse(float *dest,
                                                              int maxLength);

// Apply stored impulse response to AEC filter
FFI_PLUGIN_EXPORT void flutter_recorder_aec_applyImpulseResponse();

// Get captured calibration signals for visualization
FFI_PLUGIN_EXPORT int
flutter_recorder_aec_getCalibrationRefSignal(float *dest, int maxLength);
FFI_PLUGIN_EXPORT int
flutter_recorder_aec_getCalibrationMicSignal(float *dest, int maxLength);

// Set AEC delay from calibration result
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setDelay(float delayMs);

// Apply full calibration result: delay + impulse response
FFI_PLUGIN_EXPORT void flutter_recorder_aec_applyCalibration(float delayMs);

/////////////////////////
/// AEC Testing
/////////////////////////

// Start capturing test data (raw mic + cancelled output)
FFI_PLUGIN_EXPORT void flutter_recorder_aec_startTestCapture(size_t maxSamples);

/////////////////////////
/// AEC Calibration Logging
/////////////////////////

// Get calibration log buffer (returns pointer to internal string, do not free)
FFI_PLUGIN_EXPORT const char *flutter_recorder_aec_getCalibrationLog();

// Clear calibration log buffer
FFI_PLUGIN_EXPORT void flutter_recorder_aec_clearCalibrationLog();

// Stop capturing test data
FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopTestCapture();

// Run test analysis and return metrics
FFI_PLUGIN_EXPORT int flutter_recorder_aec_runTest(
    unsigned int sampleRate, float *outCancellationDb,
    float *outCorrelationBefore, float *outCorrelationAfter, int *outPassed,
    float *outMicEnergyDb, float *outCancelledEnergyDb);

// Get captured test signals for visualization
FFI_PLUGIN_EXPORT int flutter_recorder_aec_getTestMicSignal(float *dest,
                                                            int maxLength);
FFI_PLUGIN_EXPORT int
flutter_recorder_aec_getTestCancelledSignal(float *dest, int maxLength);

// Reset test data
FFI_PLUGIN_EXPORT void flutter_recorder_aec_resetTest();

// VSS-NLMS parameter control for experimentation
// mu_max: Maximum step size (0.0-1.0). Set to 0 to freeze weights.
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setVssMuMax(float mu);
// leakage: Weight decay factor (0.99-1.0). Set to 1.0 for no decay.
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setVssLeakage(float lambda);
// alpha: Smoothing factor for VSS statistics (0.9-0.999).
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setVssAlpha(float alpha);
// Getters for current values
FFI_PLUGIN_EXPORT float flutter_recorder_aec_getVssMuMax();
FFI_PLUGIN_EXPORT float flutter_recorder_aec_getVssLeakage();
FFI_PLUGIN_EXPORT float flutter_recorder_aec_getVssAlpha();

// Filter length control (2048, 4096, 8192 recommended)
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setFilterLength(int length);
FFI_PLUGIN_EXPORT int flutter_recorder_aec_getFilterLength();

// Position-based sync for sample-accurate AEC
// Get total frames written to reference buffer (output side)
FFI_PLUGIN_EXPORT size_t flutter_recorder_aec_getOutputFrameCount();
// Get total frames captured by recorder (input side)
FFI_PLUGIN_EXPORT size_t flutter_recorder_aec_getCaptureFrameCount();
// Record frame counters at calibration start
FFI_PLUGIN_EXPORT void flutter_recorder_aec_recordCalibrationFrameCounters();
// Set calibrated offset: captureFrame - offset = outputFrame
FFI_PLUGIN_EXPORT void flutter_recorder_aec_setCalibratedOffset(int64_t offset);
// Get current calibrated offset
FFI_PLUGIN_EXPORT int64_t flutter_recorder_aec_getCalibratedOffset();

/////////////////////////
/// iOS Hardware Control
/////////////////////////
FFI_PLUGIN_EXPORT void flutter_recorder_ios_force_speaker_output(bool enabled);

/////////////////////////
/// Aligned Calibration Capture (frame-aligned signals from processAudio)
/////////////////////////
FFI_PLUGIN_EXPORT void
flutter_recorder_aec_startAlignedCalibrationCapture(size_t maxSamples);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopAlignedCalibrationCapture();
FFI_PLUGIN_EXPORT int flutter_recorder_aec_runAlignedCalibrationWithImpulse(
    unsigned int sampleRate, int *outDelaySamples, float *outDelayMs,
    float *outGain, float *outCorrelation, int *outImpulseLength,
    int64_t *outCalibratedOffset, int signalType);

/////////////////////////
/// Native Scheduler
/// Sample-accurate timing for recording start/stop in audio callback
/////////////////////////

// Reset the native scheduler state
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_reset();

// Set base loop parameters for quantization
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_setBaseLoop(int64_t loopFrames,
                                                               int64_t loopStartFrame);

// Clear base loop (free recording mode)
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_clearBaseLoop();

// Schedule quantized recording start
// Returns event ID (0 if failed)
FFI_PLUGIN_EXPORT uint32_t flutter_recorder_scheduler_scheduleStart(const char* path);

// Schedule quantized recording stop
// Returns event ID (0 if failed)
FFI_PLUGIN_EXPORT uint32_t flutter_recorder_scheduler_scheduleStop(int64_t startFrame);

// Schedule event at specific frame
// action: 0=None, 1=StartRecording, 2=StopRecording, 3=StartPlayback, 4=StopPlayback
FFI_PLUGIN_EXPORT uint32_t flutter_recorder_scheduler_scheduleEvent(
    int action, int64_t targetFrame, const char* path);

// Cancel a scheduled event by ID
// Returns 1 if cancelled, 0 if not found
FFI_PLUGIN_EXPORT int flutter_recorder_scheduler_cancelEvent(uint32_t eventId);

// Cancel all pending events
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_cancelAll();

// Poll for fired event notification
// Returns 1 if notification available, 0 if queue empty
FFI_PLUGIN_EXPORT int flutter_recorder_scheduler_pollNotification(
    uint32_t* outEventId, int* outAction, int64_t* outFiredFrame, int32_t* outLatency);

// Check if there are pending notifications
FFI_PLUGIN_EXPORT int flutter_recorder_scheduler_hasNotifications();

// Get current global frame position
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getGlobalFrame();

// Get base loop length in frames
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getBaseLoopFrames();

// Get next loop boundary frame
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getNextLoopBoundary();

// Set latency compensation in frames (applied at recording start)
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_setLatencyCompensation(int64_t frames);

// Get latency compensation in frames
FFI_PLUGIN_EXPORT int64_t flutter_recorder_scheduler_getLatencyCompensation();

// Set auto-stop enabled (when true, STOP is scheduled upfront with START)
FFI_PLUGIN_EXPORT void flutter_recorder_scheduler_setAutoStop(bool enabled);

// Get auto-stop enabled state
FFI_PLUGIN_EXPORT bool flutter_recorder_scheduler_isAutoStopEnabled();

/////////////////////////
/// Auto-Record (hands-free first-loop capture)
/// Long-press to arm; the first detected onset becomes the loop downbeat (the
/// lead-in silence is trimmed by rewinding the ring buffer). With a preset bar
/// count + a known tempo (framesPerBar) the take auto-stops on the bar.
/////////////////////////

// Arm auto-record.
//   wavPath        : where the take is written on stop
//   barCount       : preset phrase length in bars; <= 0 = no preset length
//   framesPerBar   : tempo in frames per bar (one bar = 4 beats, 4/4); > 0
//                    enables the preset auto-stop; 0 = tempo unknown
//   sampleRate     : capture sample rate (0 = keep current)
//   measureAmbient : if != 0, keep measuring the ambient level (don't listen for
//                    onsets) until flutter_recorder_endAutoRecordMeasure() — the
//                    "hold the button" model: the longer you hold, the better
//                    the ambient estimate, hence the trigger threshold.
FFI_PLUGIN_EXPORT void flutter_recorder_armAutoRecord(const char* wavPath,
                                                      int barCount,
                                                      int64_t framesPerBar,
                                                      unsigned int sampleRate,
                                                      int measureAmbient);

// End the ambient-measure window (held button released): lock the trigger to the
// measured ambient level and start listening for onsets.
FFI_PLUGIN_EXPORT void flutter_recorder_endAutoRecordMeasure();

// Disarm auto-record (an in-progress take is left for the normal stop path).
FFI_PLUGIN_EXPORT void flutter_recorder_disarmAutoRecord();

// 0 = idle, 1 = armed (waiting for onset, or still measuring ambient), 2 = recording
FFI_PLUGIN_EXPORT int flutter_recorder_getAutoRecordState();

// 1 while the armed detector is still measuring ambient (button held), else 0.
FFI_PLUGIN_EXPORT int flutter_recorder_isAutoRecordMeasuringAmbient();

// Best current tempo estimate in BPM (0 until Phase 2's estimator locks).
FFI_PLUGIN_EXPORT float flutter_recorder_getAutoRecordTempoBpm();

// Current measured noise floor in dBFS (for the UI threshold line).
FFI_PLUGIN_EXPORT float flutter_recorder_getAutoRecordNoiseFloorDb();

// Current onset trigger level in dBFS = noiseFloorDb + onsetThresholdDb.
FFI_PLUGIN_EXPORT float flutter_recorder_getAutoRecordTriggerLevelDb();

// Onset-detector sensitivity: dB above the (ambient) noise floor that counts as
// an attack. Lower = more sensitive (soft-onset instruments). Default ~12 dB.
FFI_PLUGIN_EXPORT void flutter_recorder_setAutoRecordOnsetThresholdDb(float db);

/////////////////////////
/// Native Ring Buffer
/// Latency compensation via continuous capture with pre-roll
/////////////////////////

// Create/configure the native ring buffer
// capacitySeconds: How many seconds of audio to keep (typically 5)
// sampleRate: Sample rate in Hz
// channels: Number of channels (1=mono, 2=stereo)
FFI_PLUGIN_EXPORT void flutter_recorder_createRingBuffer(
    size_t capacitySeconds, unsigned int sampleRate, unsigned int channels);

// Destroy/reset the native ring buffer
FFI_PLUGIN_EXPORT void flutter_recorder_destroyRingBuffer();

// Read pre-roll samples for latency compensation
// dest: Destination buffer (must be pre-allocated)
// frameCount: Number of frames to read
// rewindFrames: How many frames back in time to start reading
// Returns: Number of frames actually read
FFI_PLUGIN_EXPORT size_t flutter_recorder_readPreRoll(
    float* dest, size_t frameCount, size_t rewindFrames);

// Get current audio level in dB (RMS)
FFI_PLUGIN_EXPORT float flutter_recorder_getAudioLevelDb();

// Get total frames written to the ring buffer
FFI_PLUGIN_EXPORT size_t flutter_recorder_getRingBufferFramesWritten();

// Get available frames in the ring buffer
FFI_PLUGIN_EXPORT size_t flutter_recorder_getRingBufferAvailable();

// Phase 3c — live monophonic pitch estimate (chromatic instrument tuner) from
// the most recent slice of the capture ring buffer. Writes:
//   *outFrequencyHz : detected fundamental in Hz; 0 = no clear pitch
//   *outClarity     : 0..1; YIN confidence the pitch is real — gate the
//                     display on this so a strummed chord (no single
//                     fundamental) reads as "no pitch", not garbage.
// `maxAnalyzeFrames` bounds the analysis window (0 = ~2048 frames, ≈ 40-50 ms);
// it's clamped up to a sane minimum so the detector isn't starved. Cheap
// enough to poll at ~10 Hz; lower the poll rate on weak hardware. Safe to
// call from the UI thread.
FFI_PLUGIN_EXPORT void flutter_recorder_estimatePitch(
    unsigned int maxAnalyzeFrames, float* outFrequencyHz, float* outClarity);

// Reset the ring buffer (clear all data)
FFI_PLUGIN_EXPORT void flutter_recorder_resetRingBuffer();

// Get recorded audio as WAV data in native memory
// Returns pointer to WAV data (header + samples) - builds on first call
// Pointer valid until next recording or freeRecordedAudio
FFI_PLUGIN_EXPORT const uint8_t* flutter_recorder_getRecordedWav(size_t* outSize);

// Get WAV size without building (for checking if data available)
FFI_PLUGIN_EXPORT size_t flutter_recorder_getRecordedWavSize();

// Free the recorded audio and WAV buffers
FFI_PLUGIN_EXPORT void flutter_recorder_freeRecordedAudio();

FFI_PLUGIN_EXPORT int flutter_recorder_wasDuplexDenied();

/////////////////////////
// PHASE 2e: Ableton Link control
/////////////////////////
//
// These call into AudioEngine::linkClock() directly. `link_setEnabled` is
// NOT realtime-safe (Link's enable() spins up its network thread) — Dart
// must call from the main thread, never from the audio callback. The
// `numPeers` / `isEnabled` readers are wait-free.

// Enable / disable Ableton Link participation. NOT realtime-safe.
FFI_PLUGIN_EXPORT void flutter_recorder_link_setEnabled(int enabled);

// Wait-free reads. Safe from any thread.
FFI_PLUGIN_EXPORT int flutter_recorder_link_isEnabled();
FFI_PLUGIN_EXPORT uint32_t flutter_recorder_link_numPeers();

/////////////////////////
// Audio-callback profiling (pops/clicks hunt)
/////////////////////////
//
// Fills `out[0..5]` with the data_callback timing stats:
//   [0] lastMicros    — duration of the most recent callback
//   [1] maxMicros     — worst duration since the last reset
//   [2] budgetMicros  — buffer period; a callback over this = underrun
//   [3] overrunCount  — callbacks that exceeded budget
//   [4] nearMissCount — callbacks over 80% of budget
//   [5] totalCount    — callbacks measured since reset
// `out` must hold at least 6 int64_t. Wait-free; safe from any thread.
FFI_PLUGIN_EXPORT void flutter_recorder_getCallbackStats(int64_t *out);
FFI_PLUGIN_EXPORT void flutter_recorder_resetCallbackStats();

#ifdef __cplusplus
}
#endif

#endif // FLUTTER_RECORDER_H