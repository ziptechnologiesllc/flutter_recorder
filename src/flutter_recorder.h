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
                      unsigned int channels);

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

/////////////////////////
/// MONITORING
/////////////////////////
FFI_PLUGIN_EXPORT void flutter_recorder_setMonitoring(bool enabled);
FFI_PLUGIN_EXPORT void flutter_recorder_setMonitoringMode(int mode);

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

/////////////////////////
/// AEC (Acoustic Echo Cancellation)
/////////////////////////
FFI_PLUGIN_EXPORT void *
flutter_recorder_aec_createReferenceBuffer(unsigned int sampleRate,
                                           unsigned int channels);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_destroyReferenceBuffer();
FFI_PLUGIN_EXPORT void *flutter_recorder_aec_getOutputCallback();
FFI_PLUGIN_EXPORT void flutter_recorder_aec_resetBuffer();

/////////////////////////
/// AEC Calibration
/////////////////////////
FFI_PLUGIN_EXPORT uint8_t *flutter_recorder_aec_generateCalibrationSignal(
    unsigned int sampleRate, unsigned int channels, size_t *outSize);
FFI_PLUGIN_EXPORT void
flutter_recorder_aec_startCalibrationCapture(size_t maxSamples);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopCalibrationCapture();
FFI_PLUGIN_EXPORT void flutter_recorder_aec_captureForAnalysis();
FFI_PLUGIN_EXPORT int
flutter_recorder_aec_runCalibrationAnalysis(unsigned int sampleRate,
                                            float *outDelayMs, float *outEchoGain,
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
    unsigned int sampleRate,
    float *outCancellationDb,
    float *outCorrelationBefore,
    float *outCorrelationAfter,
    int *outPassed,
    float *outMicEnergyDb,
    float *outCancelledEnergyDb);

// Get captured test signals for visualization
FFI_PLUGIN_EXPORT int flutter_recorder_aec_getTestMicSignal(float *dest, int maxLength);
FFI_PLUGIN_EXPORT int flutter_recorder_aec_getTestCancelledSignal(float *dest, int maxLength);

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
FFI_PLUGIN_EXPORT void flutter_recorder_aec_startAlignedCalibrationCapture(size_t maxSamples);
FFI_PLUGIN_EXPORT void flutter_recorder_aec_stopAlignedCalibrationCapture();
FFI_PLUGIN_EXPORT int flutter_recorder_aec_runAlignedCalibrationWithImpulse(
    unsigned int sampleRate,
    int *outDelaySamples,
    float *outDelayMs,
    float *outGain,
    float *outCorrelation,
    int *outImpulseLength,
    int64_t *outCalibratedOffset);

#ifdef __cplusplus
}
#endif

#endif // FLUTTER_RECORDER_H