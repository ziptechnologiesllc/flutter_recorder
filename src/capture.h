#ifndef CAPTURE_H
#define CAPTURE_H

#include "common.h"
#include "enums.h"
#include "miniaudio.h"
#include "wav.h"

#include "filters/filters.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct CaptureDevice {
  char *name;
  unsigned int isDefault;
  unsigned int id;
};

// ── Audio-callback profiling (pops/clicks hunt) ───────────────────────────
// Lock-free read of the data_callback timing stats. `out` must point at an
// array of at least 6 int64: [lastMicros, maxMicros, budgetMicros,
// overrunCount, nearMissCount, totalCount]. captureResetCallbackStats()
// zeroes the max + counters (keeps last/budget). See capture.cpp for the
// definition of "overrun".
void captureGetCallbackStats(int64_t *out);
void captureResetCallbackStats();

class Capture {
public:
  Capture();
  ~Capture();

  /// stores a list of available capture devices
  /// detected by miniaudio
  std::vector<CaptureDevice> listCaptureDevices();

  /// @brief initialize the capture with a [deviceID]. A list of devices
  ///     can be acquired with [listCaptureDevices].
  ///     If [deviceID] is -1, the default will be used
  /// @param filters the filters
  /// @param deviceID the device ID chosen to be initialized
  /// @param pcmFormat the PCM format
  /// @param sampleRate the sample rate
  /// @param channels the number of channels
  /// @param androidInputPreset Android input preset. 0 leaves it unset.
  /// @param captureOnly if true, use capture-only mode (no playback).
  ///        Use this when SoLoud has its own playback device.
  ///        If false, use duplex mode for slave mode where the recorder
  ///        drives SoLoud's output inside its own callback (one clock).
  /// @return `captureNoError` if no error or else `captureInitFailed`
  CaptureErrors init(Filters *filters, int deviceID, PCMFormat pcmFormat,
                     unsigned int sampleRate, unsigned int channels,
                     int androidInputPreset, bool captureOnly = false);

  /// @brief Must be called when there is no more need of the capture or when
  /// closing the app
  void dispose();

  bool isInited();

  bool isDeviceStarted();

  /// Returns true if duplex mode was requested but capture couldn't get
  /// exclusive mode. Dart should fall back to standard (non-slave) SoLoud
  /// when this is true.
  bool wasDuplexDenied() const { return mDuplexDenied; }

  CaptureErrors start();

  void stop();

  // Getters for actual device parameters (populated after init)
  unsigned int getSampleRate() const { return device.sampleRate; }
  unsigned int getCaptureChannels() const { return device.capture.channels; }
  unsigned int getPlaybackChannels() const { return device.playback.channels; }
  int getCaptureFormat() const { return (int)device.capture.format; }
  int getPlaybackFormat() const { return (int)device.playback.format; }

  void startStreamingData();
  void stopStreamingData();

  void setSilenceDetection(bool enable);

  void setSilenceThresholdDb(float silenceThresholdDb);
  void setSilenceDuration(float silenceDuration);
  void setSecondsOfAudioToWriteBefore(float secondsOfAudioToWriteBefore);

  CaptureErrors startRecording(const char *path);

  void setPauseRecording(bool pause);

  void stopRecording();

  /// Write preroll data directly to the WAV file (for latency compensation)
  /// @param samples Float samples (interleaved if stereo)
  /// @param numSamples Total number of samples (frames * channels)
  void writePrerollToWav(const float *samples, size_t numSamples);

  float *getWave(bool *isTheSameAsBefore);

  float getVolumeDb();

  ma_device_config deviceConfig;

  /// Wheter or not the callback is detecting silence.
  bool isDetectingSilence;

  /// The threshold for detecting silence.
  float silenceThresholdDb;

  /// The duration of silence in seconds after which the silence is considered
  /// silence.
  float silenceDuration;

  /// ms of audio to write occurred before starting recording againg after
  /// silence.
  float secondsOfAudioToWriteBefore;

  ///
  WriteAudio::Wav wav;

  /// true when the capture device is recording.
  bool isRecording;

  /// true when the capture device is paused.
  bool isRecordingPaused;

  /// true when the capture device is streaming data.
  bool isStreamingData;

  /// true when monitoring (input passthrough to output) is enabled.
  bool monitoringEnabled;

  /// Monitoring mode: 0=stereo, 1=leftMono, 2=rightMono, 3=mono
  int monitoringMode;

  /// the number of bytes per sample
  int bytesPerSample;

  Filters *mFilters = nullptr; // Initialize to null for thread-safety

  /// @brief Start capturing samples for AEC calibration
  /// @param maxSamples Maximum number of mono samples to capture
  void startCalibrationCapture(size_t maxSamples);

  /// @brief Stop capturing samples for calibration
  void stopCalibrationCapture();

  /// @brief Read captured calibration samples
  /// @param dest Destination buffer for mono samples
  /// @param maxSamples Maximum number of samples to read
  /// @return Number of samples actually read
  size_t readCalibrationSamples(float *dest, size_t maxSamples);

  /// @brief Check if calibration capture is active
  bool isCalibrationCaptureActive() const;

  /// @brief Get total frames captured since device started
  /// This counter is used for sample-accurate AEC synchronization
  size_t getTotalFramesCaptured() const {
    return mTotalFramesCaptured.load(std::memory_order_acquire);
  }

  /// @brief Reset the frame counter (call before calibration)
  void resetFrameCounter() {
    mTotalFramesCaptured.store(0, std::memory_order_release);
  }

  /// Calibration capture buffer and state (public for data callback access).
  /// Lock-free on the RT thread: the callback only reads mCalibrationActive
  /// (acquire) and advances mCalibrationWritePos (release); the buffer is
  /// (re)allocated exclusively on the API thread BEFORE active is set, so the
  /// callback never observes a resizing vector. mCalibrationMutex serializes
  /// API-thread callers only — the audio callback must never take it.
  std::vector<float> mCalibrationBuffer;
  std::atomic<size_t> mCalibrationWritePos{0};
  std::atomic<bool> mCalibrationActive{false};
  std::mutex mCalibrationMutex;

  /// Total frames captured since device started (for AEC sync)
  /// Atomic for lock-free access from data callback
  std::atomic<size_t> mTotalFramesCaptured{0};

  /// Buffer for format conversion (e.g. S16 -> F32) used in data_callback
  /// Defined here to avoid reallocating in the audio thread
  std::vector<float> mConversionBuffer;

  /// Buffer for playback format conversion (F32 -> S16)
  /// SoLoud outputs f32, but device may need s16 for low-latency fast path
  /// All processing (AEC, monitoring) happens in f32, then converted at the end
  std::vector<float> mPlaybackBuffer;

private:
  /// @brief Create the explicit miniaudio context if it does not exist yet.
  /// The one context-lifetime mechanism: every platform gets an explicit
  /// context up-front (Windows: WASAPI priority with auto fallback; Apple:
  /// AVAudioSession-neutral CoreAudio config; others: auto backend). No-op
  /// when `mContextInited` is already true.
  /// @return `captureNoError` on success or reuse, `captureInitFailed`
  /// otherwise.
  CaptureErrors ensureContext();

  ma_context context;
  ma_device_info *pPlaybackInfos;
  ma_uint32 playbackCount;
  ma_device_info *pCaptureInfos;
  ma_uint32 captureCount;
  ma_result result;
  ma_device device;

  /// true when the capture device is initialized.
  bool mInited;

  /// true when duplex was requested but capture fell to shared mode
  /// (exclusive denied by HAL). Dart should use standard SoLoud instead of
  /// slave.
  bool mDuplexDenied = false;

  /// true while `context` is a live, explicitly created miniaudio context.
  /// Replaces upstream's `mUsesContext`: on the merged code every platform
  /// creates an explicit context up-front (see ensureContext). Windows keeps
  /// the context alive across init/dispose cycles (WASAPI re-init deadlock)
  /// and releases it in the destructor.
  bool mContextInited;
};

#endif // CAPTURE_H
