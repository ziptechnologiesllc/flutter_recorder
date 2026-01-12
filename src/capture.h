#ifndef CAPTURE_H
#define CAPTURE_H

#include "enums.h"
#include "common.h"
#include "wav.h"
#include "miniaudio.h"

#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include "filters/filters.h"

struct CaptureDevice
{
    char *name;
    unsigned int isDefault;
    unsigned int id;
};

class Capture
{
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
    /// @param captureOnly if true, use capture-only mode (no playback).
    ///        Use this when SoLoud has its own playback device.
    ///        If false, use duplex mode for slave mode where recorder drives output.
    /// @return `captureNoError` if no error or else `captureInitFailed`
    CaptureErrors init(
        Filters *filters,
        int deviceID,
        PCMFormat pcmFormat,
        unsigned int sampleRate,
        unsigned int channels,
        bool captureOnly = false);

    /// @brief Must be called when there is no more need of the capture or when closing the app
    void dispose();

    bool isInited();

    bool isDeviceStarted();

    CaptureErrors start();

    void stop();

    void startStreamingData();
    void stopStreamingData();

    void setSilenceDetection(bool enable);

    void setSilenceThresholdDb(float silenceThresholdDb);
    void setSilenceDuration(float silenceDuration);
    void setSecondsOfAudioToWriteBefore(float secondsOfAudioToWriteBefore);

    CaptureErrors startRecording(const char *path);

    void setPauseRecording(bool pause);

    void stopRecording();

    float *getWave(bool *isTheSameAsBefore);

    float getVolumeDb();

    ma_device_config deviceConfig;

    /// Wheter or not the callback is detecting silence.
    bool isDetectingSilence;

    /// The threshold for detecting silence.
    float silenceThresholdDb;

    /// The duration of silence in seconds after which the silence is considered silence.
    float silenceDuration;

    /// ms of audio to write occurred before starting recording againg after silence.
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

    Filters *mFilters;

    /// @brief Start capturing samples for AEC calibration
    /// @param maxSamples Maximum number of mono samples to capture
    void startCalibrationCapture(size_t maxSamples);

    /// @brief Stop capturing samples for calibration
    void stopCalibrationCapture();

    /// @brief Read captured calibration samples
    /// @param dest Destination buffer for mono samples
    /// @param maxSamples Maximum number of samples to read
    /// @return Number of samples actually read
    size_t readCalibrationSamples(float* dest, size_t maxSamples);

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

    /// Calibration capture buffer and state (public for data callback access)
    std::vector<float> mCalibrationBuffer;
    size_t mCalibrationWritePos;
    bool mCalibrationActive;
    std::mutex mCalibrationMutex;

    /// Total frames captured since device started (for AEC sync)
    /// Atomic for lock-free access from data callback
    std::atomic<size_t> mTotalFramesCaptured{0};

private:
    ma_context context;
    ma_device_info *pPlaybackInfos;
    ma_uint32 playbackCount;
    ma_device_info *pCaptureInfos;
    ma_uint32 captureCount;
    ma_result result;
    ma_device device;


    /// true when the capture device is initialized.
    bool mInited;

    /// true when the context has been initialized (keep alive across init/dispose cycles)
    bool mContextInited;
};

#endif // CAPTURE_H