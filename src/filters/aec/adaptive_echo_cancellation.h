#ifndef ADAPTIVE_ECHO_CANCELLATION_H
#define ADAPTIVE_ECHO_CANCELLATION_H

#include "../../enums.h"
#include "../generic_filter.h"
#include "delay_estimator.h"
#include "neural_post_filter.h"
#include "nlms_filter.h"
#include "reference_buffer.h"
#include "vss_nlms_filter.h"

#include <map>
#include <memory>
#include <vector>

/**
 * Adaptive Echo Cancellation Filter using NLMS algorithm.
 *
 * This filter removes acoustic echo from the microphone input by:
 * 1. Reading the reference signal (speaker output) from a shared buffer
 * 2. Applying an adaptive NLMS filter to estimate the echo
 * 3. Subtracting the estimated echo from the microphone input
 *
 * The filter continuously adapts to changes in the acoustic environment.
 */
class AdaptiveEchoCancellation : public GenericFilter {
public:
  // Parameter indices
  enum Params {
    StepSize = 0, // NLMS adaptation rate
    DelayMs,      // Estimated acoustic delay in ms
    Enabled,      // Master enable/bypass
    ParamCount
  };

  /**
   * @param sampleRate Audio sample rate in Hz
   * @param channels Number of audio channels
   */
  AdaptiveEchoCancellation(unsigned int sampleRate, unsigned int channels = 2);
  ~AdaptiveEchoCancellation() override = default;

  // GenericFilter interface implementation
  int getParamCount() const override;
  float getParamMax(int param) const override;
  float getParamMin(int param) const override;
  float getParamDef(int param) const override;
  std::string getParamName(int param) const override;
  void setParamValue(int param, float value) override;
  float getParamValue(int param) const override;

  // Process audio through the AEC filter
  void process(void *pInput, ma_uint32 frameCount, unsigned int channels,
               ma_format format) override;

  // Process with explicit timestamp for synchronization
  void processWithTimestamp(void *pInput, ma_uint32 frameCount,
                            unsigned int channels, ma_format format,
                            AECReferenceBuffer::TimePoint timestamp);

  /**
   * Reset the filter state.
   */
  void reset();

  /**
   * Get the current echo return loss (ERL) in dB.
   * Higher values indicate better echo cancellation.
   */
  float getEchoReturnLoss() const;

  /**
   * Set the impulse response from calibration.
   * Pre-initializes NLMS filter coefficients for immediate cancellation.
   *
   * @param coeffs Impulse response coefficients
   * @param length Number of coefficients
   */
  /**
   * Set the impulse response from calibration.
   */
  void setImpulseResponse(const float *coeffs, int length);

  /**
   * Measure hardware latency using cross-correlation.
   * Updates the DelayMs parameter automatically.
   * @param refBuffer Reference signal buffer (1-2 seconds)
   * @param micBuffer Microphone signal buffer (1-2 seconds)
   * @return Measured delay in milliseconds
   */
  float measureHardwareLatency(const std::vector<float> &refBuffer,
                               const std::vector<float> &micBuffer);

  // Stats
  AecStats getStats();
  void updateStats(float ref, float mic, float out);

  NeuralPostFilter *getNeuralFilter() { return mNeuralFilter.get(); }

  // VSS-NLMS parameter control for experimentation
  void setVssMuMax(float mu);
  void setVssLeakage(float lambda);
  void setVssAlpha(float alpha);
  float getVssMuMax() const;
  float getVssLeakage() const;
  float getVssAlpha() const;

  // Filter length control
  void setFilterLength(int length);
  int getFilterLength() const;

  // Sample-accurate synchronization (frame counter based)
  // Call this BEFORE process() with the capture frame count at block start
  void setCaptureFrameCount(size_t captureFrameCount);

  // Set the calibrated offset: captureFrame - offset = corresponding
  // outputFrame This is calculated during calibration as: offset =
  // (captureFramesAtCalib - outputFramesAtCalib) + acousticDelaySamples
  void setCalibratedOffset(int64_t offset);
  int64_t getCalibratedOffset() const { return mCalibratedOffset; }

  // Set pure acoustic delay in samples (for slave mode where thread timing is irrelevant)
  void setAcousticDelaySamples(size_t samples) { mAcousticDelaySamples = samples; }
  size_t getAcousticDelaySamples() const { return mAcousticDelaySamples; }

  // Set buffer configuration for theoretical delay calculation
  void setBufferConfig(size_t bufferSizeFrames, size_t pipelinePeriods = 3) {
    mBufferSizeFrames = bufferSizeFrames;
    mPipelinePeriods = pipelinePeriods;
  }

  // Calculate theoretical delay based on buffer config
  // Full round-trip delay = output buffering + acoustic path + input buffering
  // For duplex device: both input and output have mPipelinePeriods of buffering
  size_t getTheoreticalDelaySamples() const {
    // Output buffering (DAC pipeline): N periods
    size_t outputLatency = mPipelinePeriods * mBufferSizeFrames;
    // Input buffering (ADC pipeline): typically same as output
    size_t inputLatency = mPipelinePeriods * mBufferSizeFrames;
    // Acoustic delay: speaker → air → mic (3-5ms typical for laptop)
    size_t acousticDelay = (mSampleRate * 4) / 1000; // 4ms default
    return outputLatency + inputLatency + acousticDelay;
  }

  // Enable/disable position-based sync (vs legacy timestamp/delay based)
  void setUsePositionSync(bool enable) { mUsePositionSync = enable; }
  bool getUsePositionSync() const { return mUsePositionSync; }

  // Calibration capture: capture frame-aligned ref/mic for delay estimation
  void startCalibrationCapture(size_t maxSamples = 96000); // 2 seconds @ 48kHz
  void stopCalibrationCapture();
  const std::vector<float> &getAlignedRef() const { return mAlignedRefCapture; }
  const std::vector<float> &getAlignedMic() const { return mAlignedMicCapture; }
  bool isCalibrationCaptureComplete() const;

  // AEC Mode Control (A/B Testing)
  void setAecMode(AecMode mode);
  AecMode getAecMode() const;

  /**
   * Loads a TFLite model for the neural post-filter.
   * @param modelPath Path to the .tflite model file.
   * @return true if successful.
   */
  bool loadNeuralModel(const std::string &modelPath);

private:
  struct ParamRange {
    float defaultVal;
    float minVal;
    float maxVal;
  };

  unsigned int mSampleRate;
  unsigned int mChannels;

  // Parameter storage
  std::map<Params, ParamRange> mParams;
  std::vector<float> mValues;

  // NLMS filter instances (one per channel)
  std::vector<std::unique_ptr<NLMSFilter>> mFilters;
  // VSS-NLMS filter instances (Parallel path for new algo)
  std::vector<std::unique_ptr<VssNlmsFilter>> mVssFilters;

  // Delay in samples for reference signal alignment (fallback if no timestamp)
  unsigned int mDelaySamples;

  // Temporary buffer for reference signal
  std::vector<float> mRefBuffer;
  // Temporary buffer for linear AEC output
  std::vector<float> mLinearOutputBuffer;

  // Timestamp-based synchronization
  bool mUseTimestampSync;
  AECReferenceBuffer::TimePoint mCurrentCallbackTimestamp;

  std::unique_ptr<NeuralPostFilter> mNeuralFilter;

  AecStats mCurrentStats = {0};

  // Sample-accurate sync state
  size_t mCaptureFrameCount = 0; // Set before each process() call
  int64_t mCalibratedOffset = 0; // Capture frame - offset = output frame
  bool mUsePositionSync = false; // Use position-based sync vs legacy delay
  size_t mAcousticDelaySamples = 0; // Pure acoustic delay (for slave mode)

  // Buffer configuration for theoretical delay calculation
  size_t mBufferSizeFrames = 128; // Default audio buffer size
  size_t mPipelinePeriods = 3;    // Typical pipeline depth for PipeWire/ALSA

  // Calibration capture state (for frame-aligned delay estimation)
  bool mCalibrationCaptureEnabled = false;
  size_t mCalibrationMaxSamples = 0;
  std::vector<float> mAlignedRefCapture;
  std::vector<float> mAlignedMicCapture;

  // AEC Mode
  AecMode mAecMode = aecModeHybrid;

  void validateParam(int param) const;
  void updateDelay();

  // Templated processing for different sample formats
  template <typename T>
  void processAudio(void *pInput, ma_uint32 frameCount, unsigned int channels);

  // Format conversion helpers
  float normalizeSample(unsigned char sample);
  float normalizeSample(int16_t sample);
  float normalizeSample(int32_t sample);
  float normalizeSample(float sample);

  template <typename T> T denormalizeSample(float sample);
};

#endif // ADAPTIVE_ECHO_CANCELLATION_H
