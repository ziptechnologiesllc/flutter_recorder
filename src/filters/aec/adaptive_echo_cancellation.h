#ifndef ADAPTIVE_ECHO_CANCELLATION_H
#define ADAPTIVE_ECHO_CANCELLATION_H

#include "../../enums.h"
#include "../generic_filter.h"
#include "nlms_filter.h"
#include "reference_buffer.h"

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
  void setImpulseResponse(const float *coeffs, int length);

  // Stats
  AecStats getStats();
  void updateStats(float ref, float mic, float out);

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

  // Delay in samples for reference signal alignment (fallback if no timestamp)
  unsigned int mDelaySamples;

  // Temporary buffer for reference signal
  std::vector<float> mRefBuffer;

  // Timestamp-based synchronization
  bool mUseTimestampSync;
  AECReferenceBuffer::TimePoint mCurrentCallbackTimestamp;

  AecStats mCurrentStats = {0};

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
