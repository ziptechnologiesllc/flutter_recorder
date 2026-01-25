#include "adaptive_echo_cancellation.h"
#include "aec_test.h"

#include "neural_post_filter.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdint.h>
#include <string>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// Get sandbox-accessible temp directory (Standard C++ version)
static std::string getTempDir() { return "/tmp/"; }

extern void aecLog(const char *fmt, ...);

AdaptiveEchoCancellation::AdaptiveEchoCancellation(unsigned int sampleRate,
                                                   unsigned int channels)
    : mSampleRate(sampleRate), mChannels(channels),
      mParams{
          // Parameter: default, min, max
          {StepSize,
           {0.005f, 0.001f, 0.2f}}, // NLMS step size (matches nlms_filter.h)
          {DelayMs,
           {30.0f, 0.0f,
            100.0f}}, // Acoustic delay in ms (30ms default for phone)
          {Enabled, {1.0f, 0.0f, 1.0f}} // Enable flag (1=on, 0=bypass)
      },
      mValues(ParamCount, 0.0f), mDelaySamples(0), mUseTimestampSync(true),
      mCurrentCallbackTimestamp(AECReferenceBuffer::Clock::now()),
      mNeuralFilter(std::make_unique<NeuralPostFilter>(sampleRate, channels)) {
  // Initialize parameter values with defaults
  for (auto const &it : mParams) {
    mValues[it.first] = it.second.defaultVal;
  }
}

int AdaptiveEchoCancellation::getParamCount() const {
  return (int)mParams.size();
}

float AdaptiveEchoCancellation::getParamMax(int param) const {
  validateParam(param);
  return mParams.at((Params)param).maxVal;
}

float AdaptiveEchoCancellation::getParamMin(int param) const {
  validateParam(param);
  return mParams.at((Params)param).minVal;
}

float AdaptiveEchoCancellation::getParamDef(int param) const {
  validateParam(param);
  return mParams.at((Params)param).defaultVal;
}

std::string AdaptiveEchoCancellation::getParamName(int param) const {
  validateParam(param);
  switch (param) {
  case StepSize:
    return "Step Size";
  case DelayMs:
    return "Acoustic Delay (ms)";
  case Enabled:
    return "Enabled";
  default:
    return "";
  }
}

void AdaptiveEchoCancellation::setParamValue(int param, float value) {
  validateParam(param);
  mValues[param] = value;

  if (param == DelayMs) {
    updateDelay();
  }
}

float AdaptiveEchoCancellation::getParamValue(int param) const {
  validateParam(param);
  return mValues[param];
}

void AdaptiveEchoCancellation::process(void *pInput, ma_uint32 frameCount,
                                       unsigned int channels,
                                       ma_format format) {
  processWithTimestamp(pInput, frameCount, channels, format,
                       mCurrentCallbackTimestamp);
}

void AdaptiveEchoCancellation::processWithTimestamp(
    void *pInput, ma_uint32 frameCount, unsigned int channels, ma_format format,
    AECReferenceBuffer::TimePoint timestamp) {
  // Check if filter is enabled
  if (mValues[Enabled] < 0.5f) {
    return; // Bypass - no processing
  }

  // Check if reference buffer is available
  if (g_aecReferenceBuffer == nullptr) {
    return; // No reference signal - can't do echo cancellation
  }

  // Store timestamp for use in processAudio
  mCurrentCallbackTimestamp = timestamp;
  mUseTimestampSync = true;

  switch (format) {
  case ma_format_u8:
    processAudio<unsigned char>(pInput, frameCount, channels);
    break;
  case ma_format_s16:
    processAudio<int16_t>(pInput, frameCount, channels);
    break;
  case ma_format_s32:
    processAudio<int32_t>(pInput, frameCount, channels);
    break;
  case ma_format_f32:
    processAudio<float>(pInput, frameCount, channels);
    break;
  default:
    std::cerr << "AdaptiveEchoCancellation: Unsupported format\n";
    break;
  }
}

template <typename T>
void AdaptiveEchoCancellation::processAudio(void *pInput, ma_uint32 frameCount,
                                            unsigned int channels) {

  T *input = static_cast<T *>(pInput);

  // Accumulators for smoothed metrics over multiple blocks
  static int debugCounter = 0;
  static float totalRefEnergy = 0.0f;
  static float totalMicEnergy = 0.0f;
  static float totalCrossCorr = 0.0f;        // Accumulated cross-correlation
  static float totalRefEnergyForCorr = 0.0f; // Ref energy for correlation calc
  static int debugSamples = 0;

  static int callCount = 0;
  if (++callCount % 187 == 0) {
    aecLog(
        "[AEC processAudio] call #%d, frames=%u ch=%u enabled=%.1f refBuf=%p\n",
        callCount, frameCount, channels, mValues[Enabled],
        g_aecReferenceBuffer);
  }

  // Ensure we have enough filters for the channels
  while (mFilters.size() < channels) {
    mFilters.push_back(
        std::make_unique<NLMSFilter>(NLMSFilter::DEFAULT_FILTER_LENGTH));
    mVssFilters.push_back(
        std::make_unique<VssNlmsFilter>(VssNlmsFilter::DEFAULT_FILTER_LENGTH));
  }

  size_t totalSamples = frameCount * channels;
  if (mLinearOutputBuffer.size() < totalSamples) {
    mLinearOutputBuffer.resize(totalSamples);
  }
  if (mLocalMicBuffer.size() < frameCount) {
    mLocalMicBuffer.resize(frameCount);
    mLocalErrorBuffer.resize(frameCount);
  }

  // Determine reference frames to read
  bool syncValid = false;
  size_t startOutputFrame = 0;

  if (mUsePositionSync && mCalibratedOffset != 0) {
    int64_t absoluteStart =
        static_cast<int64_t>(mCaptureFrameCount) - mCalibratedOffset;
    if (absoluteStart >= 0) {
      startOutputFrame = static_cast<size_t>(absoluteStart);
      syncValid = true;
    }
  }

  // Fallback to timestamp sync if position sync is not available or failed
  if (!syncValid && g_aecReferenceBuffer != nullptr) {
    // We use the current callback timestamp (if available) or "now" as a last
    // resort
    AECReferenceBuffer::TimePoint targetTime = mCurrentCallbackTimestamp;
    if (targetTime.time_since_epoch().count() == 0) {
      targetTime = AECReferenceBuffer::now();
    }

    // Default hardware delay fallback (estimated 150ms if not calibrated)
    float delayMs = mValues[DelayMs];
    if (delayMs <= 0.0f)
      delayMs = 150.0f;

    int64_t frame =
        g_aecReferenceBuffer->findFrameForTimestamp(targetTime, delayMs);
    if (frame >= 0) {
      // findFrameForTimestamp returns the "head", we want the block start
      startOutputFrame = static_cast<size_t>(frame) - frameCount;
      syncValid = true;
    }
  }

  // Output sync is determined. If no sync and not in calibration/bypass mode,
  // return early or bypass.

  // If no sync and not in calibration/bypass mode, return early
  if (!syncValid && !mCalibrationCaptureEnabled && mAecMode != aecModeBypass) {
    for (ma_uint32 i = 0; i < totalSamples; ++i) {
      input[i] = input[i]; // Bypass
    }
    return;
  }

  // BLOCK PROCESSING
  for (unsigned int ch = 0; ch < channels; ++ch) {
    // 1. Prepare Mono Microphone Block
    for (ma_uint32 i = 0; i < frameCount; ++i) {
      mLocalMicBuffer[i] = normalizeSample(input[i * channels + ch]);
    }

    // 2. Read Reference Context Window for this channel
    size_t filterLen = mVssFilters[ch]->getFilterLength();
    size_t contextSize = (filterLen - 1) + frameCount;
    if (mRefContextBuffer.size() < contextSize) {
      mRefContextBuffer.resize(contextSize);
    }

    if (syncValid && g_aecReferenceBuffer != nullptr) {
      g_aecReferenceBuffer->readContextWindow(mRefContextBuffer.data(),
                                              filterLen - 1, frameCount,
                                              startOutputFrame);
    } else {
      std::fill(mRefContextBuffer.begin(), mRefContextBuffer.end(), 0.0f);
    }

    // 3. Process Block
    if (mAecMode == aecModeAlgo || mAecMode == aecModeHybrid) {
      mVssFilters[ch]->processBlock(mRefContextBuffer.data(),
                                    mLocalMicBuffer.data(),
                                    mLocalErrorBuffer.data(), frameCount);
    } else {
      // Bypass or Neural-only: error is just the normalized mic
      std::copy(mLocalMicBuffer.begin(), mLocalMicBuffer.begin() + frameCount,
                mLocalErrorBuffer.begin());
    }

    // 4. Store to Linear Output
    for (ma_uint32 i = 0; i < frameCount; ++i) {
      mLinearOutputBuffer[i * channels + ch] = mLocalErrorBuffer[i];
    }
  }

  // Accumulate stats (using channel 0 for debug)
  if (debugSamples < 48000) { // Limit accumulation to avoid overflow
    size_t filterLen = mVssFilters[0]->getFilterLength();
    for (ma_uint32 i = 0; i < frameCount; ++i) {
      float mic = mLocalMicBuffer[i];
      float err = mLocalErrorBuffer[i];
      // ref current is at mRefContextBuffer[filterLen - 1 + i]
      float ref = mRefContextBuffer[filterLen - 1 + i];

      totalMicEnergy += mic * mic;
      totalRefEnergy += ref * ref;
      totalCrossCorr += mic * ref;
      totalRefEnergyForCorr += ref * ref;
      debugSamples++;

      // Calibration capture
      {
        std::lock_guard<std::mutex> lock(mCalibrationMutex);
        if (mCalibrationCaptureEnabled &&
            mAlignedRefCapture.size() < mCalibrationMaxSamples) {
          mAlignedRefCapture.push_back(ref);
          mAlignedMicCapture.push_back(mic);
        }
      }

      if (AECTest::isCapturing()) {
        AECTest::captureSample(mic, err, ref);
      }
    }
  }

  // SECOND STAGE: Neural Post-Filter
  if (mAecMode == aecModeNeural || mAecMode == aecModeHybrid) {
    // Neural filter needs reference block (not context)
    if (mRefBuffer.size() < frameCount)
      mRefBuffer.resize(frameCount);
    if (syncValid && g_aecReferenceBuffer != nullptr) {
      g_aecReferenceBuffer->readFramesAtPosition(mRefBuffer.data(), frameCount,
                                                 startOutputFrame);
    } else {
      std::fill(mRefBuffer.begin(), mRefBuffer.end(), 0.0f);
    }

    mNeuralFilter->process(mLinearOutputBuffer.data(), mRefBuffer.data(),
                           mLinearOutputBuffer.data(), frameCount);
  }

  // Write back
  for (unsigned int i = 0; i < totalSamples; ++i) {
    input[i] = denormalizeSample<T>(mLinearOutputBuffer[i]);
  }

  // Debug output every ~1s (~187 callbacks @ 48kHz/256 frames)
  if (++debugCounter % 187 == 0) {
    float avgRefEnergy = debugSamples > 0 ? totalRefEnergy / debugSamples : 0;
    float avgMicEnergy = debugSamples > 0 ? totalMicEnergy / debugSamples : 0;

    // Compute cross-correlation to check alignment (normalized)
    float correlation = 0.0f;
    if (totalRefEnergyForCorr > 1e-10f && totalMicEnergy > 1e-10f) {
      correlation =
          totalCrossCorr / std::sqrt(totalRefEnergyForCorr * totalMicEnergy);
    }

    float attenuation = 0.0f;
    if (avgMicEnergy > 1e-10f) {
      attenuation = 10.0f * std::log10(avgRefEnergy / avgMicEnergy);
    }

    {
      std::lock_guard<std::mutex> lock(mStatsMutex);
      mCurrentStats.maxAttenuationDb =
          std::max(mCurrentStats.maxAttenuationDb, attenuation);
      mCurrentStats.avgAttenuationDb =
          0.9f * mCurrentStats.avgAttenuationDb + 0.1f * attenuation;
      mCurrentStats.minAttenuationDb =
          std::min(mCurrentStats.minAttenuationDb, attenuation);
      mCurrentStats.correlation = correlation;
      mCurrentStats.refEnergy = avgRefEnergy;
      mCurrentStats.micEnergy = avgMicEnergy;
    }

    // Get filter metrics
    float coeffEnergy = 0.0f;
    float echoEst = 0.0f;
    if (!mVssFilters.empty()) {
      coeffEnergy = mVssFilters[0]->getCoeffEnergy();
      echoEst = mVssFilters[0]->getLastEchoEstimate();
    }

    // Convert energy to dB for readability
    float refDb =
        avgRefEnergy > 1e-10f ? 10.0f * std::log10(avgRefEnergy) : -100.0f;
    float micDb =
        avgMicEnergy > 1e-10f ? 10.0f * std::log10(avgMicEnergy) : -100.0f;
    float echoEstDb = std::abs(echoEst) > 1e-10f
                          ? 20.0f * std::log10(std::abs(echoEst))
                          : -100.0f;

    // Status based on coefficient energy (indicates calibration was applied)
    const char *status = coeffEnergy > 0.001f ? "ACTIVE" : "NO COEFFS";

    float currentDelayMs = mValues[DelayMs];

    aecLog("[AEC] delay=%.1fms ref=%.0fdB mic=%.0fdB ŷ=%.0fdB corr=%.2f "
           "coef=%.4f | %s\n",
           currentDelayMs, refDb, micDb, echoEstDb, correlation, coeffEnergy,
           status);

    if (correlation > 0.1f) {
      aecLog("[AEC Stats] Attenuation: %.2f dB (Avg: %.2f dB), Corr: %.4f, "
             "Ref: %.6f, Mic: %.6f\n",
             attenuation, mCurrentStats.avgAttenuationDb, correlation,
             avgRefEnergy, avgMicEnergy);
    }

    // Reset loop accumulators
    totalMicEnergy = 0;
    totalRefEnergy = 0;
    totalCrossCorr = 0;
    totalRefEnergyForCorr = 0;
    debugSamples = 0;
  }
}

void AdaptiveEchoCancellation::reset() {
  for (auto &filter : mFilters) {
    filter->reset();
  }
  for (auto &filter : mVssFilters) {
    filter->reset();
  }
  std::lock_guard<std::mutex> lock(mStatsMutex);
  mCurrentStats = {}; // Reset all stats
}

AecStats AdaptiveEchoCancellation::getStats() const {
  std::lock_guard<std::mutex> lock(mStatsMutex);
  AecStats stats = mCurrentStats; // Make a copy under lock

  if (!mVssFilters.empty()) {
    stats.filterLength = static_cast<int>(mVssFilters[0]->getFilterLength());
    stats.muMax = mVssFilters[0]->getMuMax();
    stats.muEffective = mVssFilters[0]->getLastStepSize();
    stats.instantCorrelation = mVssFilters[0]->getLastCorrelation();

    float lastErr = mVssFilters[0]->getLastError();
    stats.lastErrorDb = std::abs(lastErr) > 1e-10f
                            ? 20.0f * std::log10(std::abs(lastErr))
                            : -100.0f;
  } else {
    stats.filterLength = 0;
    stats.muMax = 0.0f;
    stats.muEffective = 0.0f;
    stats.lastErrorDb = -100.0f;
    stats.instantCorrelation = 0.0f;
  }
  return stats;
}

std::vector<float> AdaptiveEchoCancellation::getAlignedRef() const {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  return mAlignedRefCapture;
}

std::vector<float> AdaptiveEchoCancellation::getAlignedMic() const {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  return mAlignedMicCapture;
}

void AdaptiveEchoCancellation::setImpulseResponse(const float *coeffs,
                                                  int length) {
  for (auto &filter : mFilters) {
    filter->setCoefficients(coeffs, length);
  }
  for (auto &filter : mVssFilters) {
    filter->setWeights(coeffs, length);
  }

  float energy = 0.0f;
  for (int i = 0; i < length; ++i) {
    energy += coeffs[i] * coeffs[i];
  }
  aecLog("[AEC] Set impulse response: %d coefficients, energy=%.4f\n", length,
         energy);
}

float AdaptiveEchoCancellation::measureHardwareLatency(
    const std::vector<float> &refBuffer, const std::vector<float> &micBuffer) {
  if (refBuffer.empty() || micBuffer.empty())
    return 0.0f;

  int lagSamples = DelayEstimator::estimateDelay(refBuffer, micBuffer);
  float lagMs = (static_cast<float>(lagSamples) / mSampleRate) * 1000.0f;

  aecLog("[AEC] Measured Hardware Latency: %d samples (%.2f ms)\n", lagSamples,
         lagMs);

  setParamValue(Params::DelayMs, lagMs);
  return lagMs;
}

void AdaptiveEchoCancellation::validateParam(int param) const {
  if (param < 0 || param >= ParamCount) {
    throw std::invalid_argument("Invalid parameter index");
  }
}

void AdaptiveEchoCancellation::updateDelay() {
  float delayMs = mValues[DelayMs];
  mDelaySamples = static_cast<unsigned int>((delayMs / 1000.0f) * mSampleRate);
}

float AdaptiveEchoCancellation::normalizeSample(unsigned char sample) {
  return (sample - 128) / 128.0f;
}

float AdaptiveEchoCancellation::normalizeSample(int16_t sample) {
  return sample / 32768.0f;
}

float AdaptiveEchoCancellation::normalizeSample(int32_t sample) {
  return sample / 2147483648.0f;
}

float AdaptiveEchoCancellation::normalizeSample(float sample) { return sample; }

template <>
unsigned char
AdaptiveEchoCancellation::denormalizeSample<unsigned char>(float sample) {
  float val = sample * 128.0f + 128.0f;
  return static_cast<unsigned char>(std::max(0.0f, std::min(val, 255.0f)));
}

template <>
int16_t AdaptiveEchoCancellation::denormalizeSample<int16_t>(float sample) {
  float val = sample * 32768.0f;
  return static_cast<int16_t>(std::max(-32768.0f, std::min(val, 32767.0f)));
}

template <>
int32_t AdaptiveEchoCancellation::denormalizeSample<int32_t>(float sample) {
  float val = sample * 2147483648.0f;
  val = std::max(-2147483648.0f, std::min(val, 2147483647.0f));
  return static_cast<int32_t>(val);
}

template <>
float AdaptiveEchoCancellation::denormalizeSample<float>(float sample) {
  return sample;
}

void AdaptiveEchoCancellation::updateStats(float ref, float mic, float out) {
  // Stats are now updated in blocks inside processAudio for efficiency.
  // This legacy per-sample update is no longer used but kept for interface
  // compatibility.
}

void AdaptiveEchoCancellation::setVssMuMax(float mu) {
  for (auto &filter : mVssFilters) {
    filter->setStepSize(mu);
  }
}

void AdaptiveEchoCancellation::setVssLeakage(float lambda) {
  for (auto &filter : mVssFilters) {
    filter->setLeakage(lambda);
  }
}

void AdaptiveEchoCancellation::setVssAlpha(float alpha) {
  for (auto &filter : mVssFilters) {
    filter->setSmoothingFactor(alpha);
  }
}

float AdaptiveEchoCancellation::getVssMuMax() const {
  return mVssFilters.empty() ? 0.0f : mVssFilters[0]->getMuMax();
}

float AdaptiveEchoCancellation::getVssLeakage() const {
  return mVssFilters.empty() ? 1.0f : mVssFilters[0]->getLeakage();
}

float AdaptiveEchoCancellation::getVssAlpha() const {
  return mVssFilters.empty() ? 0.95f : mVssFilters[0]->getAlpha();
}

void AdaptiveEchoCancellation::setFilterLength(int length) {
  if (length < 256 || length > 16384) {
    return;
  }

  for (auto &filter : mVssFilters) {
    filter->resize(static_cast<size_t>(length));
  }
  for (auto &filter : mFilters) {
    filter->resize(static_cast<size_t>(length));
  }
}

int AdaptiveEchoCancellation::getFilterLength() const {
  return mVssFilters.empty()
             ? 0
             : static_cast<int>(mVssFilters[0]->getFilterLength());
}

void AdaptiveEchoCancellation::setCaptureFrameCount(size_t captureFrameCount) {
  mCaptureFrameCount = captureFrameCount;
}

void AdaptiveEchoCancellation::setCalibratedOffset(int64_t offset) {
  mCalibratedOffset = offset;
  mUsePositionSync = true;
}

void AdaptiveEchoCancellation::startCalibrationCapture(size_t maxSamples) {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  mAlignedRefCapture.clear();
  mAlignedMicCapture.clear();
  mAlignedRefCapture.reserve(maxSamples);
  mAlignedMicCapture.reserve(maxSamples);
  mCalibrationMaxSamples = maxSamples;
  mCalibrationCaptureEnabled = true;
}

void AdaptiveEchoCancellation::stopCalibrationCapture() {
  std::lock_guard<std::mutex> lock(mCalibrationMutex);
  mCalibrationCaptureEnabled = false;
}

bool AdaptiveEchoCancellation::isCalibrationCaptureComplete() const {
  return !mCalibrationCaptureEnabled &&
         mAlignedRefCapture.size() >= mCalibrationMaxSamples;
}

void AdaptiveEchoCancellation::setAecMode(AecMode mode) {
  mAecMode = mode;
  bool neuralEnabled = (mode == aecModeNeural || mode == aecModeHybrid);
  if (mNeuralFilter) {
    mNeuralFilter->setEnabled(neuralEnabled);
  }
}

AecMode AdaptiveEchoCancellation::getAecMode() const { return mAecMode; }

bool AdaptiveEchoCancellation::loadNeuralModel(const std::string &modelPath) {
  if (mNeuralFilter) {
    return mNeuralFilter->loadModel(modelPath);
  }
  return false;
}

// Explicit instantiations for the supported formats
template void AdaptiveEchoCancellation::processAudio<unsigned char>(
    void *pInput, ma_uint32 frameCount, unsigned int channels);
template void AdaptiveEchoCancellation::processAudio<int16_t>(
    void *pInput, ma_uint32 frameCount, unsigned int channels);
template void AdaptiveEchoCancellation::processAudio<int32_t>(
    void *pInput, ma_uint32 frameCount, unsigned int channels);
template void AdaptiveEchoCancellation::processAudio<float>(
    void *pInput, ma_uint32 frameCount, unsigned int channels);
