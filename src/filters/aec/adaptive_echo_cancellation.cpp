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
#if TARGET_OS_MAC && !TARGET_OS_IPHONE
#include <Foundation/Foundation.h>
#endif
#endif

// Get sandbox-accessible temp directory
static std::string getTempDir() {
#ifdef __APPLE__
#if TARGET_OS_MAC && !TARGET_OS_IPHONE
  NSString* tempDir = NSTemporaryDirectory();
  if (tempDir) {
    return std::string([tempDir UTF8String]);
  }
#endif
#endif
  return "/tmp/";
}

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

  // Calculate initial delay in samples
  updateDelay();

  // Create FIR filter for each channel
  for (unsigned int ch = 0; ch < channels; ++ch) {
    mFilters.push_back(
        std::make_unique<NLMSFilter>(NLMSFilter::DEFAULT_FILTER_LENGTH));
    mVssFilters.push_back(
        std::make_unique<VssNlmsFilter>(VssNlmsFilter::DEFAULT_FILTER_LENGTH));
  }

  // Pre-allocate reference buffer for batch processing
  // Size: max expected frame count * channels (assume 4096 as max)
  mRefBuffer.resize(4096 * channels, 0.0f);
}

int AdaptiveEchoCancellation::getParamCount() const { return ParamCount; }

float AdaptiveEchoCancellation::getParamMax(int param) const {
  validateParam(param);
  return mParams.at(static_cast<Params>(param)).maxVal;
}

float AdaptiveEchoCancellation::getParamMin(int param) const {
  validateParam(param);
  return mParams.at(static_cast<Params>(param)).minVal;
}

float AdaptiveEchoCancellation::getParamDef(int param) const {
  validateParam(param);
  return mParams.at(static_cast<Params>(param)).defaultVal;
}

std::string AdaptiveEchoCancellation::getParamName(int param) const {
  validateParam(param);
  switch (static_cast<Params>(param)) {
  case StepSize:
    return "Step Size";
  case DelayMs:
    return "Delay (ms)";
  case Enabled:
    return "Enabled";
  default:
    return "Unknown";
  }
}

void AdaptiveEchoCancellation::setParamValue(int param, float value) {
  validateParam(param);
  const auto &range = mParams.at(static_cast<Params>(param));
  // std::clamp is C++17, using max/min for C++14 compatibility
  value = std::max(range.minVal, std::min(value, range.maxVal));
  mValues[param] = value;

  // Apply changes to filters
  switch (static_cast<Params>(param)) {
  case StepSize:
    for (auto &filter : mFilters) {
      filter->setStepSize(value);
    }
    break;
  case DelayMs:
    updateDelay();
    break;
  case Enabled:
    if (value < 0.5f) {
      // Reset filters when disabled
      reset();
    }
    break;
  default:
    break;
  }
}

float AdaptiveEchoCancellation::getParamValue(int param) const {
  validateParam(param);
  return mValues[param];
}

void AdaptiveEchoCancellation::process(void *pInput, ma_uint32 frameCount,
                                       unsigned int channels,
                                       ma_format format) {
  // Use current timestamp for synchronization
  mCurrentCallbackTimestamp = AECReferenceBuffer::now();
  mUseTimestampSync = true;

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
  static int callCount = 0;
  callCount++;
  // Log sparingly: first 5 calls, then every 500 calls (~5 seconds at typical buffer sizes)
  if (callCount <= 5 || callCount % 500 == 0) {
    aecLog("[AEC processAudio] call #%d, frames=%u ch=%u enabled=%.1f refBuf=%p\n",
           callCount, frameCount, channels, mValues[Enabled], g_aecReferenceBuffer);
  }

  T *input = static_cast<T *>(pInput);

  // Ensure we have enough filters for the channels
  while (mFilters.size() < channels) {
    mFilters.push_back(
        std::make_unique<NLMSFilter>(NLMSFilter::DEFAULT_FILTER_LENGTH));
    mVssFilters.push_back(
        std::make_unique<VssNlmsFilter>(VssNlmsFilter::DEFAULT_FILTER_LENGTH));
  }

  // Read reference signal from shared buffer
  size_t totalSamples = frameCount * channels;
  if (mRefBuffer.size() < totalSamples) {
    mRefBuffer.resize(totalSamples);
  }
  if (mLinearOutputBuffer.size() < totalSamples) {
    mLinearOutputBuffer.resize(totalSamples);
  }

  // Read reference signal
  size_t framesRead = 0;

  if (mUsePositionSync && mCalibratedOffset != 0) {
    // NEW: Position-based sync using frame counters
    // Calculate the output frame position that corresponds to this capture block
    // captureFrame - offset = outputFrame
    int64_t startOutputFrame =
        static_cast<int64_t>(mCaptureFrameCount) - mCalibratedOffset;

    if (startOutputFrame >= 0) {
      framesRead = g_aecReferenceBuffer->readFramesAtPosition(
          mRefBuffer.data(), frameCount, static_cast<size_t>(startOutputFrame));

      static int posReadDebugCount = 0;
      if (++posReadDebugCount % 500 == 0) {
        aecLog("[AEC PosSync] capFrame=%zu offset=%lld outFrame=%lld read=%zu\n",
               mCaptureFrameCount, (long long)mCalibratedOffset,
               (long long)startOutputFrame, framesRead);
      }
    }
  } else {
    // LEGACY: Simple sample-based delay (for backwards compatibility)
    float delayMs = mValues[DelayMs];
    size_t delaySamples =
        static_cast<size_t>((delayMs / 1000.0f) * mSampleRate) * channels;
    framesRead = g_aecReferenceBuffer->readFrames(mRefBuffer.data(), frameCount,
                                                  delaySamples);
  }

  // Accumulators for smoothed metrics over multiple blocks
  static int debugCounter = 0;
  static float totalRefEnergy = 0.0f;
  static float totalMicEnergy = 0.0f;
  static float totalCrossCorr = 0.0f;        // Accumulated cross-correlation
  static float totalRefEnergyForCorr = 0.0f; // Ref energy for correlation calc
  static int debugSamples = 0;

  // If we couldn't read the reference (data not available), skip processing
  if (framesRead == 0) {
    return;
  }

  // DEBUG: Dump ref and mic to files for alignment verification
  static FILE* refFile = nullptr;
  static FILE* micFile = nullptr;
  static int dumpFrames = 0;
  static const int MAX_DUMP_FRAMES = 48000 * 5; // 5 seconds
  static std::string tempDir;

  if (dumpFrames == 0 && refFile == nullptr) {
    tempDir = getTempDir();
    std::string refPath = tempDir + "aec_ref.raw";
    std::string micPath = tempDir + "aec_mic.raw";
    refFile = fopen(refPath.c_str(), "wb");
    micFile = fopen(micPath.c_str(), "wb");
    if (refFile && micFile) {
      aecLog("[AEC DEBUG] Dumping to: %s\n", tempDir.c_str());
      aecLog("[AEC DEBUG] Files: aec_ref.raw, aec_mic.raw\n");
    } else {
      aecLog("[AEC DEBUG] Failed to open files in %s\n", tempDir.c_str());
    }
  }

  if (refFile && micFile && dumpFrames < MAX_DUMP_FRAMES) {
    // Write channel 0 only (mono) for easier analysis
    for (ma_uint32 frame = 0; frame < frameCount; ++frame) {
      float micSample = normalizeSample(static_cast<T*>(pInput)[frame * channels]);
      float refSample = mRefBuffer[frame * channels];
      fwrite(&refSample, sizeof(float), 1, refFile);
      fwrite(&micSample, sizeof(float), 1, micFile);
    }
    dumpFrames += frameCount;

    if (dumpFrames >= MAX_DUMP_FRAMES) {
      fclose(refFile);
      fclose(micFile);
      refFile = nullptr;
      micFile = nullptr;
      aecLog("[AEC DEBUG] Finished dumping %d frames to files\n", dumpFrames);
    }
  }

  // Calibration capture: save frame-aligned ref+mic for delay estimation
  // These are perfectly aligned since they come from the same callback
  if (mCalibrationCaptureEnabled && mAlignedRefCapture.size() < mCalibrationMaxSamples) {
    for (ma_uint32 frame = 0; frame < frameCount; ++frame) {
      if (mAlignedRefCapture.size() >= mCalibrationMaxSamples) break;
      float micSample = normalizeSample(static_cast<T*>(pInput)[frame * channels]);
      float refSample = mRefBuffer[frame * channels];
      mAlignedRefCapture.push_back(refSample);
      mAlignedMicCapture.push_back(micSample);
    }

    // Log progress periodically
    static int calibCapLogCount = 0;
    if (++calibCapLogCount % 100 == 0) {
      aecLog("[AEC CalibCapture] %zu/%zu samples\n",
             mAlignedRefCapture.size(), mCalibrationMaxSamples);
    }
  }

  // NLMS ADAPTIVE ECHO CANCELLATION
  for (ma_uint32 frame = 0; frame < frameCount; ++frame) {
    for (unsigned int ch = 0; ch < channels; ++ch) {
      size_t idx = frame * channels + ch;

      // Normalize mic sample to float
      float micSample = normalizeSample(input[idx]);

      // Get reference sample (already float, properly delayed by calibration)
      float refSample = mRefBuffer[idx];

      // Accumulate for debug output
      totalRefEnergy += refSample * refSample;
      totalMicEnergy += micSample * micSample;
      totalCrossCorr += micSample * refSample;
      totalRefEnergyForCorr += refSample * refSample;
      debugSamples++;

      // Fixed FIR echo cancellation (Old Path):
      // float error = mFilters[ch]->process(micSample, refSample);

      // VSS-NLMS Adaptive Echo Cancellation (New Path):
      // 1. Estimate echo using adaptive filter
      // 2. Subtract from mic signal (happens inside processSample)
      float error = mVssFilters[ch]->processSample(refSample, micSample);

      // Store stage 1 result in float buffer for stage 2
      mLinearOutputBuffer[idx] = error;

      // Capture samples for AEC test (channel 0 only to avoid duplicates)
      if (ch == 0 && AECTest::isCapturing()) {
        AECTest::captureSample(micSample, error, refSample);
      }
    }
  }

  // SECOND STAGE: Neural Post-Filter
  // This stage runs on the output of the NLMS filters to remove
  // residual echo and non-linearities.
  // We reuse mRefBuffer for the reference signal.
  mNeuralFilter->process(mLinearOutputBuffer.data(), mRefBuffer.data(),
                         mLinearOutputBuffer.data(), frameCount);

  // Write final results back to the original input buffer
  for (unsigned int i = 0; i < totalSamples; ++i) {
    input[i] = denormalizeSample<T>(mLinearOutputBuffer[i]);
  }

  // Debug output every ~1s (~500 callbacks at 256 frames/callback @ 48kHz)
  if (++debugCounter % 500 == 0) {
    float avgRefEnergy = debugSamples > 0 ? totalRefEnergy / debugSamples : 0;
    float avgMicEnergy = debugSamples > 0 ? totalMicEnergy / debugSamples : 0;

    // Compute cross-correlation to check alignment (normalized)
    float correlation = 0.0f;
    if (totalRefEnergyForCorr > 1e-10f && totalMicEnergy > 1e-10f) {
      correlation =
          totalCrossCorr / std::sqrt(totalRefEnergyForCorr * totalMicEnergy);
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

    // Reset accumulators
    totalRefEnergy = 0.0f;
    totalMicEnergy = 0.0f;
    totalCrossCorr = 0.0f;
    totalRefEnergyForCorr = 0.0f;
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
}

float AdaptiveEchoCancellation::getEchoReturnLoss() const {
  // Use the stats calculated in updateStats() which reflects the active filter
  // performance
  return mCurrentStats.echoReturnLossDb;
}

void AdaptiveEchoCancellation::setImpulseResponse(const float *coeffs,
                                                  int length) {
  // Pre-initialize all channel filters with the calibrated impulse response
  // This gives NLMS a starting point for immediate cancellation
  for (auto &filter : mFilters) {
    filter->setCoefficients(coeffs, length);
  }
  for (auto &filter : mVssFilters) {
    filter->setWeights(coeffs, length);
  }

  // Calculate coefficient energy for debug output
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

  // Use DelayEstimator to find lag
  int lagSamples = DelayEstimator::estimateDelay(refBuffer, micBuffer);

  // Convert to ms
  float lagMs = (static_cast<float>(lagSamples) / mSampleRate) * 1000.0f;

  aecLog("[AEC] Measured Hardware Latency: %d samples (%.2f ms)\n", lagSamples,
         lagMs);

  // Update the DelayMs parameter
  setParamValue(Params::DelayMs, lagMs);

  return lagMs;
}

void AdaptiveEchoCancellation::validateParam(int param) const {
  if (param < 0 || param >= ParamCount) {
    throw std::invalid_argument("Invalid parameter index");
  }
}

void AdaptiveEchoCancellation::updateDelay() {
  // Convert delay from ms to samples
  float delayMs = mValues[DelayMs];
  mDelaySamples = static_cast<unsigned int>((delayMs / 1000.0f) * mSampleRate);
}

// Sample normalization helpers
float AdaptiveEchoCancellation::normalizeSample(unsigned char sample) {
  return (sample - 128) / 128.0f;
}

float AdaptiveEchoCancellation::normalizeSample(int16_t sample) {
  return sample / 32768.0f;
}

float AdaptiveEchoCancellation::normalizeSample(int32_t sample) {
  return sample / 2147483648.0f;
}

float AdaptiveEchoCancellation::normalizeSample(float sample) {
  return sample; // Already normalized
}

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
  // Note: min/max on float first to avoid overflow wrap-around
  val = std::max(-2147483648.0f, std::min(val, 2147483647.0f));
  return static_cast<int32_t>(val);
}

template <>
float AdaptiveEchoCancellation::denormalizeSample<float>(float sample) {
  return sample;
}

void AdaptiveEchoCancellation::updateStats(float ref, float mic, float out) {
  // Simple leaky integrator stats
  float micEnergy = mic * mic;
  float outEnergy = out * out;

  // Smooth energies (tau ~= 100ms at 48kHz, alpha=0.0005)
  // For ~100 samples per call (block), alpha=0.1
  const float alpha = 0.01f;
  static float smoothMic = 0.0f;
  static float smoothOut = 0.0f;

  smoothMic = (1.0f - alpha) * smoothMic + alpha * micEnergy;
  smoothOut = (1.0f - alpha) * smoothOut + alpha * outEnergy;

  // Calculate instantaneous attenuation (positive dB)
  float atten = 0.0f;
  if (smoothMic > 1e-9f && smoothOut > 1e-9f) {
    float ratio = smoothMic / smoothOut;
    // ratio > 1 means mic > out (attenuation, since "out" is error)
    // Wait, GenericFilter input is (mic, ref). output is error.
    // If output is smaller than mic input, we have attenuation.
    if (ratio > 1.0f) {
      atten = 10.0f * std::log10(ratio);
    }
  }

  mCurrentStats.maxAttenuationDb = atten; // For now just current attenuation
  mCurrentStats.echoReturnLossDb = atten; // Proxy
  // Correlation is harder to calculate cheaply per-sample, skipping for now or
  // user approximation
  mCurrentStats.correlation = 0.0f;
}

AecStats AdaptiveEchoCancellation::getStats() { return mCurrentStats; }

// VSS-NLMS parameter control
void AdaptiveEchoCancellation::setVssMuMax(float mu) {
  for (auto &filter : mVssFilters) {
    filter->setStepSize(mu);
  }
  aecLog("[AEC] Set VSS mu_max=%.4f for %zu filters\n", mu, mVssFilters.size());
}

void AdaptiveEchoCancellation::setVssLeakage(float lambda) {
  for (auto &filter : mVssFilters) {
    filter->setLeakage(lambda);
  }
  aecLog("[AEC] Set VSS leakage=%.6f for %zu filters\n", lambda,
         mVssFilters.size());
}

void AdaptiveEchoCancellation::setVssAlpha(float alpha) {
  for (auto &filter : mVssFilters) {
    filter->setSmoothingFactor(alpha);
  }
  aecLog("[AEC] Set VSS alpha=%.4f for %zu filters\n", alpha,
         mVssFilters.size());
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

void AdaptiveEchoCancellation::setCaptureFrameCount(size_t captureFrameCount) {
  mCaptureFrameCount = captureFrameCount;
}

void AdaptiveEchoCancellation::setCalibratedOffset(int64_t offset) {
  mCalibratedOffset = offset;
  mUsePositionSync = true;  // Enable position sync when offset is set
  aecLog("[AEC] Set calibrated offset=%lld, position sync enabled\n",
         (long long)offset);
}

// Calibration capture methods
void AdaptiveEchoCancellation::startCalibrationCapture(size_t maxSamples) {
  mAlignedRefCapture.clear();
  mAlignedMicCapture.clear();
  mAlignedRefCapture.reserve(maxSamples);
  mAlignedMicCapture.reserve(maxSamples);
  mCalibrationMaxSamples = maxSamples;
  mCalibrationCaptureEnabled = true;
  aecLog("[AEC] Calibration capture started (max %zu samples)\n", maxSamples);
}

void AdaptiveEchoCancellation::stopCalibrationCapture() {
  mCalibrationCaptureEnabled = false;
  aecLog("[AEC] Calibration capture stopped: ref=%zu mic=%zu samples\n",
         mAlignedRefCapture.size(), mAlignedMicCapture.size());
}

bool AdaptiveEchoCancellation::isCalibrationCaptureComplete() const {
  return !mCalibrationCaptureEnabled &&
         mAlignedRefCapture.size() >= mCalibrationMaxSamples;
}
