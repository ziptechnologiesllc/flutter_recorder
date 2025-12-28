#include "adaptive_echo_cancellation.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdint.h>

AdaptiveEchoCancellation::AdaptiveEchoCancellation(unsigned int sampleRate,
                                                     unsigned int channels)
    : mSampleRate(sampleRate),
      mChannels(channels),
      mParams{
          // Parameter: default, min, max
          {StepSize, {0.05f, 0.001f, 0.2f}},  // NLMS step size (matches nlms_filter.h)
          {DelayMs, {30.0f, 0.0f, 100.0f}},   // Acoustic delay in ms (30ms default for phone)
          {Enabled, {1.0f, 0.0f, 1.0f}}       // Enable flag (1=on, 0=bypass)
      },
      mValues(ParamCount, 0.0f),
      mDelaySamples(0),
      mUseTimestampSync(false),
      mCurrentCallbackTimestamp(AECReferenceBuffer::Clock::now()) {
    // Initialize parameter values with defaults
    for (const auto& [param, range] : mParams) {
        mValues[param] = range.defaultVal;
    }

    // Calculate initial delay in samples
    updateDelay();

    // Create NLMS filter for each channel
    for (unsigned int ch = 0; ch < channels; ++ch) {
        mFilters.push_back(std::make_unique<NLMSFilter>(
            NLMSFilter::DEFAULT_FILTER_LENGTH,
            mValues[StepSize],
            NLMSFilter::DEFAULT_REGULARIZATION));
    }

    // Pre-allocate reference buffer for batch processing
    // Size: max expected frame count * channels (assume 4096 as max)
    mRefBuffer.resize(4096 * channels, 0.0f);
}

int AdaptiveEchoCancellation::getParamCount() const {
    return ParamCount;
}

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
    const auto& range = mParams.at(static_cast<Params>(param));
    value = std::clamp(value, range.minVal, range.maxVal);
    mValues[param] = value;

    // Apply changes to filters
    switch (static_cast<Params>(param)) {
        case StepSize:
            for (auto& filter : mFilters) {
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

void AdaptiveEchoCancellation::process(void* pInput, ma_uint32 frameCount,
                                        unsigned int channels, ma_format format) {
    // Use current timestamp for synchronization
    mCurrentCallbackTimestamp = AECReferenceBuffer::now();
    mUseTimestampSync = true;

    processWithTimestamp(pInput, frameCount, channels, format, mCurrentCallbackTimestamp);
}

void AdaptiveEchoCancellation::processWithTimestamp(void* pInput, ma_uint32 frameCount,
                                                     unsigned int channels, ma_format format,
                                                     AECReferenceBuffer::TimePoint timestamp) {
    // Check if filter is enabled
    if (mValues[Enabled] < 0.5f) {
        return;  // Bypass - no processing
    }

    // Check if reference buffer is available
    if (g_aecReferenceBuffer == nullptr) {
        return;  // No reference signal - can't do echo cancellation
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
void AdaptiveEchoCancellation::processAudio(void* pInput, ma_uint32 frameCount,
                                             unsigned int channels) {
    T* input = static_cast<T*>(pInput);

    // Ensure we have enough filters for the channels
    while (mFilters.size() < channels) {
        mFilters.push_back(std::make_unique<NLMSFilter>(
            NLMSFilter::DEFAULT_FILTER_LENGTH,
            mValues[StepSize],
            NLMSFilter::DEFAULT_REGULARIZATION));
    }

    // Read reference signal from shared buffer
    size_t totalSamples = frameCount * channels;
    if (mRefBuffer.size() < totalSamples) {
        mRefBuffer.resize(totalSamples);
    }

    // Read reference signal with timestamp-based alignment
    // This uses the calibrated delay (DelayMs) and the current callback timestamp
    // to find the corresponding samples in the reference buffer
    float delayMs = mValues[DelayMs];

    // Use simple sample-based delay instead of timestamp synchronization
    // Timestamps were causing jitter and misalignment
    //
    // IMPORTANT: readFrames() subtracts samplesToRead from the read position:
    //   readPos = writePos - delaySamples - samplesToRead
    // This effectively adds one block duration (~2.67ms at 128 frames/48kHz) to the delay.
    // We compensate by subtracting this from our delay calculation.
    //
    // Also add a small jitter buffer (1ms) to account for timing variations between
    // the SoLoud output callback and the mic input callback.
    const float blockDurationMs = (128.0f / 48000.0f) * 1000.0f;  // ~2.67ms
    const float jitterBufferMs = 1.0f;  // Extra margin for callback timing jitter
    float adjustedDelayMs = delayMs - blockDurationMs + jitterBufferMs;
    adjustedDelayMs = std::max(adjustedDelayMs, 0.0f);  // Don't go negative

    size_t delaySamples = static_cast<size_t>((adjustedDelayMs / 1000.0f) * mSampleRate) * channels;

    size_t framesRead = g_aecReferenceBuffer->readFrames(
        mRefBuffer.data(), frameCount, delaySamples);

    // Accumulators for smoothed metrics over multiple blocks
    static int debugCounter = 0;
    static float totalRefEnergy = 0.0f;
    static float totalMicEnergy = 0.0f;
    static float totalCrossCorr = 0.0f;  // Accumulated cross-correlation
    static float totalRefEnergyForCorr = 0.0f;  // Ref energy for correlation calc
    static int debugSamples = 0;


    // If we couldn't read the reference (data not available), skip processing
    if (framesRead == 0) {
        return;
    }

    // NLMS ADAPTIVE ECHO CANCELLATION
    // The NLMS filter learns the complete speaker→mic transfer function
    // including device distortion, acoustic path, and room reflections.
    // Filter coefficients are pre-initialized from calibration impulse response.

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

            // NLMS adaptive filter:
            // 1. Computes echo estimate via convolution with learned coefficients
            // 2. Subtracts echo estimate from mic signal
            // 3. Updates coefficients to minimize error
            float cancelledSample = mFilters[ch]->process(micSample, refSample);

            // Denormalize and write back
            input[idx] = denormalizeSample<T>(cancelledSample);
        }
    }

    // Debug output every ~100ms (~50 callbacks at 256 frames/callback @ 48kHz)
    if (++debugCounter % 50 == 0) {
        float avgRefEnergy = debugSamples > 0 ? totalRefEnergy / debugSamples : 0;
        float avgMicEnergy = debugSamples > 0 ? totalMicEnergy / debugSamples : 0;

        // Get NLMS convergence metrics
        float coeffEnergy = 0.0f, errorEnergyAvg = 0.0f, coeffChangeRate = 0.0f;
        if (!mFilters.empty()) {
            mFilters[0]->getConvergenceMetrics(coeffEnergy, errorEnergyAvg, coeffChangeRate);
        }

        // Calculate correlation
        float correlation = 0.0f;
        if (totalRefEnergyForCorr > 1e-8f && totalMicEnergy > 1e-8f) {
            correlation = totalCrossCorr / std::sqrt(totalRefEnergyForCorr * totalMicEnergy);
            correlation = std::clamp(correlation, -1.0f, 1.0f);
        }

        // Calculate attenuation (ref energy vs error energy)
        float attenuationDb = 0.0f;
        if (avgRefEnergy > 1e-10f && errorEnergyAvg > 1e-10f) {
            attenuationDb = 10.0f * std::log10(avgRefEnergy / errorEnergyAvg);
        }

        // Convert energy to dB for readability
        float refDb = avgRefEnergy > 1e-10f ? 10.0f * std::log10(avgRefEnergy) : -100.0f;
        float micDb = avgMicEnergy > 1e-10f ? 10.0f * std::log10(avgMicEnergy) : -100.0f;

        // Status based on NLMS convergence
        const char* status = "WAITING";
        if (attenuationDb > 20.0f) status = "✓ CANCELLING";
        else if (attenuationDb > 10.0f) status = "GOOD";
        else if (coeffEnergy > 0.01f) status = "ADAPTING";
        else if (attenuationDb > 5.0f) status = "WEAK";

        // Get current alpha (variable step-size) for debugging
        float alpha = 0.0f;
        if (!mFilters.empty()) {
            alpha = mFilters[0]->getAlpha();
        }

        printf("[AEC] delay=%.0fms α=%.2f ref=%.0fdB mic=%.0fdB corr=%.2f coef=%.3f atten=%.0fdB | %s\n",
               delayMs, alpha, refDb, micDb, correlation, coeffEnergy, attenuationDb, status);
        fflush(stdout);

        // Reset accumulators
        totalRefEnergy = 0.0f;
        totalMicEnergy = 0.0f;
        totalCrossCorr = 0.0f;
        totalRefEnergyForCorr = 0.0f;
        debugSamples = 0;
    }
}

void AdaptiveEchoCancellation::reset() {
    for (auto& filter : mFilters) {
        filter->reset();
    }
}

float AdaptiveEchoCancellation::getEchoReturnLoss() const {
    if (mFilters.empty()) return 0.0f;
    // Return average ERL across channels
    float totalERL = 0.0f;
    for (const auto& filter : mFilters) {
        totalERL += filter->getEchoReturnLoss();
    }
    return totalERL / mFilters.size();
}

void AdaptiveEchoCancellation::setImpulseResponse(const float* coeffs, int length) {
    // Pre-initialize all channel filters with the calibrated impulse response
    // This gives NLMS a starting point for immediate cancellation
    for (auto& filter : mFilters) {
        filter->setCoefficients(coeffs, length);
    }

    // Calculate coefficient energy for debug output
    float energy = 0.0f;
    for (int i = 0; i < length; ++i) {
        energy += coeffs[i] * coeffs[i];
    }
    printf("[AEC] Set impulse response: %d coefficients, energy=%.4f\n", length, energy);
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
    return sample;  // Already normalized
}

template <>
unsigned char AdaptiveEchoCancellation::denormalizeSample<unsigned char>(float sample) {
    return static_cast<unsigned char>(std::clamp(sample * 128.0f + 128.0f, 0.0f, 255.0f));
}

template <>
int16_t AdaptiveEchoCancellation::denormalizeSample<int16_t>(float sample) {
    return static_cast<int16_t>(std::clamp(sample * 32768.0f, -32768.0f, 32767.0f));
}

template <>
int32_t AdaptiveEchoCancellation::denormalizeSample<int32_t>(float sample) {
    return static_cast<int32_t>(std::clamp(sample * 2147483648.0f, -2147483648.0f, 2147483647.0f));
}

template <>
float AdaptiveEchoCancellation::denormalizeSample<float>(float sample) {
    return sample;
}
