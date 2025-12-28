#ifndef NLMS_FILTER_H
#define NLMS_FILTER_H

#include <algorithm>
#include <cmath>
#include <cstring>

/**
 * Static FIR Echo Cancellation Filter
 *
 * Uses calibrated impulse response coefficients to estimate and subtract
 * the acoustic echo. No adaptive learning - coefficients are fixed after
 * calibration for stable, distortion-free operation with music.
 *
 * For music/looper use cases, adaptive filters cause distortion because:
 * - Music is highly correlated (unlike speech)
 * - Continuous audio prevents convergence
 * - Adaptation artifacts create audible artifacts
 *
 * Static FIR avoids these issues by using pre-measured impulse response.
 */
class NLMSFilter {
public:
  static constexpr int DEFAULT_FILTER_LENGTH = 2048; // ~43ms @ 48kHz
  static constexpr float DEFAULT_REGULARIZATION = 1e-6f;

  NLMSFilter(int filterLength = DEFAULT_FILTER_LENGTH, float stepSize = 0.1f,
             float regularization = DEFAULT_REGULARIZATION)
      : mFilterLength(filterLength), mHistoryIndex(0), mStepSize(stepSize),
        mRegularization(regularization), mRefEnergy(0.0f), mErrorEnergy(0.0f),
        mCoeffEnergy(0.0f), mSampleCount(0) {
    mCoeffs = new float[filterLength]();
    // Optimize: Double size for "mirrored" ring buffer to avoid modulo in inner
    // loops
    mRefHistory = new float[filterLength * 2]();
  }

  ~NLMSFilter() {
    delete[] mCoeffs;
    delete[] mRefHistory;
  }

  NLMSFilter(const NLMSFilter &) = delete;
  NLMSFilter &operator=(const NLMSFilter &) = delete;

  float process(float micSample, float refSample) {
    // Write sample to both ends of the buffer (mirrored ring buffer)
    mRefHistory[mHistoryIndex] = refSample;
    mRefHistory[mHistoryIndex + mFilterLength] = refSample;

    // FIR convolution without modulo
    // We want x[n], x[n-1], ... x[n-L+1]
    // In our double buffer, current is at `mHistoryIndex + mFilterLength`
    // History is contiguous backwards from there.
    float echoEstimate = 0.0f;
    int dataPtrBase = mHistoryIndex + mFilterLength;

// Pragma for auto-vectorization hint (widely supported)
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < mFilterLength; ++i) {
      // Read backwards: x[n-i]
      // echoEstimate += mCoeffs[i] * mRefHistory[idx];
      // Optimized:
      echoEstimate += mCoeffs[i] * mRefHistory[dataPtrBase - i];
    }

    // Error calculation
    float error = micSample - echoEstimate;

    // Track energies
    updateEnergies(refSample, error);

    // Advance index
    // Resume code with standard increment:
    mHistoryIndex = (mHistoryIndex + 1) % mFilterLength;
    return error;
  }

  // Adapt coefficients using NLMS (or NPVSS)
  // Should be called after process()
  void adapt(float error) {
    // Standard NLMS update:
    // h(n+1) = h(n) + mu * e(n) * x(n) / (x'x + epsilon)

    // Calculate variable step size (simplified NPVSS)
    // For efficiency in this loop, we use a normalized step size
    // mu_normalized = mu / (energy + regularization)
    float normalization = mRefEnergy + mRegularization;
    float currStep = mStepSize / normalization;

    // Stability check: if energy is too low, don't adapt (avoid divergence)
    if (mRefEnergy < 1e-5f)
      return;

    // Update coefficients
    // Current sample x[n] is at `(mHistoryIndex - 1) + mFilterLength`

    int prevIdx = (mHistoryIndex == 0) ? mFilterLength - 1 : mHistoryIndex - 1;
    int dataPtrBase = prevIdx + mFilterLength;

#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < mFilterLength; ++i) {
      // mCoeffs[i] += currStep * error * mRefHistory[idx];
      // Optimized:
      mCoeffs[i] += currStep * error * mRefHistory[dataPtrBase - i];
    }
  }

  void setStepSize(float stepSize) { mStepSize = stepSize; }
  void setCoefficients(const float *coeffs, int length) {
    int copyLen = std::min(length, mFilterLength);
    std::memcpy(mCoeffs, coeffs, copyLen * sizeof(float));
    if (copyLen < mFilterLength) {
      std::memset(mCoeffs + copyLen, 0,
                  (mFilterLength - copyLen) * sizeof(float));
    }
    mCoeffEnergy = getCoeffEnergy();
  }

  int getFilterLength() const { return mFilterLength; }
  const float *getCoeffs() const { return mCoeffs; }

  // Simple energy tracker for ref signal
  void updateEnergies(float refSample, float errorSample) {
    // Exponential moving average for energy
    const float alpha = 0.001f;
    // Estimate energy over the filter length (approx)
    // For NLMS normalization we ideally want dot(x,x).
    // For efficiency we can track it iteratively or approximate.
    // Iterative update of sum-squares is expensive to maintain perfectly due to
    // drift. We'll use a leaky integrator approach as a proxy for signal power.
    // Actually for True NLMS we need sum(x[n-i]^2).
    // Let's implement an efficient sliding window energy if possible,
    // or just use the leaky integrator * filterLength approximation.

    // Leaky integrator energy estimate
    float refSq = refSample * refSample;
    mRefEnergy = (1.0f - alpha) * mRefEnergy + alpha * (refSq * mFilterLength);

    mErrorEnergy += errorSample * errorSample;
    mSampleCount++;
  }

  float getCoeffEnergy() const {
    float energy = 0.0f;
    for (int i = 0; i < mFilterLength; ++i)
      energy += mCoeffs[i] * mCoeffs[i];
    return energy;
  }

  // Interface compatibility
  float getAlpha() const { return mStepSize; }

  float getEchoReturnLoss() const {
    float energy = getCoeffEnergy();
    if (energy < 1e-10f)
      return 0.0f;
    return -10.0f * std::log10(energy);
  }

  void getConvergenceMetrics(float &coeffEnergy, float &errorEnergyAvg,
                             float &coeffChangeRate) {
    coeffEnergy = getCoeffEnergy();
    errorEnergyAvg = mSampleCount > 0 ? mErrorEnergy / mSampleCount : 0.0f;
    coeffChangeRate = 0;
    mErrorEnergy = 0.0f;
    mSampleCount = 0;
  }

  void reset() {
    std::memset(mRefHistory, 0, mFilterLength * 2 * sizeof(float));
    mHistoryIndex = 0;
    mRefEnergy = 0.0f;
  }

private:
  int mFilterLength;
  float *mCoeffs;
  float *mRefHistory;
  int mHistoryIndex;

  float mStepSize;
  float mRegularization;

  float mRefEnergy; // Approximated energy of reference vector
  float mErrorEnergy;
  float mCoeffEnergy;
  size_t mSampleCount;
};

#endif // NLMS_FILTER_H
