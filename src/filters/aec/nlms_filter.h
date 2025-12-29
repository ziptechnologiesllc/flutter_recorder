#ifndef NLMS_FILTER_H
#define NLMS_FILTER_H

#include <algorithm>
#include <cmath>
#include <cstring>

/**
 * NPVSS-NLMS (Nonparametric Variable Step Size NLMS) Filter
 *
 * Based on: Benesty et al. "A Nonparametric VSS NLMS Algorithm"
 * IEEE Signal Processing Letters, Vol. 13, No. 10, October 2006
 *
 * Algorithm from Table I of paper:
 *   σ̂²ₑ(n) = λσ̂²ₑ(n-1) + (1-λ)|e(n)|²
 *   μ̃(n) = max{1 - σ²ᵥ/σ̂²ₑ(n), 0}
 *   h(n+1) = h(n) + [μ̃(n)/(||x(n)||² + δ)] * e(n) * x(n)
 *
 * Key properties:
 * - When error >> noise: μ ≈ 1 (fast convergence)
 * - When error ≈ noise: μ ≈ 0 (low misadjustment)
 * - σ²ᵥ estimated during silence periods only
 */
class NLMSFilter {
public:
  // Filter length: ~64ms @ 48kHz for acoustic echo
  static constexpr int DEFAULT_FILTER_LENGTH = 3072;

  // From paper: δ = 0.01 (regularization to avoid division by small numbers)
  static constexpr float DEFAULT_REGULARIZATION = 0.01f;

  // Stability bounds (not in paper, added for robustness)
  static constexpr float MAX_COEFF_VALUE = 2.0f;   // Maximum coefficient magnitude
  static constexpr float MAX_MU = 1.0f;            // Maximum step size

  NLMSFilter(int filterLength = DEFAULT_FILTER_LENGTH,
             float stepSize = 0.5f,  // Not used, NPVSS controls this
             float regularization = DEFAULT_REGULARIZATION)
      : mFilterLength(filterLength),
        mHistoryIndex(0),
        mDelta(regularization),
        // λ = 1 - 1/(3L) from paper Section III
        mLambda(1.0f - 1.0f / (3.0f * filterLength)),
        // NPVSS state
        mSigmaE2(0.001f),     // Smoothed error power σ̂²ₑ (initialize small)
        mSigmaV2(1e-6f),      // Noise power σ²ᵥ (estimated during silence)
        mCurrentMu(0.0f),
        mXNormSq(0.0f),       // ||x(n)||² - sum of squared reference samples
        // Metrics
        mRefEnergy(0.0f),
        mErrorEnergy(0.0f),
        mSampleCount(0) {
    mCoeffs = new float[filterLength]();
    // Double-size ring buffer for efficient convolution
    mRefHistory = new float[filterLength * 2]();
  }

  ~NLMSFilter() {
    delete[] mCoeffs;
    delete[] mRefHistory;
  }

  NLMSFilter(const NLMSFilter &) = delete;
  NLMSFilter &operator=(const NLMSFilter &) = delete;

  /**
   * Process one sample: compute echo estimate and subtract from mic
   * @param micSample Input microphone sample d(n)
   * @param refSample Reference (playback) sample x(n)
   * @return Error signal e(n) = d(n) - ŷ(n)
   */
  float process(float micSample, float refSample) {
    // Update ||x(n)||² incrementally:
    // Remove oldest sample's contribution, add new sample's contribution
    int oldestIdx = mHistoryIndex;
    float oldestSample = mRefHistory[oldestIdx];
    mXNormSq -= oldestSample * oldestSample;
    mXNormSq += refSample * refSample;
    // Prevent negative due to floating point errors
    mXNormSq = std::max(mXNormSq, 0.0f);

    // Store reference sample in ring buffer (mirrored for efficiency)
    mRefHistory[mHistoryIndex] = refSample;
    mRefHistory[mHistoryIndex + mFilterLength] = refSample;

    // Compute echo estimate: ŷ(n) = h'(n) * x(n)
    float echoEstimate = 0.0f;
    int baseIdx = mHistoryIndex + mFilterLength;

// Vectorized FIR convolution
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < mFilterLength; ++i) {
      echoEstimate += mCoeffs[i] * mRefHistory[baseIdx - i];
    }

    // Error signal: e(n) = d(n) - ŷ(n)
    float error = micSample - echoEstimate;

    // Store for adapt()
    mLastError = error;
    mLastRefSample = refSample;

    // Update energy tracking for metrics
    mRefEnergy += refSample * refSample;
    mErrorEnergy += error * error;
    mSampleCount++;

    // Advance ring buffer index
    mHistoryIndex = (mHistoryIndex + 1) % mFilterLength;

    return error;
  }

  /**
   * Adapt filter coefficients using NPVSS-NLMS (Table I from paper)
   * Call after process() with the returned error
   */
  void adapt(float error) {
    // Update smoothed error power: σ̂²ₑ(n) = λσ̂²ₑ(n-1) + (1-λ)|e(n)|²
    float errorSq = error * error;
    mSigmaE2 = mLambda * mSigmaE2 + (1.0f - mLambda) * errorSq;

    // Compute variable step size: μ̃(n) = max{1 - σ²ᵥ/σ̂²ₑ(n), 0}
    float mu_tilde = 0.0f;
    if (mSigmaE2 > mSigmaV2) {
      mu_tilde = 1.0f - (mSigmaV2 / mSigmaE2);
      mu_tilde = std::max(mu_tilde, 0.0f);
      mu_tilde = std::min(mu_tilde, MAX_MU);
    }
    mCurrentMu = mu_tilde;

    // Don't adapt if step size is effectively zero or no reference energy
    if (mu_tilde < 1e-6f || mXNormSq < 1e-8f) {
      return;
    }

    // NLMS update: h(n+1) = h(n) + [μ̃(n)/(||x(n)||² + δ)] * e(n) * x(n)
    float normFactor = mXNormSq + mDelta;
    float adaptStep = mu_tilde / normFactor;

    // Get reference history for update
    int prevIdx = (mHistoryIndex == 0) ? mFilterLength - 1 : mHistoryIndex - 1;
    int baseIdx = prevIdx + mFilterLength;

    float scaledError = adaptStep * error;

#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < mFilterLength; ++i) {
      mCoeffs[i] += scaledError * mRefHistory[baseIdx - i];
      // Clip coefficients to prevent explosion (stability bound)
      mCoeffs[i] = std::max(-MAX_COEFF_VALUE, std::min(mCoeffs[i], MAX_COEFF_VALUE));
    }
  }

  /**
   * Set noise floor estimate σ²ᵥ
   * Should be called with measurements taken during silence periods
   */
  void setNoiseFloor(float sigmaV2) {
    mSigmaV2 = std::max(sigmaV2, 1e-10f);
  }

  void setStepSize(float stepSize) {
    // NPVSS controls step size automatically
    (void)stepSize;
  }

  void setCoefficients(const float *coeffs, int length) {
    int copyLen = std::min(length, mFilterLength);
    std::memcpy(mCoeffs, coeffs, copyLen * sizeof(float));
    if (copyLen < mFilterLength) {
      std::memset(mCoeffs + copyLen, 0,
                  (mFilterLength - copyLen) * sizeof(float));
    }
  }

  int getFilterLength() const { return mFilterLength; }
  const float *getCoeffs() const { return mCoeffs; }

  float getCoeffEnergy() const {
    float energy = 0.0f;
    for (int i = 0; i < mFilterLength; ++i) {
      energy += mCoeffs[i] * mCoeffs[i];
    }
    return energy;
  }

  // Get current variable step size (for debugging)
  float getAlpha() const { return mCurrentMu; }

  // Get noise floor estimate (for debugging)
  float getNoiseFloor() const { return mSigmaV2; }

  // Get smoothed error power (for debugging)
  float getErrorPower() const { return mSigmaE2; }

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
    coeffChangeRate = mCurrentMu;
    // Reset accumulators
    mErrorEnergy = 0.0f;
    mRefEnergy = 0.0f;
    mSampleCount = 0;
  }

  void reset() {
    std::memset(mCoeffs, 0, mFilterLength * sizeof(float));
    std::memset(mRefHistory, 0, mFilterLength * 2 * sizeof(float));
    mHistoryIndex = 0;
    mSigmaE2 = 0.001f;
    mXNormSq = 0.0f;
    mCurrentMu = 0.0f;
    mRefEnergy = 0.0f;
    mErrorEnergy = 0.0f;
    mSampleCount = 0;
  }

private:
  int mFilterLength;
  float *mCoeffs;
  float *mRefHistory;
  int mHistoryIndex;

  float mDelta;   // Regularization δ
  float mLambda;  // Smoothing factor λ = 1 - 1/(3L)

  // NPVSS state variables
  float mSigmaE2;   // Smoothed error power σ̂²ₑ(n)
  float mSigmaV2;   // Noise power estimate σ²ᵥ (set externally)
  float mCurrentMu; // Current variable step size μ̃(n)
  float mXNormSq;   // ||x(n)||² - squared norm of reference vector

  // Temporary storage for adapt()
  float mLastError = 0.0f;
  float mLastRefSample = 0.0f;

  // Metrics
  float mRefEnergy;
  float mErrorEnergy;
  size_t mSampleCount;
};

#endif // NLMS_FILTER_H
