#ifndef NLMS_FILTER_H
#define NLMS_FILTER_H

#include <algorithm>
#include <cmath>
#include <cstring>

// External logging function defined in calibration.cpp
extern void aecLog(const char *fmt, ...);

/**
 * Simple Fixed FIR Echo Filter
 *
 * Uses calibrated impulse response (no adaptation) to estimate echo.
 * Optimized for looper applications where:
 * - The acoustic path is fixed (user doesn't move)
 * - Double-talk is constant (user always singing while loop plays)
 * - Calibration provides accurate impulse response via click measurement
 */
class NLMSFilter {
public:
  // Filter length: ~170ms @ 48kHz for long reverb tails
  static constexpr int DEFAULT_FILTER_LENGTH = 8192;

  NLMSFilter(int filterLength = DEFAULT_FILTER_LENGTH,
             float stepSize = 0.0f,       // Unused, kept for API compatibility
             float regularization = 0.0f) // Unused
      : mFilterLength(filterLength), mHistoryIndex(0), mXNormSq(0.0f),
        mHasCoeffs(false) {
    mCoeffs = new float[filterLength]();
    // Double-size ring buffer for efficient convolution
    mRefHistory = new float[filterLength * 2]();
    (void)stepSize;
    (void)regularization;
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
    // Update ||x(n)||² incrementally (for debug metrics)
    int oldestIdx = mHistoryIndex;
    float oldestSample = mRefHistory[oldestIdx];
    mXNormSq -= oldestSample * oldestSample;
    mXNormSq += refSample * refSample;
    mXNormSq = std::max(mXNormSq, 0.0f);

    // Store reference sample in ring buffer (mirrored for efficiency)
    mRefHistory[mHistoryIndex] = refSample;
    mRefHistory[mHistoryIndex + mFilterLength] = refSample;

    // Compute echo estimate: ŷ(n) = h' * x(n)
    float echoEstimate = 0.0f;

    if (mHasCoeffs) {
      int baseIdx = mHistoryIndex + mFilterLength;

// Vectorized FIR convolution
#pragma clang loop vectorize(enable) interleave(enable)
      for (int i = 0; i < mFilterLength; ++i) {
        echoEstimate += mCoeffs[i] * mRefHistory[baseIdx - i];
      }
    }

    // Adaptive gain control: prevent over-cancellation and eliminate "burbling"
    // artifacts. We use slow energy tracking to estimate the relationship
    // between echo and mic signals.
    constexpr float ENERGY_SMOOTH =
        0.0001f; // ~200ms @ 48kHz for power averaging
    mMicEnergy = (1.0f - ENERGY_SMOOTH) * mMicEnergy +
                 ENERGY_SMOOTH * micSample * micSample;
    mEchoEstEnergy = (1.0f - ENERGY_SMOOTH) * mEchoEstEnergy +
                     ENERGY_SMOOTH * echoEstimate * echoEstimate;

    // Calculate target gain: limit echo estimate so it doesn't exceed mic
    // energy. This handles double-talk and imperfect filter coefficients.
    float targetGain = 1.0f;
    if (mEchoEstEnergy > 1e-9f && mMicEnergy > 1e-9f) {
      float energyRatio = std::sqrt(mMicEnergy / mEchoEstEnergy);
      targetGain = std::min(1.0f, energyRatio);
    } else if (mEchoEstEnergy > mMicEnergy && mEchoEstEnergy > 1e-9f) {
      // If mic is silent but we estimate echo, be conservative
      targetGain = 0.0f;
    }

    // Smooth the gain factor itself to eliminate rapid amplitude modulation
    // (ringmod-like sound). Using slow smoothing prevents the gain from
    // following the signal envelope.
    constexpr float GAIN_SMOOTH =
        0.0002f; // ~100ms smoothing for the gain factor itself
    mAppliedGain =
        (1.0f - GAIN_SMOOTH) * mAppliedGain + GAIN_SMOOTH * targetGain;

    // Apply smoothed gain to echo estimate
    float adaptedEchoEstimate = mAppliedGain * echoEstimate;

    // Debug: track echo estimate
    mLastEchoEstimate = adaptedEchoEstimate;

    // Error signal: e(n) = d(n) - ŷ(n)
    float error = micSample - adaptedEchoEstimate;

    // Advance ring buffer index
    mHistoryIndex = (mHistoryIndex + 1) % mFilterLength;

    return error;
  }

  /**
   * No-op adaptation (kept for API compatibility)
   */
  void adapt(float error) {
    (void)error;
    // No adaptation - we use fixed calibrated coefficients
  }

  void setNoiseFloor(float sigmaV2) { (void)sigmaV2; }
  void setStepSize(float stepSize) { (void)stepSize; }

  /**
   * Set coefficients from calibration (impulse response).
   */
  void setCoefficients(const float *coeffs, int length) {
    int copyLen = std::min(length, mFilterLength);
    std::memcpy(mCoeffs, coeffs, copyLen * sizeof(float));
    if (copyLen < mFilterLength) {
      std::memset(mCoeffs + copyLen, 0,
                  (mFilterLength - copyLen) * sizeof(float));
    }
    mHasCoeffs = true;

    // Debug: print coefficient stats and first few values
    float energy = 0.0f;
    float peak = 0.0f;
    int peakIdx = 0;
    for (int i = 0; i < copyLen; ++i) {
      energy += coeffs[i] * coeffs[i];
      if (std::abs(coeffs[i]) > peak) {
        peak = std::abs(coeffs[i]);
        peakIdx = i;
      }
    }
    aecLog(
        "[FIR Filter] Set %d coefficients, energy=%.4f, peak=%.4f at idx %d\n",
        copyLen, energy, peak, peakIdx);
    aecLog("[FIR Filter] First 10 coeffs: ");
    for (int i = 0; i < std::min(10, copyLen); ++i) {
      aecLog("%.4f ", coeffs[i]);
    }
    aecLog("\n");
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

  // API compatibility methods (return fixed values)
  float getAlpha() const { return 0.0f; } // No adaptation
  float getNoiseFloor() const { return 0.0f; }
  float getErrorPower() const { return 0.0f; }
  float getLastEchoEstimate() const { return mLastEchoEstimate; }
  float getRefHistoryEnergy() const { return mXNormSq; }

  float getEchoReturnLoss() const {
    float energy = getCoeffEnergy();
    if (energy < 1e-10f)
      return 0.0f;
    return -10.0f * std::log10(energy);
  }

  void getConvergenceMetrics(float &coeffEnergy, float &errorEnergyAvg,
                             float &coeffChangeRate) {
    coeffEnergy = getCoeffEnergy();
    errorEnergyAvg = 0.0f;
    coeffChangeRate = 0.0f; // No adaptation
  }

  void reset() {
    // Keep coefficients, just clear history
    std::memset(mRefHistory, 0, mFilterLength * 2 * sizeof(float));
    mHistoryIndex = 0;
    mXNormSq = 0.0f;
  }

  /**
   * Resize the filter to a new length.
   * This reallocates internal buffers and resets state.
   */
  void resize(size_t newLength) {
    if (newLength == static_cast<size_t>(mFilterLength)) {
      return;
    }

    // Deallocate old buffers
    delete[] mCoeffs;
    delete[] mRefHistory;

    // Allocate new buffers
    mFilterLength = static_cast<int>(newLength);
    mCoeffs = new float[mFilterLength]();
    mRefHistory = new float[mFilterLength * 2]();

    // Reset state
    mHistoryIndex = 0;
    mXNormSq = 0.0f;
    mHasCoeffs = false;
  }

private:
  int mFilterLength;
  float *mCoeffs;
  float *mRefHistory;
  int mHistoryIndex;
  float mXNormSq; // Reference energy for debug
  bool mHasCoeffs;
  float mLastEchoEstimate = 0.0f;

  // Adaptive gain control state
  float mMicEnergy = 0.0f;     // Smoothed mic energy
  float mEchoEstEnergy = 0.0f; // Smoothed echo estimate energy
  float mAppliedGain = 1.0f;   // Transitioned gain factor
};

#endif // NLMS_FILTER_H
