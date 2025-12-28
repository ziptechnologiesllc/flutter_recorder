#ifndef NLMS_FILTER_H
#define NLMS_FILTER_H

#include <cmath>
#include <cstring>
#include <algorithm>

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
    static constexpr int DEFAULT_FILTER_LENGTH = 2048;  // ~43ms @ 48kHz
    static constexpr float DEFAULT_REGULARIZATION = 1e-4f;  // Unused, kept for compatibility

    /**
     * @param filterLength Number of filter taps (L)
     * @param stepSize Unused - kept for interface compatibility
     * @param regularization Unused - kept for interface compatibility
     */
    NLMSFilter(int filterLength = DEFAULT_FILTER_LENGTH,
               float stepSize = 1.0f,
               float regularization = DEFAULT_REGULARIZATION)
        : mFilterLength(filterLength),
          mHistoryIndex(0),
          mErrorEnergy(0.0f),
          mPrevCoeffEnergy(0.0f),
          mCoeffChangeRate(0.0f),
          mSampleCount(0) {
        // Allocate and zero-initialize buffers
        mCoeffs = new float[filterLength]();
        mRefHistory = new float[filterLength]();
    }

    ~NLMSFilter() {
        delete[] mCoeffs;
        delete[] mRefHistory;
    }

    // Prevent copying (owns raw pointers)
    NLMSFilter(const NLMSFilter&) = delete;
    NLMSFilter& operator=(const NLMSFilter&) = delete;

    /**
     * Process a single sample through the static FIR filter.
     *
     * @param micSample Microphone input sample d(n) (contains speech + echo)
     * @param refSample Reference sample x(n) (speaker output)
     * @return Echo-cancelled output sample e(n)
     */
    float process(float micSample, float refSample) {
        // Store reference in history (circular buffer)
        mRefHistory[mHistoryIndex] = refSample;

        // Calculate echo estimate ŷ(n) using FIR filter (convolution)
        // ŷ(n) = Σ h[i] * x[n-i]
        float echoEstimate = 0.0f;
        for (int i = 0; i < mFilterLength; ++i) {
            int idx = (mHistoryIndex - i + mFilterLength) % mFilterLength;
            echoEstimate += mCoeffs[i] * mRefHistory[idx];
        }

        // Error signal e(n) = d(n) - ŷ(n)
        // This is the echo-cancelled output
        float error = micSample - echoEstimate;

        // Track error energy for metrics (no adaptation)
        mErrorEnergy += error * error;
        mSampleCount++;

        // Advance history index
        mHistoryIndex = (mHistoryIndex + 1) % mFilterLength;

        return error;
    }

    /**
     * Process a block of samples (more efficient for batch processing).
     */
    void processBlock(float* micSamples, const float* refSamples, size_t sampleCount) {
        for (size_t i = 0; i < sampleCount; ++i) {
            micSamples[i] = process(micSamples[i], refSamples[i]);
        }
    }

    /**
     * Reset filter state (history only - coefficients preserved).
     */
    void reset() {
        std::memset(mRefHistory, 0, mFilterLength * sizeof(float));
        mHistoryIndex = 0;
        mErrorEnergy = 0.0f;
        mSampleCount = 0;
    }

    /**
     * Set the adaptation step size.
     * Note: Static FIR doesn't adapt - this is kept for interface compatibility.
     */
    void setStepSize(float stepSize) {
        (void)stepSize;  // Unused
    }

    /**
     * Set filter coefficients from calibration impulse response.
     * This is the primary way to configure the filter.
     */
    void setCoefficients(const float* coeffs, int length) {
        int copyLen = std::min(length, mFilterLength);
        std::memcpy(mCoeffs, coeffs, copyLen * sizeof(float));
        if (copyLen < mFilterLength) {
            std::memset(mCoeffs + copyLen, 0, (mFilterLength - copyLen) * sizeof(float));
        }
        mPrevCoeffEnergy = getCoeffEnergy();
    }

    /**
     * Get the current step size.
     * Static FIR always returns 0 (no adaptation).
     */
    float getStepSize() const { return 0.0f; }

    /**
     * Get the filter length.
     */
    int getFilterLength() const { return mFilterLength; }

    /**
     * Get current filter coefficients (for debugging/visualization).
     */
    const float* getCoeffs() const { return mCoeffs; }

    /**
     * Get estimated echo return loss in dB.
     */
    float getEchoReturnLoss() const {
        float energy = getCoeffEnergy();
        if (energy < 1e-10f) return 0.0f;
        return -10.0f * std::log10(energy);
    }

    /**
     * Get coefficient energy (sum of squared coefficients).
     */
    float getCoeffEnergy() const {
        float energy = 0.0f;
        for (int i = 0; i < mFilterLength; ++i) {
            energy += mCoeffs[i] * mCoeffs[i];
        }
        return energy;
    }

    /**
     * Get convergence metrics and reset accumulators.
     */
    void getConvergenceMetrics(float& coeffEnergy, float& errorEnergyAvg, float& coeffChangeRate) {
        coeffEnergy = getCoeffEnergy();
        errorEnergyAvg = mSampleCount > 0 ? mErrorEnergy / mSampleCount : 0.0f;
        coeffChangeRate = 0.0f;  // Static FIR - no coefficient changes
        mErrorEnergy = 0.0f;
        mSampleCount = 0;
    }

    /**
     * Check if filter has valid coefficients.
     */
    bool isConverged(float threshold = 0.001f) const {
        (void)threshold;
        // Static FIR is "converged" if coefficients have been set
        return getCoeffEnergy() > 1e-6f;
    }

    /**
     * Get the current alpha (step-size) for debugging.
     * Static FIR always returns 0.
     */
    float getAlpha() const { return 0.0f; }

private:
    int mFilterLength;

    float* mCoeffs;       // Filter coefficients h(n) - from calibration
    float* mRefHistory;   // Reference signal history x(n) (circular buffer)
    int mHistoryIndex;    // Current position in history buffer

    // Metrics tracking
    float mErrorEnergy;
    float mPrevCoeffEnergy;
    float mCoeffChangeRate;
    size_t mSampleCount;
};

#endif // NLMS_FILTER_H
