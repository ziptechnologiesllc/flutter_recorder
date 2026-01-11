#ifndef VSS_NLMS_FILTER_H
#define VSS_NLMS_FILTER_H

#include <cstddef>
#include <vector>

/**
 * Variable Step-Size Normalized Least Mean Squares (VSS-NLMS) Filter.
 *
 * This adaptive filter is designed for Acoustic Echo Cancellation (AEC) in
 * double-talk scenarios (e.g., loop station where musician plays over a loop).
 *
 * Key Features:
 * - Correlation-based Step Size: Detects double-talk by monitoring the
 *   cross-correlation between the error signal (mic) and reference (loop).
 *   - High Correlation -> Echo dominates -> Fast Adaptation
 *   - Low Correlation -> Double-talk (Instrument) dominates -> Freeze
 * Adaptation
 *
 * - SIMD Optimization: Uses ARM NEON (Mobile/Mac) or AVX2 (Desktop) for
 *   accelerated convolution and weight updates. 'Chad' mode enabled.
 */
class VssNlmsFilter {
public:
  static constexpr int DEFAULT_FILTER_LENGTH =
      8192; // ~170ms at 48kHz, covers long reverb tails

  /**
   * @param taps Filter length in samples. Will be rounded up to nearest
   * multiple of 8 for SIMD.
   */
  VssNlmsFilter(size_t taps = DEFAULT_FILTER_LENGTH);
  ~VssNlmsFilter() = default;

  /**
   * Process a single sample.
   *
   * @param aligned_ref The reference sample (x) from the delay line, aligned in
   * time with the echo.
   * @param mic_input   The microphone sample (d) containing echo + near-end
   * signal.
   * @return            The clean signal (e) = mic_input - estimated_echo.
   */
  float processSample(float aligned_ref, float mic_input);

  /**
   * Reset filter weights and history.
   */
  void reset();

  /**
   * Resize the filter to a new length.
   * Resets weights and history. Length will be rounded up to multiple of 8.
   * @param newLength New filter length in samples.
   */
  void resize(size_t newLength);

  /**
   * Get the current filter length.
   * @return Filter length in samples.
   */
  size_t getFilterLength() const { return filter_length; }

  /**
   * Set the maximum step size (learning rate).
   * @param mu Max step size (0.0 to 2.0). Default is 1.2.
   */
  void setStepSize(float mu);

  /**
   * Set the smoothing factor for VSS statistics.
   * @param a Alpha (0.0 to 1.0). Default is 0.05.
   * Lower values = faster adaptation to transients.
   */
  void setSmoothingFactor(float a);

  /**
   * Set the leakage factor.
   * @param lambda Leakage (0.0 to 1.0). Default is 0.9999.
   * 1.0 = No leakage (Standard LMS). Lower values add stability.
   */
  void setLeakage(float lambda);

  /**
   * Set the filter weights directly (e.g. from calibration).
   * @param coeffs The new weights.
   * @param count  Number of weights to copy (up to filter_length).
   */
  void setWeights(const float *coeffs, size_t count);

  /**
   * Get the current filter weights.
   * @return Copy of the weights vector.
   */
  std::vector<float> getWeights() const;

  /**
   * Train the filter on a known reference/mic pair (Offline Learning).
   * Used for "Warm Start" calibration using a chirp signal.
   *
   * @param ref_signal Reference signal buffer.
   * @param mic_signal Microphone signal buffer (must be aligned).
   */
  void warmStartWeights(const std::vector<float> &ref_signal,
                        const std::vector<float> &mic_signal);

  // Diagnostics
  float getCoeffEnergy() const;
  float getLastError() const { return mLastE; }
  float getLastStepSize() const { return mLastStep; }
  float getLastCorrelation() const { return mLastCorrelation; }
  float getLastEchoEstimate() const { return mLastYEst; }

  // Parameter getters for experimentation
  float getMuMax() const { return mu_max; }
  float getAlpha() const { return alpha; }
  float getLeakage() const { return leakage; }
  float getEpsilon() const { return epsilon; }

private:
  // SIMD helpers require aligned memory or careful handling.
  // We use standard vectors but handle unaligned loads safely in the
  // implementation.
  std::vector<float> weights;
  std::vector<float> x_history;
  size_t filter_length;

  // VSS Statistics
  float p_est = 0.0f; // Cross-correlation estimate (smoothed)
  float var_x = 0.0f; // Power of Reference (smoothed)
  float var_e = 0.0f; // Power of Error (smoothed)

  // Tuning Parameters (optimized via sweep test - 9.05dB cancellation)
  float alpha =
      0.05f; // Smoothing factor (lower = faster tracking for transients)
  float mu_max = 1.2f;    // Max step size (higher = faster convergence)
  float epsilon = 1e-6f;  // Small constant to prevent division by zero
  float leakage = 0.9999f; // Leakage factor (slight decay for stability)

  // Diagnostics
  float mLastE = 0.0f;
  float mLastStep = 0.0f;
  float mLastCorrelation = 0.0f;
  float mLastYEst = 0.0f; // Last echo estimate for diagnostics

  // SIMD helper functions defined in cpp
  void updateHistory(float new_sample);
};

#endif // VSS_NLMS_FILTER_H
