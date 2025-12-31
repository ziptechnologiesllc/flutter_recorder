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
      2048; // ~42ms at 48kHz, matches calibration IR_LENGTH

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
   * Set the maximum step size (learning rate).
   * @param mu Max step size (0.0 to 1.0). Default is 0.5.
   */
  void setStepSize(float mu);

  /**
   * Set the smoothing factor for VSS statistics.
   * @param a Alpha (0.0 to 1.0). Default is 0.99.
   * Lower values = faster adaptation, less stable.
   */
  void setSmoothingFactor(float a);

  /**
   * Set the leakage factor.
   * @param lambda Leakage (0.0 to 1.0). Default is 0.9999.
   * 1.0 = No leakage (Standard LMS).
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

  // Tuning Parameters
  // Tuning Parameters
  float alpha = 0.95f;   // Smoothing factor (0.9 - 0.999)
  float mu_max = 0.5f;   // Max step size
  float epsilon = 1e-6f; // Small constant to prevent division by zero
  float leakage = 1.0f;  // Leakage factor (1.0 = no leakage)

  // Diagnostics
  float mLastE = 0.0f;
  float mLastStep = 0.0f;
  float mLastCorrelation = 0.0f;
  float mLastYEst = 0.0f;  // Last echo estimate for diagnostics

  // SIMD helper functions defined in cpp
  void updateHistory(float new_sample);
};

#endif // VSS_NLMS_FILTER_H
