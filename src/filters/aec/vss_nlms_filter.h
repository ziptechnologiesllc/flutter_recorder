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
      4096; // ~85ms at 48kHz, covers direct path + early reflections + room tail

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
   * Freeze/unfreeze weight adaptation.
   * When frozen, the filter performs pure FIR convolution with no updates.
   * Use after calibration for stable, predictable echo cancellation.
   */
  void setFrozen(bool frozen) { mFrozen = frozen; }
  bool isFrozen() const { return mFrozen; }

  /**
   * Raw adaptation mode (for the shadow filter's background path).
   * When true the double-talk gate is bypassed (always adapts at full mu_max)
   * and leakage is disabled — the reckless "offline NLMS" regime that
   * converges fastest. The shadow architecture handles double-talk safety by
   * only promoting the background when it measurably beats the foreground,
   * so the background itself never needs to be cautious.
   */
  void setRawAdaptation(bool raw) { mRawAdapt = raw; }
  bool isRawAdaptation() const { return mRawAdapt; }

  /**
   * Get the current filter weights.
   * @return Copy of the weights vector.
   */
  std::vector<float> getWeights() const;

  /**
   * Zero-alloc read access to the live weights (for the shadow-filter EMA blend
   * and promotion copy on the audio thread — getWeights() allocates a vector).
   */
  const std::vector<float> &getWeightsRef() const { return weights; }

  /**
   * Like setWeights() but with NO getCoeffEnergy() scan and NO logging — for the
   * per-block EMA promotion path that runs ~23x/sec on the audio thread, where
   * the chatty setWeights() (printf + full 4096-tap energy loop) is an RT hazard.
   */
  void setWeightsQuiet(const float *coeffs, size_t count);

  /**
   * Score an EXTERNAL weight vector out-of-sample against THIS filter's current
   * reference history (no history shift, no adaptation, no stats). Returns the
   * residual mic - (w · x_history). Used by the shadow filter to measure the
   * candidate (an EMA of the background weights) on audio it did not adapt to,
   * so the error we gate promotion on is the error of the exact vector we
   * promote. *outBlown is set true if the prediction is non-finite or |y|>8
   * (the candidate bypasses processSample's in-filter divergence guard, so this
   * restores that guarantee for the promoted vector).
   * @param w   External weights, must be at least filter_length long.
   * @param mic_input Microphone sample (d).
   * @param outBlown Optional out-param flagged on divergence; may be nullptr.
   */
  float scoreAgainstHistory(const float *w, float mic_input,
                            bool *outBlown) const;

  /**
   * Return the MEAN annealed step ratio (mu_eff / mu_max) accumulated since the
   * last call, and reset the accumulator. A value near the anneal floor means
   * the filter has SETTLED (var_e is a small fraction of mic energy). The
   * shadow filter folds the background into its EMA only once this is low, so
   * the high-misadjustment early iterates stay out of the average. Summarising
   * over the whole block (not point-sampling mLastStep once per 2048 samples)
   * avoids aliasing on the per-sample step.
   */
  float consumeBlockMuRatio();

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
  std::vector<float> x_history; // 2x filter_length; live window starts at hist_head
  size_t hist_head = 0;         // offset of the newest sample (see updateHistory)
  size_t filter_length;

  // VSS Statistics
  float p_est = 0.0f; // Cross-correlation estimate (smoothed)
  float var_x = 0.0f; // Power of Reference (smoothed)
  float var_e = 0.0f; // Power of Error (smoothed)
  float var_mic = 0.0f; // Power of Mic input (smoothed) — drives the
                        // convergence-annealed step size in raw mode
  float mEnergyXAvg = 0.0f; // smoothed input energy — regularises the NLMS
                            // step so quiet passages can't make it explode

  // Tuning Parameters
  float alpha =
      0.05f; // Smoothing factor (lower = faster tracking for transients)
  float mu_max = 0.5f;    // Max step size (offline NLMS converged to ~12dB at this rate)
  float epsilon = 1e-6f;  // Small constant to prevent division by zero
  float leakage = 0.9999f; // Leakage factor (slight decay for stability)

  // Double-talk freeze threshold on correlation_metric (squared error/ref
  // correlation). An offline NLMS that adapts every sample converges to
  // ~12dB on this echo, so the previous DTD gating (which froze on a
  // moderate echo) was the real blocker — NLMS is inherently robust to
  // uncorrelated near-end. Keep only a last-resort freeze for essentially
  // pure near-end / silence (metric ~0).
  float mCorrelationThreshold = 0.01f;

  // Diagnostics
  float mLastE = 0.0f;
  float mLastStep = 0.0f;
  float mLastCorrelation = 0.0f;
  float mLastYEst = 0.0f; // Last echo estimate for diagnostics

  // Per-block accumulation of the annealed step ratio (mu_eff/mu_max), drained
  // by consumeBlockMuRatio(). Only accumulated while adapting (!mFrozen).
  float mBlockMuSum = 0.0f;
  int mBlockMuCount = 0;

  // Freeze flag - when true, no weight updates occur (pure FIR mode)
  bool mFrozen = false;

  // Raw adaptation - when true, double-talk gate + leakage are bypassed
  // (background/shadow filter mode).
  bool mRawAdapt = false;

  // SIMD helper functions defined in cpp
  void updateHistory(float new_sample);
};

#endif // VSS_NLMS_FILTER_H
