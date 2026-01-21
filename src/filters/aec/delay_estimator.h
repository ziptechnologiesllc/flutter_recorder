#ifndef DELAY_ESTIMATOR_H
#define DELAY_ESTIMATOR_H

#include <cstddef>
#include <vector>

/**
 * Delay Estimator.
 *
 * Calculates the time delay (lag) between a reference signal and a microphone
 * signal using Normalized Cross-Correlation (NCC).
 *
 * Usage:
 * 1. Capture ~1-2 seconds of Reference audio and Microphone audio while playing
 * a loud, distinct sound.
 * 2. Pass both buffers to `estimateDelay`.
 * 3. Use the returned lag to align the buffers before starting AEC.
 *
 * Key Features:
 * - SIMD Optimized (NEON/AVX2) for fast correlation calculation.
 * - Sub-sample precision (conceptually, though returns integer samples for
 * now).
 * - Robust to gain differences (Normalized).
 */
class DelayEstimator {
public:
  /**
   * Estimate the delay (lag) of the mic signal relative to the reference
   * signal.
   *
   * Positive lag means Mic is BEHIND Reference (Mic = Ref[t - lag]).
   * This is the standard physical case (speaker -> air -> mic).
   *
   * @param ref_signal  The reference (speaker) signal.
   * @param mic_signal  The microphone signal.
   * @param max_lag     Maximum lag to search for (in samples). E.g., 24000 for
   * 500ms at 48kHz. If 0, defaults to half the buffer size.
   * @return            The estimated lag in samples.
   */
  static int estimateDelay(const std::vector<float> &ref_signal,
                           const std::vector<float> &mic_signal,
                           int max_lag = 0);

  /**
   * Estimate delay with a targeted search around a theoretical value.
   * This is more robust than full-range search as it avoids false peaks.
   *
   * @param ref_signal       The reference (speaker) signal.
   * @param mic_signal       The microphone signal.
   * @param centerLag        Expected delay (theoretical or from previous calibration)
   * @param searchWindow     How far to search on either side of centerLag (samples)
   * @return                 The estimated lag in samples.
   */
  static int estimateDelayTargeted(const std::vector<float> &ref_signal,
                                   const std::vector<float> &mic_signal,
                                   int centerLag,
                                   int searchWindow = 480); // ±10ms at 48kHz

private:
  // Helper for SIMD dot product
  static float dotProduct(const float *a, const float *b, size_t length);
};

#endif // DELAY_ESTIMATOR_H
