#include "delay_estimator.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// External logging function defined in calibration.cpp
extern void aecLog(const char *fmt, ...);

// ==========================================
// SIMD Architecture Detection & Includes
// ==========================================
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define USE_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX
#endif

// ==========================================
// SIMD Helpers (in anonymous namespace to avoid ODR violations)
// ==========================================

namespace {

#ifdef USE_NEON
inline float hsum_float32x4_de(float32x4_t v) {
  float32x2_t high = vget_high_f32(v);
  float32x2_t low = vget_low_f32(v);
  float32x2_t sum = vpadd_f32(low, high);
  return vget_lane_f32(vpadd_f32(sum, sum), 0);
}
#endif

#ifdef USE_AVX
inline float hsum_float256_de(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 sum = _mm_add_ps(hi, lo);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}
#endif

} // anonymous namespace

// ==========================================
// Implementation
// ==========================================

float DelayEstimator::dotProduct(const float *a, const float *b,
                                 size_t length) {
  float result = 0.0f;
  size_t i = 0;

#ifdef USE_NEON
  float32x4_t v_sum = vdupq_n_f32(0.0f);
  for (; i + 4 <= length; i += 4) {
    float32x4_t v_a = vld1q_f32(a + i);
    float32x4_t v_b = vld1q_f32(b + i);
    v_sum = vmlaq_f32(v_sum, v_a, v_b);
  }
  result = hsum_float32x4_de(v_sum);
#elif defined(USE_AVX)
  __m256 v_sum = _mm256_setzero_ps();
  for (; i + 8 <= length; i += 8) {
    __m256 v_a = _mm256_loadu_ps(a + i); // loadu handles unaligned
    __m256 v_b = _mm256_loadu_ps(b + i);
    v_sum = _mm256_add_ps(v_sum, _mm256_mul_ps(v_a, v_b));
  }
  result = hsum_float256_de(v_sum);
#endif

  // Handle tail
  for (; i < length; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

int DelayEstimator::estimateDelay(const std::vector<float> &ref_signal,
                                  const std::vector<float> &mic_signal,
                                  int max_lag) {
  if (ref_signal.empty() || mic_signal.empty())
    return 0;

  size_t n = std::min(ref_signal.size(), mic_signal.size());
  if (n < 256)
    return 0; // Too short to estimate reliably

  if (max_lag <= 0) {
    // 500ms @ 48kHz - Android can have very high audio latency (300-400ms)
    // This covers typical Android AAudio/OpenSL latency plus acoustic delay
    max_lag = 24000;
  }
  if (max_lag > static_cast<int>(n) - 1) {
    max_lag = static_cast<int>(n) - 1;
  }

  // =====================================================
  // NORMALIZED CROSS-CORRELATION (Python-compatible)
  // This matches the algorithm in analyze_aec.py:
  // - Searches BOTH positive and negative lags
  // - Uses ABSOLUTE correlation for peak finding
  // - Handles phase inversion (negative correlation)
  // =====================================================

  // Step 1: Calculate means
  double refMean = 0, micMean = 0;
  for (size_t i = 0; i < n; ++i) {
    refMean += ref_signal[i];
    micMean += mic_signal[i];
  }
  refMean /= n;
  micMean /= n;

  // Step 2: Calculate standard deviations
  double refStd = 0, micStd = 0;
  for (size_t i = 0; i < n; ++i) {
    refStd += (ref_signal[i] - refMean) * (ref_signal[i] - refMean);
    micStd += (mic_signal[i] - micMean) * (mic_signal[i] - micMean);
  }
  refStd = std::sqrt(refStd / n);
  micStd = std::sqrt(micStd / n);

  if (refStd < 1e-10 || micStd < 1e-10) {
    aecLog("[DelayEstimator] Insufficient signal variance (ref=%.6f mic=%.6f)\n",
           refStd, micStd);
    return 0;
  }

  aecLog("[DelayEstimator] Signal stats: ref mean=%.4f std=%.4f, mic mean=%.4f std=%.4f\n",
         refMean, refStd, micMean, micStd);

  // Step 3: Create normalized copies (like Python)
  std::vector<float> refNorm(n), micNorm(n);
  for (size_t i = 0; i < n; ++i) {
    refNorm[i] = (ref_signal[i] - refMean) / refStd;
    micNorm[i] = (mic_signal[i] - micMean) / micStd;
  }

  // Step 4: Search BOTH positive and negative lags
  // Like Python's scipy.signal.correlate(mode='full')
  // Use ABSOLUTE correlation for peak finding (handles phase inversion)
  double maxAbsCorr = 0;
  int bestLag = 0;
  double bestCorrSigned = 0;

  // Positive lags: mic is delayed relative to ref (expected for echo)
  for (int tau = 0; tau < max_lag && tau < static_cast<int>(n); ++tau) {
    size_t len = n - tau;
    if (len < 128) break;

    double sum = 0;
    for (size_t i = 0; i < len; ++i) {
      sum += refNorm[i] * micNorm[i + tau];
    }
    double corr = sum / len;

    if (std::abs(corr) > maxAbsCorr) {
      maxAbsCorr = std::abs(corr);
      bestCorrSigned = corr;
      bestLag = tau;
    }
  }

  // Negative lags: ref is delayed relative to mic
  for (int tau = 1; tau < max_lag && tau < static_cast<int>(n); ++tau) {
    size_t len = n - tau;
    if (len < 128) break;

    double sum = 0;
    for (size_t i = 0; i < len; ++i) {
      sum += refNorm[i + tau] * micNorm[i];
    }
    double corr = sum / len;

    if (std::abs(corr) > maxAbsCorr) {
      maxAbsCorr = std::abs(corr);
      bestCorrSigned = corr;
      bestLag = -tau;  // Negative lag
    }
  }

  aecLog("[DelayEstimator] Best lag %d samples (%.2fms), correlation=%.4f (signed=%.4f)\n",
         bestLag, bestLag * 1000.0 / 48000.0, maxAbsCorr, bestCorrSigned);

  return bestLag;
}

int DelayEstimator::estimateDelayTargeted(const std::vector<float> &ref_signal,
                                          const std::vector<float> &mic_signal,
                                          int centerLag,
                                          int searchWindow) {
  if (ref_signal.empty() || mic_signal.empty())
    return centerLag;

  size_t n = std::min(ref_signal.size(), mic_signal.size());
  if (n < 256)
    return centerLag;

  // Define search range around center
  int minLag = std::max(0, centerLag - searchWindow);
  int maxLag = std::min(static_cast<int>(n) - 128, centerLag + searchWindow);

  if (minLag >= maxLag)
    return centerLag;

  aecLog("[DelayEstimator Targeted] Searching around %d samples (±%d), range [%d, %d]\n",
         centerLag, searchWindow, minLag, maxLag);

  // Calculate means for normalization
  double refMean = 0, micMean = 0;
  for (size_t i = 0; i < n; ++i) {
    refMean += ref_signal[i];
    micMean += mic_signal[i];
  }
  refMean /= n;
  micMean /= n;

  // Calculate standard deviations
  double refStd = 0, micStd = 0;
  for (size_t i = 0; i < n; ++i) {
    refStd += (ref_signal[i] - refMean) * (ref_signal[i] - refMean);
    micStd += (mic_signal[i] - micMean) * (mic_signal[i] - micMean);
  }
  refStd = std::sqrt(refStd / n);
  micStd = std::sqrt(micStd / n);

  if (refStd < 1e-10 || micStd < 1e-10) {
    aecLog("[DelayEstimator Targeted] Insufficient variance, returning center\n");
    return centerLag;
  }

  // Create normalized copies
  std::vector<float> refNorm(n), micNorm(n);
  for (size_t i = 0; i < n; ++i) {
    refNorm[i] = (ref_signal[i] - refMean) / refStd;
    micNorm[i] = (mic_signal[i] - micMean) / micStd;
  }

  // Search only within window around center
  double maxAbsCorr = 0;
  int bestLag = centerLag;
  double bestCorrSigned = 0;

  for (int tau = minLag; tau <= maxLag; ++tau) {
    size_t len = n - tau;
    if (len < 128) continue;

    double sum = 0;
    for (size_t i = 0; i < len; ++i) {
      sum += refNorm[i] * micNorm[i + tau];
    }
    double corr = sum / len;

    if (std::abs(corr) > maxAbsCorr) {
      maxAbsCorr = std::abs(corr);
      bestCorrSigned = corr;
      bestLag = tau;
    }
  }

  aecLog("[DelayEstimator Targeted] Best lag %d samples (%.2fms), corr=%.4f, delta from center=%d\n",
         bestLag, bestLag * 1000.0 / 48000.0, maxAbsCorr, bestLag - centerLag);

  return bestLag;
}
