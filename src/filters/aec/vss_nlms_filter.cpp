#include "vss_nlms_filter.h"
#include <algorithm>
#include <cmath>
#include <cstring>

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
// SIMD Helper Functions (in anonymous namespace to avoid ODR violations)
// ==========================================

namespace {

#ifdef USE_NEON
// Horizontal sum for float32x4_t
inline float hsum_float32x4_nlms(float32x4_t v) {
  float32x2_t high = vget_high_f32(v);
  float32x2_t low = vget_low_f32(v);
  float32x2_t sum = vpadd_f32(low, high);
  return vget_lane_f32(vpadd_f32(sum, sum), 0);
}
#endif

#ifdef USE_AVX
// Horizontal sum for __m256
inline float hsum_float256_nlms(__m256 v) {
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
// Class Implementation
// ==========================================

VssNlmsFilter::VssNlmsFilter(size_t taps) : filter_length(taps) {
  // Round up to nearest multiple of 8 for safe SIMD unrolling
  if (filter_length % 8 != 0) {
    filter_length = ((filter_length + 7) / 8) * 8;
  }

  // Resize vectors and initialize to zero
  // Note: Standard vector allocation might not be 32-byte aligned,
  // so we use unaligned load/store intrinsics (loadu/storeu) which are fast on
  // modern CPUs.
  weights.resize(filter_length, 0.0f);
  x_history.resize(filter_length, 0.0f);
}

void VssNlmsFilter::reset() {
  std::fill(weights.begin(), weights.end(), 0.0f);
  std::fill(x_history.begin(), x_history.end(), 0.0f);
  p_est = 0.0f;
  var_x = 0.0f;
  var_e = 0.0f;
  mLastE = 0.0f;
  mLastStep = 0.0f;
}

void VssNlmsFilter::resize(size_t newLength) {
  // Round up to nearest multiple of 8 for SIMD
  if (newLength % 8 != 0) {
    newLength = ((newLength + 7) / 8) * 8;
  }

  filter_length = newLength;

  // Resize and reinitialize vectors
  weights.resize(filter_length);
  x_history.resize(filter_length);

  // Reset all state
  reset();
}

void VssNlmsFilter::setStepSize(float mu) {
  mu_max = std::max(
      0.0f, std::min(mu, 2.0f)); // Allow up to 2.0 for aggressive tuning
}

void VssNlmsFilter::setSmoothingFactor(float a) {
  alpha = std::max(0.0f, std::min(a, 1.0f));
}

void VssNlmsFilter::setLeakage(float lambda) {
  leakage = std::max(0.0f, std::min(lambda, 1.0f));
}

void VssNlmsFilter::setWeights(const float *coeffs, size_t count) {
  if (!coeffs)
    return;
  size_t copy_len = std::min(count, filter_length);
  std::memcpy(weights.data(), coeffs, copy_len * sizeof(float));
  // Zero out the rest if any
  if (copy_len < filter_length) {
    std::fill(weights.begin() + copy_len, weights.end(), 0.0f);
  }
}

float VssNlmsFilter::getCoeffEnergy() const {
  float energy = 0.0f;
  for (float w : weights) {
    energy += w * w;
  }
  return energy;
}

void VssNlmsFilter::updateHistory(float new_sample) {
  // Shift history: x[n] -> x[n+1]
  // memmove is generally highly optimized.
  // For a circular buffer approach, we'd need more logic, but for < 4096 taps,
  // this is negligible.
  if (filter_length > 1) {
    std::memmove(&x_history[1], &x_history[0],
                 (filter_length - 1) * sizeof(float));
  }
  x_history[0] = new_sample;
}

float VssNlmsFilter::processSample(float aligned_ref, float mic_input) {
  // 1. Update Reference History
  updateHistory(aligned_ref);

  float y_est = 0.0f;
  float energy_x_inst = 0.0f;

  // ============================================
  // 2. SIMD CONVOLUTION (Prediction)
  // ============================================
  // Calculate y_est = weights * x_history
  // and energy_x_inst = x_history * x_history

  // Raw pointers for speed
  const float *p_x = x_history.data();
  float *p_w =
      weights.data(); // we will modify weights later, but for conv it reads

#ifdef USE_NEON
  float32x4_t v_y_est = vdupq_n_f32(0.0f);
  float32x4_t v_energy = vdupq_n_f32(0.0f);

  for (size_t i = 0; i < filter_length; i += 4) {
    float32x4_t v_x = vld1q_f32(p_x + i);
    float32x4_t v_w = vld1q_f32(p_w + i);

    v_y_est = vmlaq_f32(v_y_est, v_w, v_x);   // y += w * x
    v_energy = vmlaq_f32(v_energy, v_x, v_x); // energy += x * x
  }
  y_est = hsum_float32x4_nlms(v_y_est);
  energy_x_inst = hsum_float32x4_nlms(v_energy);

#elif defined(USE_AVX)
  __m256 v_y_est = _mm256_setzero_ps();
  __m256 v_energy = _mm256_setzero_ps();

  for (size_t i = 0; i < filter_length; i += 8) {
    __m256 v_x = _mm256_loadu_ps(p_x + i);
    __m256 v_w = _mm256_loadu_ps(p_w + i);

    // FMA is nice but simple mul/add is safer for compatibility if FMA flag is
    // tricky
    v_y_est = _mm256_add_ps(v_y_est, _mm256_mul_ps(v_w, v_x));
    v_energy = _mm256_add_ps(v_energy, _mm256_mul_ps(v_x, v_x));
  }
  y_est = hsum_float256_nlms(v_y_est);
  energy_x_inst = hsum_float256_nlms(v_energy);

#else
  // Scalar fallback
  for (size_t i = 0; i < filter_length; i++) {
    y_est += p_w[i] * p_x[i];
    energy_x_inst += p_x[i] * p_x[i];
  }
#endif

  // 3. Calculate Error (Clean Signal)
  float e = mic_input - y_est;
  mLastE = e;
  mLastYEst = y_est; // Store for diagnostics

  // 4. Update VSS Statistics
  // Is the error correlated with the reference?
  // If e contains the loop (echo), it will be correlated with x.
  // If e contains only clean guitar, it will be uncorrelated with x.

  // We update p_est (Cross-Correlation Estimate)
  p_est = alpha * p_est + (1.0f - alpha) * (e * aligned_ref);
  var_x = alpha * var_x + (1.0f - alpha) * (aligned_ref * aligned_ref);
  var_e = alpha * var_e + (1.0f - alpha) * (e * e);

  // 5. Calculate Dynamic Step Size (Variable Step Size)
  // Metric: (E[e*x])^2 / (E[e^2] * E[x^2]) -> Normalized Cross Correlation
  float correlation_metric = 0.0f;
  float denominator = var_e * var_x + epsilon;
  if (denominator > 1e-12f) {
    correlation_metric = (p_est * p_est) / denominator;
  }

  // Adapt step size: High correlation -> Large step. Low correlation -> Small
  // step.
  float mu_eff = mu_max * correlation_metric;

  // Clamp for stability
  if (mu_eff > mu_max)
    mu_eff = mu_max;
  mLastStep = mu_eff;
  mLastCorrelation = correlation_metric;

  // 6. Update Weights
  // NLMS rule: w[n+1] = w[n] + (mu / (||x||^2 + eps)) * e[n] * x[n]
  float norm_factor = energy_x_inst + epsilon;
  float step = mu_eff / norm_factor;
  float final_step = step * e; // Pre-multiply error

  static int vssLogCount = 0;
  if (vssLogCount++ % 48000 == 0) {  // Log once per second at 48kHz
    aecLog("[VSS_RT] mu_eff=%.6f p_est=%.6f var_e=%.6f var_x=%.6f corr=%.6f\n",
           mu_eff, p_est, var_e, var_x, correlation_metric);
  }

  // ============================================
  // 7. SIMD WEIGHT UPDATE
  // ============================================

#ifdef USE_NEON
  float32x4_t v_step = vdupq_n_f32(final_step);
  float32x4_t v_leak = vdupq_n_f32(leakage);

  for (size_t i = 0; i < filter_length; i += 4) {
    float32x4_t v_w = vld1q_f32(p_w + i);
    float32x4_t v_x = vld1q_f32(p_x + i);

    // w = (w * leakage) + (step * x)
    v_w = vmulq_f32(v_w, v_leak);
    v_w = vmlaq_f32(v_w, v_step, v_x);

    vst1q_f32(p_w + i, v_w); // Store back
  }

#elif defined(USE_AVX)
  __m256 v_step = _mm256_set1_ps(final_step);
  __m256 v_leak = _mm256_set1_ps(leakage);

  for (size_t i = 0; i < filter_length; i += 8) {
    __m256 v_w = _mm256_loadu_ps(p_w + i);
    __m256 v_x = _mm256_loadu_ps(p_x + i);

    v_w = _mm256_mul_ps(v_w, v_leak);
    v_w = _mm256_add_ps(v_w, _mm256_mul_ps(v_step, v_x));

    _mm256_storeu_ps(p_w + i, v_w);
  }
#else
  for (size_t i = 0; i < filter_length; i++) {
    p_w[i] = (p_w[i] * leakage) + (final_step * p_x[i]);
  }
#endif

  return e;
}

// ============================================
// Calibration / Warm Start
// ============================================

// ============================================
// Calibration / Warm Start
// ============================================

std::vector<float> VssNlmsFilter::getWeights() const { return weights; }

void VssNlmsFilter::warmStartWeights(const std::vector<float> &ref_signal,
                                     const std::vector<float> &mic_signal) {
  if (ref_signal.size() != mic_signal.size()) {
    return;
  }

  size_t len = ref_signal.size();

  // Debug: Calculate energy of inputs
  float refSum = 0, micSum = 0;
  for (float v : ref_signal)
    refSum += v * v;
  for (float v : mic_signal)
    micSum += v * v;

  aecLog("[VSS_DEBUG] WarmStart: %zu samples. RefEnergy=%.2f MicEnergy=%.2f\n",
         len, refSum, micSum);

  // Preserve original settings
  float original_mu = mu_max;
  float original_alpha = alpha;

  // Aggressive adaptation for warm start
  mu_max = 1.0f;
  alpha = 0.5f; // Fast smoothing for warm start

  aecLog(
      "[VSS_DEBUG] WarmStart: Set mu_max=%.2f alpha=%.2f. Starting loop...\n",
      mu_max, alpha);

  for (size_t i = 0; i < len; ++i) {
    float output = processSample(ref_signal[i], mic_signal[i]);

    // Debug first few samples to see if things explode immediately
    if (i < 5 || i % 10000 == 0) {
      aecLog("[VSS_DEBUG] i=%zu x=%.4f d=%.4f e=%.4f w[0]=%.4f p_est=%.4f\n", i,
             ref_signal[i], mic_signal[i], output, weights[0], p_est);
    }
    // Debug logging every 480 samples
    if (i % 480 == 0) {
      // Note: mu_eff, e, y, d, x_energy, ref_energy are not directly available
      // here. 'output' is 'e' from processSample. 'mLastStep' is 'mu_eff'.
      // 'mic_signal[i]' is 'd'.
      // 'ref_signal[i]' is 'x'.
      // 'y_est' is not directly available here.
      // 'energy_x_inst' is not directly available here.
      aecLog("[VSS_RT_TEST] i=%zu mu_eff=%.6f e=%.6f d=%.6f\n", i, mLastStep,
             output, mic_signal[i]);
    }
  }

  // Debug: Check weights energy
  float wEnergy = 0.0f;
  for (float w : weights)
    wEnergy += w * w;
  aecLog("[VSS_DEBUG] WarmStart Complete: WeightsEnergy=%.6f\n", wEnergy);

  // Restore settings
  mu_max = original_mu;
  alpha = original_alpha;
}
