#include "vss_nlms_filter.h"
#include <algorithm>
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
  x_history.resize(filter_length * 2, 0.0f);
  mHistoryIndex = 0;
}

void VssNlmsFilter::reset() {
  std::fill(weights.begin(), weights.end(), 0.0f);
  std::fill(x_history.begin(), x_history.end(), 0.0f);
  mHistoryIndex = 0;
  p_est = 0.0f;
  var_x = 0.0f;
  var_e = 0.0f;
  mLastE = 0.0f;
  mLastStep = 0.0f;
  x_energy_total = 0.0f;
  mMicEnergy = 0.0f;
  mEchoEstEnergy = 0.0f;
  mAppliedGain = 1.0f;
}

void VssNlmsFilter::resize(size_t newLength) {
  // Round up to nearest multiple of 8 for SIMD
  if (newLength % 8 != 0) {
    newLength = ((newLength + 7) / 8) * 8;
  }

  filter_length = newLength;

  // Resize and reinitialize vectors
  weights.resize(filter_length);
  x_history.resize(filter_length * 2);

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
  // Mirrored circular buffer update
  // We move the index backwards and mirror the sample to provide a contiguous
  // chronological window starting at history[index].
  mHistoryIndex = (mHistoryIndex == 0) ? filter_length - 1 : mHistoryIndex - 1;

  float old_sample = x_history[mHistoryIndex]; // About to be overwritten
  x_history[mHistoryIndex] = new_sample;
  x_history[mHistoryIndex + filter_length] = new_sample;

  // Incremental energy update: E = E + x[n]^2 - x[n-L]^2
  x_energy_total += (new_sample * new_sample) - (old_sample * old_sample);
  if (x_energy_total < 0.0f)
    x_energy_total = 0.0f; // Safety against rounding drift
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
  const float *p_x = x_history.data() + mHistoryIndex;
  float *p_w =
      weights.data(); // we will modify weights later, but for conv it reads

#ifdef USE_NEON
  float32x4_t v_y_est = vdupq_n_f32(0.0f);

  for (size_t i = 0; i < filter_length; i += 4) {
    float32x4_t v_x = vld1q_f32(p_x + i);
    float32x4_t v_w = vld1q_f32(p_w + i);

    v_y_est = vmlaq_f32(v_y_est, v_w, v_x); // y += w * x
  }
  y_est = hsum_float32x4_nlms(v_y_est);
  energy_x_inst = x_energy_total;

#elif defined(USE_AVX)
  __m256 v_y_est = _mm256_setzero_ps();
  __m256 v_energy = _mm256_setzero_ps();

  for (size_t i = 0; i < filter_length; i += 8) {
    __m256 v_x = _mm256_loadu_ps(p_x + i);
    __m256 v_w = _mm256_loadu_ps(p_w + i);

    v_y_est = _mm256_add_ps(v_y_est, _mm256_mul_ps(v_w, v_x));
  }
  y_est = hsum_float256_nlms(v_y_est);
  energy_x_inst = x_energy_total;

#else
  // Scalar fallback
  for (size_t i = 0; i < filter_length; i++) {
    y_est += p_w[i] * p_x[i];
  }
  energy_x_inst = x_energy_total;
#endif

  // 3. Ported Gain Limiting logic
  // This prevents the echo estimate from exceeding the mic input amplitude,
  // which eliminates "burbling" and "ringmod" sounds when the filter is
  // partially diverged.
  constexpr float GAIN_ENERGY_SMOOTH =
      0.0005f; // Fast tracking for energy peaks
  mMicEnergy = (1.0f - GAIN_ENERGY_SMOOTH) * mMicEnergy +
               GAIN_ENERGY_SMOOTH * (mic_input * mic_input);
  mEchoEstEnergy = (1.0f - GAIN_ENERGY_SMOOTH) * mEchoEstEnergy +
                   GAIN_ENERGY_SMOOTH * (y_est * y_est);

  float targetGain = 1.0f;
  if (mEchoEstEnergy > 1e-9f && mMicEnergy > 1e-9f) {
    float energyRatio = std::sqrt(mMicEnergy / mEchoEstEnergy);
    targetGain = std::min(1.0f, energyRatio);
  } else if (mEchoEstEnergy > mMicEnergy && mEchoEstEnergy > 1e-9f) {
    // If mic is silent but we estimate echo, be conservative
    targetGain = 0.0f;
  }

  // Smooth the gain factor itself to eliminate rapid amplitude modulation
  constexpr float GAIN_SMOOTH_FACTOR = 0.0002f; // ~100ms smoothing
  mAppliedGain = (1.0f - GAIN_SMOOTH_FACTOR) * mAppliedGain +
                 GAIN_SMOOTH_FACTOR * targetGain;

  // Apply smoothed gain to echo estimate
  float adapted_y_est = mAppliedGain * y_est;

  // 4. Calculate Error (Clean Signal)
  float e = mic_input - adapted_y_est;
  mLastE = e;
  mLastYEst = adapted_y_est; // Store for diagnostics

  // 5. Update VSS Statistics
  // With alpha = 0.999, these statistics are averaged over ~1000 samples,
  // preventing the step size from modulating at audio frequencies.
  p_est = alpha * p_est + (1.0f - alpha) * (e * aligned_ref);
  var_x = alpha * var_x + (1.0f - alpha) * (aligned_ref * aligned_ref);
  var_e = alpha * var_e + (1.0f - alpha) * (e * e);

  // 6. Calculate Dynamic Step Size (Variable Step Size)
  // Metric: (E[e*x])^2 / (E[e^2] * E[x^2]) -> Normalized Cross Correlation
  float correlation_metric = 0.0f;
  float denominator = var_e * var_x + epsilon;
  if (denominator > 1e-12f) {
    correlation_metric = (p_est * p_est) / denominator;
  }

  // Adapt step size: High correlation -> Large step.
  // We use sqrt(correlation_metric) to use the linear correlation coefficient
  // rather than squared correlation. This increases sensitivity for small
  // correlations, helping the filter start converging from zero.
  float mu_eff = mu_max * std::sqrt(correlation_metric);

  // Clamp for stability
  if (mu_eff > mu_max)
    mu_eff = mu_max;
  mLastStep = mu_eff;
  mLastCorrelation = correlation_metric;

  // 7. Update Weights
  // NLMS rule: w[n+1] = w[n] + (mu / (||x||^2 + eps)) * e[n] * x[n]
  float norm_factor = energy_x_inst + epsilon;
  float step = mu_eff / norm_factor;
  float final_step = step * e; // Pre-multiply error

  static int vssLogCount = 0;
  if (vssLogCount++ % 48000 == 0) {
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
    if (leakage < 0.99999f) {
      v_w = vmulq_f32(v_w, v_leak);
    }
    v_w = vmlaq_f32(v_w, v_step, v_x);

    vst1q_f32(p_w + i, v_w); // Store back
  }

#elif defined(USE_AVX)
  __m256 v_step = _mm256_set1_ps(final_step);
  __m256 v_leak = _mm256_set1_ps(leakage);

  for (size_t i = 0; i < filter_length; i += 8) {
    __m256 v_w = _mm256_loadu_ps(p_w + i);
    __m256 v_x = _mm256_loadu_ps(p_x + i);

    if (leakage < 0.99999f) {
      v_w = _mm256_mul_ps(v_w, v_leak);
    }
    v_w = _mm256_add_ps(v_w, _mm256_mul_ps(v_step, v_x));

    _mm256_storeu_ps(p_w + i, v_w);
  }
#else
  for (size_t i = 0; i < filter_length; i++) {
    float w = p_w[i];
    if (leakage < 0.99999f) {
      w *= leakage;
    }
    p_w[i] = w + (final_step * p_x[i]);
  }
#endif

  return e;
}

void VssNlmsFilter::processBlock(const float *ref_context,
                                 const float *mic_input, float *out_error,
                                 size_t num_samples) {
  for (size_t s = 0; s < num_samples; ++s) {
    // Current microphone sample
    float mic_sample = mic_input[s];

    // Reference context window for the current sample
    // ref_context points to: [History (filter_length-1)] [Current Block
    // (num_samples)] For sample 's', the relevant filter window starts at
    // ref_context + s
    const float *p_x = ref_context + s;
    const float *p_w = weights.data();

    float y_est = 0.0f;

    // ============================================
    // SIMD CONVOLUTION (Prediction)
    // ============================================
#ifdef USE_NEON
    float32x4_t v_y_est = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < filter_length; i += 4) {
      float32x4_t v_x = vld1q_f32(p_x + i);
      float32x4_t v_w = vld1q_f32(p_w + i);
      v_y_est = vmlaq_f32(v_y_est, v_w, v_x);
    }
    y_est = hsum_float32x4_nlms(v_y_est);
#elif defined(USE_AVX)
    __m256 v_y_est = _mm256_setzero_ps();
    for (size_t i = 0; i < filter_length; i += 8) {
      __m256 v_x = _mm256_loadu_ps(p_x + i);
      __m256 v_w = _mm256_loadu_ps(p_w + i);
      v_y_est = _mm256_add_ps(v_y_est, _mm256_mul_ps(v_w, v_x));
    }
    y_est = hsum_float256_nlms(v_y_est);
#else
    for (size_t i = 0; i < filter_length; i++) {
      y_est += p_w[i] * p_x[i];
    }
#endif

    // Gain Limiting
    constexpr float GAIN_ENERGY_SMOOTH = 0.0005f;
    mMicEnergy = (1.0f - GAIN_ENERGY_SMOOTH) * mMicEnergy +
                 GAIN_ENERGY_SMOOTH * (mic_sample * mic_sample);
    mEchoEstEnergy = (1.0f - GAIN_ENERGY_SMOOTH) * mEchoEstEnergy +
                     GAIN_ENERGY_SMOOTH * (y_est * y_est);

    float targetGain = 1.0f;
    if (mEchoEstEnergy > 1e-9f && mMicEnergy > 1e-9f) {
      targetGain = std::min(1.0f, std::sqrt(mMicEnergy / mEchoEstEnergy));
    } else if (mEchoEstEnergy > mMicEnergy && mEchoEstEnergy > 1e-9f) {
      targetGain = 0.0f;
    }

    constexpr float GAIN_SMOOTH_FACTOR = 0.0002f;
    mAppliedGain = (1.0f - GAIN_SMOOTH_FACTOR) * mAppliedGain +
                   GAIN_SMOOTH_FACTOR * targetGain;

    float adapted_y_est = mAppliedGain * y_est;
    float e = mic_sample - adapted_y_est;
    out_error[s] = e;

    mLastE = e;
    mLastYEst = adapted_y_est;

    // VSS Statistics
    float ref_val =
        p_x[filter_length - 1]; // Current sample in the context window
    p_est = alpha * p_est + (1.0f - alpha) * (e * ref_val);
    var_x = alpha * var_x + (1.0f - alpha) * (ref_val * ref_val);
    var_e = alpha * var_e + (1.0f - alpha) * (e * e);

    float correlation_metric = 0.0f;
    float denominator = var_e * var_x + epsilon;
    if (denominator > 1e-12f) {
      correlation_metric = (p_est * p_est) / denominator;
    }

    float mu_eff = std::min(mu_max, mu_max * std::sqrt(correlation_metric));
    mLastStep = mu_eff;
    mLastCorrelation = correlation_metric;

    // Weight Update Energy (Total energy of current window)
    float energy_x = 0.0f;
#ifdef USE_NEON
    float32x4_t v_en = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < filter_length; i += 4) {
      float32x4_t v_xi = vld1q_f32(p_x + i);
      v_en = vmlaq_f32(v_en, v_xi, v_xi);
    }
    energy_x = hsum_float32x4_nlms(v_en);
#elif defined(USE_AVX)
    __m256 v_en = _mm256_setzero_ps();
    for (size_t i = 0; i < filter_length; i += 8) {
      __m256 v_xi = _mm256_loadu_ps(p_x + i);
      v_en = _mm256_add_ps(v_en, _mm256_mul_ps(v_xi, v_xi));
    }
    energy_x = hsum_float256_nlms(v_en);
#else
    for (size_t i = 0; i < filter_length; i++) {
      energy_x += p_x[i] * p_x[i];
    }
#endif

    float step = mu_eff / (energy_x + epsilon);
    float final_step = step * e;

    // ============================================
    // SIMD WEIGHT UPDATE
    // ============================================
    float *p_w_mutable = weights.data();
#ifdef USE_NEON
    float32x4_t v_step = vdupq_n_f32(final_step);
    float32x4_t v_leak = vdupq_n_f32(leakage);
    for (size_t i = 0; i < filter_length; i += 4) {
      float32x4_t v_wi = vld1q_f32(p_w_mutable + i);
      float32x4_t v_xi = vld1q_f32(p_x + i);
      if (leakage < 0.99999f)
        v_wi = vmulq_f32(v_wi, v_leak);
      v_wi = vmlaq_f32(v_wi, v_step, v_xi);
      vst1q_f32(p_w_mutable + i, v_wi);
    }
#elif defined(USE_AVX)
    __m256 v_step = _mm256_set1_ps(final_step);
    __m256 v_leak = _mm256_set1_ps(leakage);
    for (size_t i = 0; i < filter_length; i += 8) {
      __m256 v_wi = _mm256_loadu_ps(p_w_mutable + i);
      __m256 v_xi = _mm256_loadu_ps(p_x + i);
      if (leakage < 0.99999f)
        v_wi = _mm256_mul_ps(v_wi, v_leak);
      v_wi = _mm256_add_ps(v_wi, _mm256_mul_ps(v_step, v_xi));
      _mm256_storeu_ps(p_w_mutable + i, v_wi);
    }
#else
    for (size_t i = 0; i < filter_length; i++) {
      float w = p_w_mutable[i];
      if (leakage < 0.99999f)
        w *= leakage;
      p_w_mutable[i] = w + (final_step * p_x[i]);
    }
#endif
  }
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
