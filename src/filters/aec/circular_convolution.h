#ifndef AEC_CIRCULAR_CONVOLUTION_H
#define AEC_CIRCULAR_CONVOLUTION_H

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

/**
 * FFT-based EXACT period-P circular convolution for the LSAEC seed and
 * per-track contribution jobs (worker thread only — never the RT thread).
 *
 * WHY: the direct implementation is O(P·L) — ~1e9 multiply-adds for a 5 s
 * loop against a 4096-tap IR (~1 s of worker time per track, several seconds
 * for long loops). That latency is most of the gap between "overdub launches"
 * and "its echo model is live", i.e. the feed-forward convergence floor.
 * FFT brings it to O(N log N) ≈ tens of milliseconds.
 *
 * METHOD: zero-pad signal (P) and kernel (L ≤ P) to a power-of-two
 * N ≥ P+L-1, multiply spectra (one complex FFT each — sizes here don't merit
 * a real-input split), inverse-transform, then wrap the linear tail back onto
 * the head modulo P. Wrapping a LINEAR convolution is exactly the period-P
 * circular convolution — same values the direct sum produces (verified
 * against it in tests/test_circular_convolution.cpp to ~1e-6 abs).
 *
 * The L ≤ P truncation contract from computeSeedConvolution (a kernel longer
 * than the period aliases multiple taps onto one phase — the -30 dB ERLE
 * incident) is the CALLER's job, same as before.
 */
namespace aec_conv {

// In-place iterative radix-2 Cooley-Tukey. n must be a power of two.
inline void fftRadix2(std::vector<std::complex<double>> &a, bool inverse) {
  const size_t n = a.size();
  // Bit-reversal permutation.
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (size_t k = 0; k < len / 2; ++k) {
        const std::complex<double> u = a[i + k];
        const std::complex<double> v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (inverse) {
    for (auto &x : a)
      x /= static_cast<double>(n);
  }
}

/**
 * out[phi] = sum_k kernel[k] * signal[(phi - k) mod P], phi in [0, P).
 * Requires kernel.size() <= P (caller truncates). `out` is resized to P.
 * Falls back to the direct sum for small jobs where FFT setup would lose.
 */
inline void circularConvolve(const std::vector<float> &signal,   // one period, len P
                             const std::vector<float> &kernel,   // L <= P taps
                             std::vector<float> &out) {
  const size_t P = signal.size();
  const size_t L = kernel.size();
  out.assign(P, 0.0f);
  if (P == 0 || L == 0)
    return;

  // Small-job cutoff: direct O(P·L) beats FFT setup below ~1M MACs.
  if (P * L <= (1u << 20)) {
    for (size_t phi = 0; phi < P; ++phi) {
      double acc = 0.0;
      for (size_t k = 0; k < L; ++k) {
        const size_t idx = (phi + P - k) % P;
        acc += static_cast<double>(kernel[k]) * static_cast<double>(signal[idx]);
      }
      out[phi] = static_cast<float>(acc);
    }
    return;
  }

  size_t n = 1;
  while (n < P + L - 1)
    n <<= 1;

  std::vector<std::complex<double>> fa(n), fb(n);
  for (size_t i = 0; i < P; ++i)
    fa[i] = signal[i];
  for (size_t i = 0; i < L; ++i)
    fb[i] = kernel[i];

  fftRadix2(fa, false);
  fftRadix2(fb, false);
  for (size_t i = 0; i < n; ++i)
    fa[i] *= fb[i];
  fftRadix2(fa, true);

  // Linear result has P+L-1 meaningful samples; fold the tail mod P to turn
  // linear into exact circular.
  for (size_t i = 0; i < P + L - 1; ++i) {
    const double v = fa[i].real();
    out[i % P] += static_cast<float>(v);
  }
}

} // namespace aec_conv

#endif // AEC_CIRCULAR_CONVOLUTION_H
