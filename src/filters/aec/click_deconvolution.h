#ifndef AEC_CLICK_DECONVOLUTION_H
#define AEC_CLICK_DECONVOLUTION_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "circular_convolution.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Divide the exactly-known calibration probe click back out of the averaged
 * click response (worker thread only — runs once per calibration).
 *
 * WHY: click calibration averages raw mic windows around each click, so what
 * it stores is (room IR ⊛ click), not the room IR. The 48-sample raised
 * cosine has DC gain ~23.5 and ~1 ms of width, which leaves the stored
 * template roughly 8x hot, smeared by a millisecond, and containing nothing
 * above ~2 kHz (the probe itself has no energy there). Used directly as a
 * subtraction template that caps cancellation depth well below the linear
 * ceiling of the room.
 *
 * METHOD: frequency-domain Tikhonov division. The click is known
 * analytically, so H(f) = A(f)·conj(C(f)) / (|C(f)|² + λ·max|C|²). Bins
 * where the probe has essentially no energy (below the coherent-band cut)
 * are ZEROED rather than divided — there is no information there to recover,
 * and regularized division would only shape noise.
 */
namespace aec_cal {

/**
 * @param ir             In/out: averaged (IR ⊛ click) response; replaced by
 *                       the deconvolved IR of the same length.
 * @param clickSamples   Probe click length in samples (raised cosine).
 * @param clickAmplitude Probe click peak amplitude.
 */
inline void deconvolveProbeClick(std::vector<float> &ir, int clickSamples,
                                 float clickAmplitude) {
  const size_t irLen = ir.size();
  if (irLen == 0 || clickSamples < 2)
    return;

  size_t n = 1;
  while (n < irLen + static_cast<size_t>(clickSamples))
    n <<= 1;

  std::vector<std::complex<double>> A(n), C(n);
  for (size_t i = 0; i < irLen; ++i)
    A[i] = ir[i];
  for (int i = 0; i < clickSamples; ++i) {
    // EXACT mirror of generateClickSignal's raised cosine — the probe must
    // match what was actually played or the division is against the wrong
    // kernel.
    const double t = static_cast<double>(i) / (clickSamples - 1);
    C[i] = clickAmplitude * 0.5 * (1.0 - std::cos(2.0 * M_PI * t));
  }

  aec_conv::fftRadix2(A, false);
  aec_conv::fftRadix2(C, false);

  double maxMag2 = 0.0;
  for (const auto &c : C)
    maxMag2 = std::max(maxMag2, std::norm(c));
  if (maxMag2 <= 0.0)
    return;

  // λ = 1e-2 caps the boost near the probe's spectral nulls; the coherent
  // cut (1e-3 of peak power ≈ −30 dB) zeroes everything outside the band
  // the probe actually excited — for the 48-sample raised cosine that is
  // roughly DC..2 kHz.
  const double tikhonov = 1e-2 * maxMag2;
  const double coherentCut = 1e-3 * maxMag2;

  for (size_t i = 0; i < n; ++i) {
    const double mag2 = std::norm(C[i]);
    A[i] = (mag2 < coherentCut)
               ? std::complex<double>(0.0, 0.0)
               : A[i] * std::conj(C[i]) / (mag2 + tikhonov);
  }

  aec_conv::fftRadix2(A, true);
  for (size_t i = 0; i < irLen; ++i)
    ir[i] = static_cast<float>(A[i].real());
}

} // namespace aec_cal

#endif // AEC_CLICK_DECONVOLUTION_H
