#ifndef AEC_SPECTRAL_RESIDUAL_SUPPRESSOR_H
#define AEC_SPECTRAL_RESIDUAL_SUPPRESSOR_H

#include <array>
#include <cmath>
#include <cstddef>

/**
 * SpectralResidualSuppressor — the NONLINEAR post-filter that cleans up what
 * LINEAR loop-synchronous cancellation structurally cannot.
 *
 * WHY a second stage exists at all: SynchronousEchoTemplate subtracts a
 * per-phase echo ESTIMATE from the mic — a purely LINEAR operation. Linear
 * cancellation has a hard floor at high frequencies. A sub-sample timing
 * error between the stored echo and the real echo is a negligible fraction of
 * a wavelength at 100 Hz but a LARGE fraction of one at 3 kHz, so the very
 * same alignment that cancels the bass to nothing barely dents the treble.
 * Add speaker nonlinearity and small loop-to-loop IR drift (both worst at HF)
 * and the residual that survives is band-limited to the TOP of the spectrum —
 * heard as a thin "ghost" and, most audibly, the metronome click (≈0.8–1.5
 * kHz fundamental + harmonics) that refuses to fully cancel. No amount of
 * faster convergence or deeper averaging removes it: it isn't a convergence
 * problem, the linear estimate is already as good as a linear estimate gets.
 *
 * The only way past a linear floor is a nonlinear post-filter. This is a
 * per-band residual-echo suppressor: split the residual into a LOW band (the
 * musical fundamental — NEVER touched) and a few HIGH bands, and in each high
 * band apply a Wiener-style gain that ducks the band ONLY when its energy is
 * explained by residual echo and not by the performer. "Explained by echo" is
 * measured against the echo ESTIMATE the linear stage already computes: per
 * band we learn kappa_b, the fraction of expected echo power that SURVIVES
 * linear cancellation in that band (naturally small at LF, large at HF), and
 * only while the far-end is present AND the near-end is absent — so the
 * performer's own high-frequency content (a sung consonant, a bright pick),
 * which is NOT correlated with the loop's echo, pushes the band energy above
 * the echo estimate and the gain opens back to unity, preserving it.
 *
 * STRUCTURE — a COMPLEMENTARY crossover tree: each higher band is the signal
 * MINUS its own lowpass, so at unity gain the bands sum back to the exact
 * input sample-for-sample. That buys ZERO added latency and ZERO coloration
 * when nothing is being suppressed — critical in a looper, where the
 * suppressed signal IS the recording and any fixed delay here would desync
 * every overdub against the sample-accurate loop grid. Ducking a band simply
 * removes that band's content; the low band is always passed untouched so the
 * fundamental of voice/guitar is never gated.
 *
 * BOUNDED BY CONSTRUCTION: per-band gain floored at kGainMin (never a full
 * spectral hole -> no "musical noise"), only the HF bands are eligible, and
 * kappa adapts only in clean echo-only conditions (gated by the caller's
 * existing near-end detector), so over-suppression can't run away.
 *
 * Single-threaded: processSample() runs on the capture/audio thread only. No
 * allocation after construction.
 */
class SpectralResidualSuppressor {
public:
  // HF bands above the always-passed low band. Three is enough resolution to
  // put the metronome click (≈0.8–1.5 kHz) in the FIRST high band and the
  // "air"/hiss residual in the top band, without STFT-grade bin counts.
  static constexpr int kBands = 3;
  static constexpr int kMaxChannels = 8;

  SpectralResidualSuppressor(unsigned int sampleRate, unsigned int channels)
      : mSampleRate(sampleRate ? sampleRate : 48000), mChannels(channels) {
    // Complementary-tree crossovers. fc[0] separates the passed-through low
    // band from the processed highs; fc[1..] subdivide the highs. The low
    // crossover sits above most voice/guitar fundamentals AND above the range
    // where linear cancellation already works well, so the untouched low band
    // costs no cancellation. Everything above is attribution-gated, so reaching
    // down to 1 kHz does NOT dull the performance — a near-end consonant in
    // that band opens the gain right back up.
    setCrossover(0, 1000.0f);
    setCrossover(1, 2000.0f);
    setCrossover(2, 4000.0f);
    reset();
  }

  void setEnabled(bool e) { mEnabled = e; }
  bool enabled() const { return mEnabled; }

  void reset() {
    for (auto &cs : mState)
      cs = ChannelState{};
  }

  // Caller sets, once per block, whether conditions are clean enough to adapt
  // the per-band coupling kappa (far-end present AND near-end absent). When
  // false, kappa is HELD — the suppressor keeps working with the last good
  // estimate but never learns residual coupling from double-talk (which would
  // read the performer as "surviving echo" and over-suppress them).
  void setCouplingUpdateAllowed(bool a) { mAllowKappa = a; }

  // Hot path: one sample of one channel. `echoAnchor` = the reference-gated
  // echo estimate (refGate * est), whose per-band POWER stands in for the
  // expected echo power at the mic; `residual` = the linear residual to clean.
  // Returns the HF-suppressed residual (== residual EXACTLY when every band
  // gain is 1, by the complementary-tree identity below).
  inline float processSample(unsigned int ch, float echoAnchor, float residual) {
    if (!mEnabled || ch >= mChannels || ch >= kMaxChannels)
      return residual;
    ChannelState &s = mState[ch];

    // --- split the ECHO ANCHOR into bands (power only) ---
    const float xLp0 = s.xLp[0].lp(echoAnchor, mCoef[0]);
    const float xHf = echoAnchor - xLp0;
    const float xb0 = s.xLp[1].lp(xHf, mCoef[1]);
    const float xr1 = xHf - xb0;
    const float xb1 = s.xLp[2].lp(xr1, mCoef[2]);
    const float xb2 = xr1 - xb1;
    const float xBand[kBands] = {xb0, xb1, xb2};

    // --- split the RESIDUAL into bands (kept for reconstruction) ---
    const float rLp0 = s.rLp[0].lp(residual, mCoef[0]);
    const float rHf = residual - rLp0;
    const float rb0 = s.rLp[1].lp(rHf, mCoef[1]);
    const float rr1 = rHf - rb0;
    const float rb1 = s.rLp[2].lp(rr1, mCoef[2]);
    const float rb2 = rr1 - rb1;
    const float rBand[kBands] = {rb0, rb1, rb2};

    // Complementary-tree identity: rLp0 + rb0 + rb1 + rb2 == residual exactly.
    // So at unity gain the reconstruction below is a no-op (zero latency).
    float out = rLp0; // LOW band ALWAYS passed untouched
    for (int b = 0; b < kBands; ++b) {
      // Fast per-band power envelopes.
      s.xPow[b] += kEnvRate * (xBand[b] * xBand[b] - s.xPow[b]);
      s.rPow[b] += kEnvRate * (rBand[b] * rBand[b] - s.rPow[b]);

      // Adapt the residual-echo coupling kappa_b ONLY in clean echo-only
      // conditions: far-end present in this band (anchor above the floor) and
      // the caller says near-end is absent. inst = fraction of expected echo
      // power that leaked into the residual — clamped to 1 (can't leak MORE
      // echo than exists; anything above 1 is near-end the gate should have
      // caught).
      if (mAllowKappa && s.xPow[b] > kAnchorFloor) {
        float inst = s.rPow[b] / (s.xPow[b] + kEps);
        if (inst > 1.0f)
          inst = 1.0f;
        s.kappa[b] += kKappaRate * (inst - s.kappa[b]);
      }

      // Wiener-style gain: subtract the estimated residual-echo power
      // (kappa_b * expected echo power) from the band's total, slightly
      // over-subtracted (kBeta). When the band is pure residual echo,
      // rPow ≈ kappa*xPow -> gain -> kGainMin. When the performer adds
      // uncorrelated energy, rPow climbs above kappa*xPow -> gain -> 1.
      const float echoPow = s.kappa[b] * s.xPow[b];
      float target = 1.0f;
      if (s.rPow[b] > kEps) {
        target = (s.rPow[b] - kBeta * echoPow) / (s.rPow[b] + kEps);
        if (target < kGainMin)
          target = kGainMin;
        else if (target > 1.0f)
          target = 1.0f;
      }
      const float rate = (target < s.gain[b]) ? kGainAttack : kGainRelease;
      s.gain[b] += rate * (target - s.gain[b]);

      out += s.gain[b] * rBand[b];
    }
    return out;
  }

  // Telemetry (channel 0) for a debug overlay: how hard each HF band is being
  // ducked and how much echo it thinks survives there.
  float bandGain(int b) const {
    return (b >= 0 && b < kBands) ? mState[0].gain[b] : 1.0f;
  }
  float bandKappa(int b) const {
    return (b >= 0 && b < kBands) ? mState[0].kappa[b] : 0.0f;
  }

private:
  // 2nd-order Butterworth lowpass, Direct Form I. Coefficients pre-normalized
  // by a0: c = {b0, b1, b2, a1, a2}.
  struct Biquad {
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    inline float lp(float x, const std::array<float, 5> &c) {
      const float y =
          c[0] * x + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2;
      x2 = x1;
      x1 = x;
      y2 = y1;
      y1 = y;
      return y;
    }
  };

  struct ChannelState {
    std::array<Biquad, kBands> xLp; // echo-anchor lowpasses (one per crossover)
    std::array<Biquad, kBands> rLp; // residual lowpasses
    std::array<float, kBands> xPow{{0, 0, 0}};
    std::array<float, kBands> rPow{{0, 0, 0}};
    std::array<float, kBands> kappa{{kKappaInit, kKappaInit, kKappaInit}};
    std::array<float, kBands> gain{{1.0f, 1.0f, 1.0f}};
  };

  void setCrossover(int i, float fc) {
    // RBJ 2nd-order Butterworth lowpass (Q = 1/sqrt2).
    const float w0 =
        2.0f * 3.14159265358979f * fc / static_cast<float>(mSampleRate);
    const float cw = std::cos(w0), sw = std::sin(w0);
    const float Q = 0.70710678f;
    const float alpha = sw / (2.0f * Q);
    const float a0 = 1.0f + alpha;
    const float b0 = (1.0f - cw) * 0.5f / a0;
    const float b1 = (1.0f - cw) / a0;
    const float b2 = (1.0f - cw) * 0.5f / a0;
    const float a1 = (-2.0f * cw) / a0;
    const float a2 = (1.0f - alpha) / a0;
    mCoef[i] = {b0, b1, b2, a1, a2};
  }

  unsigned int mSampleRate;
  unsigned int mChannels;
  bool mEnabled = true;
  bool mAllowKappa = false;
  std::array<std::array<float, 5>, kBands> mCoef;
  std::array<ChannelState, kMaxChannels> mState;

  // --- tuning (per-sample rates tuned for 48 kHz; 44.1 kHz is close enough) --
  static constexpr float kEnvRate = 0.01f;     // ~2 ms band power envelope
  static constexpr float kKappaRate = 0.0002f; // ~100 ms coupling adaptation
  static constexpr float kKappaInit = 0.25f;   // moderate until it learns
  static constexpr float kAnchorFloor = 1e-7f; // far-end-present threshold (pow)
  static constexpr float kBeta = 1.3f;         // slight over-subtraction
  static constexpr float kGainMin = 0.12f;     // ≈ -18 dB floor (no full holes)
  static constexpr float kGainAttack = 0.06f;  // ~0.4 ms duck
  static constexpr float kGainRelease = 0.003f;// ~10 ms recover
  static constexpr float kEps = 1e-12f;
};

#endif // AEC_SPECTRAL_RESIDUAL_SUPPRESSOR_H
