#include "../../fft/soloud_fft.h"
#include "spectral_governor.h"
#include "synchronous_echo_template.h"

#include "circular_convolution.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#ifdef __ANDROID__
#include <android/log.h>
#endif

extern void aecLog(const char *fmt, ...); // see nlms_filter.h

namespace {
// CONFIDENCE-ANNEALED synchronous-average rate: E[phi] += alpha*(mic - E[phi])
// once per loop pass at this phase. alpha starts HOT (fast convergence) and
// anneals toward a WARM floor (so it keeps tracking the slowly-drifting echo
// while still averaging the period-incoherent performer down).
constexpr float kAlphaMax = 0.5f;  // hot: unsettled phase, learn fast
// Warm floor = steady-state averaging depth. At 0.20 the template tracked fast
// but capped depth (near-end leaks into E at ~alpha per pass and the estimate
// stays noisy). The single-clock drift measurement (±0.35 ppm) says the echo
// path is static unless the device physically moves, so depth wins: 0.06
// averages ~3x deeper and costs only a few extra passes after a path change.
constexpr float kAlphaMin = 0.06f;
constexpr float kConfTau = 8.0f;   // anneal time-constant in loop passes

// Far-end learn gate on a SMOOTHED reference-power envelope (~1.5 ms EMA), so
// genuinely silent-speaker passages never absorb near-end energy but waveform
// zero-crossings do NOT punch unlearned pinholes into the template.
constexpr float kFarEndFloorPow = 1e-6f; // ~-60 dB envelope
constexpr float kRefEnvRate = 0.014f;    // per-sample EMA, tau ~1.5 ms @ 48 kHz

// Reference-presence subtraction gate rates/floor now live as tunable
// members (mSubGateAttackRate/mSubGateReleaseRate/mSubGateFloorPow — see
// setSubGateTuning in the header). Defaults preserve the old constants:
// fast attack so cancellation re-engages the instant the speaker starts;
// slow release (~reverb time) so the echo tail keeps cancelling as the
// reference decays; refGate = env/(env+floor), a soft 0..1 knee.

constexpr float kConfMax = 4096.0f; // confidence saturates (~enough passes)

// Block-level double-talk freeze. A block whose total residual energy spikes
// well above the smoothed floor (mResidBaseline) is near-end -> skip learning
// that block so the performer isn't averaged into E. One smoothed decision per
// block (per-frame thresholding thrashed — audio energy is too spiky).
constexpr uint32_t kSettleBlocks = 64; // blocks to establish the baseline first
constexpr float kSpikeRatio = 25.0f;   // block residual > 25x floor = near-end
// Freeze only where the template has CONVERGED (enough passes at these
// phases). Before that, a residual spike is just the loop's own transients
// not yet learned — freezing there starves exactly the loudest phases forever
// (they never converge, so they always spike, so they always freeze: the
// "display converges but I still hear the hits" failure). While confidence is
// low we learn unconditionally; performer bleed is period-incoherent and
// self-heals by averaging — the core LSAEC premise.
constexpr float kFreezeMinConf = 6.0f; // passes before the detector may freeze
constexpr float kBaseRate = 0.08f;     // residual-floor EMA rate per block
constexpr float kEps = 1e-12f;

// Output safety-suppression gain: smoothed at BLOCK (not sample) granularity
// — see mOutputSuppressGain's header comment for why per-sample was tried
// and rejected (broadband static). ~15ms reaction time: fast enough to
// audibly matter within a couple of blocks, slow enough that the gain
// itself never becomes a source of zipper artifacts.
constexpr float kOutputGateTauMs = 15.0f;
// Separately, the mic/raw ENERGY RATIO is smoothed first (see
// mSmoothedEnergyRatio) so single-block statistical noise in the near-end/
// echo cross-correlation term averages out BEFORE the margin decision — a
// slightly longer window than the gain's own smoothing, since it needs to
// average across genuinely independent blocks' noise, not just de-click a
// gain transition. This is what lets kOutputGateMargin below be tight
// enough to catch sustained-but-moderate staleness (confirmed on-device: a
// wide single-block margin missed a real, consistently mildly-negative-ERLE
// ghost) without false-positiving on healthy audio's per-block variance.
constexpr float kEnergyRatioTauMs = 25.0f;
// Engage once the SMOOTHED raw residual energy exceeds mic energy by this
// factor. Tight relative to the old single-block margin (was 3.0) because
// the smoothing above already absorbs the noise a wide margin used to guard
// against.
constexpr float kOutputGateMargin = 1.3f;

// Maximum supported loop period. Sized once at construction so process()
// never allocates on the audio thread. Overridable at compile time
// (-DLSAEC_MAX_SECONDS=N). Raised 16 -> 64 for multi-length loops: the
// composite echo's true period is the LCM of the active loop lengths
// (x1/x2/x4/x8 picker -> max multiple x base), so even a 2x overdub over an
// 8.6 s base (17.2 s) blew the old cap — which silently disabled ALL
// cancellation (the "second loop doesn't cancel at all" report) — and an
// x8 phrase loop needs far more. Memory scales ~1.15 MB per second @48 kHz
// across template+confidence+capture+seed buffers: 128 s ~ 148 MB (pro app, desktop-class). Fine on
// desktop; on phones consider a platform -DLSAEC_MAX_SECONDS if memory
// pressure shows up. Stage 1 (IR-domain canceller, no length cap) removes
// this constant entirely.
#ifndef LSAEC_MAX_SECONDS
#define LSAEC_MAX_SECONDS 128
#endif
constexpr unsigned int kMaxSeconds = LSAEC_MAX_SECONDS;

// ---- Convergence seed tuning (see class comment) --------------------------
// Confidence assigned to every phase immediately after seeding: high enough
// that alpha lands essentially at the warm floor (not the hot ceiling — the
// seed IS the converged estimate, so treat it like one) and high enough to
// clear kFreezeMinConf so E3 double-talk protection is active from sample one
// instead of waiting through a fake "unsettled" window.
constexpr float kSeedConfidence = 24.0f; // 3x kConfTau
// Phases applied to mTemplate per process() call while draining a finished
// seed job. Bounds the apply cost to O(1) per callback regardless of loop
// length (a 16s loop is 768000 phases; applying it in one shot could exceed
// the RT budget, so it's spread over ~188 callbacks — well under one pass).
constexpr int64_t kSeedApplyChunk = 4096;
// Worker poll interval (mirrors SpectralGovernor's cadence) — seeding is a
// rare, one-shot event per loop-period change, not a steady-state load.
constexpr int kSeedPollMs = 8;

// Feed-forward is now split into two independently-gated mechanisms so they
// can be re-enabled one at a time WITH evidence (the 2026-07-18 "terrible
// regression" was the WHOLE stack enabled on a poisoned IR — inverted/8x-hot,
// pre-P1/P2 calibration).
//
// kSeedEnabled — the COMPOSITE convergence seed (armSeedCaptureIfPossible):
//   captures one loop period of aligned reference, convolves it with the
//   calibrated IR off-thread, and stamps the template so a new loop converges
//   in ~1 pass instead of ~20. Protected by the self-fit scalar
//   (alpha = <mic,seed>/<seed,seed>, clamp +-2, discard |alpha|<0.05) which
//   AUTO-corrects a wrong-sign/scale IR — it structurally cannot anti-cancel
//   the way the day-1 unfitted seed did. Offline proof (Desktop aec_ref/mic,
//   cross-validated): a correct IR seed gives ~19 dB / ~99% cancellation with
//   NO learn-up. Requires a fresh P1/P2 calibration to reach that; degrades
//   gracefully (small alpha, no benefit, no harm) on a stale IR.
//
// kFeedForwardEnabled — the PER-TRACK exact edits (registerTrackAudio /
//   setTrackActive / setTrackGain): the truly-instant path (IR (x) each loop's
//   OWN known audio, added/subtracted arithmetically on mute/unmute, zero
//   learn-up even for 16-bar loops). NOW ENABLED: the #54 Dart-thread template-
//   write race is fixed (setTrackActive/setTrackGain validate-and-enqueue; the
//   audio thread applies via drainPendingTrackEdits at the top of process()).
//   Loop registration + mute/unmute wiring were already in place. This is the
//   fix for muted-track bleed: without the exact edit, the composite template
//   keeps subtracting a muted track's echo it can no longer hear, adding it
//   back inverted (bleed) until the slow EMA un-learns it. The exact edit
//   removes E_track from the template the instant the track mutes. Needs a
//   fresh P1/P2 calibration for the IR to be right; self-fit + safety clamp
//   guard a stale one; revert to false if it regresses.
// Overridable for the offline harness (A/B the seed's contribution to
// convergence/divergence on replayed captures); production default unchanged.
#ifndef LSAEC_SEED_ENABLED
#define LSAEC_SEED_ENABLED 1
#endif
constexpr bool kSeedEnabled = LSAEC_SEED_ENABLED != 0;
constexpr bool kFeedForwardEnabled = true;
} // namespace

SynchronousEchoTemplate::SynchronousEchoTemplate(unsigned int sampleRate,
                                                 unsigned int channels)
    : mSampleRate(sampleRate), mChannels(channels),
      mSuppressor(sampleRate, channels) {
  mCapacityFrames = static_cast<size_t>(sampleRate) * kMaxSeconds;
  mTemplate.assign(mCapacityFrames * channels, 0.0f);
  mConfidence.assign(mCapacityFrames, 0.0f);
  mRefCapture.assign(mCapacityFrames, 0.0f);
  mMicCapture.assign(mCapacityFrames, 0.0f);
  // ~400ms of learned-frame undo at 48k (see header): 19200 entries.
  mUndoRing.assign(19200, LearnUndo{});
  mSeedOutput.assign(mCapacityFrames, 0.0f);
  mSeedThreadRunning.store(true, std::memory_order_relaxed);
  mSeedWorker = std::thread([this] { seedWorkerLoop(); });
}

SynchronousEchoTemplate::~SynchronousEchoTemplate() {
  mSeedThreadRunning.store(false, std::memory_order_relaxed);
  if (mSeedWorker.joinable())
    mSeedWorker.join();
}

void SynchronousEchoTemplate::setSubGateTuning(float attackMs, float releaseMs,
                                               float floorDb) {
  // ms time-constant -> per-sample EMA rate (rate = 1/tau_samples), clamped
  // to sane bounds so a bad slider value can't disable or destabilize the
  // gate. dB (power) -> linear floor.
  const float fs = static_cast<float>(mSampleRate ? mSampleRate : 48000);
  auto msToRate = [fs](float ms) {
    const float tauSamples = std::max(1.0f, ms * fs / 1000.0f);
    return std::min(1.0f, 1.0f / tauSamples);
  };
  mSubGateAttackRate.store(msToRate(std::max(0.05f, attackMs)),
                           std::memory_order_relaxed);
  mSubGateReleaseRate.store(msToRate(std::max(0.5f, releaseMs)),
                            std::memory_order_relaxed);
  const float db = std::min(-10.0f, std::max(-90.0f, floorDb));
  mSubGateFloorPow.store(std::pow(10.0f, db / 10.0f),
                         std::memory_order_relaxed);
  aecLog("[LSAEC] sub-gate tuning: attack=%.2fms release=%.1fms floor=%.1fdB\n",
         attackMs, releaseMs, floorDb);
}

void SynchronousEchoTemplate::setSeedImpulseResponse(const float *coeffs,
                                                     int length) {
  if (!coeffs || length <= 0)
    return;
  {
    std::lock_guard<std::mutex> lock(mSeedIRMutex);
    mSeedIR.assign(coeffs, coeffs + length);
  }
  // The IR just arrived or changed. Any track registered BEFORE now (e.g.
  // recorded before this session's live calibration) computed against an
  // empty/old IR and got no usable contribution — recompute them all
  // against the new IR so record-vs-calibrate order stops mattering.
  requeueAllTrackContributions();
}

void SynchronousEchoTemplate::reset() {
  std::fill(mTemplate.begin(), mTemplate.end(), 0.0f);
  std::fill(mConfidence.begin(), mConfidence.end(), 0.0f);
  mConfidenceSum = 0.0;
  mActiveLoopFrames = 0;
  mActiveLoopFramesAtomic.store(0, std::memory_order_relaxed);
  mRefEnvelope = 0.0f;
  mSubGateEnv = 0.0f;
  mResidBaseline = 0.0f;
  mLearnedBlocks = 0;
  mFreezeCount = 0;
  mReopenCount = 0;
  mSuppressor.reset();
  // Abandon any in-progress capture/job (bump the generation so a job the
  // worker completes for this now-stale epoch is dropped at drain time
  // without touching mSeedBusy — see mSeedGeneration comment in the header).
  ++mSeedGeneration;
  if (mSeedBusy) {
    mSeedAborts.fetch_add(1, std::memory_order_relaxed);
    aecLog("[SEED] aborted (reset)\n");
  }
  mSeedBusy = false;
  mSeedCaptureRemaining = 0;
  mSeedApplyPos = 0;
  mSeedFitActive = false;
  mSeedFitDone = false;
  mSeedFitFrames = 0;
}

void SynchronousEchoTemplate::seedWorkerLoop() {
  while (mSeedThreadRunning.load(std::memory_order_relaxed)) {
    bool didWork = false;

    if (mSeedJobPosted.load(std::memory_order_acquire)) {
      computeSeedConvolution(mSeedJobPeriod);
      mSeedJobPosted.store(false, std::memory_order_release);
      mSeedOutputReady.store(true, std::memory_order_release);
      aecLog("[SEED] worker convolution done P=%lld\n",
             (long long)mSeedJobPeriod);
      didWork = true;
    }

    // Per-track registration jobs (see registerTrackAudio doc comment) —
    // rare, one-shot-per-track, so a plain mutex-protected queue is fine;
    // this never runs on the audio thread.
    TrackRegJob job;
    bool haveJob = false;
    {
      std::lock_guard<std::mutex> lock(mTrackJobMutex);
      if (!mPendingTrackJobs.empty()) {
        job = std::move(mPendingTrackJobs.back());
        mPendingTrackJobs.pop_back();
        haveJob = true;
      }
    }
    if (haveJob) {
      computeTrackContribution(job.trackIndex, job.audio);
      didWork = true;
    }

    if (!didWork)
      std::this_thread::sleep_for(std::chrono::milliseconds(kSeedPollMs));
  }
}

void SynchronousEchoTemplate::armSeedCaptureIfPossible(const float *alignedRef) {
  if (!kSeedEnabled)
    return;
  if (mSeedBusy || !alignedRef)
    return; // already capturing/awaiting a job, or nothing to capture from
  // Don't arm until the reference is actually AUDIBLE: arming at anchor
  // time, before playback rings out, captured a loop of silence — Wiener
  // coherence 0.00, IR fit alpha 0.000, seed discarded, and (with arms=1
  // and no retry) convergence then rode slow EMA for the whole session
  // ("doesn't converge on an existing session", caught live by the
  // [SEED] diagnostics the first minute they were enabled).
  if (mRefEnvelope <= kFarEndFloorPow) {
    mReferenceChangePending.store(true, std::memory_order_relaxed);
    return; // keep the arm request pending until the far end is live
  }
  // NOTE: no calibrated-IR requirement anymore — the one-loop Wiener seed
  // measures the live transfer function from the captured pass itself, so
  // an uncalibrated device converges just as fast. The IR remains a
  // fallback inside computeSeedConvolution when the pass has no usable
  // coherence.
  ++mSeedGeneration; // this arm owns a fresh epoch; no prior job can match it
  mSeedBusy = true;
  mSeedCaptureRemaining = mActiveLoopFrames;
  mSeedApplyPos = 0;
  mSeedArms.fetch_add(1, std::memory_order_relaxed);
  aecLog("[SEED] armed gen=%lld P=%lld\n", (long long)mSeedGeneration,
         (long long)mActiveLoopFrames);
}

void SynchronousEchoTemplate::computeSeedConvolution(int64_t P) {
  std::vector<float> ir;
  {
    std::lock_guard<std::mutex> lock(mSeedIRMutex);
    ir = mSeedIR; // snapshot — a mid-job calibration update won't tear this read
  }
  if (P <= 0)
    return; // ir may legitimately be empty: the Wiener path needs no IR
  const size_t Pu = static_cast<size_t>(P);
  // CRITICAL: circular convolution is only well-posed as "one clean copy of
  // h wrapped around a period-P cycle" when L <= P. If the calibrated IR
  // (4096 taps = 85 ms @ 48 kHz) is LONGER than the loop period — a short
  // loop, entirely plausible right after calibrating — using all L taps
  // makes `k % Pu` wrap MULTIPLE times, so several taps land on the SAME
  // output phase and sum constructively. A real IR carries meaningful energy
  // across most of its length (this session's calibration measured
  // echoGain=sqrt(Σh²)=4.05), so that pileup produces an output tens of dB
  // louder than the mic — confirmed on-device: ERLE cratered to -30 dB on
  // the very first seed after a fresh calibration. Fix: use only the first
  // min(L, P) taps — the standard, correct treatment for a period-P circular
  // convolution with a kernel that may exceed the period. A P-tap prefix
  // still carries the direct-path + early-reflection energy that matters
  // most; the discarded reverb tail was going to alias into garbage anyway.
  const size_t L = std::min(ir.size(), Pu);

  // Defense in depth: a real acoustic echo (speaker leaking into the mic) is
  // essentially always QUIETER than the direct/reference signal — the room
  // attenuates it. A seed predicting echo energy comparable to or louder
  // than the reference is never physically correct, and applying it doesn't
  // just under/over-cancel by a little: with confidence set to max on
  // arrival, an over-estimated seed dominates the subtraction outright,
  // producing a phase-inverted copy of whatever the reference was instead of
  // Amplitude bound on the seed. This was a reference-relative unity cap
  // ("echo can't exceed the reference") — but that is FALSE on real hardware
  // with hot mic gain: measured echoGain is 2.7-3x (iOS/macOS), so capping
  // the seed at 1x the reference lands it at only ~35% of the true echo. The
  // remaining ~65% then has to be closed by per-pass learning at the low
  // post-seed alpha (~0.08 after the seed stamps confidence 24) — ~20 loop
  // passes, i.e. the "takes too long to converge" the user reported. The IR
  // already encodes the true echo gain (and calibration success now requires
  // a real quality metric), so trust it: use only a loose ABSOLUTE blow-up
  // guard, same as the per-track path (kMaxContribAbs).
  //
  // The on-device failure that originally motivated the unity cap (a seed
  // that recorded as a phase-inverted, ~30ms-delayed copy of the OTHER loop
  // layer) was a STALENESS bug — a seed computed against a reference that no
  // longer matched the audible mix — not an amplitude bug. That is now
  // defended structurally by (a) aborting+restarting the capture on any
  // mid-capture mix change (consistent-mix capture, see the arm logic) and
  // (b) the reference-presence subtraction gate (mSubGateEnv) which refuses
  // to subtract ANY stored estimate while the far-end is silent. Neither
  // existed when the unity cap was chosen.
  constexpr float kMaxSeedAbs = 8.0f; // matches per-track kMaxContribAbs

  // Exact circular convolution over the (possibly truncated) kernel: the
  // reference is exactly periodic at P, so the true steady-state echo is
  // h ⊛ ref computed modulo P — not a linear-convolution approximation with
  // edge effects. seed[phi] predicts what per-pass learning would converge
  // to, using the SAME alignedRef delay convention the live filter already
  // uses (see calibration.cpp analyzeAligned). FFT-based (worker thread):
  // was a direct O(P·L) loop — ~4 s of compute for a 5 s loop, i.e. most of
  // the audible convergence latency after a period change. Now ~50 ms.
  std::vector<float> refPeriod(mRefCapture.begin(),
                               mRefCapture.begin() + Pu);

  // ---- ONE-LOOP WIENER SEED (preferred) --------------------------------
  // The captured period contains BOTH the aligned reference and the raw mic
  // at the same phases: one pass of live data fully determines the linear
  // transfer function ref->mic. Estimate per-bin H = Sxy/Sxx with a Welch
  // sweep over the (circular) period, COHERENCE-GATE each bin (bins where
  // the mic isn't explained by the reference — the performer jamming, room
  // noise — contribute NOTHING, so the seed is double-talk-immune by
  // construction), and synthesize E = H (x) ref via overlap-add. Offline
  // validation of this exact estimator on real session pairs measured
  // 17-33 dB of identifiable-bleed removal from one pass — versus ~5 dB
  // starting depth for the chirp-IR seed it replaces (stale calibrations
  // fit alpha ~0.7 at best, ~0 at worst). The IR-convolution path below
  // remains as fallback when the Wiener estimate has no usable coherence
  // (mic muted, silent pass) or the period is too short for the analysis
  // window.
  constexpr int kWienerWin = 1024; // FFT::fft/ifft, 512 packed complex bins
  constexpr int kWienerHop = 256;  // 75% overlap; hann OLA sums to a constant
  constexpr double kBinCohGate = 0.30;
  bool wienerUsed = false;
  if (Pu >= static_cast<size_t>(kWienerWin) * 2) {
    static thread_local std::vector<float> hann;
    if (hann.size() != static_cast<size_t>(kWienerWin)) {
      hann.assign(kWienerWin, 0.0f);
      for (int i = 0; i < kWienerWin; ++i)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979f * i /
                                          (kWienerWin - 1)));
    }
    const int nBins = kWienerWin / 2;
    std::vector<double> sxx(nBins, 0.0), syy(nBins, 0.0);
    std::vector<double> sxyRe(nBins, 0.0), sxyIm(nBins, 0.0);
    std::vector<float> xw(kWienerWin), yw(kWienerWin);

    // Pass A: accumulate cross/auto spectra over the circular period.
    for (size_t start = 0; start < Pu; start += kWienerHop) {
      for (int i = 0; i < kWienerWin; ++i) {
        const size_t idx = (start + i) % Pu;
        xw[i] = refPeriod[idx] * hann[i];
        yw[i] = mMicCapture[idx] * hann[i];
      }
      FFT::fft(xw.data(), kWienerWin);
      FFT::fft(yw.data(), kWienerWin);
      for (int b = 0; b < nBins; ++b) {
        const double xr = xw[2 * b], xi = xw[2 * b + 1];
        const double yr = yw[2 * b], yi = yw[2 * b + 1];
        sxx[b] += xr * xr + xi * xi;
        syy[b] += yr * yr + yi * yi;
        sxyRe[b] += yr * xr + yi * xi;
        sxyIm[b] += yi * xr - yr * xi;
      }
    }

    // Per-bin coherence-gated Wiener filter.
    std::vector<double> hRe(nBins, 0.0), hIm(nBins, 0.0);
    double cohSum = 0.0;
    int cohBins = 0;
    for (int b = 0; b < nBins; ++b) {
      const double denom = sxx[b] * syy[b] + 1e-24;
      const double coh = (sxyRe[b] * sxyRe[b] + sxyIm[b] * sxyIm[b]) / denom;
      if (sxx[b] > 1e-12) {
        cohSum += coh;
        ++cohBins;
      }
      if (coh >= kBinCohGate && sxx[b] > 1e-12) {
        hRe[b] = sxyRe[b] / (sxx[b] + 1e-20);
        hIm[b] = sxyIm[b] / (sxx[b] + 1e-20);
      }
    }
    const double meanCoh = cohBins > 0 ? cohSum / cohBins : 0.0;

    // GCC-PHAT alignment probe (diagnostic, ~free): ifft of the phase-
    // normalized cross-spectrum peaks at the ref->mic delay. alignedRef is
    // SUPPOSED to be sample-aligned with the mic's echo; a consistent
    // nonzero lag here means the latency compensation is off by that many
    // samples — which degrades HF cancellation first (3 samples is already
    // ~90 degrees at 8 kHz) and can flip apparent polarity. Logged every
    // seed so drift is visible across sessions.
    {
      std::vector<float> phat(kWienerWin, 0.0f);
      for (int b = 1; b < nBins; ++b) {
        const double mag =
            std::sqrt(sxyRe[b] * sxyRe[b] + sxyIm[b] * sxyIm[b]) + 1e-20;
        phat[2 * b] = static_cast<float>(sxyRe[b] / mag);
        phat[2 * b + 1] = static_cast<float>(sxyIm[b] / mag);
      }
      FFT::ifft(phat.data(), kWienerWin);
      int bestLag = 0;
      float bestV = -1e30f;
      for (int l = -256; l <= 256; ++l) {
        const int idx = (l + kWienerWin) % kWienerWin;
        if (phat[idx] > bestV) {
          bestV = phat[idx];
          bestLag = l;
        }
      }
      aecLog("[SEED] ref->mic alignment lag: %+d samples (%.2f ms)\n",
             bestLag, bestLag * 1000.0f / 48000.0f);
      // Closed-loop auto-correction: fold the measured residual into the
      // running correction the AEC's reference read applies. Negative lag
      // (mic leads ref => reference read too late) reduces the effective
      // delay. Gated on usable coherence and a sane magnitude; clamped so
      // one garbage pass can never run the read out of range. Each landed
      // seed re-measures AGAINST the corrected reference, so the reading
      // trends to zero when the sign/loop is right — and would visibly
      // diverge in this very log if it were wrong (self-validating).
      if (meanCoh >= 0.2 && bestLag != 0 && std::abs(bestLag) <= 250) {
        const int64_t prev =
            mAlignLagCorrection.load(std::memory_order_relaxed);
        int64_t next = prev + bestLag;
        if (next > 480) next = 480;
        if (next < -480) next = -480;
        mAlignLagCorrection.store(next, std::memory_order_relaxed);
        aecLog("[SEED] align auto-correct: %+lld -> %+lld samples\n",
               (long long)prev, (long long)next);
      }
    }

    // Floor raised 0.05 -> 0.15: a meanCoh=0.06 seed (near-end-dominated
    // capture at ~0 dB SER) synthesized mostly-garbage output yet fit
    // alpha~1 downstream — the beats-current landing gate now catches that
    // too, but a seed this incoherent is never worth the fit pass.
    if (meanCoh >= 0.15) {
      // Pass B: synthesize E = H (x) ref with hann-weighted overlap-add.
      // Analysis hann applied once + 75% overlap => constant OLA gain of
      // exactly 2.0 for this window/hop pair; divide it out.
      std::vector<double> acc(Pu, 0.0);
      for (size_t start = 0; start < Pu; start += kWienerHop) {
        for (int i = 0; i < kWienerWin; ++i) {
          const size_t idx = (start + i) % Pu;
          xw[i] = refPeriod[idx] * hann[i];
        }
        FFT::fft(xw.data(), kWienerWin);
        for (int b = 0; b < nBins; ++b) {
          const double xr = xw[2 * b], xi = xw[2 * b + 1];
          const double zr = hRe[b] * xr - hIm[b] * xi;
          const double zi = hRe[b] * xi + hIm[b] * xr;
          xw[2 * b] = static_cast<float>(zr);
          xw[2 * b + 1] = static_cast<float>(zi);
        }
        FFT::ifft(xw.data(), kWienerWin);
        for (int i = 0; i < kWienerWin; ++i) {
          const size_t idx = (start + i) % Pu;
          acc[idx] += xw[i];
        }
      }
      for (size_t phi = 0; phi < Pu; ++phi)
        mSeedOutput[phi] = static_cast<float>(acc[phi] * 0.5);
      wienerUsed = true;
      aecLog("[SEED] one-loop Wiener seed (meanCoh=%.2f)\n", meanCoh);
    } else {
      aecLog("[SEED] Wiener coherence too low (%.2f) — IR fallback\n",
             meanCoh);
    }
  }

  if (!wienerUsed) {
    if (ir.empty() || L == 0) {
      // No usable Wiener estimate AND no calibrated IR: emit a zero seed —
      // the self-fit measures alpha ~ 0 and discards it cleanly.
      std::fill(mSeedOutput.begin(), mSeedOutput.begin() + Pu, 0.0f);
      return;
    }
    std::vector<float> kernel(ir.begin(), ir.begin() + L);
    std::vector<float> seed;
    aec_conv::circularConvolve(refPeriod, kernel, seed);
    for (size_t phi = 0; phi < Pu; ++phi)
      mSeedOutput[phi] = seed[phi];
  }

  // SANITIZE, then clamp. The old peak-scale pass was NaN-blind:
  // std::max(x, NaN) keeps x (NaN comparisons are false), so non-finite
  // taps sailed past the clamp INTO the template — the template then
  // subtracts NaN at those phases forever, poisoning every downstream
  // consumer (recordings included: the TB330FU "brick wall" takes, where
  // the first loop recorded after a fresh calibration carried NaN and
  // FLT_MAX samples). A single huge-finite tap was as bad in the other
  // direction: scale = kMaxSeedAbs/1e38 collapsed the whole seed to zero.
  // Per-tap: non-finite -> 0, |tap| > kMaxSeedAbs -> clamped, count+log so
  // the upstream estimator fault (per-bin Sxy/Sxx with an empty bin is the
  // prime suspect) stays visible instead of laundered.
  {
    size_t badTaps = 0;
    for (size_t phi = 0; phi < Pu; ++phi) {
      float v = mSeedOutput[phi];
      if (!std::isfinite(v)) {
        mSeedOutput[phi] = 0.0f;
        ++badTaps;
      } else if (std::fabs(v) > kMaxSeedAbs) {
        mSeedOutput[phi] = v > 0 ? kMaxSeedAbs : -kMaxSeedAbs;
        ++badTaps;
      }
    }
    if (badTaps > 0) {
      aecLog("[SEED SCRUB] %zu/%zu taps non-finite or beyond %.1f — seed "
             "estimator emitted garbage (source=%s)\n",
             badTaps, Pu, kMaxSeedAbs, wienerUsed ? "wiener" : "ir-conv");
#ifdef __ANDROID__
      __android_log_print(ANDROID_LOG_ERROR, "FlutterRecorder",
                          "[SEED SCRUB] %zu/%zu bad taps (source=%s)",
                          badTaps, Pu, wienerUsed ? "wiener" : "ir-conv");
#endif
    }
  }
}

// ---- Per-track exact subtraction (see header doc comment) -----------------

int SynchronousEchoTemplate::findTrackSlot(int trackIndex) const {
  if (trackIndex < 0) return -1;
  for (int i = 0; i < kMaxTracks; ++i) {
    if (mTrackContributions[i].slotIndex.load(std::memory_order_acquire) ==
        trackIndex)
      return i;
  }
  return -1;
}

int SynchronousEchoTemplate::findOrAllocTrackSlot(int trackIndex) {
  if (trackIndex < 0) return -1;
  const int existing = findTrackSlot(trackIndex);
  if (existing >= 0) return existing;
  // Claim the first free slot via CAS — lock-free, safe to race against
  // other registerTrackAudio calls (only ever called off the audio thread,
  // but findTrackSlot()/lookups above ARE called from the audio thread, so
  // this whole scheme has to be lock-free end to end).
  for (int i = 0; i < kMaxTracks; ++i) {
    int expected = -1;
    if (mTrackContributions[i].slotIndex.compare_exchange_strong(
            expected, trackIndex, std::memory_order_acq_rel)) {
      return i;
    }
    // Lost the race, or this slot was already trackIndex's (another thread
    // registered the same track concurrently) — either way, re-check.
    if (expected == trackIndex) return i;
  }
  return -1; // table full
}

void SynchronousEchoTemplate::releaseTrackSlot(int trackIndex) {
  const int slot = findTrackSlot(trackIndex);
  if (slot < 0) return;
  TrackContribution &tc = mTrackContributions[slot];
  tc.computed.store(false, std::memory_order_release);
  tc.active = false;
  tc.targetGain = 1.0f;
  tc.appliedGain = 0.0f;
  tc.E.clear();
  tc.slotIndex.store(-1, std::memory_order_release);
}

void SynchronousEchoTemplate::registerTrackAudio(int trackIndex,
                                                 const float *audioMono,
                                                 int64_t frames) {
  if (!kFeedForwardEnabled)
    return;
  if (trackIndex < 0 || !audioMono || frames <= 0)
    return;
  const int slot = findOrAllocTrackSlot(trackIndex);
  if (slot < 0) {
    aecLog("[LSAEC] track %d registration FAILED: slot table full (%d "
           "tracks already registered this session)\n",
           trackIndex, kMaxTracks);
    return;
  }
  // The job carries its OWN period (audio.size()) rather than reading
  // mActiveLoopFrames here — this method can be called from any thread
  // (e.g. Dart's registration FFI call), and mActiveLoopFrames is a plain
  // (non-atomic) audio-thread-owned int; reading it from an arbitrary
  // thread would be a data race. setTrackActive() below defensively
  // verifies the computed contribution's length still matches the CURRENT
  // composite period before applying it, so a stale-period registration
  // is a safe no-op rather than a misaligned corruption.
  // Retain the audio in the slot so the contribution can be recomputed
  // later if the IR arrives/changes after this registration (see
  // requeueAllTrackContributions). Written on the Dart thread only.
  mTrackContributions[slot].audio.assign(audioMono, audioMono + frames);

  // Measure this track's length as a multiple of the engine BASE period
  // (safe any-thread read of the dedicated atomic). Rounded: overdub
  // lengths can land a few frames off an exact multiple.
  {
    const int64_t baseP = mEngineLoopFrames.load(std::memory_order_relaxed);
    int mult = 1;
    if (baseP > 0) {
      const double m = static_cast<double>(frames) / static_cast<double>(baseP);
      mult = static_cast<int>(m + 0.5);
      if (mult < 1) mult = 1;
      if (mult > 8) mult = 8;
    }
    mTrackContributions[slot].lengthMultiple.store(mult,
                                                   std::memory_order_relaxed);
    if (mult > 1)
      aecLog("[LSAEC] track %d is a %dx-length loop (%lld frames vs base "
             "%lld)\n",
             trackIndex, mult, (long long)frames, (long long)baseP);
  }

  TrackRegJob job;
  job.trackIndex = trackIndex;
  job.audio.assign(audioMono, audioMono + frames);
  {
    std::lock_guard<std::mutex> lock(mTrackJobMutex);
    // A track can be re-registered (e.g. after a split/cut-in-half changes
    // its content) — drop any earlier still-pending job for the same
    // track so the worker doesn't waste time computing a superseded one.
    mPendingTrackJobs.erase(
        std::remove_if(mPendingTrackJobs.begin(), mPendingTrackJobs.end(),
                       [trackIndex](const TrackRegJob &j) {
                         return j.trackIndex == trackIndex;
                       }),
        mPendingTrackJobs.end());
    mPendingTrackJobs.push_back(std::move(job));
  }
  // Mark not-yet-computed / not-active so a stale contribution from a
  // PRIOR registration of this same track can't be toggled active while
  // the new job is still pending.
  mTrackContributions[slot].computed.store(false, std::memory_order_release);
  mTrackContributions[slot].active = false;
}

void SynchronousEchoTemplate::requeueAllTrackContributions() {
  std::lock_guard<std::mutex> lock(mTrackJobMutex);
  for (int slot = 0; slot < kMaxTracks; ++slot) {
    TrackContribution &tc = mTrackContributions[slot];
    const int ti = tc.slotIndex.load(std::memory_order_acquire);
    if (ti < 0 || tc.audio.empty())
      continue;
    // Drop any still-pending job for this track, then queue a fresh one
    // from the retained audio so it recomputes against the current IR.
    mPendingTrackJobs.erase(
        std::remove_if(mPendingTrackJobs.begin(), mPendingTrackJobs.end(),
                       [ti](const TrackRegJob &j) { return j.trackIndex == ti; }),
        mPendingTrackJobs.end());
    TrackRegJob job;
    job.trackIndex = ti;
    job.audio = tc.audio; // copy; worker computes from its own copy
    mPendingTrackJobs.push_back(std::move(job));
    tc.computed.store(false, std::memory_order_release);
    aecLog("[LSAEC] track %d re-queued for recompute (IR changed)\n", ti);
  }
}

void SynchronousEchoTemplate::computeTrackContribution(
    int trackIndex, const std::vector<float> &audio) {
  if (trackIndex < 0 || audio.empty())
    return;
  // registerTrackAudio() already allocated (or found) this track's slot
  // before queueing the job — this just re-resolves the same slot
  // (idempotent lookup, not a new allocation).
  const int slot = findTrackSlot(trackIndex);
  if (slot < 0) {
    aecLog("[LSAEC] track %d contribution SKIPPED: no slot found (should "
           "be unreachable — registerTrackAudio allocates before queueing)\n",
           trackIndex);
    return;
  }
  std::vector<float> ir;
  {
    std::lock_guard<std::mutex> lock(mSeedIRMutex);
    ir = mSeedIR;
  }
  TrackContribution &tc = mTrackContributions[slot];
  if (ir.empty()) {
    tc.E.clear();
    // Not a real seed available -- this session hasn't run a live
    // calibration yet (loading a SAVED calibration only restores
    // delay/offset, not the impulse response -- a known limitation). Falls
    // back to the composite template's existing behavior for this track
    // until a live calibration runs. Logged once per attempt since a
    // silent no-op here is exactly the kind of gap that looks like "the
    // fix isn't working" when it's actually "never got a chance to."
    aecLog("[LSAEC] track %d registration skipped: no calibrated IR yet "
           "(run a live calibration this session)\n", trackIndex);
    return;
  }
  // The template is phase-indexed to the CURRENT composite period, not to
  // whatever length this track's own recording happened to land at. A
  // track's raw audio can legitimately span multiple base-loop periods
  // (an Nx overdub) — and per a separate, known recording-pipeline issue,
  // can occasionally land on a non-exact-multiple length. Either way, fold
  // it down to exactly one period by summing every P-th sample before
  // convolving, so setTrackActive()'s later length check actually matches
  // instead of silently skipping the toggle (which looks identical to the
  // per-track fix "not working" — this was live-tested and confirmed: an
  // overdub landed at 239937 frames against a 136704-frame base period,
  // and its mute/unmute toggle never applied).
  const int64_t periodSigned =
      mActiveLoopFramesAtomic.load(std::memory_order_relaxed);
  const size_t rawLen = audio.size();
  const size_t P = (periodSigned > 0)
                        ? std::min(static_cast<size_t>(periodSigned), rawLen)
                        : rawLen;

  // A track spanning >= 2 fold periods would have its DISTINCT periods
  // averaged by the fold below — subtracting that average is wrong half
  // the time (worse than nothing). Leave it uncomputed: the composite
  // period multiplier grows the template to this track's true period and
  // EMA/reseed cancel it there; the exact edit returns on the next
  // recompute once the active period matches (P == rawLen -> identity).
  if (rawLen >= 2 * P) {
    tc.E.clear();
    aecLog("[LSAEC] track %d contribution deferred: %zu frames spans "
           "multiple %zu-frame periods (composite path covers it)\n",
           trackIndex, rawLen, P);
    return;
  }

  std::vector<float> folded(P, 0.0f);
  for (size_t i = 0; i < rawLen; ++i)
    folded[i % P] += audio[i];

  // Rotate onto the template's MIC-phase time base: registered audio is
  // OUTPUT-clock (sample 0 plays at loop phase 0) but alignedRef — and
  // therefore the seed and the learned template — sees the output LAMBDA
  // frames late (acoustic delay + one callback buffer). Without this the
  // contribution landed LAMBDA (~60 ms) EARLY, and subtracting a wrong-phase
  // echo estimate ADDS energy: the "cancelled my own sounds, kept the
  // bleed" regression. shifted[p] = folded[(p - LAMBDA) mod P].
  const int64_t lambdaRaw =
      mReferenceShiftFrames.load(std::memory_order_relaxed);
  const size_t lambda =
      static_cast<size_t>(((lambdaRaw % static_cast<int64_t>(P)) +
                           static_cast<int64_t>(P)) %
                          static_cast<int64_t>(P));
  if (lambda != 0) {
    std::vector<float> shifted(P);
    for (size_t p = 0; p < P; ++p)
      shifted[p] = folded[(p + P - lambda) % P];
    folded.swap(shifted);
  }

  // Same truncation rationale as computeSeedConvolution: a period-P
  // circular convolution is only well-posed with a kernel <= P taps.
  const size_t L = std::min(ir.size(), P);

  float audioPeak = 0.0f;
  for (size_t phi = 0; phi < P; ++phi)
    audioPeak = std::max(audioPeak, std::fabs(folded[phi]));

  // FFT circular convolution (worker thread) — see computeSeedConvolution;
  // the direct loop cost ~1 s per registered track, which was the gap
  // between "overdub launches" and "its echo model is live".
  std::vector<float> kernel(ir.begin(), ir.begin() + L);
  std::vector<float> mono;
  aec_conv::circularConvolve(folded, kernel, mono);
  float contribPeak = 0.0f;
  for (size_t phi = 0; phi < P; ++phi)
    contribPeak = std::max(contribPeak, std::fabs(mono[phi]));
  // Do NOT normalize the contribution to the track's own amplitude. The
  // acoustic echo legitimately EXCEEDS the digital source when the mic gain
  // is hot (measured on real hardware: echoGain ~1.5, IR⊛audio peaking at
  // ~13x the source) — clamping to the source peak structurally under-
  // cancels, so muting only removes a sliver and the ghost survives. The
  // calibrated IR already encodes the true echo gain; trust it. Keep only a
  // loose ABSOLUTE guard against a pathological/NaN IR producing a runaway
  // subtraction (full-scale is 1.0; >8x full-scale is not a real echo).
  constexpr float kMaxContribAbs = 8.0f;
  if (contribPeak > kMaxContribAbs) {
    const float scale = kMaxContribAbs / contribPeak;
    for (size_t phi = 0; phi < P; ++phi)
      mono[phi] *= scale;
  }

  // Expand mono -> interleaved, matching mTemplate's layout
  // (E[phi*channels+ch]) at the CURRENT active (effective) period. A 1x
  // track under a grown composite period (a 2x/4x overdub is playing)
  // repeats within it — TILE the P-length contribution up to activeP so
  // setTrackActive's length check matches and the exact edit still lands.
  // A MULTI-period track (rawLen > activeP, i.e. registered before the
  // period grew, or clamped by capacity) cannot be represented at activeP:
  // leave it uncomputed — the composite EMA/reseed path covers it
  // (harmless, just not instant), instead of applying a folded average
  // whose subtraction is wrong half the time.
  const size_t activeP = static_cast<size_t>(
      std::max<int64_t>(mActiveLoopFramesAtomic.load(std::memory_order_relaxed),
                        static_cast<int64_t>(P)));
  if (activeP % P != 0) {
    tc.E.clear();
    aecLog("[LSAEC] track %d contribution NOT applied: length %zu doesn't "
           "divide active period %zu\n",
           trackIndex, P, activeP);
    return;
  }
  tc.E.assign(activeP * mChannels, 0.0f);
  for (size_t phi = 0; phi < activeP; ++phi) {
    const size_t base = phi * mChannels;
    for (unsigned int ch = 0; ch < mChannels; ++ch)
      tc.E[base + ch] = mono[phi % P];
  }
  tc.computed.store(true, std::memory_order_release);
  aecLog("[LSAEC] track %d contribution computed: raw=%zu folded-to-P=%zu "
         "peak=%.4f (audio peak=%.4f)\n",
         trackIndex, rawLen, P, contribPeak, audioPeak);
}

bool SynchronousEchoTemplate::setTrackActive(int trackIndex, bool active) {
  if (!kFeedForwardEnabled)
    return false;
  if (trackIndex < 0)
    return false;
  // DART THREAD: atomic-only validation, then enqueue. mTemplate and
  // tc.active/appliedGain are audio-thread-only, so the O(P) edit is deferred
  // to drainPendingTrackEdits() (top of process()). Returning true lets the
  // caller skip the reference-changed reseed; a period change between here and
  // the drain drops the edit (size re-check in applyTrackActive) and triggers
  // its own reseed, so the skip is safe.
  const int slot = findTrackSlot(trackIndex);
  if (slot < 0)
    return false; // never registered — composite template covers it
  // Record the DESIRED state regardless of compute progress: the drain
  // reconciles it once the contribution is ready (deferred activation).
  mTrackContributions[slot].wantActive.store(active, std::memory_order_relaxed);
  if (!mTrackContributions[slot].computed.load(std::memory_order_acquire))
    return false; // not ready yet — will auto-apply on compute completion
  {
    std::lock_guard<std::mutex> lk(mPendingEditsMutex);
    mPendingTrackEdits.push_back({trackIndex, /*isGain=*/false, active, 0.0f});
  }
  return true;
}

// AUDIO THREAD ONLY (via drainPendingTrackEdits): the exact toggle edit.
void SynchronousEchoTemplate::applyTrackActive(int trackIndex, bool active) {
  const int slot = findTrackSlot(trackIndex);
  if (slot < 0)
    return;
  TrackContribution &tc = mTrackContributions[slot];
  if (!tc.computed.load(std::memory_order_acquire))
    return;
  if (tc.active == active)
    return; // already in the target state
  // Defensive: the contribution's length must still match the CURRENT
  // composite template's active span, or this track's audio was registered
  // against a period that's since changed. Applying a mismatched-length
  // contribution would misalign phase and silently corrupt the template.
  const size_t expected =
      static_cast<size_t>(mActiveLoopFrames) * mChannels;
  if (tc.E.size() != expected) {
    aecLog("[LSAEC] track %d toggle SKIPPED (size mismatch: have %zu, "
           "need %zu — period changed since registration)\n",
           trackIndex, tc.E.size(), expected);
    return;
  }
  // Activate at the track's CURRENT mixer gain scaled by the fitted IR
  // trust (contributions come from the same IR as the seed and inherit its
  // measured sign/scale error); deactivate removes exactly what was applied.
  const float effective = tc.targetGain * mIrFitScale;
  const float gain = active ? effective : -tc.appliedGain;
  for (size_t i = 0; i < tc.E.size(); ++i)
    mTemplate[i] += gain * tc.E[i];
  tc.appliedGain = active ? effective : 0.0f;
  // NOTE: deliberately do NOT stamp mConfidence here. An earlier version
  // pinned every phase to kSeedConfidence on each toggle to "protect" the
  // exact edit — but that made the WHOLE template max-confidence, which (a)
  // makes the E3 double-talk freeze eligible everywhere so learning stalls
  // and cancellation visibly degrades over the next several loops, and (b)
  // blocks the very learning that would finish the job. It turns out no
  // protection is needed: right after a mute the mic no longer contains
  // that track's echo, so ordinary Pass-2 learning already pulls E[phi] in
  // the SAME direction as this subtraction — it reinforces the edit and
  // cleans up any residual (e.g. from the amplitude clamp) within ~1 pass,
  // rather than diluting it. Leaving confidence untouched keeps the
  // template adaptive.
  tc.active = active;
  aecLog("[LSAEC] track %d exact-subtraction toggle: active=%d gain=%.3f\n",
         trackIndex, active ? 1 : 0, tc.appliedGain);
  // The exact edit removes only the IR-PREDICTED (linear) part of this
  // track's echo — measured on real full-volume captures, that's only ~1/4
  // of the bleed; the composite template has LEARNED the rest (nonlinear,
  // speaker-distorted) and keeps subtracting it after a mute: the audible
  // "muted track still in the template" ghost. Two-part cleanup, both made
  // safe by the beats-current seed gate:
  //  1. Trim confidence so per-pass EMA unlearns the nonlinear remainder
  //     at a re-heated alpha (~0.2 vs the converged 0.06) for the next
  //     couple passes, then anneals back.
  //  2. Self-arm a reseed of the NEW mix — callers still skip their own
  //     notify (the old livelock reason), but one recapture per applied
  //     toggle is cheap and the landing gate guarantees it only replaces
  //     the template if it actually beats the ghost-y current state.
  {
    const size_t n =
        std::min(static_cast<size_t>(mActiveLoopFrames), mCapacityFrames);
    for (size_t phi = 0; phi < n; ++phi)
      mConfidence[phi] *= 0.35f;
    mConfidenceSum *= 0.35;
    notifyReferenceChanged();
  }
}

bool SynchronousEchoTemplate::setTrackGain(int trackIndex, float gain) {
  if (!kFeedForwardEnabled)
    return false;
  if (trackIndex < 0)
    return false;
  // DART THREAD: validate (atomic-only) + enqueue; the audio thread records
  // tc.targetGain and applies the exact gain delta in applyTrackGain.
  const int slot = findTrackSlot(trackIndex);
  if (slot < 0)
    return false;
  {
    std::lock_guard<std::mutex> lk(mPendingEditsMutex);
    mPendingTrackEdits.push_back({trackIndex, /*isGain=*/true, false, gain});
  }
  return true;
}

// AUDIO THREAD ONLY (via drainPendingTrackEdits): record the target gain and,
// if the track is live, apply the exact (target - applied) delta.
void SynchronousEchoTemplate::applyTrackGain(int trackIndex, float gain) {
  const int slot = findTrackSlot(trackIndex);
  if (slot < 0)
    return;
  TrackContribution &tc = mTrackContributions[slot];
  tc.targetGain = gain;
  if (!tc.active)
    return; // nothing summed in — the recorded target covers activation
  if (!tc.computed.load(std::memory_order_acquire))
    return;
  const size_t expected =
      static_cast<size_t>(mActiveLoopFrames) * mChannels;
  if (tc.E.size() != expected)
    return;
  const float effective = gain * mIrFitScale;
  const float delta = effective - tc.appliedGain;
  if (delta != 0.0f) {
    for (size_t i = 0; i < tc.E.size(); ++i)
      mTemplate[i] += delta * tc.E[i];
  }
  tc.appliedGain = effective;
  aecLog("[LSAEC] track %d exact gain edit: applied=%.3f\n", trackIndex, gain);
}

// AUDIO THREAD ONLY: drain the Dart-enqueued per-track edits and apply them to
// mTemplate in FIFO order. try_lock so the render thread never blocks; a
// contended block just drains next callback (edits aren't sample-critical).
void SynchronousEchoTemplate::drainPendingTrackEdits() {
  // Deferred-activation reconcile: a contribution that finished computing
  // AFTER its setTrackActive call applies here, on the audio thread, the
  // callback after the worker completes. O(kMaxTracks) atomic loads.
  int maxMult = 1;
  for (int i = 0; i < kMaxTracks; ++i) {
    TrackContribution &tc = mTrackContributions[i];
    const int idx = tc.slotIndex.load(std::memory_order_acquire);
    if (idx < 0)
      continue;
    // Composite-period multiplier: the LCM of ACTIVE (playing) tracks'
    // periods. Multiples are powers of two (x1/2/4/8 picker), so LCM ==
    // max. Tracked regardless of `computed` — the AUDIO is periodic at the
    // track's length whether or not its exact-edit contribution is ready.
    if (tc.wantActive.load(std::memory_order_relaxed)) {
      const int m = tc.lengthMultiple.load(std::memory_order_relaxed);
      if (m > maxMult) maxMult = m;
    }
    if (!tc.computed.load(std::memory_order_acquire))
      continue;
    const bool want = tc.wantActive.load(std::memory_order_relaxed);
    if (want != tc.active)
      applyTrackActive(idx, want);
  }
  mDesiredMultiplier = maxMult;
  std::unique_lock<std::mutex> lk(mPendingEditsMutex, std::try_to_lock);
  if (!lk.owns_lock() || mPendingTrackEdits.empty())
    return;
  mDrainScratch.swap(mPendingTrackEdits); // take ownership; leave pending empty
  lk.unlock();
  for (const auto &e : mDrainScratch) {
    if (e.isGain)
      applyTrackGain(e.trackIndex, e.gain);
    else
      applyTrackActive(e.trackIndex, e.active);
  }
  mDrainScratch.clear();
}

float SynchronousEchoTemplate::meanConfidence() const {
  if (mActiveLoopFrames <= 0)
    return 0.0f;
  // mActiveLoopFrames can be left set to an OVERSIZED value by process()'s
  // capacity guard (a loop period beyond kMaxSeconds falls back to pure
  // passthrough and stores the raw, over-capacity loopFrames for telemetry
  // continuity) — clamp the divisor the same way the old O(P) loop did.
  // mConfidenceSum is an incrementally-maintained running sum over the
  // active window (every write site updates it by delta — see the header
  // comment) rather than re-summed here every call: this was previously an
  // O(min(P,capacity)) loop — up to 768,000 iterations for a 16s/48kHz
  // loop — called unconditionally on the audio thread ~4x/sec for E5
  // telemetry (live in every production build, not just AEC_DEBUG_LOGGING
  // builds) plus again as an aecLog() argument (evaluated regardless of
  // whether logging is compiled in, since aecLog is a plain function and
  // C++ evaluates arguments before the no-op call).
  const size_t n =
      std::min(static_cast<size_t>(mActiveLoopFrames), mCapacityFrames);
  if (n == 0) return 0.0f;
  // Normalize by the ANNEAL scale, not the saturation cap: kConfMax (4096)
  // exists only to stop the counter; learning behavior is governed by
  // kConfTau (8) and is fully annealed by c ~ 24 (kSeedConfidence). The old
  // /kConfMax normalization reported a fully-converged template as 0.6% --
  // "why is confidence always 1%?" was a display artifact, not a
  // convergence problem. 1 - exp(-c/tau): seeded ~95%, fresh 0%.
  const double meanC = mConfidenceSum / static_cast<double>(n);
  return static_cast<float>(1.0 - std::exp(-meanC / kConfTau));
}

void SynchronousEchoTemplate::process(float *micInOut, const float *alignedRef,
                                      unsigned int frameCount,
                                      unsigned int channels,
                                      int64_t blockStartFrame,
                                      int64_t loopFrames,
                                      int64_t loopStartFrame, bool learn) {
  if (!micInOut || frameCount == 0)
    return;

  // Apply any per-track template edits enqueued from the Dart thread (#54):
  // mTemplate and tc.active/appliedGain are audio-thread-only, so
  // setTrackActive/setTrackGain deferred their O(P) edits to here. Runs before
  // any mTemplate read below so this block sees a consistent template.
  drainPendingTrackEdits();

  // Double-talk REWIND: on the detector's rising edge, roll back the last
  // ~400ms of learned updates (newest first) — the detector's latency
  // window, which otherwise bakes the onset of the performer's sound into
  // E[phi]. Chunked across callbacks (learning is already frozen while the
  // hold is up, so nothing new interleaves).
  {
    const bool hold = SpectralGovernor::instance().nearEndHold();
    if (hold && !mPrevNearEndHold) {
      // Rate limit (3.5 Hz warble fix): one full rollback per cooldown.
      // A hold edge inside the cooldown still freezes learning (the hold
      // gates `learn` upstream) but does NOT roll back state — it was the
      // unconditional per-edge rollback that turned governor chatter into
      // a ~10 dB cancellation square wave. The genuine use (undo the
      // detector-latency window of a real performer onset) fires at most
      // once per cooldown, and the E3 per-block spike freeze still guards
      // the blocks in between.
      const int64_t kRewindCooldownFrames =
          static_cast<int64_t>(mSampleRate) * 2; // 2 s
      // PARTIAL rewind (transient-bleed fix): the rewind exists to undo the
      // DETECTOR-LATENCY window — the ~50-110 ms of performer onset learned
      // before the (debounced) hold engaged. Rolling back the full 400 ms
      // ring on every hold was measured live undoing the freshest learning
      // ~every cooldown during busy playing — exactly the transient-phase
      // refinements — so hits never stayed converged ("bleed on
      // transients"). 150 ms covers the worst engage latency with margin
      // while preserving the rest of the recently-learned template.
      const int64_t kRewindMaxEntries =
          (static_cast<int64_t>(mSampleRate) * 150) / 1000;
      if (blockStartFrame - mLastRewindFrame >= kRewindCooldownFrames) {
        mRewindRemaining =
            std::min(static_cast<int64_t>(mUndoCount), kRewindMaxEntries);
        mLastRewindFrame = blockStartFrame;
        ++mRewindCount;
        aecLog("[LSAEC] double-talk rewind #%u armed (%lld of %lld entries)\n",
               mRewindCount, (long long)mRewindRemaining,
               (long long)mUndoCount);
      } else {
        aecLog("[LSAEC] rewind SUPPRESSED (cooldown) — freeze only\n");
      }
    }
    mPrevNearEndHold = hold;
    if (mRewindRemaining > 0) {
      constexpr int64_t kRewindChunk = 4096;
      int64_t todo = std::min(mRewindRemaining, kRewindChunk);
      while (todo-- > 0) {
        mUndoHead = (mUndoHead + mUndoRing.size() - 1) % mUndoRing.size();
        const LearnUndo &u = mUndoRing[mUndoHead];
        const size_t base = static_cast<size_t>(u.phi) * mChannels;
        for (unsigned int ch = 0; ch < mChannels && ch < 2; ++ch)
          mTemplate[base + ch] = u.oldT[ch];
        mConfidenceSum += static_cast<double>(u.oldConf) -
                          static_cast<double>(mConfidence[u.phi]);
        mConfidence[u.phi] = u.oldConf;
        --mUndoCount;
        --mRewindRemaining;
      }
      if (mRewindRemaining == 0)
        aecLog("[LSAEC] double-talk rewind complete\n");
    }
  }

  // Composite-period multiplier: the engine reports the BASE loop period,
  // but the composite echo repeats at the LCM of the ACTIVE tracks'
  // lengths (max power-of-two multiple x base — see lengthMultiple).
  // Running at the base period made a 2x overdub's echo alternate halves
  // every pass: the per-phase average cancelled neither ("second loop
  // doesn't cancel at all"). Halving under capacity pressure keeps the
  // effective period an exact multiple of every SHORTER track's period, so
  // those still cancel; only the over-long track degrades.
  mEngineLoopFrames.store(loopFrames, std::memory_order_relaxed);
  if (loopFrames > 0 && mDesiredMultiplier > 1) {
    int mult = mDesiredMultiplier;
    while (mult > 1 &&
           static_cast<size_t>(loopFrames) * static_cast<size_t>(mult) >
               mCapacityFrames)
      mult >>= 1;
    if (mult > 1)
      loopFrames *= mult;
  }

  // No loop yet, or period beyond capacity -> passthrough. A period beyond
  // kMaxSeconds (16s) is a SILENT, total cancellation outage — zero taps
  // applied, raw mic straight through — confirmed on-device as the cause of
  // "ghost" bleed-through of prior loop layers into a new take once a
  // composite period grows past this cap. Surface it (rate-limited log +
  // isOverCapacity() telemetry flag) so it's diagnosable instead of looking
  // like a mystery regression.
  if (loopFrames <= 0 ||
      static_cast<size_t>(loopFrames) > mCapacityFrames ||
      channels > mChannels) {
    const bool wasOverCapacity = mOverCapacity;
    mOverCapacity =
        static_cast<size_t>(loopFrames) > mCapacityFrames; // not the ch/0 cases
    if (mOverCapacity && !wasOverCapacity) {
      aecLog("[LSAEC] CAPACITY EXCEEDED: loopFrames=%lld > cap=%zu — "
             "cancellation OFF (passthrough) until period drops back under "
             "%us\n",
             (long long)loopFrames, mCapacityFrames, kMaxSeconds);
    }
    mActiveLoopFrames = (loopFrames > 0) ? loopFrames : 0;
    mActiveLoopFramesAtomic.store(mActiveLoopFrames, std::memory_order_relaxed);
    return;
  }
  mOverCapacity = false;

  // Loop period changed (a layer added/removed). NOT always a wipe anymore:
  // looper workflows change period between INTEGER MULTIPLES constantly
  // (base P -> 2P take added -> back to P on mute), and wiping on every
  // transition meant the template lived only seconds at a time in a real
  // session — measured live as leak 0.3-0.7 with boost pinned at max,
  // conf 0%, ERLE ~6dB, seed re-arming 11 times in minutes: convergence
  // could never PERSIST. But an echo that is periodic at P is exactly
  // periodic at kP too, so:
  //   grow  P -> kP : TILE the template k times — every phase stays
  //                   correct, convergence carries over untouched.
  //   shrink kP -> P: FOLD the k segments by averaging — post-track-edit
  //                   they agree, and averaging reduces performer noise.
  //   unrelated     : clear, as before.
  if (loopFrames != mActiveLoopFrames) {
    const size_t span = static_cast<size_t>(loopFrames) * channels;
    const size_t pspan = static_cast<size_t>(loopFrames);
    const int64_t oldP = mActiveLoopFrames;
    if (oldP > 0 && loopFrames > oldP && loopFrames % oldP == 0) {
      const size_t k = static_cast<size_t>(loopFrames / oldP);
      const size_t oldSpan = static_cast<size_t>(oldP) * channels;
      for (size_t c = 1; c < k; ++c) {
        std::copy(mTemplate.begin(), mTemplate.begin() + oldSpan,
                  mTemplate.begin() + c * oldSpan);
        std::copy(mConfidence.begin(),
                  mConfidence.begin() + static_cast<size_t>(oldP),
                  mConfidence.begin() + c * static_cast<size_t>(oldP));
      }
      mConfidenceSum *= static_cast<double>(k);
      aecLog("[LSAEC] period grew %lldx: template TILED, convergence kept\n",
             (long long)k);
    } else if (oldP > 0 && loopFrames < oldP && oldP % loopFrames == 0) {
      const size_t k = static_cast<size_t>(oldP / loopFrames);
      const float inv = 1.0f / static_cast<float>(k);
      double newConfSum = 0.0;
      for (size_t i = 0; i < span; ++i) {
        float acc = 0.0f;
        for (size_t c = 0; c < k; ++c)
          acc += mTemplate[i + c * span];
        mTemplate[i] = acc * inv;
      }
      for (size_t i = 0; i < pspan; ++i) {
        float acc = 0.0f;
        for (size_t c = 0; c < k; ++c)
          acc += mConfidence[i + c * static_cast<size_t>(loopFrames)];
        const float folded = acc * inv;
        mConfidence[i] = folded;
        newConfSum += folded;
      }
      mConfidenceSum = newConfSum;
      aecLog("[LSAEC] period shrank %lldx: template FOLDED, convergence kept\n",
             (long long)k);
    } else {
      std::fill(mTemplate.begin(), mTemplate.begin() + span, 0.0f);
      std::fill(mConfidence.begin(), mConfidence.begin() + pspan, 0.0f);
      // The clear above zeroes exactly the NEW active window [0, pspan), so
      // the running sum is exactly 0 immediately after — matches the
      // mConfidenceSum invariant (sum over the CURRENT active window).
      mConfidenceSum = 0.0;
    }
    mActiveLoopFrames = loopFrames;
    mActiveLoopFramesAtomic.store(loopFrames, std::memory_order_relaxed);
    mResidBaseline = 0.0f;
    mLearnedBlocks = 0;

    // The template was just cleared, so NO per-track contribution is summed
    // in anymore — but the slots' bookkeeping still said active. A later
    // deactivation would then SUBTRACT a contribution the template doesn't
    // contain (negative injection), and stale-period contributions would be
    // size-mismatched anyway. Reset the bookkeeping; contributions recompute
    // against the new period when the IR/requeue path fires. O(64) scan on
    // the audio thread — negligible.
    for (auto &tc : mTrackContributions) {
      tc.active = false;
      tc.appliedGain = 0.0f;
    }

    // A period change supersedes any pending mix-changed notification (the
    // capture below already covers the new state) and abandons any old-period
    // capture/job in flight: bump the generation so its late completion is
    // discarded at drain time (below) without touching mSeedBusy, which the
    // fresh arm immediately below is about to own for the NEW period.
    mReferenceChangePending.store(false, std::memory_order_relaxed);
    if (mSeedBusy) {
      ++mSeedGeneration;
      mSeedBusy = false;
      mSeedFitActive = false;
      mSeedFitDone = false;
      mSeedFitFrames = 0;
      mSeedAborts.fetch_add(1, std::memory_order_relaxed);
      aecLog("[SEED] aborted (period change)\n");
    }
    armSeedCaptureIfPossible(alignedRef);
  } else if (mReferenceChangePending.exchange(false, std::memory_order_relaxed)) {
    // Same period, but the audible mix changed (mute/unmute/pause/stop) — the
    // template's phase content is now stale even though P didn't move.
    // Re-arm WITHOUT touching mTemplate/mConfidence: unaffected phases keep
    // cancelling with what they have until the reseed lands and overwrites.
    //
    // If a capture/job is ALREADY in flight, its captured samples span up to
    // one full loop period of wall-clock time — long enough for a SECOND mix
    // change (another mute/unmute) to land mid-capture. Letting it "keep
    // running" (the previous behavior) means the capture straddles TWO
    // different mixes: early phases reflect the mix active when it was
    // armed, later phases reflect whatever's playing now. Convolving that
    // temporally-inconsistent capture against the calibrated IR bakes a
    // hybrid into E[phi] — confirmed on a real recording this session as an
    // audible "ghost" of a loop layer that wasn't even playing anymore, once
    // some phases were seeded from the stale portion. Abort and restart
    // clean, exactly like a period change does, so the eventual capture is
    // always a single, internally-consistent mix throughout.
    if (mSeedBusy) {
      ++mSeedGeneration;
      mSeedBusy = false;
      mSeedFitActive = false;
      mSeedFitDone = false;
      mSeedFitFrames = 0;
      mSeedAborts.fetch_add(1, std::memory_order_relaxed);
      aecLog("[SEED] aborted (reference change)\n");
    }
    armSeedCaptureIfPossible(alignedRef);
  }

  // Drain a finished seed job, chunked so any single callback's apply cost is
  // bounded regardless of loop length. Only a job stamped with the CURRENT
  // generation is applied — see mSeedGeneration comment in the header for why
  // period alone isn't a sufficient staleness check (a quick mute-then-unmute
  // can return to the same P). A stale job's completion is dropped WITHOUT
  // touching mSeedBusy, which by then belongs to whatever was armed after
  // the abandonment, not to this late-arriving job.
  if (mSeedOutputReady.load(std::memory_order_acquire)) {
    if (mSeedJobGeneration == mSeedGeneration) {
      if (!mSeedFitActive && !mSeedFitDone && mSeedApplyPos == 0 &&
          mSeedFitFrames == 0) {
        // Fit-before-trust: spend one loop pass measuring how the computed
        // seed actually fits the live mic before applying it (see the
        // mSeedFitActive doc in the header — an unfitted seed with a
        // wrong-sign/hot IR was measured ANTI-cancelling on device).
        mSeedFitActive = true;
        mSeedFitNum = 0.0;
        mSeedFitDen = 0.0;
        mSeedFitMicSq = 0.0;
        mSeedFitCurSq = 0.0;
        aecLog("[SEED] fit pass started\n");
      }
      if (!mSeedFitActive && mSeedFitDone) {
        const int64_t end =
            std::min(mSeedApplyPos + kSeedApplyChunk, mActiveLoopFrames);
        for (int64_t phi = mSeedApplyPos; phi < end; ++phi) {
          const float v =
              mSeedFitAlpha * mSeedOutput[static_cast<size_t>(phi)];
          const size_t base = static_cast<size_t>(phi) * channels;
          for (unsigned int ch = 0; ch < channels; ++ch)
            mTemplate[base + ch] = v;
          mConfidenceSum += kSeedConfidence - mConfidence[static_cast<size_t>(phi)];
          mConfidence[static_cast<size_t>(phi)] = kSeedConfidence;
        }
        mSeedApplyPos = end;
        if (mSeedApplyPos >= mActiveLoopFrames) {
          mSeedOutputReady.store(false, std::memory_order_relaxed);
          mSeedBusy = false;
          mSeedFitFrames = 0;
          mSeedFitDone = false;
          mSeedRetryCount = 0;
          mSeedLands.fetch_add(1, std::memory_order_relaxed);
          aecLog("[SEED] LANDED (alpha=%.3f)\n", mSeedFitAlpha);
        }
      }
    } else {
      mSeedOutputReady.store(false, std::memory_order_relaxed); // stale; drop only
      mSeedFitActive = false;
      mSeedFitDone = false;
      mSeedFitFrames = 0;
    }
  }

  const int64_t P = loopFrames;

  // Snapshot the live-tunable subtraction-gate params once per block (relaxed
  // atomics; a slider drag lands between callbacks, never mid-loop).
  const float sgAttack = mSubGateAttackRate.load(std::memory_order_relaxed);
  const float sgRelease = mSubGateReleaseRate.load(std::memory_order_relaxed);
  const float sgFloor = mSubGateFloorPow.load(std::memory_order_relaxed);

  // ---- Pass 1a: cancel (raw residual); accumulate block raw/mic energy.
  //
  // SAFETY INVARIANT, not a timer: correct cancellation can only ever REDUCE
  // energy toward zero OVER A BLOCK — subtracting a real echo from a signal
  // that contains it brings the result closer to the true near-end signal,
  // never further away, in aggregate. So a block's raw-residual energy
  // should never exceed its mic energy. The only way it can is if E[phi] is
  // WRONG (stale: still predicts echo for content that no longer exists,
  // e.g. a track that just muted/paused/stopped through ANY path, wired or
  // not). Subtracting a wrong estimate doesn't fail toward silence — it can
  // overshoot PAST zero into an inverted, full-strength copy of whatever was
  // stale, confirmed on-device this session as a full recording replaced by
  // a phase-inverted ghost of a paused loop layer. This suppression makes
  // that failure mode structurally impossible, independent of WHY the
  // template went stale — no notify(), no timer, no reseed race to get
  // right. It replaces a time-based "trust gate"
  // (mMixChangeFrame/mTrustRampFrames/trustGate(), removed) that tried to
  // guess how long to distrust the template; that guess was wrong by up to
  // an order of magnitude (100ms vs. a reseed that can take a full loop
  // pass) and user-confirmed not to work.
  //
  // Deliberately decided at BLOCK granularity, not per-sample: a per-sample
  // version of this exact invariant was tried first and produced broadband
  // static, confirmed on-device — near-end and echo interfere constantly,
  // so their instantaneous sum legitimately crosses a per-sample threshold
  // many times per cycle even when cancellation is CORRECT, not just when
  // it's stale. Same lesson mRefEnvelope's smoothed-vs-per-sample fix
  // already taught elsewhere in this file.
  const size_t totalSamples = static_cast<size_t>(frameCount) * channels;
  if (mRawResidual.size() < totalSamples)
    mRawResidual.resize(totalSamples); // grow-only; Pass 2's learning input

  double blockResid = 0.0;    // gated-output residual energy — E3 freeze / Pass-1b input
  double blockMicEnergy = 0.0;
  float lastRefGate = 1.0f;   // UI overlay feed — see mRefGateSmooth
  for (unsigned int f = 0; f < frameCount; ++f) {
    int64_t phi = (blockStartFrame + static_cast<int64_t>(f) - loopStartFrame) % P;
    if (phi < 0)
      phi += P; // C++ % can be negative; loop phase must be in [0, P)
    const size_t base = static_cast<size_t>(phi) * channels;

    // Reference-presence subtraction gate (see mSubGateEnv doc in header).
    // How much far-end (speaker) signal is actually present at THIS frame:
    // fast attack so cancellation re-engages instantly when playback starts,
    // slow release so the echo tail keeps cancelling as the reference decays.
    // When there's no reference to read, don't gate (refGate = 1) so this is
    // never a regression from the prior unconditional-subtraction behavior.
    float refGate = 1.0f;
    if (alignedRef) {
      float refPow = 0.0f;
      for (unsigned int ch = 0; ch < channels; ++ch) {
        const float r = alignedRef[f * channels + ch];
        refPow += r * r;
      }
      const float rate = (refPow > mSubGateEnv) ? sgAttack : sgRelease;
      mSubGateEnv += rate * (refPow - mSubGateEnv);
      refGate = mSubGateEnv / (mSubGateEnv + sgFloor); // soft 0..1
      lastRefGate = refGate;
    }

    for (unsigned int ch = 0; ch < channels; ++ch) {
      const size_t i = f * channels + ch;
      const float mic = micInOut[i];
      const float est = mTemplate[base + ch];
      // Pass 2 learns from the UNGATED LMS error (mic - est) so the estimate
      // still converges to the true echo whenever learning is active (which
      // only happens while the reference has energy, i.e. refGate ~ 1 anyway).
      mRawResidual[i] = mic - est;
      // The OUTPUT (and what gets recorded) subtracts only as much of the
      // stored echo as the current far-end justifies: full while the speaker
      // plays, ~0 in a silent room so a stale E[phi] can't become an
      // inverted ghost.
      const float outR = mic - refGate * est;
      // Stage-2 nonlinear polish: duck the HF sub-bands whose leftover energy
      // is explained by residual echo (the ghost / metronome click linear
      // cancellation can't reach). The anchor is the reference-GATED echo
      // estimate, so in a silent room (refGate~0) the anchor is ~0 and the
      // suppressor passes the mic through untouched. Zero-latency at unity.
      const float suppressed = mSuppressor.processSample(ch, refGate * est, outR);
      micInOut[i] = suppressed; // Pass 1b scales this in place
      // E3 freeze and the Pass-1b safety clamp reason about LINEAR
      // cancellation correctness, so they see the pre-suppressor residual —
      // the suppressor is a final output polish, not part of that decision.
      blockResid += static_cast<double>(outR) * outR;
      blockMicEnergy += static_cast<double>(mic) * mic;
    }

    // Seed self-scaling fit: accumulate <mic, seed> and <seed, seed> at this
    // frame's phase for one full pass, then derive the trust scale alpha.
    if (mSeedFitActive) {
      float residMono = 0.0f; // current template's residual (mic - E), mono
      float estMono = 0.0f;
      for (unsigned int ch = 0; ch < channels; ++ch) {
        residMono += mRawResidual[f * channels + ch];
        estMono += mTemplate[base + (ch % channels)];
      }
      residMono /= static_cast<float>(channels);
      estMono /= static_cast<float>(channels);
      const float micMono = residMono + estMono; // mic = raw + est
      const float sv = mSeedOutput[static_cast<size_t>(phi)];
      mSeedFitNum += static_cast<double>(micMono) * sv;
      mSeedFitDen += static_cast<double>(sv) * sv;
      mSeedFitMicSq += static_cast<double>(micMono) * micMono;
      mSeedFitCurSq += static_cast<double>(residMono) * residMono;
      if (++mSeedFitFrames >= mActiveLoopFrames) {
        float alpha = (mSeedFitDen > 1e-9)
                          ? static_cast<float>(mSeedFitNum / mSeedFitDen)
                          : 0.0f;
        alpha = std::max(-2.0f, std::min(2.0f, alpha));
        mSeedFitActive = false;
        mSeedFitFrames = 0;
        mSeedLastAlpha.store(alpha, std::memory_order_relaxed);
        // Beats-current gate (see mSeedFitMicSq doc in the header): the
        // seed's predicted residual after applying alpha*seed, in closed
        // form from the fit sums, vs what the CURRENT template already
        // achieves over the same pass. Landing stamp-REPLACES E[phi] at
        // kSeedConfidence, so a seed that doesn't clearly beat the present
        // state is a pure downgrade — measured collapsing a 15 dB converged
        // template to ~2 dB on a real overdub before this gate existed.
        const double predicted =
            mSeedFitMicSq -
            (mSeedFitNum * mSeedFitNum) / std::max(mSeedFitDen, 1e-9);
        const double current = mSeedFitCurSq;
        const bool beatsCurrent = predicted < current * 0.8; // >= ~1 dB better
        aecLog("[SEED] fit alpha=%.3f predResid=%.3g curResid=%.3g%s\n",
               alpha, predicted, current,
               beatsCurrent ? "" : " (does NOT beat current)");
        if (std::fabs(alpha) < 0.05f) {
          // Seed doesn't correlate with the mic (misaligned / no echo):
          // discard rather than inject noise. Keep the prior track scale.
          mSeedOutputReady.store(false, std::memory_order_relaxed);
          mSeedBusy = false;
          mSeedDiscards.fetch_add(1, std::memory_order_relaxed);
          aecLog("[SEED] DISCARDED (alpha=%.3f)\n", alpha);
          // A discard means THIS capture was unusable (silent/contaminated
          // pass), not that seeding is hopeless — retry up to 3 times
          // before waiting for a real reference-change notify.
          if (++mSeedRetryCount <= 3) {
            mReferenceChangePending.store(true, std::memory_order_relaxed);
            aecLog("[SEED] retry %d/3 armed\n", mSeedRetryCount);
          }
        } else if (!beatsCurrent) {
          // The template we already have outperforms this seed. Discard
          // WITHOUT retrying: the capture wasn't unusable — the state is
          // simply already better than what a reseed can offer (a converged
          // EMA template on a coherence-starved mix). Retrying would churn
          // the same losing comparison every pass.
          mSeedOutputReady.store(false, std::memory_order_relaxed);
          mSeedBusy = false;
          mSeedDiscards.fetch_add(1, std::memory_order_relaxed);
          aecLog("[SEED] DISCARDED (worse than current template)\n");
        } else {
          mSeedFitAlpha = alpha;
          mIrFitScale = alpha; // per-track edits share the IR's fitted scale
          mSeedFitDone = true; // gate the apply open (and keep re-fit shut)
          mSeedApplyPos = 0;   // chunked apply starts next callback
        }
      }
    }

    // Convergence-seed capture: while armed, record one period of the ALIGNED
    // reference (mono-mixed) at its exact phase. Uses the same phi as the
    // cancel pass above, so once mSeedCaptureRemaining reaches 0 every phase
    // in [0,P) has been written exactly once, regardless of block alignment.
    if (mSeedBusy && mSeedCaptureRemaining > 0 && alignedRef) {
      float rm = 0.0f;
      for (unsigned int ch = 0; ch < channels; ++ch)
        rm += alignedRef[f * channels + ch];
      mRefCapture[static_cast<size_t>(phi)] = rm / static_cast<float>(channels);
      float micMono = 0.0f;
      for (unsigned int ch = 0; ch < channels; ++ch)
        micMono += mRawResidual[f * channels + ch] + mTemplate[base + ch];
      mMicCapture[static_cast<size_t>(phi)] = micMono / static_cast<float>(channels);
      if (--mSeedCaptureRemaining == 0 &&
          !mSeedJobPosted.load(std::memory_order_relaxed)) {
        mSeedJobPeriod = P;
        mSeedJobGeneration = mSeedGeneration; // stamp: which arm this job belongs to
        mSeedJobPosted.store(true, std::memory_order_release);
        aecLog("[SEED] capture complete, job posted gen=%lld\n",
               (long long)mSeedGeneration);
      }
    }
  }

  // ---- Pass 1b: smoothed safety-suppression gain, applied to the whole block.
  // Two-stage smoothing, deliberately separated:
  //   1. Smooth the RAW ENERGY RATIO first (mSmoothedEnergyRatio) — this is
  //      what averages out single-block statistical noise, letting the
  //      margin below be tight.
  //   2. Smooth the DERIVED GAIN second (mOutputSuppressGain) — this is
  //      what prevents the gain itself from stepping abruptly at the
  //      callback rate (a click), independent of stage 1.
  // A single-stage version (smooth only the gain, decide per-block with a
  // wide margin) was tried first: it missed a real, sustained-but-moderate
  // ghost (mildly negative ERLE, not a large blowout) because the margin
  // had to be wide enough to reject single-block noise, which also hid
  // genuine moderate staleness. Averaging the RATIO first means a much
  // tighter margin can be used without reopening the original per-sample-
  // static problem this whole mechanism exists to avoid.
  {
    const float blockMs =
        mSampleRate > 0
            ? 1000.0f * static_cast<float>(frameCount) / static_cast<float>(mSampleRate)
            : 0.0f;
    const float ratioAlpha =
        blockMs > 0.0f ? 1.0f - std::exp(-blockMs / kEnergyRatioTauMs) : 1.0f;
    const float blockRatio = static_cast<float>(
        (blockMicEnergy + kEps) / (blockResid + kEps)); // >=1 healthy, <1 raw exceeds mic
    mSmoothedEnergyRatio += ratioAlpha * (blockRatio - mSmoothedEnergyRatio);

    const float idealGain =
        (mSmoothedEnergyRatio < 1.0f / kOutputGateMargin)
            ? std::sqrt(std::max(0.0f, mSmoothedEnergyRatio * kOutputGateMargin))
            : 1.0f;
    const float gainAlpha =
        blockMs > 0.0f ? 1.0f - std::exp(-blockMs / kOutputGateTauMs) : 1.0f;
    mOutputSuppressGain += gainAlpha * (idealGain - mOutputSuppressGain);
  }
  // micInOut already holds the reference-gated residual from Pass 1a; the
  // block suppression gain is now just a secondary backstop scaling it (in
  // the common cases it stays ~1 because the reference gate already keeps
  // the residual from exceeding the mic energy).
  for (size_t i = 0; i < totalSamples; ++i)
    micInOut[i] *= mOutputSuppressGain;

  // UI overlay feed: block-smoothed gate opening (see refGateOpen()).
  mRefGateSmooth += 0.2f * (lastRefGate - mRefGateSmooth);

  // Tell the suppressor whether it may adapt its per-band residual-echo
  // coupling (kappa) on the NEXT block: only when the far-end is actually
  // playing AND the near-end is absent, so the performer is never learned as
  // "surviving echo" and over-suppressed. Far-end presence reuses the same
  // smoothed reference envelope the subtraction gate uses; near-end absence
  // reuses the E3 residual-vs-baseline test, but with a TIGHTER ratio than
  // the freeze threshold — kappa drives how hard we duck, so it must only
  // move under confident echo-only conditions, not merely "not a huge spike".
  // When unsure (baseline not established, or any elevated residual) kappa is
  // held at its conservative init, which under- rather than over-suppresses.
  {
    constexpr float kSuppCleanRatio = 4.0f; // << kSpikeRatio (freeze uses 25)
    const bool farEndPresent = mSubGateEnv > sgFloor;
    const bool nearEndAbsent =
        mLearnedBlocks >= kSettleBlocks &&
        blockResid <=
            static_cast<double>(kSuppCleanRatio) *
                (static_cast<double>(mResidBaseline) + kEps);
    mSuppressor.setCouplingUpdateAllowed(farEndPresent && nearEndAbsent);
  }

  if (!learn)
    return; // cancel only

  // Governor learning boost: loaded once per block (relaxed; RT-safe).
  const float learnBoost = mLearnBoost.load(std::memory_order_relaxed);

  // ---- Block-level double-talk freeze (confidence-gated, governor-aware) ----
  // Interlock with the spectral governor: sustained COHERENT leakage (echo the
  // template stopped matching — e.g. speaker thermal drift over a long soak)
  // raises the boost; a boosted governor must also relax the freeze, or frozen
  // blocks ignore the re-heat and the stale template can never re-learn (the
  // slow decay-to-uncancelled death spiral). Real double-talk is INcoherent:
  // boost decays and full freeze protection returns.
  {
    // Schmitt trigger on the interlock threshold: the raw `< 2.0f` compare
    // chattered as the governor's boost seesawed ±0.19/tick across 2.0,
    // toggling freeze protection at up to the 4.76 Hz governor tick (one
    // leg of the 3.5 Hz warble). Hot at >= 2.25, cold again at <= 1.75.
    if (learnBoost >= 2.25f)
      mBoostInterlockHot = true;
    else if (learnBoost <= 1.75f)
      mBoostInterlockHot = false;
    const int64_t phi0 =
        ((blockStartFrame - loopStartFrame) % P + P) % P;
    const float blockConf = mConfidence[static_cast<size_t>(phi0)];
    if (!mBoostInterlockHot && mLearnedBlocks >= kSettleBlocks &&
        blockConf >= kFreezeMinConf &&
        blockResid >
            kSpikeRatio * (static_cast<double>(mResidBaseline) + kEps)) {
      ++mFreezeCount;
      return; // near-end over a converged template -> do not update E
    }
  }

  // ---- Pass 2: learn (annealed alpha, per-sample far-end gated) ----
  for (unsigned int f = 0; f < frameCount; ++f) {
    int64_t phi = (blockStartFrame + static_cast<int64_t>(f) - loopStartFrame) % P;
    if (phi < 0)
      phi += P;
    const size_t pphi = static_cast<size_t>(phi);
    const size_t base = pphi * channels;

    if (alignedRef) {
      float rp = 0.0f;
      for (unsigned int ch = 0; ch < channels; ++ch) {
        const float r = alignedRef[f * channels + ch];
        rp += r * r;
      }
      mRefEnvelope += kRefEnvRate * (rp - mRefEnvelope);
      if (mRefEnvelope <= kFarEndFloorPow)
        continue; // speaker genuinely silent: nothing to learn
    }

    const float c = mConfidence[pphi];
    float alpha =
        (kAlphaMin + (kAlphaMax - kAlphaMin) * std::exp(-c / kConfTau)) *
        learnBoost;
    if (alpha > kAlphaMax)
      alpha = kAlphaMax; // bounded authority: boost can re-heat, never exceed hot
    // Robust (clipped) update: bound each sample's learning contribution to
    // a few x the running residual RMS. Impulsive NEAR-END transients --
    // keystrokes, chair creaks, a cough -- are enormous outliers at one
    // phase for one pass; unclipped they enter E[phi] at alpha and get
    // SUBTRACTED on later passes (audible phase-inverted "typing ghosts",
    // user-reported). Genuine echo drift is small and sign-consistent
    // across passes and passes through the clip untouched. Classic
    // Huber-style robustification of an averaging estimator.
    float clipLim = 3.4e38f; // effectively no clip until a baseline exists
    if (mResidBaseline > 0.0f && frameCount > 0) {
      const float rms = std::sqrt(
          mResidBaseline / (static_cast<float>(frameCount) * channels));
      clipLim = 3.0f * rms;
    }
    // Undo-ring entry BEFORE the write (double-talk rewind, see header).
    {
      LearnUndo &u = mUndoRing[mUndoHead];
      u.phi = static_cast<uint32_t>(pphi);
      u.oldConf = mConfidence[pphi];
      for (unsigned int ch = 0; ch < channels && ch < 2; ++ch)
        u.oldT[ch] = mTemplate[base + ch];
      mUndoHead = (mUndoHead + 1) % mUndoRing.size();
      if (mUndoCount < mUndoRing.size()) ++mUndoCount;
    }
    for (unsigned int ch = 0; ch < channels; ++ch) {
      float upd = mRawResidual[f * channels + ch];
      if (upd > clipLim) upd = clipLim;
      else if (upd < -clipLim) upd = -clipLim;
      mTemplate[base + ch] += alpha * upd;
    }
    if (c < kConfMax) {
      mConfidenceSum += 1.0;
      mConfidence[pphi] = c + 1.0f;
    }
  }

  // Update the smoothed residual floor (clamped so a borderline near-end block
  // can't inflate the baseline the detector compares against).
  const float contrib =
      std::min(static_cast<float>(blockResid),
               kSpikeRatio * (mResidBaseline + static_cast<float>(kEps)));
  mResidBaseline = (1.0f - kBaseRate) * mResidBaseline + kBaseRate * contrib;
  ++mLearnedBlocks;
}
