#include "synchronous_echo_template.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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

// Maximum supported loop period: 16 s. Sized once at construction so process()
// never allocates on the audio thread.
constexpr unsigned int kMaxSeconds = 16;

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
} // namespace

SynchronousEchoTemplate::SynchronousEchoTemplate(unsigned int sampleRate,
                                                 unsigned int channels)
    : mSampleRate(sampleRate), mChannels(channels),
      mSuppressor(sampleRate, channels) {
  mCapacityFrames = static_cast<size_t>(sampleRate) * kMaxSeconds;
  mTemplate.assign(mCapacityFrames * channels, 0.0f);
  mConfidence.assign(mCapacityFrames, 0.0f);
  mRefCapture.assign(mCapacityFrames, 0.0f);
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
  if (mSeedBusy)
    mSeedAborts.fetch_add(1, std::memory_order_relaxed);
  mSeedBusy = false;
  mSeedCaptureRemaining = 0;
  mSeedApplyPos = 0;
}

void SynchronousEchoTemplate::seedWorkerLoop() {
  while (mSeedThreadRunning.load(std::memory_order_relaxed)) {
    bool didWork = false;

    if (mSeedJobPosted.load(std::memory_order_acquire)) {
      computeSeedConvolution(mSeedJobPeriod);
      mSeedJobPosted.store(false, std::memory_order_release);
      mSeedOutputReady.store(true, std::memory_order_release);
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
  if (mSeedBusy || !alignedRef)
    return; // already capturing/awaiting a job, or nothing to capture from
  bool haveSeed = false;
  if (mSeedIRMutex.try_lock()) {
    haveSeed = !mSeedIR.empty();
    mSeedIRMutex.unlock();
  }
  if (!haveSeed)
    return;
  ++mSeedGeneration; // this arm owns a fresh epoch; no prior job can match it
  mSeedBusy = true;
  mSeedCaptureRemaining = mActiveLoopFrames;
  mSeedApplyPos = 0;
  mSeedArms.fetch_add(1, std::memory_order_relaxed);
}

void SynchronousEchoTemplate::computeSeedConvolution(int64_t P) {
  std::vector<float> ir;
  {
    std::lock_guard<std::mutex> lock(mSeedIRMutex);
    ir = mSeedIR; // snapshot — a mid-job calibration update won't tear this read
  }
  if (ir.empty() || P <= 0)
    return;
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
  // uses (see calibration.cpp analyzeAligned: h is trained on a delay-
  // pre-aligned ref/mic pair).
  float seedPeak = 0.0f;
  for (size_t phi = 0; phi < Pu; ++phi) {
    double acc = 0.0;
    for (size_t k = 0; k < L; ++k) {
      const size_t idx = (phi + Pu - k) % Pu; // k < L <= Pu: no wraparound aliasing
      acc += static_cast<double>(ir[k]) * static_cast<double>(mRefCapture[idx]);
    }
    const float v = static_cast<float>(acc);
    mSeedOutput[phi] = v;
    seedPeak = std::max(seedPeak, std::fabs(v));
  }

  if (seedPeak > kMaxSeedAbs) {
    const float scale = kMaxSeedAbs / seedPeak;
    for (size_t phi = 0; phi < Pu; ++phi)
      mSeedOutput[phi] *= scale;
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
  tc.E.clear();
  tc.slotIndex.store(-1, std::memory_order_release);
}

void SynchronousEchoTemplate::registerTrackAudio(int trackIndex,
                                                 const float *audioMono,
                                                 int64_t frames) {
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

  std::vector<float> folded(P, 0.0f);
  for (size_t i = 0; i < rawLen; ++i)
    folded[i % P] += audio[i];

  // Same truncation rationale as computeSeedConvolution: a period-P
  // circular convolution is only well-posed with a kernel <= P taps.
  const size_t L = std::min(ir.size(), P);

  float audioPeak = 0.0f;
  for (size_t phi = 0; phi < P; ++phi)
    audioPeak = std::max(audioPeak, std::fabs(folded[phi]));

  std::vector<float> mono(P, 0.0f);
  float contribPeak = 0.0f;
  for (size_t phi = 0; phi < P; ++phi) {
    double acc = 0.0;
    for (size_t k = 0; k < L; ++k) {
      const size_t idx = (phi + P - k) % P;
      acc += static_cast<double>(ir[k]) * static_cast<double>(folded[idx]);
    }
    const float v = static_cast<float>(acc);
    mono[phi] = v;
    contribPeak = std::max(contribPeak, std::fabs(v));
  }
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

  // Expand mono -> interleaved, matching mTemplate's layout (E[phi*channels+ch]).
  tc.E.assign(P * mChannels, 0.0f);
  for (size_t phi = 0; phi < P; ++phi) {
    const size_t base = phi * mChannels;
    for (unsigned int ch = 0; ch < mChannels; ++ch)
      tc.E[base + ch] = mono[phi];
  }
  tc.computed.store(true, std::memory_order_release);
  aecLog("[LSAEC] track %d contribution computed: raw=%zu folded-to-P=%zu "
         "peak=%.4f (audio peak=%.4f)\n",
         trackIndex, rawLen, P, contribPeak, audioPeak);
}

void SynchronousEchoTemplate::setTrackActive(int trackIndex, bool active) {
  if (trackIndex < 0)
    return;
  // Lock-free lookup only — audio thread, must never allocate or block.
  // registerTrackAudio() is what allocates a slot; a track with no slot
  // simply hasn't been registered yet (or registration failed, e.g. table
  // full) and falls back to the composite template, same as before.
  const int slot = findTrackSlot(trackIndex);
  if (slot < 0)
    return;
  TrackContribution &tc = mTrackContributions[slot];
  if (!tc.computed.load(std::memory_order_acquire))
    return; // not ready yet — composite template handles this track for now
  if (tc.active == active)
    return; // already in the target state
  // Defensive: the contribution's length must still match the CURRENT
  // composite template's active span, or this track's audio was
  // registered against a period that's since changed (a period change, or
  // a stale job — see registerTrackAudio's comment). Applying a
  // mismatched-length contribution would misalign phase and silently
  // corrupt the template rather than simply not helping — skip instead.
  const size_t expected =
      static_cast<size_t>(mActiveLoopFrames) * mChannels;
  if (tc.E.size() != expected) {
    aecLog("[LSAEC] track %d toggle SKIPPED (size mismatch: have %zu, "
           "need %zu — period changed since registration)\n",
           trackIndex, tc.E.size(), expected);
    return;
  }
  const float sign = active ? 1.0f : -1.0f;
  for (size_t i = 0; i < tc.E.size(); ++i)
    mTemplate[i] += sign * tc.E[i];
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
  aecLog("[LSAEC] track %d exact-subtraction toggle: active=%d\n",
         trackIndex, active ? 1 : 0);
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
  return static_cast<float>(mConfidenceSum / (static_cast<double>(n) * kConfMax));
}

void SynchronousEchoTemplate::process(float *micInOut, const float *alignedRef,
                                      unsigned int frameCount,
                                      unsigned int channels,
                                      int64_t blockStartFrame,
                                      int64_t loopFrames,
                                      int64_t loopStartFrame, bool learn) {
  if (!micInOut || frameCount == 0)
    return;

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

  // Loop period changed (a layer added/removed): the old per-phase estimates
  // remap to different content, so clear the in-use span ONCE at the boundary.
  if (loopFrames != mActiveLoopFrames) {
    const size_t span = static_cast<size_t>(loopFrames) * channels;
    const size_t pspan = static_cast<size_t>(loopFrames);
    std::fill(mTemplate.begin(), mTemplate.begin() + span, 0.0f);
    std::fill(mConfidence.begin(), mConfidence.begin() + pspan, 0.0f);
    // The clear above zeroes exactly the NEW active window [0, pspan), so
    // the running sum is exactly 0 immediately after — matches the
    // mConfidenceSum invariant (sum over the CURRENT active window).
    mConfidenceSum = 0.0;
    mActiveLoopFrames = loopFrames;
    mActiveLoopFramesAtomic.store(loopFrames, std::memory_order_relaxed);
    mResidBaseline = 0.0f;
    mLearnedBlocks = 0;

    // A period change supersedes any pending mix-changed notification (the
    // capture below already covers the new state) and abandons any old-period
    // capture/job in flight: bump the generation so its late completion is
    // discarded at drain time (below) without touching mSeedBusy, which the
    // fresh arm immediately below is about to own for the NEW period.
    mReferenceChangePending.store(false, std::memory_order_relaxed);
    if (mSeedBusy) {
      ++mSeedGeneration;
      mSeedBusy = false;
      mSeedAborts.fetch_add(1, std::memory_order_relaxed);
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
      mSeedAborts.fetch_add(1, std::memory_order_relaxed);
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
      const int64_t end = std::min(mSeedApplyPos + kSeedApplyChunk, mActiveLoopFrames);
      for (int64_t phi = mSeedApplyPos; phi < end; ++phi) {
        const float v = mSeedOutput[static_cast<size_t>(phi)];
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
        mSeedLands.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      mSeedOutputReady.store(false, std::memory_order_relaxed); // stale; drop only
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

    // Convergence-seed capture: while armed, record one period of the ALIGNED
    // reference (mono-mixed) at its exact phase. Uses the same phi as the
    // cancel pass above, so once mSeedCaptureRemaining reaches 0 every phase
    // in [0,P) has been written exactly once, regardless of block alignment.
    if (mSeedBusy && mSeedCaptureRemaining > 0 && alignedRef) {
      float rm = 0.0f;
      for (unsigned int ch = 0; ch < channels; ++ch)
        rm += alignedRef[f * channels + ch];
      mRefCapture[static_cast<size_t>(phi)] = rm / static_cast<float>(channels);
      if (--mSeedCaptureRemaining == 0 &&
          !mSeedJobPosted.load(std::memory_order_relaxed)) {
        mSeedJobPeriod = P;
        mSeedJobGeneration = mSeedGeneration; // stamp: which arm this job belongs to
        mSeedJobPosted.store(true, std::memory_order_release);
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
    const int64_t phi0 =
        ((blockStartFrame - loopStartFrame) % P + P) % P;
    const float blockConf = mConfidence[static_cast<size_t>(phi0)];
    if (learnBoost < 2.0f && mLearnedBlocks >= kSettleBlocks &&
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
    for (unsigned int ch = 0; ch < channels; ++ch)
      mTemplate[base + ch] += alpha * mRawResidual[f * channels + ch];
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
