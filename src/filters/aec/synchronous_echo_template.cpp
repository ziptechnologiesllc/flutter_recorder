#include "synchronous_echo_template.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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
    : mSampleRate(sampleRate), mChannels(channels) {
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

void SynchronousEchoTemplate::setSeedImpulseResponse(const float *coeffs,
                                                     int length) {
  if (!coeffs || length <= 0)
    return;
  std::lock_guard<std::mutex> lock(mSeedIRMutex);
  mSeedIR.assign(coeffs, coeffs + length);
}

void SynchronousEchoTemplate::reset() {
  std::fill(mTemplate.begin(), mTemplate.end(), 0.0f);
  std::fill(mConfidence.begin(), mConfidence.end(), 0.0f);
  mActiveLoopFrames = 0;
  mRefEnvelope = 0.0f;
  mResidBaseline = 0.0f;
  mLearnedBlocks = 0;
  mFreezeCount = 0;
  mReopenCount = 0;
  // Abandon any in-progress capture/apply. A job the worker completes for the
  // now-stale period is discarded by the period check in process() — no
  // handshake needed with the worker thread here.
  mSeedBusy = false;
  mSeedCaptureRemaining = 0;
  mSeedApplyPos = 0;
}

void SynchronousEchoTemplate::seedWorkerLoop() {
  while (mSeedThreadRunning.load(std::memory_order_relaxed)) {
    if (!mSeedJobPosted.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kSeedPollMs));
      continue;
    }
    computeSeedConvolution(mSeedJobPeriod);
    mSeedJobPosted.store(false, std::memory_order_release);
    mSeedOutputReady.store(true, std::memory_order_release);
  }
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
  const size_t L = ir.size();
  // Exact circular convolution: the reference is exactly periodic at P, so
  // the true steady-state echo is h ⊛ ref computed modulo P — not a linear-
  // convolution approximation with edge effects. seed[phi] predicts what
  // per-pass learning would converge to, using the SAME alignedRef delay
  // convention the live filter already uses (see calibration.cpp
  // analyzeAligned: h is trained on a delay-pre-aligned ref/mic pair).
  for (size_t phi = 0; phi < Pu; ++phi) {
    double acc = 0.0;
    for (size_t k = 0; k < L; ++k) {
      const size_t idx = (phi + Pu - (k % Pu)) % Pu;
      acc += static_cast<double>(ir[k]) * static_cast<double>(mRefCapture[idx]);
    }
    mSeedOutput[phi] = static_cast<float>(acc);
  }
}

float SynchronousEchoTemplate::meanConfidence() const {
  if (mActiveLoopFrames <= 0)
    return 0.0f;
  const size_t n = static_cast<size_t>(mActiveLoopFrames);
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i)
    acc += mConfidence[i];
  return static_cast<float>(acc / (static_cast<double>(n) * kConfMax));
}

void SynchronousEchoTemplate::process(float *micInOut, const float *alignedRef,
                                      unsigned int frameCount,
                                      unsigned int channels,
                                      int64_t blockStartFrame,
                                      int64_t loopFrames,
                                      int64_t loopStartFrame, bool learn) {
  if (!micInOut || frameCount == 0)
    return;

  // No loop yet, or period beyond capacity -> passthrough.
  if (loopFrames <= 0 ||
      static_cast<size_t>(loopFrames) > mCapacityFrames ||
      channels > mChannels) {
    mActiveLoopFrames = (loopFrames > 0) ? loopFrames : 0;
    return;
  }

  // Loop period changed (a layer added/removed): the old per-phase estimates
  // remap to different content, so clear the in-use span ONCE at the boundary.
  if (loopFrames != mActiveLoopFrames) {
    const size_t span = static_cast<size_t>(loopFrames) * channels;
    const size_t pspan = static_cast<size_t>(loopFrames);
    std::fill(mTemplate.begin(), mTemplate.begin() + span, 0.0f);
    std::fill(mConfidence.begin(), mConfidence.begin() + pspan, 0.0f);
    mActiveLoopFrames = loopFrames;
    mResidBaseline = 0.0f;
    mLearnedBlocks = 0;

    // Arm a convergence-seed capture for the new period, if we have a
    // calibrated IR and aren't already mid-seed for a (now stale) period.
    // try_lock: setSeedImpulseResponse is rare and off the audio thread; on
    // the vanishingly-unlikely contention case we just skip seeding this
    // cycle and fall back to normal per-pass learning (always correct, just
    // not fast).
    if (!mSeedBusy && alignedRef) { // no reference this call -> can't capture, skip
      bool haveSeed = false;
      if (mSeedIRMutex.try_lock()) {
        haveSeed = !mSeedIR.empty();
        mSeedIRMutex.unlock();
      }
      if (haveSeed) {
        mSeedBusy = true;
        mSeedCaptureRemaining = loopFrames;
        mSeedApplyPos = 0;
      }
    }
  }

  // Drain a finished seed job, chunked so any single callback's apply cost is
  // bounded regardless of loop length. mSeedJobPeriod must still match the
  // ACTIVE period — a job that finishes after a further period change (the
  // capture was abandoned by the branch above) is stale and discarded here.
  if (mSeedOutputReady.load(std::memory_order_acquire)) {
    if (mSeedJobPeriod == mActiveLoopFrames) {
      const int64_t end = std::min(mSeedApplyPos + kSeedApplyChunk, mActiveLoopFrames);
      for (int64_t phi = mSeedApplyPos; phi < end; ++phi) {
        const float v = mSeedOutput[static_cast<size_t>(phi)];
        const size_t base = static_cast<size_t>(phi) * channels;
        for (unsigned int ch = 0; ch < channels; ++ch)
          mTemplate[base + ch] = v;
        mConfidence[static_cast<size_t>(phi)] = kSeedConfidence;
      }
      mSeedApplyPos = end;
      if (mSeedApplyPos >= mActiveLoopFrames) {
        mSeedOutputReady.store(false, std::memory_order_relaxed);
        mSeedBusy = false;
      }
    } else {
      // Stale job for an old period — drop it, no apply.
      mSeedOutputReady.store(false, std::memory_order_relaxed);
      mSeedBusy = false;
    }
  }

  const int64_t P = loopFrames;

  // ---- Pass 1: cancel in place (out = mic - E[phi]); accumulate block residual.
  double blockResid = 0.0;
  for (unsigned int f = 0; f < frameCount; ++f) {
    int64_t phi = (blockStartFrame + static_cast<int64_t>(f) - loopStartFrame) % P;
    if (phi < 0)
      phi += P; // C++ % can be negative; loop phase must be in [0, P)
    const size_t base = static_cast<size_t>(phi) * channels;
    for (unsigned int ch = 0; ch < channels; ++ch) {
      const size_t i = f * channels + ch;
      const float u = micInOut[i] - mTemplate[base + ch];
      micInOut[i] = u; // micInOut now holds the residual (the cancelled output)
      blockResid += static_cast<double>(u) * u;
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
        mSeedJobPosted.store(true, std::memory_order_release);
      }
    }
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
      mTemplate[base + ch] += alpha * micInOut[f * channels + ch];
    if (c < kConfMax)
      mConfidence[pphi] = c + 1.0f;
  }

  // Update the smoothed residual floor (clamped so a borderline near-end block
  // can't inflate the baseline the detector compares against).
  const float contrib =
      std::min(static_cast<float>(blockResid),
               kSpikeRatio * (mResidBaseline + static_cast<float>(kEps)));
  mResidBaseline = (1.0f - kBaseRate) * mResidBaseline + kBaseRate * contrib;
  ++mLearnedBlocks;
}
