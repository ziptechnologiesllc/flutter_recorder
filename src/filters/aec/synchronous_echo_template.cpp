#include "synchronous_echo_template.h"

#include <algorithm>
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
} // namespace

SynchronousEchoTemplate::SynchronousEchoTemplate(unsigned int sampleRate,
                                                 unsigned int channels)
    : mSampleRate(sampleRate), mChannels(channels) {
  mCapacityFrames = static_cast<size_t>(sampleRate) * kMaxSeconds;
  mTemplate.assign(mCapacityFrames * channels, 0.0f);
  mConfidence.assign(mCapacityFrames, 0.0f);
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
  }

  if (!learn)
    return; // cancel only

  // ---- Block-level double-talk freeze (confidence-gated) ----
  {
    const int64_t phi0 =
        ((blockStartFrame - loopStartFrame) % P + P) % P;
    const float blockConf = mConfidence[static_cast<size_t>(phi0)];
    if (mLearnedBlocks >= kSettleBlocks && blockConf >= kFreezeMinConf &&
        blockResid >
            kSpikeRatio * (static_cast<double>(mResidBaseline) + kEps)) {
      ++mFreezeCount;
      return; // near-end over a converged template -> do not update E
    }
  }

  // Governor learning boost: loaded once per block (relaxed; RT-safe).
  const float learnBoost = mLearnBoost.load(std::memory_order_relaxed);

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
