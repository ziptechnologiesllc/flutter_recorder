#include "inference.h"

#include <cmath>

namespace flowstate {
namespace audio_engine {

// ---------------------------------------------------------------------------
// Length-only tempo inference
// ---------------------------------------------------------------------------
//
// The intuition: a recorded loop of duration D seconds at tempo T BPM with Q
// beats per loop satisfies T·D / 60 = Q exactly. So given D, the candidates
// are pairs (T, Q) where Q is an integer divisor of T·D/60. We restrict to
// musically plausible Q (1..32 with strong bias toward 3, 4) and T (60..200,
// gaussian preference around 120).
//
// This is not as accurate as full audio analysis, but it gets the common case
// right and is instantly available the moment a recording stops.

namespace {

struct QuantumCandidate {
  std::uint32_t quantum;
  double bias;  // 0..1; multiplicative musicality prior
};

// Ordered roughly by musical commonness. The bias values are empirical —
// tuned so that "obvious" cases (4-beat loops at 80-140 BPM) win cleanly,
// while still allowing 3/4 and compound meters to surface when the BPM math
// puts them in a more natural range.
constexpr QuantumCandidate kCandidates[] = {
    {4, 1.00},   // 4/4 single bar — by far the most common
    {3, 0.85},   // 3/4 waltz, single bar
    {8, 0.80},   // 2 bars of 4/4 or 1 bar of compound 8
    {6, 0.75},   // 6/8 compound duple, or 2 bars of 3/4
    {2, 0.60},   // half-bar of 4/4 or 1 bar of 2/4
    {12, 0.55},  // 3 bars of 4/4 (less common)
    {16, 0.55},  // 4 bars of 4/4
    {1, 0.40},   // single beat (rare for a "loop")
    {24, 0.35},  // 6 bars
    {32, 0.30},  // 8 bars
};

constexpr double kMinPlausibleBpm = 60.0;
constexpr double kMaxPlausibleBpm = 200.0;
constexpr double kBpmCenter = 120.0;
constexpr double kBpmSigma = 70.0;  // wider gaussian than first instinct so
                                    // 80 BPM and 140 BPM both score well.

}  // namespace

TempoInference inferTempoFromLength(std::int64_t loopFrames,
                                     std::uint32_t sampleRate) noexcept {
  if (loopFrames <= 0 || sampleRate == 0) {
    return {0.0, 0, 0.0f};
  }
  const double duration = static_cast<double>(loopFrames) /
                          static_cast<double>(sampleRate);

  TempoInference best{120.0, 4, 0.0f};
  double bestScore = -1.0;

  for (const QuantumCandidate& c : kCandidates) {
    const double bpm = static_cast<double>(c.quantum) * 60.0 / duration;
    if (bpm < kMinPlausibleBpm || bpm > kMaxPlausibleBpm) continue;

    // Gaussian preference centered at kBpmCenter. Pulls toward common
    // recording tempos without rejecting extremes outright.
    const double delta = (bpm - kBpmCenter) / kBpmSigma;
    const double bpmScore = std::exp(-delta * delta);
    const double score = bpmScore * c.bias;

    if (score > bestScore) {
      bestScore = score;
      best = TempoInference{bpm, c.quantum, static_cast<float>(score)};
    }
  }

  return best;
}

// ---------------------------------------------------------------------------
// Audio-aware tempo + key (Phase 3a-v2 / Phase 3b stubs)
// ---------------------------------------------------------------------------

TempoInference inferTempoFromAudio(const float* /*samples*/,
                                    std::int64_t frameCount,
                                    std::uint32_t /*channels*/,
                                    std::uint32_t sampleRate) noexcept {
  // Phase 3a v2 will implement onset-envelope + autocorrelation. Until then,
  // fall back to length-only inference so callers can wire to this function
  // and "upgrade" later without changing the integration site.
  return inferTempoFromLength(frameCount, sampleRate);
}

KeyInference inferKey(const float* /*samples*/, std::int64_t /*frameCount*/,
                       std::uint32_t /*channels*/,
                       std::uint32_t /*sampleRate*/) noexcept {
  // Phase 3b. Placeholder returns "unknown".
  return KeyInference{255, false, 0.0f};
}

}  // namespace audio_engine
}  // namespace flowstate
