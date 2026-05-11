// Tempo / quantum / key inference for recorded clips.
//
// Two tiers of inference:
//   1. Length-only — instant; needs just (loopFrames, sampleRate). Uses the
//      constraint that musical loops are usually integer numbers of beats at
//      a sensible tempo, scored by quantum simplicity + BPM musicality.
//      Good enough for the common case; immediately fixes the "q=4 hardcoded"
//      problem.
//   2. Audio-aware — runs on the worker thread after WAV write. Uses onset
//      detection + autocorrelation for tempo, FFT chromagram + Krumhansl-
//      Schmuckler for key. More accurate; arrives a few hundred ms later
//      via the event bus.
//
// Both are pure functions. No I/O, no threading, no allocations on the hot
// path beyond what's noted. Designed to be drop-in callable from anywhere.

#ifndef FLOWSTATE_AUDIO_ENGINE_INFERENCE_H_
#define FLOWSTATE_AUDIO_ENGINE_INFERENCE_H_

#include <cstdint>

namespace flowstate {
namespace audio_engine {

// ---------------------------------------------------------------------------
// Tempo + quantum inference
// ---------------------------------------------------------------------------

struct TempoInference {
  double        bpm;
  std::uint32_t quantum;
  // 0..1. Length-only inference's confidence reflects how well the chosen
  // candidate scored vs. all considered candidates; audio-aware inference
  // additionally folds in onset-pattern alignment.
  float         confidence;
};

// Length-only inference: given a loop's frame count and sample rate, pick the
// most musically plausible (BPM, quantum) pair. Quantum candidates considered:
// {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}. BPMs outside [60, 200] are rejected.
//
// Returns {0, 0, 0} for invalid inputs (zero frames or zero sample rate).
TempoInference inferTempoFromLength(std::int64_t loopFrames,
                                    std::uint32_t sampleRate) noexcept;

// Audio-aware inference: same output type, but uses onset envelope +
// autocorrelation on the actual samples to disambiguate musically equivalent
// length-only candidates. Implementation lands in Phase 3a-v2; declared here
// so the interface is settled.
//
// `samples` is interleaved float32, channelCount channels, frameCount frames.
// Implementation will downmix to mono internally.
TempoInference inferTempoFromAudio(const float* samples,
                                    std::int64_t frameCount,
                                    std::uint32_t channels,
                                    std::uint32_t sampleRate) noexcept;

// ---------------------------------------------------------------------------
// Key inference (Phase 3b)
// ---------------------------------------------------------------------------

struct KeyInference {
  std::uint8_t pitchClass;    // 0..11 (C, C#, D, ..., B); 255 = unknown
  bool         isMinor;
  float        confidence;
};

// Krumhansl-Schmuckler key estimation on the given samples. Phase 3b
// implementation; declared here.
KeyInference inferKey(const float* samples,
                      std::int64_t frameCount,
                      std::uint32_t channels,
                      std::uint32_t sampleRate) noexcept;

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_INFERENCE_H_
