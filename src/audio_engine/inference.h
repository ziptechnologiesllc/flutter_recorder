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
#include <vector>

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

// Krumhansl-Schmuckler key estimation on the given samples.
KeyInference inferKey(const float* samples,
                      std::int64_t frameCount,
                      std::uint32_t channels,
                      std::uint32_t sampleRate) noexcept;

// ---------------------------------------------------------------------------
// Chord / note annotation (Phase 3c)
// ---------------------------------------------------------------------------

// A labelled span of a loop. "Chord" is loose here: a span may be a triad
// (major or minor) or a single sustained pitch class — monophonic content
// like a sung melody note or a one-note-at-a-time line. The producer runs a
// chromagram → harmonic-template match → Viterbi pipeline, so labels are
// *stable*: a span only changes when the audio evidence persistently
// disagrees with the current label, not on every flickery sixteenth.
enum class ChordQuality : std::uint8_t {
  Major = 0,   // major triad rooted at pitchClass
  Minor = 1,   // minor triad rooted at pitchClass
  Note  = 2,   // single pitch class (monophonic content)
};

struct ChordSegment {
  std::int32_t  startSixteenth;  // inclusive
  std::int32_t  endSixteenth;    // exclusive
  std::uint8_t  pitchClass;      // 0..11; 255 = N/C (no clear pitch)
  std::uint8_t  quality;         // ChordQuality; ignored when pitchClass == 255
  float         confidence;      // 0..1
};

// Recognize the chord / note progression of a loop.
//
// Pipeline:
//   1. FFT chromagram (same path as inferKey), aggregated into half-beat
//      (eighth-note) analysis windows — long enough that a strummed or
//      arpeggiated chord settles into a stable chroma, short enough to catch
//      a quick IV→V at the end of a bar.
//   2. Each window is scored (Pearson correlation) against 36 harmonic-
//      weighted templates: 12 major triads, 12 minor triads, 12 single notes.
//      A window matching nothing well is labelled N/C.
//   3. A Viterbi decode over the window sequence with a sticky self-transition
//      (the continuity prior — chords persist) and a key-diatonic discount
//      (changing *to* a chord/note outside the inferred key costs extra).
//      This replaces the old per-window argmax + median filter and is what
//      makes the output stable on real instruments.
//   4. Merge consecutive identical labels into segments on the sixteenth grid.
//
// `keyPitchClass` / `keyIsMinor` come from inferKey; pass 255 for unknown to
// disable the key bias. Returns segments in order; may be empty if the loop
// is too short or has no detectable tonal content.
std::vector<ChordSegment> recognizeChords(
    const float* samples,
    std::int64_t frameCount,
    std::uint32_t channels,
    std::uint32_t sampleRate,
    double bpm,
    std::uint32_t quantum,
    std::uint8_t keyPitchClass = 255,
    bool keyIsMinor = false) noexcept;

// ---------------------------------------------------------------------------
// Live chord / note estimate (Phase 3c — tuner-style meter)
// ---------------------------------------------------------------------------

struct ChordEstimate {
  std::uint8_t pitchClass;   // 0..11; 255 = N/C (no clear pitch)
  std::uint8_t quality;      // ChordQuality; ignored when pitchClass == 255
  float        confidence;   // 0..1
};

// Single best chord/note for a short window of audio (e.g. the last ~1 s of
// the monitor mix), using the same harmonic-weighted templates as
// recognizeChords but with no temporal model — the caller is expected to
// debounce (require N agreeing estimates before committing a display change).
ChordEstimate estimateChord(const float* samples,
                            std::int64_t frameCount,
                            std::uint32_t channels,
                            std::uint32_t sampleRate) noexcept;

// Lower-level: best chord/note for an already-computed, L1-normalized 12-bin
// chromagram. Exposed so a live meter running on the audio thread can reuse
// its own FFT/chroma plumbing instead of re-deriving one.
ChordEstimate estimateChordFromChroma(const float chroma12[12]) noexcept;

// ---------------------------------------------------------------------------
// Monophonic pitch detection (Phase 3c — chromatic instrument tuner)
// ---------------------------------------------------------------------------

struct PitchEstimate {
  float frequencyHz;   // 0 = no clear pitch (silence, noise, or polyphony)
  float clarity;       // 0..1; YIN aperiodicity → confidence the pitch is real
};

// YIN fundamental-frequency estimate (de Cheveigné & Kawahara 2002 — the
// algorithm behind most instrument tuners) on a short window. Downmixes to
// mono internally; uses the most recent frames if `frameCount` exceeds the
// internal window cap. Searches the instrument range and returns {0, 0} when
// there's no clear single pitch — which is the right answer for a strummed
// chord, so the tuner only "locks" when you're sounding one string.
PitchEstimate detectPitch(const float* samples,
                          std::int64_t frameCount,
                          std::uint32_t channels,
                          std::uint32_t sampleRate) noexcept;

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_INFERENCE_H_
