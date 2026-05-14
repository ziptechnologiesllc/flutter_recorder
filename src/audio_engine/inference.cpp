#include "inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "../fft/soloud_fft.h"

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

// ---------------------------------------------------------------------------
// Audio-aware tempo inference
// ---------------------------------------------------------------------------
//
// Pipeline:
//   1. Downmix to mono, hop in ~512-sample windows
//   2. Per-hop energy envelope (RMS)
//   3. Novelty = positive first difference of envelope (emphasizes attacks)
//   4. Autocorrelation of novelty over plausible lag range (60–200 BPM)
//   5. Pick the strongest peak; refine with parabolic interpolation
//   6. Normalize BPM into [60, 200] by halving / doubling
//   7. Derive quantum from the loop's total length and that BPM
//   8. Snap BPM exactly so quantum × 60 / bpm == loopDurationSec
//   9. Score confidence from peak-to-mean autocorrelation contrast
//
// On low confidence (sparse / non-rhythmic loops), fall back to the length-
// only inference so the caller always gets *something* reasonable.

namespace {

constexpr int kHopFrames = 512;
constexpr int kFrameFrames = 1024;

// Confidence below this → fall back to length-only inference.
constexpr float kMinConfidenceForAudio = 0.20f;

}  // namespace

TempoInference inferTempoFromAudio(const float* samples,
                                    std::int64_t frameCount,
                                    std::uint32_t channels,
                                    std::uint32_t sampleRate) noexcept {
  // Length-only fallback for invalid or too-short inputs.
  if (samples == nullptr || frameCount <= 0 || channels == 0 ||
      sampleRate == 0) {
    return inferTempoFromLength(frameCount, sampleRate);
  }
  const std::int64_t numHops =
      (frameCount - kFrameFrames) / kHopFrames + 1;
  if (numHops < 8) {
    // Less than ~0.1s of envelope. Autocorrelation results would be noise;
    // length-based reasoning is more reliable.
    return inferTempoFromLength(frameCount, sampleRate);
  }

  // ---- 1+2: Energy envelope (RMS per hop, downmixed to mono) ----------
  std::vector<float> envelope(static_cast<std::size_t>(numHops), 0.0f);
  for (std::int64_t h = 0; h < numHops; ++h) {
    const std::int64_t start = h * kHopFrames;
    double sumSq = 0.0;
    int validSamples = 0;
    for (int i = 0; i < kFrameFrames; ++i) {
      const std::int64_t s = start + i;
      if (s >= frameCount) break;
      // Downmix this frame to mono by averaging channels.
      double mono = 0.0;
      for (std::uint32_t c = 0; c < channels; ++c) {
        mono += samples[s * channels + c];
      }
      mono /= channels;
      sumSq += mono * mono;
      ++validSamples;
    }
    envelope[static_cast<std::size_t>(h)] = (validSamples > 0)
        ? static_cast<float>(std::sqrt(sumSq / validSamples))
        : 0.0f;
  }

  // ---- 3: Novelty function (positive first difference) ----------------
  std::vector<float> novelty(static_cast<std::size_t>(numHops), 0.0f);
  for (std::int64_t h = 1; h < numHops; ++h) {
    const float diff = envelope[static_cast<std::size_t>(h)] -
                       envelope[static_cast<std::size_t>(h - 1)];
    novelty[static_cast<std::size_t>(h)] = diff > 0.0f ? diff : 0.0f;
  }

  // ---- 4: Autocorrelation over plausible lag range --------------------
  // minPeriodHops corresponds to fastest BPM, maxPeriodHops to slowest.
  const double secPerHop =
      static_cast<double>(kHopFrames) / static_cast<double>(sampleRate);
  const int minLag =
      std::max(1, static_cast<int>(std::ceil(60.0 / (200.0 * secPerHop))));
  const int maxLag = std::min(
      static_cast<int>(numHops - 1),
      static_cast<int>(std::floor(60.0 / (60.0 * secPerHop))));

  if (maxLag <= minLag) {
    return inferTempoFromLength(frameCount, sampleRate);
  }

  std::vector<float> autocorr(static_cast<std::size_t>(maxLag + 1), 0.0f);
  for (int lag = minLag; lag <= maxLag; ++lag) {
    double sum = 0.0;
    for (std::int64_t h = 0; h + lag < numHops; ++h) {
      sum += static_cast<double>(novelty[static_cast<std::size_t>(h)]) *
             static_cast<double>(
                 novelty[static_cast<std::size_t>(h + lag)]);
    }
    // Normalize by overlap length so longer-loop, shorter-lag bins don't
    // dominate just by summing more terms.
    autocorr[static_cast<std::size_t>(lag)] =
        static_cast<float>(sum / static_cast<double>(numHops - lag));
  }

  // ---- 5: Find strongest peak ----------------------------------------
  int bestLag = minLag;
  float bestVal = autocorr[static_cast<std::size_t>(minLag)];
  for (int lag = minLag + 1; lag <= maxLag; ++lag) {
    const float v = autocorr[static_cast<std::size_t>(lag)];
    if (v > bestVal) {
      bestVal = v;
      bestLag = lag;
    }
  }

  // ---- 5b: Parabolic interpolation for sub-hop precision -------------
  double refinedLag = static_cast<double>(bestLag);
  if (bestLag > minLag && bestLag < maxLag) {
    const float y_m = autocorr[static_cast<std::size_t>(bestLag - 1)];
    const float y_0 = autocorr[static_cast<std::size_t>(bestLag)];
    const float y_p = autocorr[static_cast<std::size_t>(bestLag + 1)];
    const float denom = (y_m - 2.0f * y_0 + y_p);
    if (std::fabs(denom) > 1e-9f) {
      const double offset =
          0.5 * static_cast<double>(y_m - y_p) / static_cast<double>(denom);
      // Clamp to ±1 hop so a degenerate peak doesn't push us off the grid.
      refinedLag += std::max(-1.0, std::min(1.0, offset));
    }
  }

  // ---- 6: Lag → period → BPM, normalize to [60, 200] -----------------
  const double periodSec = refinedLag * secPerHop;
  if (periodSec <= 0.0) {
    return inferTempoFromLength(frameCount, sampleRate);
  }
  double bpm = 60.0 / periodSec;
  // Octave-jump correction. Loop bound at 8 to avoid pathological inputs
  // spinning forever.
  int octaveJumps = 0;
  while (bpm > 200.0 && octaveJumps < 8) { bpm /= 2.0; ++octaveJumps; }
  while (bpm < 60.0  && octaveJumps < 8) { bpm *= 2.0; ++octaveJumps; }

  // ---- 7+8: Snap to integer quantum given loop length ----------------
  const double loopDurSec =
      static_cast<double>(frameCount) / static_cast<double>(sampleRate);
  std::int32_t quantum =
      static_cast<std::int32_t>(std::lround(bpm * loopDurSec / 60.0));
  if (quantum < 1) quantum = 1;
  if (quantum > 64) quantum = 64;
  bpm = (static_cast<double>(quantum) * 60.0) / loopDurSec;

  // ---- 9: Confidence from peak-to-mean autocorrelation ratio ---------
  double mean = 0.0;
  for (int lag = minLag; lag <= maxLag; ++lag) {
    mean += static_cast<double>(autocorr[static_cast<std::size_t>(lag)]);
  }
  mean /= static_cast<double>(maxLag - minLag + 1);
  float confidence = 0.0f;
  if (mean > 1e-12) {
    // Peak/mean ratio of 1 = featureless; 5+ = clear periodicity. Map to
    // [0, 1] with a soft ceiling.
    const double ratio = static_cast<double>(bestVal) / mean;
    confidence = static_cast<float>(
        std::min(1.0, std::max(0.0, (ratio - 1.0) / 4.0)));
  }

  if (confidence < kMinConfidenceForAudio) {
    // No clear rhythmic period detected. Length-only is more useful.
    return inferTempoFromLength(frameCount, sampleRate);
  }

  return TempoInference{bpm, static_cast<std::uint32_t>(quantum), confidence};
}

// ---------------------------------------------------------------------------
// Key inference (Krumhansl-Schmuckler on a chromagram)
// ---------------------------------------------------------------------------
//
// Pipeline:
//   1. Window the audio into overlapping ~85 ms frames (4096 samples @ 48 kHz)
//   2. Apply Hann window, run radix-2 Cooley-Tukey FFT (in-place)
//   3. For each FFT bin in the musical range (~55 Hz – 4 kHz), map to the
//      nearest pitch class via midi = 69 + 12·log2(f/440), pc = midi % 12
//   4. Accumulate magnitudes into a 12-dim chromagram, then normalize
//   5. Correlate (Pearson) against 24 rotated Krumhansl-Schmuckler key
//      profiles (12 major + 12 minor)
//   6. Pick the highest correlation; that's the key
//   7. Confidence = the max correlation value (typically 0.5–0.9 for clear
//      tonal content, lower for noise/silence)

namespace {

// FFT is provided by the existing `FFT::fft(float*, unsigned)` from
// fft/soloud_fft.h. Output is a `float` buffer where bin `b` is encoded as
// (real = buf[b*2], imag = buf[b*2+1]) for b in [0, N/2). The fft processes
// data in place, so the input buffer is overwritten.

// ----- Krumhansl-Schmuckler key strength profiles -----------------------
//
// 12 values each, indexed by pitch class (0=C, 1=C#, ..., 11=B), normalized
// such that values are relative key strengths. Source: Krumhansl & Kessler
// (1982). Empirically derived from probe-tone experiments.

constexpr float kProfileMajor[12] = {
    6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f,
    2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f,
};
constexpr float kProfileMinor[12] = {
    6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f,
    2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f,
};

constexpr std::size_t kFftSize = 4096;
constexpr std::size_t kFftHop = 2048;
constexpr double      kMinFreq = 55.0;     // ~A1, below this is rumble
constexpr double      kMaxFreq = 4186.0;   // C8, top of piano range

// Per-frame chromagram: 12-element vector per FFT hop. Used by both
// inferKey (averages across the whole loop) and recognizeChords (aggregates
// per sixteenth-note window).
struct PerFrameChromagram {
  // [frame_index * 12 + pc] -> magnitude contribution
  std::vector<float> data;
  std::int64_t numFrames{0};
};

// Compute a per-FFT-frame chromagram from interleaved samples. Returns an
// empty result for invalid inputs.
PerFrameChromagram computePerFrameChromagram(
    const float* samples, std::int64_t frameCount,
    std::uint32_t channels, std::uint32_t sampleRate) noexcept {
  PerFrameChromagram out;
  if (samples == nullptr || frameCount <= 0 || channels == 0 ||
      sampleRate == 0) {
    return out;
  }
  if (static_cast<std::size_t>(frameCount) < kFftSize) return out;

  // Pre-compute Hann window.
  std::vector<float> hann(kFftSize);
  for (std::size_t i = 0; i < kFftSize; ++i) {
    hann[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f *
                                       static_cast<float>(i) /
                                       static_cast<float>(kFftSize - 1)));
  }

  // Bin → pitch class mapping (skip bins outside musical range).
  std::vector<int> binToPC(kFftSize / 2, -1);
  for (std::size_t bin = 1; bin < kFftSize / 2; ++bin) {
    const double freq = static_cast<double>(bin) *
                        static_cast<double>(sampleRate) /
                        static_cast<double>(kFftSize);
    if (freq < kMinFreq || freq > kMaxFreq) continue;
    const double midi = 69.0 + 12.0 * std::log2(freq / 440.0);
    int pc = static_cast<int>(std::lround(midi)) % 12;
    if (pc < 0) pc += 12;
    binToPC[bin] = pc;
  }

  const std::int64_t numFrames =
      (frameCount - static_cast<std::int64_t>(kFftSize)) /
          static_cast<std::int64_t>(kFftHop) + 1;
  if (numFrames < 1) return out;

  out.numFrames = numFrames;
  out.data.assign(static_cast<std::size_t>(numFrames) * 12, 0.0f);

  std::vector<float> buf(kFftSize);
  for (std::int64_t f = 0; f < numFrames; ++f) {
    const std::int64_t start = f * static_cast<std::int64_t>(kFftHop);
    for (std::size_t i = 0; i < kFftSize; ++i) {
      const std::int64_t s = start + static_cast<std::int64_t>(i);
      double mono = 0.0;
      for (std::uint32_t c = 0; c < channels; ++c) {
        mono += samples[s * channels + c];
      }
      mono /= channels;
      buf[i] = static_cast<float>(mono) * hann[i];
    }
    FFT::fft(buf.data(), static_cast<unsigned int>(kFftSize));
    for (std::size_t bin = 1; bin < kFftSize / 2; ++bin) {
      const int pc = binToPC[bin];
      if (pc < 0) continue;
      const float re = buf[bin * 2];
      const float im = buf[bin * 2 + 1];
      const float mag = std::sqrt(re * re + im * im);
      out.data[static_cast<std::size_t>(f) * 12 +
               static_cast<std::size_t>(pc)] += mag;
    }
  }
  return out;
}

// Pearson correlation between a 12-element chromagram and a 12-element
// rotated key profile. Returns a value in [-1, 1].
float pearsonCorrelation12(const float* a, const float* b) noexcept {
  float meanA = 0.0f, meanB = 0.0f;
  for (int i = 0; i < 12; ++i) {
    meanA += a[i];
    meanB += b[i];
  }
  meanA /= 12.0f;
  meanB /= 12.0f;
  float num = 0.0f, dA = 0.0f, dB = 0.0f;
  for (int i = 0; i < 12; ++i) {
    const float da = a[i] - meanA;
    const float db = b[i] - meanB;
    num += da * db;
    dA  += da * da;
    dB  += db * db;
  }
  if (dA <= 0.0f || dB <= 0.0f) return 0.0f;
  return num / std::sqrt(dA * dB);
}

}  // namespace

KeyInference inferKey(const float* samples, std::int64_t frameCount,
                       std::uint32_t channels,
                       std::uint32_t sampleRate) noexcept {
  PerFrameChromagram pfc = computePerFrameChromagram(
      samples, frameCount, channels, sampleRate);
  if (pfc.numFrames < 1) {
    return KeyInference{255, false, 0.0f};
  }

  // Sum across all frames into one 12-element chromagram.
  std::array<double, 12> chroma{};
  for (std::int64_t f = 0; f < pfc.numFrames; ++f) {
    for (int pc = 0; pc < 12; ++pc) {
      chroma[static_cast<std::size_t>(pc)] +=
          pfc.data[static_cast<std::size_t>(f) * 12 +
                   static_cast<std::size_t>(pc)];
    }
  }

  // Normalize the chromagram.
  double total = 0.0;
  for (double v : chroma) total += v;
  if (total <= 0.0) {
    return KeyInference{255, false, 0.0f};
  }
  float chromaNorm[12];
  for (int i = 0; i < 12; ++i) {
    chromaNorm[i] = static_cast<float>(chroma[static_cast<std::size_t>(i)] /
                                        total);
  }

  // Correlate against every rotated key profile (24 keys total).
  float bestCorr = -2.0f;
  std::uint8_t bestPC = 0;
  bool bestMinor = false;

  for (int rot = 0; rot < 12; ++rot) {
    float rotMajor[12];
    float rotMinor[12];
    for (int i = 0; i < 12; ++i) {
      // The key whose tonic is at pitch class `rot` has profile peak at
      // index `rot`. So rotMajor[i] = kProfileMajor[(i - rot + 12) % 12].
      const int idx = (i - rot + 12) % 12;
      rotMajor[i] = kProfileMajor[idx];
      rotMinor[i] = kProfileMinor[idx];
    }
    const float cMaj = pearsonCorrelation12(chromaNorm, rotMajor);
    const float cMin = pearsonCorrelation12(chromaNorm, rotMinor);
    if (cMaj > bestCorr) {
      bestCorr = cMaj;
      bestPC = static_cast<std::uint8_t>(rot);
      bestMinor = false;
    }
    if (cMin > bestCorr) {
      bestCorr = cMin;
      bestPC = static_cast<std::uint8_t>(rot);
      bestMinor = true;
    }
  }

  // Correlation can be negative for content that anti-correlates with every
  // key (very atonal); clamp to [0, 1] for display.
  const float confidence = std::max(0.0f, std::min(1.0f, bestCorr));
  return KeyInference{bestPC, bestMinor, confidence};
}

// ---------------------------------------------------------------------------
// Chord / note progression (Phase 3c)
// ---------------------------------------------------------------------------
//
// Two ideas make this stable on real instruments:
//   1. Half-beat analysis windows + harmonic-weighted templates, instead of
//      sixteenth windows + binary triads. A single sixteenth of a strummed
//      chord is whatever string is ringing loudest at that instant; a half-
//      beat settles. Binary {1,0,0,..} templates fight the smeared spectrum
//      of a real instrument (octave doublings, harmonics) — we instead bake
//      the first few harmonics of each chord tone into the template.
//   2. A Viterbi decode with a sticky self-transition. Chords persist; the
//      old per-window argmax had zero continuity cost so it flickered. The
//      self-transition bonus means a label only changes when the evidence
//      persistently disagrees — which also makes "minimum segment length"
//      emerge for free instead of being a post-hoc filter.
//
// State space: 12 major triads + 12 minor triads + 12 single notes + N/C.
// The single-note states catch monophonic content (sung melody, one-note
// guitar lines) — for an a-cappella the decode becomes a melody contour.

namespace {

// 36 = 12 maj + 12 min + 12 single-note. N/C is state index 36.
constexpr int kNumChordTemplates = 36;
constexpr int kNumChordStates    = kNumChordTemplates + 1;   // + N/C
constexpr int kNoChordState      = kNumChordTemplates;       // = 36

inline bool stateIsMajor(int s) noexcept { return s >= 0  && s < 12; }
inline bool stateIsMinor(int s) noexcept { return s >= 12 && s < 24; }
inline bool stateIsNote(int s)  noexcept { return s >= 24 && s < 36; }
inline int  stateRootPc(int s)  noexcept { return s % 12; }

// Lay a played pitch class into a 12-bin template: a strong weight on the pc
// itself (the fundamental + its octaves), a *small* nudge to its perfect
// fifth above (pc+7 — the 3rd/6th harmonics) and major third above (pc+4 —
// the 5th harmonic). Earlier this used 0.5 / 0.25 for the harmonic terms,
// which backfired: in a triad the fifth is *also* a chord tone, so the
// fifth ended up the template's tallest bin and a 1-peak "single note"
// template would out-correlate the triad on a real strum (→ everything got
// labelled a note). Keeping the harmonic terms tiny and giving the *root* a
// heavier base weight (real chords double the root in the bass) puts the
// template's peak back where the ear puts it.
inline void addTone(float w[12], int pc, float baseWeight) noexcept {
  w[pc % 12]       += baseWeight;
  w[(pc + 7) % 12] += 0.20f;   // 3rd/6th harmonics → perfect fifth above
  w[(pc + 4) % 12] += 0.10f;   // 5th harmonic → major third above
}

constexpr float kRootWeight  = 1.50f;   // root sits above the third/fifth
constexpr float kOtherWeight  = 1.00f;

void buildChordTemplates(float out[kNumChordTemplates][12]) noexcept {
  for (int s = 0; s < kNumChordTemplates; ++s) {
    for (int i = 0; i < 12; ++i) out[s][i] = 0.0f;
    const int r = stateRootPc(s);
    if (stateIsMajor(s)) {
      addTone(out[s], r, kRootWeight);
      addTone(out[s], (r + 4) % 12, kOtherWeight);  // major third
      addTone(out[s], (r + 7) % 12, kOtherWeight);  // perfect fifth
    } else if (stateIsMinor(s)) {
      addTone(out[s], r, kRootWeight);
      addTone(out[s], (r + 3) % 12, kOtherWeight);  // minor third
      addTone(out[s], (r + 7) % 12, kOtherWeight);  // perfect fifth
    } else {  // single note
      addTone(out[s], r, kRootWeight);
    }
  }
}

void stateToLabel(int s, std::uint8_t* outPc, std::uint8_t* outQuality) noexcept {
  if (s == kNoChordState) { *outPc = 255; *outQuality = 0; return; }
  *outPc = static_cast<std::uint8_t>(stateRootPc(s));
  *outQuality = stateIsMajor(s) ? 0 : (stateIsMinor(s) ? 1 : 2);
}

// Best-correlating template index in [0, kNumChordTemplates); writes the
// correlation (in [-1, 1]) to *outCorr.
int bestChordTemplate(const float chroma[12],
                      const float templates[kNumChordTemplates][12],
                      float* outCorr) noexcept {
  int best = 0;
  float bestC = -2.0f;
  for (int s = 0; s < kNumChordTemplates; ++s) {
    const float c = pearsonCorrelation12(chroma, templates[s]);
    if (c > bestC) { bestC = c; best = s; }
  }
  *outCorr = bestC;
  return best;
}

// "Matches nothing in particular" reference score. A window whose best
// template correlation falls below this is labelled N/C. (Tune against real
// recordings — see AUDIO_ENGINE_ARCHITECTURE.md §3c.)
constexpr float kNoChordScore = 0.32f;

// Viterbi transition costs, subtracted from the running path score. λ is THE
// stability knob: bigger ⇒ stickier ⇒ fewer label changes. Emission is in
// [-1, 1], so a change is worth it only when the new label beats the old by
// more than ~λ, sustained. Looped guitar parts hold a chord for bars at a
// time, so we want this firmly sticky; a sub-beat min-segment merge after the
// decode mops up whatever flicker still gets through. kOutOfKeyExtra makes a
// change *to* a chord/note outside the inferred key cost more — diatonic
// neighbours are cheap.
constexpr float kChangePenalty = 0.72f;
// Key bias is currently OFF. On short/sparse loops the key inference is close
// to a coin flip, and a *wrong* key drags the chords toward the wrong diatonic
// set (e.g. it reported a held F#-minor chord as F# major because the key came
// back F# major). Once the key is derived *from* the detected chords (root
// histogram weighted by duration) rather than the other way around, re-enable
// this. The plumbing (keyPitchClass param, chordDiatonic, outOfKeyCost) is
// kept so flipping it back on is a one-line change.
constexpr float kOutOfKeyExtra = 0.0f;
// Head start a triad gets over a "single note" of the same root in the
// per-window emission — see the note where it's applied. ~0.12 flips a close
// chord-vs-note call without touching a clear monophonic line.
constexpr float kTriadBonus = 0.12f;
// Segments shorter than this (sixteenths) get absorbed into a neighbour.
constexpr std::int32_t kMinSegmentSixteenths = 4;  // one beat

// Major / natural-minor scale-degree masks, bit i set ⇒ semitone i above the
// tonic is in the scale. Major = {0,2,4,5,7,9,11}; natural minor = {0,2,3,5,7,8,10}.
constexpr unsigned kMajorScaleMask = 0b101010110101u;
constexpr unsigned kMinorScaleMask = 0b010110101101u;

inline bool pcInKeyScale(int pc, int keyPc, bool keyMinor) noexcept {
  const int rel = ((pc - keyPc) % 12 + 12) % 12;
  return ((keyMinor ? kMinorScaleMask : kMajorScaleMask) >> rel) & 1u;
}

// Is chord-template state `s` diatonic to the given key? Approx but adequate
// for a bias term: all of the state's pitch classes must lie in the scale.
bool chordDiatonic(int s, int keyPc, bool keyMinor) noexcept {
  if (keyPc < 0 || keyPc > 11) return true;   // no key → never penalize
  const int r = stateRootPc(s);
  if (stateIsNote(s)) return pcInKeyScale(r, keyPc, keyMinor);
  const int third = stateIsMajor(s) ? (r + 4) % 12 : (r + 3) % 12;
  const int fifth = (r + 7) % 12;
  return pcInKeyScale(r, keyPc, keyMinor) &&
         pcInKeyScale(third, keyPc, keyMinor) &&
         pcInKeyScale(fifth, keyPc, keyMinor);
}

}  // namespace

ChordEstimate estimateChordFromChroma(const float chroma12[12]) noexcept {
  float total = 0.0f;
  for (int i = 0; i < 12; ++i) total += chroma12[i];
  if (total < 1e-6f) return ChordEstimate{255, 0, 0.0f};

  float templates[kNumChordTemplates][12];
  buildChordTemplates(templates);
  float corr = -2.0f;
  const int best = bestChordTemplate(chroma12, templates, &corr);
  if (corr < kNoChordScore) {
    return ChordEstimate{255, 0, std::max(0.0f, corr)};
  }
  std::uint8_t pc = 0, q = 0;
  stateToLabel(best, &pc, &q);
  return ChordEstimate{pc, q, std::min(1.0f, corr)};
}

ChordEstimate estimateChord(const float* samples, std::int64_t frameCount,
                             std::uint32_t channels,
                             std::uint32_t sampleRate) noexcept {
  PerFrameChromagram pfc = computePerFrameChromagram(
      samples, frameCount, channels, sampleRate);
  if (pfc.numFrames < 1) return ChordEstimate{255, 0, 0.0f};

  float chroma[12] = {0};
  for (std::int64_t f = 0; f < pfc.numFrames; ++f) {
    for (int pc = 0; pc < 12; ++pc) {
      chroma[pc] += pfc.data[static_cast<std::size_t>(f) * 12 +
                             static_cast<std::size_t>(pc)];
    }
  }
  float total = 0.0f;
  for (float v : chroma) total += v;
  if (total <= 0.0f) return ChordEstimate{255, 0, 0.0f};
  for (int i = 0; i < 12; ++i) chroma[i] /= total;
  return estimateChordFromChroma(chroma);
}

std::vector<ChordSegment> recognizeChords(
    const float* samples, std::int64_t frameCount,
    std::uint32_t channels, std::uint32_t sampleRate,
    double bpm, std::uint32_t quantum,
    std::uint8_t keyPitchClass, bool keyIsMinor) noexcept {
  std::vector<ChordSegment> result;
  if (bpm <= 0.0 || quantum == 0) return result;

  PerFrameChromagram pfc = computePerFrameChromagram(
      samples, frameCount, channels, sampleRate);
  if (pfc.numFrames < 1) return result;

  // Output grid: sixteenth notes. Analysis grid: half-beats (eighth notes).
  const int numSixteenths = static_cast<int>(quantum) * 4;
  const int numWindows    = static_cast<int>(quantum) * 2;   // half-beats
  if (numSixteenths < 1 || numWindows < 2) return result;

  const double secPerHop =
      static_cast<double>(kFftHop) / static_cast<double>(sampleRate);
  const double secPerFftWindow =
      static_cast<double>(kFftSize) / static_cast<double>(sampleRate);
  const double loopDurSec =
      static_cast<double>(frameCount) / static_cast<double>(sampleRate);
  const double winDurSec = loopDurSec / static_cast<double>(numWindows);

  // ---- Aggregate the per-FFT-frame chromagram into half-beat windows ----
  std::vector<std::array<float, 12>> winChroma(
      static_cast<std::size_t>(numWindows), std::array<float, 12>{});
  std::vector<int> winFrames(static_cast<std::size_t>(numWindows), 0);
  for (std::int64_t f = 0; f < pfc.numFrames; ++f) {
    const double midSec = static_cast<double>(f) * secPerHop +
                          secPerFftWindow * 0.5;
    int w = static_cast<int>(std::floor(midSec / winDurSec));
    if (w < 0) w = 0;
    if (w >= numWindows) w = numWindows - 1;
    for (int pc = 0; pc < 12; ++pc) {
      winChroma[static_cast<std::size_t>(w)][static_cast<std::size_t>(pc)] +=
          pfc.data[static_cast<std::size_t>(f) * 12 +
                   static_cast<std::size_t>(pc)];
    }
    ++winFrames[static_cast<std::size_t>(w)];
  }
  for (int w = 0; w < numWindows; ++w) {
    float total = 0.0f;
    for (float v : winChroma[static_cast<std::size_t>(w)]) total += v;
    if (total > 1e-12f) {
      for (int i = 0; i < 12; ++i) {
        winChroma[static_cast<std::size_t>(w)][static_cast<std::size_t>(i)] /=
            total;
      }
    }
  }

  // ---- Emission scores: correlation of each window vs each template ----
  float templates[kNumChordTemplates][12];
  buildChordTemplates(templates);
  std::vector<std::array<float, kNumChordStates>> emission(
      static_cast<std::size_t>(numWindows));
  for (int w = 0; w < numWindows; ++w) {
    auto& row = emission[static_cast<std::size_t>(w)];
    if (winFrames[static_cast<std::size_t>(w)] == 0) {
      // No spectral evidence in this window: neutral for every chord, N/C
      // slightly preferred. The self-transition normally carries a held
      // label straight through a one-window dropout anyway.
      for (int s = 0; s < kNumChordStates; ++s) row[s] = 0.0f;
      row[kNoChordState] = kNoChordScore;
      continue;
    }
    for (int s = 0; s < kNumChordTemplates; ++s) {
      row[s] = pearsonCorrelation12(
          winChroma[static_cast<std::size_t>(w)].data(), templates[s]);
      // Looped material is overwhelmingly chordal, and a strummed guitar
      // chord whose root dominates the chroma (bass doubling, low strings)
      // matches a 1-peak "single note" template *better* than the 3-peak
      // triad — so a triad needs a small head start to win that contest.
      // Small enough that a genuinely monophonic line (note correlates ~1.0,
      // any triad ~0.6) still reads as a note.
      if (stateIsMajor(s) || stateIsMinor(s)) row[s] += kTriadBonus;
    }
    row[kNoChordState] = kNoChordScore;
  }

  // Per-destination-state extra penalty for changing *into* a non-diatonic
  // chord/note. (Staying in one is free — the bias only resists committing.)
  std::array<float, kNumChordStates> outOfKeyCost{};
  for (int s = 0; s < kNumChordTemplates; ++s) {
    outOfKeyCost[static_cast<std::size_t>(s)] =
        chordDiatonic(s, static_cast<int>(keyPitchClass), keyIsMinor)
            ? 0.0f
            : kOutOfKeyExtra;
  }
  outOfKeyCost[kNoChordState] = 0.0f;

  // ---- Viterbi over the window sequence ----
  std::vector<std::array<float, kNumChordStates>> delta(
      static_cast<std::size_t>(numWindows));
  std::vector<std::array<std::int16_t, kNumChordStates>> psi(
      static_cast<std::size_t>(numWindows));
  for (int s = 0; s < kNumChordStates; ++s) {
    delta[0][static_cast<std::size_t>(s)] = emission[0][static_cast<std::size_t>(s)];
    psi[0][static_cast<std::size_t>(s)] = -1;
  }
  for (int w = 1; w < numWindows; ++w) {
    const auto& prev = delta[static_cast<std::size_t>(w - 1)];
    auto& cur  = delta[static_cast<std::size_t>(w)];
    auto& cpsi = psi[static_cast<std::size_t>(w)];
    const auto& emit = emission[static_cast<std::size_t>(w)];
    for (int s = 0; s < kNumChordStates; ++s) {
      const float changeCost = kChangePenalty + outOfKeyCost[static_cast<std::size_t>(s)];
      float bestV = -1e30f;
      int   bestSp = 0;
      for (int sp = 0; sp < kNumChordStates; ++sp) {
        const float v = prev[static_cast<std::size_t>(sp)] -
                        (sp == s ? 0.0f : changeCost);
        if (v > bestV) { bestV = v; bestSp = sp; }
      }
      cur[static_cast<std::size_t>(s)] = bestV + emit[static_cast<std::size_t>(s)];
      cpsi[static_cast<std::size_t>(s)] = static_cast<std::int16_t>(bestSp);
    }
  }

  // Backtrack.
  std::vector<int> winState(static_cast<std::size_t>(numWindows), kNoChordState);
  {
    int bestS = 0;
    float bestV = -1e30f;
    for (int s = 0; s < kNumChordStates; ++s) {
      const float v = delta[static_cast<std::size_t>(numWindows - 1)]
                           [static_cast<std::size_t>(s)];
      if (v > bestV) { bestV = v; bestS = s; }
    }
    winState[static_cast<std::size_t>(numWindows - 1)] = bestS;
    for (int w = numWindows - 1; w > 0; --w) {
      winState[static_cast<std::size_t>(w - 1)] =
          psi[static_cast<std::size_t>(w)]
             [static_cast<std::size_t>(winState[static_cast<std::size_t>(w)])];
    }
  }

  // ---- Merge consecutive identical labels into segments (sixteenth grid) --
  // Each half-beat window spans exactly 2 sixteenths.
  auto windowScore = [&](int w) {
    return std::max(0.0f, emission[static_cast<std::size_t>(w)]
                                  [static_cast<std::size_t>(
                                      winState[static_cast<std::size_t>(w)])]);
  };
  std::uint8_t segPc = 0, segQ = 0;
  stateToLabel(winState[0], &segPc, &segQ);
  int   segStartWin = 0;
  float segScoreSum = windowScore(0);
  int   segWinCount = 1;
  for (int w = 1; w < numWindows; ++w) {
    std::uint8_t pc = 0, q = 0;
    stateToLabel(winState[static_cast<std::size_t>(w)], &pc, &q);
    const bool same = (pc == segPc) && (pc == 255 || q == segQ);
    if (same) { segScoreSum += windowScore(w); ++segWinCount; continue; }
    result.push_back(ChordSegment{
        segStartWin * 2, w * 2, segPc, segQ,
        segWinCount > 0 ? std::min(1.0f, segScoreSum / segWinCount) : 0.0f,
    });
    segStartWin = w;
    segPc = pc;
    segQ = q;
    segScoreSum = windowScore(w);
    segWinCount = 1;
  }
  result.push_back(ChordSegment{
      segStartWin * 2, numWindows * 2, segPc, segQ,
      segWinCount > 0 ? std::min(1.0f, segScoreSum / segWinCount) : 0.0f,
  });

  // ---- Absorb sub-beat slivers into their stronger neighbour --------------
  // Even with a sticky Viterbi, a window or two at a chord boundary (or a
  // strum attack) can pop out as its own tiny segment. Looped parts never
  // change harmony faster than a beat, so merge anything shorter than that
  // into whichever neighbour has the higher confidence, then re-coalesce
  // adjacent identical labels.
  {
    bool changed = true;
    while (changed && result.size() > 1) {
      changed = false;
      for (std::size_t i = 0; i < result.size(); ++i) {
        if (result[i].endSixteenth - result[i].startSixteenth >=
            kMinSegmentSixteenths) {
          continue;
        }
        std::size_t into;
        if (i == 0) {
          into = 1;
        } else if (i + 1 == result.size()) {
          into = i - 1;
        } else {
          into = (result[i - 1].confidence >= result[i + 1].confidence)
                     ? i - 1
                     : i + 1;
        }
        result[into].startSixteenth =
            std::min(result[into].startSixteenth, result[i].startSixteenth);
        result[into].endSixteenth =
            std::max(result[into].endSixteenth, result[i].endSixteenth);
        result.erase(result.begin() + static_cast<std::ptrdiff_t>(i));
        changed = true;
        break;  // indices shifted — restart the scan
      }
    }
    // Re-label a "single note" span as the triad it's bracketed by, when a
    // neighbour is a triad on the same root — that's almost always the decay
    // tail (or attack) of that chord reading as just its bass note, not a
    // real melody note. (A genuine melody note is flanked by other notes or
    // by chords on *different* roots, so this leaves melodies alone.)
    for (std::size_t i = 0; i < result.size(); ++i) {
      if (result[i].quality != 2 || result[i].pitchClass > 11) continue;
      const std::uint8_t pc = result[i].pitchClass;
      int relabelQ = -1;
      float bestConf = -1.0f;
      if (i > 0 && result[i - 1].quality <= 1 && result[i - 1].pitchClass == pc) {
        relabelQ = result[i - 1].quality;
        bestConf = result[i - 1].confidence;
      }
      if (i + 1 < result.size() && result[i + 1].quality <= 1 &&
          result[i + 1].pitchClass == pc &&
          result[i + 1].confidence > bestConf) {
        relabelQ = result[i + 1].quality;
      }
      if (relabelQ >= 0) result[i].quality = static_cast<std::uint8_t>(relabelQ);
    }
    for (std::size_t i = 1; i < result.size();) {
      if (result[i].pitchClass == result[i - 1].pitchClass &&
          (result[i].pitchClass == 255 ||
           result[i].quality == result[i - 1].quality)) {
        result[i - 1].endSixteenth = result[i].endSixteenth;
        result[i - 1].confidence =
            std::max(result[i - 1].confidence, result[i].confidence);
        result.erase(result.begin() + static_cast<std::ptrdiff_t>(i));
      } else {
        ++i;
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Monophonic pitch detection (YIN)
// ---------------------------------------------------------------------------
// de Cheveigné & Kawahara 2002. Steps:
//   1. Difference function d(τ) over the instrument lag range.
//   2. Cumulative-mean-normalized difference d'(τ).
//   3. Absolute threshold — first τ where d' dips below it (walk to the dip's
//      local minimum). Fallback to the global minimum of d' if nothing dips.
//   4. Parabolic interpolation around that τ for sub-sample precision.
//   5. f0 = sampleRate / τ; clarity = 1 − d'(τ*).
// Bounded τ range keeps the O(W·τ_max) difference function affordable to poll
// at ~10 Hz; tier down the window/poll on weak hardware.

namespace {

constexpr float       kYinThreshold   = 0.15f;   // d'(τ) must dip below this
constexpr double      kTunerMinFreqHz = 50.0;    // ~G#1 — below drop-tunings
constexpr double      kTunerMaxFreqHz = 1500.0;  // above the guitar's top frets
constexpr std::size_t kYinMaxWindow   = 4096;    // analysis window cap

}  // namespace

PitchEstimate detectPitch(const float* samples, std::int64_t frameCount,
                          std::uint32_t channels,
                          std::uint32_t sampleRate) noexcept {
  if (samples == nullptr || frameCount <= 0 || channels == 0 ||
      sampleRate == 0) {
    return PitchEstimate{0.0f, 0.0f};
  }

  // Use the most recent up-to-kYinMaxWindow frames.
  const std::size_t W = static_cast<std::size_t>(std::min<std::int64_t>(
      frameCount, static_cast<std::int64_t>(kYinMaxWindow)));
  const std::size_t tauMax = std::min<std::size_t>(
      W / 2, static_cast<std::size_t>(static_cast<double>(sampleRate) /
                                      kTunerMinFreqHz));
  const std::size_t tauMin = std::max<std::size_t>(
      2, static_cast<std::size_t>(static_cast<double>(sampleRate) /
                                  kTunerMaxFreqHz));
  if (tauMax <= tauMin + 2 || W < 64) return PitchEstimate{0.0f, 0.0f};

  // Downmix the most-recent W frames to mono.
  const std::int64_t start = frameCount - static_cast<std::int64_t>(W);
  static thread_local std::vector<float> mono;
  mono.assign(W, 0.0f);
  double energy = 0.0;
  for (std::size_t i = 0; i < W; ++i) {
    double m = 0.0;
    for (std::uint32_t c = 0; c < channels; ++c) {
      m += samples[(start + static_cast<std::int64_t>(i)) * channels + c];
    }
    m /= channels;
    mono[i] = static_cast<float>(m);
    energy += m * m;
  }
  if (energy / static_cast<double>(W) < 1e-7) return PitchEstimate{0.0f, 0.0f};

  // (1) Difference function d(τ).
  static thread_local std::vector<double> d;
  d.assign(tauMax + 1, 0.0);
  for (std::size_t tau = tauMin; tau <= tauMax; ++tau) {
    double sum = 0.0;
    const std::size_t n = W - tau;
    for (std::size_t i = 0; i < n; ++i) {
      const double diff = static_cast<double>(mono[i]) -
                          static_cast<double>(mono[i + tau]);
      sum += diff * diff;
    }
    d[tau] = sum;
  }

  // (2) Cumulative mean normalized difference d'(τ).
  static thread_local std::vector<double> dp;
  dp.assign(tauMax + 1, 1.0);
  double running = 0.0;
  for (std::size_t tau = tauMin; tau <= tauMax; ++tau) {
    running += d[tau];
    dp[tau] = (running > 0.0)
                  ? d[tau] * static_cast<double>(tau - tauMin + 1) / running
                  : 1.0;
  }

  // (3) Absolute threshold.
  std::size_t tauStar = 0;
  for (std::size_t tau = tauMin + 1; tau < tauMax; ++tau) {
    if (dp[tau] < kYinThreshold) {
      while (tau + 1 <= tauMax && dp[tau + 1] < dp[tau]) ++tau;
      tauStar = tau;
      break;
    }
  }
  if (tauStar == 0) {
    double best = 1.0;
    std::size_t bestTau = 0;
    for (std::size_t tau = tauMin; tau <= tauMax; ++tau) {
      if (dp[tau] < best) { best = dp[tau]; bestTau = tau; }
    }
    if (bestTau == 0 || best > 0.5) return PitchEstimate{0.0f, 0.0f};
    tauStar = bestTau;
  }

  // (4) Parabolic interpolation around tauStar (on the raw difference d).
  double betterTau = static_cast<double>(tauStar);
  if (tauStar > tauMin && tauStar < tauMax) {
    const double y0 = d[tauStar - 1], y1 = d[tauStar], y2 = d[tauStar + 1];
    const double denom = (y0 + y2 - 2.0 * y1);
    if (std::fabs(denom) > 1e-12) {
      const double adj = 0.5 * (y0 - y2) / denom;
      if (adj > -1.0 && adj < 1.0) betterTau += adj;
    }
  }
  if (betterTau <= 0.0) return PitchEstimate{0.0f, 0.0f};

  const double f0 = static_cast<double>(sampleRate) / betterTau;
  if (f0 < kTunerMinFreqHz || f0 > kTunerMaxFreqHz) {
    return PitchEstimate{0.0f, 0.0f};
  }
  const float clarity = static_cast<float>(
      std::max(0.0, std::min(1.0, 1.0 - dp[tauStar])));
  return PitchEstimate{static_cast<float>(f0), clarity};
}

}  // namespace audio_engine
}  // namespace flowstate
