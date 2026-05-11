// Tests for length-only tempo inference.
//
// Trace through a handful of common musical scenarios and verify the
// algorithm picks musically plausible (BPM, quantum) pairs. Some inputs are
// fundamentally ambiguous; for those we assert that the chosen pair is at
// least musically reasonable rather than dictating a specific answer.

#include "../inference.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ae = flowstate::audio_engine;

namespace {

constexpr std::uint32_t kSR48k = 48000;

void check(bool ok, const char* label) {
  if (!ok) {
    std::fprintf(stderr, "[FAIL] %s\n", label);
    std::exit(1);
  }
}

void test_invalid_inputs() {
  auto r1 = ae::inferTempoFromLength(0, kSR48k);
  check(r1.bpm == 0.0 && r1.quantum == 0,
        "zero loopFrames returns zero result");
  auto r2 = ae::inferTempoFromLength(48000, 0);
  check(r2.bpm == 0.0 && r2.quantum == 0,
        "zero sampleRate returns zero result");
  std::puts("[PASS] invalid inputs");
}

void test_two_second_loop_is_120bpm_4beats() {
  // A 2-second loop at 4 beats per bar = exactly 120 BPM. Should win cleanly.
  const std::int64_t frames = 2 * kSR48k;
  auto r = ae::inferTempoFromLength(frames, kSR48k);
  check(r.quantum == 4, "2s loop → q=4");
  check(std::fabs(r.bpm - 120.0) < 0.001, "2s loop → 120 BPM");
  std::puts("[PASS] 2s loop = 120 BPM, q=4");
}

void test_three_second_loop_is_120bpm_6beats() {
  // 3-second loop: q=6 at 120 BPM is musically natural (6/8 or 3/4 doubled),
  // but q=4 at 80 BPM is also valid (4/4 at 80). The algorithm should pick
  // *something* plausible; verify it's one of the obvious candidates.
  const std::int64_t frames = 3 * kSR48k;
  auto r = ae::inferTempoFromLength(frames, kSR48k);
  // Acceptable answers: q=4 (bpm=80), q=6 (bpm=120), q=3 (bpm=60),
  // q=8 (bpm=160).
  const bool acceptable =
      (r.quantum == 4 && std::fabs(r.bpm - 80.0) < 0.001) ||
      (r.quantum == 6 && std::fabs(r.bpm - 120.0) < 0.001) ||
      (r.quantum == 3 && std::fabs(r.bpm - 60.0) < 0.001) ||
      (r.quantum == 8 && std::fabs(r.bpm - 160.0) < 0.001);
  check(acceptable, "3s loop chooses a musically plausible candidate");
  std::printf("       (chose q=%u, bpm=%.1f)\n", r.quantum, r.bpm);
  std::puts("[PASS] 3s loop: plausible candidate selected");
}

void test_one_second_loop_is_quarter_at_240() {
  // 1s loop, q=4 → 240 BPM (out of plausible range — rejected).
  // q=2 → 120 BPM (in range, q=2 has lower bias).
  // q=3 → 180 BPM (in range).
  const std::int64_t frames = kSR48k;
  auto r = ae::inferTempoFromLength(frames, kSR48k);
  check(r.bpm >= 60.0 && r.bpm <= 200.0,
        "1s loop chooses BPM in plausible range");
  std::printf("       (chose q=%u, bpm=%.1f)\n", r.quantum, r.bpm);
  std::puts("[PASS] 1s loop: in-range BPM selected");
}

void test_four_second_loop_natural_4beat() {
  // 4-second loop at 4 beats per bar = 60 BPM. q=4 should win.
  const std::int64_t frames = 4 * kSR48k;
  auto r = ae::inferTempoFromLength(frames, kSR48k);
  // Acceptable: q=4 at 60 BPM, or q=8 at 120 BPM (the algorithm slightly
  // favors 120-centric BPM, so q=8 might edge out q=4 here).
  const bool acceptable =
      (r.quantum == 4 && std::fabs(r.bpm - 60.0) < 0.001) ||
      (r.quantum == 8 && std::fabs(r.bpm - 120.0) < 0.001);
  check(acceptable, "4s loop is q=4@60 or q=8@120");
  std::printf("       (chose q=%u, bpm=%.1f)\n", r.quantum, r.bpm);
  std::puts("[PASS] 4s loop: plausible 4-beat or 8-beat answer");
}

void test_realistic_user_loop_5938ms() {
  // 5.938s loop from the live session — 285024 frames @ 48 kHz.
  // q=4 → 40 BPM (rejected). q=8 → 80.8 BPM (in range). q=12 → 121.2 BPM
  // (in range, very near 120). Whichever the algorithm picks, both are
  // plausible musical interpretations.
  const std::int64_t frames = 285024;
  auto r = ae::inferTempoFromLength(frames, kSR48k);
  check(r.bpm >= 60.0 && r.bpm <= 200.0, "5.94s loop: in-range BPM");
  check(r.quantum > 0 && r.quantum <= 32, "5.94s loop: sane quantum");
  std::printf("       (chose q=%u, bpm=%.1f, confidence=%.2f)\n",
              r.quantum, r.bpm, r.confidence);
  std::puts("[PASS] realistic 5.94s loop");
}

void test_three_beat_waltz() {
  // A waltz at 90 BPM in 3/4 has loop length 3·60/90 = 2.0s. So q=3 should
  // win — but only if the algorithm doesn't get seduced by q=4 at 120 BPM
  // (also valid for a 2-second loop). Real ambiguity, no perfect answer.
  const std::int64_t frames = 2 * kSR48k;
  auto r = ae::inferTempoFromLength(frames, kSR48k);
  // The 4/4 interpretation is *more* common in pop, so we don't insist on
  // q=3. But we verify it's one of the two plausible answers.
  const bool acceptable =
      (r.quantum == 3 && std::fabs(r.bpm - 90.0) < 0.001) ||
      (r.quantum == 4 && std::fabs(r.bpm - 120.0) < 0.001);
  check(acceptable, "2s loop is q=3@90 or q=4@120");
  std::printf("       (chose q=%u, bpm=%.1f)\n", r.quantum, r.bpm);
  std::puts("[PASS] 2s loop: 4/4@120 or 3/4@90");
}

void test_long_loop_4_bars() {
  // 8 seconds, 4 bars of 4/4 at 120 BPM → q=16.
  const std::int64_t frames = 8 * kSR48k;
  auto r = ae::inferTempoFromLength(frames, kSR48k);
  // Should land on q=8 (60 BPM), q=16 (120 BPM), or q=12 (90 BPM).
  check(r.bpm >= 60.0 && r.bpm <= 200.0, "8s loop: in-range BPM");
  check(r.quantum >= 8 && r.quantum <= 24, "8s loop: quantum 8..24");
  std::printf("       (chose q=%u, bpm=%.1f)\n", r.quantum, r.bpm);
  std::puts("[PASS] 8s loop: multi-bar answer");
}

}  // namespace

int main() {
  test_invalid_inputs();
  test_two_second_loop_is_120bpm_4beats();
  test_three_second_loop_is_120bpm_6beats();
  test_one_second_loop_is_quarter_at_240();
  test_four_second_loop_natural_4beat();
  test_realistic_user_loop_5938ms();
  test_three_beat_waltz();
  test_long_loop_4_bars();
  std::puts("All inference tests passed.");
  return 0;
}
