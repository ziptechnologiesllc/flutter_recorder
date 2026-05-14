// Tests for LocalClock musical-time math.
//
// All numbers chosen so the expected values are exact integers / simple
// fractions, so we can use exact comparisons instead of fudge factors.

#include "../local_clock.h"

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

bool nearlyEqual(double a, double b, double eps = 1e-9) {
  return std::fabs(a - b) <= eps;
}

void test_invalid_when_unset() {
  ae::LocalClock c;
  check(!c.isValid(), "clock is invalid before any tempo set");
  check(c.tempoBPM() == 0.0, "tempo zero when unset");
  check(c.quantum() == 0, "quantum zero when unset");
  check(c.phaseInBar(1234, kSR48k) == 0.0,
        "phase returns 0 when unset");
  check(c.nextDownbeatFrame(1234, kSR48k) == 1234,
        "nextDownbeat returns input when unset");
  std::puts("[PASS] unset clock returns zero defaults");
}

void test_set_clear_round_trip() {
  ae::LocalClock c;
  c.setTempo(120.0, 4, 1000);
  check(c.isValid(), "valid after setTempo");
  check(c.tempoBPM() == 120.0, "tempo stored");
  check(c.quantum() == 4, "quantum stored");
  check(c.anchorFrame() == 1000, "anchor stored");

  c.clear();
  check(!c.isValid(), "invalid after clear");
  check(c.tempoBPM() == 0.0, "tempo zeroed after clear");

  std::puts("[PASS] set/clear round trip");
}

void test_phase_at_anchor() {
  ae::LocalClock c;
  // 120 BPM, 4/4 ⇒ 24000 frames per beat ⇒ 96000 frames per bar at 48 kHz.
  c.setTempo(120.0, 4, /*anchor=*/0);
  check(c.phaseInBar(0, kSR48k) == 0.0, "phase = 0 at anchor");
  check(c.phaseInBar(96000, kSR48k) == 0.0, "phase wraps to 0 at next bar");
  check(nearlyEqual(c.phaseInBar(48000, kSR48k), 0.5),
        "phase = 0.5 at half bar");
  check(nearlyEqual(c.phaseInBar(24000, kSR48k), 0.25),
        "phase = 0.25 after one beat");
  std::puts("[PASS] phaseInBar math at common positions");
}

void test_phase_with_nonzero_anchor() {
  ae::LocalClock c;
  // Anchor at frame 1000; one bar = 96000 frames.
  c.setTempo(120.0, 4, /*anchor=*/1000);
  check(c.phaseInBar(1000, kSR48k) == 0.0, "phase = 0 at the anchor");
  check(c.phaseInBar(1000 + 96000, kSR48k) == 0.0, "phase wraps next bar");
  check(nearlyEqual(c.phaseInBar(1000 + 48000, kSR48k), 0.5),
        "phase = 0.5 at half bar past anchor");
  std::puts("[PASS] phaseInBar respects anchor");
}

void test_beat_count() {
  ae::LocalClock c;
  c.setTempo(120.0, 4, /*anchor=*/0);
  check(c.beatAtFrame(0, kSR48k) == 0, "beat 0 at anchor");
  check(c.beatAtFrame(24000, kSR48k) == 1, "beat 1 after one beat");
  check(c.beatAtFrame(96000, kSR48k) == 4, "beat 4 after one bar (4/4)");
  check(c.beatAtFrame(96001, kSR48k) == 4, "beat 4 just past one bar");
  // Frame before anchor.
  check(c.beatAtFrame(-100, kSR48k) == 0, "beat 0 before anchor");
  std::puts("[PASS] beat count math");
}

void test_next_downbeat() {
  ae::LocalClock c;
  c.setTempo(120.0, 4, /*anchor=*/0);
  // At anchor → returns anchor.
  check(c.nextDownbeatFrame(0, kSR48k) == 0, "next downbeat at anchor = anchor");
  // Mid-bar → next bar boundary.
  check(c.nextDownbeatFrame(48000, kSR48k) == 96000,
        "next downbeat from mid-bar");
  // Just before a downbeat.
  check(c.nextDownbeatFrame(95999, kSR48k) == 96000,
        "next downbeat one frame before");
  // Exactly on a downbeat → that frame.
  check(c.nextDownbeatFrame(96000, kSR48k) == 96000,
        "next downbeat at downbeat = that frame");
  // One frame past → next bar.
  check(c.nextDownbeatFrame(96001, kSR48k) == 192000,
        "next downbeat one frame past");
  std::puts("[PASS] nextDownbeatFrame math");
}

void test_anchor_nonzero_downbeat() {
  ae::LocalClock c;
  c.setTempo(120.0, 4, /*anchor=*/1000);
  check(c.nextDownbeatFrame(1000, kSR48k) == 1000,
        "next downbeat = anchor itself");
  check(c.nextDownbeatFrame(50000, kSR48k) == 97000,
        "next downbeat after anchor + 49000");
  std::puts("[PASS] nextDownbeatFrame respects anchor");
}

void test_frame_for_beat() {
  ae::LocalClock c;
  // 120 BPM @ 48 kHz ⇒ 24000 frames per beat. Anchor at 1000.
  c.setTempo(120.0, 4, /*anchor=*/1000);
  check(c.frameForBeat(0, kSR48k) == 1000, "beat 0 starts at anchor");
  check(c.frameForBeat(1, kSR48k) == 25000, "beat 1 starts one beat past");
  check(c.frameForBeat(4, kSR48k) == 97000, "beat 4 starts at next bar");
  check(c.frameForBeat(100, kSR48k) == 2401000, "beat 100 math");
  std::puts("[PASS] frameForBeat math");
}

void test_realistic_tempo() {
  // Loop length: 285024 frames @ 48 kHz, 4 beats ⇒ should infer ~40.45 BPM.
  // Verify the round trip: setTempo with computed BPM, ask for boundary at
  // the predicted spot, get back the right answer.
  ae::LocalClock c;
  const double loopFrames = 285024.0;
  const std::uint32_t quantum = 4;
  const double bpm =
      (static_cast<double>(kSR48k) * 60.0 * static_cast<double>(quantum)) /
      loopFrames;
  c.setTempo(bpm, quantum, /*anchor=*/0);
  check(c.isValid(), "realistic clock valid");
  check(c.nextDownbeatFrame(0, kSR48k) == 0, "boundary at 0");
  check(c.nextDownbeatFrame(1, kSR48k) == 285024,
        "boundary at exactly one loop length away");
  check(c.nextDownbeatFrame(285024, kSR48k) == 285024,
        "second loop boundary lands at 2*loopFrames");
  std::puts("[PASS] realistic loop length round trip");
}

}  // namespace

int main() {
  test_invalid_when_unset();
  test_set_clear_round_trip();
  test_phase_at_anchor();
  test_phase_with_nonzero_anchor();
  test_beat_count();
  test_next_downbeat();
  test_anchor_nonzero_downbeat();
  test_frame_for_beat();
  test_realistic_tempo();
  std::puts("All LocalClock tests passed.");
  return 0;
}
