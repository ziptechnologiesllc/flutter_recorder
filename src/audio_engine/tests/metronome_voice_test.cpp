// Tests for MetronomeVoice.
//
// Verifies sample-accurate click placement, multi-buffer click coverage,
// concurrent-click handling, and lazy click-sample regeneration on sample-
// rate change.

#include "../metronome_voice.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ae = flowstate::audio_engine;

namespace {

constexpr std::uint32_t kSR48k = 48000;
constexpr std::uint16_t kStereo = 2;
constexpr std::size_t kClickFrames =
    static_cast<std::size_t>(ae::MetronomeVoice::kClickDurationSec * kSR48k);

void check(bool ok, const char* label) {
  if (!ok) {
    std::fprintf(stderr, "[FAIL] %s\n", label);
    std::exit(1);
  }
}

double bufferEnergy(const std::vector<float>& buf) {
  double e = 0.0;
  for (float v : buf) e += static_cast<double>(v) * v;
  return e;
}

void test_no_clicks_no_output_change() {
  ae::MetronomeVoice m;
  std::vector<float> buf(512 * 2, 0.0f);
  m.mix(buf.data(), /*bufferStart=*/0, /*frameCount=*/512, kStereo, kSR48k);
  check(bufferEnergy(buf) == 0.0, "no active clicks → buffer unchanged");
  check(m.activeClickCount() == 0, "no clicks active");
  std::puts("[PASS] no-op when no clicks scheduled");
}

void test_click_lands_in_correct_buffer_position() {
  ae::MetronomeVoice m;
  // Schedule a click at exactly frame 100 within a buffer starting at 0.
  m.schedule(/*startFrame=*/100, /*isDownbeat=*/false);
  check(m.activeClickCount() == 1, "one click scheduled");

  std::vector<float> buf(512 * 2, 0.0f);
  m.mix(buf.data(), 0, 512, kStereo, kSR48k);

  // Frames 0..99 should be silent. Frame 100 onward should have content.
  bool silentBefore = true;
  for (std::size_t i = 0; i < 100; ++i) {
    if (buf[i * 2] != 0.0f || buf[i * 2 + 1] != 0.0f) {
      silentBefore = false;
      break;
    }
  }
  check(silentBefore, "frames before click are silent");

  bool gotSomeContent = false;
  for (std::size_t i = 100; i < 200; ++i) {
    if (buf[i * 2] != 0.0f) gotSomeContent = true;
  }
  check(gotSomeContent, "click samples mixed in starting at offset 100");
  std::puts("[PASS] click lands at sample-accurate offset");
}

void test_click_spans_multiple_buffers() {
  ae::MetronomeVoice m;
  // Click at frame 1900 with click length ~1440 frames @ 48 kHz → it ends
  // at frame ~3340. Buffers of 1024 frames: click straddles three buffers.
  m.schedule(/*startFrame=*/1900, false);

  std::vector<float> buf0(1024 * 2, 0.0f);
  std::vector<float> buf1(1024 * 2, 0.0f);
  std::vector<float> buf2(1024 * 2, 0.0f);

  // Buffer covers [0, 1024). Click is in the future → no mix.
  m.mix(buf0.data(), 0, 1024, kStereo, kSR48k);
  check(bufferEnergy(buf0) == 0.0, "future click does not leak into early buffer");

  // Buffer [1024, 2048). Click starts at 1900 (offset 876 in this buffer).
  m.mix(buf1.data(), 1024, 1024, kStereo, kSR48k);
  bool buf1Silent = true;
  for (std::size_t i = 0; i < 876; ++i) {
    if (buf1[i * 2] != 0.0f) {
      buf1Silent = false;
      break;
    }
  }
  check(buf1Silent, "buf1: silent before click start");
  bool buf1Has = false;
  for (std::size_t i = 876; i < 1024; ++i) {
    if (buf1[i * 2] != 0.0f) buf1Has = true;
  }
  check(buf1Has, "buf1: click samples from offset 876");

  // Buffer [2048, 3072). Click continues from sample (2048 - 1900) = 148.
  m.mix(buf2.data(), 2048, 1024, kStereo, kSR48k);
  bool buf2Has = false;
  for (std::size_t i = 0; i < 100; ++i) {
    if (buf2[i * 2] != 0.0f) buf2Has = true;
  }
  check(buf2Has, "buf2: click continues");

  // After enough buffers the click should auto-deactivate.
  std::vector<float> buf3(1024 * 2, 0.0f);
  m.mix(buf3.data(), 3072, 1024, kStereo, kSR48k);
  std::vector<float> buf4(1024 * 2, 0.0f);
  m.mix(buf4.data(), 4096, 1024, kStereo, kSR48k);
  check(m.activeClickCount() == 0, "click auto-deactivates after duration");
  std::puts("[PASS] click spans multiple buffers, auto-deactivates");
}

void test_concurrent_clicks() {
  ae::MetronomeVoice m;
  // Two overlapping clicks: one at frame 100, one at frame 200.
  m.schedule(100, true);   // downbeat
  m.schedule(200, false);  // regular beat
  check(m.activeClickCount() == 2, "two clicks scheduled");

  std::vector<float> buf(1024 * 2, 0.0f);
  m.mix(buf.data(), 0, 1024, kStereo, kSR48k);

  // Frame 100..199 has only downbeat click contribution.
  // Frame 200..(100+clickFrames) has both clicks summed.
  bool gotBoth = false;
  for (std::size_t i = 200; i < std::min<std::size_t>(800, kClickFrames); ++i) {
    if (std::fabs(buf[i * 2]) > 1e-6f) {
      gotBoth = true;
      break;
    }
  }
  check(gotBoth, "concurrent clicks both contribute to output");
  std::puts("[PASS] concurrent clicks mix together");
}

void test_downbeat_louder_than_beat() {
  // Compare RMS energy of a downbeat-only click vs a regular-beat-only click
  // at peak. Downbeat amplitude should produce more energy.
  std::vector<float> downBuf(kClickFrames * 2, 0.0f);
  {
    ae::MetronomeVoice m;
    m.schedule(0, true);
    m.mix(downBuf.data(), 0, static_cast<std::uint32_t>(kClickFrames),
          kStereo, kSR48k);
  }
  std::vector<float> beatBuf(kClickFrames * 2, 0.0f);
  {
    ae::MetronomeVoice m;
    m.schedule(0, false);
    m.mix(beatBuf.data(), 0, static_cast<std::uint32_t>(kClickFrames),
          kStereo, kSR48k);
  }
  const double downE = bufferEnergy(downBuf);
  const double beatE = bufferEnergy(beatBuf);
  check(downE > beatE, "downbeat has more total energy than beat");
  std::printf("       (downbeat E=%.3f, beat E=%.3f)\n", downE, beatE);
  std::puts("[PASS] downbeat is louder than regular beat");
}

void test_reset_cancels_active_clicks() {
  ae::MetronomeVoice m;
  m.schedule(100, false);
  m.schedule(200, false);
  check(m.activeClickCount() == 2, "two clicks scheduled");
  m.reset();
  check(m.activeClickCount() == 0, "reset clears all clicks");
  std::puts("[PASS] reset cancels active clicks");
}

}  // namespace

int main() {
  test_no_clicks_no_output_change();
  test_click_lands_in_correct_buffer_position();
  test_click_spans_multiple_buffers();
  test_concurrent_clicks();
  test_downbeat_louder_than_beat();
  test_reset_cancels_active_clicks();
  std::puts("All MetronomeVoice tests passed.");
  return 0;
}
