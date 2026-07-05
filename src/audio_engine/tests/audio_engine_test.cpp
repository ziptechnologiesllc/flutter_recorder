// Simulation tests for AudioEngine::process().
//
// Drives the engine with synthetic buffers and asserts on observable
// behavior (published snapshot, drained events). No audio hardware
// required — this is the pure-function harness called out in the
// architecture doc §9.2.

#include "../audio_engine.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace ae = flowstate::audio_engine;

namespace {

void check(bool ok, const char* label) {
  if (!ok) {
    std::fprintf(stderr, "[FAIL] %s\n", label);
    std::exit(1);
  }
}

// Reset shared engine state between tests. Because the engine is a Meyers
// singleton, the same instance persists across tests in this binary. We
// scrub it by posting commands or via the test-only entry below.
void resetEngine(ae::AudioEngine& e) {
  ae::Command clear{};
  clear.type = ae::Command::Type::ClearBaseLoop;
  e.postCommand(clear);
  // Advance one buffer so the command takes effect.
  e.process(/*bufferStartFrame=*/0, /*frameCount=*/256,
            /*sampleRate=*/48000, /*channels=*/2);
  // Drain any leftover events from prior tests.
  ae::Event drained;
  int budget = 1024;
  while (budget-- > 0 && e.drainEvent(&drained)) {}
}

void test_process_advances_frame() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Baseline snapshot starting frame.
  e.process(0, 256, 48000, 2);
  auto s0 = e.loadSnapshot();
  check(s0.currentFrame == 256, "after one 256-frame buffer at 0, frame=256");
  check(s0.sampleRate == 48000, "sample rate published");
  check(s0.channels == 2, "channel count published");
  check(s0.bufferSize == 256, "buffer size published");

  // Next buffer.
  e.process(256, 256, 48000, 2);
  auto s1 = e.loadSnapshot();
  check(s1.currentFrame == 512, "after second buffer, frame=512");

  std::puts("[PASS] process advances current frame");
}

void test_set_base_loop_command() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Post SetBaseLoop with explicit anchor and length.
  ae::Command cmd{};
  cmd.type = ae::Command::Type::SetBaseLoop;
  cmd.lengthFrames = 96000;      // 2 seconds @ 48 kHz
  cmd.targetFrame  = 12345;       // arbitrary anchor
  check(e.postCommand(cmd), "post SetBaseLoop succeeded");

  // Process one buffer — the command should be consumed and applied.
  e.process(0, 256, 48000, 2);

  auto s = e.loadSnapshot();
  check(s.baseLoopFrames == 96000, "baseLoopFrames published");
  check(s.baseLoopStart  == 12345, "baseLoopStart published");

  // Verify a BaseLoopSet event was emitted with the right payload.
  ae::Event ev;
  check(e.drainEvent(&ev), "BaseLoopSet event present");
  check(ev.type == ae::Event::Type::BaseLoopSet, "event type is BaseLoopSet");
  check(ev.framesProcessed == 96000, "event payload carries loop length");
  check(ev.frame == 256, "event frame is the buffer end where it fired");

  // No further events expected.
  check(!e.drainEvent(&ev), "no extra events");

  std::puts("[PASS] SetBaseLoop updates snapshot + emits event");
}

void test_clear_base_loop_command() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Establish a base loop first.
  ae::Command set{};
  set.type = ae::Command::Type::SetBaseLoop;
  set.lengthFrames = 48000;
  set.targetFrame  = 100;
  e.postCommand(set);
  e.process(0, 256, 48000, 2);

  // Drain the BaseLoopSet event so we can isolate the clear behavior.
  ae::Event dropped;
  while (e.drainEvent(&dropped)) {}

  // Now clear.
  ae::Command clear{};
  clear.type = ae::Command::Type::ClearBaseLoop;
  e.postCommand(clear);
  e.process(256, 256, 48000, 2);

  auto s = e.loadSnapshot();
  check(s.baseLoopFrames == 0, "cleared baseLoopFrames");
  check(s.baseLoopStart  == 0, "cleared baseLoopStart");

  ae::Event ev;
  check(e.drainEvent(&ev), "BaseLoopCleared event present");
  check(ev.type == ae::Event::Type::BaseLoopCleared, "event type is cleared");

  std::puts("[PASS] ClearBaseLoop resets state + emits event");
}

void test_command_posted_from_another_thread() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Background producer pushes commands while the "audio thread" runs
  // process() on the main thread. Verifies SPSC correctness with the
  // realistic two-thread topology.
  std::atomic<bool> done{false};
  std::thread producer([&] {
    ae::Command c{};
    c.type = ae::Command::Type::SetBaseLoop;
    c.lengthFrames = 24000;
    c.targetFrame  = 999;
    while (!e.postCommand(c)) {
      std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  });

  // Spin process() until the command is consumed and reflected in the
  // snapshot, or until we time out.
  for (int i = 0; i < 1000; ++i) {
    e.process(static_cast<std::int64_t>(i) * 256, 256, 48000, 2);
    if (e.loadSnapshot().baseLoopFrames == 24000) break;
    std::this_thread::yield();
  }

  producer.join();
  check(done.load(), "producer completed");
  check(e.loadSnapshot().baseLoopFrames == 24000,
        "cross-thread SetBaseLoop reflected in snapshot");

  std::puts("[PASS] cross-thread command delivery");
}

// ── Phase 2c.1: mute/unmute/gain queue ────────────────────────────────────

void test_queue_mute_stab_fires_immediately() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Establish tempo so the engine is in a quantizable state. quantize=0 in
  // the command should still bypass it and fire on the same buffer.
  ae::Command setTempo{};
  setTempo.type = ae::Command::Type::SetTempo;
  const double bpm = 120.0;
  std::int64_t bpmBits;
  std::memcpy(&bpmBits, &bpm, sizeof(double));
  setTempo.lengthFrames = bpmBits;
  setTempo.targetFrame  = 0;
  setTempo.flags        = 4;
  e.postCommand(setTempo);
  e.process(0, 256, 48000, 2);
  ae::Event ev;
  while (e.drainEvent(&ev)) {}

  // Queue a stab unmute on track 7 with velocity 0.8, quantize=0 (free).
  ae::Command q{};
  q.type      = ae::Command::Type::QueueUnmute;
  q.id        = 7;
  q.flags     = 0;  // immediate
  std::uint32_t velBits;
  const float velocity = 0.8f;
  std::memcpy(&velBits, &velocity, sizeof(float));
  q.soundHash = velBits;
  q.targetFrame = 0;
  check(e.postCommand(q), "stab unmute posted");

  e.process(256, 256, 48000, 2);

  bool fired = false;
  while (e.drainEvent(&ev)) {
    if (ev.type == ae::Event::Type::VoiceUnmuted && ev.id == 7) {
      // Velocity round-trip via float bit-cast in soundHash.
      float roundtrip;
      std::memcpy(&roundtrip, &ev.soundHash, sizeof(float));
      check(roundtrip > 0.79f && roundtrip < 0.81f,
            "velocity round-trips ~0.8");
      // Fire frame should be on the same buffer (between 256 and 512).
      check(ev.frame >= 256 && ev.frame <= 512, "fired on current buffer");
      fired = true;
    }
  }
  check(fired, "VoiceUnmuted event emitted for stab mode");
  std::puts("[PASS] QueueUnmute stab fires immediately");
}

void test_queue_mute_quantized_fires_at_bar_boundary() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // 120 BPM, q=4 → bar at 96000 frames.
  ae::Command setTempo{};
  setTempo.type = ae::Command::Type::SetTempo;
  const double bpm = 120.0;
  std::int64_t bpmBits;
  std::memcpy(&bpmBits, &bpm, sizeof(double));
  setTempo.lengthFrames = bpmBits;
  setTempo.targetFrame  = 0;
  setTempo.flags        = 4;
  e.postCommand(setTempo);
  e.process(0, 256, 48000, 2);
  ae::Event ev;
  while (e.drainEvent(&ev)) {}

  // Queue a mute on track 2 with quantize=16 (one bar). Posted at frame
  // 256, so fire frame should be frame 96000 (first bar boundary).
  ae::Command q{};
  q.type   = ae::Command::Type::QueueMute;
  q.id     = 2;
  q.flags  = 16;
  q.targetFrame = 0;
  e.postCommand(q);

  // Step buffers until well past the expected fire frame.
  e.process(256, 256, 48000, 2);
  // Nothing should fire yet.
  while (e.drainEvent(&ev)) {
    check(ev.type != ae::Event::Type::VoiceMuted,
          "no premature fire before bar boundary");
  }

  // Advance to frame 96100 — should fire on this buffer.
  e.process(512, 96100 - 512, 48000, 2);
  bool fired = false;
  while (e.drainEvent(&ev)) {
    if (ev.type == ae::Event::Type::VoiceMuted && ev.id == 2) {
      check(ev.frame == 96000, "fire frame is exactly the bar boundary");
      fired = true;
    }
  }
  check(fired, "VoiceMuted fired at bar boundary");
  std::puts("[PASS] QueueMute fires at bar boundary");
}

void test_upsert_replaces_pending_entry() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Tempo so quantize math works.
  ae::Command setTempo{};
  setTempo.type = ae::Command::Type::SetTempo;
  const double bpm = 120.0;
  std::int64_t bpmBits;
  std::memcpy(&bpmBits, &bpm, sizeof(double));
  setTempo.lengthFrames = bpmBits;
  setTempo.targetFrame  = 0;
  setTempo.flags        = 4;
  e.postCommand(setTempo);
  e.process(0, 256, 48000, 2);
  ae::Event ev;
  while (e.drainEvent(&ev)) {}

  // First queue mute on track 5 at bar boundary (frame 96000).
  ae::Command qMute{};
  qMute.type  = ae::Command::Type::QueueMute;
  qMute.id    = 5;
  qMute.flags = 16;
  e.postCommand(qMute);

  // Then immediately queue unmute on the same track, also at bar boundary.
  ae::Command qUn{};
  qUn.type  = ae::Command::Type::QueueUnmute;
  qUn.id    = 5;
  qUn.flags = 16;
  const float v = 1.0f;
  std::uint32_t vb;
  std::memcpy(&vb, &v, sizeof(float));
  qUn.soundHash = vb;
  e.postCommand(qUn);

  // First buffer: both commands drain with currentFrame=512, both resolve
  // to bar boundary at frame 96000. Upsert keeps only the latest (Unmute).
  e.process(256, 256, 48000, 2);
  while (e.drainEvent(&ev)) {}

  // Advance past the bar boundary so the pending Unmute fires.
  e.process(512, 96100 - 512, 48000, 2);
  int muteCount = 0;
  int unmuteCount = 0;
  while (e.drainEvent(&ev)) {
    if (ev.type == ae::Event::Type::VoiceMuted && ev.id == 5) ++muteCount;
    if (ev.type == ae::Event::Type::VoiceUnmuted && ev.id == 5) ++unmuteCount;
  }
  check(muteCount == 0, "upsert dropped the prior Mute entry");
  check(unmuteCount == 1, "only the latest Unmute fired");
  std::puts("[PASS] upsert replaces pending entry per track");
}

void test_cancel_pending_queue() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  ae::Command setTempo{};
  setTempo.type = ae::Command::Type::SetTempo;
  const double bpm = 120.0;
  std::int64_t bpmBits;
  std::memcpy(&bpmBits, &bpm, sizeof(double));
  setTempo.lengthFrames = bpmBits;
  setTempo.targetFrame  = 0;
  setTempo.flags        = 4;
  e.postCommand(setTempo);
  e.process(0, 256, 48000, 2);
  ae::Event ev;
  while (e.drainEvent(&ev)) {}

  // Queue a mute at bar boundary, process once so the entry is in the
  // queue with fireFrame=96000, then cancel before advancing to fire.
  ae::Command qMute{};
  qMute.type  = ae::Command::Type::QueueMute;
  qMute.id    = 11;
  qMute.flags = 16;
  e.postCommand(qMute);

  e.process(256, 256, 48000, 2);  // drains qMute, entry now pending at 96000
  while (e.drainEvent(&ev)) {}

  ae::Command qCancel{};
  qCancel.type = ae::Command::Type::CancelPendingQueue;
  qCancel.id   = 11;
  e.postCommand(qCancel);

  // Now advance past the bar boundary — without the cancel, this would
  // have fired VoiceMuted; with the cancel, nothing fires.
  e.process(512, 96100 - 512, 48000, 2);
  while (e.drainEvent(&ev)) {
    check(!(ev.type == ae::Event::Type::VoiceMuted && ev.id == 11),
          "cancelled entry must not fire");
  }
  std::puts("[PASS] CancelPendingQueue suppresses fire");
}

void test_gain_change_round_trips_float() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Free-mode (no tempo). quantize=0 with currentFrame fallback.
  ae::Command q{};
  q.type   = ae::Command::Type::SetTrackGain;
  q.id     = 3;
  q.flags  = 0;
  const float gain = 0.42f;
  std::uint32_t gb;
  std::memcpy(&gb, &gain, sizeof(float));
  q.soundHash = gb;
  e.postCommand(q);

  e.process(0, 256, 48000, 2);
  ae::Event ev;
  bool fired = false;
  while (e.drainEvent(&ev)) {
    if (ev.type == ae::Event::Type::GainChanged && ev.id == 3) {
      float roundtrip;
      std::memcpy(&roundtrip, &ev.soundHash, sizeof(float));
      check(roundtrip > 0.41f && roundtrip < 0.43f,
            "gain round-trips ~0.42");
      fired = true;
    }
  }
  check(fired, "GainChanged event emitted");
  std::puts("[PASS] GainChanged round-trips float payload");
}

}  // namespace

void test_metronome_emits_beat_events() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Set a tempo: 120 BPM, q=4, anchor at 0. Frames per beat at 48 kHz = 24000.
  ae::Command setTempo{};
  setTempo.type = ae::Command::Type::SetTempo;
  // Pack tempoBPM into lengthFrames via bit-cast (matches Dart side).
  const double bpm = 120.0;
  std::int64_t bpmBits;
  std::memcpy(&bpmBits, &bpm, sizeof(double));
  setTempo.lengthFrames = bpmBits;
  setTempo.targetFrame  = 0;
  setTempo.flags        = 4;  // quantum
  e.postCommand(setTempo);

  // Enable metronome.
  ae::Command setMetro{};
  setMetro.type  = ae::Command::Type::SetMetronome;
  setMetro.flags = 0x1;  // enabled, all beats (not downbeat-only)
  e.postCommand(setMetro);

  // First process(): drains both commands, but no beats emit yet — the
  // metronome anchors on first observation and skips historical beats.
  e.process(0, 256, 48000, 2);
  ae::Event ev;
  // Drain any events from the command handlers (TempoSet etc.).
  while (e.drainEvent(&ev)) {}

  // Advance well past beat 1 (which is at frame 24000).
  e.process(256, 24320, 48000, 2);

  // We expect one BeatFired event for beat 1.
  bool gotBeat1 = false;
  while (e.drainEvent(&ev)) {
    if (ev.type == ae::Event::Type::BeatFired ||
        ev.type == ae::Event::Type::DownbeatFired) {
      check(ev.id == 1, "first emitted beat is beat 1");
      check(ev.frame == 24000, "beat 1 frame is at 24000");
      // beat 1 is not a downbeat (q=4 → downbeats at 0, 4, 8, ...).
      check(ev.type == ae::Event::Type::BeatFired, "beat 1 is non-downbeat");
      gotBeat1 = true;
    }
  }
  check(gotBeat1, "received BeatFired for beat 1");

  // Advance to beat 4 (= a downbeat). Frame 96000.
  e.process(24576, 96000 - 24576 + 100, 48000, 2);
  bool sawDownbeat = false;
  bool sawBeat4Downbeat = false;
  while (e.drainEvent(&ev)) {
    if (ev.type == ae::Event::Type::DownbeatFired) {
      sawDownbeat = true;
      if (ev.id == 4) {
        check(ev.frame == 96000, "downbeat 4 at 96000");
        sawBeat4Downbeat = true;
      }
    }
  }
  check(sawDownbeat, "got at least one downbeat");
  check(sawBeat4Downbeat, "got DownbeatFired for beat 4");

  std::puts("[PASS] metronome emits beat + downbeat events");
}

void test_metronome_downbeat_only() {
  auto& e = ae::AudioEngine::instance();
  resetEngine(e);

  // Tempo 120 BPM, q=4, anchor 0.
  ae::Command setTempo{};
  setTempo.type = ae::Command::Type::SetTempo;
  const double bpm = 120.0;
  std::int64_t bpmBits;
  std::memcpy(&bpmBits, &bpm, sizeof(double));
  setTempo.lengthFrames = bpmBits;
  setTempo.targetFrame  = 0;
  setTempo.flags        = 4;
  e.postCommand(setTempo);

  // Enable metronome with downbeat-only flag.
  ae::Command setMetro{};
  setMetro.type  = ae::Command::Type::SetMetronome;
  setMetro.flags = 0x1 | 0x2;
  e.postCommand(setMetro);

  e.process(0, 256, 48000, 2);
  ae::Event ev;
  while (e.drainEvent(&ev)) {}

  // Run through ~2 bars: from frame 256 to frame 200000.
  e.process(256, 200000 - 256, 48000, 2);

  int downbeats = 0;
  int beats = 0;
  while (e.drainEvent(&ev)) {
    if (ev.type == ae::Event::Type::DownbeatFired) downbeats++;
    if (ev.type == ae::Event::Type::BeatFired) beats++;
  }
  check(downbeats >= 2, "got at least 2 downbeats in 2 bars");
  check(beats == 0, "downbeat-only mode suppresses non-downbeat events");
  std::printf("       (downbeats=%d, beats=%d)\n", downbeats, beats);
  std::puts("[PASS] downbeat-only mode");
}

int main() {
  test_process_advances_frame();
  test_set_base_loop_command();
  test_clear_base_loop_command();
  test_command_posted_from_another_thread();
  test_metronome_emits_beat_events();
  test_metronome_downbeat_only();
  test_queue_mute_stab_fires_immediately();
  test_queue_mute_quantized_fires_at_bar_boundary();
  test_upsert_replaces_pending_entry();
  test_cancel_pending_queue();
  test_gain_change_round_trips_float();
  std::puts("All AudioEngine tests passed.");
  return 0;
}
