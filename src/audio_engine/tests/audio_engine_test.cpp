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

}  // namespace

int main() {
  test_process_advances_frame();
  test_set_base_loop_command();
  test_clear_base_loop_command();
  test_command_posted_from_another_thread();
  std::puts("All AudioEngine tests passed.");
  return 0;
}
