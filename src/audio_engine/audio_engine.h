// AudioEngine — the single authority on audio state.
//
// Phase 1 scope (this file): publishes a Snapshot of basic transport state
// from inside the audio thread, drains a command queue (no-ops for now), and
// produces an empty event stream. No DSP, no recording, no base loop yet —
// those land in later phases.
//
// Threading invariants (see docs/AUDIO_ENGINE_ARCHITECTURE.md §3):
//
//   - process() runs ONLY on the audio thread. No allocations, no locks,
//     no syscalls, no file I/O, no Dart callbacks.
//   - postCommand() is callable from any thread (typically Dart main).
//     SPSC: only ONE thread is allowed to push at a time. For Phase 1+ Dart
//     is the only producer.
//   - drainEvent() is callable from any thread (typically Dart main).
//     SPSC: only ONE thread is allowed to pop. For Phase 1+ Dart is the only
//     consumer.
//   - loadSnapshot() is callable from any thread; uses the seqlock.

#ifndef FLOWSTATE_AUDIO_ENGINE_AUDIO_ENGINE_H_
#define FLOWSTATE_AUDIO_ENGINE_AUDIO_ENGINE_H_

#include <cstdint>

#include "command.h"
#include "event.h"
#include "local_clock.h"
#include "seqlock.h"
#include "snapshot.h"
#include "spsc_queue.h"
#include "sync_source.h"

namespace flowstate {
namespace audio_engine {

class AudioEngine {
 public:
  // Inbox capacity sized for bursts; commands are user-driven so realistic
  // load is a handful per second.
  static constexpr std::size_t kInboxCapacity = 64;
  // Outbox sized larger because the audio thread can emit several events per
  // buffer in worst case (e.g. simultaneous recording stop + voice end).
  static constexpr std::size_t kOutboxCapacity = 256;
  // Max commands drained per process() call to bound audio-thread work.
  static constexpr int kMaxCommandsPerBuffer = 16;

  static AudioEngine& instance() noexcept;

  // Any thread. Returns false if the inbox is full (rare; bug if it happens).
  bool postCommand(const Command& cmd) noexcept;

  // Any thread. Returns false if no event is queued.
  bool drainEvent(Event* out) noexcept;

  // Any thread. Returns the most recently published snapshot.
  Snapshot loadSnapshot() const noexcept;

  // Audio thread only. Drains a bounded number of commands, runs Phase 1
  // bookkeeping, publishes the snapshot. Future phases extend this.
  void process(std::int64_t bufferStartFrame, std::uint32_t frameCount,
               std::uint32_t sampleRate, std::uint16_t channels) noexcept;

 private:
  AudioEngine() = default;
  ~AudioEngine() = default;
  AudioEngine(const AudioEngine&) = delete;
  AudioEngine& operator=(const AudioEngine&) = delete;

  // Audio-thread only. Dispatch on cmd.type and mutate state accordingly.
  // May emit events. Must not block, allocate, or call into Dart.
  void handleCommand(const Command& cmd) noexcept;

  // Audio-thread only. Best-effort push to the outbox; on overflow, attempts
  // to enqueue an Error event so Dart notices the drop. Never blocks.
  void emitEvent(const Event& event) noexcept;

  SPSCQueue<Command, kInboxCapacity>  mInbox;
  SPSCQueue<Event, kOutboxCapacity>   mOutbox;
  Seqlock<Snapshot>                   mSnapshot;

  // -------------------------------------------------------------------------
  // Audio-thread-only state. Only process() reads OR writes these fields.
  // No external code touches them; no synchronization needed.
  // -------------------------------------------------------------------------
  std::int64_t  mCurrentFrame{0};      // bufferEnd of most recent process()

  // Sync sources. Each implementation lives as a member here so the active
  // pointer always points at a valid object regardless of switches. Phase 2
  // ships only LocalClock; AbletonLink joins this list in Phase 4.
  LocalClock    mLocalClock;
  SyncSource*   mActiveSource{&mLocalClock};
  SyncSourceKind mActiveSourceKind{SyncSourceKind::Local};

  // Legacy base-loop fields — kept for parallel-run during 2 → 2c. Removed
  // in Phase 2c.
  std::int64_t  mBaseLoopFrames{0};
  std::int64_t  mBaseLoopStart{0};

  // Metronome control (Phase 3 implementation; the flag is wired now so
  // SetMetronome commands stop being silent no-ops).
  bool          mMetronomeEnabled{false};
  bool          mMetronomeDownbeatOnly{false};
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_AUDIO_ENGINE_H_
