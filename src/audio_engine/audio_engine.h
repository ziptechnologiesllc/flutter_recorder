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

#include <atomic>
#include <cstdint>

#include "command.h"
#include "event.h"
#include "local_clock.h"
#include "metronome_voice.h"
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
  // Worker inbox: a second SPSC queue dedicated to messages from the
  // C++ worker thread (tempo inference, key inference, etc.). Keeping
  // it separate preserves single-producer semantics on each queue.
  static constexpr std::size_t kWorkerInboxCapacity = 16;
  // Outbox sized larger because the audio thread can emit several events per
  // buffer in worst case (e.g. simultaneous recording stop + voice end).
  static constexpr std::size_t kOutboxCapacity = 256;
  // Max commands drained per process() call to bound audio-thread work.
  static constexpr int kMaxCommandsPerBuffer = 16;

  static AudioEngine& instance() noexcept;

  // Dart main isolate is the sole producer. Returns false if full.
  bool postCommand(const Command& cmd) noexcept;

  // Worker thread is the sole producer of this queue. Used for tempo /
  // key inference results posted back to the audio thread. Returns false
  // if the worker inbox is full (would indicate inference is firing far
  // faster than process() drains it — pathological).
  bool postWorkerCommand(const Command& cmd) noexcept;

  // Any thread. Returns false if no event is queued.
  bool drainEvent(Event* out) noexcept;

  // Any thread. Returns the most recently published snapshot.
  Snapshot loadSnapshot() const noexcept;

  // Audio thread only. Drains a bounded number of commands, runs Phase 1
  // bookkeeping, publishes the snapshot. Future phases extend this.
  void process(std::int64_t bufferStartFrame, std::uint32_t frameCount,
               std::uint32_t sampleRate, std::uint16_t channels) noexcept;

  // Audio thread only. Mixes any active metronome clicks into the given
  // float output buffer. Must be called AFTER process() in the same
  // data_callback so that beat scheduling has already happened. The buffer
  // is the same one SoLoud writes to; we additively mix clicks on top of
  // SoLoud's output.
  //
  // bufferStartFrame must match what was passed to process(); frameCount
  // and channels match the output buffer layout.
  void mixMetronomeIntoOutput(float* output, std::int64_t bufferStartFrame,
                               std::uint32_t frameCount,
                               std::uint16_t channels,
                               std::uint32_t sampleRate) noexcept;

  // Phase 2c.4 — bridge for code that runs on the audio thread but lives
  // outside AudioEngine (specifically, NativeScheduler::processEvents,
  // which runs immediately before AudioEngine::process() in the same
  // data_callback). Pushes the event onto the outbox just like
  // process()'s internal emitEvent does.
  //
  // Single-producer invariant: callers must be on the audio thread, and
  // there must only be one such call site per data_callback invocation.
  // NativeScheduler::processEvents respects this — it runs sequentially
  // with AudioEngine::process(), never concurrent.
  void emitFromAudioThread(const Event& event) noexcept;

  // Lock-free, callable from any thread. Returns the bufferEnd frame of the
  // most-recent process() call. Phase 2a canonicalises this as the single
  // source of truth for "what frame is it now"; NativeScheduler and Dart
  // consumers route through here.
  std::int64_t getCurrentFrame() const noexcept {
    return mPublishedFrame.load(std::memory_order_acquire);
  }

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

  SPSCQueue<Command, kInboxCapacity>       mInbox;        // Dart   → audio
  SPSCQueue<Command, kWorkerInboxCapacity> mWorkerInbox;  // Worker → audio
  SPSCQueue<Event, kOutboxCapacity>        mOutbox;
  Seqlock<Snapshot>                        mSnapshot;

  // -------------------------------------------------------------------------
  // Audio-thread-only state. Only process() reads OR writes these fields.
  // No external code touches them; no synchronization needed.
  // -------------------------------------------------------------------------
  std::int64_t  mCurrentFrame{0};      // bufferEnd of most recent process()

  // Phase 2a: published mirror of mCurrentFrame for cross-component reads
  // (NativeScheduler, Dart via FFI, anyone needing "what frame is it"). The
  // audio thread is the sole writer, so this is a single-producer atomic;
  // a relaxed store + acquire-load contract gives wait-free read access.
  // [getCurrentFrame] is the public way in.
  std::atomic<std::int64_t> mPublishedFrame{0};

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

  // Metronome control. Phase 3c v1 emits BeatFired/DownbeatFired events at
  // sample-accurate frames; UI / Dart-side click playback consumes them.
  // Audio-thread audible click mixing arrives in Phase 3c v2.
  bool          mMetronomeEnabled{false};
  bool          mMetronomeDownbeatOnly{false};

  // Last beat number whose event we've already emitted. Sentinel value
  // (UINT32_MAX) means "uninitialized; first observation should anchor
  // rather than emit a flurry of historical beats." Reset to sentinel
  // whenever tempo or sync source changes.
  static constexpr std::uint32_t kBeatCounterUninit =
      static_cast<std::uint32_t>(-1);
  std::uint32_t mLastEmittedBeat{kBeatCounterUninit};

  // Sample-accurate click voice. Audio-thread only.
  MetronomeVoice mMetronome;

  // ── Phase 2c: pending mute/unmute/gain queue ─────────────────────────────
  // Tap-to-mute (and stab/free, and queued gain change) lands here as a
  // PendingEntry whose fire frame is resolved against the active SyncSource.
  // Each process() call scans the array and emits VoiceMuted/VoiceUnmuted/
  // GainChanged for entries whose fire frame falls within the buffer window.
  //
  // Per-track invariant: at most one pending entry per track index at a time.
  // A new Queue* for an already-queued track replaces the prior entry.
  enum class PendingAction : std::uint8_t {
    None = 0,
    Mute,
    Unmute,
    SetGain,
    // Phase 1 native transport. Fire-frame logic identical to Mute/Unmute;
    // the difference is that when these fire, the audio thread calls the
    // bridged SoLoud setter directly (setPause/stop) instead of relying on
    // Dart to apply the change.
    Pause,
    Unpause,
    Stop,
  };

  struct PendingEntry {
    PendingAction action{PendingAction::None};
    std::uint32_t trackIndex{0};
    std::int64_t  fireFrame{0};
    float         value{0.0f};  // velocity (Unmute) or gain (SetGain); ignored for Mute
  };

  // 32 entries covers any plausible burst (one tap per track per beat across
  // a 16-track session would still leave headroom). Bounded so the audio
  // thread's scan is constant-time on a tight upper limit.
  static constexpr std::size_t kMaxPendingQueueEntries = 32;
  PendingEntry mPendingQueue[kMaxPendingQueueEntries];
  std::size_t  mPendingQueueCount{0};

  // ── Phase 1 native transport: trackIndex → SoLoud handle table ─────────
  // Dart announces the SoLoud voice handle each loop player creates (via
  // RegisterTrackHandle command); the audio thread reads this table when a
  // pending Pause/Unpause/Stop/Mute/Unmute/SetGain entry fires, then calls
  // the bridged SoLoud setter directly. soloudHandle == 0 marks an empty
  // slot (SoLoud reserves 0 as "invalid handle"). Audio-thread-only — all
  // mutation flows through handleCommand which already runs on the audio
  // thread, no synchronisation needed.
  struct TrackHandle {
    std::uint32_t trackIndex{0};
    std::uint32_t soloudHandle{0};  // 0 = empty slot
  };
  static constexpr std::size_t kMaxTrackHandles = 64;
  TrackHandle mTrackHandles[kMaxTrackHandles]{};

  // Linear scan; at 64 entries this is ~200 ns on a modern CPU — trivial
  // compared to a buffer's mix budget. Returns 0 if the trackIndex is not
  // registered (caller must skip the bridged setter call in that case).
  std::uint32_t lookupSoloudHandle(std::uint32_t trackIndex) const noexcept;

  // Insert / replace / remove the table entry for `trackIndex`. Returns
  // false if a registration ran into a full table (no empty slot AND no
  // pre-existing entry for the same trackIndex); the audio thread emits an
  // Error event so Dart can surface the leak.
  bool registerTrackHandle(std::uint32_t trackIndex,
                           std::uint32_t soloudHandle) noexcept;
  bool unregisterTrackHandle(std::uint32_t trackIndex) noexcept;

  // Resolve `quantizeSixteenths` against the active SyncSource. Returns the
  // first sample-accurate frame >= `currentFrame` that lies on the requested
  // grid. quantizeSixteenths==0 returns `currentFrame` itself (immediate /
  // stab mode). Returns currentFrame as a safe fallback when no SyncSource
  // is valid (tempo not set yet) — caller has already chosen the boundary.
  std::int64_t resolveQuantizedFireFrame(std::int64_t currentFrame,
                                          std::uint32_t quantizeSixteenths,
                                          std::uint32_t sampleRate) const noexcept;

  // Add or replace the pending entry for `entry.trackIndex`. Returns false
  // if the queue is at capacity AND the track has no existing entry (the
  // overwrite case always succeeds). Audio-thread only.
  bool upsertPendingEntry(const PendingEntry& entry) noexcept;

  // Remove the pending entry (if any) for `trackIndex`. Returns true if an
  // entry was removed.
  bool removePendingEntry(std::uint32_t trackIndex) noexcept;

  // Scan the pending queue and emit events for entries whose fireFrame <=
  // bufferEndFrame. Removes fired entries from the queue. Audio-thread only.
  void firePendingThroughFrame(std::int64_t bufferEndFrame) noexcept;

  // Helper called from process(); not a public API.
  void checkAndEmitBeats(std::int64_t bufferEndFrame,
                          std::uint32_t sampleRate) noexcept;
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_AUDIO_ENGINE_H_
