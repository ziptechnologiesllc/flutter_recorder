#include "audio_engine.h"

#include <cstring>

#include "inference.h"

namespace flowstate {
namespace audio_engine {

namespace {

// Type-pun helpers for SetTempo command payloads. Dart and C++ must agree
// that we bit-cast a double into the int64_t `lengthFrames` slot.
inline double unpackTempoFromLengthFrames(std::int64_t lengthFrames) noexcept {
  double out;
  std::memcpy(&out, &lengthFrames, sizeof(double));
  return out;
}

inline std::int64_t framesPerBarOrZero(double bpm, std::uint32_t quantum,
                                        std::uint32_t sampleRate) noexcept {
  if (bpm <= 0.0 || quantum == 0 || sampleRate == 0) return 0;
  return static_cast<std::int64_t>(
      (static_cast<double>(sampleRate) * 60.0 *
       static_cast<double>(quantum)) /
      bpm);
}

}  // namespace

AudioEngine& AudioEngine::instance() noexcept {
  // Meyers singleton: a single instance across all translation units, lazily
  // constructed on first use. Thread-safe initialization is guaranteed by
  // C++11.
  static AudioEngine sInstance;
  return sInstance;
}

bool AudioEngine::postCommand(const Command& cmd) noexcept {
  return mInbox.push(cmd);
}

bool AudioEngine::drainEvent(Event* out) noexcept {
  return mOutbox.pop(out);
}

Snapshot AudioEngine::loadSnapshot() const noexcept {
  return mSnapshot.load();
}

void AudioEngine::process(std::int64_t bufferStartFrame,
                          std::uint32_t frameCount,
                          std::uint32_t sampleRate,
                          std::uint16_t channels) noexcept {
  // 1. Advance our transport. mCurrentFrame is the END of this buffer once
  //    process() returns — matches the convention used by NativeScheduler
  //    so the two systems agree on "global frame" during the parallel-run
  //    period of the refactor.
  mCurrentFrame = bufferStartFrame + static_cast<std::int64_t>(frameCount);

  // 2. Drain a bounded number of commands. Each one mutates audio-thread
  //    state and may emit events. Bounded to keep the audio callback within
  //    a predictable budget regardless of inbox burst size.
  Command cmd;
  for (int i = 0; i < kMaxCommandsPerBuffer; ++i) {
    if (!mInbox.pop(&cmd)) break;
    handleCommand(cmd);
  }

  // 3. Compose and publish the snapshot. Query active SyncSource for the
  //    musical state — wait-free reads of LocalClock atomics.
  Snapshot s{};
  s.currentFrame   = mCurrentFrame;
  s.sampleRate     = sampleRate;
  s.channels       = channels;
  s.bufferSize     = static_cast<std::uint16_t>(frameCount);
  s.syncSourceKind = static_cast<std::uint8_t>(mActiveSourceKind);
  s.sigNumerator   = 4;   // 4/4 default until Phase 11+
  s.sigDenominator = 4;

  if (mActiveSource != nullptr && mActiveSource->isValid()) {
    s.tempoBPM          = mActiveSource->tempoBPM();
    s.quantum           = mActiveSource->quantum();
    s.phaseInBar        = static_cast<float>(
        mActiveSource->phaseInBar(mCurrentFrame, sampleRate));
    s.currentBeat       = mActiveSource->beatAtFrame(mCurrentFrame, sampleRate);
    s.nextDownbeatFrame =
        mActiveSource->nextDownbeatFrame(mCurrentFrame, sampleRate);
  }

  // Legacy base-loop fields (parallel-run during 2 → 2c).
  s.baseLoopStart  = mBaseLoopStart;
  s.baseLoopFrames = mBaseLoopFrames;

  // activeRecordingMask, activeVoiceMask, level dBs filled by later phases.
  mSnapshot.store(s);
}

void AudioEngine::handleCommand(const Command& cmd) noexcept {
  switch (cmd.type) {
    case Command::Type::SetBaseLoop: {
      // Legacy path. Set the explicit base-loop fields AND infer (BPM,
      // quantum) from the loop length so callers reading musical state from
      // the snapshot see plausible values without a separate SetTempo. The
      // inference uses length only (no audio access here); a follow-up
      // worker-thread pass with audio analysis posts a refined SetTempo if
      // it disagrees.
      mBaseLoopFrames = cmd.lengthFrames;
      mBaseLoopStart  = cmd.targetFrame;
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      if (sr > 0 && cmd.lengthFrames > 0) {
        const TempoInference t =
            inferTempoFromLength(cmd.lengthFrames, sr);
        if (t.quantum > 0 && t.bpm > 0.0) {
          mLocalClock.setTempo(t.bpm, t.quantum, cmd.targetFrame);
        }
      }
      emitEvent(Event{
          /*type=*/Event::Type::BaseLoopSet,
          /*reserved=*/{0, 0, 0},
          /*id=*/0,
          /*frame=*/mCurrentFrame,
          /*framesProcessed=*/mBaseLoopFrames,
          /*soundHash=*/0,
          /*code=*/0,
      });
      break;
    }

    case Command::Type::ClearBaseLoop:
      mBaseLoopFrames = 0;
      mBaseLoopStart  = 0;
      mLocalClock.clear();
      emitEvent(Event{
          /*type=*/Event::Type::BaseLoopCleared,
          /*reserved=*/{0, 0, 0},
          /*id=*/0,
          /*frame=*/mCurrentFrame,
          /*framesProcessed=*/0,
          /*soundHash=*/0,
          /*code=*/0,
      });
      break;

    case Command::Type::SetTempo: {
      // Payload encoding:
      //   tempoBPM bit-cast into lengthFrames (int64 slot)
      //   quantum  in flags
      //   anchorFrame in targetFrame
      const double bpm = unpackTempoFromLengthFrames(cmd.lengthFrames);
      const std::uint32_t q = cmd.flags;
      mLocalClock.setTempo(bpm, q, cmd.targetFrame);

      // Update legacy mirror so callers still reading baseLoopFrames see
      // the equivalent value.
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      mBaseLoopFrames = framesPerBarOrZero(bpm, q, sr);
      mBaseLoopStart  = cmd.targetFrame;

      emitEvent(Event{
          /*type=*/Event::Type::TempoSet,
          /*reserved=*/{0, 0, 0},
          /*id=*/0,
          /*frame=*/mCurrentFrame,
          /*framesProcessed=*/mBaseLoopFrames,
          /*soundHash=*/0,
          /*code=*/q,
      });
      break;
    }

    case Command::Type::ClearTempo:
      mLocalClock.clear();
      mBaseLoopFrames = 0;
      mBaseLoopStart  = 0;
      emitEvent(Event{
          /*type=*/Event::Type::TempoCleared,
          /*reserved=*/{0, 0, 0},
          /*id=*/0,
          /*frame=*/mCurrentFrame,
          /*framesProcessed=*/0,
          /*soundHash=*/0,
          /*code=*/0,
      });
      break;

    case Command::Type::SetSyncSource:
      // Phase 2 has only one source. Validate the request and emit an event
      // so callers know the switch was acknowledged (or ignored).
      switch (static_cast<SyncSourceKind>(cmd.id)) {
        case SyncSourceKind::Local:
          mActiveSource     = &mLocalClock;
          mActiveSourceKind = SyncSourceKind::Local;
          emitEvent(Event{
              /*type=*/Event::Type::SyncSourceChanged,
              /*reserved=*/{0, 0, 0},
              /*id=*/static_cast<std::uint32_t>(SyncSourceKind::Local),
              /*frame=*/mCurrentFrame,
              /*framesProcessed=*/0,
              /*soundHash=*/0,
              /*code=*/0,
          });
          break;
        case SyncSourceKind::None:
        case SyncSourceKind::AbletonLink:
        case SyncSourceKind::MidiClock:
          // Not implemented in Phase 2; surface an error so Dart can show
          // a friendly "Link not connected yet" message instead of silently
          // doing nothing.
          emitEvent(Event{
              /*type=*/Event::Type::Error,
              /*reserved=*/{0, 0, 0},
              /*id=*/cmd.id,
              /*frame=*/mCurrentFrame,
              /*framesProcessed=*/0,
              /*soundHash=*/0,
              /*code=*/2,  // kSyncSourceNotImplemented
          });
          break;
      }
      break;

    case Command::Type::SetMetronome:
      mMetronomeEnabled       = (cmd.flags & 0x1u) != 0;
      mMetronomeDownbeatOnly  = (cmd.flags & 0x2u) != 0;
      // No event emitted; Dart already knows the setting it just posted.
      // Beat/Downbeat events fire from the metronome itself in Phase 3.
      break;

    case Command::Type::None:
    case Command::Type::StartRecording:
    case Command::Type::StopRecording:
    case Command::Type::StartPlayback:
    case Command::Type::StopPlayback:
    case Command::Type::SetLatencyComp:
      // Phase 2c+ will implement these. For now, drained but ignored.
      break;
  }
}

void AudioEngine::emitEvent(const Event& event) noexcept {
  // Best-effort push. If the outbox is full, drop the event and signal the
  // overflow via an Error event on the next opportunity. We never block on
  // the audio thread, and we never grow the outbox.
  if (!mOutbox.push(event)) {
    Event err{};
    err.type = Event::Type::Error;
    err.frame = mCurrentFrame;
    err.code = 1;  // kEventOutboxFull (TODO: enumerate error codes)
    // If this push also fails, give up silently — Dart will notice via
    // missing events that should have arrived.
    (void)mOutbox.push(err);
  }
}

}  // namespace audio_engine
}  // namespace flowstate
