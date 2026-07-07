#include "audio_engine.h"

#include <cstring>

#include "inference.h"
#include "../soloud_slave_bridge.h"  // g_soloudSetVolume / setPause / stop
#include "../filters/filters.h"      // mFilters -> notifyAecReferenceChanged

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

inline float unpackFloatFromSoundHash(std::uint32_t soundHash) noexcept {
  float out;
  std::memcpy(&out, &soundHash, sizeof(float));
  return out;
}

inline std::uint32_t packFloatToSoundHash(float value) noexcept {
  std::uint32_t out;
  std::memcpy(&out, &value, sizeof(float));
  return out;
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

bool AudioEngine::postWorkerCommand(const Command& cmd) noexcept {
  return mWorkerInbox.push(cmd);
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
  // Phase 2a: publish for any cross-component / cross-thread reader. Done
  // up front so the value is correct even when called between sub-phases of
  // this same buffer (e.g. a future helper that reads its own current frame).
  mPublishedFrame.store(mCurrentFrame, std::memory_order_release);

  // Phase 2e: refresh the Link clock's frame↔microsecond anchor + cached
  // tempo once per buffer (no-op until Phase 2e.2 hooks the real Link
  // SessionState capture). Cheap if the active source isn't Link.
  mLinkClock.onAudioBuffer(bufferStartFrame, sampleRate);

  // 2. Drain a bounded number of commands from BOTH inboxes. Each one
  //    mutates audio-thread state and may emit events. Bounded to keep
  //    the audio callback within a predictable budget regardless of
  //    burst size. Dart inbox first (user input has lower latency budget),
  //    worker inbox second.
  Command cmd;
  for (int i = 0; i < kMaxCommandsPerBuffer; ++i) {
    if (!mInbox.pop(&cmd)) break;
    handleCommand(cmd);
  }
  for (int i = 0; i < kMaxCommandsPerBuffer; ++i) {
    if (!mWorkerInbox.pop(&cmd)) break;
    handleCommand(cmd);
  }

  // 3. Emit BeatFired / DownbeatFired events for any beats that crossed
  //    this buffer. Must happen after command drain (so a same-buffer
  //    SetTempo correctly resets) and before snapshot publish (so the
  //    snapshot's currentBeat is consistent with the events Dart observes).
  checkAndEmitBeats(mCurrentFrame, sampleRate);

  // 3b. Fire any pending mute/unmute/gain entries whose fire frame falls
  //     within this buffer. Same ordering rationale as beats — must happen
  //     after command drain (so a same-buffer QueueMute can fire immediately
  //     in stab mode) and before snapshot publish.
  firePendingThroughFrame(mCurrentFrame);

  // 4. Compose and publish the snapshot. Query active SyncSource for the
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
          mLastEmittedBeat = kBeatCounterUninit;
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
      mLastEmittedBeat = kBeatCounterUninit;
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
      mLastEmittedBeat = kBeatCounterUninit;

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
      mLastEmittedBeat = kBeatCounterUninit;
      mMetronome.reset();
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
      switch (static_cast<SyncSourceKind>(cmd.id)) {
        case SyncSourceKind::Local:
          mActiveSource     = &mLocalClock;
          mActiveSourceKind = SyncSourceKind::Local;
          mLastEmittedBeat  = kBeatCounterUninit;
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
        case SyncSourceKind::AbletonLink:
          // Phase 2e: route musical time through the Link clock. In 2e.1
          // the clock returns isValid()=false so AudioEngine effectively
          // sees "no tempo" until 2e.2 hooks the real Link SessionState.
          // The switch still happens so SyncSourceChanged surfaces to Dart
          // (UI badge state, NetworkTimingService toggle plumbing).
          mActiveSource     = &mLinkClock;
          mActiveSourceKind = SyncSourceKind::AbletonLink;
          mLastEmittedBeat  = kBeatCounterUninit;
          emitEvent(Event{
              /*type=*/Event::Type::SyncSourceChanged,
              /*reserved=*/{0, 0, 0},
              /*id=*/static_cast<std::uint32_t>(SyncSourceKind::AbletonLink),
              /*frame=*/mCurrentFrame,
              /*framesProcessed=*/0,
              /*soundHash=*/0,
              /*code=*/0,
          });
          break;
        case SyncSourceKind::None:
        case SyncSourceKind::MidiClock:
          // Not implemented; surface an error so Dart can show a friendly
          // "MIDI clock not supported" message instead of silently no-op.
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
      // Re-anchor the beat counter so we don't emit a backlog when the
      // metronome is turned on mid-session.
      mLastEmittedBeat = kBeatCounterUninit;
      // No event emitted for the SetMetronome itself; Dart already knows
      // the setting it just posted.
      break;

    case Command::Type::ReportKeyInferred: {
      // Worker thread → audio thread → Dart. We just unpack and forward as
      // an Event. The audio thread is the sole producer on the outbox so
      // this preserves SPSC semantics on both queues.
      Event ev{};
      ev.type             = Event::Type::KeyInferred;
      ev.id               = cmd.id;
      ev.frame            = mCurrentFrame;
      ev.framesProcessed  = 0;
      ev.soundHash        = cmd.soundHash;  // float-cast confidence
      ev.code             = cmd.flags & 0x1u;  // isMinor
      emitEvent(ev);
      break;
    }

    case Command::Type::QueueMute: {
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      PendingEntry pe{};
      pe.action     = PendingAction::Mute;
      pe.trackIndex = cmd.id;
      pe.fireFrame  = (cmd.targetFrame != 0)
                          ? cmd.targetFrame
                          : resolveQuantizedFireFrame(mCurrentFrame,
                                                       cmd.flags, sr);
      pe.value      = 0.0f;
      (void)upsertPendingEntry(pe);
      break;
    }

    case Command::Type::QueueUnmute: {
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      PendingEntry pe{};
      pe.action     = PendingAction::Unmute;
      pe.trackIndex = cmd.id;
      pe.fireFrame  = (cmd.targetFrame != 0)
                          ? cmd.targetFrame
                          : resolveQuantizedFireFrame(mCurrentFrame,
                                                       cmd.flags, sr);
      pe.value      = unpackFloatFromSoundHash(cmd.soundHash);
      (void)upsertPendingEntry(pe);
      break;
    }

    case Command::Type::SetTrackGain: {
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      PendingEntry pe{};
      pe.action     = PendingAction::SetGain;
      pe.trackIndex = cmd.id;
      pe.fireFrame  = (cmd.targetFrame != 0)
                          ? cmd.targetFrame
                          : resolveQuantizedFireFrame(mCurrentFrame,
                                                       cmd.flags, sr);
      pe.value      = unpackFloatFromSoundHash(cmd.soundHash);
      (void)upsertPendingEntry(pe);
      break;
    }

    case Command::Type::CancelPendingQueue:
      (void)removePendingEntry(cmd.id);
      break;

    // ── Phase 1 native transport ────────────────────────────────────────
    case Command::Type::QueuePause: {
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      PendingEntry pe{};
      pe.action     = PendingAction::Pause;
      pe.trackIndex = cmd.id;
      pe.fireFrame  = (cmd.targetFrame != 0)
                          ? cmd.targetFrame
                          : resolveQuantizedFireFrame(mCurrentFrame,
                                                       cmd.flags, sr);
      pe.value      = 0.0f;
      (void)upsertPendingEntry(pe);
      break;
    }

    case Command::Type::QueueUnpause: {
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      PendingEntry pe{};
      pe.action     = PendingAction::Unpause;
      pe.trackIndex = cmd.id;
      pe.fireFrame  = (cmd.targetFrame != 0)
                          ? cmd.targetFrame
                          : resolveQuantizedFireFrame(mCurrentFrame,
                                                       cmd.flags, sr);
      pe.value      = 0.0f;
      (void)upsertPendingEntry(pe);
      break;
    }

    case Command::Type::QueueStop: {
      const std::uint32_t sr = mSnapshot.load().sampleRate;
      PendingEntry pe{};
      pe.action     = PendingAction::Stop;
      pe.trackIndex = cmd.id;
      pe.fireFrame  = (cmd.targetFrame != 0)
                          ? cmd.targetFrame
                          : resolveQuantizedFireFrame(mCurrentFrame,
                                                       cmd.flags, sr);
      pe.value      = 0.0f;
      (void)upsertPendingEntry(pe);
      break;
    }

    case Command::Type::RegisterTrackHandle: {
      if (!registerTrackHandle(cmd.id, cmd.soundHash)) {
        // Table full — emit Error so Dart can surface a leak. The bridged
        // setter calls for this trackIndex will be no-ops until something
        // unregisters.
        emitEvent(Event{
            /*type=*/Event::Type::Error,
            /*reserved=*/{0, 0, 0},
            /*id=*/cmd.id,
            /*frame=*/mCurrentFrame,
            /*framesProcessed=*/0,
            /*soundHash=*/cmd.soundHash,
            /*code=*/1,  // 1 = track handle table full
        });
      }
      break;
    }

    case Command::Type::UnregisterTrackHandle:
      (void)unregisterTrackHandle(cmd.id);
      break;

    case Command::Type::SetLinkEnabled: {
      const bool enable = (cmd.flags & 0x1u) != 0;
      mLinkClock.setEnabled(enable);
      // 2e.2 will publish a SyncSourceChanged-like event so Dart knows
      // peer-count / connect state changed. For now flipping the flag is
      // enough — the active source switch is a separate command.
      break;
    }

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

std::int64_t AudioEngine::resolveQuantizedFireFrame(
    std::int64_t currentFrame, std::uint32_t quantizeSixteenths,
    std::uint32_t sampleRate) const noexcept {
  if (quantizeSixteenths == 0) return currentFrame;
  if (mActiveSource == nullptr || !mActiveSource->isValid()) return currentFrame;
  if (sampleRate == 0) return currentFrame;

  const double bpm = mActiveSource->tempoBPM();
  if (bpm <= 0.0) return currentFrame;

  // Sixteenth-resolution math. anchor = frame for beat 0. We walk forward in
  // sixteenth steps until we land on a grid-aligned position.
  const std::int64_t anchorFrame =
      mActiveSource->frameForBeat(0, sampleRate);
  const double framesPerSixteenth =
      (static_cast<double>(sampleRate) * 60.0) / (bpm * 4.0);
  if (framesPerSixteenth <= 0.0) return currentFrame;

  // Current position expressed in sixteenths since the anchor. Negative
  // results (we're before anchor) snap forward to sixteenth 0 first.
  const double rawSixteenth =
      static_cast<double>(currentFrame - anchorFrame) / framesPerSixteenth;
  const double currentSixteenth = (rawSixteenth < 0.0) ? 0.0 : rawSixteenth;

  // Ceiling-divide currentSixteenth onto the grid of multiples of
  // quantizeSixteenths. Add a tiny epsilon so "almost on the boundary" still
  // advances to the *next* boundary (avoids "fire this buffer" surprises at
  // sub-sample alignment).
  constexpr double kEpsilonSixteenths = 1.0e-6;
  const double scaled =
      (currentSixteenth + kEpsilonSixteenths) /
      static_cast<double>(quantizeSixteenths);
  // ceil of a non-negative double.
  std::int64_t scaledCeil = static_cast<std::int64_t>(scaled);
  if (static_cast<double>(scaledCeil) < scaled) ++scaledCeil;

  const double alignedSixteenth =
      static_cast<double>(scaledCeil) *
      static_cast<double>(quantizeSixteenths);
  return anchorFrame +
         static_cast<std::int64_t>(alignedSixteenth * framesPerSixteenth);
}

bool AudioEngine::upsertPendingEntry(const PendingEntry& entry) noexcept {
  // Replace any existing entry for this trackIndex.
  for (std::size_t i = 0; i < mPendingQueueCount; ++i) {
    if (mPendingQueue[i].trackIndex == entry.trackIndex &&
        mPendingQueue[i].action != PendingAction::None) {
      mPendingQueue[i] = entry;
      return true;
    }
  }
  if (mPendingQueueCount >= kMaxPendingQueueEntries) {
    emitEvent(Event{
        /*type=*/Event::Type::Error,
        /*reserved=*/{0, 0, 0},
        /*id=*/entry.trackIndex,
        /*frame=*/mCurrentFrame,
        /*framesProcessed=*/0,
        /*soundHash=*/0,
        /*code=*/3,  // kPendingQueueFull
    });
    return false;
  }
  mPendingQueue[mPendingQueueCount++] = entry;
  return true;
}

bool AudioEngine::removePendingEntry(std::uint32_t trackIndex) noexcept {
  for (std::size_t i = 0; i < mPendingQueueCount; ++i) {
    if (mPendingQueue[i].trackIndex == trackIndex &&
        mPendingQueue[i].action != PendingAction::None) {
      // Swap-with-last to keep the array compact without shifting.
      mPendingQueue[i] = mPendingQueue[mPendingQueueCount - 1];
      --mPendingQueueCount;
      return true;
    }
  }
  return false;
}

// ── Phase 1 native transport: trackIndex → SoLoud handle table ───────────
//
// All three methods are audio-thread-only — they're invoked from
// handleCommand (which runs on the audio thread) and from
// firePendingThroughFrame (also audio-thread). No synchronisation needed.

std::uint32_t AudioEngine::lookupSoloudHandle(
    std::uint32_t trackIndex) const noexcept {
  if (trackIndex == 0) return 0;
  for (std::size_t i = 0; i < kMaxTrackHandles; ++i) {
    if (mTrackHandles[i].trackIndex == trackIndex &&
        mTrackHandles[i].soloudHandle != 0) {
      return mTrackHandles[i].soloudHandle;
    }
  }
  return 0;
}

bool AudioEngine::registerTrackHandle(std::uint32_t trackIndex,
                                       std::uint32_t soloudHandle) noexcept {
  if (trackIndex == 0 || soloudHandle == 0) return false;

  // Single pass: replace an existing entry for the same trackIndex, OR fall
  // back to the first empty slot we saw. The two cases share the loop so
  // the audio thread does at most one scan per registration.
  std::size_t emptySlot = kMaxTrackHandles;  // sentinel = "none found yet"
  for (std::size_t i = 0; i < kMaxTrackHandles; ++i) {
    if (mTrackHandles[i].trackIndex == trackIndex &&
        mTrackHandles[i].soloudHandle != 0) {
      mTrackHandles[i].soloudHandle = soloudHandle;
      return true;
    }
    if (emptySlot == kMaxTrackHandles &&
        mTrackHandles[i].soloudHandle == 0) {
      emptySlot = i;
    }
  }
  if (emptySlot == kMaxTrackHandles) return false;
  mTrackHandles[emptySlot].trackIndex   = trackIndex;
  mTrackHandles[emptySlot].soloudHandle = soloudHandle;
  return true;
}

bool AudioEngine::unregisterTrackHandle(std::uint32_t trackIndex) noexcept {
  if (trackIndex == 0) return false;
  for (std::size_t i = 0; i < kMaxTrackHandles; ++i) {
    if (mTrackHandles[i].trackIndex == trackIndex &&
        mTrackHandles[i].soloudHandle != 0) {
      mTrackHandles[i].soloudHandle = 0;
      mTrackHandles[i].trackIndex   = 0;
      return true;
    }
  }
  return false;
}

void AudioEngine::firePendingThroughFrame(
    std::int64_t bufferEndFrame) noexcept {
  // Two-pass: emit events for entries due this buffer, then compact the
  // array. We rebuild in-place to keep allocation-free behavior on the
  // audio thread.
  std::size_t write = 0;
  for (std::size_t read = 0; read < mPendingQueueCount; ++read) {
    const PendingEntry& pe = mPendingQueue[read];
    if (pe.action == PendingAction::None) continue;
    if (pe.fireFrame > bufferEndFrame) {
      if (write != read) mPendingQueue[write] = pe;
      ++write;
      continue;
    }
    Event::Type evType = Event::Type::Error;
    switch (pe.action) {
      case PendingAction::Mute:    evType = Event::Type::VoiceMuted;     break;
      case PendingAction::Unmute:  evType = Event::Type::VoiceUnmuted;   break;
      case PendingAction::SetGain: evType = Event::Type::GainChanged;    break;
      case PendingAction::Pause:   evType = Event::Type::VoicePaused;    break;
      case PendingAction::Unpause: evType = Event::Type::VoiceUnpaused;  break;
      case PendingAction::Stop:    evType = Event::Type::PlaybackEnded;  break;
      case PendingAction::None:    continue;  // unreachable; defensive
    }

    // Phase 1: apply the audio change RIGHT HERE on the audio thread via
    // the SoLoud setter bridge, before emitting the event. Dart still sees
    // the event for UI bookkeeping (mute button state, transport indicator,
    // PerformanceRecorder) but the audible change no longer waits on Dart.
    // SoLoud handle 0 means "trackIndex isn't registered yet" — skip the
    // setter (Dart's still applying via the event listener as a fallback).
    const std::uint32_t soloudHandle = lookupSoloudHandle(pe.trackIndex);
    if (soloudHandle != 0) {
      switch (pe.action) {
        case PendingAction::Mute:
          if (g_soloudSetVolume) g_soloudSetVolume(soloudHandle, 0.0f);
          break;
        case PendingAction::Unmute:
          if (g_soloudSetVolume) g_soloudSetVolume(soloudHandle, pe.value);
          break;
        case PendingAction::SetGain:
          if (g_soloudSetVolume) g_soloudSetVolume(soloudHandle, pe.value);
          break;
        case PendingAction::Pause:
          if (g_soloudSetPause) g_soloudSetPause(soloudHandle, true);
          break;
        case PendingAction::Unpause:
          if (g_soloudSetPause) g_soloudSetPause(soloudHandle, false);
          break;
        case PendingAction::Stop:
          if (g_soloudStop) g_soloudStop(soloudHandle);
          // Stop reclaims the voice — unregister so subsequent fires don't
          // touch a slot SoLoud may have already reused.
          (void)unregisterTrackHandle(pe.trackIndex);
          break;
        case PendingAction::None:
          break;
      }

      // LSAEC: an on/off-type mix change (NOT a continuous gain automation,
      // which would thrash the seed-capture worker) makes the template's
      // per-phase content stale even though the loop period didn't move —
      // re-arm a convergence-seed reseed so cancellation catches up in ~1
      // pass instead of several. See synchronous_echo_template.h.
      switch (pe.action) {
        case PendingAction::Mute:
        case PendingAction::Unmute:
        case PendingAction::Pause:
        case PendingAction::Unpause:
        case PendingAction::Stop:
          if (mFilters) mFilters->notifyAecReferenceChanged();
          break;
        default:
          break;
      }
    }

    Event ev{};
    ev.type            = evType;
    ev.id              = pe.trackIndex;
    ev.frame           = pe.fireFrame;
    ev.framesProcessed = 0;
    ev.soundHash       = (pe.action == PendingAction::Unmute ||
                          pe.action == PendingAction::SetGain)
                             ? packFloatToSoundHash(pe.value)
                             : 0u;
    ev.code            = 0;
    emitEvent(ev);
    // Don't copy into write slot — entry is consumed.
  }
  mPendingQueueCount = write;
}

void AudioEngine::checkAndEmitBeats(std::int64_t bufferEndFrame,
                                     std::uint32_t sampleRate) noexcept {
  if (!mMetronomeEnabled) return;
  if (mActiveSource == nullptr || !mActiveSource->isValid()) return;
  if (sampleRate == 0) return;

  // Beat number at the end of this buffer. Beats are 0-indexed from the
  // active source's anchor.
  const std::uint32_t endBeat =
      mActiveSource->beatAtFrame(bufferEndFrame, sampleRate);

  // First time through (or just after a tempo / source change): just anchor
  // and emit nothing. Avoids a flurry of "historical" beats when the user
  // toggles the metronome mid-session.
  if (mLastEmittedBeat == kBeatCounterUninit) {
    mLastEmittedBeat = endBeat;
    return;
  }

  if (endBeat <= mLastEmittedBeat) {
    // No new beat boundary crossed. Common case in a 256-frame buffer at
    // 120 BPM (one beat ≈ 24000 frames apart).
    return;
  }

  // Safety cap. Real audio buffers cover well under one beat at sensible
  // tempos (256 frames @ 48 kHz @ 200 BPM ≈ 0.05 beats). The cap exists to
  // catch pathological cases — clock jumps, system suspend, manual frame
  // seeks — where emitting every elided beat would flood the outbox.
  // Tempo-change resets are already handled separately via kBeatCounterUninit.
  // 32 beats is several full bars even at slow tempos; anything beyond is a
  // glitch worth anchoring through.
  constexpr std::uint32_t kMaxBeatsPerBuffer = 32;
  if (endBeat - mLastEmittedBeat > kMaxBeatsPerBuffer) {
    mLastEmittedBeat = endBeat;
    return;
  }

  const std::uint32_t q = mActiveSource->quantum();
  for (std::uint32_t beat = mLastEmittedBeat + 1; beat <= endBeat; ++beat) {
    const bool isDownbeat = (q > 0) && (beat % q == 0);
    if (mMetronomeDownbeatOnly && !isDownbeat) continue;

    const std::int64_t beatFrame =
        mActiveSource->frameForBeat(beat, sampleRate);
    emitEvent(Event{
        /*type=*/isDownbeat ? Event::Type::DownbeatFired
                            : Event::Type::BeatFired,
        /*reserved=*/{0, 0, 0},
        /*id=*/beat,
        /*frame=*/beatFrame,
        /*framesProcessed=*/0,
        /*soundHash=*/0,
        /*code=*/0,
    });
    // Schedule a sample-accurate audible click at the same frame. The voice
    // mixes into the output buffer post-SoLoud in data_callback. We schedule
    // every beat (including non-downbeats) when the metronome is fully
    // enabled; downbeat-only mode is handled by the early continue above.
    mMetronome.schedule(beatFrame, isDownbeat);
  }
  mLastEmittedBeat = endBeat;
}

void AudioEngine::mixMetronomeIntoOutput(
    float* output, std::int64_t bufferStartFrame,
    std::uint32_t frameCount, std::uint16_t channels,
    std::uint32_t sampleRate) noexcept {
  // The metronome voice itself handles the "no active clicks" fast path.
  // We unconditionally call it so it can lazy-regenerate the click samples
  // on first use / sample-rate change.
  mMetronome.mix(output, bufferStartFrame, frameCount, channels, sampleRate);
}

void AudioEngine::emitFromAudioThread(const Event& event) noexcept {
  emitEvent(event);
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
