// Commands flow Dart -> audio engine via SPSC queue.
//
// POD only: trivially copyable, no pointers that need ownership semantics.
// Field layout is fixed at 32 bytes so the inbox capacity is predictable.

#ifndef FLOWSTATE_AUDIO_ENGINE_COMMAND_H_
#define FLOWSTATE_AUDIO_ENGINE_COMMAND_H_

#include <cstdint>
#include <type_traits>

namespace flowstate {
namespace audio_engine {

struct Command {
  enum class Type : std::uint8_t {
    None = 0,
    StartRecording,    // begin recording slot `id` at `targetFrame`
                       //   (0 = immediate, else quantized to that frame)
    StopRecording,     // stop recording slot `id` at `targetFrame`
    SetBaseLoop,       // legacy: set base loop length+anchor; deprecated in
                       //   Phase 2c. Today: derives tempo (assuming
                       //   quantum=4) and forwards to LocalClock.
    ClearBaseLoop,     // legacy: clears tempo too. Deprecated in 2c.
    StartPlayback,     // begin voice `id` for sound `soundHash` at targetFrame
    StopPlayback,      // stop voice `id` at targetFrame
    SetLatencyComp,    // free-mode pre-roll in frames (`lengthFrames`)
    SetTempo,          // tempoBPM in lengthFrames (type-pun: see Dart side),
                       //   quantum in flags, anchor frame in targetFrame
    ClearTempo,        // back to free mode (no quantization)
    SetSyncSource,     // `id` = SyncSourceKind enum value
    SetMetronome,      // flags: bit 0 = enable, bit 1 = downbeat-only
    ReportKeyInferred, // worker → audio: id = pitchClass (0-11, 255=unknown),
                       //   flags bit 0 = isMinor, soundHash = float-cast
                       //   confidence

    // ── Phase 2c: tap-to-mute / MIDI Performance ──────────────────────────
    // Track-level mute/unmute and gain control, quantized to a launch grid
    // (bar/beat/sixteenth/free). The audio thread schedules the fire frame
    // and emits VoiceMuted/VoiceUnmuted/GainChanged events at that frame.
    // SoLoud playback gain is applied Dart-side on event receipt — the
    // audio thread is just the scheduler + truth source for "when".
    //
    // Field layout for all four:
    //   id            = track index (Dart-maintained UUID↔index map)
    //   flags         = quantize in 1/16 units: 0=immediate/free, 1=1/16,
    //                   2=1/8, 4=1/4 (beat), 8=1/2 (half-bar), 16=bar
    //   soundHash     = bit-cast float payload:
    //                     QueueUnmute  → velocity ∈ [0,1]
    //                     SetTrackGain → gain ∈ [0, ~4]
    //                   Unused for QueueMute / CancelPendingQueue.
    //   targetFrame   = optional explicit fire frame; 0 = let engine resolve
    //                   from `flags` (the common case).
    QueueMute,
    QueueUnmute,
    SetTrackGain,
    CancelPendingQueue,

    // ── Phase 1 native transport: per-track pause/unpause/stop on the
    //    audio thread, plus the trackIndex→SoLoud-handle table they read.
    //    All four share the layout of the Phase 2c mute commands so a
    //    single fire-frame resolver works across the board.
    //
    // QueuePause / QueueUnpause / QueueStop:
    //   id           = trackIndex
    //   flags        = LaunchQuantize (1/16 units; 0 = immediate)
    //   targetFrame  = optional explicit fire frame; 0 = engine resolves
    //   soundHash    = unused (no payload float)
    //
    // RegisterTrackHandle: Dart announces the SoLoud handle that backs a
    // trackIndex so the audio thread can call the bridged setters at fire
    // time. UnregisterTrackHandle removes the mapping on stop / cleanup.
    //   id           = trackIndex
    //   soundHash    = SoLoud voice handle (uint32_t)
    //   flags        = unused
    //   targetFrame  = unused
    QueuePause,
    QueueUnpause,
    QueueStop,
    RegisterTrackHandle,
    UnregisterTrackHandle,
  };

  Type           type{Type::None};
  std::uint8_t   reserved_0[3]{};  // pad to 4-byte boundary for id
  std::uint32_t  id{0};            // recording slot id or voice id
  std::int64_t   targetFrame{0};   // 0 = immediate
  std::int64_t   lengthFrames{0};  // SetBaseLoop / SetLatencyComp payload
  std::uint32_t  soundHash{0};     // StartPlayback: which sound to play
  std::uint32_t  flags{0};         // bitfield (looping, mute, etc.)
};

static_assert(std::is_trivially_copyable<Command>::value,
              "Command must be trivially copyable");
static_assert(sizeof(Command) == 32,
              "Command must be exactly 32 bytes for predictable inbox sizing");

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_COMMAND_H_
