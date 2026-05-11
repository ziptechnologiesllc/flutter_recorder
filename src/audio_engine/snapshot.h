// Snapshot of audio engine state, published via Seqlock every audio buffer.
//
// Phase 2: snapshot grew past a single cache line because we now publish the
// full musical transport (tempo, quantum, phase, current beat, next downbeat)
// alongside the legacy base-loop fields. Two cache lines is the worst case;
// we can split into per-row seqlocks if profiling shows the write cost
// matters. Today it doesn't — once per buffer (~5.3 ms) is plenty cheap.
//
// Per-voice and per-recording detail will live in separate seqlock-protected
// arrays in Phase 2c so the hot transport read stays cache-friendly.

#ifndef FLOWSTATE_AUDIO_ENGINE_SNAPSHOT_H_
#define FLOWSTATE_AUDIO_ENGINE_SNAPSHOT_H_

#include <cstdint>
#include <type_traits>

namespace flowstate {
namespace audio_engine {

struct Snapshot {
  // --- Transport (musical time) ---
  std::int64_t  currentFrame{0};       // bufferEnd of most recent process()
  std::int64_t  nextDownbeatFrame{0};  // 0 if no tempo set
  double        tempoBPM{0.0};         // 0 if no tempo set
  std::uint32_t quantum{0};            // beats per loop / bar; 0 if no tempo
  std::uint32_t currentBeat{0};        // total beats since anchor (monotonic)
  float         phaseInBar{0.0f};      // [0, 1); 0 if no tempo
  std::uint8_t  sigNumerator{4};
  std::uint8_t  sigDenominator{4};
  std::uint8_t  syncSourceKind{0};     // matches SyncSourceKind enum
  std::uint8_t  reserved_0{0};

  // --- Legacy base-loop fields (kept during parallel-run; remove in 2c) ---
  std::int64_t  baseLoopStart{0};
  std::int64_t  baseLoopFrames{0};     // 0 == no base loop

  // --- Device config ---
  std::uint32_t sampleRate{0};
  std::uint16_t channels{0};
  std::uint16_t bufferSize{0};

  // --- Activity bitmasks ---
  std::uint8_t  activeRecordingMask{0};  // bit i: Recording slot i is active
  std::uint8_t  reserved_1{0};
  std::uint16_t activeVoiceMask{0};      // bit i: Voice slot i is active
  float         captureLevelDb{-100.0f};
  float         playbackLevelDb{-100.0f};
  std::uint32_t reserved_2{0};
};

static_assert(std::is_trivially_copyable<Snapshot>::value,
              "Snapshot must be trivially copyable");
static_assert(sizeof(Snapshot) <= 128,
              "Snapshot must fit in two cache lines; consider splitting if "
              "this assertion ever fires");

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_SNAPSHOT_H_
