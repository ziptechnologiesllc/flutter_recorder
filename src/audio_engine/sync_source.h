// SyncSource — abstract interface for whatever drives musical time.
//
// Audio-thread reads must be wait-free; readers spin at most a handful of
// instructions per query. Implementations:
//   LocalClock        — free-running tempo, set by user / tap / inference
//   AbletonLinkClock  — driven by an Ableton Link session (Phase 4)
//   MidiClock         — external MIDI clock (future)
//
// The active source is held as a raw pointer by AudioEngine. Lifetimes are
// owned by AudioEngine (each implementation is a member), so the pointer is
// always valid; we just swap which member is "active" when SetSyncSource
// fires. Single writer (audio thread) → no atomics needed for the pointer
// itself in this phase.

#ifndef FLOWSTATE_AUDIO_ENGINE_SYNC_SOURCE_H_
#define FLOWSTATE_AUDIO_ENGINE_SYNC_SOURCE_H_

#include <cstdint>

namespace flowstate {
namespace audio_engine {

class SyncSource {
 public:
  virtual ~SyncSource() = default;

  // True if the source has a tempo set. When false, phase / boundary queries
  // return zero defaults — callers should branch on this rather than treat
  // them as meaningful musical positions.
  virtual bool isValid() const noexcept = 0;

  // Audio-thread queries. Must be wait-free.

  // Tempo in beats per minute.
  virtual double tempoBPM() const noexcept = 0;

  // Beats per loop / bar (e.g. 4 for 4/4 single-bar loops, 8 for 4/4 two-bar
  // loops). Phase 2 treats this as "loop length in beats"; full time-signature
  // support arrives with bar/beat-level effects in Phase 11+.
  virtual std::uint32_t quantum() const noexcept = 0;

  // Phase within the bar, in [0, 1). 0.0 == downbeat.
  virtual double phaseInBar(std::int64_t frame,
                             std::uint32_t sampleRate) const noexcept = 0;

  // Total beats elapsed since the source's anchor (monotonic, integer).
  virtual std::uint32_t beatAtFrame(std::int64_t frame,
                                     std::uint32_t sampleRate) const noexcept = 0;

  // Global frame of the next bar-boundary (downbeat) at or after `frame`.
  // If frame already coincides with a downbeat, returns frame itself.
  virtual std::int64_t nextDownbeatFrame(
      std::int64_t frame, std::uint32_t sampleRate) const noexcept = 0;
};

// Enumerated kinds, useful for SetSyncSource commands and snapshot reporting.
enum class SyncSourceKind : std::uint8_t {
  None = 0,
  Local = 1,
  AbletonLink = 2,
  MidiClock = 3,
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_SYNC_SOURCE_H_
