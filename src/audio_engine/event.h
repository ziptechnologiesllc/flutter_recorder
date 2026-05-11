// Events flow audio engine -> Dart via SPSC queue.
//
// POD only: trivially copyable, fixed layout. Dart drains these on a 120 Hz
// polling timer and folds them into AppState.

#ifndef FLOWSTATE_AUDIO_ENGINE_EVENT_H_
#define FLOWSTATE_AUDIO_ENGINE_EVENT_H_

#include <cstdint>
#include <type_traits>

namespace flowstate {
namespace audio_engine {

struct Event {
  enum class Type : std::uint8_t {
    None = 0,
    RecordingStarted,    // audio thread began capturing slot `id` at `frame`
    RecordingStopped,    // ended at `frame`; captured `framesProcessed`;
                         //   worker may produce a WAV at the slot's path
    PlaybackStarted,     // voice `id` began at `frame`, length `framesProcessed`
                         //   for sound `soundHash`
    PlaybackEnded,       // voice `id` ended at `frame`
    BaseLoopSet,         // legacy: base loop installed (kept during 2 parallel-run)
    BaseLoopCleared,     // legacy
    TempoSet,            // new tempo active at `frame`; loop length frames
                         //   in `framesProcessed`; `code` packs quantum
    TempoCleared,        // tempo cleared at `frame`
    SyncSourceChanged,   // `id` = new SyncSourceKind enum value
    DownbeatFired,       // metronome downbeat at `frame` (beat = `id`)
    BeatFired,           // metronome non-downbeat beat at `frame`
    TempoInferred,       // worker thread: `id` = clipId,
                         //   bpm bit-cast into `framesProcessed` lower 8 bytes
    KeyInferred,         // worker thread: `id` = clipId, `code` = packed key
    Error,               // diagnostic; semantic in `code`
  };

  Type           type{Type::None};
  std::uint8_t   reserved_0[3]{};
  std::uint32_t  id{0};
  std::int64_t   frame{0};           // global frame when event happened
  std::int64_t   framesProcessed{0}; // RecordingStopped: extracted frames
                                     // PlaybackStarted: voice length
                                     // BaseLoopSet: loop length
  std::uint32_t  soundHash{0};       // PlaybackStarted: SoLoud sound hash
  std::uint32_t  code{0};            // Error: error code, otherwise unused
};

static_assert(std::is_trivially_copyable<Event>::value,
              "Event must be trivially copyable");
static_assert(sizeof(Event) == 32,
              "Event must be exactly 32 bytes for predictable outbox sizing");

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_EVENT_H_
