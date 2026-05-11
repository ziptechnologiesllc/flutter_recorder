// Snapshot of audio engine state, published via Seqlock every audio buffer.
//
// Sized to fit comfortably in one cache line so the seqlock writer stays
// fast on the audio thread. Per-recording / per-voice detail beyond the
// bitmasks is deferred to Phase 3.

#ifndef FLOWSTATE_AUDIO_ENGINE_SNAPSHOT_H_
#define FLOWSTATE_AUDIO_ENGINE_SNAPSHOT_H_

#include <cstdint>
#include <type_traits>

namespace flowstate {
namespace audio_engine {

struct Snapshot {
  std::int64_t  currentFrame{0};
  std::int64_t  baseLoopStart{0};
  std::int64_t  baseLoopFrames{0};      // 0 == no base loop set
  std::uint32_t sampleRate{0};
  std::uint16_t channels{0};
  std::uint16_t bufferSize{0};
  std::uint8_t  activeRecordingMask{0}; // bit i: Recording slot i is active
  std::uint8_t  reserved_0{0};
  std::uint16_t activeVoiceMask{0};     // bit i: Voice slot i is active
  float         captureLevelDb{-100.0f};
  float         playbackLevelDb{-100.0f};
  std::uint32_t reserved_1{0};
};

static_assert(std::is_trivially_copyable<Snapshot>::value,
              "Snapshot must be trivially copyable");
static_assert(sizeof(Snapshot) <= 64,
              "Snapshot must fit in one cache line");

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_SNAPSHOT_H_
