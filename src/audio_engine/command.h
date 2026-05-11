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
    SetBaseLoop,       // set the base loop: lengthFrames, anchored at frame
                       //   given by current global frame minus lengthFrames
                       //   (engine computes this from its transport)
    ClearBaseLoop,     // free mode; no quantization
    StartPlayback,     // begin voice `id` for sound `soundHash` at targetFrame
    StopPlayback,      // stop voice `id` at targetFrame
    SetLatencyComp,    // free-mode pre-roll in frames (`lengthFrames`)
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
