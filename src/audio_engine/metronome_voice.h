// Sample-accurate metronome click voice.
//
// The audio engine schedules clicks when beat boundaries cross a buffer; the
// data_callback then asks this voice to mix any active clicks into the output
// before the buffer goes through AEC reference / device-format conversion.
//
// Threading: schedule() and mix() are both called exclusively from the audio
// thread (or test code single-threaded). No locks; no allocations on the hot
// path. Click sample buffers are regenerated lazily on the first mix() call
// with a new sample rate, which only happens at device init.

#ifndef FLOWSTATE_AUDIO_ENGINE_METRONOME_VOICE_H_
#define FLOWSTATE_AUDIO_ENGINE_METRONOME_VOICE_H_

#include <array>
#include <cstdint>
#include <vector>

namespace flowstate {
namespace audio_engine {

class MetronomeVoice {
 public:
  // 30 ms of click length at the active sample rate. Long enough to be
  // audible / "tactile," short enough to never collide with the next beat
  // at sane tempos.
  static constexpr double kClickDurationSec = 0.030;

  // How many simultaneous clicks we can have ringing. Plenty for any tempo
  // up to several hundred BPM; the click is shorter than the gap between
  // beats anyway.
  static constexpr std::size_t kMaxConcurrentClicks = 4;

  // Pitch + amplitude defaults. Downbeat is higher / louder so it stands
  // out from the other beats.
  static constexpr double kDownbeatFreqHz = 1500.0;
  static constexpr float  kDownbeatAmp = 0.55f;
  static constexpr double kBeatFreqHz = 800.0;
  static constexpr float  kBeatAmp = 0.40f;

  MetronomeVoice() = default;
  MetronomeVoice(const MetronomeVoice&) = delete;
  MetronomeVoice& operator=(const MetronomeVoice&) = delete;

  // Schedule a click to play starting at the given global frame. If all
  // concurrency slots are full, the oldest click is replaced (latest beat
  // wins; humanly imperceptible at any sane tempo).
  void schedule(std::int64_t startFrame, bool isDownbeat) noexcept;

  // Mix any active clicks into the output buffer. The output is interleaved
  // float; the click is mono and is replicated across all output channels.
  // bufferStartFrame is the global frame of output[0]; frameCount is the
  // number of frames (samples per channel) in the buffer.
  //
  // sampleRate must match the device sample rate. If it differs from the
  // last call's sampleRate, the click sample buffers are regenerated.
  void mix(float* output, std::int64_t bufferStartFrame,
           std::uint32_t frameCount, std::uint16_t channels,
           std::uint32_t sampleRate) noexcept;

  // Cancel all active clicks. Called when sync source changes mid-stream.
  void reset() noexcept;

  // For tests / diagnostics.
  std::size_t activeClickCount() const noexcept;

 private:
  struct ActiveClick {
    std::int64_t startFrame;     // global frame where click begins
    std::int64_t framesMixed;    // frames of click already mixed into output
    bool         isDownbeat;
    bool         active;
  };

  // Regenerate click sample buffers for the given sample rate. Cheap (~1ms
  // of pre-computation); called only when sampleRate changes.
  void regenerateClicks(std::uint32_t sampleRate) noexcept;

  std::array<ActiveClick, kMaxConcurrentClicks> mActiveClicks{};
  std::vector<float>                            mDownbeatSample;
  std::vector<float>                            mBeatSample;
  std::uint32_t                                 mSampleRate{0};
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_METRONOME_VOICE_H_
