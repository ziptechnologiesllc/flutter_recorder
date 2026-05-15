// AbletonLinkClock — SyncSource backed by an Ableton Link session.
//
// Phase 2e.1 (this file): scaffolding only. The class is a real SyncSource
// implementation with stubbed bodies that report `isValid() == false` so the
// AudioEngine falls back to LocalClock until Phase 2e.2 wires the actual
// `ableton::Link` instance + CMake linkage.
//
// Phase 2e.2: replace mEnabled with a real `ableton::Link link{120.0};`,
// query its SessionState in tempoBPM() / beatAtFrame() / phaseInBar() via
// frame↔microsecond conversion anchored on each AudioEngine::process()
// callback.
//
// Audio-thread readers are wait-free; Dart-side `SetLinkEnabled` commands
// flow through AudioEngine's command queue so flipping the enable flag and
// switching the active SyncSource both happen on the audio thread.

#ifndef FLOWSTATE_AUDIO_ENGINE_ABLETON_LINK_CLOCK_H_
#define FLOWSTATE_AUDIO_ENGINE_ABLETON_LINK_CLOCK_H_

#include <atomic>
#include <cstdint>

#include "sync_source.h"

namespace flowstate {
namespace audio_engine {

class AbletonLinkClock final : public SyncSource {
 public:
  AbletonLinkClock() = default;
  AbletonLinkClock(const AbletonLinkClock&) = delete;
  AbletonLinkClock& operator=(const AbletonLinkClock&) = delete;

  // Toggle Link-session participation. Phase 2e.1 just flips an atomic;
  // 2e.2 will call `link.enable(b)` on the real Link instance.
  void setEnabled(bool enabled) noexcept;
  bool isEnabled() const noexcept;

  // Refreshed once per audio buffer by AudioEngine::process() so the wall-
  // clock ↔ frame conversion used in SyncSource queries stays current.
  // 2e.2 will capture `link.captureAudioSessionState()` here too.
  void onAudioBuffer(std::int64_t bufferStartFrame,
                     std::uint32_t sampleRate) noexcept;

  // Number of peers in the Link session. 0 when disabled or solo.
  std::uint32_t numPeers() const noexcept;

  // SyncSource interface. 2e.1 returns LocalClock-equivalent defaults
  // (isValid=false → AudioEngine treats us as "no tempo"); 2e.2 routes
  // these through Link's SessionState.
  bool isValid() const noexcept override;
  double tempoBPM() const noexcept override;
  std::uint32_t quantum() const noexcept override;
  double phaseInBar(std::int64_t frame,
                    std::uint32_t sampleRate) const noexcept override;
  std::uint32_t beatAtFrame(std::int64_t frame,
                             std::uint32_t sampleRate) const noexcept override;
  std::int64_t nextDownbeatFrame(
      std::int64_t frame, std::uint32_t sampleRate) const noexcept override;
  std::int64_t frameForBeat(std::uint32_t beat,
                             std::uint32_t sampleRate) const noexcept override;

 private:
  std::atomic<bool>          mEnabled{false};
  std::atomic<std::uint32_t> mNumPeers{0};
  // Phase 2e.2 will add:
  //   ableton::Link              mLink{120.0};
  //   std::atomic<double>        mCachedTempoBPM{0.0};
  //   std::atomic<std::int64_t>  mAnchorFrame{0};
  //   std::atomic<std::int64_t>  mAnchorMicros{0};
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_ABLETON_LINK_CLOCK_H_
