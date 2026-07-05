// LocalClock — free-running tempo source driven by user-provided values.
//
// State is three numbers:
//   tempoBPM     — beats per minute (0.0 means "no tempo set")
//   quantum      — beats per bar (4 is the default for 4/4 loops)
//   anchorFrame  — global frame where phase=0 (the downbeat)
//
// Single writer (audio thread inside AudioEngine::handleCommand), multi-
// reader (audio thread query methods + worker thread occasional reads).
// We use plain atomics on each field; the tempo math is robust to seeing
// any two of {tempo, quantum, anchor} from one buffer combined with a third
// from the next — the worst case is one buffer of musical position glitch
// while a tempo change settles. Not worth a seqlock for the writer cost.

#ifndef FLOWSTATE_AUDIO_ENGINE_LOCAL_CLOCK_H_
#define FLOWSTATE_AUDIO_ENGINE_LOCAL_CLOCK_H_

#include <atomic>
#include <cstdint>

#include "sync_source.h"

namespace flowstate {
namespace audio_engine {

class LocalClock final : public SyncSource {
 public:
  LocalClock() = default;
  LocalClock(const LocalClock&) = delete;
  LocalClock& operator=(const LocalClock&) = delete;

  // Writers — called only from the audio thread inside AudioEngine. Stores
  // are release-ordered so a Dart-side poller through the snapshot sees a
  // consistent view via the snapshot's own seqlock.
  void setTempo(double bpm, std::uint32_t quantum,
                std::int64_t anchorFrame) noexcept;
  void clear() noexcept;

  // SyncSource interface.
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

  // Convenience: anchor frame as published.
  std::int64_t anchorFrame() const noexcept;

 private:
  std::atomic<double>        mTempoBPM{0.0};
  std::atomic<std::uint32_t> mQuantum{0};
  std::atomic<std::int64_t>  mAnchorFrame{0};
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_LOCAL_CLOCK_H_
