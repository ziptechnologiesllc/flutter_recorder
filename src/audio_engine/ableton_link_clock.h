// AbletonLinkClock — SyncSource backed by an Ableton Link session.
//
// Phase 2e.2: the real integration. When FLOWSTATE_ABLETON_LINK is defined
// at compile time, this class wraps an `ableton::Link` instance and routes
// musical-time queries through its SessionState. When the macro is NOT
// defined (e.g. a platform target where Link wasn't wired into CMake yet),
// every SyncSource method falls back to a "no tempo" stub and the engine
// runs purely on LocalClock — the public API is identical either way.
//
// Threading:
//   - setEnabled(): NOT realtime-safe (Link's enable() opens sockets,
//     starts background threads). Call from Dart's main thread via the
//     dedicated FFI entry point, never from the audio callback.
//   - onAudioBuffer(): audio-thread only. Calls
//     `link.captureAudioSessionState()` (RT-safe per Link's contract) and
//     refreshes a cache of tempo + (anchorFrame ↔ anchorMicros) so the
//     wait-free SyncSource queries below can convert frame counts into
//     wall-clock microseconds.
//   - SyncSource methods: wait-free reads of cached atomics; callable from
//     any thread.

#ifndef FLOWSTATE_AUDIO_ENGINE_ABLETON_LINK_CLOCK_H_
#define FLOWSTATE_AUDIO_ENGINE_ABLETON_LINK_CLOCK_H_

#include <atomic>
#include <cstdint>
#include <memory>

#include "sync_source.h"

namespace flowstate {
namespace audio_engine {

// Forward-declared pImpl so Link headers don't leak into every translation
// unit that includes audio_engine.h. The pImpl is allocated unconditionally
// (a 16-byte stub when Link isn't compiled in); kept simple — this clock
// only exists once per process anyway.
struct AbletonLinkClockImpl;

class AbletonLinkClock final : public SyncSource {
 public:
  AbletonLinkClock();
  ~AbletonLinkClock();
  AbletonLinkClock(const AbletonLinkClock&) = delete;
  AbletonLinkClock& operator=(const AbletonLinkClock&) = delete;

  // ── Main-thread API (NOT realtime-safe) ─────────────────────────────────
  // Call from Dart via dedicated FFI entry points, never from the audio
  // callback. enable() spins up Link's background thread on first true.
  void setEnabled(bool enabled) noexcept;
  bool isEnabled() const noexcept;

  // ── Audio-thread API ────────────────────────────────────────────────────
  // Called by AudioEngine::process() once per buffer. Captures Link's
  // SessionState (RT-safe) and refreshes the frame↔microsecond anchor +
  // cached tempo / quantum / phase atomics that the wait-free SyncSource
  // queries below read.
  void onAudioBuffer(std::int64_t bufferStartFrame,
                     std::uint32_t sampleRate) noexcept;

  // ── Telemetry (wait-free, any thread) ───────────────────────────────────
  std::uint32_t numPeers() const noexcept;

  // ── SyncSource interface (wait-free, any thread) ────────────────────────
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
  // Cached per-buffer state, refreshed in onAudioBuffer(). Reads are
  // wait-free; writes are single-producer (audio thread). The SyncSource
  // queries combine these with the caller's `frame` argument to derive
  // beat / phase / next-downbeat without re-entering Link from any thread
  // but the audio one.
  std::atomic<bool>          mEnabled{false};
  std::atomic<std::uint32_t> mNumPeers{0};
  std::atomic<double>        mCachedTempoBPM{0.0};
  std::atomic<std::int64_t>  mAnchorFrame{0};
  std::atomic<std::int64_t>  mAnchorMicros{0};

  // Implementation detail (`ableton::Link` instance) hidden behind pImpl so
  // <ableton/Link.hpp> doesn't transitively appear in every audio_engine.h
  // consumer.
  std::unique_ptr<AbletonLinkClockImpl> mImpl;
};

}  // namespace audio_engine
}  // namespace flowstate

#endif  // FLOWSTATE_AUDIO_ENGINE_ABLETON_LINK_CLOCK_H_
