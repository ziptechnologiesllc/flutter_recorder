// Phase 2e.2 — real Ableton Link integration. The class is also compilable
// with FLOWSTATE_ABLETON_LINK undefined (e.g. on a platform where Link
// hasn't been wired into CMake yet) — in that case every SyncSource method
// reports "no tempo" and the engine runs on LocalClock instead.

#include "ableton_link_clock.h"

#include <algorithm>
#include <chrono>

#if defined(FLOWSTATE_ABLETON_LINK)
#include <ableton/Link.hpp>
#endif

namespace flowstate {
namespace audio_engine {

namespace {
// Default quantum (beats per bar). Matches LocalClock's default and the
// `_quantum = 4.0` used elsewhere in the Dart timing layer. Configurable
// via setQuantum() in a later phase if we expose it to Dart.
constexpr double kDefaultQuantum = 4.0;
}  // namespace

#if defined(FLOWSTATE_ABLETON_LINK)
struct AbletonLinkClockImpl {
  // Default Link tempo (120 BPM) until either the user dials in a session
  // tempo or a peer's tempo overrides ours.
  ableton::Link link{120.0};
};
#else
struct AbletonLinkClockImpl {};  // empty stub when Link isn't wired
#endif

AbletonLinkClock::AbletonLinkClock()
    : mImpl(std::make_unique<AbletonLinkClockImpl>()) {}

AbletonLinkClock::~AbletonLinkClock() = default;

void AbletonLinkClock::setEnabled(bool enabled) noexcept {
#if defined(FLOWSTATE_ABLETON_LINK)
  // NOT realtime-safe. Caller must be on Dart's main thread (we route
  // through a dedicated FFI entry point, not the audio command queue).
  mImpl->link.enable(enabled);
#endif
  mEnabled.store(enabled, std::memory_order_release);
}

bool AbletonLinkClock::isEnabled() const noexcept {
  return mEnabled.load(std::memory_order_acquire);
}

void AbletonLinkClock::onAudioBuffer(std::int64_t bufferStartFrame,
                                      std::uint32_t sampleRate) noexcept {
#if defined(FLOWSTATE_ABLETON_LINK)
  if (!mImpl->link.isEnabled() || sampleRate == 0) {
    mEnabled.store(false, std::memory_order_release);
    mCachedTempoBPM.store(0.0, std::memory_order_release);
    return;
  }
  mEnabled.store(true, std::memory_order_release);
  mNumPeers.store(static_cast<std::uint32_t>(mImpl->link.numPeers()),
                  std::memory_order_release);

  // Capture session state — RT-safe per Link's contract; only valid in this
  // audio callback. We extract tempo + (anchor frame ↔ anchor microseconds)
  // and stash them in atomics so the SyncSource queries can run wait-free
  // from any thread without re-entering Link.
  const auto state = mImpl->link.captureAudioSessionState();
  const auto micros = mImpl->link.clock().micros();
  mCachedTempoBPM.store(state.tempo(), std::memory_order_release);
  mAnchorFrame.store(bufferStartFrame, std::memory_order_release);
  mAnchorMicros.store(micros.count(), std::memory_order_release);
#else
  (void)bufferStartFrame;
  (void)sampleRate;
#endif
}

std::uint32_t AbletonLinkClock::numPeers() const noexcept {
  return mNumPeers.load(std::memory_order_acquire);
}

// ── SyncSource interface ────────────────────────────────────────────────
//
// All five musical-time queries derive their answer from the cached anchor
// (refreshed once per buffer in onAudioBuffer) + the caller's `frame` and
// `sampleRate`. We avoid re-querying Link here so these can run on any
// thread without locks.

namespace {

// Convert a global frame count to a wall-clock microsecond using the
// most-recent anchor captured from Link. Both anchor fields are atomically
// updated together each buffer; a torn read between them would land us one
// buffer (~5 ms) off, which is below human-perceptible misalignment for a
// metronome read. If perfect coherency mattered we'd seqlock them; the
// trade-off in audio-thread cost isn't worth it.
std::chrono::microseconds anchoredMicrosForFrame(
    std::int64_t frame, std::uint32_t sampleRate,
    std::int64_t anchorFrame, std::int64_t anchorMicros) {
  const std::int64_t deltaFrames = frame - anchorFrame;
  const std::int64_t deltaMicros =
      (deltaFrames * 1'000'000) / static_cast<std::int64_t>(sampleRate);
  return std::chrono::microseconds(anchorMicros + deltaMicros);
}

}  // namespace

bool AbletonLinkClock::isValid() const noexcept {
  // Valid only when Link is enabled AND we've seen at least one buffer
  // where it published a tempo (mCachedTempoBPM > 0 after onAudioBuffer).
  return mEnabled.load(std::memory_order_acquire) &&
         mCachedTempoBPM.load(std::memory_order_acquire) > 0.0;
}

double AbletonLinkClock::tempoBPM() const noexcept {
  return mCachedTempoBPM.load(std::memory_order_acquire);
}

std::uint32_t AbletonLinkClock::quantum() const noexcept {
  return static_cast<std::uint32_t>(kDefaultQuantum);
}

double AbletonLinkClock::phaseInBar(std::int64_t frame,
                                     std::uint32_t sampleRate) const noexcept {
#if defined(FLOWSTATE_ABLETON_LINK)
  if (!isValid() || sampleRate == 0) return 0.0;
  const auto t = anchoredMicrosForFrame(
      frame, sampleRate, mAnchorFrame.load(std::memory_order_acquire),
      mAnchorMicros.load(std::memory_order_acquire));
  // Reading SessionState from a non-audio thread is documented to be safe
  // via captureAppSessionState, but that's not RT-safe. For phase queries
  // we use the cached snapshot's beat math instead: phaseAtTime is purely
  // a function of (time, quantum, tempo, anchor), and we have all four.
  // We approximate it by deriving phase from the cached tempo + anchor
  // delta. Drift between buffers is ~5 ms which doesn't show in a phase
  // display.
  const double bpm = mCachedTempoBPM.load(std::memory_order_acquire);
  const double secondsSinceAnchor = (t.count() - mAnchorMicros.load(
                                          std::memory_order_acquire)) /
                                     1'000'000.0;
  const double beatsSinceAnchor = secondsSinceAnchor * bpm / 60.0;
  const double phase = std::fmod(beatsSinceAnchor, kDefaultQuantum);
  return phase < 0 ? (phase + kDefaultQuantum) / kDefaultQuantum
                   : phase / kDefaultQuantum;
#else
  (void)frame;
  (void)sampleRate;
  return 0.0;
#endif
}

std::uint32_t AbletonLinkClock::beatAtFrame(
    std::int64_t frame, std::uint32_t sampleRate) const noexcept {
#if defined(FLOWSTATE_ABLETON_LINK)
  if (!isValid() || sampleRate == 0) return 0;
  const double bpm = mCachedTempoBPM.load(std::memory_order_acquire);
  const std::int64_t deltaFrames =
      frame - mAnchorFrame.load(std::memory_order_acquire);
  const double secondsSinceAnchor =
      static_cast<double>(deltaFrames) / static_cast<double>(sampleRate);
  const double beatsSinceAnchor = secondsSinceAnchor * bpm / 60.0;
  return static_cast<std::uint32_t>(std::max(0.0, std::floor(beatsSinceAnchor)));
#else
  (void)frame;
  (void)sampleRate;
  return 0;
#endif
}

std::int64_t AbletonLinkClock::nextDownbeatFrame(
    std::int64_t frame, std::uint32_t sampleRate) const noexcept {
#if defined(FLOWSTATE_ABLETON_LINK)
  if (!isValid() || sampleRate == 0) return frame;
  const double bpm = mCachedTempoBPM.load(std::memory_order_acquire);
  if (bpm <= 0.0) return frame;
  const double framesPerBeat =
      (static_cast<double>(sampleRate) * 60.0) / bpm;
  const double framesPerBar = framesPerBeat * kDefaultQuantum;
  const std::int64_t deltaFrames =
      frame - mAnchorFrame.load(std::memory_order_acquire);
  const double barsSinceAnchor =
      static_cast<double>(deltaFrames) / framesPerBar;
  const double nextBar = std::ceil(barsSinceAnchor);
  return mAnchorFrame.load(std::memory_order_acquire) +
         static_cast<std::int64_t>(nextBar * framesPerBar);
#else
  (void)sampleRate;
  return frame;
#endif
}

std::int64_t AbletonLinkClock::frameForBeat(
    std::uint32_t beat, std::uint32_t sampleRate) const noexcept {
#if defined(FLOWSTATE_ABLETON_LINK)
  if (!isValid() || sampleRate == 0) return 0;
  const double bpm = mCachedTempoBPM.load(std::memory_order_acquire);
  if (bpm <= 0.0) return 0;
  const double framesPerBeat =
      (static_cast<double>(sampleRate) * 60.0) / bpm;
  return mAnchorFrame.load(std::memory_order_acquire) +
         static_cast<std::int64_t>(beat * framesPerBeat);
#else
  (void)beat;
  (void)sampleRate;
  return 0;
#endif
}

}  // namespace audio_engine
}  // namespace flowstate
