// Phase 2e.1 — scaffolded AbletonLinkClock. All SyncSource methods are
// stubbed to LocalClock-equivalent defaults so the AudioEngine can hold a
// valid AbletonLinkClock member and route SetSyncSource(AbletonLink) onto
// it without crashing. Phase 2e.2 swaps these for real `ableton::Link`
// SessionState queries; see the header for the migration plan.

#include "ableton_link_clock.h"

namespace flowstate {
namespace audio_engine {

void AbletonLinkClock::setEnabled(bool enabled) noexcept {
  mEnabled.store(enabled, std::memory_order_release);
  // Phase 2e.2: mLink.enable(enabled);
}

bool AbletonLinkClock::isEnabled() const noexcept {
  return mEnabled.load(std::memory_order_acquire);
}

void AbletonLinkClock::onAudioBuffer(std::int64_t /*bufferStartFrame*/,
                                      std::uint32_t /*sampleRate*/) noexcept {
  // Phase 2e.2: capture link.captureAudioSessionState(), refresh
  // mCachedTempoBPM + mAnchor{Frame,Micros} for the wall-clock↔frame
  // conversion used by the SyncSource methods below.
}

std::uint32_t AbletonLinkClock::numPeers() const noexcept {
  // Phase 2e.2: return link.numPeers();
  return mNumPeers.load(std::memory_order_acquire);
}

// ── SyncSource ──────────────────────────────────────────────────────────
//
// Phase 2e.1 stubs: report invalid so AudioEngine falls back to LocalClock
// for tempo/beat/phase math until 2e.2 hooks real Link queries.

bool AbletonLinkClock::isValid() const noexcept {
  return false;
}

double AbletonLinkClock::tempoBPM() const noexcept {
  return 0.0;
}

std::uint32_t AbletonLinkClock::quantum() const noexcept {
  return 0;
}

double AbletonLinkClock::phaseInBar(std::int64_t /*frame*/,
                                     std::uint32_t /*sampleRate*/) const noexcept {
  return 0.0;
}

std::uint32_t AbletonLinkClock::beatAtFrame(
    std::int64_t /*frame*/, std::uint32_t /*sampleRate*/) const noexcept {
  return 0;
}

std::int64_t AbletonLinkClock::nextDownbeatFrame(
    std::int64_t frame, std::uint32_t /*sampleRate*/) const noexcept {
  // Not valid → caller branches on isValid() and skips us. Return `frame`
  // as a safe default rather than zero (matches LocalClock's "no anchor"
  // fallback).
  return frame;
}

std::int64_t AbletonLinkClock::frameForBeat(
    std::uint32_t /*beat*/, std::uint32_t /*sampleRate*/) const noexcept {
  return 0;
}

}  // namespace audio_engine
}  // namespace flowstate
