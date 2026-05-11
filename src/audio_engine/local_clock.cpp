#include "local_clock.h"

#include <cmath>

namespace flowstate {
namespace audio_engine {

namespace {

// Helper: frames per bar at the given tempo + quantum + sample rate.
// Returns 0 if any input is invalid.
inline double framesPerBar(double tempoBPM, std::uint32_t quantum,
                            std::uint32_t sampleRate) noexcept {
  if (tempoBPM <= 0.0 || quantum == 0 || sampleRate == 0) {
    return 0.0;
  }
  // frames/beat = sampleRate * 60 / bpm.  frames/bar = frames/beat * quantum.
  return (static_cast<double>(sampleRate) * 60.0 *
          static_cast<double>(quantum)) /
         tempoBPM;
}

inline double framesPerBeat(double tempoBPM,
                             std::uint32_t sampleRate) noexcept {
  if (tempoBPM <= 0.0 || sampleRate == 0) return 0.0;
  return (static_cast<double>(sampleRate) * 60.0) / tempoBPM;
}

}  // namespace

void LocalClock::setTempo(double bpm, std::uint32_t quantum,
                          std::int64_t anchorFrame) noexcept {
  // Discard nonsense; clear() is the right way to zero out.
  if (bpm <= 0.0 || quantum == 0) {
    clear();
    return;
  }
  // Order: write anchor first, then quantum, then tempo. Readers that race
  // and observe a half-update will see (new anchor, old tempo) which yields
  // a one-buffer position glitch rather than a divide-by-zero or NaN.
  mAnchorFrame.store(anchorFrame, std::memory_order_release);
  mQuantum.store(quantum, std::memory_order_release);
  mTempoBPM.store(bpm, std::memory_order_release);
}

void LocalClock::clear() noexcept {
  // Zero tempo first so isValid() returns false even if a reader sees
  // half-cleared state.
  mTempoBPM.store(0.0, std::memory_order_release);
  mQuantum.store(0, std::memory_order_release);
  mAnchorFrame.store(0, std::memory_order_release);
}

bool LocalClock::isValid() const noexcept {
  return mTempoBPM.load(std::memory_order_acquire) > 0.0 &&
         mQuantum.load(std::memory_order_acquire) > 0;
}

double LocalClock::tempoBPM() const noexcept {
  return mTempoBPM.load(std::memory_order_acquire);
}

std::uint32_t LocalClock::quantum() const noexcept {
  return mQuantum.load(std::memory_order_acquire);
}

std::int64_t LocalClock::anchorFrame() const noexcept {
  return mAnchorFrame.load(std::memory_order_acquire);
}

double LocalClock::phaseInBar(std::int64_t frame,
                               std::uint32_t sampleRate) const noexcept {
  const double bpm = tempoBPM();
  const std::uint32_t q = quantum();
  const std::int64_t anchor = anchorFrame();
  const double fpb = framesPerBar(bpm, q, sampleRate);
  if (fpb <= 0.0) return 0.0;

  const double rel = static_cast<double>(frame - anchor);
  // C-style fmod can return negative for negative rel; we want [0, 1).
  double p = std::fmod(rel, fpb) / fpb;
  if (p < 0.0) p += 1.0;
  return p;
}

std::uint32_t LocalClock::beatAtFrame(
    std::int64_t frame, std::uint32_t sampleRate) const noexcept {
  const double bpm = tempoBPM();
  const std::int64_t anchor = anchorFrame();
  const double fpb = framesPerBeat(bpm, sampleRate);
  if (fpb <= 0.0) return 0;
  if (frame <= anchor) return 0;
  const double beats = static_cast<double>(frame - anchor) / fpb;
  if (beats <= 0.0) return 0;
  return static_cast<std::uint32_t>(beats);
}

std::int64_t LocalClock::nextDownbeatFrame(
    std::int64_t frame, std::uint32_t sampleRate) const noexcept {
  const double bpm = tempoBPM();
  const std::uint32_t q = quantum();
  const std::int64_t anchor = anchorFrame();
  const double fpb = framesPerBar(bpm, q, sampleRate);
  if (fpb <= 0.0) return frame;

  const double rel = static_cast<double>(frame - anchor);
  // How many complete bars (rounded down toward -inf) have passed since
  // anchor? std::floor handles negative rel correctly.
  const double barsElapsed = std::floor(rel / fpb);
  // Frame of the bar that just completed at-or-before `frame`:
  const double thisBarStart =
      static_cast<double>(anchor) + barsElapsed * fpb;
  // If frame coincides with thisBarStart, return frame (caller convention).
  if (std::fabs(static_cast<double>(frame) - thisBarStart) < 0.5) {
    return frame;
  }
  // Otherwise, next downbeat is one bar later.
  const double nextStart = thisBarStart + fpb;
  return static_cast<std::int64_t>(std::llround(nextStart));
}

}  // namespace audio_engine
}  // namespace flowstate
