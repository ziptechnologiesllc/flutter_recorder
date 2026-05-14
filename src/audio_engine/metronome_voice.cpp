#include "metronome_voice.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace flowstate {
namespace audio_engine {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Generate a Hann-windowed sine click into `dst`. Hann window kills the click
// at both ends of the sample so we don't introduce DC steps that thump.
void generateClickInto(std::vector<float>& dst, double freqHz, float amplitude,
                       std::uint32_t sampleRate) {
  const std::size_t frames = static_cast<std::size_t>(
      MetronomeVoice::kClickDurationSec * sampleRate);
  dst.assign(frames, 0.0f);
  if (frames < 2 || sampleRate == 0) return;

  const double dt = 1.0 / static_cast<double>(sampleRate);
  for (std::size_t i = 0; i < frames; ++i) {
    const double t = static_cast<double>(i) * dt;
    const double s = std::sin(2.0 * kPi * freqHz * t);
    const double w = 0.5 * (1.0 - std::cos(2.0 * kPi *
                                            static_cast<double>(i) /
                                            static_cast<double>(frames - 1)));
    dst[i] = static_cast<float>(amplitude * s * w);
  }
}

}  // namespace

void MetronomeVoice::regenerateClicks(std::uint32_t sampleRate) noexcept {
  generateClickInto(mDownbeatSample, kDownbeatFreqHz, kDownbeatAmp, sampleRate);
  generateClickInto(mBeatSample,     kBeatFreqHz,     kBeatAmp,     sampleRate);
  mSampleRate = sampleRate;
}

void MetronomeVoice::schedule(std::int64_t startFrame,
                              bool isDownbeat) noexcept {
  // Try to find a free slot.
  for (ActiveClick& slot : mActiveClicks) {
    if (!slot.active) {
      slot.startFrame   = startFrame;
      slot.framesMixed  = 0;
      slot.isDownbeat   = isDownbeat;
      slot.active       = true;
      return;
    }
  }
  // All slots busy — replace the slot with the earliest startFrame (the one
  // closest to finishing anyway). Should not happen at sane tempos given
  // kMaxConcurrentClicks = 4 and clicks ~30 ms.
  ActiveClick* oldest = &mActiveClicks[0];
  for (ActiveClick& slot : mActiveClicks) {
    if (slot.startFrame < oldest->startFrame) oldest = &slot;
  }
  oldest->startFrame   = startFrame;
  oldest->framesMixed  = 0;
  oldest->isDownbeat   = isDownbeat;
  oldest->active       = true;
}

void MetronomeVoice::reset() noexcept {
  for (ActiveClick& slot : mActiveClicks) {
    slot.active = false;
  }
}

std::size_t MetronomeVoice::activeClickCount() const noexcept {
  std::size_t n = 0;
  for (const ActiveClick& slot : mActiveClicks) {
    if (slot.active) ++n;
  }
  return n;
}

void MetronomeVoice::mix(float* output, std::int64_t bufferStartFrame,
                         std::uint32_t frameCount, std::uint16_t channels,
                         std::uint32_t sampleRate) noexcept {
  if (output == nullptr || frameCount == 0 || channels == 0) return;
  if (sampleRate == 0) return;

  // Lazy click regeneration on first mix call (or device sample rate change).
  if (sampleRate != mSampleRate) {
    regenerateClicks(sampleRate);
  }
  const std::size_t clickFrames = mBeatSample.size();
  if (clickFrames == 0) return;  // regeneration failed; nothing to mix.

  const std::int64_t bufferEndFrame =
      bufferStartFrame + static_cast<std::int64_t>(frameCount);

  for (ActiveClick& slot : mActiveClicks) {
    if (!slot.active) continue;

    const std::int64_t clickEndFrame =
        slot.startFrame + static_cast<std::int64_t>(clickFrames);

    // Click already finished before this buffer began?
    if (clickEndFrame <= bufferStartFrame) {
      slot.active = false;
      continue;
    }
    // Click hasn't started yet by the end of this buffer?
    if (slot.startFrame >= bufferEndFrame) {
      continue;
    }

    // Intersect [slot.startFrame, clickEndFrame) with [bufferStart, bufferEnd).
    const std::int64_t overlapStart =
        std::max(slot.startFrame, bufferStartFrame);
    const std::int64_t overlapEnd =
        std::min(clickEndFrame, bufferEndFrame);
    if (overlapStart >= overlapEnd) continue;

    const std::size_t framesToMix =
        static_cast<std::size_t>(overlapEnd - overlapStart);
    const std::size_t bufferOffset =
        static_cast<std::size_t>(overlapStart - bufferStartFrame);
    const std::size_t clickOffset =
        static_cast<std::size_t>(overlapStart - slot.startFrame);

    const float* sample =
        slot.isDownbeat ? mDownbeatSample.data() : mBeatSample.data();

    // Mix mono click into every output channel.
    if (channels == 1) {
      for (std::size_t i = 0; i < framesToMix; ++i) {
        output[bufferOffset + i] += sample[clickOffset + i];
      }
    } else if (channels == 2) {
      // Stereo: interleaved L R L R ...
      for (std::size_t i = 0; i < framesToMix; ++i) {
        const float s = sample[clickOffset + i];
        const std::size_t outIdx = (bufferOffset + i) * 2;
        output[outIdx]     += s;
        output[outIdx + 1] += s;
      }
    } else {
      // Generic N-channel.
      for (std::size_t i = 0; i < framesToMix; ++i) {
        const float s = sample[clickOffset + i];
        const std::size_t outIdx = (bufferOffset + i) * channels;
        for (std::uint16_t c = 0; c < channels; ++c) {
          output[outIdx + c] += s;
        }
      }
    }

    slot.framesMixed += static_cast<std::int64_t>(framesToMix);
    if (slot.framesMixed >= static_cast<std::int64_t>(clickFrames)) {
      slot.active = false;
    }
  }
}

}  // namespace audio_engine
}  // namespace flowstate
