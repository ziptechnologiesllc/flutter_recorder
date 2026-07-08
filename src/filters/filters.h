#ifndef FILTERS_H
#define FILTERS_H

#include "../enums.h"
#include "generic_filter.h"
#include "aec/neural_post_filter.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct FilterObject {
  RecorderFilterType type;
  std::unique_ptr<GenericFilter> filter;

  FilterObject(RecorderFilterType t, std::unique_ptr<GenericFilter> f)
      : type(t), filter(std::move(f)) {}

  bool operator==(RecorderFilterType const &i) { return (i == type); }
};

/// Class to manage global filters.
class Filters {
  /// Setting the global filter to NULL will clear the global filter.
  /// The default maximum number of global filters active is 4, but this
  /// can be changed in a global constant in soloud.h (and rebuilding SoLoud).
public:
  Filters(unsigned int samplerate, unsigned int channels = 2);
  ~Filters();

  /// Return -1 if the filter is not active or its index
  int isFilterActive(RecorderFilterType filter);

  CaptureErrors addFilter(RecorderFilterType filterType);

  CaptureErrors removeFilter(RecorderFilterType filterType);

  std::vector<std::string> getFilterParamNames(RecorderFilterType filterType);

  /// If [handle]==0 the operation is done to global filters.
  void setFilterParams(RecorderFilterType filterType, int attributeId,
                       float value);

  /// If [handle]==0 the operation is done to global filters.
  float getFilterParams(RecorderFilterType filterType, int attributeId);

  /// Set AEC impulse response from calibration
  void setAecImpulseResponse(const float *coeffs, int length);

  AecStats getAecStats();

  // VSS-NLMS parameter control for experimentation
  void setAecVssMuMax(float mu);
  void setAecVssLeakage(float lambda);
  void setAecVssAlpha(float alpha);
  float getAecVssMuMax() const;
  float getAecVssLeakage() const;
  float getAecVssAlpha() const;

  // Filter length control
  void setAecFilterLength(int length);
  int getAecFilterLength() const;

  // LSAEC E5: lock-free gated-ERLE telemetry snapshot (zeroed if no AEC filter).
  AecTelemetrySnapshot getAecTelemetry() const;

  // Sample-accurate AEC synchronization
  // Call before processing filters with current capture frame count
  void setAecCaptureFrameCount(size_t captureFrameCount);
  // Set calibrated offset for position-based sync
  void setAecCalibratedOffset(int64_t offset);
  int64_t getAecCalibratedOffset() const;
  // Set acoustic delay for slave mode (pure room delay, no thread timing)
  void setAecAcousticDelaySamples(size_t samples);

  // Aligned calibration capture (for accurate delay estimation)
  void startAecCalibrationCapture(size_t maxSamples);
  void stopAecCalibrationCapture();
  const std::vector<float> &getAecAlignedRef() const;
  const std::vector<float> &getAecAlignedMic() const;

  // AEC Mode Control
  void setAecMode(AecMode mode);
  AecMode getAecMode() const;

  // LSAEC: the audible mix changed WITHOUT a loop-period change — a track
  // was muted/unmuted/paused/stopped. The template is indexed by loop phase,
  // not by mix content, so it keeps cancelling against a now-stale reference
  // shape until it slowly relearns. This re-arms the same convergence-seed
  // capture used for a brand-new loop (recapture one period of the NEW
  // reference, reconvolve with the calibrated room IR) so cancellation
  // catches up in ~1 pass instead of several. Cheap/safe to call often —
  // internally gated so it's a no-op while a seed job is already in flight.
  void notifyAecReferenceChanged();

  // LSAEC per-track exact subtraction — see SynchronousEchoTemplate's doc
  // comment for the full rationale. registerAecTrackAudio: call once a
  // track's audio is known (any thread); computes that track's echo
  // contribution off-thread. setAecTrackActive: call at the SAME sample-
  // accurate instant the SoLoud mute/unmute/pause/unpause/stop setter
  // fires, so the AEC state change and the audible state change are
  // atomic. Both no-ops if LSAEC isn't active.
  void registerAecTrackAudio(int trackIndex, const float *audioMono,
                             int64_t frames);
  void setAecTrackActive(int trackIndex, bool active);

  // Neural Model Control
  bool loadNeuralModel(NeuralModelType modelType, const std::string &assetBasePath);
  NeuralModelType getLoadedNeuralModel() const;
  void setNeuralEnabled(bool enabled);
  bool isNeuralEnabled() const;

  unsigned int mSamplerate;
  unsigned int mChannels;

  // AEC mode requested before/after the filter exists. setAecMode always
  // stores here; addFilter applies it to a freshly created AEC instance so a
  // mode set at boot (before saved-calibration adds the filter) is not lost.
  AecMode mRequestedAecMode = aecModeAlgo;

  std::vector<std::unique_ptr<FilterObject>> filters;
  mutable std::mutex mFiltersMutex;  // Protects 'filters' vector for thread-safety

  // Lock contention tracking for debug overlay
  std::atomic<uint64_t> mFilterMissCount{0};    // Times we skipped due to lock
  std::atomic<uint64_t> mFilterProcessCount{0}; // Times we successfully processed

  // Lock-free filter count for hot path (updated when filters added/removed)
  std::atomic<size_t> mFilterCountCached{0};

  // Thread-safe method to process all filters (for use in audio callback)
  void processAllFilters(void* pInput, ma_uint32 frameCount,
                         unsigned int channels, ma_format format);

  // Thread-safe method to get filter count (for use in audio callback)
  size_t getFilterCount() const;

  // Lock-free check if any filters exist (for hot path optimization)
  bool hasFilters() const {
    return mFilterCountCached.load(std::memory_order_relaxed) > 0;
  }

  // Debug stats for overlay
  uint64_t getFilterMissCount() const { return mFilterMissCount.load(std::memory_order_relaxed); }
  uint64_t getFilterProcessCount() const { return mFilterProcessCount.load(std::memory_order_relaxed); }
  void resetFilterStats() {
    mFilterMissCount.store(0, std::memory_order_relaxed);
    mFilterProcessCount.store(0, std::memory_order_relaxed);
  }
};

// The single global filter chain instance (flutter_recorder.cpp). Exposed
// here, matching the g_soloudSetVolume-style extern-global convention already
// used across this plugin, so same-plugin native code outside capture.cpp
// (e.g. audio_engine.cpp's mute/pause handling) can reach it directly without
// a new FFI surface.
extern std::unique_ptr<Filters> mFilters;

#endif // PLAYER_H
