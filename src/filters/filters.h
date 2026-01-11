#ifndef FILTERS_H
#define FILTERS_H

#include "../enums.h"
#include "generic_filter.h"
#include "aec/neural_post_filter.h"

#include <functional>
#include <memory>
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

  // Sample-accurate AEC synchronization
  // Call before processing filters with current capture frame count
  void setAecCaptureFrameCount(size_t captureFrameCount);
  // Set calibrated offset for position-based sync
  void setAecCalibratedOffset(int64_t offset);
  int64_t getAecCalibratedOffset() const;

  // Aligned calibration capture (for accurate delay estimation)
  void startAecCalibrationCapture(size_t maxSamples);
  void stopAecCalibrationCapture();
  const std::vector<float> &getAecAlignedRef() const;
  const std::vector<float> &getAecAlignedMic() const;

  // AEC Mode Control
  void setAecMode(AecMode mode);
  AecMode getAecMode() const;

  // Neural Model Control
  bool loadNeuralModel(NeuralModelType modelType, const std::string &assetBasePath);
  NeuralModelType getLoadedNeuralModel() const;
  void setNeuralEnabled(bool enabled);
  bool isNeuralEnabled() const;

  unsigned int mSamplerate;
  unsigned int mChannels;

  std::vector<std::unique_ptr<FilterObject>> filters;
};

#endif // PLAYER_H
