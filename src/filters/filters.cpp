#include "filters.h"
#include "aec/adaptive_echo_cancellation.h"
#include "autogain.h"
#include "echo_cancellation.h"

#include <memory>
#include <string>
#include <vector>

// External logging function defined in calibration.cpp
extern void aecLog(const char *fmt, ...);

Filters::Filters(unsigned int samplerate, unsigned int channels)
    : mSamplerate(samplerate), mChannels(channels) {}

Filters::~Filters() {}

int Filters::isFilterActive(RecorderFilterType filter) {
  for (int i = 0; i < filters.size(); i++) {
    if (filters[i].get()->type == filter)
      return i;
  }
  return -1;
}

std::vector<std::string>
Filters::getFilterParamNames(RecorderFilterType filterType) {
  std::vector<std::string> ret;
  switch (filterType) {
  case RecorderFilterType::autogain: {
    AutoGain f;
    int nParams = f.getParamCount();
    for (int i = 0; i < nParams; i++)
      ret.push_back(f.getParamName(i));
  } break;
  case RecorderFilterType::echoCancellation: {
    EchoCancellation f;
    int nParams = f.getParamCount();
    for (int i = 0; i < nParams; i++)
      ret.push_back(f.getParamName(i));
  } break;
  case RecorderFilterType::adaptiveEchoCancellation: {
    AdaptiveEchoCancellation f(mSamplerate, mChannels);
    int nParams = f.getParamCount();
    for (int i = 0; i < nParams; i++)
      ret.push_back(f.getParamName(i));
  } break;
  default:
    break;
  }

  return ret;
}

CaptureErrors Filters::addFilter(RecorderFilterType filterType) {
  // Check if the new filter is already here.
  // Only one kind of filter allowed.
  if (isFilterActive(filterType) >= 0)
    return CaptureErrors::filterAlreadyAdded;

  std::unique_ptr<GenericFilter> newFilter;
  switch (filterType) {
  case autogain:
    newFilter = std::make_unique<AutoGain>(mSamplerate);
    break;
  case echoCancellation:
    newFilter = std::make_unique<EchoCancellation>(mSamplerate);
    break;
  case adaptiveEchoCancellation:
    newFilter =
        std::make_unique<AdaptiveEchoCancellation>(mSamplerate, mChannels);
    break;
  default:
    return CaptureErrors::filterNotFound;
  }

  std::unique_ptr<FilterObject> nfo =
      std::make_unique<FilterObject>(filterType, std::move(newFilter));
  /// In [filters] we add the new filter to the list. All these filters must be
  /// processed inside the callback.
  filters.push_back(std::move(nfo));

  aecLog("[Filters] Added filter type %d, now have %zu filters\n",
         static_cast<int>(filterType), filters.size());

  return CaptureErrors::captureNoError;
}

CaptureErrors Filters::removeFilter(RecorderFilterType filterType) {
  int index = isFilterActive(filterType);
  if (index < 0)
    return CaptureErrors::filterNotFound;

  filters[index].get()->filter.reset();

  /// remove the filter from the list
  filters.erase(filters.begin() + index);

  return CaptureErrors::captureNoError;
}

void Filters::setFilterParams(RecorderFilterType filterType, int attributeId,
                              float value) {
  int index = isFilterActive(filterType);
  if (index < 0)
    return;
  filters[index].get()->filter.get()->setParamValue(attributeId, value);
}

float Filters::getFilterParams(RecorderFilterType filterType, int attributeId) {
  int index = isFilterActive(filterType);
  // If not active return its default value
  if (index < 0) {
    switch (filterType) {
    case autogain:
      return AutoGain(0).getParamDef(attributeId);
    case echoCancellation:
      return EchoCancellation(0).getParamDef(attributeId);
    case adaptiveEchoCancellation:
      return AdaptiveEchoCancellation(mSamplerate, mChannels)
          .getParamDef(attributeId);
    default:
      return 9999.f;
    }
  }

  float ret = filters[index].get()->filter.get()->getParamValue(attributeId);

  return ret;
}

void Filters::setAecImpulseResponse(const float *coeffs, int length) {
  int index = isFilterActive(RecorderFilterType::adaptiveEchoCancellation);
  if (index < 0) {
    printf("[Filters] AEC not active, cannot set impulse response\n");
    return;
  }

  // Cast to AEC filter and call setImpulseResponse
  AdaptiveEchoCancellation *aec = dynamic_cast<AdaptiveEchoCancellation *>(
      filters[index].get()->filter.get());
  if (aec) {
    aec->setImpulseResponse(coeffs, length);
  } else {
    printf("[Filters] Failed to cast to AEC filter\n");
  }
}

AecStats Filters::getAecStats() {
  AecStats zero = {0};
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return zero;

  // Cast and call
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getStats();
}

// VSS-NLMS parameter control
void Filters::setAecVssMuMax(float mu) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->setVssMuMax(mu);
}

void Filters::setAecVssLeakage(float lambda) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->setVssLeakage(lambda);
}

void Filters::setAecVssAlpha(float alpha) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->setVssAlpha(alpha);
}

float Filters::getAecVssMuMax() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return 0.5f;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getVssMuMax();
}

float Filters::getAecVssLeakage() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return 1.0f;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getVssLeakage();
}

float Filters::getAecVssAlpha() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return 0.95f;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getVssAlpha();
}

// Filter length control
void Filters::setAecFilterLength(int length) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->setFilterLength(length);
}

int Filters::getAecFilterLength() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return 8192; // Default
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getFilterLength();
}

// Sample-accurate AEC synchronization
void Filters::setAecCaptureFrameCount(size_t captureFrameCount) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->setCaptureFrameCount(captureFrameCount);
}

void Filters::setAecCalibratedOffset(int64_t offset) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->setCalibratedOffset(offset);
}

int64_t Filters::getAecCalibratedOffset() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return 0;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getCalibratedOffset();
}

void Filters::startAecCalibrationCapture(size_t maxSamples) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0) {
    aecLog("[Filters] AEC not active, cannot start calibration capture\n");
    return;
  }
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->startCalibrationCapture(maxSamples);
}

void Filters::stopAecCalibrationCapture() {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0) {
    aecLog("[Filters] AEC not active, cannot stop calibration capture\n");
    return;
  }
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->stopCalibrationCapture();
}

// Static empty vectors for when AEC is not active
static std::vector<float> sEmptyVector;

const std::vector<float> &Filters::getAecAlignedRef() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return sEmptyVector;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getAlignedRef();
}

const std::vector<float> &Filters::getAecAlignedMic() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return sEmptyVector;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getAlignedMic();
}

void Filters::setAecMode(AecMode mode) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  aec->setAecMode(mode);
}

AecMode Filters::getAecMode() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return aecModeHybrid;
  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  return aec->getAecMode();
}

// Neural Model Control
bool Filters::loadNeuralModel(NeuralModelType modelType,
                               const std::string &assetBasePath) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return false; // AEC filter not active

  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  NeuralPostFilter *neuralFilter = aec->getNeuralFilter();

  if (!neuralFilter)
    return false;

  return neuralFilter->loadModelByType(modelType, assetBasePath);
}

NeuralModelType Filters::getLoadedNeuralModel() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return NeuralModelType::NONE;

  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  NeuralPostFilter *neuralFilter = aec->getNeuralFilter();

  if (!neuralFilter)
    return NeuralModelType::NONE;

  return neuralFilter->getLoadedModelType();
}

void Filters::setNeuralEnabled(bool enabled) {
  int idx = isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return;

  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  NeuralPostFilter *neuralFilter = aec->getNeuralFilter();

  if (neuralFilter) {
    neuralFilter->setEnabled(enabled);
  }
}

bool Filters::isNeuralEnabled() const {
  int idx =
      const_cast<Filters *>(this)->isFilterActive(adaptiveEchoCancellation);
  if (idx < 0)
    return false;

  AdaptiveEchoCancellation *aec =
      static_cast<AdaptiveEchoCancellation *>(filters[idx].get()->filter.get());
  NeuralPostFilter *neuralFilter = aec->getNeuralFilter();

  if (!neuralFilter)
    return false;

  return neuralFilter->isEnabled();
}
