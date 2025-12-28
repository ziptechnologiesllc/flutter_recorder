#include "filters.h"
#include "autogain.h"
#include "echo_cancellation.h"
#include "aec/adaptive_echo_cancellation.h"

#include <vector>
#include <string>
#include <memory>

Filters::Filters(unsigned int samplerate, unsigned int channels)
    : mSamplerate(samplerate), mChannels(channels) {}

Filters::~Filters() {}

int Filters::isFilterActive(RecorderFilterType filter)
{
    for (int i = 0; i < filters.size(); i++)
    {
        if (filters[i].get()->type == filter)
            return i;
    }
    return -1;
}

std::vector<std::string> Filters::getFilterParamNames(RecorderFilterType filterType)
{
    std::vector<std::string> ret;
    switch (filterType)
    {
    case RecorderFilterType::autogain:
    {
        AutoGain f;
        int nParams = f.getParamCount();
        for (int i = 0; i < nParams; i++)
            ret.push_back(f.getParamName(i));
    }
    break;
    case RecorderFilterType::echoCancellation:
    {
        EchoCancellation f;
        int nParams = f.getParamCount();
        for (int i = 0; i < nParams; i++)
            ret.push_back(f.getParamName(i));
    }
    break;
    case RecorderFilterType::adaptiveEchoCancellation:
    {
        AdaptiveEchoCancellation f(mSamplerate, mChannels);
        int nParams = f.getParamCount();
        for (int i = 0; i < nParams; i++)
            ret.push_back(f.getParamName(i));
    }
    break;
    default:
        break;
    }

    return ret;
}

CaptureErrors Filters::addFilter(RecorderFilterType filterType)
{
    // Check if the new filter is already here.
    // Only one kind of filter allowed.
    if (isFilterActive(filterType) >= 0)
        return CaptureErrors::filterAlreadyAdded;

    std::unique_ptr<GenericFilter> newFilter;
    switch (filterType)
    {
    case autogain:
        newFilter = std::make_unique<AutoGain>(mSamplerate);
        break;
    case echoCancellation:
        newFilter = std::make_unique<EchoCancellation>(mSamplerate);
        break;
    case adaptiveEchoCancellation:
        newFilter = std::make_unique<AdaptiveEchoCancellation>(mSamplerate, mChannels);
        break;
    default:
        return CaptureErrors::filterNotFound;
    }

    std::unique_ptr<FilterObject> nfo = std::make_unique<FilterObject>(filterType, std::move(newFilter));
    /// In [filters] we add the new filter to the list. All these filters must be processed inside the callback.
    filters.push_back(std::move(nfo));

    return CaptureErrors::captureNoError;
}

CaptureErrors Filters::removeFilter(RecorderFilterType filterType)
{
    int index = isFilterActive(filterType);
    if (index < 0)
        return CaptureErrors::filterNotFound;

    filters[index].get()->filter.reset();

    /// remove the filter from the list
    filters.erase(filters.begin() + index);

    return CaptureErrors::captureNoError;
}

void Filters::setFilterParams(RecorderFilterType filterType, int attributeId, float value)
{
    int index = isFilterActive(filterType);
    if (index < 0)
        return;
    filters[index].get()->filter.get()->setParamValue(attributeId, value);
}

float Filters::getFilterParams(RecorderFilterType filterType, int attributeId)
{
    int index = isFilterActive(filterType);
    // If not active return its default value
    if (index < 0) {
        switch (filterType)
        {
        case autogain:
            return AutoGain(0).getParamDef(attributeId);
        case echoCancellation:
            return EchoCancellation(0).getParamDef(attributeId);
        case adaptiveEchoCancellation:
            return AdaptiveEchoCancellation(mSamplerate, mChannels).getParamDef(attributeId);
        default:
            return 9999.f;
        }
    }

    float ret = filters[index].get()->filter.get()->getParamValue(attributeId);

    return ret;
}

void Filters::setAecImpulseResponse(const float* coeffs, int length) {
    int index = isFilterActive(RecorderFilterType::adaptiveEchoCancellation);
    if (index < 0) {
        printf("[Filters] AEC not active, cannot set impulse response\n");
        return;
    }

    // Cast to AEC filter and call setImpulseResponse
    AdaptiveEchoCancellation* aec =
        dynamic_cast<AdaptiveEchoCancellation*>(filters[index].get()->filter.get());
    if (aec) {
        aec->setImpulseResponse(coeffs, length);
    } else {
        printf("[Filters] Failed to cast to AEC filter\n");
    }
}
