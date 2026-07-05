#include "aec_test.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// External logging function defined in calibration.cpp
extern void aecLog(const char *fmt, ...);

// Static member initialization
std::vector<float> AECTest::sRefCapture;
std::vector<float> AECTest::sMicCapture;
std::vector<float> AECTest::sCancelledCapture;
bool AECTest::sIsCapturing = false;
size_t AECTest::sMaxSamples = 0;

void AECTest::startTestCapture(size_t maxSamples) {
    // Clear previous data
    sRefCapture.clear();
    sMicCapture.clear();
    sCancelledCapture.clear();

    // Reserve space
    sRefCapture.reserve(maxSamples);
    sMicCapture.reserve(maxSamples);
    sCancelledCapture.reserve(maxSamples);

    sMaxSamples = maxSamples;
    sIsCapturing = true;

    aecLog("[AEC TEST] Started test capture (max %zu samples)\n", maxSamples);
}

void AECTest::stopTestCapture() {
    sIsCapturing = false;
    aecLog("[AEC TEST] Stopped test capture. Captured: ref=%zu, mic=%zu, cancelled=%zu samples\n",
           sRefCapture.size(), sMicCapture.size(), sCancelledCapture.size());
}

bool AECTest::isCapturing() {
    return sIsCapturing;
}

void AECTest::captureSample(float micSample, float cancelledSample, float refSample) {
    if (!sIsCapturing) return;
    if (sMicCapture.size() >= sMaxSamples) return;

    sRefCapture.push_back(refSample);
    sMicCapture.push_back(micSample);
    sCancelledCapture.push_back(cancelledSample);

    // Debug: print energy every ~48000 samples (1 second at 48kHz)
    static float refEnergy = 0, micEnergy = 0;
    refEnergy += refSample * refSample;
    micEnergy += micSample * micSample;
    if (sMicCapture.size() % 48000 == 0) {
        float refRms = std::sqrt(refEnergy / 48000.0f);
        float micRms = std::sqrt(micEnergy / 48000.0f);
        aecLog("[AEC TEST] After %zu samples: refRMS=%.4f (%.1fdB), micRMS=%.4f (%.1fdB)\n",
                sMicCapture.size(), refRms,
                refRms > 1e-10f ? 20*std::log10(refRms) : -100.0f,
                micRms,
                micRms > 1e-10f ? 20*std::log10(micRms) : -100.0f);
        refEnergy = 0;
        micEnergy = 0;
    }
}

float AECTest::computeRMS(const std::vector<float>& signal) {
    if (signal.empty()) return 0.0f;

    double sum = 0.0;
    for (float s : signal) {
        sum += s * s;
    }
    return static_cast<float>(std::sqrt(sum / signal.size()));
}

float AECTest::computePeak(const std::vector<float>& signal) {
    float peak = 0.0f;
    for (float s : signal) {
        if (std::abs(s) > peak) peak = std::abs(s);
    }
    return peak;
}

float AECTest::computeCorrelation(const std::vector<float>& a, const std::vector<float>& b) {
    size_t len = std::min(a.size(), b.size());
    if (len == 0) return 0.0f;

    // Compute means
    double meanA = 0, meanB = 0;
    for (size_t i = 0; i < len; i++) {
        meanA += a[i];
        meanB += b[i];
    }
    meanA /= len;
    meanB /= len;

    // Compute correlation coefficient
    double sumAB = 0, sumA2 = 0, sumB2 = 0;
    for (size_t i = 0; i < len; i++) {
        double da = a[i] - meanA;
        double db = b[i] - meanB;
        sumAB += da * db;
        sumA2 += da * da;
        sumB2 += db * db;
    }

    double denom = std::sqrt(sumA2 * sumB2);
    if (denom < 1e-10) return 0.0f;

    return static_cast<float>(sumAB / denom);
}

AecTestResult AECTest::analyze(unsigned int sampleRate) {
    AecTestResult result = {};

    aecLog("\n========== AEC TEST RESULTS ==========\n");

    // Check if we have data
    aecLog("[AEC TEST] Captured samples: ref=%zu, mic=%zu, cancelled=%zu\n",
            sRefCapture.size(), sMicCapture.size(), sCancelledCapture.size());

    if (sMicCapture.empty() || sCancelledCapture.empty()) {
        aecLog("[AEC TEST] ERROR: No captured data to analyze\n");
        aecLog("==========================================\n\n");
        result.passed = false;
        return result;
    }

    size_t numSamples = std::min(sMicCapture.size(), sCancelledCapture.size());

    // Compute signal stats
    float refRms = computeRMS(sRefCapture);
    float refPeak = computePeak(sRefCapture);
    float micRms = computeRMS(sMicCapture);
    float micPeak = computePeak(sMicCapture);
    float cancelledRms = computeRMS(sCancelledCapture);
    float cancelledPeak = computePeak(sCancelledCapture);

    // Convert to dB
    float refRmsDb = refRms > 1e-10f ? 20.0f * std::log10(refRms) : -100.0f;
    float micRmsDb = micRms > 1e-10f ? 20.0f * std::log10(micRms) : -100.0f;
    float cancelledRmsDb = cancelledRms > 1e-10f ? 20.0f * std::log10(cancelledRms) : -100.0f;

    // Store results
    result.micEnergyDb = micRmsDb;
    result.cancelledEnergyDb = cancelledRmsDb;
    result.cancellationDb = micRmsDb - cancelledRmsDb;  // Positive = good cancellation
    result.peakReductionDb = micPeak > 1e-10f && cancelledPeak > 1e-10f
        ? 20.0f * std::log10(micPeak / cancelledPeak) : 0.0f;

    // Compute correlations
    result.correlationBefore = computeCorrelation(sRefCapture, sMicCapture);
    result.correlationAfter = computeCorrelation(sRefCapture, sCancelledCapture);

    // Determine pass/fail
    result.passed = result.cancellationDb >= PASS_THRESHOLD_DB;

    // Print detailed results
    aecLog("[AEC TEST] Sample rate: %u Hz\n", sampleRate);
    aecLog("[AEC TEST] Samples analyzed: %zu (%.2f sec)\n",
           numSamples, (float)numSamples / sampleRate);
    aecLog("\n");

    aecLog("[AEC TEST] Reference signal:\n");
    aecLog("           samples=%zu, peak=%.4f, RMS=%.4f (%.1f dB)\n",
           sRefCapture.size(), refPeak, refRms, refRmsDb);

    aecLog("[AEC TEST] Raw mic signal (before AEC):\n");
    aecLog("           samples=%zu, peak=%.4f, RMS=%.4f (%.1f dB)\n",
           sMicCapture.size(), micPeak, micRms, micRmsDb);

    aecLog("[AEC TEST] Cancelled signal (after AEC):\n");
    aecLog("           samples=%zu, peak=%.4f, RMS=%.4f (%.1f dB)\n",
           sCancelledCapture.size(), cancelledPeak, cancelledRms, cancelledRmsDb);
    aecLog("\n");

    aecLog("[AEC TEST] Correlation (ref vs raw mic): %.4f\n", result.correlationBefore);
    aecLog("[AEC TEST] Correlation (ref vs cancelled): %.4f\n", result.correlationAfter);
    aecLog("\n");

    aecLog("[AEC TEST] === CANCELLATION METRICS ===\n");
    aecLog("[AEC TEST] Energy reduction: %.2f dB\n", result.cancellationDb);
    aecLog("[AEC TEST] Peak reduction: %.2f dB\n", result.peakReductionDb);
    aecLog("[AEC TEST] Correlation reduction: %.4f -> %.4f (delta: %.4f)\n",
           result.correlationBefore, result.correlationAfter,
           result.correlationBefore - result.correlationAfter);
    aecLog("\n");

    // Sample-level debug - just show first 10 to keep output shorter
    aecLog("[AEC TEST] First 10 sample triplets (ref, mic, cancelled):\n");
    for (int i = 0; i < 10 && i < (int)numSamples; i++) {
        float ref = i < (int)sRefCapture.size() ? sRefCapture[i] : 0;
        float mic = sMicCapture[i];
        float can = sCancelledCapture[i];
        aecLog("  [%2d] ref=%.4f  mic=%.4f  cancelled=%.4f  diff=%.4f\n",
               i, ref, mic, can, mic - can);
    }
    aecLog("\n");

    // Find peak locations
    int micPeakIdx = 0, cancelledPeakIdx = 0;
    for (int i = 0; i < (int)numSamples; i++) {
        if (std::abs(sMicCapture[i]) >= micPeak * 0.99f) micPeakIdx = i;
        if (std::abs(sCancelledCapture[i]) >= cancelledPeak * 0.99f) cancelledPeakIdx = i;
    }
    aecLog("[AEC TEST] Mic peak at sample %d (%.2f ms)\n",
           micPeakIdx, micPeakIdx * 1000.0f / sampleRate);
    aecLog("[AEC TEST] Cancelled peak at sample %d (%.2f ms)\n",
           cancelledPeakIdx, cancelledPeakIdx * 1000.0f / sampleRate);
    aecLog("\n");

    // Pass/fail
    aecLog("[AEC TEST] RESULT: %s\n", result.passed ? "PASS" : "FAIL");
    aecLog("[AEC TEST] (threshold: %.1f dB, achieved: %.2f dB)\n",
           PASS_THRESHOLD_DB, result.cancellationDb);
    aecLog("==========================================\n\n");

    return result;
}

int AECTest::getRefSignal(float* dest, int maxLength) {
    if (dest == nullptr || maxLength <= 0) return 0;
    int copyLen = std::min(maxLength, static_cast<int>(sRefCapture.size()));
    std::memcpy(dest, sRefCapture.data(), copyLen * sizeof(float));
    return copyLen;
}

int AECTest::getMicSignal(float* dest, int maxLength) {
    if (dest == nullptr || maxLength <= 0) return 0;
    int copyLen = std::min(maxLength, static_cast<int>(sMicCapture.size()));
    std::memcpy(dest, sMicCapture.data(), copyLen * sizeof(float));
    return copyLen;
}

int AECTest::getCancelledSignal(float* dest, int maxLength) {
    if (dest == nullptr || maxLength <= 0) return 0;
    int copyLen = std::min(maxLength, static_cast<int>(sCancelledCapture.size()));
    std::memcpy(dest, sCancelledCapture.data(), copyLen * sizeof(float));
    return copyLen;
}

void AECTest::reset() {
    sRefCapture.clear();
    sMicCapture.clear();
    sCancelledCapture.clear();
    sIsCapturing = false;
    sMaxSamples = 0;
    aecLog("[AEC TEST] Reset test data\n");
}
