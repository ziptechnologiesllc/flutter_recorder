#ifndef AEC_TEST_H
#define AEC_TEST_H

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * AEC Test System
 *
 * Automated testing for echo cancellation quality.
 * Plays a test signal through speakers while recording with AEC enabled,
 * then analyzes how well the echo was cancelled.
 *
 * Test sequence:
 * 1. Start test capture (captures raw mic + AEC-cancelled output)
 * 2. Play test audio through speakers (handled by caller via SoLoud)
 * 3. Stop test capture
 * 4. Analyze: compare raw mic vs cancelled output
 * 5. Report metrics (cancellation dB, correlation, pass/fail)
 */

struct AecTestResult {
    float micEnergyDb;          // Energy of raw mic signal (before AEC)
    float cancelledEnergyDb;    // Energy of cancelled signal (after AEC)
    float cancellationDb;       // How much was cancelled (mic - cancelled)
    float correlationBefore;    // Correlation of ref with raw mic
    float correlationAfter;     // Correlation of ref with cancelled (should be ~0)
    float peakReductionDb;      // Peak amplitude reduction
    bool passed;                // cancellationDb > threshold
};

class AECTest {
public:
    // Threshold for pass/fail (dB of cancellation required)
    static constexpr float PASS_THRESHOLD_DB = 6.0f;

    /**
     * Start capturing test signals.
     * Call this BEFORE playing the test audio.
     *
     * Captures:
     * - Reference signal (from AEC reference buffer)
     * - Raw mic signal (before AEC filter)
     * - Cancelled signal (after AEC filter)
     *
     * @param maxSamples Maximum samples to capture per signal
     */
    static void startTestCapture(size_t maxSamples);

    /**
     * Stop capturing test signals.
     * Call this AFTER test audio has finished playing.
     */
    static void stopTestCapture();

    /**
     * Check if test capture is currently active.
     */
    static bool isCapturing();

    /**
     * Called by AEC filter during processing to capture samples.
     * This captures BOTH the raw mic input and the cancelled output.
     *
     * @param micSample Raw microphone sample (before cancellation)
     * @param cancelledSample Output after echo cancellation
     * @param refSample Reference sample used for cancellation
     */
    static void captureSample(float micSample, float cancelledSample, float refSample);

    /**
     * Run analysis on captured signals.
     * Computes cancellation metrics and determines pass/fail.
     *
     * @param sampleRate Sample rate for any time-based calculations
     * @return Test results with all metrics
     */
    static AecTestResult analyze(unsigned int sampleRate);

    /**
     * Get captured reference signal for visualization.
     * @param dest Destination buffer
     * @param maxLength Maximum samples to copy
     * @return Number of samples copied
     */
    static int getRefSignal(float* dest, int maxLength);

    /**
     * Get captured raw mic signal (before AEC) for visualization.
     * @param dest Destination buffer
     * @param maxLength Maximum samples to copy
     * @return Number of samples copied
     */
    static int getMicSignal(float* dest, int maxLength);

    /**
     * Get captured cancelled signal (after AEC) for visualization.
     * @param dest Destination buffer
     * @param maxLength Maximum samples to copy
     * @return Number of samples copied
     */
    static int getCancelledSignal(float* dest, int maxLength);

    /**
     * Reset/clear all captured test data.
     */
    static void reset();

private:
    // Captured signals
    static std::vector<float> sRefCapture;
    static std::vector<float> sMicCapture;
    static std::vector<float> sCancelledCapture;

    // Capture state
    static bool sIsCapturing;
    static size_t sMaxSamples;

    // Helper functions
    static float computeRMS(const std::vector<float>& signal);
    static float computePeak(const std::vector<float>& signal);
    static float computeCorrelation(const std::vector<float>& a, const std::vector<float>& b);
};

#endif // AEC_TEST_H
