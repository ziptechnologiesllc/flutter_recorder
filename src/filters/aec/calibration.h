#ifndef AEC_CALIBRATION_H
#define AEC_CALIBRATION_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>

/**
 * AEC Calibration System
 *
 * Measures the acoustic delay between speaker output and microphone input
 * using cross-correlation of calibration signals (white noise + sine sweep).
 *
 * Calibration sequence:
 * 1. Generate calibration signal (white noise burst + sine sweep)
 * 2. Play through speakers while recording from microphone
 * 3. Cross-correlate reference (playback) with recorded (mic) signal
 * 4. Find peak correlation = optimal delay
 * 5. Estimate echo gain at that delay
 */

struct CalibrationResult {
    int delaySamples;      // Optimal delay in samples
    int delayMs;           // Optimal delay in milliseconds
    float echoGain;        // Echo attenuation factor (0-1)
    float correlation;     // Peak correlation coefficient (quality metric)
    bool success;          // Whether calibration succeeded
    std::vector<float> impulseResponse;  // NLMS initial coefficients from FFT deconvolution
};

class AECCalibration {
public:
    static constexpr int WHITE_NOISE_DURATION_MS = 1500;   // 1.5 seconds of white noise
    static constexpr int MAX_DELAY_SEARCH_MS = 150;  // Search up to 150ms delay
    static constexpr float SIGNAL_AMPLITUDE = 0.3f;  // Calibration signal level
    static constexpr float MIN_CORRELATION_THRESHOLD = 0.2f;  // Minimum correlation for valid result

    /**
     * Generate calibration WAV file in memory.
     * Contains: 500ms white noise + 1000ms logarithmic sine sweep
     *
     * @param sampleRate Sample rate in Hz
     * @param channels Number of channels (1 or 2)
     * @param outSize Output: size of returned buffer in bytes
     * @return Pointer to WAV data (caller must free with delete[])
     */
    static uint8_t* generateCalibrationWav(
        unsigned int sampleRate,
        unsigned int channels,
        size_t* outSize);

    /**
     * Capture reference and mic signals for analysis.
     * Call this AFTER calibration audio has finished playing.
     *
     * @param referenceBuffer Data from AEC reference buffer (clean playback)
     * @param referenceLen Length in samples
     * @param micBuffer Data from microphone ring buffer
     * @param micLen Length in samples
     */
    static void captureSignals(
        const float* referenceBuffer,
        size_t referenceLen,
        const float* micBuffer,
        size_t micLen);

    /**
     * Run cross-correlation analysis on captured signals.
     *
     * @param sampleRate Sample rate for converting delay to milliseconds
     * @return Calibration results including optimal delay
     */
    static CalibrationResult analyze(unsigned int sampleRate);

    /**
     * Clear captured signal buffers.
     */
    static void reset();

    /**
     * Get captured reference signal for visualization.
     * @param dest Destination buffer
     * @param maxLength Maximum samples to copy
     * @return Number of samples copied
     */
    static int getRefSignal(float* dest, int maxLength);

    /**
     * Get captured mic signal for visualization.
     * @param dest Destination buffer
     * @param maxLength Maximum samples to copy
     * @return Number of samples copied
     */
    static int getMicSignal(float* dest, int maxLength);

private:
    /**
     * Generate white noise samples.
     */
    static void generateWhiteNoise(
        float* buffer,
        size_t samples,
        float amplitude);

    /**
     * Generate logarithmic sine sweep from startFreq to endFreq.
     */
    static void generateSineSweep(
        float* buffer,
        size_t samples,
        unsigned int sampleRate,
        float startFreq,
        float endFreq,
        float amplitude);

    /**
     * Find optimal delay using normalized cross-correlation.
     * Returns delay in samples where correlation is maximum.
     */
    static int findOptimalDelay(
        const float* reference,
        size_t refLen,
        const float* recorded,
        size_t recLen,
        int maxDelaySamples,
        float* outCorrelation);

    /**
     * Estimate echo gain at given delay.
     * Returns ratio of recorded amplitude to reference amplitude.
     */
    static float estimateEchoGain(
        const float* reference,
        const float* recorded,
        size_t len,
        int delay);

    /**
     * Write WAV header to buffer.
     */
    static void writeWavHeader(
        uint8_t* buffer,
        unsigned int sampleRate,
        unsigned int channels,
        size_t numSamples);

    /**
     * Compute impulse response via FFT deconvolution.
     * H(f) = FFT(mic) / FFT(ref), then h(t) = IFFT(H(f))
     *
     * @param reference Reference signal (calibration audio)
     * @param mic Recorded mic signal
     * @param len Length of signals
     * @param filterLength Desired impulse response length
     * @return Impulse response coefficients
     */
    static std::vector<float> computeImpulseResponse(
        const float* reference,
        const float* mic,
        size_t len,
        int filterLength);

    /**
     * Find next power of 2 >= n.
     */
    static size_t nextPowerOf2(size_t n);

    // Internal buffers for captured signals
    static std::vector<float> sRefCapture;
    static std::vector<float> sMicCapture;

    // Generated calibration signal (stored for use as reference)
    static std::vector<float> sGeneratedSignal;
};

#endif // AEC_CALIBRATION_H
