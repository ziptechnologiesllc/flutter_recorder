#ifndef AEC_CALIBRATION_H
#define AEC_CALIBRATION_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>

// Calibration signal type selection
enum class CalibrationSignalType {
    Chirp = 0,  // Logarithmic sine sweep (current default)
    Click = 1   // Impulse train (5 clicks for IR averaging)
};

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
    float delayMs;         // Optimal delay in milliseconds (float for sub-ms precision)
    float echoGain;        // Echo attenuation factor (0-1)
    float correlation;     // Peak correlation coefficient (quality metric)
    bool success;          // Whether calibration succeeded
    std::vector<float> impulseResponse;  // NLMS initial coefficients from FFT deconvolution
    int64_t calibratedOffset;  // Sample-accurate sync: captureFrame - offset = outputFrame
};

class AECCalibration {
public:
    // Click-based calibration: 5 clicks averaged for noise reduction
    static constexpr int CLICK_COUNT = 5;              // Number of clicks to average
    static constexpr int CLICK_SAMPLES = 48;           // Samples per click (~1ms @ 48kHz, audible pulse)
    static constexpr int CLICK_SPACING_MS = 600;       // 600ms between clicks for full IR capture
    static constexpr int IR_LENGTH = 2048;             // Impulse response taps (~42ms @ 48kHz)
    static constexpr float CLICK_AMPLITUDE = 1.0f;     // Full scale for max SNR
    static constexpr int TAIL_MS = 400;                // Silence after last click for full decay
    static constexpr float MIN_PEAK_THRESHOLD = 0.005f; // Lower threshold for detection

    /**
     * Generate calibration WAV file in memory.
     *
     * Chirp: Logarithmic sine sweep 20Hz-Nyquist, ~0.6s duration
     * Click: 5 impulses spaced 600ms apart + 400ms tail, ~3s duration
     *
     * @param sampleRate Sample rate in Hz
     * @param channels Number of channels (1 or 2)
     * @param outSize Output: size of returned buffer in bytes
     * @param signalType Signal type (Chirp or Click)
     * @return Pointer to WAV data (caller must free with delete[])
     */
    static uint8_t* generateCalibrationWav(
        unsigned int sampleRate,
        unsigned int channels,
        size_t* outSize,
        CalibrationSignalType signalType = CalibrationSignalType::Chirp);

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
     * Run analysis using frame-aligned ref/mic buffers from AEC processAudio.
     * These buffers are captured from the same callback at the same time,
     * providing perfect frame alignment for accurate delay estimation.
     *
     * @param alignedRef Reference signal from AEC processAudio (already aligned)
     * @param alignedMic Mic signal from AEC processAudio (already aligned)
     * @param sampleRate Sample rate for converting delay to milliseconds
     * @param signalType Signal type used for calibration (affects analysis method)
     * @return Calibration results including optimal delay
     */
    static CalibrationResult analyzeAligned(
        const std::vector<float>& alignedRef,
        const std::vector<float>& alignedMic,
        unsigned int sampleRate,
        CalibrationSignalType signalType = CalibrationSignalType::Chirp);

    /**
     * Clear captured signal buffers.
     */
    static void reset();

    /**
     * Record frame counters at calibration start.
     * Call this when the calibration signal starts playing.
     *
     * @param outputFrames Current output frame count from reference buffer
     * @param captureFrames Current capture frame count from Capture
     */
    static void recordFrameCountersAtStart(size_t outputFrames, size_t captureFrames);

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
     * Write WAV header to buffer.
     */
    static void writeWavHeader(
        uint8_t* buffer,
        unsigned int sampleRate,
        unsigned int channels,
        size_t numSamples);

    // Internal buffers for captured signals
    static std::vector<float> sRefCapture;
    static std::vector<float> sMicCapture;

    // Generated calibration signal (stored for use as reference)
    static std::vector<float> sGeneratedSignal;

    // Signal type used for last generation
    static CalibrationSignalType sLastSignalType;

    // Frame counters recorded at calibration start
    static size_t sOutputFramesAtStart;
    static size_t sCaptureFramesAtStart;
};

#endif // AEC_CALIBRATION_H
