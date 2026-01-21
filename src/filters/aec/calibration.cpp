#include "calibration.h"
#include "delay_estimator.h"
#include "vss_nlms_filter.h"
#include "../../soloud_slave_bridge.h"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>

#ifdef __ANDROID__
#include <android/log.h>
#define AEC_LOG_TAG "AECCalibration"
#endif

// Log file for calibration debug output (visible to Flutter test)
static std::ofstream sLogFile;
static bool sLogFileOpened = false;

// Store log messages in memory so they can be retrieved via FFI
static std::string sLogBuffer;
static std::mutex sLogBufferMutex;  // Protects sLogBuffer for thread-safety
static const size_t MAX_LOG_SIZE = 64 * 1024; // 64KB max

void aecLog(const char *fmt, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

  // Append to in-memory buffer (truncate if too large)
  {
    std::lock_guard<std::mutex> lock(sLogBufferMutex);
    if (sLogBuffer.size() < MAX_LOG_SIZE) {
      sLogBuffer += buffer;
    }
  }

#ifdef __ANDROID__
  // On Android, use logcat
  __android_log_print(ANDROID_LOG_INFO, AEC_LOG_TAG, "%s", buffer);
#else
  // Also print to stderr
  fprintf(stderr, "%s", buffer);
  fflush(stderr);
#endif
}

// FFI export to retrieve log buffer
// Note: Returns a copy to avoid dangling pointer issues with concurrent access
static std::string sLogBufferCopy;  // Holds copy for FFI return

extern "C" {
const char *flutter_recorder_aec_getCalibrationLog() {
  std::lock_guard<std::mutex> lock(sLogBufferMutex);
  sLogBufferCopy = sLogBuffer;  // Copy under lock
  return sLogBufferCopy.c_str();
}

void flutter_recorder_aec_clearCalibrationLog() {
  std::lock_guard<std::mutex> lock(sLogBufferMutex);
  sLogBuffer.clear();
}
}

// Static member initialization
std::vector<float> AECCalibration::sRefCapture;
std::vector<float> AECCalibration::sMicCapture;
std::vector<float> AECCalibration::sGeneratedSignal;
CalibrationSignalType AECCalibration::sLastSignalType = CalibrationSignalType::Chirp;
size_t AECCalibration::sOutputFramesAtStart = 0;
size_t AECCalibration::sCaptureFramesAtStart = 0;

// Helper: Generate chirp signal (log sweep)
static void generateChirpSignal(std::vector<float>& signal, unsigned int sampleRate) {
  const float DURATION_SEC = 0.5f;
  size_t chirpSamples = static_cast<size_t>(sampleRate * DURATION_SEC);
  size_t paddingFrames = sampleRate / 10;  // 100ms tail
  size_t totalFrames = chirpSamples + paddingFrames;

  signal.resize(totalFrames, 0.0f);

  double f_start = 20.0;
  double f_end = sampleRate / 2.0;
  double k = std::pow(f_end / f_start, 1.0 / chirpSamples);
  const float MAX_AMPLITUDE = 0.5f;

  for (size_t i = 0; i < chirpSamples; ++i) {
    double phase = 2.0 * M_PI * f_start * (std::pow(k, i) - 1.0) /
                   (std::log(k) * sampleRate);
    signal[i] = MAX_AMPLITUDE * std::sin(phase);

    // Fade out last 50 samples to prevent popping
    if (i > chirpSamples - 50) {
      signal[i] *= (chirpSamples - i) / 50.0f;
    }
  }

  aecLog("\n[AEC Calibration] Generated Log Chirp (%.2fs, %zu samples)\n",
         DURATION_SEC, chirpSamples);
}

// Helper: Generate click train signal (impulses for direct IR measurement)
static void generateClickSignal(std::vector<float>& signal, unsigned int sampleRate) {
  // Use constants from AECCalibration class
  size_t spacingSamples = (AECCalibration::CLICK_SPACING_MS * sampleRate) / 1000;
  size_t tailSamples = (AECCalibration::TAIL_MS * sampleRate) / 1000;
  size_t totalSamples = (AECCalibration::CLICK_COUNT - 1) * spacingSamples +
                        AECCalibration::CLICK_SAMPLES + tailSamples;

  signal.resize(totalSamples, 0.0f);

  for (int click = 0; click < AECCalibration::CLICK_COUNT; click++) {
    size_t clickStart = click * spacingSamples;
    for (int i = 0; i < AECCalibration::CLICK_SAMPLES; i++) {
      // Raised cosine pulse to reduce spectral splatter
      float t = (float)i / (AECCalibration::CLICK_SAMPLES - 1);
      float envelope = 0.5f * (1.0f - std::cos(2.0f * M_PI * t));
      signal[clickStart + i] = AECCalibration::CLICK_AMPLITUDE * envelope;
    }
  }

  float totalDurationMs = (float)totalSamples * 1000.0f / sampleRate;
  aecLog("\n[AEC Calibration] Generated Click Train (%d clicks @ %dms spacing, %.1fms total, %zu samples)\n",
         AECCalibration::CLICK_COUNT, AECCalibration::CLICK_SPACING_MS,
         totalDurationMs, totalSamples);
}

// Helper: Find peaks above threshold in signal
static std::vector<size_t> findPeaks(const std::vector<float>& signal,
                                     float threshold,
                                     size_t minSpacing) {
  std::vector<size_t> peaks;

  for (size_t i = 1; i < signal.size() - 1; ++i) {
    float val = std::abs(signal[i]);
    if (val > threshold &&
        val >= std::abs(signal[i - 1]) &&
        val >= std::abs(signal[i + 1])) {
      // Check minimum spacing from last peak
      if (peaks.empty() || (i - peaks.back()) >= minSpacing) {
        peaks.push_back(i);
      } else if (val > std::abs(signal[peaks.back()])) {
        // Replace last peak if this one is larger
        peaks.back() = i;
      }
    }
  }

  return peaks;
}

// Helper: Calculate median of vector
static int medianValue(std::vector<int>& values) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// Click-specific calibration analysis
static CalibrationResult analyzeClickCalibration(
    const std::vector<float>& alignedRef,
    const std::vector<float>& alignedMic,
    unsigned int sampleRate) {

  CalibrationResult result = {0, 0.0f, 0.0f, 0.0f, false, {}};

  aecLog("[AEC Click Calibration] Starting click-based analysis...\n");

  // Find click peaks in reference signal
  // Minimum spacing between clicks: slightly less than CLICK_SPACING_MS
  size_t minSpacing = (AECCalibration::CLICK_SPACING_MS * sampleRate / 1000) * 0.8;
  std::vector<size_t> refPeaks = findPeaks(alignedRef, AECCalibration::MIN_PEAK_THRESHOLD, minSpacing);

  aecLog("[AEC Click Calibration] Found %zu reference peaks\n", refPeaks.size());
  for (size_t i = 0; i < refPeaks.size() && i < 10; ++i) {
    aecLog("  Peak %zu: sample %zu (%.1fms), amplitude %.4f\n",
           i, refPeaks[i], refPeaks[i] * 1000.0f / sampleRate,
           alignedRef[refPeaks[i]]);
  }

  if (refPeaks.empty()) {
    aecLog("[AEC Click Calibration] Error: No reference peaks found\n");
    return result;
  }

  // Find peaks in mic signal with lower threshold (signal is attenuated through room)
  float micThreshold = AECCalibration::MIN_PEAK_THRESHOLD * 0.1f;
  std::vector<size_t> micPeaks = findPeaks(alignedMic, micThreshold, minSpacing);

  aecLog("[AEC Click Calibration] Found %zu mic peaks\n", micPeaks.size());
  for (size_t i = 0; i < micPeaks.size() && i < 10; ++i) {
    aecLog("  Peak %zu: sample %zu (%.1fms), amplitude %.4f\n",
           i, micPeaks[i], micPeaks[i] * 1000.0f / sampleRate,
           alignedMic[micPeaks[i]]);
  }

  // Match ref peaks to mic peaks and calculate delays
  // Store both refPeak and delay for each match
  struct ClickMatch {
    size_t refPeak;
    int delay;
  };
  std::vector<ClickMatch> matches;
  std::vector<int> delays;
  size_t maxDelaySearch = sampleRate / 10;  // 100ms max delay

  for (size_t refPeak : refPeaks) {
    // Find closest mic peak after ref peak
    int bestDelay = -1;
    float bestAmp = 0.0f;

    for (size_t micPeak : micPeaks) {
      if (micPeak > refPeak && (micPeak - refPeak) < maxDelaySearch) {
        float amp = std::abs(alignedMic[micPeak]);
        if (amp > bestAmp) {
          bestAmp = amp;
          bestDelay = micPeak - refPeak;
        }
      }
    }

    if (bestDelay > 0) {
      matches.push_back({refPeak, bestDelay});
      delays.push_back(bestDelay);
      aecLog("[AEC Click Calibration] Matched ref@%zu -> mic delay=%d samples (%.2fms)\n",
             refPeak, bestDelay, bestDelay * 1000.0f / sampleRate);
    }
  }

  int crossCorrDelay = 0;
  bool usedCrossCorrelation = false;

  if (delays.empty()) {
    // Fallback: use cross-correlation
    aecLog("[AEC Click Calibration] No peak matches, falling back to cross-correlation\n");
    crossCorrDelay = DelayEstimator::estimateDelay(alignedRef, alignedMic);
    aecLog("[AEC Click Calibration] Cross-correlation found delay: %d samples (%.2fms)\n",
           crossCorrDelay, crossCorrDelay * 1000.0f / sampleRate);
    // Store the raw delay (without causality offset) for reporting
    // The causality margin is applied internally during IR extraction
    result.delaySamples = std::max(0, crossCorrDelay);
    usedCrossCorrelation = true;
  } else {
    // Use median delay (robust to outliers)
    result.delaySamples = std::max(0, medianValue(delays) - 32);
  }

  result.delayMs = (float)(result.delaySamples * 1000) / (float)sampleRate;

  aecLog("[AEC Click Calibration] Estimated delay: %d samples (%.2fms)\n",
         result.delaySamples, result.delayMs);

  // Extract averaged impulse response from each click
  // Window size: enough to capture room reverb (use IR_LENGTH)
  size_t irLen = AECCalibration::IR_LENGTH;
  std::vector<float> avgIR(irLen, 0.0f);
  int validClicks = 0;

  if (usedCrossCorrelation) {
    // Peak matching failed - extract IR using cross-correlation delay and reference peaks
    aecLog("[AEC Click Calibration] Using cross-correlation delay for IR extraction\n");
    int acousticDelay = crossCorrDelay; // Use the raw delay (includes causality margin)

    for (size_t refPeak : refPeaks) {
      // Extract IR starting from refPeak + acousticDelay - 32 (causality margin)
      int irStartSigned = static_cast<int>(refPeak) + acousticDelay - 32;
      if (irStartSigned < 0) continue;
      size_t irStart = static_cast<size_t>(irStartSigned);
      if (irStart + irLen > alignedMic.size()) continue;

      // Accumulate IR from this click
      for (size_t j = 0; j < irLen; ++j) {
        avgIR[j] += alignedMic[irStart + j];
      }
      validClicks++;
      aecLog("[AEC Click Calibration] Extracted IR at ref@%zu + delay=%d = mic@%zu\n",
             refPeak, acousticDelay, irStart);
    }
  } else {
    // Peak matching succeeded - use matched delays
    int medianDelay = result.delaySamples + 32; // Add back the 32 we subtracted
    int delayTolerance = std::max(48, medianDelay / 10); // 10% or 1ms minimum

    for (const auto& match : matches) {
      // Reject clicks with significantly different delay (outliers)
      if (std::abs(match.delay - medianDelay) > delayTolerance) {
        aecLog("[AEC Click Calibration] Rejecting click at ref@%zu: delay=%d too far from median=%d\n",
               match.refPeak, match.delay, medianDelay);
        continue;
      }

      // Use THIS click's individual delay for IR extraction (more precise)
      size_t irStart = match.refPeak + match.delay - 32; // Subtract causality margin
      if (irStart + irLen > alignedMic.size()) continue;

      // Accumulate IR from this click
      for (size_t j = 0; j < irLen; ++j) {
        avgIR[j] += alignedMic[irStart + j];
      }
      validClicks++;
    }
  }

  if (validClicks > 0) {
    // Average
    float scale = 1.0f / validClicks;
    float energy = 0.0f;
    float maxAmp = 0.0f;
    size_t peakIdx = 0;
    float peakVal = 0.0f;

    for (size_t i = 0; i < irLen; ++i) {
      avgIR[i] *= scale;
      energy += avgIR[i] * avgIR[i];
      if (std::abs(avgIR[i]) > maxAmp) {
        maxAmp = std::abs(avgIR[i]);
        peakVal = avgIR[i];
        peakIdx = i;
      }
    }

    // CRITICAL: Check polarity of IR
    // The IR should have a POSITIVE peak for proper echo cancellation.
    // If peak is negative (inverted microphone or acoustic path), invert the IR.
    // This ensures: y_est = IR * ref produces a positive echo estimate
    // that can be subtracted from the mic signal.
    if (peakVal < 0) {
      aecLog("[AEC Click Calibration] Detected INVERTED IR (peak=%.4f) - correcting polarity\n", peakVal);
      for (size_t i = 0; i < irLen; ++i) {
        avgIR[i] = -avgIR[i];
      }
      peakVal = -peakVal;
    }

    result.impulseResponse = avgIR;
    result.echoGain = std::sqrt(energy);
    result.success = (energy > 1e-6f);

    aecLog("[AEC Click Calibration] Averaged IR from %d clicks: energy=%.4f, peak=%.4f at idx %zu\n",
           validClicks, energy, maxAmp, peakIdx);
  } else {
    aecLog("[AEC Click Calibration] Warning: No valid clicks for IR extraction\n");
    result.success = false;
  }

  aecLog("[AEC Click Calibration] First 10 IR samples: ");
  for (int i = 0; i < 10 && i < (int)result.impulseResponse.size(); ++i)
    aecLog("%.4f ", result.impulseResponse[i]);
  aecLog("\n");

  aecLog("[AEC Click Calibration] Result: delay=%.2fms gain=%.3f success=%d\n",
         result.delayMs, result.echoGain, result.success);

  return result;
}

uint8_t *AECCalibration::generateCalibrationWav(unsigned int sampleRate,
                                                unsigned int channels,
                                                size_t *outSize,
                                                CalibrationSignalType signalType) {
  // Store signal type for analysis
  sLastSignalType = signalType;

  // Generate the appropriate signal type
  switch (signalType) {
    case CalibrationSignalType::Click:
      generateClickSignal(sGeneratedSignal, sampleRate);
      break;
    case CalibrationSignalType::Chirp:
    default:
      generateChirpSignal(sGeneratedSignal, sampleRate);
      break;
  }

  size_t totalFrames = sGeneratedSignal.size();

  // WAV format: 44 byte header + float32 samples
  size_t dataSize = totalFrames * channels * sizeof(float);
  size_t totalSize = 44 + dataSize;

  uint8_t *buffer = new uint8_t[totalSize];

  // Write WAV header
  writeWavHeader(buffer, sampleRate, channels, totalFrames);

  // Generate audio data
  float *audioData = reinterpret_cast<float *>(buffer + 44);
  std::memset(audioData, 0, dataSize);

  // Copy to all channels (interleaved)
  for (size_t frame = 0; frame < totalFrames; ++frame) {
    for (unsigned int ch = 0; ch < channels; ++ch) {
      audioData[frame * channels + ch] = sGeneratedSignal[frame];
    }
  }

  *outSize = totalSize;
  return buffer;
}

void AECCalibration::captureSignals(const float *referenceBuffer,
                                    size_t referenceLen, const float *micBuffer,
                                    size_t micLen) {
  // Use the ACTUAL speaker output from the reference buffer
  // This captures the real transfer function including speaker frequency
  // response, DAC characteristics, and any distortion - critical for accurate
  // deconvolution
  sRefCapture.assign(referenceBuffer, referenceBuffer + referenceLen);
  sMicCapture.assign(micBuffer, micBuffer + micLen);

  aecLog(
      "[AEC Calibration] Captured signals: ref=%zu samples, mic=%zu samples\n",
      referenceLen, micLen);
}

CalibrationResult AECCalibration::analyze(unsigned int sampleRate) {
  CalibrationResult result = {0, 0.0f, 0.0f, 0.0f, false, {}};

  if (sMicCapture.empty() || sRefCapture.empty()) {
    aecLog("[AEC Calibration] analyze: Missing data (ref=%zu, mic=%zu)\n",
           sRefCapture.size(), sMicCapture.size());
    return result;
  }

  aecLog("[AEC Calibration] Warm Start Analysis (Chirp method)...\n");

  // Step 0: Pre-align signals using frame counters
  // In SLAVE MODE: ref and mic are captured in the same audio callback,
  // so they're already perfectly aligned - skip pre-alignment.
  // In NON-SLAVE MODE: The ref and mic signals are captured by independent
  // threads with no synchronization. Use the recorded frame counters to align.
  std::vector<float> preAlignedRef, preAlignedMic;
  bool inSlaveMode = soloud_isSlaveMode();
  int64_t counterDiff = 0;  // Used for calibratedOffset calculation at end

  if (inSlaveMode) {
    // Slave mode: signals are already aligned from the same callback
    // counterDiff stays 0 - no timing offset between ref and mic
    preAlignedRef = sRefCapture;
    preAlignedMic = sMicCapture;
    aecLog("[AEC Calibration] Slave mode: signals already aligned (same callback)\n");
  } else {
    // Non-slave mode: need to pre-align using frame counters
    counterDiff = static_cast<int64_t>(sCaptureFramesAtStart) -
                  static_cast<int64_t>(sOutputFramesAtStart);

    aecLog("[AEC Calibration] Frame counter diff: capture=%zu - output=%zu = %lld\n",
           sCaptureFramesAtStart, sOutputFramesAtStart, (long long)counterDiff);

    if (counterDiff >= 0) {
      // Capture started after output -> mic is "ahead" in time
      // Trim mic start to align with ref start
      size_t offset = std::min(static_cast<size_t>(counterDiff), sMicCapture.size());
      preAlignedMic.assign(sMicCapture.begin() + offset, sMicCapture.end());
      preAlignedRef = sRefCapture;
      aecLog("[AEC Calibration] Pre-aligned: trimmed %zu samples from mic start\n", offset);
    } else {
      // Output started after capture -> ref is "ahead" in time
      // Trim ref start to align with mic start
      size_t offset = std::min(static_cast<size_t>(-counterDiff), sRefCapture.size());
      preAlignedRef.assign(sRefCapture.begin() + offset, sRefCapture.end());
      preAlignedMic = sMicCapture;
      aecLog("[AEC Calibration] Pre-aligned: trimmed %zu samples from ref start\n", offset);
    }
  }

  // Ensure we have enough data after alignment
  if (preAlignedRef.size() < 1024 || preAlignedMic.size() < 1024) {
    aecLog("[AEC Calibration] Error: Not enough aligned data (ref=%zu, mic=%zu)\n",
           preAlignedRef.size(), preAlignedMic.size());
    return result;
  }

  // Step 1: Alignment using DelayEstimator on pre-aligned signals
  // This now finds the ACOUSTIC delay only, not the thread timing offset
  int estimatedDelay = DelayEstimator::estimateDelay(preAlignedRef, preAlignedMic);

  // Apply offset to place peak inside filter (causality margin)
  // Shift Mic BACK relative to Ref, so delay increases effectively?
  // No, if we subtract from delay, we consume LESS of the Mic start.
  // Mic[delay] aligns with Ref[0].
  // Mic[delay - 32] aligns with Ref[0].
  // So Ref[0] sees "early" Mic.
  // Then Ref[32] sees Mic[delay].
  // So Peak is at index 32. Correct.
  int offset = 32;
  estimatedDelay -= offset;

  result.delaySamples = estimatedDelay;
  if (result.delaySamples < 0)
    result.delaySamples = 0;
  result.delayMs = (float)(result.delaySamples * 1000) / (float)sampleRate;

  aecLog(
      "[AEC Calibration] Alignment Delay: %d samples (%.2fms) [Offset -%d]\n",
      result.delaySamples, result.delayMs, offset);

  // Step 2: Prepare Aligned Buffers for Training
  // Use the pre-aligned signals and apply the acoustic delay offset
  // Overlap length = min(Ref.size, Mic.size - delay)
  size_t trainingLen = 0;
  if (estimatedDelay >= 0 && estimatedDelay < (int)preAlignedMic.size()) {
    trainingLen =
        std::min(preAlignedRef.size(), preAlignedMic.size() - estimatedDelay);
  }

  // Limit training to Chirp Duration + Reverb Tail to avoid training on silence
  // Chirp is 0.5s (~24000 samples @ 48k). Add ~200ms reverb tail.
  size_t chirpLen = (size_t)(0.5 * sampleRate);
  size_t maxTrain = chirpLen + 9600; // +200ms reverb tail @ 48kHz
  if (trainingLen > maxTrain) {
    trainingLen = maxTrain;
  }

  if (trainingLen < 1024) {
    aecLog("[AEC Calibration] Error: Insufficient overlap for training (%zu)\n",
           trainingLen);
    return result;
  }

  std::vector<float> alignedRef(trainingLen);
  std::vector<float> alignedMic(trainingLen);

  // Copy from pre-aligned ref (start at 0)
  std::memcpy(alignedRef.data(), preAlignedRef.data(),
              trainingLen * sizeof(float));

  // Copy from pre-aligned mic (start at acoustic delay)
  std::memcpy(alignedMic.data(), preAlignedMic.data() + estimatedDelay,
              trainingLen * sizeof(float));

  // Step 3: Offline Training
  VssNlmsFilter trainer(
      2048); // Use longer filter for calibration to cover tail

  aecLog("[AEC Calibration] Training filter on %zu samples...\n", trainingLen);
  // trainer.warmStartWeights(alignedRef, alignedMic);

  // Manual Warm Start with Logging
  trainer.setStepSize(1.0f);        // Aggressive mu
  trainer.setSmoothingFactor(0.9f); // Fast smoothing
  trainer.setLeakage(1.0f);         // No leakage during warm start!

  // Debug: Check signal levels
  float maxRef = 0.0f, maxMic = 0.0f;
  for (float v : alignedRef)
    maxRef = std::max(maxRef, std::abs(v));
  for (float v : alignedMic)
    maxMic = std::max(maxMic, std::abs(v));
  aecLog("[AEC Calibration] Signal Max Levels: Ref=%.4f Mic=%.4f\n", maxRef,
         maxMic);

  for (size_t i = 0; i < trainingLen; ++i) {
    trainer.processSample(alignedRef[i], alignedMic[i]);

    if (i < 10 || (i % 5000) == 0) {
      aecLog("[VSS] i=%zu x=%.4f d=%.4f e=%.4f mu=%.6f wEnerg=%.6f\n", i,
             alignedRef[i], alignedMic[i], trainer.getLastError(),
             trainer.getLastStepSize(), trainer.getCoeffEnergy());
    }
  }

  // Restore (optional, as trainer is temporary)
  // Restore (optional, as trainer is temporary)
  trainer.setStepSize(0.5f);
  trainer.setSmoothingFactor(0.95f);
  trainer.setLeakage(1.0f);

  // Step 4: Extract Weights
  std::vector<float> weights = trainer.getWeights();

  // Store as result IR
  // Resize if needed (IR_LENGTH is usually 2048 in struct? Check calibration.h)
  // Assuming IR_LENGTH matching weight count.
  size_t weightCount = weights.size();
  result.impulseResponse.resize(IR_LENGTH, 0.0f);

  // Copy weights
  float energy = 0.0f;
  float peak = 0.0f;

  for (size_t i = 0; i < std::min((size_t)IR_LENGTH, weightCount); ++i) {
    result.impulseResponse[i] = weights[i];
    energy += weights[i] * weights[i];
    if (std::abs(weights[i]) > peak)
      peak = std::abs(weights[i]);
  }

  // Metrics
  result.echoGain = std::sqrt(energy); // Rough gain estimate
  // Use the LAST correlation value? No, that might be zero at end of silence.
  // Ideally we track max correlation, but for now let's use the last known
  // correlation from the active part. Actually, let's just use
  // trainer.getLastCorrelation() but be aware it might be low.
  result.correlation = trainer.getLastCorrelation();

  // Sanity check: If energy is super low, maybe failed?
  if (energy < 1e-6f) {
    aecLog("[AEC Calibration] Warning: Low energy weights. Training might have "
           "failed.\n");
    result.success = false;
  } else {
    result.success = true;
  }

  // Debug Print Weights
  aecLog("[AEC Calibration] First 10 weights: ");
  for (int i = 0; i < 10; ++i)
    aecLog("%.4f ", result.impulseResponse[i]);
  aecLog("\n");

  aecLog(
      "[AEC Calibration] Result: delay=%.2fms gain=%.3f (energy) corr=%.4f\n",
      result.delayMs, result.echoGain, result.correlation);

  // Calculate the calibrated offset for position-based sync
  // Formula: captureFrame - offset = corresponding outputFrame
  // offset = (captureAtStart - outputAtStart) + acousticDelay
  // Note: counterDiff was already calculated at the start of this function
  result.calibratedOffset = counterDiff + result.delaySamples;

  aecLog("[AEC Calibration] Position sync offset: capStart=%zu outStart=%zu "
         "diff=%lld acoustic=%d => offset=%lld\n",
         sCaptureFramesAtStart, sOutputFramesAtStart, (long long)counterDiff,
         result.delaySamples, (long long)result.calibratedOffset);

  return result;
}

CalibrationResult AECCalibration::analyzeAligned(
    const std::vector<float>& alignedRef,
    const std::vector<float>& alignedMic,
    unsigned int sampleRate,
    CalibrationSignalType signalType) {

  CalibrationResult result = {0, 0.0f, 0.0f, 0.0f, false, {}};

  if (alignedMic.empty() || alignedRef.empty()) {
    aecLog("[AEC Calibration] analyzeAligned: Missing data (ref=%zu, mic=%zu)\n",
           alignedRef.size(), alignedMic.size());
    return result;
  }

  aecLog("[AEC Calibration] analyzeAligned: Using frame-aligned buffers (ref=%zu, mic=%zu), signalType=%d\n",
         alignedRef.size(), alignedMic.size(), static_cast<int>(signalType));

  // Use click-specific analysis for click signals
  if (signalType == CalibrationSignalType::Click) {
    result = analyzeClickCalibration(alignedRef, alignedMic, sampleRate);

    // Add calibrated offset for position-based sync
    int64_t counterDiff = static_cast<int64_t>(sCaptureFramesAtStart) -
                          static_cast<int64_t>(sOutputFramesAtStart);
    result.calibratedOffset = counterDiff + result.delaySamples;

    aecLog("[AEC Click Calibration] calibratedOffset: counterDiff=%lld + acoustic=%d = %lld samples\n",
           (long long)counterDiff, result.delaySamples, (long long)result.calibratedOffset);

    return result;
  }

  // Original chirp analysis below

  // Step 1: Delay estimation directly on aligned signals
  // No frame counter pre-alignment needed - signals are already frame-aligned!
  int estimatedDelay = DelayEstimator::estimateDelay(alignedRef, alignedMic);

  // Apply offset to place peak inside filter (causality margin)
  int offset = 32;
  estimatedDelay -= offset;

  result.delaySamples = estimatedDelay;
  if (result.delaySamples < 0)
    result.delaySamples = 0;
  result.delayMs = (float)(result.delaySamples * 1000) / (float)sampleRate;

  aecLog("[AEC Calibration] Aligned Delay: %d samples (%.2fms) [Offset -%d]\n",
         result.delaySamples, result.delayMs, offset);

  // Step 2: Prepare Aligned Buffers for Training
  size_t trainingLen = 0;
  if (estimatedDelay >= 0 && estimatedDelay < (int)alignedMic.size()) {
    trainingLen = std::min(alignedRef.size(), alignedMic.size() - estimatedDelay);
  }

  // Limit training length
  size_t chirpLen = (size_t)(0.5 * sampleRate);
  size_t maxTrain = chirpLen + 9600; // +200ms reverb tail
  if (trainingLen > maxTrain) {
    trainingLen = maxTrain;
  }

  if (trainingLen < 1024) {
    aecLog("[AEC Calibration] Error: Insufficient overlap for training (%zu)\n",
           trainingLen);
    return result;
  }

  std::vector<float> trainRef(trainingLen);
  std::vector<float> trainMic(trainingLen);

  std::memcpy(trainRef.data(), alignedRef.data(), trainingLen * sizeof(float));
  std::memcpy(trainMic.data(), alignedMic.data() + estimatedDelay,
              trainingLen * sizeof(float));

  // Step 3: Offline Training
  VssNlmsFilter trainer(2048);

  aecLog("[AEC Calibration] Training filter on %zu aligned samples...\n", trainingLen);

  trainer.setStepSize(1.0f);
  trainer.setSmoothingFactor(0.9f);
  trainer.setLeakage(1.0f);

  // Debug: Check signal levels
  float maxRef = 0.0f, maxMic = 0.0f;
  for (float v : trainRef)
    maxRef = std::max(maxRef, std::abs(v));
  for (float v : trainMic)
    maxMic = std::max(maxMic, std::abs(v));
  aecLog("[AEC Calibration] Aligned Signal Max Levels: Ref=%.4f Mic=%.4f\n",
         maxRef, maxMic);

  for (size_t i = 0; i < trainingLen; ++i) {
    trainer.processSample(trainRef[i], trainMic[i]);

    if (i < 10 || (i % 5000) == 0) {
      aecLog("[VSS-Aligned] i=%zu x=%.4f d=%.4f e=%.4f mu=%.6f wEnerg=%.6f\n", i,
             trainRef[i], trainMic[i], trainer.getLastError(),
             trainer.getLastStepSize(), trainer.getCoeffEnergy());
    }
  }

  trainer.setStepSize(0.5f);
  trainer.setSmoothingFactor(0.95f);
  trainer.setLeakage(1.0f);

  // Step 4: Extract Weights
  std::vector<float> weights = trainer.getWeights();
  result.impulseResponse.resize(IR_LENGTH, 0.0f);

  float energy = 0.0f;
  for (size_t i = 0; i < std::min((size_t)IR_LENGTH, weights.size()); ++i) {
    result.impulseResponse[i] = weights[i];
    energy += weights[i] * weights[i];
  }

  result.echoGain = std::sqrt(energy);
  result.correlation = trainer.getLastCorrelation();

  if (energy < 1e-6f) {
    aecLog("[AEC Calibration] Warning: Low energy weights in aligned analysis.\n");
    result.success = false;
  } else {
    result.success = true;
  }

  aecLog("[AEC Calibration] First 10 weights: ");
  for (int i = 0; i < 10; ++i)
    aecLog("%.4f ", result.impulseResponse[i]);
  aecLog("\n");

  aecLog("[AEC Calibration] Aligned Result: delay=%.2fms gain=%.3f corr=%.4f\n",
         result.delayMs, result.echoGain, result.correlation);

  // The calibrated offset for position-based sync must include BOTH:
  // 1. The frame counter difference between capture and output devices (counterDiff)
  // 2. The acoustic delay from speaker to microphone (delaySamples)
  // Formula: captureFrame - calibratedOffset = outputFrame
  // Note: The aligned signals are frame-aligned WITHIN the AEC callback, but the
  // capture device and output device frame COUNTERS are independent and may differ.
  int64_t counterDiff = static_cast<int64_t>(sCaptureFramesAtStart) -
                        static_cast<int64_t>(sOutputFramesAtStart);
  result.calibratedOffset = counterDiff + result.delaySamples;

  aecLog("[AEC Calibration] Aligned calibratedOffset: counterDiff=%lld + acoustic=%d = %lld samples\n",
         (long long)counterDiff, result.delaySamples, (long long)result.calibratedOffset);

  return result;
}

void AECCalibration::reset() {
  sRefCapture.clear();
  sMicCapture.clear();
  sGeneratedSignal.clear();
  sOutputFramesAtStart = 0;
  sCaptureFramesAtStart = 0;
}

void AECCalibration::recordFrameCountersAtStart(size_t outputFrames,
                                                 size_t captureFrames) {
  sOutputFramesAtStart = outputFrames;
  sCaptureFramesAtStart = captureFrames;
  aecLog("[AEC Calibration] Frame counters recorded: output=%zu capture=%zu\n",
         outputFrames, captureFrames);
}

int AECCalibration::getRefSignal(float *dest, int maxLength) {
  if (dest == nullptr || maxLength <= 0)
    return 0;
  int copyLen = std::min(maxLength, static_cast<int>(sRefCapture.size()));
  std::memcpy(dest, sRefCapture.data(), copyLen * sizeof(float));
  return copyLen;
}

int AECCalibration::getMicSignal(float *dest, int maxLength) {
  if (dest == nullptr || maxLength <= 0)
    return 0;
  int copyLen = std::min(maxLength, static_cast<int>(sMicCapture.size()));
  std::memcpy(dest, sMicCapture.data(), copyLen * sizeof(float));
  return copyLen;
}

void AECCalibration::writeWavHeader(uint8_t *buffer, unsigned int sampleRate,
                                    unsigned int channels, size_t numSamples) {
  // WAV header for 32-bit float PCM
  size_t dataSize = numSamples * channels * sizeof(float);

  // RIFF header
  buffer[0] = 'R';
  buffer[1] = 'I';
  buffer[2] = 'F';
  buffer[3] = 'F';
  uint32_t fileSize = static_cast<uint32_t>(36 + dataSize);
  std::memcpy(buffer + 4, &fileSize, 4);
  buffer[8] = 'W';
  buffer[9] = 'A';
  buffer[10] = 'V';
  buffer[11] = 'E';

  // fmt chunk
  buffer[12] = 'f';
  buffer[13] = 'm';
  buffer[14] = 't';
  buffer[15] = ' ';
  uint32_t fmtSize = 16;
  std::memcpy(buffer + 16, &fmtSize, 4);
  uint16_t audioFormat = 3; // IEEE float
  std::memcpy(buffer + 20, &audioFormat, 2);
  uint16_t numChannels = static_cast<uint16_t>(channels);
  std::memcpy(buffer + 22, &numChannels, 2);
  std::memcpy(buffer + 24, &sampleRate, 4);
  uint32_t byteRate = sampleRate * channels * sizeof(float);
  std::memcpy(buffer + 28, &byteRate, 4);
  uint16_t blockAlign = static_cast<uint16_t>(channels * sizeof(float));
  std::memcpy(buffer + 32, &blockAlign, 2);
  uint16_t bitsPerSample = 32;
  std::memcpy(buffer + 34, &bitsPerSample, 2);

  // data chunk
  buffer[36] = 'd';
  buffer[37] = 'a';
  buffer[38] = 't';
  buffer[39] = 'a';
  uint32_t dataChunkSize = static_cast<uint32_t>(dataSize);
  std::memcpy(buffer + 40, &dataChunkSize, 4);
}
