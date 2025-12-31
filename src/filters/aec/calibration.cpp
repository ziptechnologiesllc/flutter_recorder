#include "calibration.h"
#include "delay_estimator.h"
#include "vss_nlms_filter.h"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>

// Log file for calibration debug output (visible to Flutter test)
static std::ofstream sLogFile;
static bool sLogFileOpened = false;

// Store log messages in memory so they can be retrieved via FFI
static std::string sLogBuffer;
static const size_t MAX_LOG_SIZE = 64 * 1024; // 64KB max

void aecLog(const char *fmt, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

  // Append to in-memory buffer (truncate if too large)
  if (sLogBuffer.size() < MAX_LOG_SIZE) {
    sLogBuffer += buffer;
  }

  // Also print to stderr
  fprintf(stderr, "%s", buffer);
  fflush(stderr);
}

// FFI export to retrieve log buffer
extern "C" {
const char *flutter_recorder_aec_getCalibrationLog() {
  return sLogBuffer.c_str();
}

void flutter_recorder_aec_clearCalibrationLog() { sLogBuffer.clear(); }
}

// Static member initialization
std::vector<float> AECCalibration::sRefCapture;
std::vector<float> AECCalibration::sMicCapture;
std::vector<float> AECCalibration::sGeneratedSignal;
size_t AECCalibration::sOutputFramesAtStart = 0;
size_t AECCalibration::sCaptureFramesAtStart = 0;

uint8_t *AECCalibration::generateCalibrationWav(unsigned int sampleRate,
                                                unsigned int channels,
                                                size_t *outSize) {
  // Use a Logarithmic Sine Sweep (Chirp) for Warm Start
  // Duration: 0.5s
  const float DURATION_SEC = 0.5f;
  size_t totalFrames = static_cast<size_t>(sampleRate * DURATION_SEC);

  // Add some padding silence at end?
  // Let's add 100ms silence to let the room ring out
  size_t paddingFrames = sampleRate / 10;
  totalFrames += paddingFrames;

  // WAV format: 44 byte header + float32 samples
  size_t dataSize = totalFrames * channels * sizeof(float);
  size_t totalSize = 44 + dataSize;

  uint8_t *buffer = new uint8_t[totalSize];

  // Write WAV header
  writeWavHeader(buffer, sampleRate, channels, totalFrames);

  // Generate audio data
  float *audioData = reinterpret_cast<float *>(buffer + 44);
  std::memset(audioData, 0, dataSize);

  // Generate mono chirp
  sGeneratedSignal.resize(totalFrames, 0.0f);

  double f_start = 20.0;
  double f_end = sampleRate / 2.0;
  size_t chirpSamples = static_cast<size_t>(sampleRate * DURATION_SEC);
  double k = std::pow(f_end / f_start, 1.0 / chirpSamples);

  // CONSTANT: Max amplitude
  const float MAX_AMPLITUDE = 0.5f;

  for (size_t i = 0; i < chirpSamples; ++i) {
    // Logarithmic frequency increase
    // Phase phi(t) = 2*pi * f_start * (k^t - 1) / ln(k)
    // where t is time in seconds? No, classic formula is discretized.
    // Correct phase integration: division by sampleRate is required
    // Phase = 2*pi * f_start * (k^i - 1) / (ln(k) * Fs)
    double phase = 2.0 * M_PI * f_start * (std::pow(k, i) - 1.0) /
                   (std::log(k) * sampleRate);
    sGeneratedSignal[i] = MAX_AMPLITUDE * std::sin(phase);

    // Fade out last 50 samples to prevent popping
    if (i > chirpSamples - 50) {
      sGeneratedSignal[i] *= (chirpSamples - i) / 50.0f;
    }
  }

  // Copy to all channels (interleaved)
  for (size_t frame = 0; frame < totalFrames; ++frame) {
    for (unsigned int ch = 0; ch < channels; ++ch) {
      audioData[frame * channels + ch] = sGeneratedSignal[frame];
    }
  }

  aecLog("\n[AEC Calibration] Generated Log Chirp (%.2fs, %zu samples)\n",
         DURATION_SEC, chirpSamples);

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
  // The ref and mic signals are captured by independent threads with no
  // synchronization. Use the recorded frame counters to align them.
  int64_t counterDiff = static_cast<int64_t>(sCaptureFramesAtStart) -
                        static_cast<int64_t>(sOutputFramesAtStart);

  aecLog("[AEC Calibration] Frame counter diff: capture=%zu - output=%zu = %lld\n",
         sCaptureFramesAtStart, sOutputFramesAtStart, (long long)counterDiff);

  std::vector<float> preAlignedRef, preAlignedMic;
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
    unsigned int sampleRate) {

  CalibrationResult result = {0, 0.0f, 0.0f, 0.0f, false, {}};

  if (alignedMic.empty() || alignedRef.empty()) {
    aecLog("[AEC Calibration] analyzeAligned: Missing data (ref=%zu, mic=%zu)\n",
           alignedRef.size(), alignedMic.size());
    return result;
  }

  aecLog("[AEC Calibration] analyzeAligned: Using frame-aligned buffers (ref=%zu, mic=%zu)\n",
         alignedRef.size(), alignedMic.size());

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

  // The calibrated offset for position-based sync is simply the estimated delay
  // since the aligned signals are already synchronized at frame level
  result.calibratedOffset = result.delaySamples;

  aecLog("[AEC Calibration] Aligned calibratedOffset=%lld samples\n",
         (long long)result.calibratedOffset);

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
