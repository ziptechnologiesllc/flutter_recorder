#include "calibration.h"
#include "../../fft/soloud_fft.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

// Static member initialization
std::vector<float> AECCalibration::sRefCapture;
std::vector<float> AECCalibration::sMicCapture;
std::vector<float> AECCalibration::sGeneratedSignal;

uint8_t *AECCalibration::generateCalibrationWav(unsigned int sampleRate,
                                                unsigned int channels,
                                                size_t *outSize) {
  // Calculate total samples - white noise only
  size_t totalSamples = (sampleRate * WHITE_NOISE_DURATION_MS) / 1000;
  size_t totalFrames = totalSamples; // samples per channel

  // WAV format: 44 byte header + float32 samples
  size_t dataSize = totalFrames * channels * sizeof(float);
  size_t totalSize = 44 + dataSize;

  uint8_t *buffer = new uint8_t[totalSize];

  // Write WAV header
  writeWavHeader(buffer, sampleRate, channels, totalFrames);

  // Generate audio data
  float *audioData = reinterpret_cast<float *>(buffer + 44);

  // Generate mono signal first and store it for use as reference
  sGeneratedSignal.resize(totalSamples);

  // White noise only - works better for correlation than sine sweep
  // (phone speakers can't reproduce high frequencies in sweep)
  generateWhiteNoise(sGeneratedSignal.data(), totalSamples, SIGNAL_AMPLITUDE);

  // Copy to all channels (interleaved)
  for (size_t frame = 0; frame < totalFrames; ++frame) {
    for (unsigned int ch = 0; ch < channels; ++ch) {
      audioData[frame * channels + ch] = sGeneratedSignal[frame];
    }
  }

  printf("[AEC Calibration] Generated %zu samples (%.1fms) of white noise\n",
         totalSamples, totalSamples * 1000.0f / sampleRate);

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

  printf(
      "[AEC Calibration] Captured signals: ref=%zu samples, mic=%zu samples\n",
      referenceLen, micLen);
}

CalibrationResult AECCalibration::analyze(unsigned int sampleRate) {
  CalibrationResult result = {0, 0, 0.0f, 0.0f, false};

  if (sRefCapture.empty() || sMicCapture.empty()) {
    printf("[AEC Calibration] analyze: No data (ref=%zu mic=%zu)\n",
           sRefCapture.size(), sMicCapture.size());
    return result;
  }

  // Debug: compute signal energy
  // Debug: compute signal energy and stats over ALL samples
  float refEnergy = 0.0f, micEnergy = 0.0f;
  float refMax = 0.0f, micMax = 0.0f;

  for (float val : sRefCapture) {
    refEnergy += val * val;
    if (std::abs(val) > refMax)
      refMax = std::abs(val);
  }
  for (float val : sMicCapture) {
    micEnergy += val * val;
    if (std::abs(val) > micMax)
      micMax = std::abs(val);
  }

  // Normalize energy by length
  if (!sRefCapture.empty())
    refEnergy /= sRefCapture.size();
  if (!sMicCapture.empty())
    micEnergy /= sMicCapture.size();

  printf("[AEC Calibration] analyze: ref samples=%zu (RMS=%.6f, Peak=%.4f) mic "
         "samples=%zu (RMS=%.6f, Peak=%.4f)\n",
         sRefCapture.size(), std::sqrt(refEnergy), refMax, sMicCapture.size(),
         std::sqrt(micEnergy), micMax);

  // Calculate max delay in samples
  int maxDelaySamples = (sampleRate * MAX_DELAY_SEARCH_MS) / 1000;

  // Find optimal delay via cross-correlation
  float peakCorrelation = 0.0f;
  int optimalDelay = findOptimalDelay(sRefCapture.data(), sRefCapture.size(),
                                      sMicCapture.data(), sMicCapture.size(),
                                      maxDelaySamples, &peakCorrelation);

  // Check if correlation is strong enough
  if (peakCorrelation < MIN_CORRELATION_THRESHOLD) {
    // Correlation too weak - calibration may have failed
    result.success = false;
    result.correlation = peakCorrelation;
    return result;
  }

  // Estimate echo gain at optimal delay
  float echoGain = estimateEchoGain(
      sRefCapture.data(), sMicCapture.data(),
      std::min(sRefCapture.size(), sMicCapture.size()), optimalDelay);

  // Compute impulse response via FFT deconvolution
  // Align signals first using the optimal delay
  size_t alignedLen =
      std::min(sRefCapture.size(), sMicCapture.size() - optimalDelay);
  if (alignedLen > 1000) { // Need enough samples for good deconvolution
    result.impulseResponse = computeImpulseResponse(
        sRefCapture.data(),
        sMicCapture.data() + optimalDelay, // Align mic to reference
        alignedLen,
        2048); // Match NLMS filter length (43ms @ 48kHz -
               // mobile-optimized)
  } else {
    printf("[AEC Calibration] Warning: Not enough aligned samples for "
           "impulse "
           "response (%zu)\n",
           alignedLen);
  }

  // Populate result
  result.delaySamples = optimalDelay;
  result.delayMs = (optimalDelay * 1000) / sampleRate;
  result.echoGain = echoGain;
  result.correlation = peakCorrelation;
  result.success = true;

  printf("[AEC Calibration] Result: delay=%dms gain=%.3f corr=%.3f "
         "impulseLen=%zu\n",
         result.delayMs, result.echoGain, result.correlation,
         result.impulseResponse.size());

  return result;
}

void AECCalibration::reset() {
  sRefCapture.clear();
  sMicCapture.clear();
  sGeneratedSignal.clear();
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

void AECCalibration::generateWhiteNoise(float *buffer, size_t samples,
                                        float amplitude) {
  // Use Mersenne Twister for high-quality random numbers
  static std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < samples; ++i) {
    buffer[i] = dist(gen) * amplitude;
  }

  // Apply fade in/out to avoid clicks (10ms)
  size_t fadeLen = std::min(samples / 10, (size_t)480); // ~10ms @ 48kHz
  for (size_t i = 0; i < fadeLen; ++i) {
    float fade = static_cast<float>(i) / fadeLen;
    buffer[i] *= fade;
    buffer[samples - 1 - i] *= fade;
  }
}

void AECCalibration::generateSineSweep(float *buffer, size_t samples,
                                       unsigned int sampleRate, float startFreq,
                                       float endFreq, float amplitude) {
  // Logarithmic sine sweep (exponential chirp)
  // More energy at lower frequencies for better room characterization
  double k = std::pow(endFreq / startFreq, 1.0 / samples);
  double phase = 0.0;

  for (size_t i = 0; i < samples; ++i) {
    // Current frequency
    double freq = startFreq * std::pow(k, static_cast<double>(i));

    // Generate sample
    buffer[i] = static_cast<float>(amplitude * std::sin(phase));

    // Update phase
    phase += 2.0 * M_PI * freq / sampleRate;
    if (phase > 2.0 * M_PI) {
      phase -= 2.0 * M_PI;
    }
  }

  // Apply fade in/out to avoid clicks (10ms)
  size_t fadeLen = std::min(samples / 10, (size_t)480);
  for (size_t i = 0; i < fadeLen; ++i) {
    float fade = static_cast<float>(i) / fadeLen;
    buffer[i] *= fade;
    buffer[samples - 1 - i] *= fade;
  }
}

// Find where signal energy starts (first sample above threshold)
static size_t findSignalOnset(const float *signal, size_t len,
                              float threshold = 0.02f) {
  // Use a sliding window to detect sustained energy above threshold
  const size_t windowSize = 256; // ~5ms at 48kHz
  float windowEnergy = 0.0f;

  for (size_t i = 0; i < len; ++i) {
    windowEnergy += signal[i] * signal[i];
    if (i >= windowSize) {
      windowEnergy -= signal[i - windowSize] * signal[i - windowSize];
    }
    float avgEnergy = windowEnergy / std::min(i + 1, windowSize);
    if (avgEnergy > threshold * threshold) {
      // Found onset, return position minus window size to get start
      return (i > windowSize) ? (i - windowSize) : 0;
    }
  }
  return 0; // No onset found, assume start
}

int AECCalibration::findOptimalDelay(const float *reference, size_t refLen,
                                     const float *recorded, size_t recLen,
                                     int maxDelaySamples,
                                     float *outCorrelation) {
  int bestDelay = 0;
  float bestCorr = -1.0f;

  // Find signal onset in both buffers
  size_t refOnset = findSignalOnset(reference, refLen);
  size_t micOnset = findSignalOnset(recorded, recLen);

  printf("[AEC Calibration] Signal onset: ref=%zu mic=%zu (%.1fms vs %.1fms)\n",
         refOnset, micOnset, refOnset / 48.0f, micOnset / 48.0f);

  // Debug: print energy at different points in each signal
  auto printEnergy = [](const char *name, const float *sig, size_t len) {
    float e0 = 0, e1 = 0, e2 = 0, e3 = 0;
    size_t quarter = len / 4;
    for (size_t i = 0; i < quarter && i < len; ++i)
      e0 += sig[i] * sig[i];
    for (size_t i = quarter; i < 2 * quarter && i < len; ++i)
      e1 += sig[i] * sig[i];
    for (size_t i = 2 * quarter; i < 3 * quarter && i < len; ++i)
      e2 += sig[i] * sig[i];
    for (size_t i = 3 * quarter; i < len; ++i)
      e3 += sig[i] * sig[i];
    printf("[AEC Calibration] %s energy by quarter: [%.4f, %.4f, %.4f, %.4f]\n",
           name, std::sqrt(e0 / quarter), std::sqrt(e1 / quarter),
           std::sqrt(e2 / quarter), std::sqrt(e3 / quarter));
  };
  printEnergy("ref", reference, refLen);
  printEnergy("mic", recorded, recLen);

  // Use the portion after onset for correlation (skip initial silence)
  size_t refStart = refOnset;
  size_t micStart = micOnset;

  // Adjust available length after onset
  size_t refAvail = refLen - refStart;
  size_t micAvail = recLen - micStart;

  // Use available samples, leaving margin for delay search
  size_t windowLen = std::min(refAvail, micAvail);
  // Leave margin for delay search (max delay samples on each side)
  if (windowLen > (size_t)(maxDelaySamples * 2)) {
    windowLen -= maxDelaySamples;
  }

  printf("[AEC Calibration] Correlation window: %zu samples (%.1fms) starting "
         "at ref[%zu], mic[%zu]\n",
         windowLen, windowLen / 48.0f, refStart, micStart);

  // Pre-compute reference energy for normalization (from onset)
  float refEnergy = 0.0f;
  for (size_t i = 0; i < windowLen; ++i) {
    refEnergy += reference[refStart + i] * reference[refStart + i];
  }

  // Search both positive and negative delays around the onset difference
  // Positive delay = mic lags behind ref (acoustic delay)
  // Negative delay = ref has more lead-in silence than mic
  int startDelay = -maxDelaySamples;
  int endDelay = maxDelaySamples;

  for (int delay = startDelay; delay < endDelay; ++delay) {
    // Calculate mic offset for this delay
    int micOffset = (int)micStart + delay;
    if (micOffset < 0 || micOffset + (int)windowLen > (int)recLen) {
      continue; // Out of bounds
    }

    float corr = 0.0f;
    float recEnergy = 0.0f;

    // Compute cross-correlation
    for (size_t i = 0; i < windowLen; ++i) {
      float refSample = reference[refStart + i];
      float micSample = recorded[micOffset + i];
      corr += refSample * micSample;
      recEnergy += micSample * micSample;
    }

    // Normalized correlation coefficient
    float denom = std::sqrt(refEnergy * recEnergy);
    float normCorr = (denom > 1e-10f) ? (corr / denom) : 0.0f;

    if (normCorr > bestCorr) {
      bestCorr = normCorr;
      bestDelay = delay;
    }
  }

  // The actual acoustic delay is the difference in onsets plus the
  // correlation delay
  int totalDelaySamples = (int)(micOnset - refOnset) + bestDelay;

  printf("[AEC Calibration] Best correlation: %.4f at delay=%d samples "
         "(%.2fms)\n",
         bestCorr, bestDelay, bestDelay / 48.0f);
  printf("[AEC Calibration] Onset diff: %d samples, total delay: %d samples "
         "(%.2fms)\n",
         (int)(micOnset - refOnset), totalDelaySamples,
         totalDelaySamples / 48.0f);

  *outCorrelation = bestCorr;
  return totalDelaySamples; // Return total delay including onset difference
}

float AECCalibration::estimateEchoGain(const float *reference,
                                       const float *recorded, size_t len,
                                       int delay) {
  if (delay < 0 || delay >= (int)len) {
    return 0.0f;
  }

  // Compute RMS of both signals at optimal alignment
  size_t windowLen = len - delay;
  float refRms = 0.0f;
  float recRms = 0.0f;

  for (size_t i = 0; i < windowLen; ++i) {
    refRms += reference[i] * reference[i];
    recRms += recorded[i + delay] * recorded[i + delay];
  }

  refRms = std::sqrt(refRms / windowLen);
  recRms = std::sqrt(recRms / windowLen);

  // Echo gain is ratio of recorded to reference RMS
  return (refRms > 1e-10f) ? (recRms / refRms) : 0.0f;
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

size_t AECCalibration::nextPowerOf2(size_t n) {
  size_t power = 1;
  while (power < n) {
    power *= 2;
  }
  return power;
}

std::vector<float>
AECCalibration::computeImpulseResponse(const float *reference, const float *mic,
                                       size_t len, int filterLength) {
  // FFT deconvolution: H(f) = Mic(f) / Ref(f)
  // The impulse response h(t) = IFFT(H(f)) gives us the
  // complete speaker→mic transfer function.

  // Zero-pad to power of 2 (at least 2x signal length for linear
  // convolution)
  size_t fftSize = nextPowerOf2(len * 2);

  // SoLoud FFT uses interleaved complex format: [real0, imag0, real1,
  // imag1,
  // ...] So buffer size = fftSize * 2 (fftSize complex numbers)
  std::vector<float> refFFT(fftSize * 2, 0.0f);
  std::vector<float> micFFT(fftSize * 2, 0.0f);

  // Apply Hann window to prevent spectral leakage, then copy to FFT
  // buffers Hann window: w(n) = 0.5 * (1 - cos(2π * n / (N-1)))
  for (size_t i = 0; i < len; ++i) {
    float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (len - 1)));
    refFFT[i * 2] = reference[i] * window; // Real part (windowed)
    refFFT[i * 2 + 1] = 0.0f;              // Imaginary part
    micFFT[i * 2] = mic[i] * window;
    micFFT[i * 2 + 1] = 0.0f;
  }

  // Forward FFT of both signals
  FFT::fft(refFFT.data(), static_cast<unsigned int>(fftSize * 2));
  FFT::fft(micFFT.data(), static_cast<unsigned int>(fftSize * 2));

  // Complex division: H = Mic / Ref (with regularization to avoid
  // division by zero) For complex numbers: (a+bi) / (c+di) =
  // (ac+bd)/(c²+d²) + (bc-ad)/(c²+d²)i
  std::vector<float> transferFunc(fftSize * 2, 0.0f);

  // Find max reference power for adaptive regularization
  float maxRefPower = 0.0f;
  for (size_t i = 0; i < fftSize; ++i) {
    size_t idx = i * 2;
    float power = refFFT[idx] * refFFT[idx] + refFFT[idx + 1] * refFFT[idx + 1];
    if (power > maxRefPower)
      maxRefPower = power;
  }
  // Use 1% of peak power as regularization - prevents instability while
  // preserving detail
  float regularization = 0.01f * maxRefPower + 1e-10f;
  printf("[AEC Calibration] FFT deconvolution: fftSize=%zu maxRefPower=%.6f "
         "reg=%.6f\n",
         fftSize, maxRefPower, regularization);

  for (size_t i = 0; i < fftSize; ++i) {
    size_t idx = i * 2;
    float refReal = refFFT[idx];
    float refImag = refFFT[idx + 1];
    float micReal = micFFT[idx];
    float micImag = micFFT[idx + 1];

    // Denominator: |Ref|² + regularization
    float denom = refReal * refReal + refImag * refImag + regularization;

    // H = Mic * conj(Ref) / denom
    transferFunc[idx] = (micReal * refReal + micImag * refImag) / denom;
    transferFunc[idx + 1] = (micImag * refReal - micReal * refImag) / denom;
  }

  // Inverse FFT to get time-domain impulse response
  FFT::ifft(transferFunc.data(), static_cast<unsigned int>(fftSize * 2));

  // Extract real parts as impulse response (first filterLength samples)
  std::vector<float> impulseResponse(filterLength, 0.0f);
  int copyLen = std::min(filterLength, static_cast<int>(fftSize));

  // Find peak of impulse response to normalize
  float maxVal = 0.0f;
  for (int i = 0; i < copyLen; ++i) {
    float val = std::abs(transferFunc[i * 2]); // Real part
    if (val > maxVal)
      maxVal = val;
  }

  // Copy and optionally normalize
  // Copy and optionally normalize
  // float normFactor = (maxVal > 0.01f) ? (1.0f / maxVal) : 1.0f;
  for (int i = 0; i < copyLen; ++i) {
    // Don't normalize - keep actual gain values for NLMS
    impulseResponse[i] = transferFunc[i * 2]; // Real part only
  }

  // Debug output
  float energy = 0.0f;
  for (int i = 0; i < copyLen; ++i) {
    energy += impulseResponse[i] * impulseResponse[i];
  }
  printf("[AEC Calibration] Computed impulse response: len=%d energy=%.4f "
         "peak=%.4f\n",
         copyLen, energy, maxVal);

  return impulseResponse;
}
