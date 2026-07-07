#include "adaptive_echo_cancellation.h"
#include "aec_test.h"

#include "neural_post_filter.h"
#include "../../soloud_slave_bridge.h"
#include "spectral_governor.h"
#include "../../native_scheduler.h" // LSAEC: known loop period P + start frame
#include "../../audio_engine/audio_engine.h" // LSAEC: engine frame for div diag
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <string>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <Foundation/Foundation.h>
#endif

// Get sandbox-accessible temp directory
static std::string getTempDir() {
#ifdef __ANDROID__
  // Android doesn't have a writable /tmp/
  return "";
#elif defined(__APPLE__)
#if TARGET_OS_MAC && !TARGET_OS_IPHONE
  NSString *tempDir = NSTemporaryDirectory();
  if (tempDir) {
    return std::string([tempDir UTF8String]);
  }
#else
  // iOS: /tmp is NOT writable. Use the app's Documents dir (pullable over USB
  // via afcclient --documents now that UIFileSharingEnabled is set).
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                                       NSUserDomainMask, YES);
  if ([paths count] > 0) {
    return std::string([[paths objectAtIndex:0] UTF8String]) + "/";
  }
#endif
#endif
  return "/tmp/";
}

extern void aecLog(const char *fmt, ...);

AdaptiveEchoCancellation::AdaptiveEchoCancellation(unsigned int sampleRate,
                                                   unsigned int channels)
    : mSampleRate(sampleRate), mChannels(channels),
      mParams{
          // Parameter: default, min, max
          {StepSize,
           {0.005f, 0.001f, 0.2f}}, // NLMS step size (matches nlms_filter.h)
          {DelayMs,
           {30.0f, 0.0f,
            100.0f}}, // Acoustic delay in ms (30ms default for phone)
          {Enabled, {1.0f, 0.0f, 1.0f}} // Enable flag (1=on, 0=bypass)
      },
      mValues(ParamCount, 0.0f), mDelaySamples(0), mUseTimestampSync(true),
      mCurrentCallbackTimestamp(AECReferenceBuffer::Clock::now()),
      mNeuralFilter(std::make_unique<NeuralPostFilter>(sampleRate, channels)) {
  // Initialize parameter values with defaults
  for (auto const &it : mParams) {
    mValues[it.first] = it.second.defaultVal;
  }

  // Calculate initial delay in samples
  updateDelay();

  // Create one filter set per channel. Shadow filter (Phase A): the
  // foreground VSS filter is frozen — it produces the cancelled output using
  // the last promoted weights; the background filter adapts recklessly in raw
  // mode and is promoted into the foreground only when it measurably wins.
  for (unsigned int ch = 0; ch < channels; ++ch) {
    mFilters.push_back(
        std::make_unique<NLMSFilter>(NLMSFilter::DEFAULT_FILTER_LENGTH));
    auto fg =
        std::make_unique<VssNlmsFilter>(VssNlmsFilter::DEFAULT_FILTER_LENGTH);
    fg->setFrozen(true);
    mVssFilters.push_back(std::move(fg));
    auto bg =
        std::make_unique<VssNlmsFilter>(VssNlmsFilter::DEFAULT_FILTER_LENGTH);
    bg->setRawAdaptation(true);
    mBgFilters.push_back(std::move(bg));
    // Per-channel shadow state; candidate EMA buffer sized to match the
    // background filter length exactly.
    ShadowChannel sc;
    sc.candWeights.assign(mBgFilters.back()->getFilterLength(), 0.0f);
    mShadow.push_back(std::move(sc));
  }

  // Pre-allocate reference buffer for batch processing
  // Size: max expected frame count * channels (assume 4096 as max)
  mRefBuffer.resize(4096 * channels, 0.0f);

  // Drift-compensated reference aligner (preallocates its histories here, off
  // the audio thread).
  mDriftAligner =
      std::make_unique<DriftAligner>(sampleRate, channels);

  // LSAEC loop-synchronous echo template (preallocates its per-phase tables
  // here, off the audio thread).
  mEchoTemplate =
      std::make_unique<SynchronousEchoTemplate>(sampleRate, channels);

  // Spectral governor worker (FFT coherence -> learning boost). Constructed
  // on the FFI/Dart thread (addFilter), never the render thread.
  SpectralGovernor::instance().start();
}

int AdaptiveEchoCancellation::getParamCount() const { return ParamCount; }

float AdaptiveEchoCancellation::getParamMax(int param) const {
  validateParam(param);
  return mParams.at(static_cast<Params>(param)).maxVal;
}

float AdaptiveEchoCancellation::getParamMin(int param) const {
  validateParam(param);
  return mParams.at(static_cast<Params>(param)).minVal;
}

float AdaptiveEchoCancellation::getParamDef(int param) const {
  validateParam(param);
  return mParams.at(static_cast<Params>(param)).defaultVal;
}

std::string AdaptiveEchoCancellation::getParamName(int param) const {
  validateParam(param);
  switch (static_cast<Params>(param)) {
  case StepSize:
    return "Step Size";
  case DelayMs:
    return "Delay (ms)";
  case Enabled:
    return "Enabled";
  default:
    return "Unknown";
  }
}

void AdaptiveEchoCancellation::setParamValue(int param, float value) {
  validateParam(param);
  const auto &range = mParams.at(static_cast<Params>(param));
  // std::clamp is C++17, using max/min for C++14 compatibility
  value = std::max(range.minVal, std::min(value, range.maxVal));
  mValues[param] = value;

  // Apply changes to filters
  switch (static_cast<Params>(param)) {
  case StepSize:
    for (auto &filter : mFilters) {
      filter->setStepSize(value);
    }
    break;
  case DelayMs:
    updateDelay();
    break;
  case Enabled:
    if (value < 0.5f) {
      // Reset filters when disabled
      reset();
    }
    break;
  default:
    break;
  }
}

float AdaptiveEchoCancellation::getParamValue(int param) const {
  validateParam(param);
  return mValues[param];
}

void AdaptiveEchoCancellation::process(void *pInput, ma_uint32 frameCount,
                                       unsigned int channels,
                                       ma_format format) {
  // Use current timestamp for synchronization
  mCurrentCallbackTimestamp = AECReferenceBuffer::now();
  mUseTimestampSync = true;

  processWithTimestamp(pInput, frameCount, channels, format,
                       mCurrentCallbackTimestamp);
}

void AdaptiveEchoCancellation::processWithTimestamp(
    void *pInput, ma_uint32 frameCount, unsigned int channels, ma_format format,
    AECReferenceBuffer::TimePoint timestamp) {
  // Check if filter is enabled
  if (mValues[Enabled] < 0.5f) {
    return; // Bypass - no processing
  }

  // Check if reference buffer is available
  if (g_aecReferenceBuffer == nullptr) {
    return; // No reference signal - can't do echo cancellation
  }

  // Store timestamp for use in processAudio
  mCurrentCallbackTimestamp = timestamp;
  mUseTimestampSync = true;

  switch (format) {
  case ma_format_u8:
    processAudio<unsigned char>(pInput, frameCount, channels);
    break;
  case ma_format_s16:
    processAudio<int16_t>(pInput, frameCount, channels);
    break;
  case ma_format_s32:
    processAudio<int32_t>(pInput, frameCount, channels);
    break;
  case ma_format_f32:
    processAudio<float>(pInput, frameCount, channels);
    break;
  default:
    std::cerr << "AdaptiveEchoCancellation: Unsupported format\n";
    break;
  }
}

template <typename T>
void AdaptiveEchoCancellation::processAudio(void *pInput, ma_uint32 frameCount,
                                            unsigned int channels) {
  static int callCount = 0;
  callCount++;
  // Log sparingly: first 5 calls, then every 500 calls (~5 seconds at typical
  // buffer sizes)
  if (callCount <= 5 || callCount % 500 == 0) {
    aecLog(
        "[AEC processAudio] call #%d, frames=%u ch=%u enabled=%.1f refBuf=%p\n",
        callCount, frameCount, channels, mValues[Enabled],
        g_aecReferenceBuffer);
  }

  T *input = static_cast<T *>(pInput);

  // Ensure we have enough filters for the channels
  while (mFilters.size() < channels) {
    mFilters.push_back(
        std::make_unique<NLMSFilter>(NLMSFilter::DEFAULT_FILTER_LENGTH));
    // Shadow filter (Phase A): foreground is frozen — pure FIR, produces the
    // cancelled output the user hears, never corrupted by adaptation. The
    // background adapts recklessly in raw mode and is promoted into the
    // foreground only when it measurably wins (see promotion block below).
    auto fg =
        std::make_unique<VssNlmsFilter>(VssNlmsFilter::DEFAULT_FILTER_LENGTH);
    fg->setFrozen(true);
    mVssFilters.push_back(std::move(fg));
    auto bg =
        std::make_unique<VssNlmsFilter>(VssNlmsFilter::DEFAULT_FILTER_LENGTH);
    bg->setRawAdaptation(true);
    mBgFilters.push_back(std::move(bg));
    ShadowChannel sc;
    sc.candWeights.assign(mBgFilters.back()->getFilterLength(), 0.0f);
    mShadow.push_back(std::move(sc));
  }

  // Read reference signal from shared buffer
  size_t totalSamples = frameCount * channels;
  if (mRefBuffer.size() < totalSamples) {
    mRefBuffer.resize(totalSamples);
  }
  if (mLinearOutputBuffer.size() < totalSamples) {
    mLinearOutputBuffer.resize(totalSamples);
  }

  // Read reference signal
  size_t framesRead = 0;
  bool dcraActive = false; // true when the DriftAligner is driving the read

  // LSAEC: in slave mode with a KNOWN loop period, the loop-synchronous echo
  // template replaces the NLMS/shadow/promotion stack. The loop period P and
  // start frame are exact (the engine scheduled them); mCaptureFrameCount is
  // the engine frame of this block — same coordinate as loopStartFrame — so the
  // loop phase is pure arithmetic, no estimation.
  int64_t lsLoopFrames = 0;
  int64_t lsLoopStart = 0;
  bool useTemplate = false;
  if (mAecMode == aecModeLsaec && soloud_isSlaveMode() && mEchoTemplate) {
    lsLoopFrames = NativeScheduler::instance().getBaseLoopFrames();
    lsLoopStart = NativeScheduler::instance().getBaseLoopStartFrame();
    useTemplate = (lsLoopFrames > 0);
  }

  if (soloud_isSlaveMode()) {
    // SLAVE MODE: Reference was written in the SAME callback, perfectly time-aligned.
    // We only need to apply the ACOUSTIC delay, not the full calibratedOffset
    // (which includes thread timing differences that don't apply in slave mode).
    size_t totalWritten = g_aecReferenceBuffer->getFramesWritten();

    // Slave-mode reference delay. mAcousticDelaySamples is only set by a live
    // calibration run; on a saved-calibration boot it stays 0. Fall back to the
    // calibrated DelayMs parameter (restored from prefs) — it is the same
    // measured echo delay. Without any delay the reference is read with zero
    // lag, ~one round-trip out of sync with the mic echo, so NLMS sees corr≈0
    // and never adapts.
    size_t effectiveDelay = mAcousticDelaySamples;
    if (effectiveDelay == 0 && mValues[DelayMs] > 0.0f) {
      effectiveDelay =
          static_cast<size_t>((mValues[DelayMs] / 1000.0f) * mSampleRate);
    }

    // During calibration (no delay known yet), always read the most recent frames
    // This ensures we capture ref+mic for delay estimation even before calibration is complete
    bool isCalibrating = mCalibrationCaptureEnabled || effectiveDelay == 0;

    if (isCalibrating) {
      // During calibration: read the most recent frames (zero acoustic lag) so
      // the calibration's own delay estimator gets a raw ref/mic pair. DCRA is
      // bypassed here — it only drives the normal post-calibration path.
      if (totalWritten >= frameCount) {
        framesRead = g_aecReferenceBuffer->readFramesAtPosition(
            mRefBuffer.data(), frameCount, totalWritten - frameCount);
      }
    } else if (useTemplate) {
      // LSAEC E1 — DETERMINISTIC reference read. In slave mode the reference
      // was written in THIS callback (same clock), so the echo-aligned
      // reference is just a fixed integer offset back by the acoustic delay —
      // NO drift estimation, no cross-correlation. This reference is used only
      // for far-end gating of the template (it learns the echo from the mic).
      if (totalWritten >= frameCount + effectiveDelay) {
        framesRead = g_aecReferenceBuffer->readFramesAtPosition(
            mRefBuffer.data(), frameCount,
            totalWritten - frameCount - effectiveDelay);
      } else {
        // Not enough history yet: present silence to the gate (no learning)
        // but still let the template run (passthrough output while E is empty).
        std::fill(mRefBuffer.begin(), mRefBuffer.begin() + totalSamples, 0.0f);
        framesRead = frameCount;
      }
    } else {
      // LEGACY NORMAL AEC PATH — DRIFT-COMPENSATED reference read for the NLMS
      // stack (used when there is no known loop period). The DriftAligner
      // advances a fractional read pointer at the clock-drift-corrected rate.
      framesRead = mDriftAligner->produceAligned(
          g_aecReferenceBuffer, mRefBuffer.data(), frameCount, effectiveDelay);
      dcraActive = (framesRead > 0);
    }

    static int slaveReadDebugCount = 0;
    if (++slaveReadDebugCount % 500 == 0 ||
        (isCalibrating && slaveReadDebugCount <= 10)) {
      aecLog("[AEC Slave] totalWritten=%zu effDelay=%zu read=%zu calib=%d | "
             "DCRA drift=%.0fppm resid=%.1f bulk=%.0f\n",
             totalWritten, effectiveDelay, framesRead, isCalibrating ? 1 : 0,
             mDriftAligner->driftPpm(), mDriftAligner->residual(),
             mDriftAligner->bulkDelayFrames());
    }
  } else if (mUsePositionSync && mCalibratedOffset != 0) {
    // NON-SLAVE MODE: Position-based sync using frame counters
    // This handles separate devices with potential clock drift
    int64_t startOutputFrame =
        static_cast<int64_t>(mCaptureFrameCount) - mCalibratedOffset;

    if (startOutputFrame >= 0) {
      framesRead = g_aecReferenceBuffer->readFramesAtPosition(
          mRefBuffer.data(), frameCount, static_cast<size_t>(startOutputFrame));

      // DRIFT COMPENSATION: If read failed due to buffer overrun, adjust offset
      if (framesRead == 0 && g_aecReferenceBuffer != nullptr) {
        size_t totalWritten = g_aecReferenceBuffer->getFramesWritten();
        size_t bufferSize = g_aecReferenceBuffer->sizeInFrames();

        size_t oldestAvailable = (totalWritten > bufferSize) ?
                                  (totalWritten - bufferSize + frameCount) : 0;

        if (static_cast<size_t>(startOutputFrame) < oldestAvailable) {
          int64_t newOffset = static_cast<int64_t>(mCaptureFrameCount) -
                              static_cast<int64_t>(oldestAvailable);

          static int driftWarnCount = 0;
          if (++driftWarnCount % 100 == 1) {
            aecLog("[AEC DriftComp] Clock drift detected! Adjusting offset: %lld -> %lld\n",
                   (long long)mCalibratedOffset, (long long)newOffset);
          }

          mCalibratedOffset = newOffset;
          startOutputFrame = static_cast<int64_t>(mCaptureFrameCount) - mCalibratedOffset;
          framesRead = g_aecReferenceBuffer->readFramesAtPosition(
              mRefBuffer.data(), frameCount, static_cast<size_t>(startOutputFrame));
        }
      }

      static int posReadDebugCount = 0;
      if (++posReadDebugCount % 500 == 0) {
        aecLog("[AEC PosSync] capFrame=%zu offset=%lld outFrame=%lld read=%zu\n",
               mCaptureFrameCount, (long long)mCalibratedOffset,
               (long long)startOutputFrame, framesRead);
      }
    }
  } else {
    // LEGACY MODE or calibration in progress
    if (soloud_isSlaveMode() && mCalibratedOffset == 0) {
      // Slave mode during calibration: read frames that were just written
      // Position = totalWritten - frameCount (the frames we just wrote)
      size_t totalWritten = g_aecReferenceBuffer->getFramesWritten();
      if (totalWritten >= frameCount) {
        size_t startPos = totalWritten - frameCount;
        framesRead = g_aecReferenceBuffer->readFramesAtPosition(
            mRefBuffer.data(), frameCount, startPos);

        static int slaveCalibLog = 0;
        if (slaveCalibLog++ < 5 || slaveCalibLog % 500 == 0) {
          aecLog("[AEC SlaveCalib] Reading most recent frames: pos=%zu read=%zu\n",
                 startPos, framesRead);
        }
      }
    } else {
      // LEGACY: Simple sample-based delay (for backwards compatibility)
      float delayMs = mValues[DelayMs];
      size_t delaySamples =
          static_cast<size_t>((delayMs / 1000.0f) * mSampleRate) * channels;
      framesRead = g_aecReferenceBuffer->readFrames(mRefBuffer.data(), frameCount,
                                                    delaySamples);
    }
  }

  // Accumulators for smoothed metrics over multiple blocks
  static int debugCounter = 0;
  static float totalRefEnergy = 0.0f;
  static float totalMicEnergy = 0.0f;
  static float totalCrossCorr = 0.0f;        // Accumulated cross-correlation
  static float totalRefEnergyForCorr = 0.0f; // Ref energy for correlation calc
  static int debugSamples = 0;

  // If we couldn't read the reference (data not available), skip processing
  if (framesRead == 0) {
    return;
  }

#ifdef AEC_DUMP_FILES
  // DEBUG: Dump ref and mic to files for alignment verification.
  // Arm the dump only once the reference actually carries playback signal —
  // dumping from boot just captures pre-playback silence and tells us nothing.
  // NOTE: compiled out by default — fopen/fwrite on the RT audio thread (and,
  // on iOS where "/tmp/" isn't writable, per-block retry storms) crack audio.
  // Use the syslog [AEC DCRA]/[AEC Slave] telemetry for live diagnosis instead.
  static FILE *refFile = nullptr;
  static FILE *micFile = nullptr;
  static int dumpFrames = 0;
  static bool dumpArmed = false;
  static bool dumpDone = false;
  static const int MAX_DUMP_FRAMES = 48000 * 30; // 30 s — span several loops to
                                                 // measure period/drift offline
  static std::string tempDir;

#ifndef __ANDROID__
  if (!dumpArmed && !dumpDone) {
    float refEnergy = 0.0f;
    for (ma_uint32 f = 0; f < frameCount; ++f) {
      float r = mRefBuffer[f * channels];
      refEnergy += r * r;
    }
    // ~-70 dB mean-square gate: real playback clears it, silence does not.
    if (frameCount > 0 && (refEnergy / frameCount) > 1e-7f) {
      dumpArmed = true;
    }
  }

  if (dumpArmed && !dumpDone && refFile == nullptr) {
    tempDir = getTempDir();
    std::string refPath = tempDir + "aec_ref.raw";
    std::string micPath = tempDir + "aec_mic.raw";
    refFile = fopen(refPath.c_str(), "wb");
    micFile = fopen(micPath.c_str(), "wb");
    if (refFile && micFile) {
      aecLog("[AEC DEBUG] Playback detected — dumping ref/mic to: %s\n",
             tempDir.c_str());
    } else {
      aecLog("[AEC DEBUG] Failed to open files in %s\n", tempDir.c_str());
    }
  }
#endif

  if (refFile && micFile && dumpFrames < MAX_DUMP_FRAMES) {
    // Write channel 0 only (mono) for easier analysis
    for (ma_uint32 frame = 0; frame < frameCount; ++frame) {
      float micSample =
          normalizeSample(static_cast<T *>(pInput)[frame * channels]);
      float refSample = mRefBuffer[frame * channels];
      fwrite(&refSample, sizeof(float), 1, refFile);
      fwrite(&micSample, sizeof(float), 1, micFile);
    }
    dumpFrames += frameCount;

    if (dumpFrames >= MAX_DUMP_FRAMES) {
      fclose(refFile);
      fclose(micFile);
      refFile = nullptr;
      micFile = nullptr;
      dumpDone = true;
      aecLog("[AEC DEBUG] Finished dumping %d frames to files\n", dumpFrames);
    }
  }
#endif // AEC_DUMP_FILES

  // Calibration capture: save frame-aligned ref+mic for delay estimation
  // These are perfectly aligned since they come from the same callback
  if (mCalibrationCaptureEnabled &&
      mAlignedRefCapture.size() < mCalibrationMaxSamples) {
    for (ma_uint32 frame = 0; frame < frameCount; ++frame) {
      if (mAlignedRefCapture.size() >= mCalibrationMaxSamples)
        break;
      float micSample =
          normalizeSample(static_cast<T *>(pInput)[frame * channels]);
      float refSample = mRefBuffer[frame * channels];
      mAlignedRefCapture.push_back(refSample);
      mAlignedMicCapture.push_back(micSample);
    }

    // Log progress periodically
    static int calibCapLogCount = 0;
    if (++calibCapLogCount % 100 == 0) {
      aecLog("[AEC CalibCapture] %zu/%zu samples\n", mAlignedRefCapture.size(),
             mCalibrationMaxSamples);
    }
  }

  // NLMS ADAPTIVE ECHO CANCELLATION — shadow filter (Phase A)
  // - aecModeAlgo/aecModeHybrid: shadow filter active (bg adapts, fg frozen)
  // - aecModeFrozen/aecModeFrozenNeural: foreground only, no adaptation
  // - aecModeNeural/aecModeBypass: skip the linear stage entirely
  const bool adaptiveMode =
      (mAecMode == aecModeAlgo || mAecMode == aecModeHybrid);
  const bool linearStage = adaptiveMode || mAecMode == aecModeFrozen ||
                           mAecMode == aecModeFrozenNeural;
  for (ma_uint32 frame = 0; frame < frameCount; ++frame) {
    for (unsigned int ch = 0; ch < channels; ++ch) {
      size_t idx = frame * channels + ch;

      float micSample = normalizeSample(input[idx]);
      float refSample = mRefBuffer[idx];

      // Feed the drift estimator the JUST-PRODUCED aligned reference (ch0) and
      // mic (ch0). It periodically cross-correlates these to measure residual
      // misalignment and update the fractional read pointer / drift rate.
      if (dcraActive && ch == 0)
        mDriftAligner->appendHistory(refSample, micSample);

      // Accumulate for debug output
      totalRefEnergy += refSample * refSample;
      totalMicEnergy += micSample * micSample;
      totalCrossCorr += micSample * refSample;
      totalRefEnergyForCorr += refSample * refSample;
      debugSamples++;

      float error = micSample; // Bypass/Neural/LSAEC default — skip linear stage

      if (linearStage && !useTemplate) {
        // Foreground is frozen: produces the cancelled output the user hears
        // using the last promoted weights — never corrupted by live adaptation.
        error = mVssFilters[ch]->processSample(refSample, micSample);
        if (adaptiveMode) {
          // Background adapts recklessly (raw mode). bg->processSample shifts
          // the bg's reference history with refSample — identical to the fg's
          // history (both fed the same refSample this sample).
          float eBg = mBgFilters[ch]->processSample(refSample, micSample);
          ShadowChannel &sc = mShadow[ch];
          sc.blockFgErr += (double)error * (double)error;
          sc.blockBgErr += (double)eBg * (double)eBg; // diagnostics only

          // Score the EMA candidate OUT-OF-SAMPLE against that same (just
          // shifted) history, so blockCandErr is the true residual of the EXACT
          // weights we will promote — not the live bg's in-sample error. An
          // unprimed channel has an all-zero EMA; skip it (its gate also
          // requires primed, so this can't cause a false win).
          if (sc.primed) {
            bool blown = false;
            float eCand = mBgFilters[ch]->scoreAgainstHistory(
                sc.candWeights.data(), micSample, &blown);
            sc.blockCandErr += (double)eCand * (double)eCand;
            if (blown)
              sc.blockCandBlown = 1;
          }
          // Reference energy floor: silent blocks make all errors tiny and let
          // noise flip the promotion margin.
          sc.blockRefEnergy += (double)refSample * (double)refSample;
        }
      }

      mLinearOutputBuffer[idx] = error;

      // Capture samples for AEC test (channel 0 only to avoid duplicates)
      if (ch == 0 && AECTest::isCapturing()) {
        AECTest::captureSample(micSample, error, refSample);
      }
    }
  }

  // ---- LSAEC: loop-synchronous echo template ----------------------------
  // When a loop period is known (slave mode), the linear stage above left the
  // RAW mic in mLinearOutputBuffer; the template cancels it in place by
  // subtracting the per-phase echo estimate E[phi] and learning it via a
  // far-end-gated synchronous average. No weights -> cannot diverge. (E3's
  // transient freeze will gate `learn` per block; for now we always learn.)
  if (useTemplate) {
    // Spectral governor closed loop: read the current learning boost (one
    // relaxed load), cancel, then feed the sensor with this block's aligned
    // reference + residual (wait-free ring write on the render thread; the
    // FFT/coherence math runs on the governor's worker thread).
    SpectralGovernor &gov = SpectralGovernor::instance();
    mEchoTemplate->setLearnBoost(gov.learningBoost());
    mEchoTemplate->process(mLinearOutputBuffer.data(), mRefBuffer.data(),
                           frameCount, channels,
                           static_cast<int64_t>(mCaptureFrameCount),
                           lsLoopFrames, lsLoopStart, /*learn=*/true);
    gov.push(mRefBuffer.data(), mLinearOutputBuffer.data(), frameCount,
             channels);
    // Cheap E3 diagnostics (counter reads only): freeze climbing => near-end
    // being detected; reopen climbing => E was found stale (echo changed /
    // glitch / period mismatch). Disambiguates the degradation cause.
    static int lsLogCount = 0;
    if (++lsLogCount % 300 == 0) {
      // div = capture-counter vs engine-frame offset. It MUST be constant:
      // any movement means the template's phase source is slipping against
      // the loop origin (template smears, ERLE oscillates without converging).
      aecLog("[LSAEC] P=%lld conf=%.3f freeze=%u div=%lld\n",
             (long long)lsLoopFrames, mEchoTemplate->meanConfidence(),
             mEchoTemplate->freezeCount(),
             (long long)(static_cast<int64_t>(mCaptureFrameCount) -
                         flowstate::audio_engine::AudioEngine::instance()
                             .getCurrentFrame()));
    }
  }

  // ---- Shadow-filter promotion (settled-snapshot / Polyak-Ruppert) -------
  // We promote a per-channel EMA of the background weights (the candidate,
  // mShadow[ch].candWeights), NOT the background's instantaneous noisy weights —
  // averaging cancels the LMS misadjustment that evaporates the moment the
  // filter is frozen (the ~15x in-sample/frozen gap). Strict ordering each
  // block: (1) sanity, (2) gate on the candidate's OWN held-out error scored
  // this block, (3) promote on a sustained win streak, (4) ONLY THEN reblend
  // the EMA from this block's bg so the scored vector == the promoted vector
  // (no off-by-one), and only from a SETTLED bg (mean annealed step near the
  // floor), which keeps high-misadjustment iterates out of the average and is
  // also the double-talk guard (near-end raises the residual -> raises the
  // ratio -> halts ingest).
  if (adaptiveMode && !useTemplate) {
    const size_t kShadowPromoteBlock = 2048;   // ~43 ms @ 48 kHz
    const double kShadowPromoteMargin = 0.95;  // candidate must beat fg by >=5%
    const float kShadowMaxCoeff = 50.0f;       // divergence sanity ceiling
    const float kCandEmaBeta = 0.90f;          // tail average ~430ms (N_eff~10)
    const float kBgConvergedRatio = 0.10f;     // ingest only when bg has settled
    const int kCandMinBeatBlocks = 3;          // win-streak hysteresis (~129ms)
    const double kMinBlockRefEnergy =
        1e-4 * static_cast<double>(kShadowPromoteBlock); // silent-block floor
    mBlockSamples += frameCount;
    if (mBlockSamples >= kShadowPromoteBlock) {
      const unsigned int nch =
          (channels < mShadow.size())
              ? channels
              : static_cast<unsigned int>(mShadow.size());
      static int shadowLogCount = 0;
      const bool logNow = (++shadowLogCount % 23 == 0);

      // Each channel is acoustically independent: gate, promote, and reblend it
      // on its OWN error and convergence so a slow channel never starves a
      // converged one.
      for (unsigned int ch = 0; ch < nch; ++ch) {
        ShadowChannel &sc = mShadow[ch];

        // (1) Candidate sanity: coeffEnergy of the EMA we'd install.
        float candEnergy = 0.0f;
        for (float w : sc.candWeights)
          candEnergy += w * w;
        bool candSane = candEnergy <= kShadowMaxCoeff;

        // (2) Gate on the candidate scored THIS block (held-out == promoted).
        bool win = sc.primed && candSane && !sc.blockCandBlown &&
                   (sc.blockRefEnergy > kMinBlockRefEnergy) &&
                   (sc.blockCandErr < sc.blockFgErr * kShadowPromoteMargin);
        if (win)
          sc.winStreak++;
        else
          sc.winStreak = 0;

        // (3) Promote only after a sustained streak (rejects fluke blocks).
        bool promoted = false;
        if (win && sc.winStreak >= kCandMinBeatBlocks &&
            ch < mVssFilters.size()) {
          mVssFilters[ch]->setWeightsQuiet(sc.candWeights.data(),
                                           sc.candWeights.size());
          promoted = true;
        }

        // (4) Reblend the EMA from THIS block's bg — AFTER the decision, and
        //     only if the bg has SETTLED (mean annealed step near the floor)
        //     and is sane. consumeBlockMuRatio() is drained once per channel
        //     every block regardless (it resets the bg accumulator).
        float muRatio = 1.0f;
        if (ch < mBgFilters.size()) {
          muRatio = mBgFilters[ch]->consumeBlockMuRatio();
          float bgEnergy = mBgFilters[ch]->getCoeffEnergy();
          if (bgEnergy <= kShadowMaxCoeff && muRatio <= kBgConvergedRatio) {
            const float *bgW = mBgFilters[ch]->getWeightsRef().data();
            std::vector<float> &cw = sc.candWeights;
            if (!sc.primed) {
              std::memcpy(cw.data(), bgW, cw.size() * sizeof(float)); // seed
              sc.primed = 1;
            } else {
              for (size_t i = 0; i < cw.size(); ++i)
                cw[i] = kCandEmaBeta * cw[i] + (1.0f - kCandEmaBeta) * bgW[i];
            }
          }
        }

        // Rate-limited per-channel telemetry (~once/sec) for the #27 quiet-room
        // verification: watch cand converge toward bg, fg drop to meet it, and
        // muR enter the convergence band.
        if (promoted || logNow) {
          aecLog("[AEC Shadow] ch%u fg=%.5f cand=%.5f bg=%.5f primed=%d "
                 "streak=%d muR=%.3f%s\n",
                 ch, sc.blockFgErr, sc.blockCandErr, sc.blockBgErr,
                 (int)sc.primed, sc.winStreak, muRatio,
                 promoted ? " [PROMOTED]" : "");
        }

        // (5) Reset this channel's block accumulators.
        sc.blockFgErr = 0.0;
        sc.blockBgErr = 0.0;
        sc.blockCandErr = 0.0;
        sc.blockRefEnergy = 0.0;
        sc.blockCandBlown = 0;
      }
      mBlockSamples = 0;
    }
  }

  // SECOND STAGE: Neural Post-Filter
  // This stage runs on the output of the linear stage to remove
  // residual echo and non-linearities.
  // (LSAEC isolates the template for now — neural runs only on the legacy path.)
  if (!useTemplate &&
      (mAecMode == aecModeNeural || mAecMode == aecModeHybrid ||
       mAecMode == aecModeFrozenNeural)) {
    mNeuralFilter->process(mLinearOutputBuffer.data(), mRefBuffer.data(),
                           mLinearOutputBuffer.data(), frameCount);
  }

  // Write final results back to the original input buffer, accumulating LSAEC
  // E5 gated-ERLE telemetry in the SAME pass. At this point `input` still holds
  // the original mic (it is overwritten below) and mLinearOutputBuffer holds
  // the FINAL output (post linear + neural) — exactly what gets recorded. The
  // audio thread does only adds/compares here; all dB math is deferred to the
  // Dart-side poller. "Far-end-active" gates ERLE to samples where the speaker
  // is actually playing, so a filter that eats the performer shows up as low
  // ERLE rather than being rewarded for it.
  {
    constexpr float kFarEndFloor = 1e-6f; // ~-60 dB smoothed ref power
    const uint64_t kTelemetryWindow =
        static_cast<uint64_t>(mSampleRate) * channels / 4; // ~0.25 s
    for (unsigned int i = 0; i < totalSamples; ++i) {
      const float micS = normalizeSample(input[i]);
      const float outS = mLinearOutputBuffer[i];
      const float refS = mRefBuffer[i];
      mRefPowerEma = 0.99f * mRefPowerEma + 0.01f * (refS * refS);
      const double micE = static_cast<double>(micS) * micS;
      const double outE = static_cast<double>(outS) * outS;
      mTwMicAll += micE;
      mTwOutAll += outE;
      mTwRefAll += static_cast<double>(refS) * refS;
      if (mRefPowerEma > kFarEndFloor) {
        mTwMicFar += micE;
        mTwOutFar += outE;
        ++mTwFar;
      }
      ++mTwTotal;
      input[i] = denormalizeSample<T>(outS);
    }
    if (mTwTotal >= kTelemetryWindow) {
      AecTelemetrySnapshot snap;
      snap.micEnergyFar = mTwMicFar;
      snap.outEnergyFar = mTwOutFar;
      snap.micEnergyAll = mTwMicAll;
      snap.outEnergyAll = mTwOutAll;
      snap.refEnergyAll = mTwRefAll;
      snap.farSamples = mTwFar;
      snap.totalSamples = mTwTotal;
      snap.generation = ++mTelemetryGen;
      mTelemetry.store(snap);
      mTwMicFar = mTwOutFar = 0.0;
      mTwMicAll = mTwOutAll = mTwRefAll = 0.0;
      mTwFar = mTwTotal = 0;
    }
  }

  // Debug output every ~1s (~500 callbacks at 256 frames/callback @ 48kHz)
  if (++debugCounter % 500 == 0) {
    float avgRefEnergy = debugSamples > 0 ? totalRefEnergy / debugSamples : 0;
    float avgMicEnergy = debugSamples > 0 ? totalMicEnergy / debugSamples : 0;

    // Compute cross-correlation to check alignment (normalized)
    float correlation = 0.0f;
    if (totalRefEnergyForCorr > 1e-10f && totalMicEnergy > 1e-10f) {
      correlation =
          totalCrossCorr / std::sqrt(totalRefEnergyForCorr * totalMicEnergy);
    }

    // Get filter metrics
    float coeffEnergy = 0.0f;
    float echoEst = 0.0f;
    if (!mVssFilters.empty()) {
      coeffEnergy = mVssFilters[0]->getCoeffEnergy();
      echoEst = mVssFilters[0]->getLastEchoEstimate();
    }

    // Convert energy to dB for readability
    float refDb =
        avgRefEnergy > 1e-10f ? 10.0f * std::log10(avgRefEnergy) : -100.0f;
    float micDb =
        avgMicEnergy > 1e-10f ? 10.0f * std::log10(avgMicEnergy) : -100.0f;
    float echoEstDb = std::abs(echoEst) > 1e-10f
                          ? 20.0f * std::log10(std::abs(echoEst))
                          : -100.0f;

    // Status based on coefficient energy (indicates calibration was applied)
    const char *status = coeffEnergy > 0.001f ? "ACTIVE" : "NO COEFFS";

    float currentDelayMs = mValues[DelayMs];

    aecLog("[AEC] delay=%.1fms ref=%.0fdB mic=%.0fdB ŷ=%.0fdB corr=%.2f "
           "coef=%.4f | %s\n",
           currentDelayMs, refDb, micDb, echoEstDb, correlation, coeffEnergy,
           status);

    // Reset accumulators
    totalRefEnergy = 0.0f;
    totalMicEnergy = 0.0f;
    totalCrossCorr = 0.0f;
    totalRefEnergyForCorr = 0.0f;
    debugSamples = 0;
  }
}

void AdaptiveEchoCancellation::reset() {
  for (auto &filter : mFilters) {
    filter->reset();
  }
  for (auto &filter : mVssFilters) {
    filter->reset();
  }
  for (auto &filter : mBgFilters) {
    filter->reset();
  }
  for (auto &sc : mShadow) {
    std::fill(sc.candWeights.begin(), sc.candWeights.end(), 0.0f);
    sc.primed = 0;
    sc.winStreak = 0;
    sc.blockFgErr = 0.0;
    sc.blockBgErr = 0.0;
    sc.blockCandErr = 0.0;
    sc.blockRefEnergy = 0.0;
    sc.blockCandBlown = 0;
  }
  if (mDriftAligner)
    mDriftAligner->reset();
  if (mEchoTemplate)
    mEchoTemplate->reset();
  mBlockSamples = 0;

  // E5 telemetry window accumulators (snapshot itself is left for the poller).
  mTwMicFar = mTwOutFar = 0.0;
  mTwMicAll = mTwOutAll = mTwRefAll = 0.0;
  mTwFar = mTwTotal = 0;
  mRefPowerEma = 0.0f;
}

float AdaptiveEchoCancellation::getEchoReturnLoss() const {
  // Use the stats calculated in updateStats() which reflects the active filter
  // performance
  return mCurrentStats.echoReturnLossDb;
}

void AdaptiveEchoCancellation::setImpulseResponse(const float *coeffs,
                                                  int length) {
  // Pre-initialize all channel filters with the calibrated impulse response
  // This gives NLMS a starting point for immediate cancellation
  for (auto &filter : mFilters) {
    filter->setCoefficients(coeffs, length);
  }
  for (auto &filter : mVssFilters) {
    filter->setWeights(coeffs, length);

    // Freeze state depends on AEC mode:
    // - aecModeFrozen/aecModeFrozenNeural: Pure FIR, no adaptation
    // - aecModeAlgo/aecModeHybrid: Adaptive NLMS (may cause transient artifacts)
    bool shouldFreeze = (mAecMode == aecModeFrozen || mAecMode == aecModeFrozenNeural);
    filter->setFrozen(shouldFreeze);

    if (!shouldFreeze) {
      // CONSERVATIVE step size - prevents oscillation that causes AM distortion
      // The VSS correlation-based scaling will further reduce this during double-talk
      filter->setStepSize(0.1f);
      // Smoothing for VSS statistics (lower = faster response to transients)
      filter->setSmoothingFactor(0.05f);
      // Slight leakage for numerical stability (only applied when reference present)
      filter->setLeakage(0.9999f);
    }
  }

  // Calculate coefficient energy for debug output
  float energy = 0.0f;
  for (int i = 0; i < length; ++i) {
    energy += coeffs[i] * coeffs[i];
  }

  // Warm-start the background explorers with the calibrated IR too, so they
  // begin near-converged (their annealed step drops to the floor quickly and
  // the candidate can re-seed from a settled bg almost immediately) instead of
  // re-learning the echo path from zero.
  for (auto &filter : mBgFilters) {
    filter->setWeights(coeffs, length);
  }
  // A freshly-loaded calibration is the new foreground truth; any in-flight EMA
  // candidate must re-prove itself (beat this new fg) before it can overwrite
  // the calibration. Invalidate it so it re-seeds from the warm-started bg.
  for (auto &sc : mShadow) {
    std::fill(sc.candWeights.begin(), sc.candWeights.end(), 0.0f);
    sc.primed = 0;
    sc.winStreak = 0;
  }

  bool shouldFreeze = (mAecMode == aecModeFrozen || mAecMode == aecModeFrozenNeural);
  const char *modeStr = shouldFreeze ? "FROZEN (pure FIR)" : "ADAPTIVE (mu=0.1)";
  aecLog("[AEC] Set impulse response: %d coefficients, energy=%.4f, mode=%s\n",
         length, energy, modeStr);
}

float AdaptiveEchoCancellation::measureHardwareLatency(
    const std::vector<float> &refBuffer, const std::vector<float> &micBuffer) {
  if (refBuffer.empty() || micBuffer.empty())
    return 0.0f;

  // Use DelayEstimator to find lag
  int lagSamples = DelayEstimator::estimateDelay(refBuffer, micBuffer);

  // Convert to ms
  float lagMs = (static_cast<float>(lagSamples) / mSampleRate) * 1000.0f;

  aecLog("[AEC] Measured Hardware Latency: %d samples (%.2f ms)\n", lagSamples,
         lagMs);

  // Update the DelayMs parameter
  setParamValue(Params::DelayMs, lagMs);

  return lagMs;
}

void AdaptiveEchoCancellation::validateParam(int param) const {
  if (param < 0 || param >= ParamCount) {
    throw std::invalid_argument("Invalid parameter index");
  }
}

void AdaptiveEchoCancellation::updateDelay() {
  // Convert delay from ms to samples
  float delayMs = mValues[DelayMs];
  mDelaySamples = static_cast<unsigned int>((delayMs / 1000.0f) * mSampleRate);
}

// Sample normalization helpers
float AdaptiveEchoCancellation::normalizeSample(unsigned char sample) {
  return (sample - 128) / 128.0f;
}

float AdaptiveEchoCancellation::normalizeSample(int16_t sample) {
  return sample / 32768.0f;
}

float AdaptiveEchoCancellation::normalizeSample(int32_t sample) {
  return sample / 2147483648.0f;
}

float AdaptiveEchoCancellation::normalizeSample(float sample) {
  return sample; // Already normalized
}

template <>
unsigned char
AdaptiveEchoCancellation::denormalizeSample<unsigned char>(float sample) {
  float val = sample * 128.0f + 128.0f;
  return static_cast<unsigned char>(std::max(0.0f, std::min(val, 255.0f)));
}

template <>
int16_t AdaptiveEchoCancellation::denormalizeSample<int16_t>(float sample) {
  float val = sample * 32768.0f;
  return static_cast<int16_t>(std::max(-32768.0f, std::min(val, 32767.0f)));
}

template <>
int32_t AdaptiveEchoCancellation::denormalizeSample<int32_t>(float sample) {
  float val = sample * 2147483648.0f;
  // Note: min/max on float first to avoid overflow wrap-around
  val = std::max(-2147483648.0f, std::min(val, 2147483647.0f));
  return static_cast<int32_t>(val);
}

template <>
float AdaptiveEchoCancellation::denormalizeSample<float>(float sample) {
  return sample;
}

void AdaptiveEchoCancellation::updateStats(float ref, float mic, float out) {
  // Simple leaky integrator stats
  float micEnergy = mic * mic;
  float outEnergy = out * out;

  // Smooth energies (tau ~= 100ms at 48kHz, alpha=0.0005)
  // For ~100 samples per call (block), alpha=0.1
  const float alpha = 0.01f;
  static float smoothMic = 0.0f;
  static float smoothOut = 0.0f;

  smoothMic = (1.0f - alpha) * smoothMic + alpha * micEnergy;
  smoothOut = (1.0f - alpha) * smoothOut + alpha * outEnergy;

  // Calculate instantaneous attenuation (positive dB)
  float atten = 0.0f;
  if (smoothMic > 1e-9f && smoothOut > 1e-9f) {
    float ratio = smoothMic / smoothOut;
    // ratio > 1 means mic > out (attenuation, since "out" is error)
    // Wait, GenericFilter input is (mic, ref). output is error.
    // If output is smaller than mic input, we have attenuation.
    if (ratio > 1.0f) {
      atten = 10.0f * std::log10(ratio);
    }
  }

  mCurrentStats.maxAttenuationDb = atten; // For now just current attenuation
  mCurrentStats.echoReturnLossDb = atten; // Proxy
  // Correlation is harder to calculate cheaply per-sample, skipping for now or
  // user approximation
  mCurrentStats.correlation = 0.0f;
}

AecStats AdaptiveEchoCancellation::getStats() {
  // Populate debug fields from filter state
  if (!mVssFilters.empty()) {
    mCurrentStats.filterLength =
        static_cast<int>(mVssFilters[0]->getFilterLength());
    mCurrentStats.muMax = mVssFilters[0]->getMuMax();
    mCurrentStats.muEffective = mVssFilters[0]->getLastStepSize();
    mCurrentStats.instantCorrelation = mVssFilters[0]->getLastCorrelation();

    // Convert last error to dB
    float lastErr = mVssFilters[0]->getLastError();
    mCurrentStats.lastErrorDb = std::abs(lastErr) > 1e-10f
                                    ? 20.0f * std::log10(std::abs(lastErr))
                                    : -100.0f;
  } else {
    mCurrentStats.filterLength = 0;
    mCurrentStats.muMax = 0.0f;
    mCurrentStats.muEffective = 0.0f;
    mCurrentStats.lastErrorDb = -100.0f;
    mCurrentStats.instantCorrelation = 0.0f;
  }
  return mCurrentStats;
}

// VSS-NLMS parameter control
void AdaptiveEchoCancellation::setVssMuMax(float mu) {
  for (auto &filter : mVssFilters) {
    filter->setStepSize(mu);
  }
  aecLog("[AEC] Set VSS mu_max=%.4f for %zu filters\n", mu, mVssFilters.size());
}

void AdaptiveEchoCancellation::setVssLeakage(float lambda) {
  for (auto &filter : mVssFilters) {
    filter->setLeakage(lambda);
  }
  aecLog("[AEC] Set VSS leakage=%.6f for %zu filters\n", lambda,
         mVssFilters.size());
}

void AdaptiveEchoCancellation::setVssAlpha(float alpha) {
  for (auto &filter : mVssFilters) {
    filter->setSmoothingFactor(alpha);
  }
  aecLog("[AEC] Set VSS alpha=%.4f for %zu filters\n", alpha,
         mVssFilters.size());
}

float AdaptiveEchoCancellation::getVssMuMax() const {
  return mVssFilters.empty() ? 0.0f : mVssFilters[0]->getMuMax();
}

float AdaptiveEchoCancellation::getVssLeakage() const {
  return mVssFilters.empty() ? 1.0f : mVssFilters[0]->getLeakage();
}

float AdaptiveEchoCancellation::getVssAlpha() const {
  return mVssFilters.empty() ? 0.95f : mVssFilters[0]->getAlpha();
}

// Filter length control
void AdaptiveEchoCancellation::setFilterLength(int length) {
  // Validate power of 2 and reasonable range
  if (length < 256 || length > 16384) {
    aecLog("[AEC] Invalid filter length %d (must be 256-16384)\n", length);
    return;
  }

  // Resize all VSS filters
  for (auto &filter : mVssFilters) {
    filter->resize(static_cast<size_t>(length));
  }
  // Resize the shadow-filter background filters too — previously skipped, which
  // left promotion copying mismatched-length weights (silently truncated by
  // setWeights' std::min). Must stay in lockstep with the foreground.
  for (auto &filter : mBgFilters) {
    filter->resize(static_cast<size_t>(length));
  }
  // Also resize regular NLMS filters for consistency
  for (auto &filter : mFilters) {
    filter->resize(static_cast<size_t>(length));
  }
  // Re-size + invalidate the candidate EMA buffers to the new (SIMD-rounded)
  // length, or the per-block blend / coeff scan would index a stale length.
  for (unsigned int ch = 0; ch < mShadow.size(); ++ch) {
    size_t fl = (ch < mBgFilters.size())
                    ? mBgFilters[ch]->getFilterLength()
                    : static_cast<size_t>(length);
    mShadow[ch].candWeights.assign(fl, 0.0f);
    mShadow[ch].primed = 0;
    mShadow[ch].winStreak = 0;
  }

  aecLog("[AEC] Set filter length=%d for %zu filters\n", length,
         mVssFilters.size());
}

int AdaptiveEchoCancellation::getFilterLength() const {
  return mVssFilters.empty()
             ? 0
             : static_cast<int>(mVssFilters[0]->getFilterLength());
}

void AdaptiveEchoCancellation::setCaptureFrameCount(size_t captureFrameCount) {
  mCaptureFrameCount = captureFrameCount;
}

void AdaptiveEchoCancellation::setCalibratedOffset(int64_t offset) {
  mCalibratedOffset = offset;
  mUsePositionSync = true; // Enable position sync when offset is set
  aecLog("[AEC] Set calibrated offset=%lld, position sync enabled\n",
         (long long)offset);
}

// Calibration capture methods
void AdaptiveEchoCancellation::startCalibrationCapture(size_t maxSamples) {
  mAlignedRefCapture.clear();
  mAlignedMicCapture.clear();
  mAlignedRefCapture.reserve(maxSamples);
  mAlignedMicCapture.reserve(maxSamples);
  mCalibrationMaxSamples = maxSamples;
  mCalibrationCaptureEnabled = true;
  aecLog("[AEC] Calibration capture started (max %zu samples)\n", maxSamples);
}

void AdaptiveEchoCancellation::stopCalibrationCapture() {
  mCalibrationCaptureEnabled = false;
  aecLog("[AEC] Calibration capture stopped: ref=%zu mic=%zu samples\n",
         mAlignedRefCapture.size(), mAlignedMicCapture.size());
}

bool AdaptiveEchoCancellation::isCalibrationCaptureComplete() const {
  return !mCalibrationCaptureEnabled &&
         mAlignedRefCapture.size() >= mCalibrationMaxSamples;
}

void AdaptiveEchoCancellation::setAecMode(AecMode mode) {
  // A/B hygiene: entering or leaving LSAEC clears the learned template so each
  // comparison run converges from scratch (no stale echo estimate).
  if (mEchoTemplate && (mode == aecModeLsaec) != (mAecMode == aecModeLsaec)) {
    mEchoTemplate->reset();
  }
  mAecMode = mode;

  // Mode names for logging
  const char *modeNames[] = {"Bypass", "Algo(Adaptive)", "Neural", "Hybrid(Adaptive+Neural)",
                              "Frozen(FIR)", "FrozenNeural(FIR+Neural)"};
  const char *modeName = (mode >= 0 && mode <= 5) ? modeNames[mode] : "Unknown";
  aecLog("[AEC] Mode set to %d (%s)\n", static_cast<int>(mode), modeName);

  // Update VSS filter frozen state
  bool shouldFreeze = (mode == aecModeFrozen || mode == aecModeFrozenNeural);
  for (auto &filter : mVssFilters) {
    filter->setFrozen(shouldFreeze);
    if (!shouldFreeze) {
      // Restore adaptive settings
      filter->setStepSize(0.1f);
      filter->setSmoothingFactor(0.05f);
      filter->setLeakage(0.9999f);
    }
  }

  // Update neural filter state
  bool neuralEnabled = (mode == aecModeNeural || mode == aecModeHybrid ||
                        mode == aecModeFrozenNeural);
  if (mNeuralFilter) {
    mNeuralFilter->setEnabled(neuralEnabled);
  }
}

AecMode AdaptiveEchoCancellation::getAecMode() const { return mAecMode; }

bool AdaptiveEchoCancellation::loadNeuralModel(const std::string &modelPath) {
  if (mNeuralFilter) {
    return mNeuralFilter->loadModel(modelPath);
  }
  return false;
}
