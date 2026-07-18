#ifndef ADAPTIVE_ECHO_CANCELLATION_H
#define ADAPTIVE_ECHO_CANCELLATION_H

#include "../../enums.h"
#include "../generic_filter.h"
#include "delay_estimator.h"
#include "drift_aligner.h"
#include "neural_post_filter.h"
#include "nlms_filter.h"
#include "reference_buffer.h"
#include "synchronous_echo_template.h"
#include "vss_nlms_filter.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "../../audio_engine/seqlock.h"
// AecTelemetrySnapshot is defined in enums.h (included above) so the FFI layer
// (filters.h / flutter_recorder.cpp) can see it without pulling in this header.

/**
 * Adaptive Echo Cancellation Filter using NLMS algorithm.
 *
 * This filter removes acoustic echo from the microphone input by:
 * 1. Reading the reference signal (speaker output) from a shared buffer
 * 2. Applying an adaptive NLMS filter to estimate the echo
 * 3. Subtracting the estimated echo from the microphone input
 *
 * The filter continuously adapts to changes in the acoustic environment.
 */
class AdaptiveEchoCancellation : public GenericFilter {
public:
  // Parameter indices
  enum Params {
    StepSize = 0, // NLMS adaptation rate
    DelayMs,      // Estimated acoustic delay in ms
    Enabled,      // Master enable/bypass
    ParamCount
  };

  /**
   * @param sampleRate Audio sample rate in Hz
   * @param channels Number of audio channels
   */
  AdaptiveEchoCancellation(unsigned int sampleRate, unsigned int channels = 2);
  ~AdaptiveEchoCancellation() override = default;

  // GenericFilter interface implementation
  int getParamCount() const override;
  float getParamMax(int param) const override;
  float getParamMin(int param) const override;
  float getParamDef(int param) const override;
  std::string getParamName(int param) const override;
  void setParamValue(int param, float value) override;
  float getParamValue(int param) const override;

  // Process audio through the AEC filter
  void process(void *pInput, ma_uint32 frameCount, unsigned int channels,
               ma_format format) override;

  // Process with explicit timestamp for synchronization
  void processWithTimestamp(void *pInput, ma_uint32 frameCount,
                            unsigned int channels, ma_format format,
                            AECReferenceBuffer::TimePoint timestamp);

  /**
   * Reset the filter state.
   */
  void reset();

  /**
   * Get the current echo return loss (ERL) in dB.
   * Higher values indicate better echo cancellation.
   */
  float getEchoReturnLoss() const;

  /**
   * LSAEC E5: lock-free gated-ERLE telemetry snapshot. Safe to read from any
   * thread (Dart poller). The audio thread is the single writer; all dB math
   * is done by the reader, never on the RT thread.
   */
  AecTelemetrySnapshot getTelemetry() const { return mTelemetry.load(); }

  /**
   * LSAEC: the audible mix changed without a loop-period change (a track was
   * muted/unmuted/paused/stopped). Re-arms the template's convergence-seed
   * capture for the CURRENT period so cancellation catches up in ~1 pass
   * instead of relearning via ordinary per-pass EMA. No-op if LSAEC isn't
   * active or a seed job is already in flight.
   */
  void notifyReferenceChanged() {
    if (mEchoTemplate)
      mEchoTemplate->notifyReferenceChanged();
  }

  /**
   * LSAEC per-track exact subtraction (see SynchronousEchoTemplate's doc
   * comment for the full rationale): register a track's own known audio so
   * its echo contribution can be computed once, off-thread, and later
   * added/subtracted from the template EXACTLY on mute/unmute instead of
   * relying on reactive suppression or a reseed race. No-op if LSAEC isn't
   * active.
   */
  void registerTrackAudio(int trackIndex, const float *audioMono, int64_t frames) {
    if (mEchoTemplate)
      mEchoTemplate->registerTrackAudio(trackIndex, audioMono, frames);
  }

  /**
   * Toggle a registered track's contribution in/out of the live template.
   * Call at the SAME sample-accurate instant the SoLoud mute/unmute/pause/
   * unpause/stop setter fires, so the AEC state change and the audible
   * state change happen atomically. No-op if LSAEC isn't active or the
   * track hasn't finished registering yet (falls back to the composite
   * template's existing behavior for that track in the meantime).
   */
  bool setTrackActive(int trackIndex, bool active) {
    return mEchoTemplate ? mEchoTemplate->setTrackActive(trackIndex, active)
                         : false;
  }

  /** Exact per-track gain edit; true = mix change fully accounted for (the
   * caller can skip the reference-changed reseed). */
  bool setTrackGain(int trackIndex, float gain) {
    return mEchoTemplate ? mEchoTemplate->setTrackGain(trackIndex, gain)
                         : false;
  }

  /**
   * Release a track's per-track AEC slot — call when a loop is deleted so
   * a long session doesn't exhaust the fixed-size slot table with
   * contributions for loops that no longer exist. No-op if LSAEC isn't
   * active or the track was never registered.
   */
  void releaseTrackContribution(int trackIndex) {
    if (mEchoTemplate)
      mEchoTemplate->releaseTrackSlot(trackIndex);
  }

  /**
   * Enable/disable the LSAEC Stage-2 nonlinear HF residual-echo suppressor
   * (the post-filter that cleans the high-frequency ghost / metronome click
   * linear cancellation can't reach). Exposed for on-device A/B — linear-only
   * vs. linear+suppressor on the same take. No-op if LSAEC isn't active.
   */
  void setResidualSuppressorEnabled(bool enabled) {
    if (mEchoTemplate)
      mEchoTemplate->setResidualSuppressorEnabled(enabled);
  }
  bool residualSuppressorEnabled() const {
    return mEchoTemplate ? mEchoTemplate->residualSuppressorEnabled() : false;
  }

  /** Live-tune the LSAEC subtraction gate (attack ms / release ms / floor dB)
   * — see SynchronousEchoTemplate::setSubGateTuning. No-op without LSAEC. */
  void setSubGateTuning(float attackMs, float releaseMs, float floorDb) {
    if (mEchoTemplate)
      mEchoTemplate->setSubGateTuning(attackMs, releaseMs, floorDb);
  }

  /**
   * Set the impulse response from calibration.
   * Pre-initializes NLMS filter coefficients for immediate cancellation.
   *
   * @param coeffs Impulse response coefficients
   * @param length Number of coefficients
   */
  /**
   * Set the impulse response from calibration.
   */
  void setImpulseResponse(const float *coeffs, int length);

  /**
   * Measure hardware latency using cross-correlation.
   * Updates the DelayMs parameter automatically.
   * @param refBuffer Reference signal buffer (1-2 seconds)
   * @param micBuffer Microphone signal buffer (1-2 seconds)
   * @return Measured delay in milliseconds
   */
  float measureHardwareLatency(const std::vector<float> &refBuffer,
                               const std::vector<float> &micBuffer);

  // Stats
  AecStats getStats();
  void updateStats(float ref, float mic, float out);

  NeuralPostFilter *getNeuralFilter() { return mNeuralFilter.get(); }

  // VSS-NLMS parameter control for experimentation
  void setVssMuMax(float mu);
  void setVssLeakage(float lambda);
  void setVssAlpha(float alpha);
  float getVssMuMax() const;
  float getVssLeakage() const;
  float getVssAlpha() const;

  // Filter length control
  void setFilterLength(int length);
  int getFilterLength() const;

  // Sample-accurate synchronization (frame counter based)
  // Call this BEFORE process() with the capture frame count at block start
  void setCaptureFrameCount(size_t captureFrameCount);

  // Set the calibrated offset: captureFrame - offset = corresponding
  // outputFrame This is calculated during calibration as: offset =
  // (captureFramesAtCalib - outputFramesAtCalib) + acousticDelaySamples
  void setCalibratedOffset(int64_t offset);
  int64_t getCalibratedOffset() const { return mCalibratedOffset; }

  // Set pure acoustic delay in samples (for slave mode where thread timing is irrelevant)
  void setAcousticDelaySamples(size_t samples) { mAcousticDelaySamples = samples; }
  size_t getAcousticDelaySamples() const { return mAcousticDelaySamples; }

  // Set buffer configuration for theoretical delay calculation
  void setBufferConfig(size_t bufferSizeFrames, size_t pipelinePeriods = 3) {
    mBufferSizeFrames = bufferSizeFrames;
    mPipelinePeriods = pipelinePeriods;
  }

  // Calculate theoretical delay based on buffer config
  // Full round-trip delay = output buffering + acoustic path + input buffering
  // For duplex device: both input and output have mPipelinePeriods of buffering
  size_t getTheoreticalDelaySamples() const {
    // Output buffering (DAC pipeline): N periods
    size_t outputLatency = mPipelinePeriods * mBufferSizeFrames;
    // Input buffering (ADC pipeline): typically same as output
    size_t inputLatency = mPipelinePeriods * mBufferSizeFrames;
    // Acoustic delay: speaker → air → mic (3-5ms typical for laptop)
    size_t acousticDelay = (mSampleRate * 4) / 1000; // 4ms default
    return outputLatency + inputLatency + acousticDelay;
  }

  // Enable/disable position-based sync (vs legacy timestamp/delay based)
  void setUsePositionSync(bool enable) { mUsePositionSync = enable; }
  bool getUsePositionSync() const { return mUsePositionSync; }

  // Calibration capture: capture frame-aligned ref/mic for delay estimation
  void startCalibrationCapture(size_t maxSamples = 96000); // 2 seconds @ 48kHz
  void stopCalibrationCapture();
  const std::vector<float> &getAlignedRef() const { return mAlignedRefCapture; }
  const std::vector<float> &getAlignedMic() const { return mAlignedMicCapture; }
  bool isCalibrationCaptureComplete() const;

  // AEC Mode Control (A/B Testing)
  void setAecMode(AecMode mode);
  AecMode getAecMode() const;

  /**
   * Loads a TFLite model for the neural post-filter.
   * @param modelPath Path to the .tflite model file.
   * @return true if successful.
   */
  bool loadNeuralModel(const std::string &modelPath);

private:
  struct ParamRange {
    float defaultVal;
    float minVal;
    float maxVal;
  };

  unsigned int mSampleRate;
  unsigned int mChannels;

  // Parameter storage
  std::map<Params, ParamRange> mParams;
  std::vector<float> mValues;

  // NLMS filter instances (one per channel)
  std::vector<std::unique_ptr<NLMSFilter>> mFilters;
  // VSS-NLMS foreground filters (frozen — produce the cancelled output).
  std::vector<std::unique_ptr<VssNlmsFilter>> mVssFilters;
  // Shadow filter (Phase A): background filters adapt recklessly in raw mode.
  // Promoted into the foreground only when they measurably beat it — so a
  // background corrupted by double-talk is simply never promoted.
  std::vector<std::unique_ptr<VssNlmsFilter>> mBgFilters;
  // Shadow-filter Phase A refinement (settled snapshot / Polyak-Ruppert tail
  // averaging). We no longer promote the background's instantaneous (gradient-
  // noisy) weights. Instead we keep a per-channel EMA of the background weights
  // (candWeights) — the candidate — and PROMOTE it only when its OWN held-out
  // error (blockCandErr, scored against the bg history via scoreAgainstHistory)
  // beats the frozen foreground for several consecutive blocks. The EMA only
  // ingests the background once that channel has SETTLED (mean annealed step
  // ratio near the floor), which keeps high-misadjustment early iterates out of
  // the average and doubles as double-talk protection (near-end raises the
  // residual, raising the ratio, halting ingest).
  //
  // State is PER CHANNEL: capture channels are acoustically independent and
  // promote independently, so a slow-to-settle channel never starves a
  // converged one (the previous global gate did). One struct per channel keeps
  // the EMA buffer and its block accumulators together — no parallel vectors to
  // desync.
  struct ShadowChannel {
    std::vector<float> candWeights; // EMA of bg weights — scored & promoted
    uint8_t primed = 0;             // EMA seeded from a settled bg at least once
    int winStreak = 0;              // consecutive blocks the candidate beat fg
    double blockFgErr = 0.0;        // foreground error energy this block
    double blockBgErr = 0.0;        // background error energy this block (diagnostic)
    double blockCandErr = 0.0;      // held-out candidate error this block (the gate)
    double blockRefEnergy = 0.0;    // reference energy this block (silent-block floor)
    uint8_t blockCandBlown = 0;     // candidate scoring saw |y|>8 / non-finite
  };
  std::vector<ShadowChannel> mShadow;
  size_t mBlockSamples = 0; // frames accumulated toward the next promotion (shared)

  // Drift-compensated reference aligner (DCRA). Capture & playback are on
  // independent clocks (miniaudio bridges them drift-free), so the reference
  // must be read at a continuously drift-corrected fractional position or the
  // echo path walks and the filter cannot converge. Replaces the fixed integer
  // reference read in the slave-mode path.
  std::unique_ptr<DriftAligner> mDriftAligner;

  // LSAEC core: loop-synchronous echo template. In slave mode with a known loop
  // period it REPLACES the NLMS/shadow/promotion stack — it cannot diverge
  // (no weights) and rejects the period-incoherent performer by construction.
  std::unique_ptr<SynchronousEchoTemplate> mEchoTemplate;

  // Delay in samples for reference signal alignment (fallback if no timestamp)
  unsigned int mDelaySamples;

  // Temporary buffer for reference signal
  std::vector<float> mRefBuffer;
  // Temporary buffer for linear AEC output
  std::vector<float> mLinearOutputBuffer;

  // Timestamp-based synchronization
  bool mUseTimestampSync;
  AECReferenceBuffer::TimePoint mCurrentCallbackTimestamp;

  std::unique_ptr<NeuralPostFilter> mNeuralFilter;

  AecStats mCurrentStats = {0};

  // --- LSAEC E5 gated-ERLE telemetry (lock-free; published per ~0.25 s window).
  // The audio thread accumulates windowed energy SUMS (adds/compares only) and
  // publishes a snapshot via the seqlock; a Dart poller computes dB off-thread.
  flowstate::audio_engine::Seqlock<AecTelemetrySnapshot> mTelemetry;
  double mTwMicFar = 0.0, mTwOutFar = 0.0;                   // far-end-active sums
  double mTwMicAll = 0.0, mTwOutAll = 0.0, mTwRefAll = 0.0;  // all-sample sums
  uint64_t mTwFar = 0, mTwTotal = 0;                         // window sample counts
  uint64_t mTelemetryGen = 0;                                // published-window counter
  float mRefPowerEma = 0.0f;                                 // smoothed ref power (far gate)

  // Sample-accurate sync state
  size_t mCaptureFrameCount = 0; // Set before each process() call
  int64_t mCalibratedOffset = 0; // Capture frame - offset = output frame
  bool mUsePositionSync = false; // Use position-based sync vs legacy delay
  size_t mAcousticDelaySamples = 0; // Pure acoustic delay (for slave mode)

  // Buffer configuration for theoretical delay calculation
  size_t mBufferSizeFrames = 128; // Default audio buffer size
  size_t mPipelinePeriods = 3;    // Typical pipeline depth for PipeWire/ALSA

  // Calibration capture state (for frame-aligned delay estimation)
  bool mCalibrationCaptureEnabled = false;
  size_t mCalibrationMaxSamples = 0;
  std::vector<float> mAlignedRefCapture;
  std::vector<float> mAlignedMicCapture;

  // AEC Mode
  AecMode mAecMode = aecModeHybrid;

  void validateParam(int param) const;
  void updateDelay();

  // Templated processing for different sample formats
  template <typename T>
  void processAudio(void *pInput, ma_uint32 frameCount, unsigned int channels);

  // Format conversion helpers
  float normalizeSample(unsigned char sample);
  float normalizeSample(int16_t sample);
  float normalizeSample(int32_t sample);
  float normalizeSample(float sample);

  template <typename T> T denormalizeSample(float sample);
};

#endif // ADAPTIVE_ECHO_CANCELLATION_H
