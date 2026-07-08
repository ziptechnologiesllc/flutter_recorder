#ifndef AEC_SYNCHRONOUS_ECHO_TEMPLATE_H
#define AEC_SYNCHRONOUS_ECHO_TEMPLATE_H

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

/**
 * SynchronousEchoTemplate — the core of LSAEC (Loop-Synchronous AEC).
 *
 * WHY this exists (and why it replaces the NLMS/shadow/promotion stack):
 * CloudLoop plays back a loop of KNOWN period P (frames) and controls its
 * timing, so the speaker output — and therefore the acoustic echo of it — is
 * exactly PERIODIC at P. The performer's live overdub is NOT periodic at P.
 * That single fact is a separation axis a performer physically cannot defeat.
 *
 * So instead of solving a moving, ill-conditioned linear inverse (an adaptive
 * filter, which on a periodic music reference wanders and diverges — the
 * "recording grows over time" failure), we keep one echo estimate PER LOOP
 * PHASE: E[phi], phi in [0, P). Each loop pass, E[phi] is nudged toward the mic
 * by a small leaky average. The period-coherent echo ACCUMULATES; the
 * period-incoherent performance AVERAGES TOWARD ZERO (~sqrt(#periods)). The
 * cancelled output is simply mic - E[phi].
 *
 * There are NO filter weights, so unbounded weight inflation — the entire
 * divergence failure mode — is structurally impossible. The acoustic delay D
 * never enters the learning: the echo is periodic at P whatever the delay, so
 * E indexed by the mic's loop phase captures it directly.
 *
 * Single-threaded: process() runs on the capture/audio thread. No allocation
 * after construction (capacity is sized once for the maximum supported period).
 *
 * CONVERGENCE SEED (fast-start): plain per-pass EMA learning is fundamentally
 * rate-limited — you only get ONE update per phase per loop pass, and the
 * update size (alpha) is capped for noise-floor reasons (push it too hot and
 * a single loud performer moment gets baked into E permanently). That's why
 * "converge in 3 loops" cannot be reached safely by tuning alpha/tau alone.
 *
 * The real fix: the room/device echo path h[] is exactly what a live AEC
 * calibration already measures (see calibration.cpp's analyzeAligned — a
 * causally-offset FIR trained on a delay-aligned chirp, the SAME delay
 * convention as the `alignedRef` this class receives every callback). Since
 * the reference is exactly periodic at P, the TRUE steady-state echo is the
 * CIRCULAR convolution of h with one period of the reference — not an
 * approximation, an exact closed form, PROVIDED the kernel is truncated to
 * at most P taps (a period-P circular convolution with a longer kernel wraps
 * multiple taps onto the same output phase, which measurably crashed ERLE to
 * -30 dB on-device on this feature's first live trigger — see
 * computeSeedConvolution's truncation + defensive amplitude clamp). So:
 * capture one full period of
 * `alignedRef` after a loop-period change, convolve it with the calibrated
 * h off the audio thread, and drop the result straight into E[phi] with high
 * initial confidence. Cancellation starts near its converged depth on loop 2
 * (loop 1 is spent capturing the reference) instead of after 8-16 passes.
 * Ongoing per-pass learning still runs afterward to track small deviations
 * (device moved, speaker warmed up) — the governor (spectral_governor.h)
 * re-heats it on demand exactly as before; the seed just removes the initial
 * "many empty loops while it learns from scratch" tax entirely.
 */
class SynchronousEchoTemplate {
public:
  SynchronousEchoTemplate(unsigned int sampleRate, unsigned int channels);
  ~SynchronousEchoTemplate();

  /**
   * Supply the calibrated room/device impulse response (same taps passed to
   * AdaptiveEchoCancellation::setImpulseResponse). Copies the coefficients;
   * safe to call from any thread (calibration runs off the audio thread).
   * The NEXT loop-period change (new base loop, layer add) will capture one
   * period of reference and seed the template from it — see class comment.
   */
  void setSeedImpulseResponse(const float *coeffs, int length);

  /**
   * Learning-rate multiplier from the spectral governor (>=1). Applied on top
   * of the annealed alpha, capped at the hot ceiling — measured echo leakage
   * re-heats learning; convergence lets it anneal back down. RT-safe setter
   * (relaxed atomic store from the render thread itself).
   */
  void setLearnBoost(float b) {
    mLearnBoost.store(b < 1.0f ? 1.0f : b, std::memory_order_relaxed);
  }

  /**
   * The audible mix changed WITHOUT a loop-period change (a track was
   * muted/unmuted/paused/stopped). E[phi] is indexed by loop phase, not by
   * mix content, so it keeps cancelling against the OLD mix shape until
   * ordinary per-pass learning slowly corrects it. This re-arms the same
   * convergence-seed capture used for a brand-new loop period — recapture
   * one period of the NEW reference, reconvolve with the calibrated room IR
   * — WITHOUT wiping the existing template first (unaffected phases keep
   * cancelling normally during the ~1-period gap until the reseed lands).
   * RT-safe from any thread: sets a flag; process() performs the actual arm
   * on the audio thread, where alignedRef is available. Safe/cheap to call
   * often — a no-op while a seed job is already in flight.
   */
  void notifyReferenceChanged() {
    mReferenceChangePending.store(true, std::memory_order_relaxed);
  }

  /** Clear all learned echo (on enable/disable/reset). */
  void reset();

  /**
   * Cancel one interleaved block IN PLACE.
   *
   * @param micInOut       Interleaved mic in; cleaned (mic - echo) out.
   * @param alignedRef     Interleaved echo-aligned reference for far-end
   *                       gating (only LEARN where the speaker is playing).
   *                       May be null -> always learn.
   * @param frameCount     Frames in this block.
   * @param channels       Interleave width (must be <= construction channels).
   * @param blockStartFrame Engine frame counter of the first sample (same
   *                       coordinate as loopStartFrame).
   * @param loopFrames     Loop period P in frames (0 or >capacity -> passthrough).
   * @param loopStartFrame Engine frame where loop phase 0 falls.
   * @param learn          When false, cancel but do NOT update the template
   *                       (double-talk / transient freeze; see LSAEC E3).
   */
  void process(float *micInOut, const float *alignedRef,
               unsigned int frameCount, unsigned int channels,
               int64_t blockStartFrame, int64_t loopFrames,
               int64_t loopStartFrame, bool learn);

  /** Active loop period actually in use (0 if passthrough). */
  int64_t activeLoopFrames() const { return mActiveLoopFrames; }

  /** Mean per-phase confidence over the active period (0..1), for telemetry. */
  float meanConfidence() const;

  /** E3 diagnostics (monotonic counters since reset). */
  uint32_t freezeCount() const { return mFreezeCount; }
  uint32_t reopenCount() const { return mReopenCount; }

  /** True while a convergence-seed capture/compute/apply is in flight — for
   * a debug overlay to distinguish "seeding" from "ordinary per-pass
   * learning" as the reason ERLE is moving (or not). Audio-thread-owned
   * state; safe to read from any thread as a snapshot (single bool read). */
  bool isSeeding() const { return mSeedBusy; }

  /** True while the loop period exceeds kMaxSeconds (16s) capacity — the
   * template falls back to pure passthrough (zero cancellation) in this
   * state, silently, until the period drops back under the cap. Surfaced so
   * a debug overlay / telemetry can show WHY cancellation stopped instead of
   * it looking like a mystery "ghost bleed". Audio-thread-owned; safe single-
   * bool snapshot read from any thread. */
  bool isOverCapacity() const { return mOverCapacity; }

private:
  unsigned int mSampleRate;
  unsigned int mChannels;
  size_t mCapacityFrames;          // max P we can hold (per channel)
  std::vector<float> mTemplate;    // E[phi*channels + ch] — echo estimate
  std::vector<float> mConfidence;  // per-phase saturating update count (anneal)
  int64_t mActiveLoopFrames = 0;   // P currently in use (<= capacity)

  // E3 block-level double-talk freeze. A smoothed per-BLOCK residual floor; a
  // block whose residual spikes well above it is near-end -> skip learning that
  // block (per-frame thresholding thrashed — audio energy is too spiky).
  // Smoothed reference power envelope for the far-end learn gate. Gating on
  // INSTANTANEOUS per-sample power leaves unlearned pinholes at every zero
  // crossing of the reference waveform (hundreds/sec) — the mic echo (delayed
  // ~26 ms, nonzero there) pokes through each hole every pass = broadband
  // crackle that WORSENS as the rest of the template converges.
  std::atomic<float> mLearnBoost{1.0f}; // spectral-governor gain

  float mRefEnvelope = 0.0f;

  float mResidBaseline = 0.0f; // EMA of per-block residual energy
  uint32_t mLearnedBlocks = 0; // non-frozen learning blocks (arms the detector)
  uint32_t mFreezeCount = 0;   // blocks frozen (near-end) — telemetry
  uint32_t mReopenCount = 0;   // reserved (block-level recovery) — telemetry
  bool mOverCapacity = false;  // loopFrames > mCapacityFrames — see isOverCapacity()

  // ---- Trust gate (fast, time-based — NOT loop-bound) -------------------
  // The reseed mechanism fixes a stale template correctly, but takes up to
  // one full loop pass (capture + compute + apply) to land — during which
  // the OLD template is still subtracted at full strength every callback.
  // If the mix change was a track going silent (mute/pause/stop), that
  // stale subtraction has nothing left to cancel against and instead
  // INJECTS a phase-inverted copy of whatever that track used to sound
  // like — confirmed on-device as a full recording replaced by a
  // phase-inverted ghost of a paused loop layer. This gate closes
  // IMMEDIATELY (frame-accurate, no loop-period wait) on any mix/period
  // change and ramps back to full strength over kTrustRampMs — independent
  // of and much faster than the reseed, which still runs in the background
  // and restores full per-phase accuracy once it lands. Global (not
  // per-phase): simple, RT-cheap, and correct enough since the window is
  // short — a brief, honest reduction in cancellation for still-playing
  // tracks beats a loud, wrong, phase-inverted injection every time.
  int64_t mMixChangeFrame = INT64_MIN / 2; // far enough back that gate() = 1.0 from frame 0
  int64_t mTrustRampFrames = 0;            // set from sampleRate in the constructor

  // 0 immediately after a mix/period change, ramping linearly to 1 over
  // mTrustRampFrames. audio-thread-only; cheap (one subtract/divide/clamp
  // per block, not per sample).
  float trustGate(int64_t blockStartFrame) const {
    if (mTrustRampFrames <= 0)
      return 1.0f;
    const int64_t elapsed = blockStartFrame - mMixChangeFrame;
    if (elapsed <= 0)
      return 0.0f;
    if (elapsed >= mTrustRampFrames)
      return 1.0f;
    return static_cast<float>(elapsed) / static_cast<float>(mTrustRampFrames);
  }

  // ---- Convergence seed (see class comment) ----------------------------
  // Calibrated IR, settable from any thread; read only when arming a capture.
  std::mutex mSeedIRMutex;
  std::vector<float> mSeedIR;

  // Audio-thread-only state machine (no atomics needed for these — single
  // writer/reader). mSeedBusy spans from arm to fully-applied; while true, a
  // NEW loop-period change will NOT re-arm (rare-edge-case: just falls back
  // to normal per-pass learning for that cycle, no correctness issue).
  bool mSeedBusy = false;
  int64_t mSeedCaptureRemaining = 0; // frames left to capture this period
  // Set by notifyReferenceChanged() (any thread); consumed by process() on
  // the audio thread, which is the only thread allowed to touch mSeedBusy et al.
  std::atomic<bool> mReferenceChangePending{false};
  // Monotonic, audio-thread-only "epoch" for the capture/job currently owned
  // by mSeedBusy. Bumped on every arm AND on abandonment (a period change
  // while a capture/job was in flight). A posted job is stamped with the
  // generation it belongs to; at drain time only a job whose stamp still
  // matches mSeedGeneration is applied — an abandoned job's late completion
  // is dropped WITHOUT touching mSeedBusy, which by then belongs to whatever
  // capture was armed after the abandonment. (Comparing loop PERIOD alone
  // isn't sufficient: a quick mute-then-unmute can return to the same P,
  // which would let a stale job masquerade as current.)
  int64_t mSeedGeneration = 0;
  int64_t mSeedJobGeneration = -1; // audio-thread-only; worker never reads this
  void armSeedCaptureIfPossible(const float *alignedRef); // audio-thread only
  std::vector<float> mRefCapture;    // mono, pre-sized to capacity — RT-safe

  // Handoff to the worker thread. mSeedJobPosted: audio thread sets true when
  // capture completes (release); worker polls it (acquire), clears it after
  // starting the job. mSeedOutputReady: worker sets true when done (release);
  // audio thread polls it (acquire) and chunks the apply across callbacks
  // (kSeedApplyChunk phases/callback) to keep any single callback O(1)
  // regardless of loop length, then clears it (relaxed, audio-thread-only).
  std::atomic<bool> mSeedJobPosted{false};
  std::atomic<bool> mSeedOutputReady{false};
  std::vector<float> mSeedOutput;    // mono, pre-sized to capacity
  int64_t mSeedJobPeriod = 0;        // P the posted/ready job corresponds to
  int64_t mSeedApplyPos = 0;         // chunked-apply cursor

  std::atomic<bool> mSeedThreadRunning{false};
  std::thread mSeedWorker;
  void seedWorkerLoop();
  void computeSeedConvolution(int64_t P); // worker thread only
};

#endif // AEC_SYNCHRONOUS_ECHO_TEMPLATE_H
