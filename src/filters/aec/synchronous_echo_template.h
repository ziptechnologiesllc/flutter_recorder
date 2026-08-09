#ifndef AEC_SYNCHRONOUS_ECHO_TEMPLATE_H
#define AEC_SYNCHRONOUS_ECHO_TEMPLATE_H

#include <array>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "spectral_residual_suppressor.h"

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

  /**
   * Live-tune the reference-presence SUBTRACTION gate (see mSubGateEnv doc):
   * attack/release of the far-end envelope and the soft-knee floor. Exposed
   * because the release+knee pair sets how long cancellation hangs on after
   * content stops (the audible "tail"), and the right value is a per-room,
   * by-ear judgment — hand-tuned live from the AEC debug panel. RT-safe
   * (relaxed atomics; process() snapshots once per block).
   *
   * @param attackMs  envelope attack time constant, ms (default 1.0)
   * @param releaseMs envelope release time constant, ms (default ~69)
   * @param floorDb   soft-knee floor in dB power (default -54): gate is ~0.5
   *                  AT the floor, ~1 well above, ~0 well below. Raising it
   *                  closes the gate sooner as content decays.
   */
  void setSubGateTuning(float attackMs, float releaseMs, float floorDb);

  /**
   * OUTPUT-clock -> alignedRef time shift LAMBDA (frames): alignedRef(t) =
   * out(t - LAMBDA), published per block by the AEC's reference read
   * (effectiveDelay + frameCount). Registered track audio lives on the
   * OUTPUT clock (sample 0 plays at loop phase 0), while the template —
   * like the captured-alignedRef seed — is indexed by MIC phase where the
   * echo actually appears. Contribution jobs rotate registered audio right
   * by LAMBDA before convolving so exact edits land at the same phases the
   * seed/learning produce. RT-safe relaxed store.
   */
  void setReferenceShiftFrames(int64_t frames) {
    mReferenceShiftFrames.store(frames, std::memory_order_relaxed);
  }

  /** Seed lifecycle counters (monotonic since construction) — release-visible
   * convergence diagnostics. A high aborts:arms ratio with few lands means
   * mix-change notifies keep killing the capture (seed livelock) and
   * convergence is riding pure per-pass EMA. */
  uint32_t seedArms() const { return mSeedArms.load(std::memory_order_relaxed); }
  uint32_t seedDiscards() const { return mSeedDiscards.load(std::memory_order_relaxed); }
  float seedLastAlpha() const { return mSeedLastAlpha.load(std::memory_order_relaxed); }
  uint32_t seedAborts() const { return mSeedAborts.load(std::memory_order_relaxed); }
  uint32_t seedLands() const { return mSeedLands.load(std::memory_order_relaxed); }

  /** Which phase the in-flight seed is stuck in (racy snapshot, telemetry
   * only): 0 idle, 1 capturing reference, 2 job posted (worker pending),
   * 3 output ready (fit/apply in progress), 4 busy-but-none (anomalous). */
  int64_t alignLagCorrection() const {
    return mAlignLagCorrection.load(std::memory_order_relaxed);
  }

  uint32_t seedPhase() const {
    if (!mSeedBusy) return 0;
    if (mSeedCaptureRemaining > 0) return 1;
    if (mSeedJobPosted.load(std::memory_order_relaxed)) return 2;
    if (mSeedOutputReady.load(std::memory_order_relaxed)) return 3;
    return 4;
  }

  /** UI feed for the gate overlay on the scrolling monitor: the smoothed
   * far-end envelope the gate tracks (linear power) and the resulting gate
   * opening (0..1, block-smoothed). Racy single-float reads by design —
   * audio thread writes, UI snapshots at ~4 Hz. */
  float subGateEnv() const { return mSubGateEnv; }
  float refGateOpen() const { return mRefGateSmooth; }

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

  /**
   * Enable/disable the nonlinear HF residual-echo suppressor (Stage 2). The
   * linear template (Stage 1) is untouched either way — this only gates the
   * post-filter that cleans the high-frequency ghost / metronome click linear
   * cancellation structurally leaves behind. RT-safe; default enabled.
   * Exposed for on-device A/B (a linear-only vs. linear+suppressor compare).
   */
  void setResidualSuppressorEnabled(bool e) { mSuppressor.setEnabled(e); }
  bool residualSuppressorEnabled() const { return mSuppressor.enabled(); }
  /** Per-HF-band duck gain (0..1) and learned residual-echo coupling, ch 0,
   * for a debug overlay. b in [0, SpectralResidualSuppressor::kBands). */
  float suppressorBandGain(int b) const { return mSuppressor.bandGain(b); }
  float suppressorBandKappa(int b) const { return mSuppressor.bandKappa(b); }

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

  /**
   * PER-TRACK EXACT SUBTRACTION — the real fix for mute/unmute ghosts.
   *
   * Every reactive attempt to catch a stale-template ghost after the fact
   * (a time-based trust gate, then three iterations of an energy-ratio
   * suppression gate) failed on-device: an aggregate-energy heuristic
   * cannot reliably distinguish "this track's echo is now gone" from
   * "this track's echo is just quiet", because it only ever sees a TOTAL,
   * never an attribution. The fix is to not need the heuristic: since every
   * track's audio is fully known and controlled by the app (a recorded
   * loop, not a live unknown signal), that track's OWN echo contribution
   * can be computed once, analytically, and then literally ADDED or
   * SUBTRACTED from E[phi] the instant it mutes/unmutes — an exact
   * arithmetic edit, not an estimate that needs to reconverge or a
   * suppression that needs to guess.
   *
   * registerTrackAudio(): call once a track's audio is known (e.g. right
   * after it's loaded into the mixer), from ANY thread. Computes E_track
   * off-thread (reuses computeSeedConvolution's exact math — circular
   * convolution of the calibrated IR against this track's own audio,
   * instead of a live-captured reference) via the existing seed worker.
   * ASSUMES the track's period equals the CURRENT composite loop period
   * (P) and that audioMono is already phase-aligned to the loop's phase-0
   * origin — both true by construction for a quantized-to-loop-boundary
   * recording in this app's existing architecture. A track whose own
   * period is a strict sub-multiple of P, or isn't yet aligned, is future
   * work — falls back to the ordinary composite-template path (harmless,
   * just not instant) until then.
   *
   * setTrackActive(): call at the SAME sample-accurate instant the
   * SoLoud mute/unmute/pause/unpause/stop setter fires (audio thread).
   * O(P) plain addition — cheap enough (a few hundred thousand float
   * adds, no convolution) to run synchronously, unlike registration. A
   * no-op if the track was never registered or isn't computed yet, in
   * which case cancellation falls back to whatever the composite template
   * (ordinary per-pass learning / the shared reseed) already provides —
   * this mechanism is additive, never a regression from today's behavior.
   */
  void registerTrackAudio(int trackIndex, const float *audioMono, int64_t frames);
  /**
   * Toggle a track's contribution in/out of the template at its CURRENT
   * gain. Returns true when the exact edit was applied — the caller can
   * then skip the reference-changed reseed entirely (the template is
   * already correct; re-arming the one-period capture would only abort any
   * in-flight seed for nothing: the seed-livelock this return kills).
   */
  bool setTrackActive(int trackIndex, bool active);
  /**
   * Exact gain edit: template += (gain - appliedGain) * E_track for an
   * active track (instant, O(P)); for an inactive/unready track just
   * records the target gain for the next activation. Returns true when the
   * mix change is fully accounted for (same skip-the-reseed contract as
   * setTrackActive). Audio thread only.
   */
  bool setTrackGain(int trackIndex, float gain);
  // Release trackIndex's slot (if any) back to the pool — call when a
  // track is deleted so a long session doesn't exhaust all 64 slots with
  // contributions for loops that no longer exist. Any thread; lock-free.
  void releaseTrackSlot(int trackIndex);

private:
  unsigned int mSampleRate;
  unsigned int mChannels;

  // Stage 2: nonlinear HF residual-echo suppressor. Runs AFTER the linear
  // template subtraction in Pass 1, on the residual, to clean the
  // high-frequency ghost / metronome click that linear cancellation can't
  // reach (see the class's own doc comment for the full rationale). Zero
  // added latency at unity gain, so it's safe in the record path.
  SpectralResidualSuppressor mSuppressor;

  size_t mCapacityFrames;          // max P we can hold (per channel)
  std::vector<float> mTemplate;    // E[phi*channels + ch] — echo estimate
  std::vector<float> mConfidence;  // per-phase saturating update count (anneal)
  // Incrementally-maintained sum of mConfidence over the CURRENT active
  // window [0, mActiveLoopFrames) — every write to mConfidence within that
  // window must update this by the delta, and every place that clears/
  // resizes the window must reset it to match. meanConfidence() reads this
  // instead of re-summing (was an O(min(P,capacity)) loop — up to 768,000
  // iterations for a 16s/48kHz loop — called unconditionally on the audio
  // thread ~4x/sec for E5 telemetry, live in every production build).
  double mConfidenceSum = 0.0;
  int64_t mActiveLoopFrames = 0;   // P currently in use (<= capacity)
  // Cross-thread-safe mirror of mActiveLoopFrames, updated at the same
  // write sites. computeTrackContribution() runs on the worker thread and
  // needs to know the CURRENT phase period to fold a registered track's
  // raw audio (which may span multiple base-loop periods, or — per a
  // separate recording-pipeline bug — occasionally land on a non-multiple
  // length) down to exactly one period before convolving; reading the
  // plain mActiveLoopFrames from that thread would be a data race.
  std::atomic<int64_t> mActiveLoopFramesAtomic{0};
  // See setReferenceShiftFrames(). Read by the worker thread when computing
  // per-track contributions.
  std::atomic<int64_t> mReferenceShiftFrames{0};

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

  // Reference-presence gate for the SUBTRACTION (distinct from mRefEnvelope,
  // which gates LEARNING). E[phi] is a STORED per-phase echo estimate that
  // Pass 1 subtracts unconditionally — but a stored echo is only correct
  // while the speaker is actually reproducing the content that created it.
  // The instant the far-end goes silent (every track muted/paused, quiet
  // room), there is no echo present, yet E[phi] is still full of the echo it
  // learned while those tracks played — and learning is gated OFF by that
  // same silence, so it can never self-correct. Subtracting E[phi] from a
  // silent mic then EMITS -E[phi]: a phase-inverted ghost of the old echo,
  // baked into the whole take (the user's "ghosts when everything muted").
  // The fix is textbook near-end/far-end gating: scale the subtraction by
  // how much far-end signal is ACTUALLY present right now (a smoothed
  // envelope of the aligned reference). Speaker playing -> full subtraction
  // (cancellation unchanged); speaker silent -> subtract nothing (output =
  // mic, no ghost). Slow release (~reverb time) so the echo TAIL keeps
  // cancelling as the reference decays. Uses the KNOWN reference, not an
  // output-energy heuristic — this is the principled version of what the
  // block-level suppression gate only approximated.
  float mSubGateEnv = 0.0f;

  // Live-tunable subtraction-gate parameters (see setSubGateTuning). Stored
  // as the raw per-sample EMA rates / linear power the hot loop consumes;
  // defaults match the previously hardcoded constants (1 ms attack, ~69 ms
  // release, -54 dB floor — a touch above the learning floor so the gate
  // closes before learning would). Relaxed atomics: any thread may tune,
  // process() snapshots once per call.
  std::atomic<float> mSubGateAttackRate{0.02f};
  std::atomic<float> mSubGateReleaseRate{0.0003f};
  std::atomic<float> mSubGateFloorPow{4e-6f};

  // Seed lifecycle counters (see seedArms()/seedAborts()/seedLands()).
  std::atomic<uint32_t> mSeedArms{0};
  // Fit-discard diagnostics: a completed seed whose fitted alpha fell below
  // the trust floor (|alpha|<0.05 => uncorrelated with the live mic --
  // stale/misaligned calibration IR). Invisible before these: the discard
  // cleared mSeedBusy without touching arms/aborts/lands, which read as
  // "seed vanished".
  std::atomic<uint32_t> mSeedDiscards{0};
  std::atomic<float> mSeedLastAlpha{0.0f};
  std::atomic<uint32_t> mSeedAborts{0};
  std::atomic<uint32_t> mSeedLands{0};

  // Block-smoothed gate opening (0..1) for the UI overlay — last frame's
  // refGate, EMA'd once per block. Audio-thread-owned; UI reads racily.
  float mRefGateSmooth = 1.0f;

  float mResidBaseline = 0.0f; // EMA of per-block residual energy
  uint32_t mLearnedBlocks = 0; // non-frozen learning blocks (arms the detector)
  uint32_t mFreezeCount = 0;   // blocks frozen (near-end) — telemetry
  uint32_t mReopenCount = 0;   // reserved (block-level recovery) — telemetry
  bool mOverCapacity = false;  // loopFrames > mCapacityFrames — see isOverCapacity()

  // ---- Cancel-pass safety suppression -----------------------------------
  // Pass 1 caps the OUTPUT/recording's energy to never exceed the raw mic's
  // — see process()'s Pass 1 comment for the full rationale. Decided at
  // BLOCK granularity (one gain per callback, smoothed block-to-block), NOT
  // per-sample: a hard per-sample version of this exact idea was tried first
  // and produced broadband static, confirmed on-device — near-end and echo
  // interfere constantly, so their instantaneous sum legitimately crosses a
  // per-sample energy threshold many times per cycle even when cancellation
  // is working CORRECTLY, not just when it's stale. This is the same lesson
  // mRefEnvelope's per-sample-vs-smoothed fix already taught elsewhere in
  // this file. Pass 2 (learning) reads the UNCLAMPED raw residual (stored
  // here) regardless of this gain — it needs the true (mic - E[phi]) to know
  // how wrong E[phi] is, or a heavily-suppressed block would starve learning
  // of the very signal it needs to self-correct.
  std::vector<float> mRawResidual; // grow-only scratch; Pass 2's learning input

  // ---- Learning undo-ring (double-talk rewind) --------------------------
  // Every learned frame records its pre-update template values +
  // confidence so the last ~400ms of learning can be ROLLED BACK when the
  // double-talk detector engages: the detector has inherent latency
  // (spectral EMAs), and the ONSET of the performer's sound — its loudest,
  // most damaging part — always lands inside that blind window. Rewinding
  // through it makes detection latency cost nothing. Fixed-size ring,
  // preallocated; audio-thread only.
  struct LearnUndo {
    uint32_t phi;
    float oldT[2]; // per-channel pre-update template values (<=2 channels)
    float oldConf;
  };
  std::vector<LearnUndo> mUndoRing;
  size_t mUndoHead = 0;   // next write slot
  size_t mUndoCount = 0;  // valid entries (saturates at ring size)
  bool mPrevNearEndHold = false;
  int64_t mRewindRemaining = 0; // >0: chunked rewind in progress
  int mSeedRetryCount = 0; // audio-thread-only: bounded discard->retry loop

  /** Closed-loop alignment auto-correction, samples (signed). The seed's
   * GCC-PHAT measures the RESIDUAL ref->mic misalignment of the reference
   * the template actually consumed; the AEC's reference read adds this
   * correction to its acoustic delay, so successive seeds drive the
   * measurement toward zero. Written by the seed worker (only when the
   * pass had usable coherence), read on the audio thread each block. */
  std::atomic<int64_t> mAlignLagCorrection{0};
  float mOutputSuppressGain = 1.0f; // smoothed 0..1; see process()'s Pass 1
  // Smoothed mic/raw energy RATIO (not the derived gain) — averaging THIS
  // first, over several blocks, is what lets the margin below be tight
  // enough to catch sustained-but-moderate staleness (confirmed on-device:
  // a wide single-block margin missed a real, consistently-negative-ERLE
  // ghost) while still ignoring genuine single-block statistical noise.
  float mSmoothedEnergyRatio = 1.0f;

  // Replaced a fixed-100ms time-based "trust gate" that used to live here
  // (mMixChangeFrame/mTrustRampFrames/trustGate()) — it guessed how long to
  // distrust the template after a mix change, and that guess was wrong by
  // up to an order of magnitude (100ms vs. a reseed that can take a full
  // loop pass), confirmed on-device as inadequate. The Pass-1 clamp above
  // needs no guess: it reacts to whether THIS SPECIFIC sample's cancellation
  // is demonstrably wrong, not to how long ago some notification fired.

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
  // Raw-mic capture parallel to mRefCapture (same phase indexing): the
  // one-loop Wiener seed measures the LIVE transfer function ref->mic from
  // a single loop pass instead of trusting a months-old chirp IR.
  std::vector<float> mMicCapture;

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

  // ---- Seed self-scaling fit (audio-thread-only) -------------------------
  // OFFLINE-MEASURED motivation: on real cafe recordings the calibrated
  // IR's predicted echo fit the actual bleed with alpha = -0.26 — INVERTED
  // POLARITY and ~4x too hot (the calibration's echoGain=3.43 is the same
  // error seen from the other side). Applying that seed at face value made
  // the canceller ADD ~1.3x the bleed (the observed -1..-3 dB gated ERLE).
  // Rather than patch sign/gain conventions piecemeal, the seed is now
  // FITTED before it is trusted: after the job lands, one loop pass
  // accumulates num += mic·seed[phi], den += seed[phi]^2, then applies
  // alpha = clamp(num/den) * seed. A correct seed fits alpha ~ 1; a
  // wrong-sign/scale seed gets corrected; an uncorrelated (misaligned)
  // seed fits alpha ~ 0 and is discarded — the seed is structurally no
  // worse than no seed under ANY calibration bug.
  bool mSeedFitActive = false;
  // Set when the fit pass completes with a usable alpha; cleared when the
  // chunked apply finishes (or on any abort/reset/stale-drop). WITHOUT this
  // flag the post-fit-success state (fitActive=false, applyPos=0,
  // fitFrames=0) is indistinguishable from the fresh-drain state, so the
  // drain re-entered the fit forever and the apply was unreachable --
  // SEEDING stayed lit for entire sessions with arms=1/lands=0 while
  // convergence limped along on pure EMA (the measured 3-5+ loop slowness).
  bool mSeedFitDone = false;
  double mSeedFitNum = 0.0;
  double mSeedFitDen = 0.0;
  int64_t mSeedFitFrames = 0;
  float mSeedFitAlpha = 1.0f;
  // Last fitted alpha — also applied to per-track exact edits, which are
  // built from the SAME IR and inherit the same scale/sign error. 1.0
  // until the first fit lands.
  float mIrFitScale = 1.0f;

  std::atomic<bool> mSeedThreadRunning{false};
  std::thread mSeedWorker;
  void seedWorkerLoop();
  void computeSeedConvolution(int64_t P); // worker thread only

  // ---- Per-track exact subtraction (see registerTrackAudio/setTrackActive
  // doc comment above for the full rationale) ------------------------------
  //
  // trackIndex (as generated everywhere in Dart: `<hash>.hashCode &
  // 0x7FFFFFFF`, a ~31-bit value up to ~2.1 billion) is NOT a dense small
  // index — it must never be used to directly index a fixed-size array.
  // This mirrors AudioEngine::mTrackHandles's scan-and-allocate slot table
  // (audio_engine.h/.cpp) rather than array-indexing by trackIndex directly.
  // slotIndex.load()==-1 means the slot is free. Allocation (registerTrack-
  // Audio, any thread) uses compare_exchange to claim a free slot
  // lock-free; lookup (setTrackActive, AUDIO THREAD — must never block) is
  // a plain atomic-load scan over kMaxTracks (64) entries, which is O(64)
  // and negligible next to the O(P) work already done per callback.
  struct TrackContribution {
    std::atomic<int> slotIndex{-1};   // the trackIndex owning this slot, or
                                       // -1 if free. Named slotIndex (not
                                       // trackIndex) to avoid confusion with
                                       // the array position, which is NOT
                                       // the trackIndex.
    std::vector<float> E;             // per-phase*channel contribution, same
                                       // layout as mTemplate; empty until computed
    std::atomic<bool> computed{false}; // set (release) once E is fully populated
    // Desired activation state, settable from any thread BEFORE the
    // contribution finishes computing. drainPendingTrackEdits() reconciles
    // computed && (wantActive != active) on the audio thread, so a take
    // that starts looping the moment it finalizes gets its exact edit the
    // instant the worker finishes -- previously setTrackActive on an
    // uncomputed track was a silent no-op and the edit NEVER landed.
    std::atomic<bool> wantActive{false};
    bool active = false;              // audio-thread-only: mirrors what's
                                       // CURRENTLY summed into mTemplate
    // Audio-thread-only gain pair: targetGain is the mixer's current gain
    // for this track (settable any time); appliedGain is the gain actually
    // summed into mTemplate right now (0 while inactive). setTrackGain on
    // an active track applies the (target - applied) delta exactly.
    float targetGain = 1.0f;
    float appliedGain = 0.0f;
    // The track's own raw mono audio, RETAINED so the contribution can be
    // recomputed if the impulse response arrives/changes AFTER the track was
    // registered (a track recorded BEFORE a live calibration this session, or
    // after a recalibration). Written and read only from the Dart-facing
    // (non-audio) thread — registerTrackAudio writes it, requeueAll... reads
    // it — so no atomic needed; the worker thread computes from a job's OWN
    // copy, never this field.
    std::vector<float> audio;
  };
  static constexpr int kMaxTracks = 64; // matches AudioEngine::kMaxTrackHandles
  std::array<TrackContribution, kMaxTracks> mTrackContributions;

  // ---- Deferred per-track template edits (#54 race fix) -----------------
  // setTrackActive/setTrackGain run on the Dart thread, but mTemplate AND
  // tc.active/appliedGain are audio-thread-only (see the fields above). So the
  // setters now only VALIDATE with atomic reads (slot + computed) and ENQUEUE
  // here; drainPendingTrackEdits(), called at the top of process(), applies the
  // O(P) mTemplate edits and mutates tc.active/appliedGain in FIFO order on the
  // audio thread. This restores the documented "audio-thread-only" invariant
  // and removes the template-corruption data race that blocked re-enabling the
  // per-track feed-forward. The drain try_locks (never blocks the RT thread);
  // an uncontended push is a cheap vector append.
  struct PendingTrackEdit {
    int trackIndex;
    bool isGain; // false = active toggle, true = gain edit
    bool active; // active-toggle target
    float gain;  // gain-edit target
  };
  std::vector<PendingTrackEdit> mPendingTrackEdits; // guarded by mPendingEditsMutex
  std::mutex mPendingEditsMutex;
  std::vector<PendingTrackEdit> mDrainScratch; // audio-thread reuse buffer
  void drainPendingTrackEdits();               // audio thread only
  void applyTrackActive(int trackIndex, bool active); // audio thread only
  void applyTrackGain(int trackIndex, float gain);    // audio thread only

  // Find the slot already owned by trackIndex, or atomically claim a free
  // one. Callable from any thread; lock-free. Returns -1 if trackIndex is
  // invalid or the table is full (all 64 slots owned by OTHER tracks).
  int findOrAllocTrackSlot(int trackIndex);
  // Find the slot already owned by trackIndex without allocating. Callable
  // from any thread including the audio thread; lock-free, non-blocking.
  // Returns -1 if trackIndex has no slot.
  int findTrackSlot(int trackIndex) const;

  // Registration jobs queued from any thread, drained by the seed worker
  // (reuses its poll loop — registration is a rare, one-shot-per-track
  // event, not a steady-state load, exactly like the composite seed).
  struct TrackRegJob {
    int trackIndex;
    std::vector<float> audio; // mono, full-period copy
  };
  std::mutex mTrackJobMutex;
  std::vector<TrackRegJob> mPendingTrackJobs;
  void computeTrackContribution(int trackIndex, const std::vector<float> &audio); // worker thread only

  // Re-queue a compute job for every registered track that has retained
  // audio, so their contributions are (re)computed against the CURRENT
  // impulse response. Called from setSeedImpulseResponse whenever the IR
  // arrives or changes — the fix for "recorded a track before calibrating,
  // so it registered against an empty IR and never got a contribution."
  // Dart-thread only (same as registerTrackAudio).
  void requeueAllTrackContributions();
};

#endif // AEC_SYNCHRONOUS_ECHO_TEMPLATE_H
